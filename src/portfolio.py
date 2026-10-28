"""Portfolio construction: the ``configs/portfolio.yaml`` loader and the three constructions.

The design is fixed in the config before any portfolio return exists. The loader is strict: an
unknown or missing key, a wrong type, a method the design does not define, a signal column the
predictions file does not have, or a horizon / lag that disagrees with ``configs/features.yaml``,
``configs/research.yaml`` or ``configs/models.yaml`` is an error, never silently ignored.

Each construction maps one date's scores (plus beta, vol, ADV, the risk model and the current
weights) to target weights per $ of the capital that holds them: ``decile_ls``,
``neutral_volscaled`` and ``optimised``. ``*_builder`` functions bind a construction to the
config and to the backtest's market data (``src.backtest.MarketData``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.covariance import ledoit_wolf_shrinkage

from src.costs import CostModel, cost_coefficients, max_trade_weight
from src.models import BASELINES, MODEL_NAMES, load_models_config
from src.run_models import column_name
from src.signals import embargo_days, load_research_config, quantile_buckets
from src.utils import REPO_ROOT, load_config

PORTFOLIO_CONFIG_PATH = REPO_ROOT / "configs" / "portfolio.yaml"
log = logging.getLogger(__name__)
CONSTRUCTIONS = ("decile_ls", "neutral_volscaled", "optimised")
ROLES = ("primary", "benchmark", "low_turnover_variant")
UNVARIED = ("B0_reversal", "B1_composite")  # baselines without a target variant
IC_TABLE_COLUMNS = {"model", "horizon", "variant", "fold_id", "validation_ic"}

NUM = (int, float)
SignalSpec = {"predictor": str, "horizon": int, "variant": (str, type(None)), "role": str}

# Expected keys and leaf types. A dict is a section whose keys must match exactly; ``SIGNALS``
# marks the mapping of column name -> SignalSpec.
SIGNALS = object()
SCHEMA: dict[str, Any] = {
    "research_config": str,
    "features_config": str,
    "models_config": str,
    "predictions": str,
    "signals": SIGNALS,
    "sample": {"start": str, "end": str, "split_date": str, "ramp_up": str},
    "timing": {
        "lag": int, "execution": str, "daily_return": str, "drift": str, "missing_daily_return": str,
        "terminal_exit": str, "unscored_holding": str,
    },
    "rebalancing": {"scheme": str, "n_tranches": str, "capital": str},
    "universe": {"scored": bool, "score_transform": str},
    "risk_model": {"estimator": str, "window": int, "min_obs_frac": NUM, "missing": str},
    "constructions": {
        "decile_ls": {"quantiles": str, "weighting": str, "gross_leverage": NUM, "beta_hedge": bool},
        "neutral_volscaled": {
            "weighting": str, "vol_feature": str, "beta_feature": str, "neutralise": str,
            "target_vol_annual": NUM, "max_gross": NUM,
        },
        "optimised": {
            "alpha": {"model": str, "vol_feature": str, "ic_source": str, "ic_table": str, "ic_floor": NUM},
            "constraints": {
                "dollar_neutral": bool, "max_abs_beta": NUM, "beta_feature": str, "max_weight": NUM,
                "max_gross": NUM, "max_trade_adv_frac": NUM,
            },
            "lambda_risk": list,
            "lambda_tc": list,
            "selection": {"metric": str, "window": str, "ties": str, "reuse_in_scenarios": bool},
            "solver": {"package": str, "name": str},
        },
        "sector_neutral": {"enabled": bool},
    },
    "costs": {
        "commission_bps": NUM,
        "half_spread_bps": NUM,
        "impact": {"model": str, "y": NUM, "vol_feature": str, "adv_feature": str},
        "borrow_bps_per_year": NUM,
        "cash_rate": NUM,
        "base_aum_usd": NUM,
        "scenarios": {"cost_multipliers": list, "aum_usd": list},
        "breakeven": str,
    },
    "statistics": {"frequency": str, "trading_days": str, "newey_west": str, "turnover": str},
    "headline": {
        "signal": str, "construction": str, "cost_multiplier": NUM, "aum_usd": NUM, "window": str,
        "statistics": list, "benchmark": str, "alongside": list,
    },
    "outputs": {"reports_dir": str, "figures_dir": str, "daily_returns": str, "weights": str},
}

# Method choices that the design defines; anything else is rejected.
FIXED = {
    "sample.start": "first_oos_date",
    "sample.end": "last_research_date",
    "sample.ramp_up": "all_tranches_invested",
    "timing.execution": "close",
    "timing.daily_return": "ret_1d",
    "timing.drift": "buy_and_hold",
    "timing.missing_daily_return": "label_residual",
    "timing.terminal_exit": "last_price",
    "timing.unscored_holding": "sell_at_rebalance",
    "rebalancing.scheme": "staggered_tranches",
    "rebalancing.n_tranches": "horizon",
    "rebalancing.capital": "equal",
    "universe.score_transform": "zscore_by_date",
    "risk_model.estimator": "ledoit_wolf",
    "risk_model.missing": "zero_after_demeaning",
    "constructions.decile_ls.quantiles": "from_research_config",
    "constructions.decile_ls.weighting": "equal",
    "constructions.neutral_volscaled.weighting": "score_over_vol",
    "constructions.neutral_volscaled.neutralise": "project_out_dollar_and_beta",
    "constructions.optimised.alpha.model": "grinold",
    "constructions.optimised.alpha.ic_source": "validation_block",
    "constructions.optimised.selection.metric": "net_sharpe",
    "constructions.optimised.selection.window": "tuning",
    "constructions.optimised.selection.ties": "first_in_config_order",
    "constructions.optimised.solver.package": "cvxpy",
    "costs.impact.model": "square_root",
    "costs.breakeven": "ex_post_at_base_trades",
    "statistics.frequency": "daily",
    "statistics.trading_days": "from_research_config",
    "statistics.newey_west": "from_research_config",
    "statistics.turnover": "one_way_annual",
    "headline.window": "evaluation",
}
HEADLINE_STATISTICS = {"net_sharpe", "net_nw_t"}
ALONGSIDE = {"gross_sharpe", "annual_turnover", "breakeven_cost_multiplier"}


def _is_type(value: Any, expected: Any) -> bool:
    types = expected if isinstance(expected, tuple) else (expected,)
    if isinstance(value, bool) and bool not in types:
        return False  # bool is an int subclass: True must not pass as a number
    return isinstance(value, types)


def _check_schema(section: Any, schema: Any, where: str) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"{where or 'config'} must be a mapping")
    unknown, missing = set(section) - set(schema), set(schema) - set(section)
    if unknown:
        raise ValueError(f"{where or 'config'}: unknown keys {sorted(unknown)}")
    if missing:
        raise ValueError(f"{where or 'config'}: missing keys {sorted(missing)}")
    for key, expected in schema.items():
        path, value = f"{where}.{key}" if where else key, section[key]
        if expected is SIGNALS:
            if not isinstance(value, dict) or not value:
                raise ValueError(f"{path} must be a non-empty mapping of column -> spec")
            for column, spec in value.items():
                _check_schema(spec, SignalSpec, f"{path}.{column}")
        elif isinstance(expected, dict):
            _check_schema(value, expected, path)
        elif not _is_type(value, expected):
            raise ValueError(f"{path} must be of type {expected}, got {type(value).__name__}")


def _get(cfg: dict[str, Any], path: str) -> Any:
    for key in path.split("."):
        cfg = cfg[key]
    return cfg


def _positive_numbers(name: str, values: list[Any], allow_zero: bool = False) -> None:
    ok = values and all(_is_type(v, NUM) and (v >= 0 if allow_zero else v > 0) for v in values)
    if not ok or len(set(values)) != len(values) or list(values) != sorted(values):
        raise ValueError(f"{name} must be a non-empty, increasing list of {'non-negative' if allow_zero else 'positive'} numbers")


def _check_signals(cfg: dict[str, Any], rcfg: dict[str, Any], mcfg: dict[str, Any]) -> None:
    signals = cfg["signals"]
    for column, spec in signals.items():
        name, h, variant = spec["predictor"], spec["horizon"], spec["variant"]
        if name not in BASELINES + MODEL_NAMES:
            raise ValueError(f"signal {column}: unknown predictor {name!r}")
        if h not in rcfg["horizons"] or h not in mcfg["targets"]["horizons"]:
            raise ValueError(f"signal {column}: horizon {h} is not a research and model horizon")
        if (variant is None) != (name in UNVARIED):
            raise ValueError(f"signal {column}: only {UNVARIED} have no target variant")
        if variant is not None and variant not in mcfg["targets"]["variants"]:
            raise ValueError(f"signal {column}: unknown target variant {variant!r}")
        if column != column_name(name, h, variant):
            raise ValueError(f"signal {column}: the column of this predictor is {column_name(name, h, variant)!r}")
        if spec["role"] not in ROLES:
            raise ValueError(f"signal {column}: role must be one of {ROLES}")
    if sum(spec["role"] == "primary" for spec in signals.values()) != 1:
        raise ValueError("exactly one signal must have role 'primary'")


def _check_predictions(cfg: dict[str, Any], predictions: Path) -> None:
    """The signal columns exist in the predictions file (schema only; no values are read)."""
    names = set(pq.read_schema(predictions).names)
    missing = set(cfg["signals"]) - names
    if missing:
        raise ValueError(f"{predictions} has no column(s) {sorted(missing)}")
    if not {"date", "ticker"} <= names:
        raise ValueError(f"{predictions} must be indexed by (date, ticker)")


def _check_timing_and_sample(
    cfg: dict[str, Any], fcfg: dict[str, Any], rcfg: dict[str, Any], mcfg: dict[str, Any]
) -> None:
    lag = cfg["timing"]["lag"]
    if lag != fcfg["labels"]["lag"]:
        raise ValueError(f"timing.lag ({lag}) must equal labels.lag in features.yaml ({fcfg['labels']['lag']})")
    if lag + max(s["horizon"] for s in cfg["signals"].values()) > embargo_days(rcfg, fcfg):
        raise ValueError("a tranche would hold positions past the embargo into the holdout")
    split = pd.Timestamp(cfg["sample"]["split_date"])
    first_test = pd.Timestamp(mcfg["walk_forward"]["first_test_year"], 1, 1)
    if not first_test < split < pd.Timestamp(rcfg["holdout_start"]):
        raise ValueError("sample.split_date must fall after the first test year starts and before the holdout")


def _check_features(cfg: dict[str, Any], rcfg: dict[str, Any]) -> None:
    con = cfg["constructions"]
    used = {
        "timing.daily_return": cfg["timing"]["daily_return"],
        "neutral_volscaled.vol_feature": con["neutral_volscaled"]["vol_feature"],
        "neutral_volscaled.beta_feature": con["neutral_volscaled"]["beta_feature"],
        "optimised.alpha.vol_feature": con["optimised"]["alpha"]["vol_feature"],
        "optimised.constraints.beta_feature": con["optimised"]["constraints"]["beta_feature"],
        "costs.impact.vol_feature": cfg["costs"]["impact"]["vol_feature"],
        "costs.impact.adv_feature": cfg["costs"]["impact"]["adv_feature"],
    }
    for key, feature in used.items():
        if feature not in rcfg["signals"]:
            raise ValueError(f"{key}: {feature!r} is not a panel feature")


def _check_constructions(cfg: dict[str, Any]) -> None:
    con, risk = cfg["constructions"], cfg["risk_model"]
    if risk["window"] < 2 or not 0 < risk["min_obs_frac"] <= 1:
        raise ValueError("risk_model needs window >= 2 and 0 < min_obs_frac <= 1")
    if con["decile_ls"]["gross_leverage"] != 2.0:
        raise ValueError("constructions.decile_ls.gross_leverage must be 2 ($1 long + $1 short)")
    vs = con["neutral_volscaled"]
    if not 0 < vs["target_vol_annual"] < 1 or not 0 < vs["max_gross"] <= 2:
        raise ValueError("neutral_volscaled needs 0 < target_vol_annual < 1 and 0 < max_gross <= 2")
    opt = con["optimised"]
    if opt["alpha"]["ic_floor"] < 0:
        raise ValueError("optimised.alpha.ic_floor must be >= 0")
    c = opt["constraints"]
    if not c["dollar_neutral"]:
        raise ValueError("optimised.constraints.dollar_neutral: only dollar-neutral books are defined")
    if not 0 <= c["max_abs_beta"] < 1 or not 0 < c["max_weight"] <= 1:
        raise ValueError("optimised needs 0 <= max_abs_beta < 1 and 0 < max_weight <= 1")
    if not 0 < c["max_gross"] <= 2 or not 0 < c["max_trade_adv_frac"] <= 1:
        raise ValueError("optimised needs 0 < max_gross <= 2 and 0 < max_trade_adv_frac <= 1")
    _positive_numbers("optimised.lambda_risk", opt["lambda_risk"])
    _positive_numbers("optimised.lambda_tc", opt["lambda_tc"])
    if con["sector_neutral"]["enabled"]:
        raise ValueError("sector neutrality needs a point-in-time sector field, which the panel does not have")
    table = REPO_ROOT / opt["alpha"]["ic_table"]
    if not table.exists() or not IC_TABLE_COLUMNS <= set(pd.read_csv(table, nrows=0).columns):
        raise ValueError(f"optimised.alpha.ic_table: {table} must exist with columns {sorted(IC_TABLE_COLUMNS)}")


def _check_costs(cfg: dict[str, Any]) -> None:
    costs = cfg["costs"]
    for key in ("commission_bps", "half_spread_bps", "borrow_bps_per_year"):
        if costs[key] < 0:
            raise ValueError(f"costs.{key} must be >= 0")
    if costs["impact"]["y"] < 0 or costs["base_aum_usd"] <= 0:
        raise ValueError("costs need impact.y >= 0 and base_aum_usd > 0")
    if costs["cash_rate"] != 0:
        raise ValueError("costs.cash_rate: returns are excess returns, so cash earns 0")
    scen = costs["scenarios"]
    _positive_numbers("costs.scenarios.cost_multipliers", scen["cost_multipliers"], allow_zero=True)
    _positive_numbers("costs.scenarios.aum_usd", scen["aum_usd"])
    if 1.0 not in scen["cost_multipliers"] or costs["base_aum_usd"] not in scen["aum_usd"]:
        raise ValueError("cost scenarios must include multiplier 1 and the base AUM")


def _check_headline(cfg: dict[str, Any]) -> None:
    head, signals = cfg["headline"], cfg["signals"]
    if head["signal"] not in signals or signals[head["signal"]]["role"] != "primary":
        raise ValueError("headline.signal must be the primary signal")
    if head["benchmark"] not in signals or signals[head["benchmark"]]["role"] != "benchmark":
        raise ValueError("headline.benchmark must be a benchmark signal")
    if signals[head["benchmark"]]["horizon"] != signals[head["signal"]]["horizon"]:
        raise ValueError("headline.benchmark must have the headline signal's horizon")
    if head["construction"] not in CONSTRUCTIONS:
        raise ValueError(f"headline.construction must be one of {CONSTRUCTIONS}")
    if head["cost_multiplier"] != 1.0 or head["aum_usd"] != cfg["costs"]["base_aum_usd"]:
        raise ValueError("the headline is at base costs (multiplier 1) and base AUM")
    if set(head["statistics"]) != HEADLINE_STATISTICS or set(head["alongside"]) != ALONGSIDE:
        raise ValueError(f"headline reports {sorted(HEADLINE_STATISTICS)} alongside {sorted(ALONGSIDE)}")


def _check_outputs(cfg: dict[str, Any]) -> None:
    out = cfg["outputs"]
    reports = Path(out["reports_dir"])
    if reports.parts[:1] != ("reports",) or Path(out["figures_dir"]).parent != reports:
        raise ValueError("outputs.reports_dir must be under reports/ and figures_dir inside it")
    for key in ("daily_returns", "weights"):
        if Path(out[key]).parent != Path("data/processed") or not out[key].endswith(".parquet"):
            raise ValueError(f"outputs.{key} must be a parquet file in data/processed (gitignored)")


def load_portfolio_config(
    path: str | Path | None = None,
    fcfg: dict[str, Any] | None = None,
    rcfg: dict[str, Any] | None = None,
    mcfg: dict[str, Any] | None = None,
    predictions: str | Path | None = None,
) -> dict[str, Any]:
    """Load ``configs/portfolio.yaml`` and reject anything outside the pre-registered design.

    The feature, research and models configs default to the paths named in the portfolio config.
    The signal columns are checked against the predictions file (``predictions``, else the
    configured path) when it exists; only its schema is read.
    """
    with open(path or PORTFOLIO_CONFIG_PATH) as handle:
        cfg = yaml.safe_load(handle)
    _check_schema(cfg, SCHEMA, "")
    for key, expected in FIXED.items():
        if _get(cfg, key) != expected:
            raise ValueError(f"{key}: only {expected!r} is defined")
    fcfg = fcfg or load_config(REPO_ROOT / cfg["features_config"])
    rcfg = rcfg or load_research_config(REPO_ROOT / cfg["research_config"], fcfg)
    mcfg = mcfg or load_models_config(REPO_ROOT / cfg["models_config"], rcfg)
    _check_signals(cfg, rcfg, mcfg)
    _check_timing_and_sample(cfg, fcfg, rcfg, mcfg)
    _check_features(cfg, rcfg)
    _check_constructions(cfg)
    _check_costs(cfg)
    _check_headline(cfg)
    _check_outputs(cfg)
    pred_path = Path(predictions) if predictions else REPO_ROOT / cfg["predictions"]
    if predictions is not None or pred_path.exists():
        _check_predictions(cfg, pred_path)
    return cfg


def evaluation_start(cfg: dict[str, Any], dates: pd.DatetimeIndex) -> pd.Timestamp:
    """First trading day on or after ``sample.split_date``: the start of the evaluation window."""
    dates = pd.DatetimeIndex(sorted(set(dates)))
    pos = int(dates.searchsorted(pd.Timestamp(cfg["sample"]["split_date"]), side="left"))
    if pos == len(dates):
        raise ValueError("no trading date on or after sample.split_date")
    return dates[pos]


# --------------------------------------------------------------------------- shared pieces

TOL = 1e-6  # constraint tolerance on weights per $ of capital (solver accuracy)
FALLBACKS = ("hold", "flat", "partial")


@dataclass(frozen=True)
class Target:
    """Target weights of one rebalance; ``fallback`` names the fallback used, if any."""

    weights: np.ndarray
    fallback: str | None = None
    status: str = "ok"


@dataclass(frozen=True)
class RiskModel:
    """Ledoit-Wolf covariance kept in factored form: ``(1 - d) X'X / T + d * mu * I``.

    ``X`` holds the window's daily returns demeaned per name, missing returns set to 0 (the name's
    window mean), so the sample covariance is ``X'X / T``.
    """

    X: np.ndarray
    shrinkage: float
    mu: float

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    def covariance(self) -> np.ndarray:
        sample = self.X.T @ self.X / self.n_obs
        return (1 - self.shrinkage) * sample + self.shrinkage * self.mu * np.eye(self.X.shape[1])

    def variance(self, w: np.ndarray) -> float:
        """Daily ex-ante variance of weights ``w``."""
        xw = self.X @ w
        return float((1 - self.shrinkage) * (xw @ xw) / self.n_obs + self.shrinkage * self.mu * (w @ w))

    def subset(self, keep: np.ndarray) -> "RiskModel":
        """The same shrunk covariance restricted to some names (a principal submatrix)."""
        return RiskModel(self.X[:, keep], self.shrinkage, self.mu)


def ledoit_wolf_risk(window: np.ndarray, min_obs_frac: float) -> tuple[RiskModel, np.ndarray]:
    """Risk model from a ``T x n`` window of daily returns (NaN = no panel row) and the mask of
    names with at least ``min_obs_frac`` of the window observed."""
    window = np.asarray(window, dtype="float64")
    observed = np.isfinite(window)
    n_obs = observed.sum(axis=0)
    covered = n_obs >= min_obs_frac * window.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_obs > 0, np.where(observed, window, 0.0).sum(axis=0) / n_obs, 0.0)
    X = np.where(observed, window - mean, 0.0)
    n = X.shape[1]
    sample_trace = float((X * X).sum()) / max(X.shape[0], 1)
    mu = sample_trace / n if n else 0.0
    shrinkage = float(ledoit_wolf_shrinkage(X, assume_centered=True)) if X.shape[0] > 1 and n > 1 else 1.0
    return RiskModel(X, shrinkage, mu), covered


def zscore(score: np.ndarray) -> np.ndarray:
    """Standardise the finite values (mean 0, sample sd 1, as ``src.utils.zscore_by_date``); NaN elsewhere."""
    score = np.asarray(score, dtype="float64")
    ok = np.isfinite(score)
    out = np.full(score.shape, np.nan)
    if ok.sum() > 1:
        sd = score[ok].std(ddof=1)
        if sd > 0:
            out[ok] = (score[ok] - score[ok].mean()) / sd
    return out


def neutralise(w: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Least-change projection of ``w`` onto ``{sum w = 0, beta'w = 0}``."""
    A = np.column_stack([np.ones(len(w)), beta])
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    return w - A @ coef


# --------------------------------------------------------------------------- constructions


def decile_ls(score: np.ndarray, quantiles: int, min_names: int, gross: float = 2.0) -> np.ndarray:
    """Equal weight, long the top and short the bottom quantile of ``score`` ($gross / 2 per leg).

    Buckets come from ``src.signals.quantile_buckets`` (tied scores share a bucket, NaN is
    excluded, fewer than ``min_names`` scores gives no position).
    """
    buckets = quantile_buckets(pd.DataFrame([np.asarray(score, dtype="float64")]), quantiles, min_names)
    b = buckets.to_numpy(dtype="float64")[0]
    w = np.zeros(len(b))
    for leg, sign in ((quantiles - 1, 1.0), (0, -1.0)):
        member = b == leg
        if member.any():
            w[member] = sign * gross / 2 / member.sum()
    if not (w > 0).any() or not (w < 0).any():
        return np.zeros(len(b))
    return w


def neutral_volscaled(
    score: np.ndarray, vol: np.ndarray, beta: np.ndarray, risk: RiskModel, target_vol_annual: float,
    max_gross: float, min_names: int, trading_days: int,
) -> np.ndarray:
    """Weights proportional to z(score) / vol, projected to zero net dollars and zero beta, then
    scaled to ``target_vol_annual`` ex-ante (``risk`` aligned with ``score``), gross capped at
    ``max_gross``. Names without a score, vol or beta get 0."""
    score, vol, beta = (np.asarray(a, dtype="float64") for a in (score, vol, beta))
    ok = np.isfinite(score) & np.isfinite(vol) & (vol > 0) & np.isfinite(beta)
    w = np.zeros(len(score))
    if ok.sum() < max(min_names, 3):
        return w
    z = zscore(score[ok])
    if not np.isfinite(z).all():
        return w
    w[ok] = neutralise(z / vol[ok], beta[ok])
    ann_vol = np.sqrt(max(risk.variance(w), 0.0) * trading_days)
    gross = np.abs(w).sum()
    if ann_vol <= 0 or gross <= 0:
        return np.zeros(len(score))
    scale = min(target_vol_annual / ann_vol, max_gross / gross)
    return w * scale


def grinold_alpha(score: np.ndarray, vol: np.ndarray, ic: float, horizon: int) -> np.ndarray:
    """h-day alpha ``IC * vol * sqrt(h) * z(score)``: ``vol`` is daily, so ``vol * sqrt(h)`` is the
    volatility over the forecast horizon."""
    return ic * np.asarray(vol, dtype="float64") * np.sqrt(horizon) * zscore(score)


@dataclass(frozen=True)
class OptimiserSpec:
    """Objective weights and constraints of ``optimised`` for one (lambda_risk, lambda_tc)."""

    horizon: int
    lambda_risk: float
    lambda_tc: float
    max_abs_beta: float
    max_weight: float
    max_gross: float
    max_trade_adv_frac: float
    solver: str = "CLARABEL"

    @classmethod
    def from_config(cls, pcfg: dict[str, Any], horizon: int, lambda_risk: float, lambda_tc: float) -> "OptimiserSpec":
        opt = pcfg["constructions"]["optimised"]
        c = opt["constraints"]
        return cls(horizon, float(lambda_risk), float(lambda_tc), float(c["max_abs_beta"]), float(c["max_weight"]),
                   float(c["max_gross"]), float(c["max_trade_adv_frac"]), opt["solver"]["name"])


def check_constraints(
    w: np.ndarray, current: np.ndarray, beta: np.ndarray, max_dw: np.ndarray, free: np.ndarray,
    spec: OptimiserSpec, tol: float = TOL,
) -> dict[str, bool]:
    """Which of the ``optimised`` constraints ``w`` satisfies. Names that are not ``free`` must be 0;
    the ADV cap applies to free names (a forced sale is not capped)."""
    b = np.where(free, beta, 0.0)
    return {
        "dollar_neutral": abs(w.sum()) <= tol,
        "beta": abs(b @ w) <= spec.max_abs_beta + tol,
        "max_weight": bool(np.all(np.abs(w) <= spec.max_weight + tol)),
        "max_gross": np.abs(w).sum() <= spec.max_gross + tol,
        "adv": bool(np.all(np.abs(w - current)[free] <= max_dw[free] + tol)),
        "forced_zero": bool(np.all(w[~free] == 0)),
    }


def fallback_weights(
    current: np.ndarray, beta: np.ndarray, max_dw: np.ndarray, free: np.ndarray, spec: OptimiserSpec,
) -> Target:
    """Weights when the optimiser fails, in order of preference:

    1. ``hold``: the current weights (non-free names sold), projected to zero net dollars and beta,
       then scaled down into the name cap and gross limit, if the ADV cap allows the trade;
    2. ``flat``: all weights 0, if the ADV cap allows it;
    3. ``partial``: each free name moved towards 0 by as much as the ADV cap allows. This is the
       only output that may break dollar neutrality, beta or the caps; it is flagged as such.
    """
    hold = np.zeros(len(current))
    if free.sum() >= 2:
        hold[free] = neutralise(current[free], beta[free])
    biggest, gross = np.abs(hold).max(initial=0.0), np.abs(hold).sum()
    scale = min(1.0, spec.max_weight / biggest if biggest > 0 else 1.0, spec.max_gross / gross if gross > 0 else 1.0)
    hold *= scale
    if all(check_constraints(hold, current, beta, max_dw, free, spec).values()):
        return Target(hold, "hold", "fallback")
    flat = np.zeros(len(current))
    if all(check_constraints(flat, current, beta, max_dw, free, spec).values()):
        return Target(flat, "flat", "fallback")
    partial = np.where(free, current - np.clip(current, -max_dw, max_dw), 0.0)
    return Target(partial, "partial", "fallback_constraints_violated")


def optimised(
    alpha: np.ndarray, beta: np.ndarray, current: np.ndarray, max_dw: np.ndarray, free: np.ndarray,
    risk: RiskModel, linear: np.ndarray, impact: np.ndarray, spec: OptimiserSpec,
) -> Target:
    """Maximise ``alpha'w - lambda_risk * h * w'Sigma w - lambda_tc * cost(w - current)``.

    Subject to sum w = 0, |beta'w| <= max_abs_beta, |w_i| <= max_weight, sum |w| <= max_gross and
    |w_i - current_i| <= max_dw_i for ``free`` names; every other name is sold (w = 0). ``risk`` is
    aligned with the inputs (daily covariance); ``linear`` / ``impact`` come from
    ``src.costs.cost_coefficients``. A solver error, a non-optimal status or a solution that breaks
    a constraint gives ``fallback_weights``, logged.
    """
    import cvxpy as cp

    n = len(current)
    idx = np.flatnonzero(free)
    if len(idx) < 2:
        return fallback_weights(current, beta, max_dw, free, spec)
    w = cp.Variable(len(idx))
    dw = w - current[idx]
    sub = risk.subset(idx)
    risk_term = (1 - sub.shrinkage) * cp.sum_squares(sub.X @ w) / sub.n_obs + sub.shrinkage * sub.mu * cp.sum_squares(w)
    cost_term = linear[idx] @ cp.abs(dw) + impact[idx] @ cp.power(cp.abs(dw), 1.5)
    objective = cp.Maximize(alpha[idx] @ w - spec.lambda_risk * spec.horizon * risk_term - spec.lambda_tc * cost_term)
    constraints = [
        cp.sum(w) == 0,
        cp.abs(beta[idx] @ w) <= spec.max_abs_beta,
        cp.abs(w) <= spec.max_weight,
        cp.norm1(w) <= spec.max_gross,
        cp.abs(dw) <= max_dw[idx],
    ]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=spec.solver)
        status = problem.status
    except Exception as exc:  # solver errors must not stop the backtest; they are counted instead
        status = f"error: {type(exc).__name__}"
    if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w.value is not None and np.isfinite(w.value).all():
        out = np.zeros(n)
        out[idx] = w.value
        if all(check_constraints(out, current, beta, max_dw, free, spec).values()):
            return Target(out, None, status)
        status = f"{status}, constraints violated"
    target = fallback_weights(current, beta, max_dw, free, spec)
    log.warning("optimiser %s: fallback %s", status, target.fallback)
    return Target(target.weights, target.fallback, status)


# --------------------------------------------------------------------------- builders

# A builder maps (market data, score position s, current weights, capital $, cost model) to a Target.
Builder = Callable[[Any, int, np.ndarray, float, CostModel], Target]


def _window(data: Any, s: int, names: np.ndarray, window: int) -> np.ndarray:
    """Trailing ``window`` daily returns up to and including date position ``s`` (rows before the
    start of the data are simply absent)."""
    return data.ret[max(0, s - window + 1): s + 1][:, names]


def decile_builder(quantiles: int, min_names: int, gross: float) -> Builder:
    def build(data: Any, s: int, current: np.ndarray, capital_usd: float, costs: CostModel) -> Target:
        return Target(decile_ls(data.score[s], quantiles, min_names, gross))

    return build


def volscaled_builder(pcfg: dict[str, Any], min_names: int, trading_days: int) -> Builder:
    spec, risk_cfg = pcfg["constructions"]["neutral_volscaled"], pcfg["risk_model"]

    def build(data: Any, s: int, current: np.ndarray, capital_usd: float, costs: CostModel) -> Target:
        names = np.flatnonzero(np.isfinite(data.score[s]))
        w = np.zeros(len(current))
        if len(names) >= min_names:
            risk, _ = ledoit_wolf_risk(_window(data, s, names, risk_cfg["window"]), risk_cfg["min_obs_frac"])
            w[names] = neutral_volscaled(data.score[s, names], data.vol[s, names], data.beta[s, names], risk,
                                         spec["target_vol_annual"], spec["max_gross"], min_names, trading_days)
        return Target(w)

    return build


def optimised_builder(
    pcfg: dict[str, Any], spec: OptimiserSpec, ic_for: Callable[[int], float], min_names: int,
) -> Builder:
    """``ic_for(s)`` is the IC of the fold that contains score position ``s`` (already floored)."""
    risk_cfg = pcfg["risk_model"]

    def build(data: Any, s: int, current: np.ndarray, capital_usd: float, costs: CostModel) -> Target:
        score, vol, beta, adv = data.score[s], data.vol[s], data.beta[s], data.adv[s]
        scored = np.isfinite(score)
        names = np.flatnonzero(scored | (current != 0))
        if scored.sum() < min_names:  # no usable scores: every held name is sold
            return Target(np.zeros(len(current)), None, "no_scores")
        risk, covered = ledoit_wolf_risk(_window(data, s, names, risk_cfg["window"]), risk_cfg["min_obs_frac"])
        sub = lambda a: np.asarray(a, dtype="float64")[names]  # noqa: E731
        free = (np.isfinite(sub(score)) & np.isfinite(sub(vol)) & (sub(vol) > 0) & np.isfinite(sub(beta))
                & np.isfinite(sub(adv)) & (sub(adv) > 0) & covered)
        sigma, a = np.where(free, sub(vol), 0.0), np.where(free, sub(adv), np.inf)
        alpha = np.zeros(len(names))
        alpha[free] = grinold_alpha(sub(score)[free], sigma[free], ic_for(s), spec.horizon)
        linear, impact = cost_coefficients(sigma, a, capital_usd, costs)
        max_dw = max_trade_weight(np.where(free, sub(adv), np.nan), capital_usd, spec.max_trade_adv_frac)
        b = np.where(free, sub(beta), 0.0)
        target = optimised(alpha, b, sub(current), max_dw, free, risk, linear, impact, spec)
        w = np.zeros(len(current))
        w[names] = target.weights
        return Target(w, target.fallback, target.status)

    return build

