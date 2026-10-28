"""Portfolio construction and cost design: loader and validation of ``configs/portfolio.yaml``.

The design is fixed in the config before any portfolio return exists. The loader is strict: an
unknown or missing key, a wrong type, a method the design does not define, a signal column the
predictions file does not have, or a horizon / lag that disagrees with ``configs/features.yaml``,
``configs/research.yaml`` or ``configs/models.yaml`` is an error, never silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.models import BASELINES, MODEL_NAMES, load_models_config
from src.run_models import column_name
from src.signals import embargo_days, load_research_config
from src.utils import REPO_ROOT, load_config

PORTFOLIO_CONFIG_PATH = REPO_ROOT / "configs" / "portfolio.yaml"
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
