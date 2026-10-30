"""Risk and attribution: the ``configs/attribution.yaml`` loader, factor files and the statistics.

The design is fixed in the config before any factor file is downloaded. The loader is strict: an
unknown or missing key, a wrong type, a method the design does not define, a book the portfolio
config does not define, or a test that does not match the pre-registered question is an error,
never silently ignored. Only recorded base-case books are described; no backtest is rerun.

Factor parsers drop every row on or after ``holdout_start`` before converting its values, so the
published data past the holdout never reaches a result.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd
import yaml
from scipy import stats

from src.metrics import newey_west_lag
from src.portfolio import CONSTRUCTIONS, _check_schema, _get, _is_type, load_portfolio_config
from src.signals import load_research_config
from src.utils import REPO_ROOT, load_config

ATTRIBUTION_CONFIG_PATH = REPO_ROOT / "configs" / "attribution.yaml"
ROLES = ("headline", "primary_attribution", "benchmark")
FACTORS = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom", "st_rev", "bab")
SOURCE_FORMATS = ("french_csv_zip", "aqr_xlsx")
PNL_SERIES = ("gross_pnl", "net_pnl")

NUM = (int, float)
BookSpec = {"signal": str, "construction": str, "role": str}

# Expected keys and leaf types. A dict is a section whose keys must match exactly; ``dict`` as a
# leaf is a mapping whose entries the checks below validate.
SCHEMA: dict[str, Any] = {
    "portfolio_config": str,
    "research_config": str,
    "features_config": str,
    "book_data": {
        "daily_returns": str, "weights": str, "portfolio_metadata": str, "exposures_table": str,
        "scenario": str, "tranche": str,
    },
    "books": dict,
    "series": {"primary": str, "secondary": str},
    "windows": {"evaluation": str, "tuning": str},
    "factors": {"raw_dir": str, "cut_before": str, "coverage": str, "sources": dict},
    "models": dict,
    "regression": {
        "estimator": str, "standard_errors": str, "lag_rule": str, "trading_days": str, "decomposition": str,
    },
    "own_reversal_spanning": {"book": str, "regressor_book": str, "model": str, "series": str, "window": str},
    "primary_test": {
        "book": str, "series": str, "window": str, "model": str, "factor": str, "t_threshold": NUM,
    },
    "risk": {
        "cumulative": str,
        "drawdown_episodes": int,
        "var_es": {"method": str, "confidence": list, "horizons_days": list, "units": list},
        "moments": {"skew_excess_kurtosis_on": list},
        "worst_periods": list,
    },
    "beta_analysis": {
        "series": list,
        "market_proxies": {"equal_weighted": str, "value_weighted": str},
        "explanations": {
            "benchmark": str,
            "estimation_error": str,
            "stale_beta": {"short_beta_window": int, "min_obs_frac": NUM, "exposure": str},
            "conditional_beta": str,
            "up_down_asymmetry": str,
            "factor_controls": str,
        },
    },
    "checks": {
        "reproduce_exposures_tolerance": NUM, "date_alignment": str, "decomposition_tolerance": NUM, "coverage": str,
    },
    "outputs": {"reports_dir": str, "figures_dir": str, "notebook": str},
}
SOURCE_SCHEMA = {"url": str, "format": str, "scale": NUM, "columns": dict}
AQR_SOURCE_SCHEMA = {**SOURCE_SCHEMA, "sheet": str}
MODEL_SCHEMA = {"adds": list}

# Method choices that the design defines; anything else is rejected.
FIXED = {
    "book_data.scenario": "base",
    "book_data.tranche": "total",
    "series.primary": "gross_pnl",
    "series.secondary": "net_pnl",
    "windows.evaluation": "primary",
    "windows.tuning": "secondary",
    "factors.cut_before": "holdout_start",
    "factors.coverage": "every_book_date",
    "regression.estimator": "ols",
    "regression.standard_errors": "newey_west",
    "regression.lag_rule": "from_research_config",
    "regression.trading_days": "from_research_config",
    "regression.decomposition": "mean_equals_alpha_plus_contributions",
    "own_reversal_spanning.series": "gross_pnl",
    "own_reversal_spanning.window": "evaluation",
    "primary_test.book": "lightgbm_h5_neutral_volscaled",
    "primary_test.series": "gross_pnl",
    "primary_test.window": "evaluation",
    "primary_test.factor": "st_rev",
    "risk.cumulative": "additive",
    "risk.var_es.method": "historical",
    "beta_analysis.market_proxies.equal_weighted": "market_return",
    "beta_analysis.market_proxies.value_weighted": "mkt_rf",
    "beta_analysis.explanations.benchmark": "compare_proxies",
    "beta_analysis.explanations.estimation_error": "newey_west_interval",
    "beta_analysis.explanations.stale_beta.exposure": "rebalance_weights_times_short_beta",
    "beta_analysis.explanations.conditional_beta": "market_times_lagged_exposure",
    "beta_analysis.explanations.up_down_asymmetry": "split_on_sign_of_market",
    "beta_analysis.explanations.factor_controls": "market_beta_in_each_nested_model",
    "checks.date_alignment": "max_correlation_at_lag_zero",
    "checks.coverage": "every_book_date",
}


def _check_books(cfg: dict[str, Any], pcfg: dict[str, Any]) -> None:
    books = cfg["books"]
    if not books:
        raise ValueError("books must not be empty")
    for name, spec in books.items():
        _check_schema(spec, BookSpec, f"books.{name}")
        if spec["signal"] not in pcfg["signals"]:
            raise ValueError(f"books.{name}: {spec['signal']!r} is not a signal of the portfolio config")
        if spec["construction"] not in CONSTRUCTIONS:
            raise ValueError(f"books.{name}: construction must be one of {CONSTRUCTIONS}")
        if spec["role"] not in ROLES:
            raise ValueError(f"books.{name}: role must be one of {ROLES}")
    pairs = {(s["signal"], s["construction"]): s["role"] for s in books.values()}
    if len(pairs) != len(books):
        raise ValueError("books must not repeat a signal and construction")
    head = pcfg["headline"]
    if pairs.get((head["signal"], head["construction"])) != "headline":
        raise ValueError("books must hold the portfolio config's headline book with role 'headline'")
    if pairs.get((head["benchmark"], head["construction"])) != "benchmark":
        raise ValueError("books must hold the headline benchmark with role 'benchmark'")
    for role in ("headline", "primary_attribution"):
        if sum(r == role for r in pairs.values()) != 1:
            raise ValueError(f"exactly one book must have role {role!r}")


def _check_factors(cfg: dict[str, Any]) -> None:
    columns = []
    for name, source in cfg["factors"]["sources"].items():
        _check_schema(source, AQR_SOURCE_SCHEMA if source.get("format") == "aqr_xlsx" else SOURCE_SCHEMA,
                      f"factors.sources.{name}")
        if source["format"] not in SOURCE_FORMATS:
            raise ValueError(f"factors.sources.{name}.format must be one of {SOURCE_FORMATS}")
        if not source["url"].startswith("https://") or source["scale"] <= 0:
            raise ValueError(f"factors.sources.{name} needs an https url and a positive scale")
        columns += list(source["columns"].values())
    if sorted(columns) != sorted(FACTORS):
        raise ValueError(f"factors.sources must supply each of {FACTORS} exactly once")
    if Path(cfg["factors"]["raw_dir"]).parts[:2] != ("data", "raw"):
        raise ValueError("factors.raw_dir must be under data/raw (gitignored)")


def _check_models(cfg: dict[str, Any]) -> None:
    used: list[str] = []
    for name, spec in cfg["models"].items():
        _check_schema(spec, MODEL_SCHEMA, f"models.{name}")
        if not spec["adds"]:
            raise ValueError(f"models.{name}: each nested model must add a factor")
        used += spec["adds"]
    if sorted(used) != sorted(FACTORS) or len(set(used)) != len(used):
        raise ValueError(f"models must add each of {FACTORS} exactly once")
    if used[0] != "mkt_rf":
        raise ValueError("the first model must be the market model")


def _check_tests(cfg: dict[str, Any]) -> None:
    models, books = list(cfg["models"]), cfg["books"]
    test, span = cfg["primary_test"], cfg["own_reversal_spanning"]
    if test["model"] != models[-1] or span["model"] != models[-1]:
        raise ValueError("primary_test and own_reversal_spanning must use the largest model")
    if books[test["book"]]["role"] != "primary_attribution":
        raise ValueError("primary_test.book must be the book with role 'primary_attribution'")
    if not 0 < test["t_threshold"]:
        raise ValueError("primary_test.t_threshold must be positive")
    if span["book"] not in books or span["regressor_book"] not in books or span["book"] == span["regressor_book"]:
        raise ValueError("own_reversal_spanning needs two different declared books")
    if books[span["book"]]["construction"] != books[span["regressor_book"]]["construction"]:
        raise ValueError("own_reversal_spanning must compare books of one construction")
    if books[span["regressor_book"]]["role"] != "benchmark":
        raise ValueError("own_reversal_spanning.regressor_book must be a benchmark book")


def _check_risk_and_beta(cfg: dict[str, Any], fcfg: dict[str, Any]) -> None:
    risk, var_es = cfg["risk"], cfg["risk"]["var_es"]
    if risk["drawdown_episodes"] < 1:
        raise ValueError("risk.drawdown_episodes must be >= 1")
    if not var_es["confidence"] or not all(_is_type(c, NUM) and 0.5 < c < 1 for c in var_es["confidence"]):
        raise ValueError("risk.var_es.confidence must hold numbers between 0.5 and 1")
    if not var_es["horizons_days"] or not all(_is_type(h, int) and h >= 1 for h in var_es["horizons_days"]):
        raise ValueError("risk.var_es.horizons_days must hold positive integers")
    if set(var_es["units"]) != {"per_dollar_of_capital", "multiple_of_daily_vol"}:
        raise ValueError("risk.var_es.units must be per_dollar_of_capital and multiple_of_daily_vol")
    if set(risk["moments"]["skew_excess_kurtosis_on"]) != {"daily", "non_overlapping_h_day"}:
        raise ValueError("risk.moments must cover daily and non_overlapping_h_day returns")
    if set(risk["worst_periods"]) != {"day", "month"}:
        raise ValueError("risk.worst_periods must be day and month")
    beta = cfg["beta_analysis"]
    if sorted(beta["series"]) != sorted(PNL_SERIES):
        raise ValueError(f"beta_analysis.series must be {PNL_SERIES}")
    stale = beta["explanations"]["stale_beta"]
    if stale["short_beta_window"] < 2 or stale["short_beta_window"] >= fcfg["features"]["beta_window"]:
        raise ValueError("stale_beta.short_beta_window must be shorter than the beta_252d window")
    if stale["min_obs_frac"] != fcfg["features"]["min_obs_frac"]:
        raise ValueError("stale_beta.min_obs_frac must equal features.min_obs_frac")


def _check_outputs(cfg: dict[str, Any]) -> None:
    out, data = cfg["outputs"], cfg["book_data"]
    if Path(out["reports_dir"]).parts[:1] != ("reports",) or Path(out["figures_dir"]).parent != Path(out["reports_dir"]):
        raise ValueError("outputs.reports_dir must be under reports/ and figures_dir inside it")
    if out["reports_dir"] == "reports/portfolio":
        raise ValueError("outputs.reports_dir must not overwrite reports/portfolio")
    for key in ("daily_returns", "weights"):
        if not data[key].startswith("data/processed/") or "{signal}" not in data[key]:
            raise ValueError(f"book_data.{key} must be a per-signal path in data/processed (gitignored)")


def load_attribution_config(
    path: str | Path | None = None,
    pcfg: dict[str, Any] | None = None,
    fcfg: dict[str, Any] | None = None,
    rcfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load ``configs/attribution.yaml`` and reject anything outside the pre-registered design.

    The portfolio, feature and research configs default to the paths named in the attribution
    config. No book return, factor value or panel row is read.
    """
    with open(path or ATTRIBUTION_CONFIG_PATH) as handle:
        cfg = yaml.safe_load(handle)
    _check_schema(cfg, SCHEMA, "")
    for key, expected in FIXED.items():
        if _get(cfg, key) != expected:
            raise ValueError(f"{key}: only {expected!r} is defined")
    fcfg = fcfg or load_config(REPO_ROOT / cfg["features_config"])
    rcfg = rcfg or load_research_config(REPO_ROOT / cfg["research_config"], fcfg)
    pcfg = pcfg or load_portfolio_config(REPO_ROOT / cfg["portfolio_config"], fcfg, rcfg)
    _check_books(cfg, pcfg)
    _check_factors(cfg)
    _check_models(cfg)
    _check_tests(cfg)
    _check_risk_and_beta(cfg, fcfg)
    _check_outputs(cfg)
    return cfg


# --------------------------------------------------------------------------- factor files


def source_filename(source: dict[str, Any]) -> str:
    return source["url"].rsplit("/", 1)[-1]


def parse_french_csv(text: str, columns: dict[str, str], scale: float, cut_before: pd.Timestamp) -> pd.DataFrame:
    """The daily rows of a Ken French CSV before ``cut_before``, renamed and scaled.

    The header is the line naming every column; data rows start with a yyyymmdd date and end at the
    first other line. A later row is skipped before its values are converted. -99.99 and -999 mark
    missing values.
    """
    lines = text.splitlines()
    header = None
    for pos, line in enumerate(lines):
        cells = [c.strip() for c in line.split(",")]
        if set(columns) <= set(cells):
            header, first = cells, pos + 1
            break
    if header is None:
        raise ValueError(f"no header with columns {sorted(columns)}")
    index = {name: header.index(name) for name in columns}
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for line in lines[first:]:
        cells = [c.strip() for c in line.split(",")]
        if not (len(cells[0]) == 8 and cells[0].isdigit()):
            break
        date = pd.Timestamp(cells[0])
        if date >= cut_before:
            continue
        rows[date] = {new: float(cells[index[old]]) for old, new in columns.items()}
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return frame.where(frame > -99.0) * scale


def parse_aqr_xlsx(path: Path, sheet: str, columns: dict[str, str], scale: float, cut_before: pd.Timestamp) -> pd.DataFrame:
    """The daily rows of an AQR factor workbook before ``cut_before``; reading stops at the first later row."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows_iter = workbook[sheet].iter_rows(values_only=True)
        index = None
        for row in rows_iter:
            cells = [str(c).strip() if c is not None else "" for c in row]
            if cells and cells[0].upper() == "DATE" and set(columns) <= set(cells):
                index = {name: cells.index(name) for name in columns}
                break
        if index is None:
            raise ValueError(f"{path.name}: no DATE header with columns {sorted(columns)}")
        rows: dict[pd.Timestamp, dict[str, float]] = {}
        for row in rows_iter:
            if row[0] is None:
                continue
            date = pd.Timestamp(row[0])
            if date >= cut_before:
                break
            rows[date] = {new: float(row[index[old]]) if row[index[old]] not in (None, "") else np.nan
                          for old, new in columns.items()}
    finally:
        workbook.close()
    return pd.DataFrame.from_dict(rows, orient="index").sort_index() * scale


def load_factors(cfg: dict[str, Any], raw_dir: Path, holdout_start: pd.Timestamp) -> pd.DataFrame:
    """All factors by date, strictly before the holdout."""
    frames = []
    for source in cfg["factors"]["sources"].values():
        path = raw_dir / source_filename(source)
        if source["format"] == "french_csv_zip":
            with zipfile.ZipFile(path) as archive:
                text = io.TextIOWrapper(archive.open(archive.namelist()[0]), encoding="utf-8", errors="replace").read()
            frames.append(parse_french_csv(text, source["columns"], source["scale"], holdout_start))
        else:
            frames.append(parse_aqr_xlsx(path, source["sheet"], source["columns"], source["scale"], holdout_start))
    factors = pd.concat(frames, axis=1).sort_index()[list(FACTORS)]
    if len(factors) and factors.index.max() >= holdout_start:
        raise AssertionError("factor parser returned a row on or after holdout_start")
    return factors


# --------------------------------------------------------------------------- regressions


def ols_newey_west(y: np.ndarray, X: np.ndarray, names: list[str], lag: int) -> dict[str, Any]:
    """OLS of ``y`` on a constant and ``X`` with a Bartlett-kernel HAC covariance (no small-sample correction).

    Same estimator as ``src.portfolio_report.regression_beta``. Returns ``coef``, ``se`` and ``t``
    series indexed by ``alpha`` and ``names``, ``r2``, ``resid`` and ``n``.
    """
    y, X = np.asarray(y, dtype="float64"), np.asarray(X, dtype="float64")
    if not (np.isfinite(y).all() and np.isfinite(X).all()):
        raise ValueError("regression inputs must be complete: a missing value is an error, never dropped")
    n = len(y)
    Z = np.column_stack([np.ones(n), X])
    bread = np.linalg.inv(Z.T @ Z)
    coef = bread @ (Z.T @ y)
    resid = y - Z @ coef
    score = Z * resid[:, None]
    meat = score.T @ score
    for j in range(1, min(lag, n - 1) + 1):
        gamma = score[j:].T @ score[:-j]
        meat += (1 - j / (lag + 1)) * (gamma + gamma.T)
    se = np.sqrt(np.diag(bread @ meat @ bread))
    terms = ["alpha", *names]
    total = float(((y - y.mean()) ** 2).sum())
    return {"coef": pd.Series(coef, terms), "se": pd.Series(se, terms), "t": pd.Series(coef / se, terms),
            "r2": 1 - float(resid @ resid) / total if total > 0 else np.nan, "resid": resid, "n": n}


def primary_outcome(reversal_beta: float, reversal_t: float, alpha_t: float, threshold: float) -> str:
    """The pre-registered outcome of the confirmatory test; the four cases cover every input."""
    present = reversal_beta > 0 and reversal_t >= threshold
    if present:
        return "reversal_partial" if alpha_t >= threshold else "reversal_explained"
    return "not_reversal" if alpha_t >= threshold else "inconclusive"


def nested_models(cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Each model's full factor list, in config order."""
    out: dict[str, list[str]] = {}
    factors: list[str] = []
    for name, spec in cfg["models"].items():
        factors = factors + spec["adds"]
        out[name] = factors
    return out


def factor_tables(y: pd.Series, factors: pd.DataFrame, models: dict[str, list[str]], horizon: int,
                  trading_days: int, ids: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Summary rows and long loading rows of every nested model, with the exact mean decomposition."""
    model_rows, loading_rows = [], []
    lag = newey_west_lag(len(y), horizon)
    for model, names in models.items():
        fit = ols_newey_west(y.to_numpy(), factors[names].to_numpy(), names, lag)
        means = pd.Series({"alpha": 1.0, **factors[names].mean().to_dict()})
        contribution = fit["coef"] * means
        gap = float(y.mean() - contribution.sum())
        resid_vol = float(fit["resid"].std(ddof=1) * np.sqrt(trading_days))
        alpha_ann = float(fit["coef"]["alpha"] * trading_days)
        model_rows.append({**ids, "model": model, "n_days": fit["n"], "nw_lag": lag,
                           "mean_return_ann": float(y.mean() * trading_days), "alpha_ann": alpha_ann,
                           "alpha_t": float(fit["t"]["alpha"]), "r2": fit["r2"], "resid_vol_ann": resid_vol,
                           "information_ratio": alpha_ann / resid_vol if resid_vol > 0 else np.nan,
                           "decomposition_gap": gap})
        for term in fit["coef"].index:
            loading_rows.append({**ids, "model": model, "term": term, "coef": float(fit["coef"][term]),
                                 "se": float(fit["se"][term]), "t": float(fit["t"][term]),
                                 "contribution_bp_per_year": float(contribution[term] * trading_days * 1e4)})
    return model_rows, loading_rows


# --------------------------------------------------------------------------- risk


def drawdown_episodes(pnl: pd.Series, n: int) -> list[dict[str, Any]]:
    """The ``n`` deepest peak-to-trough falls of cumulative P&L, which starts at zero before the first date."""
    equity = pnl.cumsum()
    high = equity.cummax().clip(lower=0.0)
    under = equity < high
    block = (~under).cumsum()
    episodes = []
    for _, days in equity[under].groupby(block[under]):
        depth = float((days - high[days.index]).min())
        trough = (days - high[days.index]).idxmin()
        before = equity.index[equity.index < days.index[0]]
        after = equity.index[equity.index > days.index[-1]]
        episodes.append({"depth": depth, "peak": before[-1] if len(before) else None, "trough": trough,
                         "recovery": after[0] if len(after) else None,
                         "days_to_trough": int(equity.index.get_loc(trough) - equity.index.get_loc(days.index[0]) + 1),
                         "days_under_water": len(days)})
    return sorted(episodes, key=lambda e: e["depth"])[:n]


def var_es(pnl: pd.Series, confidence: float, horizon: int) -> dict[str, float]:
    """Historical VaR and ES as positive losses; ``horizon`` > 1 uses overlapping rolling sums."""
    x = pnl.rolling(horizon).sum().dropna().to_numpy() if horizon > 1 else pnl.to_numpy()
    cut = float(np.quantile(x, 1 - confidence))
    return {"var": -cut, "es": -float(x[x <= cut].mean())}


def moments(pnl: pd.Series, horizon: int) -> dict[str, float]:
    """Skew and excess kurtosis of daily returns and of consecutive non-overlapping h-day sums."""
    blocks = pnl.groupby(np.arange(len(pnl)) // horizon).sum()
    blocks = blocks[pnl.groupby(np.arange(len(pnl)) // horizon).size() == horizon]
    return {"skew_daily": float(stats.skew(pnl)), "excess_kurtosis_daily": float(stats.kurtosis(pnl)),
            "skew_h_day": float(stats.skew(blocks)), "excess_kurtosis_h_day": float(stats.kurtosis(blocks))}


def worst_periods(pnl: pd.Series) -> dict[str, Any]:
    monthly = pnl.groupby(pnl.index.to_period("M")).sum()
    return {"worst_day": float(pnl.min()), "worst_day_date": pnl.idxmin(), "worst_month": float(monthly.min()),
            "worst_month_period": str(monthly.idxmin())}


# --------------------------------------------------------------------------- beta


def weights_exposure(weights: pd.DataFrame, beta: pd.DataFrame, dates: pd.DatetimeIndex, n_tranches: int) -> pd.Series:
    """Daily ``sum_i w_i beta_i`` from rebalance weights, with beta known at the score date.

    ``weights`` holds per-tranche target weights per $ of the tranche's capital. Each tranche's
    exposure is held from its trade date to its next one, and tranches count ``1 / n_tranches``.
    A held name without a beta counts zero, as in the backtest.
    """
    long = beta.stack().rename("beta").rename_axis(["score_date", "ticker"]).reset_index()
    merged = weights.merge(long, on=["score_date", "ticker"], how="left")
    merged["exposure"] = merged["weight"] * merged["beta"].fillna(0.0)
    by_tranche = merged.groupby(["trade_date", "tranche"])["exposure"].sum().unstack("tranche").sort_index()
    held = by_tranche.reindex(by_tranche.index.union(dates)).ffill()
    return (held.fillna(0.0).sum(axis=1) / n_tranches).reindex(dates)


def conditional_beta(y: pd.Series, market: pd.Series, exposure: pd.Series, lag: int) -> dict[str, float]:
    """``r_t = a + b0 m_t + b1 X_(t-1) m_t``; t-statistics test ``b0 = 0`` and ``b1 = 1``."""
    frame = pd.DataFrame({"y": y, "m": market, "x": exposure.shift(1)}).dropna()
    fit = ols_newey_west(frame["y"].to_numpy(), np.column_stack([frame["m"], frame["x"] * frame["m"]]), ["b0", "b1"], lag)
    return {"b0": float(fit["coef"]["b0"]), "b0_t_vs_0": float(fit["t"]["b0"]), "b1": float(fit["coef"]["b1"]),
            "b1_t_vs_1": float((fit["coef"]["b1"] - 1) / fit["se"]["b1"]), "n_days": fit["n"]}


def up_down_beta(y: pd.Series, market: pd.Series, lag: int) -> dict[str, float]:
    """Separate market betas on up and down days, and the t-statistic of their difference."""
    m = market.to_numpy()
    up = np.where(m > 0, m, 0.0)
    fit = ols_newey_west(y.to_numpy(), np.column_stack([m, up]), ["down", "up_minus_down"], lag)
    return {"beta_down": float(fit["coef"]["down"]), "beta_down_t": float(fit["t"]["down"]),
            "beta_up": float(fit["coef"]["down"] + fit["coef"]["up_minus_down"]),
            "up_minus_down": float(fit["coef"]["up_minus_down"]), "up_minus_down_t": float(fit["t"]["up_minus_down"])}
