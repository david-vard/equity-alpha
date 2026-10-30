"""Risk and attribution: the ``configs/attribution.yaml`` loader.

The design is fixed in the config before any factor file is downloaded. The loader is strict: an
unknown or missing key, a wrong type, a method the design does not define, a book the portfolio
config does not define, or a test that does not match the pre-registered question is an error,
never silently ignored. Only recorded base-case books are described; no backtest is rerun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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
