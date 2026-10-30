"""Sensitivity: the ``configs/sensitivity.yaml`` loader and the statistics.

The design is fixed in the config before any bootstrap, repricing or per-offset statistic is
computed. The loader is strict: an unknown or missing key, a wrong type, a method the design does
not define, a grid that loses its base point, or a primary interval that differs from the declared
ones is an error, never silently ignored. Only recorded base-case books are read; no backtest is
rerun and the optimiser is not re-solved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.metrics import newey_west_lag, newey_west_tstat, sharpe
from src.portfolio import _check_schema, _get, _is_type, _positive_numbers, load_portfolio_config
from src.signals import load_research_config
from src.utils import REPO_ROOT, load_config

SENSITIVITY_CONFIG_PATH = REPO_ROOT / "configs" / "sensitivity.yaml"
PNL_SERIES = ("gross_pnl", "net_pnl")
MIN_RESAMPLES = 1000

NUM = (int, float)

# Expected keys and leaf types. A dict is a section whose keys must match exactly.
SCHEMA: dict[str, Any] = {
    "portfolio_config": str,
    "research_config": str,
    "features_config": str,
    "book_data": {
        "daily_returns": str, "portfolio_metadata": str, "summary_table": str, "scenario": str,
        "signals": str, "constructions": str,
    },
    "windows": {"evaluation": str, "tuning": str},
    "bootstrap": {
        "method": str, "series": list, "statistic": str, "n_resamples": int, "seed": int,
        "mean_block_lengths": list, "confidence": NUM, "interval": str,
        "paired_differences": {"minuend": str, "subtrahend": str},
    },
    "primary_intervals": {
        "headline": {"signal": str, "construction": str, "series": str, "window": str},
        "lightgbm_minus_b0": {"construction": str, "series": str, "window": str},
    },
    "repricing": {
        "method": str, "impact_y": list, "borrow_bps_per_year": list, "grid": str, "statistics": list,
        "breakeven": list,
    },
    "offsets": {
        "source": str, "series": list, "scale_to_tranche_capital": bool, "statistics": list,
        "dispersion": list, "inference": str,
    },
    "checks": {
        "reproduce_sharpes_tolerance": NUM, "reprice_base_tolerance": NUM, "tranches_sum_tolerance": NUM,
        "paired_dates": str, "last_date": str,
    },
    "outputs": {"reports_dir": str, "figures_dir": str, "notebook": str},
}

# Method choices that the design defines; anything else is rejected.
FIXED = {
    "book_data.scenario": "base",
    "book_data.signals": "all_in_portfolio_config",
    "book_data.constructions": "all_in_portfolio_config",
    "windows.evaluation": "primary",
    "windows.tuning": "secondary",
    "bootstrap.method": "stationary",
    "bootstrap.statistic": "annualised_sharpe",
    "bootstrap.interval": "percentile",
    "primary_intervals.headline.series": "net_pnl",
    "primary_intervals.headline.window": "evaluation",
    "primary_intervals.lightgbm_minus_b0.series": "net_pnl",
    "primary_intervals.lightgbm_minus_b0.window": "evaluation",
    "repricing.method": "base_trades_held_fixed",
    "repricing.grid": "full_cross",
    "offsets.source": "tranche_rows",
    "offsets.inference": "none",
    "checks.paired_dates": "identical",
    "checks.last_date": "before_holdout_start",
}
REPRICING_STATISTICS = {"net_sharpe", "net_nw_t", "net_return_ann"}
BREAKEVENS = {"impact_y", "borrow_bps_per_year"}
OFFSET_STATISTICS = {"sharpe", "return_ann", "turnover_ann"}
DISPERSION = {"min", "max", "range", "sd"}


def _check_bootstrap(cfg: dict[str, Any], pcfg: dict[str, Any]) -> None:
    boot = cfg["bootstrap"]
    if sorted(boot["series"]) != sorted(PNL_SERIES):
        raise ValueError(f"bootstrap.series must be {PNL_SERIES}")
    if boot["n_resamples"] < MIN_RESAMPLES:
        raise ValueError(f"bootstrap.n_resamples must be at least {MIN_RESAMPLES}")
    if not 0.5 < boot["confidence"] < 1:
        raise ValueError("bootstrap.confidence must be between 0.5 and 1")
    lengths = boot["mean_block_lengths"]
    if not lengths or not all(_is_type(n, int) for n in lengths) or len(set(lengths)) != len(lengths):
        raise ValueError("bootstrap.mean_block_lengths must be distinct integers")
    longest = max(s["horizon"] for s in pcfg["signals"].values())
    if min(lengths) < longest:
        raise ValueError(f"bootstrap.mean_block_lengths must reach the longest horizon, {longest}")
    if lengths[0] != min(lengths):
        raise ValueError("the first (primary) block length must be the shortest")
    pair, head = boot["paired_differences"], pcfg["headline"]
    if pair["minuend"] != head["signal"] or pair["subtrahend"] != head["benchmark"]:
        raise ValueError("bootstrap.paired_differences must be the headline signal minus its benchmark")


def _check_primary_intervals(cfg: dict[str, Any], pcfg: dict[str, Any]) -> None:
    primary, head = cfg["primary_intervals"], pcfg["headline"]
    if (primary["headline"]["signal"], primary["headline"]["construction"]) != (head["signal"], head["construction"]):
        raise ValueError("primary_intervals.headline must be the portfolio config's headline book")
    if primary["lightgbm_minus_b0"]["construction"] != head["construction"]:
        raise ValueError("primary_intervals.lightgbm_minus_b0 must use the headline construction")


def _check_repricing(cfg: dict[str, Any], pcfg: dict[str, Any]) -> None:
    rep, costs = cfg["repricing"], pcfg["costs"]
    _positive_numbers("repricing.impact_y", rep["impact_y"])
    _positive_numbers("repricing.borrow_bps_per_year", rep["borrow_bps_per_year"], allow_zero=True)
    if costs["impact"]["y"] not in rep["impact_y"]:
        raise ValueError("repricing.impact_y must hold the base costs.impact.y")
    if costs["borrow_bps_per_year"] not in rep["borrow_bps_per_year"]:
        raise ValueError("repricing.borrow_bps_per_year must hold the base costs.borrow_bps_per_year")
    if set(rep["statistics"]) != REPRICING_STATISTICS or set(rep["breakeven"]) != BREAKEVENS:
        raise ValueError(f"repricing reports {sorted(REPRICING_STATISTICS)} and break-evens {sorted(BREAKEVENS)}")


def _check_offsets(cfg: dict[str, Any]) -> None:
    off = cfg["offsets"]
    if sorted(off["series"]) != sorted(PNL_SERIES):
        raise ValueError(f"offsets.series must be {PNL_SERIES}")
    if not off["scale_to_tranche_capital"]:
        raise ValueError("offsets.scale_to_tranche_capital: each offset is reported as its own book")
    if set(off["statistics"]) != OFFSET_STATISTICS or set(off["dispersion"]) != DISPERSION:
        raise ValueError(f"offsets report {sorted(OFFSET_STATISTICS)} and dispersion {sorted(DISPERSION)}")


def _check_tolerances_and_outputs(cfg: dict[str, Any]) -> None:
    for key in ("reproduce_sharpes_tolerance", "reprice_base_tolerance", "tranches_sum_tolerance"):
        if not 0 < cfg["checks"][key] < 1e-6:
            raise ValueError(f"checks.{key} must be positive and below 1e-6")
    out = cfg["outputs"]
    if Path(out["reports_dir"]).parts[:1] != ("reports",) or Path(out["figures_dir"]).parent != Path(out["reports_dir"]):
        raise ValueError("outputs.reports_dir must be under reports/ and figures_dir inside it")
    if out["reports_dir"] in ("reports/portfolio", "reports/attribution"):
        raise ValueError("outputs.reports_dir must not overwrite earlier results")
    if not cfg["book_data"]["daily_returns"].startswith("data/processed/") or "{signal}" not in cfg["book_data"]["daily_returns"]:
        raise ValueError("book_data.daily_returns must be a per-signal path in data/processed (gitignored)")


def load_sensitivity_config(
    path: str | Path | None = None,
    pcfg: dict[str, Any] | None = None,
    fcfg: dict[str, Any] | None = None,
    rcfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load ``configs/sensitivity.yaml`` and reject anything outside the pre-registered design.

    The portfolio, feature and research configs default to the paths named in the sensitivity
    config. No book return is read.
    """
    with open(path or SENSITIVITY_CONFIG_PATH) as handle:
        cfg = yaml.safe_load(handle)
    _check_schema(cfg, SCHEMA, "")
    for key, expected in FIXED.items():
        if _get(cfg, key) != expected:
            raise ValueError(f"{key}: only {expected!r} is defined")
    fcfg = fcfg or load_config(REPO_ROOT / cfg["features_config"])
    rcfg = rcfg or load_research_config(REPO_ROOT / cfg["research_config"], fcfg)
    pcfg = pcfg or load_portfolio_config(REPO_ROOT / cfg["portfolio_config"], fcfg, rcfg)
    _check_bootstrap(cfg, pcfg)
    _check_primary_intervals(cfg, pcfg)
    _check_repricing(cfg, pcfg)
    _check_offsets(cfg)
    _check_tolerances_and_outputs(cfg)
    return cfg


# --------------------------------------------------------------------------- bootstrap


def stationary_indices(n: int, n_resamples: int, mean_block: int, seed: int) -> np.ndarray:
    """Resampled day positions, ``n_resamples`` x ``n``: the Politis-Romano stationary bootstrap.

    Each day starts a new block with probability ``1 / mean_block`` and otherwise follows the day
    before, wrapping from the last day to the first. The generator depends only on ``seed``, ``n``
    and ``mean_block``, so every book with the same dates gets the same resampled days.
    """
    rng = np.random.default_rng([seed, n, mean_block])
    jumps = rng.integers(0, n, size=(n_resamples, n), dtype=np.int32)
    keep = rng.random((n_resamples, n)) >= 1 / mean_block
    keep[:, 0] = False
    index = jumps
    for t in range(1, n):
        follow = keep[:, t]
        index[follow, t] = (index[follow, t - 1] + 1) % n
    return index


def resampled_sharpes(x: np.ndarray, index: np.ndarray, trading_days: int) -> np.ndarray:
    """Annualised Sharpe of ``x`` in every resample, computed as ``src.metrics.sharpe`` does."""
    draws = np.asarray(x, dtype="float64")[index]
    sd = draws.std(axis=1, ddof=1)
    return np.where(sd > 0, draws.mean(axis=1) / np.where(sd > 0, sd, 1.0) * np.sqrt(trading_days), np.nan)


def percentile_interval(samples: np.ndarray, confidence: float) -> dict[str, Any]:
    """Percentile interval of bootstrap draws, their sd and the share above zero."""
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return {"ci_low": float(low), "ci_high": float(high), "resample_sd": float(samples.std(ddof=1)),
            "share_above_zero": float((samples > 0).mean()), "contains_zero": bool(low <= 0 <= high)}


# --------------------------------------------------------------------------- repricing


def reprice(frame: pd.DataFrame, y: float, borrow_bps: float, base_y: float, base_borrow_bps: float) -> pd.Series:
    """Daily net return of the recorded trades with impact and borrow rescaled."""
    if base_y <= 0 or base_borrow_bps <= 0:
        raise ValueError("repricing needs positive base values of Y and borrow")
    return (frame["gross_pnl"] - frame["commission"] - frame["spread"]
            - frame["impact"] * (y / base_y) - frame["borrow"] * (borrow_bps / base_borrow_bps))


def breakevens(frame: pd.DataFrame, base_y: float, base_borrow_bps: float) -> dict[str, float]:
    """The Y and the borrow rate at which the mean net return is zero, the other input at base.

    A negative value means the book loses money with that cost at zero.
    """
    edge = (frame["gross_pnl"] - frame["commission"] - frame["spread"]).mean()
    impact, borrow = frame["impact"].mean(), frame["borrow"].mean()
    return {"breakeven_impact_y": float((edge - borrow) / impact * base_y) if impact > 0 else np.nan,
            "breakeven_borrow_bps_per_year": float((edge - impact) / borrow * base_borrow_bps) if borrow > 0 else np.nan}


def repricing_rows(frame: pd.DataFrame, cfg: dict[str, Any], base_y: float, base_borrow_bps: float, horizon: int,
                   trading_days: int, ids: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per grid point, with the NW lag rule of ``src.backtest.performance``."""
    lag = newey_west_lag(len(frame), horizon)
    rows = []
    for y in cfg["repricing"]["impact_y"]:
        for borrow in cfg["repricing"]["borrow_bps_per_year"]:
            net = reprice(frame, y, borrow, base_y, base_borrow_bps)
            rows.append({**ids, "impact_y": y, "borrow_bps_per_year": borrow,
                         "is_base": y == base_y and borrow == base_borrow_bps, "net_sharpe": sharpe(net, trading_days),
                         "net_nw_t": newey_west_tstat(net.to_numpy(), lag)["t"],
                         "net_return_ann": float(net.mean() * trading_days)})
    return rows


# --------------------------------------------------------------------------- offsets


def offset_rows(tranches: pd.DataFrame, trading_days: int, ids: dict[str, Any], series: list[str]) -> list[dict[str, Any]]:
    """Each offset as its own book: tranche rows are per $ of total capital, so returns scale by n."""
    n = tranches["tranche"].nunique()
    rows = []
    for tranche, frame in tranches.groupby("tranche", sort=False):
        for pnl in series:
            rows.append({**ids, "series": pnl, "offset": int(tranche), "n_days": len(frame),
                         "sharpe": sharpe(frame[pnl], trading_days),
                         "return_ann": float(frame[pnl].mean() * trading_days * n),
                         "turnover_ann": float(frame["turnover"].mean() * trading_days * n)})
    return rows


def offset_dispersion(offsets: pd.DataFrame, totals: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Minimum, maximum, range and sd of each statistic across a book's offsets, next to its total."""
    rows = []
    for key, group in offsets.groupby(keys, sort=False):
        ids = dict(zip(keys, key))
        total = totals.loc[(totals[keys] == pd.Series(ids)).all(axis=1)].iloc[0]
        for statistic in sorted(OFFSET_STATISTICS):
            x = group[statistic]
            rows.append({**ids, "statistic": statistic, "n_offsets": len(x), "total": float(total[statistic]),
                         "min": float(x.min()), "max": float(x.max()), "range": float(x.max() - x.min()),
                         "sd": float(x.std(ddof=1))})
    return pd.DataFrame(rows)
