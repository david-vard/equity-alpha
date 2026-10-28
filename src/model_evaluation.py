"""Out-of-sample evaluation of model predictions: IC, paired comparisons, deciles, importance.

Wide-frame statistics reuse ``src.signals`` and ``src.diagnostics``; the long-format helpers here
work on row arrays (date id per row) and are used for hyperparameter selection and importance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.diagnostics import beta_hedged_spread, dispersion_weighted_ic, monotonicity, rank_by_date, spread_stats
from src.metrics import newey_west_lag, newey_west_tstat, non_overlapping_positions
from src.signals import (
    cross_sectional_ic,
    ic_summary,
    quantile_buckets,
    quantile_returns,
    turnover,
)


def daily_ic_long(pred: np.ndarray, label: np.ndarray, date_ids: np.ndarray, min_names: int) -> pd.Series:
    """Daily Spearman IC from row arrays; same definition as ``cross_sectional_ic``."""
    df = pd.DataFrame({"d": date_ids, "p": pred, "y": label}).dropna()
    g = df.groupby("d", sort=True)
    rp, ry = g["p"].rank(), g["y"].rank()
    rp = rp - rp.groupby(df["d"]).transform("mean")
    ry = ry - ry.groupby(df["d"]).transform("mean")
    num = (rp * ry).groupby(df["d"]).sum()
    den = np.sqrt((rp * rp).groupby(df["d"]).sum() * (ry * ry).groupby(df["d"]).sum())
    n = g.size()
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = (num / den).where((n >= min_names) & (den > 0))
    return ic.rename("ic")


def mean_daily_ic(pred: np.ndarray, label: np.ndarray, date_ids: np.ndarray, min_names: int) -> float:
    ic = daily_ic_long(pred, label, date_ids, min_names)
    return float(ic.mean()) if ic.notna().any() else float("nan")


def ic_row(ic: pd.Series, h: int, label: pd.DataFrame, rcfg: dict[str, Any]) -> dict[str, Any]:
    """IC mean, IR, hit rate, NW t (``max(h, plug-in)`` lags) and the dispersion-weighted mean."""
    s = ic_summary(ic, h, 1, rcfg)
    disp = dispersion_weighted_ic(ic, label.std(axis=1))
    return {
        "n_dates": s["n_dates"], "ic_mean": s["ic_mean"], "ic_std": s["ic_std"], "ic_ir": s["ic_ir"],
        "ic_ir_ann_non_overlapping": s["no_ic_ir_ann"], "hit_rate": s["hit_rate"], "nw_lag": s["nw_lag"],
        "nw_t": s["nw_t"], "p_value": s["p_value"],
        "ic_mean_dispersion_weighted": disp["ic_mean_dispersion_weight"],
    }


def oos_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Pooled out-of-sample R^2 against a zero forecast: ``1 - sum (y - p)^2 / sum y^2``."""
    ok = ~np.isnan(pred) & ~np.isnan(target)
    y, p = target[ok], pred[ok]
    return float(1 - ((y - p) ** 2).sum() / (y**2).sum()) if ok.any() else float("nan")


def paired_ic_test(ic_a: pd.Series, ic_b: pd.Series, h: int) -> dict[str, float]:
    """NW t of the mean daily IC difference ``a - b`` on dates where both exist (``max(h, plug-in)``)."""
    diff = (ic_a - ic_b).dropna().to_numpy(dtype="float64")
    nw = newey_west_tstat(diff, newey_west_lag(len(diff), h))
    return {"n_dates": nw["n"], "ic_diff_mean": nw["mean"], "nw_t": nw["t"], "p_value": nw["p"]}


def decile_evaluation(
    pred: pd.DataFrame, label: pd.DataFrame, beta: pd.DataFrame, h: int, q: int, min_names: int,
    trading_days: int, offset: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Gross deciles of the prediction on non-overlapping rebalance dates.

    Spreads: ``beta_hedged`` (each leg hedged with its mean ``beta_252d`` against the equal-weighted
    market return), ``rank`` (mean within-date return rank, [-0.5, 0.5]) and ``raw``. Returns the
    spread rows, the decile-mean rows and turnover: rank autocorrelation of the prediction between
    consecutive rebalances and the one-way turnover of the bottom / top legs.
    """
    pos = non_overlapping_positions(len(pred), h, offset)
    buckets = quantile_buckets(pred, q, min_names)
    b, y = buckets.iloc[pos], label.iloc[pos]
    market = y.mean(axis=1)
    means = {"raw": quantile_returns(b, y, q), "rank": quantile_returns(b, rank_by_date(y), q)}
    betas = quantile_returns(b, beta.iloc[pos].where(y.notna()), q)
    means["beta_hedged"] = means["raw"] - betas.mul(market, axis=0)
    spreads = {k: v.iloc[:, -1] - v.iloc[:, 0] for k, v in means.items()}
    spreads["beta_hedged"] = beta_hedged_spread(means["raw"], betas, market)
    spread_rows, decile_rows = [], []
    for variant in ("beta_hedged", "rank", "raw"):
        stats = spread_stats(spreads[variant].to_numpy(), trading_days / h)
        spread_rows.append({"spread": variant, **stats, "monotonicity": monotonicity(means[variant].mean())})
        for decile, value in means[variant].mean().items():
            decile_rows.append({"spread": variant, "decile": decile, "mean": value})
    sub = pred.iloc[pos]
    autocorr = cross_sectional_ic(sub, sub.shift(1), min_names)
    legs = turnover(buckets, q, h, offset)
    to = {
        "rank_autocorrelation": float(autocorr.mean()),
        "turnover_D1": float(legs.iloc[:, 0].mean()),
        f"turnover_D{q}": float(legs.iloc[:, 1].mean()),
        "beta_gap_top_minus_bottom": float((betas.iloc[:, -1] - betas.iloc[:, 0]).mean()),
    }
    return spread_rows, decile_rows, to


def permutation_importance(
    model: Any, X: np.ndarray, label: np.ndarray, date_ids: np.ndarray, features: list[str],
    repeats: int, min_names: int, seed: int,
) -> list[dict[str, Any]]:
    """Drop in mean daily IC when one feature is shuffled across names within each date."""
    rng = np.random.default_rng(seed)
    base = mean_daily_ic(model.predict(X), label, date_ids, min_names)
    order = np.argsort(date_ids, kind="stable")
    d_sorted = date_ids[order]
    starts = np.flatnonzero(np.r_[True, d_sorted[1:] != d_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    rows = []
    for j, name in enumerate(features):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            col = Xp[order, j]
            for s, e in zip(starts, ends):
                col[s:e] = col[s:e][rng.permutation(e - s)]
            Xp[order, j] = col
            drops.append(base - mean_daily_ic(model.predict(Xp), label, date_ids, min_names))
        rows.append({"feature": name, "base_ic": base, "ic_drop_mean": float(np.mean(drops)),
                     "ic_drop_std": float(np.std(drops))})
    return rows


def shap_summary(booster: Any, X: np.ndarray, features: list[str]) -> list[dict[str, Any]]:
    """Mean |SHAP| per feature and the direction: correlation of SHAP value with feature value."""
    import shap

    values = shap.TreeExplainer(booster).shap_values(np.asarray(X, dtype="float32"))
    rows = []
    for j, name in enumerate(features):
        x, s = X[:, j], values[:, j]
        ok = ~np.isnan(x)
        direction = float(np.corrcoef(x[ok], s[ok])[0, 1]) if ok.sum() > 2 and s[ok].std() > 0 else np.nan
        rows.append({"feature": name, "mean_abs_shap": float(np.abs(s).mean()), "shap_value_corr": direction})
    return rows
