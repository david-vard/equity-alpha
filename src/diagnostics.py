"""Pre-model diagnostics: label compounding, IC across horizons, robust decile spreads, break-even costs.

Same wide ``date x ticker`` conventions as ``src.signals``: panel dates are consecutive trading days,
so shifting a frame by ``k`` rows moves it ``k`` trading days. Design notes are in
``reports/diagnostics/SUMMARY.md``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.metrics import newey_west_lag, newey_west_tstat, sharpe

TERMINAL, COMPLETE = "terminal_exit", "complete"
DAILY_MISSING, INTERIOR_GAP, LEFT_PANEL = "daily_label_missing", "interior_gap", "not_in_panel_at_window_end"


# --------------------------------------------------------------------------- label compounding


def compound_daily_labels(ret_1d: pd.DataFrame, n_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compound the 1-day forward returns of holding days 1..n_days onto row ``t``.

    Row ``t + k`` of ``fwd_ret_1d`` is the return of holding day ``k + 1`` of a position entered at
    the close of ``t + lag``, so the product over ``k = 0..n_days-1`` telescopes to
    ``fwd_ret_{n_days}d`` at ``t``. Returns the compounded return (NaN unless all ``n_days`` daily
    returns exist) and the sum of log returns, both float64.
    """
    log_sum = np.zeros(ret_1d.shape)
    complete = np.ones(ret_1d.shape, dtype=bool)
    for k in range(n_days):
        r = ret_1d.shift(-k).to_numpy(dtype="float64")
        complete &= ~np.isnan(r)
        log_sum += np.where(np.isnan(r), 0.0, np.log1p(r))
    log_sum[~complete] = np.nan
    log_frame = pd.DataFrame(log_sum, index=ret_1d.index, columns=ret_1d.columns)
    return np.expm1(log_frame), log_frame


def compounding_check(
    ret_1d: pd.DataFrame,
    label: pd.DataFrame,
    terminal: pd.DataFrame,
    present: pd.DataFrame,
    n_days: int,
    rel_tol: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare compounded daily labels with the stored ``n_days`` label, cell by cell.

    Every cell with a stored label gets one category: ``terminal_exit`` (flagged delisting exit),
    ``complete`` (all daily returns exist, so the two must agree), ``daily_label_missing`` (rows
    exist but a daily label is NaN), ``interior_gap`` (a row is missing inside the window but the
    name is back by its end) or ``not_in_panel_at_window_end`` (left the index or stopped trading).
    The error is ``(1 + compounded) / (1 + label) - 1``; a ``complete`` cell above ``rel_tol`` is a
    mismatch. Returns the per-category table and the compounded frame.
    """
    compounded, _ = compound_daily_labels(ret_1d, n_days)
    y = label.to_numpy(dtype="float64")
    has_label = ~np.isnan(y)
    p = present.to_numpy(dtype=bool)
    rows_all = np.ones_like(p)
    for k in range(n_days):
        shifted = np.zeros_like(p)
        shifted[: len(p) - k] = p[k:]
        rows_all &= shifted
    at_end = np.zeros_like(p)
    at_end[: len(p) - (n_days - 1)] = p[n_days - 1 :]
    c = compounded.to_numpy()
    category = np.full(y.shape, "", dtype=object)
    term = terminal.to_numpy() == 1
    complete = ~np.isnan(c)
    category[has_label & ~term & ~rows_all & ~at_end] = LEFT_PANEL
    category[has_label & ~term & ~rows_all & at_end] = INTERIOR_GAP
    category[has_label & ~term & rows_all & ~complete] = DAILY_MISSING
    category[has_label & ~term & complete] = COMPLETE
    category[has_label & term] = TERMINAL
    with np.errstate(invalid="ignore", divide="ignore"):
        err = np.abs((1 + c) / (1 + y) - 1)
    rows = []
    total = int(has_label.sum())
    for name in (COMPLETE, TERMINAL, DAILY_MISSING, INTERIOR_GAP, LEFT_PANEL):
        cells = category == name
        e = err[cells & ~np.isnan(err)]
        rows.append(
            {
                "category": name,
                "n_cells": int(cells.sum()),
                "share": float(cells.sum() / total) if total else np.nan,
                "n_compared": len(e),
                "max_rel_err": float(e.max()) if len(e) else np.nan,
                "median_rel_err": float(np.median(e)) if len(e) else np.nan,
                "n_above_tol": int((e > rel_tol).sum()),
                "rel_tol": rel_tol,
            }
        )
    return pd.DataFrame(rows), compounded


# --------------------------------------------------------------------------- IC across horizons


def cross_sectional_corr(x: pd.DataFrame, y: pd.DataFrame, min_names: int) -> pd.Series:
    """Daily cross-sectional Pearson correlation over jointly non-null names (linear IC)."""
    a, b = x.to_numpy(dtype="float64"), y.to_numpy(dtype="float64")
    valid = ~np.isnan(a) & ~np.isnan(b)
    n = valid.sum(axis=1)
    a, b = np.where(valid, a, 0.0), np.where(valid, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(valid, a - (a.sum(axis=1) / n)[:, None], 0.0)
        b = np.where(valid, b - (b.sum(axis=1) / n)[:, None], 0.0)
        den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        ic = np.where((n >= min_names) & (den > 0), (a * b).sum(axis=1) / den, np.nan)
    return pd.Series(ic, index=x.index, name="ic")


def ic_decomposition(
    feature: pd.DataFrame, ret_1d: pd.DataFrame, n_days: int, min_names: int
) -> dict[str, np.ndarray]:
    """Exact per-date split of the linear IC of the ``n_days`` log return into daily pieces.

    On names with the feature and all ``n_days`` daily returns (``L = sum_k l_k``, log returns),
    ``corr(f, L) = sum_k corr(f, l_k) * sd(l_k) / sd(L)`` holds exactly on every date. Returns
    ``ic_total`` (T), ``ic_daily`` and ``weight`` (T x n_days, weight = sd(l_k) / sd(L)) and ``n``.
    Dates with fewer than ``min_names`` names are NaN.
    """
    f = feature.to_numpy(dtype="float64")
    mask = ~np.isnan(f)
    for k in range(n_days):  # two passes over the shifts keep one T x N frame in memory at a time
        mask &= ret_1d.shift(-k).notna().to_numpy()
    n = mask.sum(axis=1)
    ok = n >= max(min_names, 3)

    def centre(a: np.ndarray) -> np.ndarray:
        a = np.where(mask, a, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(mask, a - (a.sum(axis=1) / n)[:, None], 0.0)

    fc = centre(f)
    sd_f = np.sqrt((fc * fc).sum(axis=1))
    total = np.zeros_like(fc)
    cov, sd = np.full((len(f), n_days), np.nan), np.full((len(f), n_days), np.nan)
    for k in range(n_days):
        r = ret_1d.shift(-k).to_numpy(dtype="float64")
        lc = centre(np.log1p(np.where(mask, r, 0.0)))
        total += lc
        cov[:, k] = (fc * lc).sum(axis=1)
        sd[:, k] = np.sqrt((lc * lc).sum(axis=1))
    sd_total = np.sqrt((total * total).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic_total = (fc * total).sum(axis=1) / (sd_f * sd_total)
        ic_daily = cov / (sd_f[:, None] * sd)
        weight = sd / sd_total[:, None]
    good = ok & (sd_f > 0) & (sd_total > 0) & np.isfinite(ic_daily).all(axis=1)
    ic_total[~good] = np.nan
    ic_daily[~good] = np.nan
    weight[~good] = np.nan
    return {"ic_total": ic_total, "ic_daily": ic_daily, "weight": weight, "n": n}


def decomposition_summary(parts: dict[str, np.ndarray]) -> dict[str, float]:
    """Time averages of the decomposition: ``ic_total_mean = no_covariance + covariance_term``.

    ``no_covariance`` = ``sum_k mean_t(w_k) mean_t(ic_k)`` is what the multi-day IC would be if the
    daily weights were unrelated to the daily ICs; ``covariance_term`` = ``sum_k cov_t(w_k, ic_k)``
    (population) is the part due to high-weight (high-dispersion) days having different ICs.
    """
    good = ~np.isnan(parts["ic_total"])
    ic, w = parts["ic_daily"][good], parts["weight"][good]
    no_cov = float((w.mean(axis=0) * ic.mean(axis=0)).sum())
    cov_term = float(((w - w.mean(axis=0)) * (ic - ic.mean(axis=0))).mean(axis=0).sum())
    return {
        "n_dates": int(good.sum()),
        "ic_total_mean": float(parts["ic_total"][good].mean()),
        "ic_daily_mean": float(ic.mean()),
        "weight_sum_mean": float(w.sum(axis=1).mean()),
        "no_covariance": no_cov,
        "covariance_term": cov_term,
        "max_identity_error": float(np.abs((w * ic).sum(axis=1) - parts["ic_total"][good]).max()),
    }


def dispersion_weighted_ic(ic: pd.Series, dispersion: pd.Series, top_share: float = 0.1) -> dict[str, float]:
    """Equal- vs dispersion-weighted mean IC, their time-series correlation and the top-dispersion IC.

    ``dispersion`` is the cross-sectional standard deviation of the label on each date; the top
    ``top_share`` of dates by dispersion are compared with the rest.
    """
    frame = pd.DataFrame({"ic": ic, "d": dispersion}).dropna()
    cut = frame["d"].quantile(1 - top_share)
    top = frame["d"] >= cut
    return {
        "ic_mean_equal_weight": float(frame["ic"].mean()),
        "ic_mean_dispersion_weight": float((frame["ic"] * frame["d"]).sum() / frame["d"].sum()),
        "corr_ic_dispersion": float(frame["ic"].corr(frame["d"])),
        "ic_mean_top_dispersion": float(frame.loc[top, "ic"].mean()),
        "ic_mean_rest": float(frame.loc[~top, "ic"].mean()),
        "top_share": top_share,
    }


# --------------------------------------------------------------------------- robust decile spreads


def rank_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Within-date uniform ranks mapped to [-0.5, 0.5]: ``(rank - 0.5) / n - 0.5`` (ties averaged)."""
    rank = frame.rank(axis=1)
    n = rank.notna().sum(axis=1)
    return rank.sub(0.5).div(n.where(n > 0), axis=0) - 0.5


def decile_medians(buckets: pd.DataFrame, label: pd.DataFrame, q: int) -> pd.DataFrame:
    """Within-date median label of each bucket (date x q, columns 1..q), NaN for an empty bucket."""
    y = label.to_numpy(dtype="float64")
    b = buckets.to_numpy()
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows of an empty bucket
        for d in range(q):
            out[d + 1] = np.nanmedian(np.where(b == d, y, np.nan), axis=1)
    return pd.DataFrame(out, index=buckets.index)


def beta_hedged_spread(
    decile_means: pd.DataFrame, decile_betas: pd.DataFrame, market: pd.Series
) -> pd.Series:
    """Top-minus-bottom spread with each leg hedged by its mean beta against the market return.

    ``(r_q - beta_q m) - (r_1 - beta_1 m) = spread - (beta_q - beta_1) m``.
    """
    spread = decile_means.iloc[:, -1] - decile_means.iloc[:, 0]
    gap = decile_betas.iloc[:, -1] - decile_betas.iloc[:, 0]
    return spread - gap * market


def spread_stats(spread: np.ndarray, periods_per_year: float) -> dict[str, float]:
    """Mean, vol, annualised Sharpe and Newey-West t (plug-in lag) of a non-overlapping spread series."""
    x = np.asarray(spread, dtype="float64")
    x = x[~np.isnan(x)]
    nw = newey_west_tstat(x, newey_west_lag(len(x), 1, plugin=False))
    return {
        "n_periods": len(x),
        "mean": nw["mean"],
        "vol": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "sharpe": sharpe(x, periods_per_year),
        "nw_t": nw["t"],
    }


def monotonicity(means: pd.Series) -> float:
    """Spearman correlation of bucket number with the mean per bucket (NaN if a bucket is empty)."""
    if means.isna().any():
        return float("nan")
    return float(np.corrcoef(np.arange(len(means)), means.rank().to_numpy())[0, 1])


# --------------------------------------------------------------------------- break-even costs


def breakeven_costs(spread: pd.Series, legs_turnover: pd.DataFrame) -> dict[str, float]:
    """Break-even cost of a dollar-neutral long-short book from its gross spread and leg turnover.

    ``spread``: oriented gross spread per rebalance ($1 long, $1 short). ``legs_turnover``: one-way
    turnover ``0.5 * sum |dw|`` of each leg per rebalance (columns = legs). One-way turnover is the
    amount sold, and the same amount is bought, so traded notional per rebalance is
    ``2 * (TO_long + TO_short)``. ``breakeven_bps_per_side`` is the cost per unit traded (each buy
    and each sell) that sets the net spread to zero; ``breakeven_bps_round_trip`` is twice that,
    i.e. ``spread / (TO_long + TO_short)``.
    """
    mean_spread = float(spread.dropna().mean())
    one_way = float(legs_turnover.sum(axis=1).dropna().mean())
    per_side = mean_spread / (2 * one_way) if one_way > 0 else np.nan
    return {
        "n_rebalances": int(spread.notna().sum()),
        "spread_bps": 1e4 * mean_spread,
        "one_way_turnover_both_legs": one_way,
        "breakeven_bps_per_side": 1e4 * per_side,
        "breakeven_bps_round_trip": 2e4 * per_side,
    }
