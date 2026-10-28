"""Generic statistics helpers, independent of the panel layout (reused by other analyses)."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def newey_west_lag(n: int, horizon: int = 1, plugin: bool = True) -> int:
    """``max(horizon, floor(4 (n / 100) ** (2 / 9)))``; ``plugin=False`` drops the horizon floor.

    Labels that span ``h`` dates overlap, so the series is MA(h - 1) and needs at least ``h`` lags.
    """
    rule = int(math.floor(4 * (max(n, 1) / 100) ** (2 / 9)))
    return max(horizon, rule) if plugin else rule


def newey_west_tstat(x, lags: int) -> dict[str, float]:
    """t-stat of the mean of ``x`` with a Bartlett-kernel HAC variance (no small-sample correction).

    Matches ``sm.OLS(x, 1).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})``.
    NaNs are dropped. Returns ``mean, se, t, p`` (two-sided, normal) and ``n``.
    """
    x = np.asarray(x, dtype="float64")
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan"), "n": n}
    e = x - x.mean()
    lags = min(lags, n - 1)
    s = float(e @ e)
    for j in range(1, lags + 1):
        s += 2 * (1 - j / (lags + 1)) * float(e[j:] @ e[:-j])
    se = math.sqrt(max(s, 0.0)) / n
    t = float(x.mean() / se) if se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(t))) if not math.isnan(t) else float("nan")
    return {"mean": float(x.mean()), "se": se, "t": t, "p": p, "n": n}


def sharpe(returns, periods_per_year: float) -> float:
    """Annualised mean / std (ddof=1) of per-period returns; NaNs dropped."""
    r = np.asarray(returns, dtype="float64")
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


def hit_rate(x) -> float:
    """Share of non-NaN values strictly above zero."""
    x = np.asarray(x, dtype="float64")
    x = x[~np.isnan(x)]
    return float((x > 0).mean()) if len(x) else float("nan")


def adjust_pvalues(p, method: str) -> np.ndarray:
    """Multiple-testing adjusted p-values: ``holm``, ``bh`` (Benjamini-Hochberg) or ``by``.

    NaN p-values stay NaN and are not counted in the family size.
    """
    p = np.asarray(p, dtype="float64")
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * ranked)
    elif method in ("bh", "by"):
        scale = m * (np.sum(1 / np.arange(1, m + 1)) if method == "by" else 1.0)
        adj = np.minimum.accumulate((scale / np.arange(1, m + 1) * ranked)[::-1])[::-1]
    else:
        raise ValueError(f"unknown method {method!r}")
    result = np.empty(m)
    result[order] = np.minimum(adj, 1.0)
    out[ok] = result
    return out


def non_overlapping_positions(n: int, horizon: int, offset: int = 0) -> np.ndarray:
    """Positions ``offset, offset + h, ...`` below ``n`` (labels of h days do not overlap)."""
    return np.arange(offset, n, horizon)
