"""Univariate signal research: research-sample loader, IC, quantile portfolios, decay, Fama-MacBeth.

Design and conventions are in README "Signal research" and configs/research.yaml. Cross-sectional
work uses wide ``date x ticker`` float32 frames, vectorised over dates; panel dates are consecutive
trading days, so shifting a wide frame by ``k`` rows shifts by ``k`` trading days.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.metrics import (
    adjust_pvalues,
    hit_rate,
    newey_west_lag,
    newey_west_tstat,
    non_overlapping_positions,
    sharpe,
)
from src.utils import REPO_ROOT, load_config

RESEARCH_CONFIG_PATH = REPO_ROOT / "configs" / "research.yaml"
# Rank-identical to another feature (monotone transform): same IC, deciles and turnover.
RANK_DUPLICATES = ("log_dollar_volume_21d",)


# --------------------------------------------------------------------------- config and loader


def load_research_config(
    path: str | Path | None = None, features_cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load ``configs/research.yaml`` and validate it, including the method choices that are fixed
    in code (a config that asks for anything else fails instead of being silently ignored)."""
    with open(path or RESEARCH_CONFIG_PATH) as handle:
        cfg = yaml.safe_load(handle)
    features_cfg = features_cfg or load_config()
    pd.Timestamp(cfg["holdout_start"])
    if cfg["newey_west"]["lag_rule"] != "max_horizon_plugin":
        raise ValueError("newey_west.lag_rule: only 'max_horizon_plugin' is implemented")
    testing = cfg["multiple_testing"]
    if not 0 < testing["alpha"] < 1 or not set(testing["methods"]) <= {"holm", "bh", "by"}:
        raise ValueError("multiple_testing needs 0 < alpha < 1 and methods within holm / bh / by")
    if cfg["decay"]["n_days"] < 1 or cfg["portfolios"]["offset"] < 0:
        raise ValueError("need decay.n_days >= 1 and portfolios.offset >= 0")
    if list(cfg["horizons"]) != list(features_cfg["labels"]["horizons"]):
        raise ValueError("research horizons must equal labels.horizons in features.yaml")
    if cfg["quantiles"] < 2 or cfg["min_names_per_date"] < cfg["quantiles"]:
        raise ValueError("need quantiles >= 2 and min_names_per_date >= quantiles")
    signals = cfg["signals"]
    for name, spec in signals.items():
        if spec["expected_sign"] not in (-1, 0, 1):
            raise ValueError(f"expected_sign of {name} must be -1, 0 or 1")
        if not spec.get("family"):
            raise ValueError(f"signal {name} needs a family")
    reps = cfg["fama_macbeth"]["multivariate_representatives"]
    for family, name in reps.items():
        if name not in signals or signals[name]["family"] != family:
            raise ValueError(f"representative {name!r} is not a {family!r} signal")
    if set(reps) != {s["family"] for s in signals.values()}:
        raise ValueError("need exactly one representative per family")
    return cfg


def embargo_days(rcfg: dict[str, Any], features_cfg: dict[str, Any]) -> int:
    """Trading days before the holdout that cannot be used: ``lag + max(max horizon, decay.n_days)``.

    The decay offset ``k = n_days - 1`` reads the 1-day return on holding day ``n_days``, which
    ends ``lag + n_days`` days after the signal, like a label of horizon ``n_days``.
    """
    return features_cfg["labels"]["lag"] + max(max(rcfg["horizons"]), rcfg["decay"]["n_days"])


def last_research_date(
    dates: pd.DatetimeIndex, rcfg: dict[str, Any], features_cfg: dict[str, Any]
) -> pd.Timestamp:
    """Last date whose label window ends strictly before the first holdout date."""
    dates = pd.DatetimeIndex(sorted(set(dates)))
    first_holdout = int(dates.searchsorted(pd.Timestamp(rcfg["holdout_start"]), side="left"))
    last_pos = first_holdout - embargo_days(rcfg, features_cfg) - 1
    if last_pos < 0:
        raise ValueError("embargo leaves no research dates")
    return dates[last_pos]


def panel_path(features_cfg: dict[str, Any]) -> Path:
    out = features_cfg["output"]
    return REPO_ROOT / out["dir"] / f"{out['name']}.parquet"


def research_cutoff(
    panel_file: str | Path, rcfg: dict[str, Any], features_cfg: dict[str, Any]
) -> pd.Timestamp:
    """Last research date, from the trading calendar in the panel's date index (no data columns)."""
    calendar = pd.read_parquet(panel_file, columns=[]).index.get_level_values("date").unique()
    return last_research_date(calendar, rcfg, features_cfg)


def load_research_panel(
    columns: list[str] | None = None,
    panel_file: str | Path | None = None,
    rcfg: dict[str, Any] | None = None,
    features_cfg: dict[str, Any] | None = None,
    cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Research-sample rows of the panel as float32; the only way research code reads the panel.

    Rows after the research cutoff (holdout and embargo) are filtered out by the reader. Pass
    ``cutoff`` (from ``research_cutoff``) to skip re-reading the calendar; ``columns=None`` reads all.
    """
    features_cfg = features_cfg or load_config()
    rcfg = rcfg or load_research_config(features_cfg=features_cfg)
    path = Path(panel_file) if panel_file else panel_path(features_cfg)
    last = cutoff if cutoff is not None else research_cutoff(path, rcfg, features_cfg)
    panel = pd.read_parquet(path, columns=columns, filters=[("date", "<=", last)])
    if len(panel) and panel.index.get_level_values("date").max() > last:
        raise AssertionError("research loader returned rows after the last research date")
    float_cols = [c for c in panel.columns if panel[c].dtype != np.float32]
    return panel.astype({c: "float32" for c in float_cols}) if float_cols else panel


# --------------------------------------------------------------------------- wide frames


@dataclass(frozen=True)
class Grid:
    """Row/column positions of a (date, ticker) index in a date x ticker frame."""

    dates: pd.DatetimeIndex
    tickers: pd.Index
    rows: np.ndarray
    cols: np.ndarray

    @classmethod
    def from_index(cls, index: pd.MultiIndex) -> "Grid":
        dates, rows = np.unique(index.get_level_values("date").to_numpy(), return_inverse=True)
        tickers, cols = np.unique(index.get_level_values("ticker").to_numpy(), return_inverse=True)
        return cls(pd.DatetimeIndex(dates, name="date"), pd.Index(tickers, name="ticker"), rows, cols)

    def wide(self, values) -> pd.DataFrame:
        out = np.full((len(self.dates), len(self.tickers)), np.nan, dtype="float32")
        out[self.rows, self.cols] = np.asarray(values, dtype="float32")
        return pd.DataFrame(out, index=self.dates, columns=self.tickers)


# --------------------------------------------------------------------------- IC


def cross_sectional_ic(feature: pd.DataFrame, label: pd.DataFrame, min_names: int) -> pd.Series:
    """Daily cross-sectional Spearman IC between two date x ticker frames.

    Ranks (ties averaged) are taken within each date over the names where both values exist;
    the IC is the Pearson correlation of those ranks. Dates with fewer than ``min_names``
    jointly valid names are NaN. Fully vectorised over dates.
    """
    valid = feature.notna().to_numpy() & label.notna().to_numpy()
    f = feature.where(valid)
    y = label.where(valid)
    rf = f.rank(axis=1).to_numpy(dtype="float64")
    ry = y.rank(axis=1).to_numpy(dtype="float64")
    n = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rf = rf - (np.nansum(rf, axis=1) / n)[:, None]
        ry = ry - (np.nansum(ry, axis=1) / n)[:, None]
        num = np.nansum(rf * ry, axis=1)
        den = np.sqrt(np.nansum(rf * rf, axis=1) * np.nansum(ry * ry, axis=1))
        ic = np.where((n >= min_names) & (den > 0), num / den, np.nan)
    return pd.Series(ic, index=feature.index, name="ic")


def exclude_terminal(label: pd.DataFrame, terminal_flag: pd.DataFrame) -> pd.DataFrame:
    """Label with delisted-stock exits (``fwd_terminal == 1``) set to NaN; other cells unchanged."""
    return label.where(~(terminal_flag.to_numpy() == 1))


def _series_stats(x: np.ndarray, lag: int, trading_days: int, h: int) -> dict[str, float]:
    x = x[~np.isnan(x)]
    nw = newey_west_tstat(x, lag)
    std = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    ir = nw["mean"] / std if std and std > 0 else float("nan")
    return {
        "n": len(x),
        "mean": nw["mean"],
        "std": std,
        "ir": ir,
        "ir_ann": ir * np.sqrt(trading_days / h),
        "hit": hit_rate(x),
        "t": nw["t"],
        "p": nw["p"],
    }


def ic_summary(ic: pd.Series, h: int, expected_sign: int, rcfg: dict[str, Any]) -> dict[str, Any]:
    """IC statistics for one feature x horizon.

    All-dates block: mean, std, IR (mean / std, per date, not annualised), hit rate, Newey-West
    t with ``max(h, plug-in)`` lags (labels overlap) and its normal p-value. ``no_*`` block: the
    non-overlapping subsample (every h-th date, offset 0; plug-in lag only) with IR annualised by
    sqrt(trading_days / h), plus the mean over all h offsets.
    """
    td = rcfg["trading_days"]
    x = ic.to_numpy(dtype="float64")
    n = int((~np.isnan(x)).sum())
    lag = newey_west_lag(n, h)
    full = _series_stats(x, lag, td, h)
    row: dict[str, Any] = {
        "n_dates": full["n"],
        "ic_mean": full["mean"],
        "ic_std": full["std"],
        "ic_ir": full["ir"],
        "hit_rate": full["hit"],
        "nw_lag": lag,
        "nw_t": full["t"],
        "p_value": full["p"],
        "abs_t_gt3": bool(abs(full["t"]) > rcfg["multiple_testing"]["t_threshold"]),
        "expected_sign": expected_sign,
        "sign_consistent": (
            bool(np.sign(full["mean"]) == expected_sign) if expected_sign != 0 else np.nan
        ),
    }
    offsets = []
    for off in range(h):
        sub = x[non_overlapping_positions(len(x), h, off)]
        n_sub = int((~np.isnan(sub)).sum())
        offsets.append(_series_stats(sub, newey_west_lag(n_sub, h, plugin=False), td, h))
    first = offsets[0]
    row.update(
        {
            "no_n": first["n"],
            "no_ic_mean": first["mean"],
            "no_ic_std": first["std"],
            "no_ic_ir_ann": first["ir_ann"],
            "no_hit_rate": first["hit"],
            "no_nw_t": first["t"],
            "no_p_value": first["p"],
            "no_ic_mean_all_offsets": float(np.mean([o["mean"] for o in offsets])),
            "no_nw_t_mean_offsets": float(np.mean([o["t"] for o in offsets])),
        }
    )
    return row


def add_multiple_testing(
    table: pd.DataFrame,
    p_col: str,
    methods: list[str],
    alpha: float,
    drop_features: tuple[str, ...] = RANK_DUPLICATES,
) -> pd.DataFrame:
    """Adjust ``p_col`` over all rows of ``table`` (one family per table).

    Adds ``<p_col>_<method>`` (adjusted p-value), ``reject_<method>`` (adjusted p < ``alpha``) and
    the same two columns with an ``_eff`` suffix computed after dropping rank-duplicate features
    (effective family size).
    """
    out = table.copy()
    keep = ~out["feature"].isin(drop_features).to_numpy()
    for method in methods:
        adjusted = adjust_pvalues(out[p_col].to_numpy(), method)
        eff = np.full(len(out), np.nan)
        eff[keep] = adjust_pvalues(out[p_col].to_numpy()[keep], method)
        out[f"{p_col}_{method}"], out[f"{p_col}_{method}_eff"] = adjusted, eff
        out[f"reject_{method}"] = adjusted < alpha
        out[f"reject_{method}_eff"] = np.where(keep, eff < alpha, False)
    return out


def ic_stability(ic: pd.Series, h: int, rcfg: dict[str, Any]) -> pd.DataFrame:
    """IC by calendar year, by configured sub-period and for the first / second half of dates."""
    td = rcfg["trading_days"]
    groups: list[tuple[str, str, pd.Series]] = []
    for year, part in ic.groupby(ic.index.year):
        groups.append(("year", str(year), part))
    for name, span in rcfg["periods"].items():
        if name == "halves":
            continue
        groups.append(("period", name, ic.loc[span[0] : span[1]]))
    mid = ic.index[len(ic) // 2]
    groups.append(("half", "first", ic[ic.index < mid]))
    groups.append(("half", "second", ic[ic.index >= mid]))
    rows = []
    for kind, label, part in groups:
        x = part.to_numpy(dtype="float64")
        n = int((~np.isnan(x)).sum())
        stats = _series_stats(x, newey_west_lag(n, h), td, h) if n > 1 else {}
        rows.append(
            {
                "kind": kind,
                "label": label,
                "start": part.index.min() if len(part) else pd.NaT,
                "end": part.index.max() if len(part) else pd.NaT,
                "n_dates": n,
                "ic_mean": stats.get("mean", np.nan),
                "ic_std": stats.get("std", np.nan),
                "ic_ir": stats.get("ir", np.nan),
                "nw_t": stats.get("t", np.nan),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- quantile portfolios


def quantile_buckets(feature: pd.DataFrame, q: int, min_names: int) -> pd.DataFrame:
    """Decile (0..q-1) of each name within its date, NaN where the feature is NaN.

    Bucket = floor((average rank - 0.5) / n * q), so tied values always share a bucket (the one
    of their average rank). Dates with fewer than ``min_names`` valid names are all NaN.
    """
    rank = feature.rank(axis=1, method="average").to_numpy(dtype="float64")
    n = np.sum(~np.isnan(rank), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        bucket = np.floor((rank - 0.5) / n * q)
    bucket = np.clip(bucket, 0, q - 1)
    bucket[np.broadcast_to(n < min_names, bucket.shape)] = np.nan
    return pd.DataFrame(bucket.astype("float32"), index=feature.index, columns=feature.columns)


def quantile_returns(buckets: pd.DataFrame, label: pd.DataFrame, q: int) -> pd.DataFrame:
    """Equal-weight mean label of each bucket per date (date x q, columns 1..q, 1 = lowest)."""
    y = label.to_numpy(dtype="float64")
    b = buckets.to_numpy()
    out = {}
    for d in range(q):
        member = (b == d) & ~np.isnan(y)
        count = member.sum(axis=1)
        total = np.where(member, y, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[d + 1] = np.where(count > 0, total / count, np.nan)
    return pd.DataFrame(out, index=buckets.index)


def decile_summary(decile_returns: pd.DataFrame, h: int, offset: int = 0) -> dict[str, Any]:
    """Mean return per decile on non-overlapping dates plus monotonicity measures.

    ``monotonicity`` is the Spearman correlation of decile number with mean return;
    ``frac_adjacent_increasing`` the share of the q - 1 adjacent steps that go up.
    """
    sub = decile_returns.iloc[non_overlapping_positions(len(decile_returns), h, offset)]
    means = sub.mean()
    ranks = means.rank()
    mono = float(np.corrcoef(np.arange(1, len(means) + 1), ranks)[0, 1]) if means.notna().all() else np.nan
    steps = np.diff(means.to_numpy())
    return {
        "means": means,
        "n_periods": int(sub.notna().all(axis=1).sum()),
        "monotonicity": mono,
        "frac_adjacent_increasing": float((steps > 0).mean()) if means.notna().all() else np.nan,
    }


def long_short_stats(
    decile_returns: pd.DataFrame, h: int, expected_sign: int, rcfg: dict[str, Any], offset: int = 0
) -> dict[str, Any]:
    """Gross D_q - D_1 spread on non-overlapping rebalance dates, raw and oriented by the expected sign.

    Per-period mean / vol, annualised Sharpe (sqrt(252 / h)), annualised mean (mean * 252 / h)
    and Newey-West t (plug-in lag). ``*_all_offsets`` is the mean over every date (= the mean
    over all h offsets). Oriented stats are NaN when ``expected_sign == 0``. No costs.
    """
    td = rcfg["trading_days"]
    spread = decile_returns.iloc[:, -1] - decile_returns.iloc[:, 0]
    sub = spread.to_numpy()[non_overlapping_positions(len(spread), h, offset)]
    sub = sub[~np.isnan(sub)]
    lag = newey_west_lag(len(sub), h, plugin=False)

    def block(x: np.ndarray, prefix: str) -> dict[str, float]:
        nw = newey_west_tstat(x, lag)
        return {
            f"{prefix}_mean": nw["mean"],
            f"{prefix}_vol": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            f"{prefix}_sharpe": sharpe(x, td / h),
            f"{prefix}_ann_return": nw["mean"] * td / h,
            f"{prefix}_nw_t": nw["t"],
            f"{prefix}_p_value": nw["p"],
        }

    raw, oriented = "gross_raw", "gross_oriented"
    row = {"n_periods": len(sub), "nw_lag": lag, **block(sub, raw)}
    row[f"{raw}_mean_all_offsets"] = float(np.nanmean(spread.to_numpy())) if spread.notna().any() else np.nan
    if expected_sign != 0:
        row.update(block(sub * expected_sign, oriented))
        row[f"{oriented}_mean_all_offsets"] = row[f"{raw}_mean_all_offsets"] * expected_sign
    else:
        row.update({k.replace(raw, oriented, 1): np.nan for k in list(row) if k.startswith(raw)})
    return row


def turnover(
    buckets: pd.DataFrame, q: int, h: int, offset: int = 0
) -> pd.DataFrame:
    """One-way equal-weight turnover of the bottom and top legs between consecutive rebalances.

    Turnover = 0.5 * sum |w_t - w_{t-1}| (weights equal within a leg, no drift), so a name that
    leaves the leg or the index counts as traded. Rows are rebalance dates after the first.
    """
    pos = non_overlapping_positions(len(buckets), h, offset)
    b = buckets.to_numpy()[pos]
    out = {}
    for name, leg in (("D1", 0), ("D10" if q == 10 else f"D{q}", q - 1)):
        member = (b == leg).astype("float64")
        size = member.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(size > 0, member / size, np.nan)
        out[name] = 0.5 * np.abs(w[1:] - w[:-1]).sum(axis=1)
    return pd.DataFrame(out, index=buckets.index[pos][1:])


# --------------------------------------------------------------------------- decay


def ic_half_life(ic_by_k: np.ndarray, t_k0: float, min_abs_t: float) -> float:
    """Rough half-life in days: first k where the 3-point smoothed |IC| falls to half of k = 0.

    NaN when the k = 0 IC is not significant (|t| < ``min_abs_t``) or never halves within K.
    Linear interpolation between the two bracketing lags.
    """
    ic = np.asarray(ic_by_k, dtype="float64")
    if len(ic) < 2 or np.isnan(t_k0) or abs(t_k0) < min_abs_t:
        return float("nan")
    padded = np.pad(ic, 1, mode="edge")
    smooth = np.convolve(padded, np.ones(3) / 3, mode="valid") * np.sign(ic[0])
    target = smooth[0] / 2
    below = np.nonzero(smooth[1:] <= target)[0]
    if len(below) == 0 or smooth[0] <= 0:
        return float("nan")
    k = below[0] + 1
    return float(k - 1 + (smooth[k - 1] - target) / (smooth[k - 1] - smooth[k]))


def ic_decay(
    feature: pd.DataFrame, ret_1d: pd.DataFrame, n_days: int, min_names: int, rcfg: dict[str, Any]
) -> pd.DataFrame:
    """IC of the feature at ``t`` against the 1-day return ``k`` days after entry, k = 0..n_days-1.

    ``ret_1d`` is the wide ``fwd_ret_1d`` frame, whose row ``t`` holds the return of the first
    holding day (the day after entry). ``shift(-k)`` moves holding day ``k + 1`` onto row ``t``.
    Shifted returns exist only for names still in the panel (index members) at ``t + k``, so
    later lags condition on future membership (mild survivorship, see README).
    Columns: ``k, ic_mean, nw_t, cum_ic, n_dates``; the half-life is in ``attrs['half_life']``.
    """
    rows = []
    for k in range(n_days):
        ic = cross_sectional_ic(feature, ret_1d.shift(-k), min_names)
        x = ic.dropna().to_numpy(dtype="float64")
        nw = newey_west_tstat(x, newey_west_lag(len(x), 1))
        rows.append({"k": k, "ic_mean": nw["mean"], "nw_t": nw["t"], "n_dates": len(x)})
    out = pd.DataFrame(rows)
    out["cum_ic"] = out["ic_mean"].cumsum()
    out.attrs["half_life"] = ic_half_life(
        out["ic_mean"].to_numpy(), float(out["nw_t"].iloc[0]), rcfg["decay"]["half_life_min_abs_t"]
    )
    return out


# --------------------------------------------------------------------------- Fama-MacBeth


@dataclass
class FamaMacBeth:
    """Per-date cross-sectional regressions: coefficients, R^2, N and collinearity diagnostics."""

    coefs: pd.DataFrame  # date x (const + regressors)
    r2: pd.Series
    n: pd.Series
    diagnostics: pd.DataFrame | None  # date x (vif_<name>..., cond, mean_abs_corr) if K > 1


def fama_macbeth(
    regressors: dict[str, pd.DataFrame], y: pd.DataFrame, min_names: int, chunk: int = 400
) -> FamaMacBeth:
    """OLS of ``y`` on the regressors with an intercept, separately on every date.

    Rows with any NaN (regressors or y) are dropped listwise. Batched normal equations over
    chunks of dates, so there is no per-date Python loop. Dates with fewer than ``min_names``
    complete rows are NaN. Diagnostics (K > 1): VIF per regressor, condition number and mean
    absolute pairwise correlation of the regressors on each date.
    """
    names = list(regressors)
    k = len(names) + 1
    arrays = [regressors[name].to_numpy() for name in names]
    y_all = y.to_numpy()
    dates = y.index
    coefs = np.full((len(dates), k), np.nan)
    r2 = np.full(len(dates), np.nan)
    n_obs = np.zeros(len(dates))
    diag = np.full((len(dates), len(names) + 2), np.nan) if len(names) > 1 else None

    for start in range(0, len(dates), chunk):
        sl = slice(start, start + chunk)
        x = np.stack([a[sl] for a in arrays], axis=2).astype("float64")
        yy = y_all[sl].astype("float64")
        valid = np.isfinite(x).all(axis=2) & np.isfinite(yy)
        x = np.where(valid[..., None], x, 0.0)
        yy = np.where(valid, yy, 0.0)
        design = np.concatenate([valid[..., None].astype("float64"), x], axis=2)
        n = valid.sum(axis=1).astype("float64")
        ok = n >= max(min_names, k + 1)
        xtx = np.einsum("tnk,tnl->tkl", design, design)
        xty = np.einsum("tnk,tn->tk", design, yy)
        xtx[~ok] = np.eye(k)
        beta = np.einsum("tkl,tl->tk", np.linalg.pinv(xtx), xty)
        resid = (yy - np.einsum("tnk,tk->tn", design, beta)) * valid
        sse = (resid**2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ybar = yy.sum(axis=1) / n
            sst = ((yy - ybar[:, None]) ** 2 * valid).sum(axis=1)
            r2_chunk = 1 - sse / sst
        beta[~ok] = np.nan
        coefs[sl] = beta
        r2[sl] = np.where(ok, r2_chunk, np.nan)
        n_obs[sl] = np.where(ok, n, 0)
        if diag is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                s, ss = xtx[:, 0, 1:], xtx[:, 1:, 1:]
                mean = s / n[:, None]
                cov = (ss - n[:, None, None] * mean[:, :, None] * mean[:, None, :]) / (n[:, None, None] - 1)
                sd = np.sqrt(np.einsum("tii->ti", cov))
                corr = cov / (sd[:, :, None] * sd[:, None, :])
            corr[~ok] = np.eye(len(names))
            bad = ~np.isfinite(corr).all(axis=(1, 2))
            corr[bad] = np.eye(len(names))
            vif = np.einsum("tii->ti", np.linalg.inv(corr))
            cond = np.linalg.cond(corr)
            off = ~np.eye(len(names), dtype=bool)
            mean_abs = np.abs(corr[:, off]).mean(axis=1)
            good = ok & ~bad
            block = np.column_stack([vif, cond, mean_abs])
            block[~good] = np.nan
            diag[sl] = block

    col_names = ["const", *names]
    diagnostics = (
        pd.DataFrame(diag, index=dates, columns=[f"vif_{n}" for n in names] + ["cond", "mean_abs_corr"])
        if diag is not None
        else None
    )
    return FamaMacBeth(
        pd.DataFrame(coefs, index=dates, columns=col_names),
        pd.Series(r2, index=dates, name="r2"),
        pd.Series(n_obs, index=dates, name="n"),
        diagnostics,
    )


def fm_summary(result: FamaMacBeth, h: int) -> pd.DataFrame:
    """Mean coefficient (return per 1 SD of the regressor over h days), NW t, avg R^2 and N."""
    rows = []
    r2, n = float(result.r2.mean()), float(result.n[result.n > 0].mean())
    for name in result.coefs.columns:
        x = result.coefs[name].dropna().to_numpy()
        nw = newey_west_tstat(x, newey_west_lag(len(x), h))
        rows.append(
            {
                "regressor": name,
                "coef_mean": nw["mean"],
                "nw_lag": newey_west_lag(len(x), h),
                "nw_t": nw["t"],
                "p_value": nw["p"],
                "n_dates": nw["n"],
                "avg_r2": r2,
                "avg_n": n,
            }
        )
    out = pd.DataFrame(rows)
    if result.diagnostics is not None:
        vif = result.diagnostics.mean()
        out["avg_vif"] = [vif.get(f"vif_{r}", np.nan) for r in out["regressor"]]
        out["avg_condition_number"] = float(vif["cond"])
        out["avg_abs_pairwise_corr"] = float(vif["mean_abs_corr"])
    return out


# --------------------------------------------------------------------------- cross-signal structure


def cross_signal_correlation(
    features: dict[str, pd.DataFrame], min_names: int
) -> pd.DataFrame:
    """Time-average of the daily cross-sectional Spearman correlation between every feature pair."""
    names = list(features)
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            value = cross_sectional_ic(features[a], features[b], min_names).mean()
            out.loc[a, b] = out.loc[b, a] = value
    return out
