"""Pre-model diagnostics: label compounding, IC across horizons, robust decile spreads, break-even costs.

Usage (from the repository root):
    python -m src.run_diagnostics [--config configs/research.yaml] [--panel PATH] [--out DIR]

Reads only research-sample rows through ``src.signals.load_research_panel``, uses the design in
``configs/research.yaml`` unchanged and writes tidy CSVs, figures and ``run_metadata.json`` to
``reports/diagnostics/``. Like the signal research, the run refuses to start unless
``configs/research.yaml`` is committed and unmodified (``--allow-dirty-config`` overrides). When
``--reference`` points at a signal-research run on the same panel and config, the raw spreads,
turnover and ICs recomputed here must reproduce it.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: figures are only saved
import matplotlib.pyplot as plt  # noqa: E402

from src.diagnostics import (  # noqa: E402
    COMPLETE,
    beta_hedged_spread,
    breakeven_costs,
    compounding_check,
    cross_sectional_corr,
    decile_medians,
    decomposition_summary,
    dispersion_weighted_ic,
    ic_decomposition,
    monotonicity,
    rank_by_date,
    spread_stats,
)
from src.metrics import newey_west_lag, newey_west_tstat, non_overlapping_positions  # noqa: E402
from src.signals import (  # noqa: E402
    RESEARCH_CONFIG_PATH,
    Grid,
    cross_sectional_ic,
    embargo_days,
    load_research_config,
    load_research_panel,
    panel_path,
    quantile_buckets,
    quantile_returns,
    research_cutoff,
    turnover,
)
from src.utils import REPO_ROOT, load_config, provenance, winsorize_by_date  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "diagnostics"
REFERENCE_DIR = REPO_ROOT / "reports" / "signal_research"
RECONCILE = ("mom_12_1", "ret_252d", "vol_63d", "ret_5d")  # features of the IC-across-horizons check
WINSOR = (0.01, 0.99)  # within-date limits for the winsorised-return spread
REL_TOL = 1e-5  # float32 labels compounded over 21 days agree to ~1e-6
BETA = "beta_252d"
VARIANTS = ("raw", "xs", "beta_hedged", "winsorised", "median", "rank", "dispersion_scaled")
BREAKEVEN_VARIANTS = ("raw", "beta_hedged")


def _ic_stats(ic: pd.Series, h: int) -> tuple[float, float]:
    x = ic.dropna().to_numpy(dtype="float64")
    nw = newey_west_tstat(x, newey_west_lag(len(x), h))
    return nw["mean"], nw["t"]


def _orientation(sign: int, ic_mean: float) -> tuple[int, str]:
    """Long side of the book: the prior sign, else the sign of the full-sample IC (descriptive)."""
    return (sign, "prior") if sign != 0 else (int(np.sign(ic_mean)) or 1, "full_sample_ic")


def reconcile_horizons(
    feat: pd.DataFrame, z: pd.DataFrame, ret: dict[int, pd.DataFrame], compounded: pd.DataFrame,
    log_21: pd.DataFrame, n_days: int, min_names: int,
) -> list[dict[str, Any]]:
    """Rank and linear IC at h = 1 and h = n_days on all rows and on rows with a complete daily window."""
    complete = compounded.notna()
    rows = []
    for ic_type, x, ic_fn in (("rank", feat, cross_sectional_ic), ("linear", z, cross_sectional_corr)):
        cases = [
            ("all_rows", "stored", ret[1], ret[n_days]),
            ("complete_window", "stored", ret[1].where(complete), ret[n_days].where(complete)),
            ("complete_window", "compounded", ret[1].where(complete), compounded),
            ("complete_window", "log_sum", ret[1].where(complete), log_21),
        ]
        for sample, label, y1, yh in cases:
            m1, t1 = _ic_stats(ic_fn(x, y1, min_names), 1)
            mh, th = _ic_stats(ic_fn(x, yh, min_names), n_days)
            rows.append({
                "ic_type": ic_type, "sample": sample, f"h{n_days}_label": label,
                "ic_mean_h1": m1, "nw_t_h1": t1, f"ic_mean_h{n_days}": mh, f"nw_t_h{n_days}": th,
                "ratio": mh / m1, "sqrt_h": float(np.sqrt(n_days)),
            })
    return rows


def run(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=RESEARCH_CONFIG_PATH)
    parser.add_argument("--features-config", type=Path, default=None)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--reference", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--allow-dirty-config", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    fcfg = load_config(args.features_config)
    rcfg = load_research_config(args.config, fcfg)
    panel_file = args.panel or panel_path(fcfg)
    identity = provenance(args.config, args.features_config, panel_file, args.allow_dirty_config)
    out, fig_dir = args.out, args.out / "figures"
    for d in (out, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    horizons, q = rcfg["horizons"], rcfg["quantiles"]
    min_names, signals = rcfg["min_names_per_date"], rcfg["signals"]
    offset, td = rcfg["portfolios"]["offset"], rcfg["trading_days"]
    n_days = max(horizons)
    cutoff = research_cutoff(panel_file, rcfg, fcfg)

    def load(columns: list[str]) -> pd.DataFrame:
        return load_research_panel(columns, panel_file, rcfg, fcfg, cutoff)

    label_cols = [f"fwd_ret_{h}d{s}" for h in horizons for s in ("", "_xs")]
    label_cols += [f"fwd_terminal_{h}d" for h in horizons]
    labels = load(label_cols)
    grid = Grid.from_index(labels.index)
    index = labels.index
    ret = {h: grid.wide(labels[f"fwd_ret_{h}d"]) for h in horizons}
    xs = {h: grid.wide(labels[f"fwd_ret_{h}d_xs"]) for h in horizons}
    terminal = {h: grid.wide(labels[f"fwd_terminal_{h}d"]) for h in horizons}
    present = grid.wide(np.ones(len(index))).notna()
    del labels
    mid = grid.dates[len(grid.dates) // 2]  # same split as ic_stability's halves
    print(f"research sample: {grid.dates[0].date()} to {grid.dates[-1].date()} "
          f"({len(grid.dates)} dates, {len(index):,} rows); split at {mid.date()}")

    # ------------------------------------------------------------------ label compounding
    comp_rows, compounded = [], {}
    for h in horizons:
        if h == 1:
            continue
        table, compounded[h] = compounding_check(ret[1], ret[h], terminal[h], present, h, REL_TOL)
        comp_rows.append(table.assign(horizon=h))
    compounding = pd.concat(comp_rows, ignore_index=True)
    mismatches = int(compounding.loc[compounding["category"] == COMPLETE, "n_above_tol"].sum())
    print(f"compounding: {mismatches} complete-window cells above tolerance ({time.time() - started:.0f}s)")
    log_21 = np.log1p(compounded[n_days])

    # ------------------------------------------------------------------ IC across horizons
    rec_rows, decomp_rows, disp_rows, disp_bins = [], [], [], []
    for name in RECONCILE:
        frame = load([name, f"z_{name}"])
        feat, z = grid.wide(frame[name]), grid.wide(frame[f"z_{name}"])
        for row in reconcile_horizons(feat, z, ret, compounded[n_days], log_21, n_days, min_names):
            rec_rows.append({"feature": name, **row})
        for transform, x in (("z", z), ("rank", rank_by_date(feat))):
            parts = ic_decomposition(x, ret[1], n_days, min_names)
            decomp_rows.append({"feature": name, "feature_transform": transform, **decomposition_summary(parts),
                                "sqrt_h": float(np.sqrt(n_days))})
        for h in (1, n_days):
            ic = cross_sectional_ic(feat, ret[h], min_names)
            dispersion = ret[h].std(axis=1)
            disp_rows.append({"feature": name, "horizon": h, **dispersion_weighted_ic(ic, dispersion)})
            if h == 1:
                bins = pd.qcut(dispersion.reindex(ic.dropna().index), 10, labels=False) + 1
                disp_bins.append(ic.dropna().groupby(bins).mean().rename(name))
        print(f"  reconcile {name:<18} done ({time.time() - started:.0f}s)")
        del frame, feat, z

    # ------------------------------------------------------------------ robust spreads and break-even
    per_h: dict[int, dict[str, Any]] = {}
    for h in horizons:
        pos = non_overlapping_positions(len(grid.dates), h, offset)
        y = ret[h].iloc[pos]
        per_h[h] = {
            "pos": pos, "y": y, "xs": xs[h].iloc[pos], "wins": winsorize_by_date(y, WINSOR),
            "rank": rank_by_date(y), "market": y.mean(axis=1), "dispersion": y.std(axis=1),
        }
    beta_wide = grid.wide(load([BETA])[BETA])

    spread_rows, decile_rows, be_rows, ic_rows = [], [], [], []
    turnover_check, xs_gap = [], 0.0
    for name, spec in signals.items():
        feat = grid.wide(load([name])[name])
        sign = spec["expected_sign"]
        buckets = quantile_buckets(feat, q, min_names)
        for h in horizons:
            d = per_h[h]
            ic_mean, ic_t = _ic_stats(cross_sectional_ic(feat, ret[h], min_names), h)
            ic_rows.append({"feature": name, "horizon": h, "ic_mean": ic_mean, "nw_t": ic_t})
            b = buckets.iloc[d["pos"]]
            means = {
                "raw": quantile_returns(b, d["y"], q),
                "xs": quantile_returns(b, d["xs"], q),
                "winsorised": quantile_returns(b, d["wins"], q),
                "median": decile_medians(b, d["y"], q),
                "rank": quantile_returns(b, d["rank"], q),
            }
            betas = quantile_returns(b, beta_wide.iloc[d["pos"]].where(d["y"].notna()), q)
            hedged = means["raw"] - betas.mul(d["market"], axis=0)
            means["beta_hedged"] = hedged
            spreads = {k: v.iloc[:, -1] - v.iloc[:, 0] for k, v in means.items()}
            spreads["beta_hedged"] = beta_hedged_spread(means["raw"], betas, d["market"])
            spreads["dispersion_scaled"] = spreads["raw"] / d["dispersion"]
            xs_gap = max(xs_gap, float((spreads["xs"] - spreads["raw"]).abs().max()))
            beta_gap = float((betas.iloc[:, -1] - betas.iloc[:, 0]).mean())
            for variant in VARIANTS:
                stats = spread_stats(spreads[variant].to_numpy(), td / h)
                mono = monotonicity(means[variant].mean()) if variant in means else np.nan
                spread_rows.append({
                    "feature": name, "family": spec["family"], "horizon": h, "expected_sign": sign,
                    "ic_mean": ic_mean, "ic_nw_t": ic_t, "variant": variant, **stats, "monotonicity": mono,
                    "sign_matches_ic": bool(np.sign(stats["mean"]) == np.sign(ic_mean)),
                    "beta_gap_top_minus_bottom": beta_gap,
                })
            for variant, frame in means.items():
                for decile, value in frame.mean().items():
                    decile_rows.append({"feature": name, "horizon": h, "variant": variant,
                                        "decile": decile, "mean": value})

            legs = turnover(buckets, q, h, offset)
            turnover_check.append({"feature": name, "horizon": h, "D1_turnover": legs.iloc[:, 0].mean(),
                                   f"D{q}_turnover": legs.iloc[:, 1].mean()})
            orient, source = _orientation(sign, ic_mean)
            every_date = quantile_returns(buckets, ret[h], q)  # raw spread on every date = all h offsets
            every_date = every_date.iloc[:, -1] - every_date.iloc[:, 0]
            for sample, lo, hi in (("full", grid.dates[0], grid.dates[-1]),
                                   ("first_half", grid.dates[0], grid.dates[grid.dates < mid][-1]),
                                   ("second_half", mid, grid.dates[-1])):
                for variant in BREAKEVEN_VARIANTS:
                    s = spreads[variant].loc[lo:hi] * orient
                    stats = spread_stats(s.to_numpy(), td / h)
                    all_offsets = 1e4 * orient * every_date.loc[lo:hi].mean() if variant == "raw" else np.nan
                    be_rows.append({
                        "feature": name, "family": spec["family"], "horizon": h, "sample": sample,
                        "start": lo.date(), "end": hi.date(), "variant": variant, "orientation": orient,
                        "orientation_source": source, **breakeven_costs(s, legs.loc[lo:hi]),
                        "spread_bps_all_offsets": all_offsets, "spread_nw_t": stats["nw_t"],
                        "gross_sharpe": stats["sharpe"],
                    })
        print(f"  spreads {name:<20} done ({time.time() - started:.0f}s)")
        del feat, buckets

    if xs_gap > 1e-6:
        raise AssertionError(f"xs and raw decile spreads differ by {xs_gap:.2e}; demeaning is not per date")

    tables = {
        "label_compounding_check": compounding,
        "ic_horizon_reconciliation": pd.DataFrame(rec_rows),
        "ic_decomposition": pd.DataFrame(decomp_rows),
        "ic_dispersion": pd.DataFrame(disp_rows),
        "ic_by_dispersion_decile": pd.concat(disp_bins, axis=1).rename_axis("dispersion_decile").reset_index(),
        "ic_scaling_all_features": _scaling(pd.DataFrame(ic_rows), horizons),
        "decile_spreads_robust": pd.DataFrame(spread_rows),
        "decile_means_robust": pd.DataFrame(decile_rows),
        "breakeven_costs": pd.DataFrame(be_rows),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)

    reference = _reference_check(args.reference, identity, tables, pd.DataFrame(turnover_check), q)
    _plot_dispersion(tables["ic_by_dispersion_decile"], fig_dir / "ic_by_dispersion_decile_h1.png")
    _plot_risk_deciles(tables["decile_means_robust"], n_days, fig_dir / f"risk_deciles_h{n_days}.png")

    metadata = {
        **identity,
        "holdout_start": rcfg["holdout_start"],
        "embargo_trading_days": embargo_days(rcfg, fcfg),
        "first_research_date": str(grid.dates[0].date()),
        "last_research_date": str(grid.dates[-1].date()),
        "split_date": str(mid.date()),
        "n_research_dates": len(grid.dates),
        "n_research_rows": len(index),
        "compounding_rel_tol": REL_TOL,
        "compounding_mismatches": mismatches,
        "max_abs_xs_minus_raw_spread": xs_gap,
        "winsor_limits": list(WINSOR),
        "reference_check": reference,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "runtime_seconds": round(time.time() - started, 1),
    }
    assert grid.dates[-1] <= cutoff < pd.Timestamp(rcfg["holdout_start"]), "output reaches the holdout"
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(json.dumps(metadata, indent=2, default=str))
    return tables


def _scaling(ic: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Mean IC per horizon for every feature and its ratio to the h = 1 IC, next to sqrt(h)."""
    wide = ic.pivot(index="feature", columns="horizon", values="ic_mean")
    wide.columns = [f"ic_h{h}" for h in wide.columns]
    for h in horizons[1:]:
        wide[f"ratio_h{h}"] = wide[f"ic_h{h}"] / wide[f"ic_h{horizons[0]}"]
        wide[f"sqrt_h{h}"] = np.sqrt(h)
    tstats = ic.pivot(index="feature", columns="horizon", values="nw_t")
    tstats.columns = [f"nw_t_h{h}" for h in tstats.columns]
    return wide.join(tstats).reset_index()


def _reference_check(
    ref_dir: Path, identity: dict[str, Any], tables: dict[str, pd.DataFrame], legs: pd.DataFrame, q: int
) -> str:
    """Raw spreads, turnover and ICs must reproduce a signal-research run on the same panel and config."""
    meta_file = ref_dir / "run_metadata.json"
    if not meta_file.exists():
        return "skipped: no reference run"
    meta = json.loads(meta_file.read_text())
    if meta.get("panel_sha256") != identity["panel_sha256"] or meta.get("config_sha256") != identity["config_sha256"]:
        return "skipped: reference run used a different panel or config"
    keys = ["feature", "horizon"]
    ls = pd.read_csv(ref_dir / "long_short.csv")[[*keys, "gross_raw_mean"]]
    raw = tables["decile_spreads_robust"].query("variant == 'raw'")[[*keys, "mean", "ic_mean"]]
    merged = raw.merge(ls, on=keys).merge(pd.read_csv(ref_dir / "ic_summary.csv")[[*keys, "ic_mean"]],
                                          on=keys, suffixes=("", "_ref"))
    to = legs.merge(pd.read_csv(ref_dir / "turnover.csv"), on=keys, suffixes=("", "_ref"))
    checks = {
        "spread": np.allclose(merged["mean"], merged["gross_raw_mean"], rtol=1e-6, atol=1e-10),
        "ic": np.allclose(merged["ic_mean"], merged["ic_mean_ref"], rtol=1e-6, atol=1e-10),
        "turnover": all(np.allclose(to[c], to[f"{c}_ref"], rtol=1e-9) for c in ("D1_turnover", f"D{q}_turnover")),
    }
    if not all(checks.values()) or len(merged) != len(raw):
        raise AssertionError(f"diagnostics do not reproduce the signal research: {checks}")
    return "passed: raw spreads, ICs and turnover match the signal research"


def _plot_dispersion(bins: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins.set_index("dispersion_decile").plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="decile of dates by cross-sectional sd of the 1-day return (10 = most dispersed)",
           ylabel="mean daily rank IC, h = 1")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_risk_deciles(means: pd.DataFrame, h: int, path: Path) -> None:
    sub = means.query("horizon == @h and feature in ['vol_21d', 'vol_63d', 'beta_252d']")
    panels = [("raw", "mean raw return"), ("beta_hedged", "mean beta-hedged return"),
              ("median", "median return"), ("rank", "mean return rank")]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 3.6))
    for ax, (variant, title) in zip(axes, panels):
        part = sub.query("variant == @variant").pivot(index="decile", columns="feature", values="mean")
        part.plot(ax=ax, marker="o", legend=ax is axes[0])
        ax.set(title=f"{title}, h = {h}d", xlabel="decile of the feature (1 = lowest)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
