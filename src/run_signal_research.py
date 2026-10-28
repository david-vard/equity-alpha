"""Univariate signal research: regenerate every table and figure from the cached panel.

Usage (from the repository root):
    python -m src.run_signal_research [--config configs/research.yaml] [--panel PATH] [--out DIR]

Reads only research-sample rows through ``src.signals.load_research_panel`` and writes tidy CSVs,
figures and ``run_metadata.json`` to ``reports/signal_research/``. The run refuses to start unless
``configs/research.yaml`` is committed and unmodified, unless ``--allow-dirty-config`` is given.
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
import pyarrow
import scipy
import statsmodels

matplotlib.use("Agg")  # headless: figures are only saved

from src import signal_plots as plots  # noqa: E402
from src.signals import (
    RANK_DUPLICATES,
    RESEARCH_CONFIG_PATH,
    Grid,
    add_multiple_testing,
    cross_sectional_ic,
    cross_signal_correlation,
    decile_summary,
    embargo_days,
    exclude_terminal,
    fama_macbeth,
    fm_summary,
    ic_decay,
    ic_stability,
    ic_summary,
    load_research_config,
    load_research_panel,
    long_short_stats,
    panel_path,
    research_cutoff,
    quantile_buckets,
    quantile_returns,
    turnover,
)
from src.metrics import non_overlapping_positions
from src.signal_report import headline_table
from src.utils import REPO_ROOT, load_config, provenance

OUT_DIR = REPO_ROOT / "reports" / "signal_research"


def _tag(row: dict[str, Any], feature: str, h: int, spec: dict[str, Any]) -> dict[str, Any]:
    return {"feature": feature, "family": spec["family"], "horizon": h, **row}


def run(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=RESEARCH_CONFIG_PATH)
    parser.add_argument("--features-config", type=Path, default=None)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--allow-dirty-config", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    fcfg = load_config(args.features_config)
    rcfg = load_research_config(args.config, fcfg)
    panel_file = args.panel or panel_path(fcfg)
    identity = provenance(args.config, args.features_config, panel_file, args.allow_dirty_config)
    out, fig_dir, cache = args.out, args.out / "figures", args.out / "cache"
    for d in (out, fig_dir, cache):
        d.mkdir(parents=True, exist_ok=True)

    horizons, q = rcfg["horizons"], rcfg["quantiles"]
    min_names, signals = rcfg["min_names_per_date"], rcfg["signals"]
    offset = rcfg["portfolios"]["offset"]
    features = list(signals)

    cutoff = research_cutoff(panel_file, rcfg, fcfg)

    def load(columns: list[str]) -> pd.DataFrame:
        return load_research_panel(columns, panel_file, rcfg, fcfg, cutoff)

    # labels once, as wide frames on a shared grid
    label_cols = [f"fwd_ret_{h}d{s}" for h in horizons for s in ("", "_xs")]
    label_cols += [f"fwd_terminal_{h}d" for h in horizons]
    labels = load(label_cols)
    grid = Grid.from_index(labels.index)
    ret = {h: grid.wide(labels[f"fwd_ret_{h}d"]) for h in horizons}
    xs = {h: grid.wide(labels[f"fwd_ret_{h}d_xs"]) for h in horizons}
    terminal = {h: grid.wide(labels[f"fwd_terminal_{h}d"]) for h in horizons}
    index = labels.index
    del labels
    print(f"research sample: {grid.dates[0].date()} to {grid.dates[-1].date()} "
          f"({len(grid.dates)} dates, {len(index):,} rows); embargo {embargo_days(rcfg, fcfg)} days")

    ic_rows, stab_rows, term_rows, dec_rows, ls_rows, to_rows = [], [], [], [], [], []
    decay_rows, fm_rows, daily_ic, spreads = [], [], {h: {} for h in horizons}, {h: {} for h in horizons}
    raw_wide: dict[str, pd.DataFrame] = {}

    for name, spec in signals.items():
        frame = load([name, f"z_{name}"])
        assert frame.index.equals(index), "feature rows differ from the label rows"
        feat = grid.wide(frame[name])
        z = grid.wide(frame[f"z_{name}"])
        raw_wide[name] = feat
        sign = spec["expected_sign"]
        buckets = quantile_buckets(feat, q, min_names)

        decay = ic_decay(feat, ret[1], rcfg["decay"]["n_days"], min_names, rcfg)
        half_life = decay.attrs["half_life"]
        decay_rows.append(decay.assign(feature=name, family=spec["family"], half_life=half_life))

        for h in horizons:
            ic = cross_sectional_ic(feat, ret[h], min_names)
            daily_ic[h][name] = ic
            ic_rows.append(_tag(ic_summary(ic, h, sign, rcfg), name, h, spec))
            stab_rows.append(ic_stability(ic, h, rcfg).assign(feature=name, family=spec["family"], horizon=h))

            ic_ex = cross_sectional_ic(feat, exclude_terminal(ret[h], terminal[h]), min_names)
            full, ex = ic_summary(ic, h, sign, rcfg), ic_summary(ic_ex, h, sign, rcfg)
            term_rows.append(
                _tag(
                    {
                        "ic_mean_all": full["ic_mean"], "nw_t_all": full["nw_t"],
                        "ic_mean_ex_terminal": ex["ic_mean"], "nw_t_ex_terminal": ex["nw_t"],
                        "dates_changed": int((~np.isclose(ic.fillna(0), ic_ex.fillna(0), atol=1e-12)).sum()),
                        "terminal_label_rows": int((terminal[h].to_numpy() == 1).sum()),
                    },
                    name, h, spec,
                )
            )

            deciles = quantile_returns(buckets, ret[h], q)
            dsum = decile_summary(deciles, h, offset)
            for d, value in dsum["means"].items():
                dec_rows.append(_tag({"decile": d, "mean_return": value}, name, h, spec))
            ls = long_short_stats(deciles, h, sign, rcfg, offset)
            ls_rows.append(
                _tag({**ls, "monotonicity": dsum["monotonicity"],
                      "frac_adjacent_increasing": dsum["frac_adjacent_increasing"]}, name, h, spec)
            )
            pos = non_overlapping_positions(len(deciles), h, offset)
            spread = (deciles.iloc[:, -1] - deciles.iloc[:, 0]).iloc[pos] * (sign or 1)
            spreads[h][name] = spread

            to = turnover(buckets, q, h, offset)
            row = {"D1_turnover": to["D1"].mean(), f"D{q}_turnover": to[f"D{q}"].mean(), "n_rebalances": len(to)}
            to_rows.append(_tag(row, name, h, spec))

            fm = fama_macbeth({name: z}, xs[h], min_names)
            slope = fm_summary(fm, h).query("regressor == @name").iloc[0].to_dict()
            fm_rows.append(_tag(slope, name, h, spec))
        print(f"  {name:<24} done ({time.time() - started:.0f}s)")
        del frame, z, buckets

    methods, alpha = rcfg["multiple_testing"]["methods"], rcfg["multiple_testing"]["alpha"]
    summary = add_multiple_testing(pd.DataFrame(ic_rows), "p_value", methods, alpha)
    long_short = add_multiple_testing(pd.DataFrame(ls_rows), "gross_raw_p_value", methods, alpha)
    fm_uni = add_multiple_testing(pd.DataFrame(fm_rows).drop(columns="regressor"), "p_value", methods, alpha)
    stability = pd.concat(stab_rows, ignore_index=True)
    decay_all = pd.concat(decay_rows, ignore_index=True)

    # multivariate Fama-MacBeth, one representative per family
    reps = list(rcfg["fama_macbeth"]["multivariate_representatives"].values())
    zframe = load([f"z_{r}" for r in reps])
    zwide = {r: grid.wide(zframe[f"z_{r}"]) for r in reps}
    del zframe
    fm_multi, collinearity = [], []
    for h in horizons:
        fm = fama_macbeth(zwide, xs[h], min_names)
        table = fm_summary(fm, h).assign(horizon=h)
        fm_multi.append(table)
        collinearity.append(
            table.loc[table["regressor"] != "const",
                      ["horizon", "regressor", "avg_vif", "avg_condition_number", "avg_abs_pairwise_corr"]]
        )
    fm_multi = pd.concat(fm_multi, ignore_index=True)
    collinearity = pd.concat(collinearity, ignore_index=True)
    del zwide

    # cross-signal structure, grouped by family
    ordered = sorted(features, key=lambda n: (signals[n]["family"], features.index(n)))
    corr = cross_signal_correlation({n: raw_wide[n] for n in ordered}, min_names)

    tables = {
        "ic_summary": summary,
        "ic_stability": stability,
        "ic_terminal_sensitivity": pd.DataFrame(term_rows),
        "decile_returns": pd.DataFrame(dec_rows),
        "long_short": long_short,
        "turnover": pd.DataFrame(to_rows),
        "ic_decay": decay_all,
        "fama_macbeth_univariate": fm_uni,
        "fama_macbeth_multivariate": fm_multi,
        "fm_collinearity": collinearity,
        "signal_correlation": corr.reset_index(names="feature"),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    headline_table(out).to_csv(out / "headline_results.csv", index=False)
    for h in horizons:
        pd.DataFrame(daily_ic[h]).to_parquet(cache / f"daily_ic_h{h}.parquet")
        pd.DataFrame(spreads[h]).to_parquet(cache / f"long_short_h{h}.parquet")

    # nothing may reach the holdout
    latest = max(max(s.index.max() for s in daily_ic[h].values()) for h in horizons)
    assert latest <= cutoff < pd.Timestamp(rcfg["holdout_start"]), "output reaches the holdout"

    # figures
    plots.plot_tstat_overview(summary, rcfg["multiple_testing"]["t_threshold"], fig_dir / "ic_tstat_overview.png")
    for h in horizons:
        plots.plot_ic_year_heatmap(stability, h, fig_dir / f"ic_by_year_h{h}.png")
        plots.plot_decile_means(tables["decile_returns"], h, features, fig_dir / f"decile_means_h{h}.png")
        best = (summary[summary["horizon"] == h].assign(a=lambda d: d["nw_t"].abs())
                .nlargest(5, "a")["feature"].tolist())
        plots.plot_cumulative_ic(pd.DataFrame(daily_ic[h]), best, f"Cumulative IC, h = {h}d (top 5 by |t|)",
                                 fig_dir / f"cumulative_ic_h{h}.png")
        plots.plot_long_short(pd.DataFrame(spreads[h]), h, best, fig_dir / f"long_short_h{h}.png")
    plots.plot_decay(decay_all, features, fig_dir / "ic_decay.png")
    plots.plot_correlation(corr, fig_dir / "signal_correlation.png")

    metadata = {
        **identity,
        "holdout_start": rcfg["holdout_start"],
        "embargo_trading_days": embargo_days(rcfg, fcfg),
        "first_research_date": str(grid.dates[0].date()),
        "last_research_date": str(grid.dates[-1].date()),
        "n_research_dates": len(grid.dates),
        "n_research_rows": len(index),
        "n_tests": len(summary),
        "rank_duplicates_in_family": list(RANK_DUPLICATES),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "pyarrow": pyarrow.__version__,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return tables


if __name__ == "__main__":
    run()
