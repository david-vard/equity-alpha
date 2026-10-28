"""Portfolio backtest: tranche-based constructions with transaction costs on the out-of-sample forecasts.

Usage (from the repository root):
    python -m src.run_portfolio --tune       [--config configs/portfolio.yaml] [--signals ...]
    python -m src.run_portfolio --evaluate   [--config configs/portfolio.yaml] [--signals ...] [--constructions ...]

Reads only research-sample rows through ``src.signals.load_research_panel`` and the OOS predictions
in ``data/processed/oof_predictions.parquet``. The design is fixed in ``configs/portfolio.yaml``:
the run refuses to start unless it, ``configs/research.yaml`` and the feature config are committed
and unmodified, unless ``--allow-dirty-config`` is given.

``--tune`` reads nothing dated after the last trading day before ``sample.split_date``: it runs
``optimised`` over the (lambda_risk, lambda_tc) grid on the tuning window and writes the choice per
signal to ``<reports_dir>/tuning/``. ``--evaluate`` then runs every signal x construction x cost
scenario over the whole sample with those frozen hyperparameters (it refuses if the tuning output
was made with another config), and writes tidy CSVs, figures and ``run_metadata.json`` to
``reports/portfolio/`` and daily returns and weights to ``data/processed/`` (gitignored).
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

matplotlib.use("Agg")  # headless: figures are only saved
import matplotlib.pyplot as plt  # noqa: E402

from src.backtest import MarketData, performance, run_backtest  # noqa: E402
from src.costs import CostModel  # noqa: E402
from src.models import MODELS_CONFIG_PATH, baseline_weights, load_models_config, zero_fill  # noqa: E402
from src.portfolio import (  # noqa: E402
    PORTFOLIO_CONFIG_PATH,
    UNVARIED,
    OptimiserSpec,
    decile_builder,
    evaluation_start,
    load_portfolio_config,
    optimised_builder,
    volscaled_builder,
)
from src.signals import (  # noqa: E402
    RESEARCH_CONFIG_PATH,
    Grid,
    cross_sectional_ic,
    load_research_config,
    load_research_panel,
    panel_path,
    research_cutoff,
)
from src.utils import DEFAULT_CONFIG_PATH, REPO_ROOT, file_sha256, load_config, provenance  # noqa: E402
from src.validation import walk_forward_folds  # noqa: E402

CONSTRUCTION_ORDER = ("decile_ls", "neutral_volscaled", "optimised")
WINDOWS = ("tuning", "evaluation", "full")


class Inputs:
    """Research-panel arrays on the research calendar (dates up to ``data_end``) plus predictions."""

    def __init__(self, panel: pd.DataFrame, preds: pd.DataFrame, pcfg: dict[str, Any], feature_cols: list[str]):
        self.grid = Grid.from_index(panel.index)
        self.dates, self.tickers = self.grid.dates, self.grid.tickers
        con, costs = pcfg["constructions"], pcfg["costs"]
        names = {
            "ret": pcfg["timing"]["daily_return"],
            "vol": con["optimised"]["alpha"]["vol_feature"],
            "beta": con["optimised"]["constraints"]["beta_feature"],
            "adv": costs["impact"]["adv_feature"],
        }
        self.arrays = {k: self.wide(panel[c]) for k, c in names.items()}
        self.labels = {c: self.wide(panel[c]) for c in panel.columns if c.startswith("fwd_ret_")}
        self.z = panel[feature_cols].to_numpy(dtype="float64") if feature_cols else None
        self.feature_cols = feature_cols
        rows = self.grid.rows, self.grid.cols
        pos = panel.index.get_indexer(preds.index)
        if (pos < 0).any():
            raise ValueError("predictions contain (date, ticker) rows that are not in the research panel")
        self.scores = {}
        for column in preds.columns.drop("fold_id", errors="ignore"):
            wide = np.full((len(self.dates), len(self.tickers)), np.nan)
            wide[rows[0][pos], rows[1][pos]] = preds[column].to_numpy(dtype="float64")
            self.scores[column] = wide
        self.first_oos = int(np.flatnonzero(np.isin(self.dates, preds.index.get_level_values("date").unique()))[0])

    def wide(self, values: pd.Series) -> np.ndarray:
        return self.grid.wide(values.to_numpy()).to_numpy(dtype="float64")


def calendar_dates(panel_file: Path) -> pd.DatetimeIndex:
    """Trading calendar from the panel's date index (no data columns are read)."""
    return pd.DatetimeIndex(pd.read_parquet(panel_file, columns=[]).index.get_level_values("date").unique()).sort_values()


def calendar_extension(panel_file: Path, last: pd.Timestamp, n_days: int, holdout_start: str) -> pd.DatetimeIndex:
    """The ``n_days`` trading dates after ``last`` (open tranches run to their label end); never a
    date on or after the holdout."""
    calendar = calendar_dates(panel_file)
    after = calendar[(calendar > last) & (calendar < pd.Timestamp(holdout_start))]
    return after[:n_days]


def market_data(inp: Inputs, column: str, h: int, lag: int, window: int, extension: pd.DatetimeIndex) -> tuple[MarketData, int]:
    """Engine inputs for one signal: from ``window`` days before the first OOS date to the end of the
    data plus ``extension``; labels whose window ends after the last engine date are masked."""
    lo = max(0, inp.first_oos - window)
    dates = inp.dates[lo:].append(extension)
    pad = len(extension)

    def take(a: np.ndarray) -> np.ndarray:
        return np.vstack([a[lo:], np.full((pad, a.shape[1]), np.nan)])

    label = take(inp.labels[f"fwd_ret_{h}d"])
    ends = np.arange(len(dates)) + lag + h
    label[ends > len(dates) - 1] = np.nan
    data = MarketData(dates, inp.tickers, take(inp.scores[column]), take(inp.arrays["ret"]), label,
                      take(inp.arrays["vol"]), take(inp.arrays["beta"]), take(inp.arrays["adv"]))
    return data, inp.first_oos - lo


def fold_ics(
    inp: Inputs, spec: dict[str, Any], h: int, lag: int, mcfg: dict[str, Any], ic_table: pd.DataFrame,
    min_names: int, ic_floor: float,
) -> dict[int, float]:
    """Test year -> IC for Grinold alpha: the mean daily IC on that fold's purged validation block.

    Models: ``validation_ic`` of the chosen hyperparameters (``reports/models/hyperparameters.csv``).
    B0 / B1: their fixed score on the same dates, from the panel. Floored at ``ic_floor``.
    """
    wf = mcfg["walk_forward"]
    folds = walk_forward_folds(inp.dates, h, lag, wf["first_test_year"], wf["inner_validation"]["years"], inp.dates[-1])
    out = {}
    if spec["predictor"] in UNVARIED:
        score = zero_fill(inp.z) @ baseline_weights(spec["predictor"], inp.feature_cols, mcfg)
        score_w = pd.DataFrame(inp.grid.wide(score).to_numpy(dtype="float64"), index=inp.dates)
        label_w = pd.DataFrame(inp.labels[f"fwd_ret_{h}d"], index=inp.dates)
        for fold in folds:
            ic = cross_sectional_ic(score_w.iloc[fold.validation], label_w.iloc[fold.validation], min_names)
            out[fold.test_year] = float(ic.mean())
    else:
        rows = ic_table[(ic_table["model"] == spec["predictor"]) & (ic_table["horizon"] == h)
                        & (ic_table["variant"] == spec["variant"])]
        by_fold = dict(zip(rows["fold_id"].astype(int), rows["validation_ic"].astype(float)))
        for fold in folds:
            if fold.fold_id not in by_fold:
                raise ValueError(f"{spec['predictor']} h={h}: no validation IC for fold {fold.fold_id}")
            out[fold.test_year] = by_fold[fold.fold_id]
    return {year: max(ic, ic_floor) if np.isfinite(ic) else ic_floor for year, ic in out.items()}


def builder_for(
    construction: str, pcfg: dict[str, Any], rcfg: dict[str, Any], h: int, ics: dict[int, float],
    dates: pd.DatetimeIndex, lambdas: tuple[float, float] | None,
):
    min_names, td = rcfg["min_names_per_date"], rcfg["trading_days"]
    if construction == "decile_ls":
        return decile_builder(rcfg["quantiles"], min_names, pcfg["constructions"]["decile_ls"]["gross_leverage"])
    if construction == "neutral_volscaled":
        return volscaled_builder(pcfg, min_names, td)
    spec = OptimiserSpec.from_config(pcfg, h, *lambdas)
    years = dates.year
    return optimised_builder(pcfg, spec, lambda s: ics.get(int(years[s]), 0.0), min_names)


def windows(pcfg: dict[str, Any], dates: pd.DatetimeIndex, first_full: pd.Timestamp | None,
            tuning_end: pd.Timestamp) -> dict[str, tuple[Any, Any]]:
    eval_start = evaluation_start(pcfg, dates) if dates[-1] >= pd.Timestamp(pcfg["sample"]["split_date"]) else None
    return {
        "tuning": (first_full, tuning_end),
        "evaluation": (max(eval_start, first_full) if eval_start is not None and first_full is not None else None, dates[-1]),
        "full": (first_full, dates[-1]),
    }


def plot_cumulative(daily: pd.DataFrame, signal: str, eval_start: pd.Timestamp | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for construction, frame in daily.groupby("construction", sort=False):
        ax.plot(frame["date"], frame["net_pnl"].cumsum(), label=f"{construction} net")
    if eval_start is not None:
        ax.axvline(eval_start, color="grey", ls="--", lw=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{signal}: cumulative net return per $ of capital (base costs and AUM)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tune", action="store_true", help="choose lambdas on the tuning window only")
    mode.add_argument("--evaluate", action="store_true", help="full sample with the frozen tuning output")
    parser.add_argument("--config", type=Path, default=PORTFOLIO_CONFIG_PATH)
    parser.add_argument("--research-config", type=Path, default=None)
    parser.add_argument("--features-config", type=Path, default=None)
    parser.add_argument("--models-config", type=Path, default=None)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--ic-table", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="reports directory (default: outputs.reports_dir)")
    parser.add_argument("--data-out", type=Path, default=None, help="directory for daily returns and weights")
    parser.add_argument("--signals", default=None, help="subset of the configured signal columns")
    parser.add_argument("--constructions", default=",".join(CONSTRUCTION_ORDER))
    parser.add_argument("--allow-dirty-config", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    with open(args.config) as handle:
        paths = yaml.safe_load(handle)  # only the config paths; validated by load_portfolio_config below
    features_path = args.features_config or REPO_ROOT / paths["features_config"]
    research_path = args.research_config or REPO_ROOT / paths["research_config"]
    models_path = args.models_config or REPO_ROOT / paths["models_config"]
    fcfg = load_config(features_path)
    rcfg = load_research_config(research_path, fcfg)
    mcfg = load_models_config(models_path, rcfg)
    pred_file = args.predictions or REPO_ROOT / paths["predictions"]
    pcfg = load_portfolio_config(args.config, fcfg, rcfg, mcfg, predictions=pred_file)
    panel_file = args.panel or panel_path(fcfg)
    allow = args.allow_dirty_config
    identity = provenance(args.config, features_path, panel_file, allow)
    research_identity = provenance(research_path, features_path, panel_file, allow)
    features_identity = provenance(features_path, features_path, panel_file, allow)

    signals = args.signals.split(",") if args.signals else list(pcfg["signals"])
    constructions = [c for c in args.constructions.split(",") if c]
    if set(signals) - set(pcfg["signals"]) or set(constructions) - set(CONSTRUCTION_ORDER):
        raise SystemExit(f"--signals must be within {list(pcfg['signals'])}, --constructions within {CONSTRUCTION_ORDER}")
    out = args.out or REPO_ROOT / pcfg["outputs"]["reports_dir"]
    data_out = args.data_out or REPO_ROOT / Path(pcfg["outputs"]["daily_returns"]).parent
    tuning_dir = out / "tuning"

    holdout = rcfg["holdout_start"]
    lag, td, min_names = pcfg["timing"]["lag"], rcfg["trading_days"], rcfg["min_names_per_date"]
    window = pcfg["risk_model"]["window"]
    opt = pcfg["constructions"]["optimised"]
    cutoff = research_cutoff(panel_file, rcfg, fcfg)
    calendar = calendar_dates(panel_file)
    split = pd.Timestamp(pcfg["sample"]["split_date"])
    tuning_end = calendar[calendar < split][-1]
    data_end = tuning_end if args.tune else cutoff
    if data_end >= pd.Timestamp(holdout):
        raise AssertionError("the data end reaches the holdout")

    # ------------------------------------------------------------------ inputs (nothing after data_end)
    horizons = sorted({pcfg["signals"][s]["horizon"] for s in signals})
    needs_baseline_ic = args.tune or "optimised" in constructions
    feature_cols = mcfg["features"]["columns"] if needs_baseline_ic and any(
        pcfg["signals"][s]["predictor"] in UNVARIED for s in signals) else []
    con = pcfg["constructions"]
    columns = [pcfg["timing"]["daily_return"], con["optimised"]["alpha"]["vol_feature"],
               con["optimised"]["constraints"]["beta_feature"], pcfg["costs"]["impact"]["adv_feature"]]
    columns += [f"fwd_ret_{h}d" for h in horizons] + list(feature_cols)
    panel = load_research_panel(list(dict.fromkeys(columns)), panel_file, rcfg, fcfg, data_end)
    preds = pd.read_parquet(pred_file, columns=signals + ["fold_id"], filters=[("date", "<=", data_end)])
    inp = Inputs(panel, preds, pcfg, list(feature_cols))
    del panel, preds
    if inp.dates[-1] > data_end:
        raise AssertionError("inputs reach beyond the data end")
    ic_table = pd.read_csv(args.ic_table or REPO_ROOT / opt["alpha"]["ic_table"])
    extension = (pd.DatetimeIndex([]) if args.tune
                 else calendar_extension(panel_file, cutoff, lag + max(horizons), holdout))
    print(f"data: {inp.dates[0].date()} to {inp.dates[-1].date()} ({len(inp.dates)} dates), first OOS "
          f"{inp.dates[inp.first_oos].date()}, mode {'tune' if args.tune else 'evaluate'}")

    base_costs = CostModel.from_config(pcfg, td)
    specs = {s: pcfg["signals"][s] for s in signals}
    prepared = {}
    for column, spec in specs.items():
        h = spec["horizon"]
        data, start = market_data(inp, column, h, lag, window, extension)
        if data.dates.max() >= pd.Timestamp(holdout):
            raise AssertionError("a holdout date reached the engine inputs")
        ics = fold_ics(inp, spec, h, lag, mcfg, ic_table, min_names, opt["alpha"]["ic_floor"]) \
            if needs_baseline_ic else {}
        prepared[column] = (spec, data, start, ics)

    meta_common = {
        **identity,
        **{f"research_{k}": v for k, v in research_identity.items() if k.startswith("config")},
        **{f"features_{k}": v for k, v in features_identity.items() if k.startswith("config")},
        "predictions_sha256": file_sha256(pred_file),
        "holdout_start": holdout,
        "last_research_date": str(cutoff.date()),
        "tuning_end": str(tuning_end.date()),
        "data_end": str(data_end.date()),
        "signals": signals,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "cvxpy": __import__("cvxpy").__version__,
    }

    # ------------------------------------------------------------------ tune
    if args.tune:
        tuning_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for column, (spec, data, start, ics) in prepared.items():
            h = spec["horizon"]
            for order, (lr, lt) in enumerate(itertools.product(opt["lambda_risk"], opt["lambda_tc"])):
                build = builder_for("optimised", pcfg, rcfg, h, ics, data.dates, (lr, lt))
                res = run_backtest(data, build, base_costs, h, lag, h, holdout, start, record_weights=False)
                if res.total.index.max() > tuning_end:
                    raise AssertionError("the tuning run reached past the tuning window")
                stats = performance(res.total, res.first_full_date, tuning_end, h, td)
                rows.append({"signal": column, "horizon": h, "grid_order": order, "lambda_risk": lr,
                             "lambda_tc": lt, **stats, "fallbacks": sum(res.fallbacks.values()),
                             **{f"fallback_{k}": v for k, v in res.fallbacks.items()}})
                print(f"  {column} lambda_risk={lr} lambda_tc={lt}: done ({time.time() - started:.0f}s)")
        grid = pd.DataFrame(rows)
        key = grid["net_sharpe"].fillna(-np.inf)
        chosen = grid.loc[key.groupby(grid["signal"]).idxmax()]  # idxmax keeps the first: config order
        chosen = chosen[["signal", "horizon", "grid_order", "lambda_risk", "lambda_tc", "net_sharpe"]]
        grid.to_csv(tuning_dir / "lambda_grid.csv", index=False)
        chosen.to_csv(tuning_dir / "selected_hyperparameters.csv", index=False)
        meta = {**meta_common, "mode": "tune", "runtime_seconds": round(time.time() - started, 1)}
        (tuning_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        print(chosen.to_string(index=False))
        return {"lambda_grid": grid, "selected_hyperparameters": chosen}

    # ------------------------------------------------------------------ evaluate
    selected = {}
    if "optimised" in constructions:
        sel_file, meta_file = tuning_dir / "selected_hyperparameters.csv", tuning_dir / "run_metadata.json"
        if not sel_file.exists() or not meta_file.exists():
            raise SystemExit(f"run --tune first: {sel_file} is missing")
        tuned = json.loads(meta_file.read_text())
        if tuned.get("config_sha256") != identity["config_sha256"]:
            raise SystemExit("the tuning output was made with another portfolio config; re-run --tune")
        sel = pd.read_csv(sel_file)
        selected = {r.signal: (float(r.lambda_risk), float(r.lambda_tc)) for r in sel.itertuples()}
        if set(signals) - set(selected):
            raise SystemExit(f"no tuned lambdas for {sorted(set(signals) - set(selected))}; re-run --tune")

    scen = pcfg["costs"]["scenarios"]
    base_aum = pcfg["costs"]["base_aum_usd"]
    scenarios = [("base", 1.0, base_aum)]
    scenarios += [(f"cost_x{m:g}", float(m), base_aum) for m in scen["cost_multipliers"] if m != 1.0]
    scenarios += [(f"aum_{a:.0e}", 1.0, a) for a in scen["aum_usd"] if a != base_aum]

    summary_rows, fallback_rows, daily_frames, weight_frames = [], [], [], []
    fig_dir = out / "figures"
    for column, (spec, data, start, ics) in prepared.items():
        h = spec["horizon"]
        base_daily = []
        for construction in constructions:
            for name, mult, aum in scenarios:
                costs = base_costs.scenario(mult, aum)
                build = builder_for(construction, pcfg, rcfg, h, ics, data.dates, selected.get(column))
                res = run_backtest(data, build, costs, h, lag, h, holdout, start, record_weights=name == "base")
                tag = {"signal": column, "horizon": h, "construction": construction, "scenario": name,
                       "cost_multiplier": mult, "aum_usd": aum,
                       "lambda_risk": selected.get(column, (np.nan, np.nan))[0] if construction == "optimised" else np.nan,
                       "lambda_tc": selected.get(column, (np.nan, np.nan))[1] if construction == "optimised" else np.nan}
                for window_name, (lo, hi) in windows(pcfg, data.dates, res.first_full_date, tuning_end).items():
                    stats = performance(res.total, lo, hi, h, td) if lo is not None else {}
                    if name != "base":
                        stats["breakeven_cost_multiplier"] = np.nan  # defined at the base-case trades only
                    summary_rows.append({**tag, "window": window_name, **stats})
                fallback_rows.append({**tag, "rebalances": res.n_rebalances, **dict(res.fallbacks)})
                total = res.total.reset_index(names="date").assign(tranche="total", **tag)
                daily_frames.append(total)
                if name == "base":
                    base_daily.append(total)
                    daily_frames.append(res.tranches.assign(tranche=res.tranches["tranche"].astype(str), **tag))
                    weight_frames.append(res.weights.assign(signal=column, construction=construction))
                print(f"  {column} {construction:<17} {name:<10} done ({time.time() - started:.0f}s)")
        eval_start = evaluation_start(pcfg, data.dates)
        plot_cumulative(pd.concat(base_daily), column, eval_start, fig_dir / f"cumulative_net_{column}.png")

    summary = pd.DataFrame(summary_rows)
    head = pcfg["headline"]
    headline = summary[
        summary["signal"].isin([head["signal"], head["benchmark"]]) & (summary["construction"] == head["construction"])
        & (summary["scenario"] == "base") & (summary["window"] == head["window"])
    ][["signal", "construction", "window", "start", "end", "n_days", "net_sharpe", "net_nw_t", "gross_sharpe",
       "annual_turnover", "breakeven_cost_multiplier"]] if len(summary) else pd.DataFrame()
    tables = {"summary": summary, "headline": headline, "fallbacks": pd.DataFrame(fallback_rows).fillna(0)}
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    data_out.mkdir(parents=True, exist_ok=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    daily["fallback"] = daily["fallback"].fillna("").astype(str)
    daily["rebalanced"] = daily["rebalanced"].astype("boolean")
    daily.to_parquet(data_out / Path(pcfg["outputs"]["daily_returns"]).name)
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    weights.to_parquet(data_out / Path(pcfg["outputs"]["weights"]).name)

    last = max(d.dates.max() for _, d, _, _ in prepared.values())
    assert last < pd.Timestamp(holdout), "output reaches the holdout"
    meta = {
        **meta_common,
        "mode": "evaluate",
        "constructions": constructions,
        "scenarios": [s[0] for s in scenarios],
        "selected_lambdas": {k: list(v) for k, v in selected.items()},
        "evaluation_start": str(evaluation_start(pcfg, inp.dates).date()) if inp.dates[-1] >= split else None,
        "last_engine_date": str(last.date()),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(headline.to_string(index=False))
    return tables


if __name__ == "__main__":
    run()
