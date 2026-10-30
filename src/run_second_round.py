"""Run the pre-registered second round: variants chosen on 2007-2013, each then run once.

Usage (from the repository root, in this order):
    python -m src.run_second_round --tune --shard 0/4      # one process per shard
    python -m src.run_second_round --select                # checks, choices and the holdout candidate
    python -m src.run_second_round --evaluate --shard 0/2  # the chosen variants, once
    python -m src.run_second_round --report                # -> reports/second_round/

``--tune`` reads nothing dated after the last tuning date. ``--select`` writes the choices and the
holdout candidate before any evaluation-window return of a variant exists, and writes only
``checks.csv`` if a check fails. ``--evaluate`` refuses a selection made with another config. Every
evaluation-window row is labelled ``second_look``. The engine, the constructions and
``configs/portfolio.yaml`` are unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.backtest import BacktestResult, MarketData, performance, run_backtest
from src.costs import CostModel
from src.metrics import sharpe
from src.models import load_models_config
from src.portfolio import UNVARIED, load_portfolio_config
from src.run_portfolio import Inputs, builder_for, calendar_dates, calendar_extension, fold_ics, market_data
from src.second_round import (
    SECOND_ROUND_CONFIG_PATH,
    Variant,
    band_builder,
    enumerate_variants,
    holdout_candidate,
    liquid_scores,
    load_second_round_config,
    select_variants,
)
from src.sensitivity import percentile_interval, resampled_sharpes, stationary_indices
from src.signals import load_research_config, load_research_panel, panel_path, research_cutoff
from src.utils import REPO_ROOT, file_sha256, load_config, provenance

NEUTRAL = "neutral_volscaled"


@dataclass
class Context:
    cfg: dict[str, Any]
    pcfg: dict[str, Any]
    rcfg: dict[str, Any]
    prepared: dict[str, tuple[dict[str, Any], MarketData, int, dict[int, float]]]
    lambdas: dict[str, tuple[float, float]]
    costs: CostModel
    tuning_end: pd.Timestamp
    identity: dict[str, Any]


def recorded_path(cfg: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / cfg["recorded"][key]


def recorded_lambdas(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(recorded_path(cfg, "hyperparameters"))


def prepare(args: argparse.Namespace, cfg: dict[str, Any], pcfg: dict[str, Any], rcfg: dict[str, Any], fcfg: dict[str, Any],
            mcfg: dict[str, Any], identity: dict[str, Any], tune: bool) -> Context:
    """Engine inputs for both signals, with nothing after the tuning end in a tuning run."""
    holdout = rcfg["holdout_start"]
    panel_file = args.panel or panel_path(fcfg)
    pred_file = args.predictions or REPO_ROOT / pcfg["predictions"]
    lag, min_names = pcfg["timing"]["lag"], rcfg["min_names_per_date"]
    opt = pcfg["constructions"]["optimised"]
    cutoff = research_cutoff(panel_file, rcfg, fcfg)
    calendar = calendar_dates(panel_file)
    tuning_end = calendar[calendar < pd.Timestamp(pcfg["sample"]["split_date"])][-1]
    data_end = tuning_end if tune else cutoff
    if data_end >= pd.Timestamp(holdout):
        raise AssertionError("the data end reaches the holdout")

    signals = [cfg["signals"]["primary"], cfg["signals"]["benchmark"]]
    specs = {s: pcfg["signals"][s] for s in signals}
    feature_cols = mcfg["features"]["columns"] if any(s["predictor"] in UNVARIED for s in specs.values()) else []
    con = pcfg["constructions"]
    columns = [pcfg["timing"]["daily_return"], con["optimised"]["alpha"]["vol_feature"],
               con["optimised"]["constraints"]["beta_feature"], pcfg["costs"]["impact"]["adv_feature"]]
    columns += [f"fwd_ret_{h}d" for h in sorted({s["horizon"] for s in specs.values()})] + list(feature_cols)
    panel = load_research_panel(list(dict.fromkeys(columns)), panel_file, rcfg, fcfg, data_end)
    preds = pd.read_parquet(pred_file, columns=signals + ["fold_id"], filters=[("date", "<=", data_end)])
    inp = Inputs(panel, preds, pcfg, list(feature_cols))
    del panel, preds
    if inp.dates[-1] > data_end:
        raise AssertionError("inputs reach beyond the data end")
    ic_table = pd.read_csv(args.ic_table or REPO_ROOT / opt["alpha"]["ic_table"])
    extension = (pd.DatetimeIndex([]) if tune else
                 calendar_extension(panel_file, cutoff, lag + max(s["horizon"] for s in specs.values()), holdout))
    prepared = {}
    for column, spec in specs.items():
        data, start = market_data(inp, column, spec["horizon"], lag, pcfg["risk_model"]["window"], extension)
        if data.dates.max() >= pd.Timestamp(holdout):
            raise AssertionError("a holdout date reached the engine inputs")
        ics = fold_ics(inp, spec, spec["horizon"], lag, mcfg, ic_table, min_names, opt["alpha"]["ic_floor"])
        prepared[column] = (spec, data, start, ics)
    chosen = recorded_lambdas(cfg).set_index("signal")
    lambdas = {s: (float(chosen.loc[s, "lambda_risk"]), float(chosen.loc[s, "lambda_tc"])) for s in signals}
    return Context(cfg, pcfg, rcfg, prepared, lambdas, CostModel.from_config(pcfg, rcfg["trading_days"]), tuning_end, identity)


def run_variant(ctx: Context, v: Variant) -> tuple[BacktestResult, dict[str, int]]:
    """One backtest of a variant, and how often its special rules applied."""
    spec, data, start, ics = ctx.prepared[v.signal]
    h, td = spec["horizon"], ctx.rcfg["trading_days"]
    lag, min_names = ctx.pcfg["timing"]["lag"], ctx.rcfg["min_names_per_date"]
    counts: Counter = Counter()
    if v.fraction is not None:
        data = replace(data, score=liquid_scores(data.score, data.adv, v.fraction))
        names = np.isfinite(data.score[start:]).sum(axis=1)
        counts["thin_dates"] = int(((names > 0) & (names < min_names)).sum())  # run-off days have no scores at all
    if v.band is not None:
        build = band_builder(ctx.pcfg, v.band, min_names, td, counts)
    else:
        build = builder_for(v.construction, ctx.pcfg, ctx.rcfg, h, ics, data.dates, v.lambdas or ctx.lambdas[v.signal])
    res = run_backtest(data, build, ctx.costs, h, lag, h, ctx.rcfg["holdout_start"], start, record_weights=False)
    return res, dict(counts)


def tune_rows(ctx: Context, variants: list[Variant], started: float) -> pd.DataFrame:
    """Tuning statistics, and each recorded point's gap from the first round."""
    rows = []
    td = ctx.rcfg["trading_days"]
    for v in variants:
        res, counts = run_variant(ctx, v)
        if res.total.index.max() > ctx.tuning_end:
            raise AssertionError("the tuning run reached past the tuning window")
        h = ctx.prepared[v.signal][0]["horizon"]
        stats = performance(res.total, res.first_full_date, ctx.tuning_end, h, td)
        rows.append({**v.row(), **stats, "fallbacks": sum(res.fallbacks.values()), **counts,
                     "recorded_gap": recorded_gap(ctx, v, res, stats)})
        print(f"  {v.name}: net Sharpe {stats['net_sharpe']:.3f} ({time.time() - started:.0f}s)")
    return pd.DataFrame(rows)


def recorded_gap(ctx: Context, v: Variant, res: BacktestResult, stats: dict[str, Any]) -> float:
    """How far a rerun recorded point is from the first round: daily for the closed-form book, on Sharpe if solved."""
    if not v.recorded:
        return np.nan
    if v.construction == NEUTRAL:
        path = recorded_path(ctx.cfg, "daily_returns").as_posix().format(signal=v.signal)
        kept = pd.read_parquet(path, columns=["date", "net_pnl"], filters=[
            ("signal", "==", v.signal), ("construction", "==", NEUTRAL), ("tranche", "==", "total"),
            ("scenario", "==", "base"), ("date", "<=", ctx.tuning_end)]).set_index("date")["net_pnl"]
        new = res.total["net_pnl"].loc[res.first_full_date:ctx.tuning_end]
        return float((new - kept.reindex(new.index)).abs().max())
    grid = pd.read_csv(recorded_path(ctx.cfg, "lambda_grid"))
    risk, tc = ctx.lambdas[v.signal]
    point = grid[(grid["signal"] == v.signal) & (grid["lambda_risk"] == risk) & (grid["lambda_tc"] == tc)].iloc[0]
    return abs(float(stats["net_sharpe"]) - float(point["net_sharpe"]))


def shard_of(items: list[Any], shard: str) -> list[Any]:
    index, count = (int(x) for x in shard.split("/"))
    return items[index::count]


def add_recorded_lambda_rows(cfg: dict[str, Any], variants: list[Variant], tuned: pd.DataFrame) -> pd.DataFrame:
    """The grid with the recorded lambda points, read from the first round rather than rerun."""
    recorded = pd.read_csv(recorded_path(cfg, "lambda_grid"))
    rows = []
    for v in variants:
        if v.family == "wider_lambda_grid" and v.recorded:
            match = recorded[(recorded["signal"] == v.signal) & (recorded["lambda_risk"] == v.lambdas[0])
                             & (recorded["lambda_tc"] == v.lambdas[1])].iloc[0]
            rows.append({**match.to_dict(), **v.row()})
    return pd.concat([tuned, pd.DataFrame(rows)], ignore_index=True)


def selection_checks(cfg: dict[str, Any], variants: list[Variant], grid: pd.DataFrame, tuning_end: pd.Timestamp) -> pd.DataFrame:
    tests = cfg["checks"]
    rows = []
    for row in grid[grid["recorded"].astype(bool) & grid["recorded_gap"].notna()].itertuples():
        limit = tests["recorded_book_tolerance"] if row.construction == NEUTRAL else tests["recorded_optimised_sharpe_tolerance"]
        rows.append({"check": "recorded_book", "variant": row.variant, "value": row.recorded_gap, "limit": limit,
                     "passed": bool(row.recorded_gap <= limit)})
    missing = sorted({v.name for v in variants} - set(grid["variant"]))
    rows.append({"check": "every_variant_tuned", "variant": "all", "value": float(len(missing)), "limit": 0.0, "passed": not missing})
    late = int((pd.to_datetime(grid["end"]) > tuning_end).sum())
    rows.append({"check": "tuning_isolation", "variant": "all", "value": float(late), "limit": 0.0, "passed": late == 0})
    return pd.DataFrame(rows)


def report(cfg: dict[str, Any], pcfg: dict[str, Any], rcfg: dict[str, Any], out: Path, data_out: Path) -> pd.DataFrame:
    """The second-look table: each chosen variant in the evaluation window, against its recorded book."""
    selected = pd.read_csv(out / "tuning" / "selected_variants.csv")
    td = rcfg["trading_days"]
    boot = cfg["bootstrap"]
    split = pd.Timestamp(pcfg["sample"]["split_date"])
    rows = []
    for row in selected.itertuples():
        path = recorded_path(cfg, "daily_returns").as_posix().format(signal=row.signal)
        base = pd.read_parquet(path, filters=[("signal", "==", row.signal), ("construction", "==", row.construction),
                                              ("tranche", "==", "total"), ("scenario", "==", "base"), ("date", ">=", split)])
        base = base.assign(date=pd.to_datetime(base["date"])).set_index("date").sort_index()
        if row.recorded:
            book = base
        else:
            book = pd.read_parquet(data_out / f"{row.variant}.parquet")
            book = book.assign(date=pd.to_datetime(book["date"])).set_index("date").sort_index().loc[split:]
        if not book.index.equals(base.index):
            raise SystemExit(f"{row.variant}: dates differ from the recorded book")
        if book.index.max() >= pd.Timestamp(rcfg["holdout_start"]):
            raise AssertionError("a holdout date reached the report")
        h = pcfg["signals"][row.signal]["horizon"]
        stats = performance(book, book.index[0], book.index[-1], h, td)
        index = stationary_indices(len(book), boot["n_resamples"], boot["mean_block_length"], boot["seed"])
        draws = resampled_sharpes(book["net_pnl"].to_numpy(), index, td)
        versus = draws - resampled_sharpes(base["net_pnl"].to_numpy(), index, td)
        rows.append({
            "evidence": cfg["evaluation"]["evidence_label"], "variant": row.variant, "family": row.family, "signal": row.signal,
            "construction": row.construction, "recorded": bool(row.recorded), "band": row.band, "fraction": row.fraction,
            "lambda_risk": row.lambda_risk, "lambda_tc": row.lambda_tc, "tuning_net_sharpe": row.net_sharpe,
            **{k: stats[k] for k in ("start", "end", "n_days", "net_sharpe", "net_nw_t", "net_return_ann", "net_vol_ann",
                                     "gross_sharpe", "annual_turnover", "breakeven_cost_multiplier")},
            **{f"net_sharpe_{k}": v for k, v in percentile_interval(draws, boot["confidence"]).items()},
            "recorded_net_sharpe": sharpe(base["net_pnl"], td), "difference": sharpe(book["net_pnl"], td) - sharpe(base["net_pnl"], td),
            **{f"difference_{k}": v for k, v in percentile_interval(versus, boot["confidence"]).items()},
        })
    return pd.DataFrame(rows)


def run(argv: list[str] | None = None) -> dict[str, pd.DataFrame] | None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("tune", "select", "evaluate", "report"):
        mode.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--shard", default="0/1", help="i/n: this process runs every n-th variant from i")
    parser.add_argument("--config", type=Path, default=SECOND_ROUND_CONFIG_PATH)
    parser.add_argument("--portfolio-config", type=Path, default=None)
    parser.add_argument("--research-config", type=Path, default=None)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--ic-table", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--data-out", type=Path, default=None)
    parser.add_argument("--allow-dirty-config", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    with open(args.config) as handle:
        raw = yaml.safe_load(handle)  # only the config paths; validated by load_second_round_config below
    features_path = REPO_ROOT / raw["features_config"]
    fcfg = load_config(features_path)
    rcfg = load_research_config(args.research_config or REPO_ROOT / raw["research_config"], fcfg)
    mcfg = load_models_config(REPO_ROOT / raw["models_config"], rcfg)
    pcfg = load_portfolio_config(args.portfolio_config or REPO_ROOT / raw["portfolio_config"], fcfg, rcfg, mcfg)
    cfg = load_second_round_config(args.config, pcfg, fcfg, rcfg)
    identity = provenance(args.config, features_path, args.panel or panel_path(fcfg), args.allow_dirty_config)

    portfolio_file = args.portfolio_config or REPO_ROOT / cfg["portfolio_config"]
    portfolio_meta = json.loads(recorded_path(cfg, "portfolio_metadata").read_text())
    if portfolio_meta["config_sha256"] != [file_sha256(portfolio_file)]:
        raise SystemExit("the recorded books were not made with the current configs/portfolio.yaml")

    out = args.out or REPO_ROOT / cfg["outputs"]["reports_dir"]
    data_out = args.data_out or REPO_ROOT / cfg["outputs"]["data_dir"]
    tuning_dir, shard_dir = out / "tuning", out / "tuning" / "shards"
    variants = enumerate_variants(cfg, pd.read_csv(recorded_path(cfg, "lambda_grid")))

    if args.tune:
        ctx = prepare(args, cfg, pcfg, rcfg, fcfg, mcfg, identity, tune=True)
        runnable = [v for v in variants if not (v.family == "wider_lambda_grid" and v.recorded)]
        shard_dir.mkdir(parents=True, exist_ok=True)
        rows = tune_rows(ctx, shard_of(runnable, args.shard), started)
        rows.to_csv(shard_dir / f"variant_grid_{args.shard.replace('/', '_of_')}.csv", index=False)
        return {"variant_grid": rows}

    if args.select:
        tuning_dir.mkdir(parents=True, exist_ok=True)
        tuned = pd.concat([pd.read_csv(p) for p in sorted(shard_dir.glob("variant_grid_*.csv"))], ignore_index=True)
        calendar = calendar_dates(args.panel or panel_path(fcfg))
        tuning_end = calendar[calendar < pd.Timestamp(pcfg["sample"]["split_date"])][-1]
        grid = add_recorded_lambda_rows(cfg, variants, tuned)
        checks = selection_checks(cfg, variants, grid, tuning_end)
        checks.to_csv(out / "checks.csv", index=False)
        if not checks["passed"].all():
            print(checks[~checks["passed"]].to_string(index=False))
            raise SystemExit("a pre-registered check failed: no choices were written")
        selected = select_variants(grid)
        recorded = recorded_lambdas(cfg).set_index("signal")
        candidate = holdout_candidate(selected, cfg["holdout_candidate"]["signal"],
                                      float(recorded.loc[cfg["holdout_candidate"]["signal"], "net_sharpe"]))
        grid.to_csv(tuning_dir / "variant_grid.csv", index=False)
        selected.to_csv(tuning_dir / "selected_variants.csv", index=False)
        candidate.to_csv(tuning_dir / "holdout_candidate.csv", index=False)
        meta = {**identity, "mode": "select", "n_variants_tried": len(variants), "holdout_start": rcfg["holdout_start"],
                "tuning_end": str(tuning_end.date())}
        (tuning_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        print(selected[["variant", "recorded", "net_sharpe", "on_grid_edge"]].to_string(index=False))
        print("holdout candidate:", candidate["variant"].tolist() or "none")
        return {"variant_grid": grid, "selected_variants": selected, "holdout_candidate": candidate, "checks": checks}

    meta_file = tuning_dir / "run_metadata.json"
    if not meta_file.exists() or not (tuning_dir / "selected_variants.csv").exists():
        raise SystemExit(f"run --select first: {tuning_dir / 'selected_variants.csv'} is missing")
    tuned_meta = json.loads(meta_file.read_text())
    if tuned_meta.get("config_sha256") != identity["config_sha256"]:
        raise SystemExit("the choices were made with another second-round config; run --tune and --select again")

    if args.evaluate:
        ctx = prepare(args, cfg, pcfg, rcfg, fcfg, mcfg, identity, tune=False)
        by_name = {v.name: v for v in variants}
        chosen = pd.read_csv(tuning_dir / "selected_variants.csv")
        runnable = [by_name[n] for n, recorded in zip(chosen["variant"], chosen["recorded"]) if not recorded]
        data_out.mkdir(parents=True, exist_ok=True)
        for v in shard_of(runnable, args.shard):
            res, counts = run_variant(ctx, v)
            res.total.reset_index(names="date").assign(tranche="total", **counts).to_parquet(data_out / f"{v.name}.parquet")
            print(f"  {v.name}: done ({time.time() - started:.0f}s)")
        return None

    table = report(cfg, pcfg, rcfg, out, data_out)
    table.to_csv(out / "evaluation.csv", index=False)
    meta = {**identity, "mode": "report", "n_variants_tried": tuned_meta["n_variants_tried"], "evidence": cfg["evaluation"]["evidence_label"],
            "holdout_start": rcfg["holdout_start"], "portfolio_config_sha256": file_sha256(portfolio_file)}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(table[["variant", "tuning_net_sharpe", "net_sharpe", "net_sharpe_ci_low", "net_sharpe_ci_high", "difference"]].to_string(index=False))
    return {"evaluation": table}


if __name__ == "__main__":
    run()
