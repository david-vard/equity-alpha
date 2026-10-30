"""Run the pre-registered holdout, once.

Usage (from the repository root):
    python -m src.run_holdout --open-holdout      # -> reports/holdout/

This is the only command that reads a row on or after ``holdout_start``. It refuses without the
flag, with a config or code change that isn't committed, if a pinned input has changed, and if the
holdout has already been opened. The one-shot lock is written before the first holdout row is read.

The LightGBM walk-forward continues every January. The three frozen books then run continuously from
the first out-of-sample date, and only the holdout window is reported. Every check is compared on
pre-holdout data, and no holdout statistic is computed or written until all of them pass. A failure
after the lock is reported as it stands: it doesn't reopen the holdout.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import BacktestResult, MarketData, performance, run_backtest
from src.costs import CostModel
from src.holdout import (
    HOLDOUT_CONFIG_PATH,
    interval_outcome,
    last_score_date,
    load_holdout_config,
    load_holdout_panel,
    open_lock,
    t_outcome,
)
from src.metrics import newey_west_lag, newey_west_tstat, sharpe
from src.models import hyperparameter_grid, load_models_config
from src.portfolio import load_portfolio_config
from src.portfolio_report import regression_beta
from src.run_models import BETA, Data, baseline_predictions, column_name, walk_forward
from src.run_portfolio import Inputs, builder_for, fold_ics, market_data
from src.second_round import liquid_scores
from src.sensitivity import breakevens, percentile_interval, resampled_sharpes, stationary_indices
from src.signals import cross_sectional_ic, load_research_config, panel_path, research_cutoff
from src.utils import REPO_ROOT, file_sha256, load_config, provenance
from src.validation import walk_forward_folds

BENCHMARK_PREDICTOR = "B0_reversal"


@dataclass(frozen=True)
class Paths:
    panel: Path
    predictions: Path
    model_hyperparameters: Path
    recorded_daily: str
    recorded_candidate: Path
    out: Path
    data_out: Path
    lock: Path


def recorded_parameters(table: pd.DataFrame, fold_id: int, horizon: int, variant: str, mcfg: dict[str, Any]) -> dict[str, Any]:
    """The grid entry the first round chose for a fold, with the config's own types."""
    row = table[(table["model"] == "lightgbm") & (table["horizon"] == horizon) & (table["variant"] == variant)
                & (table["fold_id"] == fold_id)].iloc[0]
    for params in hyperparameter_grid("lightgbm", mcfg):
        if all(float(row[k]) == float(v) for k, v in params.items()):
            return params
    raise ValueError(f"fold {fold_id}: the recorded hyperparameters are not on the grid")


def refit_scores(data: Data, cfg: dict[str, Any], mcfg: dict[str, Any], recorded: pd.DataFrame, oof: pd.DataFrame,
                 cutoff: pd.Timestamp, last_score: pd.Timestamp, lag: int, min_names: int,
                 ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Scores after the research cutoff, the new folds' choices, and the prediction checks.

    The last research fold is refit with its recorded hyperparameters and scores the embargoed dates.
    Each new fold repeats the first-round search. B0 is its fixed formula.
    """
    refit = cfg["model_refit"]
    h, variant = refit["horizon"], refit["target_variant"]
    wf = mcfg["walk_forward"]
    folds = walk_forward_folds(data.dates, h, lag, wf["first_test_year"], wf["inner_validation"]["years"], last_score)
    research_year = refit["test_years"][0] - 1
    last_fold = [f for f in folds if f.test_year == research_year]
    new_folds = [f for f in folds if f.test_year in refit["test_years"]]
    if len(last_fold) != 1 or [f.test_year for f in new_folds] != refit["test_years"]:
        raise ValueError("the folds do not cover the declared test years")
    target, columns = data.targets[h][variant], np.arange(data.X.shape[1])
    fixed = {last_fold[0].fold_id: recorded_parameters(recorded, last_fold[0].fold_id, h, variant, mcfg)}
    old, _, _ = walk_forward("lightgbm", data, columns, target, h, last_fold, mcfg, lag, min_names, fixed=fixed)
    new, chosen, _ = walk_forward("lightgbm", data, columns, target, h, new_folds, mcfg, lag, min_names)
    benchmark, _ = baseline_predictions(BENCHMARK_PREDICTOR, data, list(mcfg["features"]["columns"]), None,
                                        last_fold + new_folds, mcfg, min_names)

    lgb_column = column_name("lightgbm", h, variant)
    b0_column = column_name(BENCHMARK_PREDICTOR, h, None)
    frame = pd.DataFrame({lgb_column: np.where(np.isnan(new), old, new), b0_column: benchmark}, index=data.index)
    dates = data.index.get_level_values("date")
    in_year = dates.year == research_year
    overlap = frame[in_year & (dates <= cutoff)].join(oof[[lgb_column, b0_column]], rsuffix="_recorded", how="inner")
    checks = [{"check": f"predictions_{c}", "book": "all", "value": float((overlap[c] - overlap[f"{c}_recorded"]).abs().max()),
               "limit": cfg["checks"]["fold_2021_prediction_tolerance"]} for c in (lgb_column, b0_column)]
    later = frame[(dates > cutoff) & (dates <= last_score)].astype("float32")
    table = pd.DataFrame([{"model": "lightgbm", "horizon": h, "variant": variant, **row} for row in chosen])
    return later, table, checks


def recorded_sharpe(paths: Paths, cfg: dict[str, Any], book: str, cutoff: pd.Timestamp, trading_days: int) -> tuple[float, pd.Timestamp]:
    """The book's recorded net Sharpe to the last research date, and its first day."""
    spec = cfg["books"][book]
    if spec["liquid_fraction"] == 1:
        frame = pd.read_parquet(paths.recorded_daily.format(signal=spec["signal"]), columns=["date", "net_pnl"], filters=[
            ("signal", "==", spec["signal"]), ("construction", "==", spec["construction"]), ("tranche", "==", "total"),
            ("scenario", "==", "base"), ("date", "<=", cutoff)])
    else:
        frame = pd.read_parquet(paths.recorded_candidate, columns=["date", "net_pnl"], filters=[("date", "<=", cutoff)])
    series = frame.assign(date=pd.to_datetime(frame["date"])).set_index("date").sort_index()["net_pnl"]
    return sharpe(series, trading_days), series.index[0]


def run_book(inp: Inputs, spec: dict[str, Any], pcfg: dict[str, Any], rcfg: dict[str, Any], mcfg: dict[str, Any],
             ic_table: pd.DataFrame, costs: CostModel, guard: pd.Timestamp) -> tuple[BacktestResult, MarketData]:
    """One frozen book, continuously from the first out-of-sample date."""
    signal = pcfg["signals"][spec["signal"]]
    h, lag = signal["horizon"], pcfg["timing"]["lag"]
    data, start = market_data(inp, spec["signal"], h, lag, pcfg["risk_model"]["window"], pd.DatetimeIndex([]))
    if spec["liquid_fraction"] != 1:
        data = replace(data, score=liquid_scores(data.score, data.adv, spec["liquid_fraction"]))
    ics = fold_ics(inp, signal, h, lag, mcfg, ic_table, rcfg["min_names_per_date"],
                   pcfg["constructions"]["optimised"]["alpha"]["ic_floor"])
    build = builder_for(spec["construction"], pcfg, rcfg, h, ics, data.dates, tuple(spec["lambdas"]))
    return run_backtest(data, build, costs, h, lag, h, guard, start, record_weights=False), data


def book_row(name: str, spec: dict[str, Any], total: pd.DataFrame, cfg: dict[str, Any], base_y: float, base_borrow: float,
             horizon: int, trading_days: int) -> dict[str, Any]:
    stats = performance(total, total.index[0], total.index[-1], horizon, trading_days)
    observed = total[total["market_return"].notna()]
    beta = regression_beta(observed["net_pnl"].to_numpy(), observed["market_return"].to_numpy(), stats["nw_lag"])
    return {"book": name, "signal": spec["signal"], "role": spec["role"], "lambda_risk": spec["lambdas"][0],
            "lambda_tc": spec["lambdas"][1], "liquid_fraction": spec["liquid_fraction"],
            **{k: stats[k] for k in ("start", "end", "n_days", "nw_lag", *cfg["reporting"]["statistics"])},
            **beta, **breakevens(total, base_y, base_borrow)}


def holdout_tables(cfg: dict[str, Any], totals: dict[str, pd.DataFrame], base_y: float, base_borrow: float, horizon: int,
                   trading_days: int) -> dict[str, pd.DataFrame]:
    """Every result table, from the holdout rows of the three books."""
    boot = cfg["bootstrap"]
    n = len(totals["headline"])
    if any(not frame.index.equals(totals["headline"].index) for frame in totals.values()):
        raise ValueError("the books must share the holdout dates")
    index = stationary_indices(n, boot["n_resamples"], boot["mean_block_length"], boot["seed"])
    draws = {k: resampled_sharpes(frame["net_pnl"].to_numpy(), index, trading_days) for k, frame in totals.items()}
    rows, years = [], []
    for name, total in totals.items():
        row = book_row(name, cfg["books"][name], total, cfg, base_y, base_borrow, horizon, trading_days)
        rows.append({**row, **{f"net_sharpe_{k}": v for k, v in percentile_interval(draws[name], boot["confidence"]).items()}})
        if name in cfg["outcomes"]:
            rows[-1]["outcome"] = t_outcome(row["net_nw_t"], cfg["outcomes"][name]["t_threshold"], cfg["outcomes"][name]["labels"])
        for year, sub in total.groupby(total.index.year):
            years.append({"book": name, "year": year, "n_days": len(sub), "net_sharpe": sharpe(sub["net_pnl"], trading_days),
                          "gross_sharpe": sharpe(sub["gross_pnl"], trading_days), "net_return": float(sub["net_pnl"].sum()),
                          "gross_return": float(sub["gross_pnl"].sum())})
    interval = percentile_interval(draws["headline"] - draws["benchmark"], boot["confidence"])
    point = sharpe(totals["headline"]["net_pnl"], trading_days) - sharpe(totals["benchmark"]["net_pnl"], trading_days)
    spec = cfg["outcomes"]["lightgbm_minus_b0"]
    difference = pd.DataFrame([{"test": "lightgbm_minus_b0", "difference": point, **interval,
                                "outcome": interval_outcome(interval["ci_low"], interval["ci_high"], spec["labels"])}])
    return {"books": pd.DataFrame(rows), "lightgbm_minus_b0": difference, "by_year": pd.DataFrame(years)}


def model_ic(inp: Inputs, column: str, horizon: int, start: pd.Timestamp, min_names: int) -> pd.DataFrame:
    """The daily rank IC of the score over the holdout, with its NW t."""
    keep = inp.dates >= start
    ic = cross_sectional_ic(pd.DataFrame(inp.scores[column][keep], index=inp.dates[keep]),
                            pd.DataFrame(inp.labels[f"fwd_ret_{horizon}d"][keep], index=inp.dates[keep]), min_names).dropna()
    stats = newey_west_tstat(ic.to_numpy(), newey_west_lag(len(ic), horizon))
    return pd.DataFrame([{"score": column, "n_days": len(ic), "mean_ic": stats["mean"], "nw_t": stats["t"]}])


def run_opened(cfg: dict[str, Any], pcfg: dict[str, Any], rcfg: dict[str, Any], fcfg: dict[str, Any], mcfg: dict[str, Any],
               paths: Paths, identity: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Everything after the lock: refit, books, checks on pre-holdout data, then the holdout tables."""
    started = time.time()
    h, lag = cfg["model_refit"]["horizon"], pcfg["timing"]["lag"]
    td, min_names = rcfg["trading_days"], rcfg["min_names_per_date"]
    holdout_start = pd.Timestamp(rcfg["holdout_start"])
    features = list(mcfg["features"]["columns"])
    columns = [pcfg["timing"]["daily_return"], pcfg["constructions"]["optimised"]["alpha"]["vol_feature"], BETA,
               pcfg["costs"]["impact"]["adv_feature"], f"fwd_ret_{h}d", f"fwd_ret_{h}d_xs", f"fwd_terminal_{h}d", "fwd_ret_1d", *features]
    panel = load_holdout_panel(list(dict.fromkeys(columns)), paths.panel, paths.lock)
    dates = panel.index.get_level_values("date").unique().sort_values()
    if str(dates[-1].date()) != cfg["window"]["last_panel_date"]:
        raise ValueError("the panel doesn't end on window.last_panel_date")
    cutoff = research_cutoff(paths.panel, rcfg, fcfg)
    last_score = last_score_date(dates, lag, h)
    signals = [cfg["books"]["headline"]["signal"], cfg["books"]["benchmark"]["signal"]]

    data = Data(panel, features, [h])
    recorded = pd.read_csv(paths.model_hyperparameters)
    oof = pd.read_parquet(paths.predictions, columns=signals + ["fold_id"], filters=[("date", "<=", cutoff)])
    later, chosen, checks = refit_scores(data, cfg, mcfg, recorded, oof, cutoff, last_score, lag, min_names)
    print(f"refit done ({time.time() - started:.0f}s)")
    preds = pd.concat([oof[signals], later[signals]]).sort_index()
    del data

    keep = [c for c in panel.columns if c not in (f"fwd_ret_{h}d_xs", "fwd_ret_1d", f"fwd_terminal_{h}d")]
    inp = Inputs(panel[keep], preds, pcfg, features)
    del panel
    ic_table = pd.concat([recorded, chosen], ignore_index=True)
    costs = CostModel.from_config(pcfg, td)
    guard = dates[-1] + pd.Timedelta(days=1)
    totals = {}
    for name, spec in cfg["books"].items():
        res, _ = run_book(inp, spec, pcfg, rcfg, mcfg, ic_table, costs, guard)
        totals[name] = res.total
        target, first = recorded_sharpe(paths, cfg, name, cutoff, td)
        new = sharpe(res.total["net_pnl"].loc[first:cutoff], td)
        checks.append({"check": "recorded_net_sharpe", "book": name, "value": abs(new - target),
                       "limit": cfg["checks"]["recorded_series_sharpe_tolerance"]})
        print(f"{name} done ({time.time() - started:.0f}s)")

    table = pd.DataFrame(checks).assign(passed=lambda d: d["value"] <= d["limit"])
    paths.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths.out / "checks.csv", index=False)
    if not table["passed"].all():
        print(table[~table["passed"]].to_string(index=False))
        raise SystemExit("a pre-registered check failed: no holdout statistic was computed or written")

    frames = {k: v.loc[holdout_start:] for k, v in totals.items()}
    if min(f.index[0] for f in frames.values()) < holdout_start:
        raise AssertionError("a pre-holdout row reached the holdout tables")
    base_y, base_borrow = float(pcfg["costs"]["impact"]["y"]), float(pcfg["costs"]["borrow_bps_per_year"])
    tables = holdout_tables(cfg, frames, base_y, base_borrow, h, td)
    tables["model_ic"] = model_ic(inp, cfg["books"]["headline"]["signal"], h, holdout_start, min_names)
    tables["refit_hyperparameters"] = chosen
    for name, frame in tables.items():
        frame.to_csv(paths.out / f"{name}.csv", index=False)
    paths.data_out.mkdir(parents=True, exist_ok=True)
    later.to_parquet(paths.data_out / "predictions.parquet")
    for name, frame in totals.items():
        frame.reset_index(names="date").to_parquet(paths.data_out / f"{name}.parquet")
    meta = {**identity, "holdout_start": rcfg["holdout_start"], "last_research_date": str(cutoff.date()),
            "last_score_date": str(last_score.date()), "last_engine_date": str(dates[-1].date()),
            "runtime_seconds": round(time.time() - started, 1)}
    (paths.out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(tables["books"][["book", "net_sharpe", "net_nw_t", "outcome"]].to_string(index=False))
    print(tables["lightgbm_minus_b0"].to_string(index=False))
    return {"checks": table, **tables}


def run(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--open-holdout", action="store_true", help="read the holdout, once")
    args = parser.parse_args(argv)
    if not args.open_holdout:
        raise SystemExit("the holdout stays closed without --open-holdout")

    cfg = load_holdout_config(HOLDOUT_CONFIG_PATH)
    fcfg = load_config(REPO_ROOT / cfg["features_config"])
    rcfg = load_research_config(REPO_ROOT / cfg["research_config"], fcfg)
    mcfg = load_models_config(REPO_ROOT / cfg["models_config"], rcfg)
    pcfg = load_portfolio_config(REPO_ROOT / cfg["portfolio_config"], fcfg, rcfg, mcfg)
    panel_file = panel_path(fcfg)
    identity = provenance(HOLDOUT_CONFIG_PATH, REPO_ROOT / cfg["features_config"], panel_file, False)
    if identity["code_dirty"]:
        raise SystemExit("src/ or configs/ has uncommitted changes: commit them before opening the holdout")
    if identity["panel_sha256"] != cfg["pinned"]["panel_sha256"]:
        raise SystemExit("the panel has changed since the design was committed")

    inputs, outputs = cfg["inputs"], cfg["outputs"]
    paths = Paths(panel_file, REPO_ROOT / inputs["predictions"], REPO_ROOT / inputs["model_hyperparameters"],
                  str(REPO_ROOT / inputs["recorded_daily_returns"]), REPO_ROOT / inputs["recorded_candidate_returns"],
                  REPO_ROOT / outputs["reports_dir"], REPO_ROOT / outputs["data_dir"], REPO_ROOT / cfg["one_shot"]["lock_file"])
    open_lock(paths.lock, identity)
    return run_opened(cfg, pcfg, rcfg, fcfg, mcfg, paths, identity)


if __name__ == "__main__":
    run()
