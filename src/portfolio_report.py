"""Portfolio results: reconciliation with the modelling spread and the tidy result tables.

Usage (from the repository root):
    python -m src.portfolio_report --reconcile                # -> reports/portfolio/reconciliation.csv
    python -m src.portfolio_report --tables RUN_DIR [RUN_DIR ...] --data DATA_DIR [DATA_DIR ...]

``--reconcile`` runs ``decile_ls`` on ``B0_reversal_h5`` with zero costs, one tranche and offset 0
and compares it with the raw h = 5 spread in ``reports/models/decile_spreads.csv``; it also checks
that gross equals net at zero costs for every construction. ``--tables`` merges the outputs of
``src.run_portfolio`` runs (one per signal) into the result tables in ``reports/portfolio/``.
Nothing here changes a result: it only re-reads the runner's outputs or reruns the engine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import MarketData, run_backtest
from src.costs import CostModel
from src.diagnostics import spread_stats
from src.metrics import newey_west_lag, newey_west_tstat, sharpe
from src.models import load_models_config
from src.portfolio import (
    OptimiserSpec,
    decile_builder,
    load_portfolio_config,
    optimised_builder,
    volscaled_builder,
)
from src.run_portfolio import Inputs, calendar_extension, market_data
from src.signals import (
    load_research_config,
    load_research_panel,
    panel_path,
    quantile_buckets,
    quantile_returns,
    research_cutoff,
)
from src.utils import REPO_ROOT, load_config

OUT_DIR = REPO_ROOT / "reports" / "portfolio"
MODEL_SPREADS = REPO_ROOT / "reports" / "models" / "decile_spreads.csv"
RECONCILE_SIGNAL = "B0_reversal_h5"
ORDER = ("decile_ls", "neutral_volscaled", "optimised")


# --------------------------------------------------------------------------- reconciliation


def load_signal(column: str) -> tuple[MarketData, int, dict[str, Any], dict[str, Any], Inputs]:
    fcfg = load_config()
    rcfg = load_research_config(features_cfg=fcfg)
    mcfg = load_models_config(rcfg=rcfg)
    pcfg = load_portfolio_config(fcfg=fcfg, rcfg=rcfg, mcfg=mcfg)
    panel_file = panel_path(fcfg)
    cutoff = research_cutoff(panel_file, rcfg, fcfg)
    h, lag = pcfg["signals"][column]["horizon"], pcfg["timing"]["lag"]
    con, costs = pcfg["constructions"], pcfg["costs"]
    cols = [pcfg["timing"]["daily_return"], con["optimised"]["alpha"]["vol_feature"],
            con["optimised"]["constraints"]["beta_feature"], costs["impact"]["adv_feature"], f"fwd_ret_{h}d"]
    panel = load_research_panel(cols, panel_file, rcfg, fcfg, cutoff)
    preds = pd.read_parquet(REPO_ROOT / pcfg["predictions"], columns=[column, "fold_id"],
                            filters=[("date", "<=", cutoff)])
    inp = Inputs(panel, preds, pcfg, [])
    extension = calendar_extension(panel_file, cutoff, lag + h, rcfg["holdout_start"])
    data, start = market_data(inp, column, h, lag, pcfg["risk_model"]["window"], extension)
    return data, start, pcfg, rcfg, inp


def reconcile(out: Path = OUT_DIR, optimised_days: int = 60) -> pd.DataFrame:
    """Engine vs label-based spread, plus gross == net at zero costs for every construction."""
    data, start, pcfg, rcfg, inp = load_signal(RECONCILE_SIGNAL)
    h, lag, q, min_names, td = 5, pcfg["timing"]["lag"], rcfg["quantiles"], rcfg["min_names_per_date"], rcfg["trading_days"]
    holdout = rcfg["holdout_start"]
    zero = CostModel.from_config(pcfg, td).scenario(multiplier=0.0)
    res = run_backtest(data, decile_builder(q, min_names, 2.0), zero, h, lag, 1, holdout, start)

    # label-based spread as in src.model_evaluation.decile_evaluation (OOS dates, every h-th, offset 0)
    oos = np.arange(start, len(inp.dates) - inp.first_oos + start)
    score = pd.DataFrame(data.score[oos], index=data.dates[oos])
    label = pd.DataFrame(data.label[oos], index=data.dates[oos])
    pos = np.arange(0, len(oos), h)
    buckets = quantile_buckets(score, q, min_names).iloc[pos]
    deciles = quantile_returns(buckets, label.iloc[pos], q)
    label_spread = (deciles[q] - deciles[1]).to_numpy()
    gross = res.total["gross_pnl"].to_numpy()
    engine_spread = np.array([gross[start + p + lag + 1: start + p + lag + h + 1].sum() for p in pos])
    # names in a leg whose label is missing (they earn only their observed daily returns in the engine)
    b = buckets.to_numpy()
    legs = (b == 0) | (b == q - 1)
    missing = (legs & np.isnan(label.iloc[pos].to_numpy())).sum(axis=1)

    model = pd.read_csv(MODEL_SPREADS).query("column == @RECONCILE_SIGNAL and spread == 'raw'").iloc[0]
    rows = [
        {"source": "reports/models/decile_spreads.csv", "n_periods": int(model["n_periods"]), "mean": model["mean"],
         "vol": model["vol"], "sharpe": model["sharpe"], "nw_t": model["nw_t"]},
        {"source": "label_spread_recomputed", **spread_stats(label_spread, td / h)},
        {"source": "engine_decile_ls_zero_cost_one_tranche", **spread_stats(engine_spread, td / h)},
    ]
    diff = engine_spread - label_spread
    ok = np.isfinite(diff)
    rows.append({
        "source": "engine_minus_label_per_period", "n_periods": int(ok.sum()), "mean": float(np.nanmean(diff)),
        "vol": float(np.nanstd(diff, ddof=1)), "max_abs_diff": float(np.nanmax(np.abs(diff))),
        "periods_differing_1e-6": int((np.abs(diff[ok]) > 1e-6).sum()),
        "periods_with_missing_leg_labels": int((missing > 0).sum()),
        "max_abs_diff_without_missing_labels": float(np.nanmax(np.abs(diff[ok & (missing == 0)]))),
    })

    # gross == net at zero costs, every construction (optimised on the first ``optimised_days`` OOS days)
    builders = {
        "decile_ls": (decile_builder(q, min_names, 2.0), data),
        "neutral_volscaled": (volscaled_builder(pcfg, min_names, td), data),
    }
    short = slice(0, start + optimised_days)
    sub = MarketData(data.dates[short], data.tickers, *(getattr(data, a)[short] for a in
                                                          ("score", "ret", "label", "vol", "beta", "adv")))
    spec = OptimiserSpec.from_config(pcfg, h, pcfg["constructions"]["optimised"]["lambda_risk"][0],
                                     pcfg["constructions"]["optimised"]["lambda_tc"][0])
    builders["optimised"] = (optimised_builder(pcfg, spec, lambda s: 0.01, min_names), sub)
    for name, (build, d) in builders.items():
        r = run_backtest(d, build, zero, h, lag, h, holdout, start, record_weights=False)
        t = r.total
        rows.append({
            "source": f"zero_cost_gross_minus_net_{name}", "n_periods": len(t),
            "max_abs_diff": float((t["gross_pnl"] - t["net_pnl"]).abs().max()),
            "cost_sum": float(t[["commission", "spread", "impact", "borrow"]].to_numpy().sum()),
            "turnover_sum": float(t["turnover"].sum()),
        })
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "reconciliation.csv", index=False)
    return table


# --------------------------------------------------------------------------- result tables


def regression_beta(y: np.ndarray, x: np.ndarray, lag: int) -> dict[str, float]:
    """OLS slope of y on x with a Newey-West t for the slope."""
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    xc = x - x.mean()
    beta = float(xc @ (y - y.mean()) / (xc @ xc))
    resid = y - y.mean() - beta * xc
    score = xc * resid / (xc @ xc) * len(x)                   # influence function of the slope
    nw = newey_west_tstat(score, lag)
    return {"realised_beta": beta, "realised_beta_t": beta / (nw["se"]) if nw["se"] > 0 else np.nan}


def build_tables(run_dirs: list[Path], data_dirs: list[Path], out: Path = OUT_DIR) -> dict[str, pd.DataFrame]:
    pcfg = load_portfolio_config()
    td = load_research_config()["trading_days"]
    summary = pd.concat([pd.read_csv(d / "summary.csv") for d in run_dirs], ignore_index=True)
    fallbacks = pd.concat([pd.read_csv(d / "fallbacks.csv") for d in run_dirs], ignore_index=True).fillna(0)
    grid = pd.concat([pd.read_csv(d / "tuning" / "lambda_grid.csv") for d in run_dirs], ignore_index=True)
    chosen = pd.concat([pd.read_csv(d / "tuning" / "selected_hyperparameters.csv") for d in run_dirs], ignore_index=True)
    metas = {d.name: json.loads((d / "run_metadata.json").read_text()) for d in run_dirs}
    tune_metas = {d.name: json.loads((d / "tuning" / "run_metadata.json").read_text()) for d in run_dirs}
    daily = pd.concat([pd.read_parquet(d / "portfolio_daily_returns.parquet") for d in data_dirs], ignore_index=True)
    daily = daily[(daily["tranche"] == "total") & (daily["scenario"] == "base")]
    daily["date"] = pd.to_datetime(daily["date"])

    opt = pcfg["constructions"]["optimised"]
    lr, lt = opt["lambda_risk"], opt["lambda_tc"]
    chosen["config_sha256"] = chosen["signal"].map(lambda s: tune_metas[s]["config_sha256"])
    chosen["lambda_risk_on_grid_edge"] = chosen["lambda_risk"].isin([min(lr), max(lr)])
    chosen["lambda_tc_on_grid_edge"] = chosen["lambda_tc"].isin([min(lt), max(lt)])

    base = summary[summary["scenario"] == "base"]
    keep = ["net_sharpe", "net_nw_t", "nw_lag", "net_return_ann", "net_vol_ann", "gross_sharpe", "annual_turnover",
            "breakeven_cost_multiplier", "start", "end", "n_days"]
    head = pcfg["headline"]
    rows = []
    for signal in (head["signal"], head["benchmark"]):
        ev = base.query("signal == @signal and construction == @head['construction'] and window == 'evaluation'")
        tu = base.query("signal == @signal and construction == @head['construction'] and window == 'tuning'")
        if len(ev):
            rows.append({"signal": signal, "construction": head["construction"], "window": "evaluation",
                         **ev.iloc[0][keep].to_dict(),
                         "tuning_window_net_sharpe_in_sample_lambdas": tu.iloc[0]["net_sharpe"] if len(tu) else np.nan,
                         "tuning_window_net_nw_t_in_sample_lambdas": tu.iloc[0]["net_nw_t"] if len(tu) else np.nan})
    headline = pd.DataFrame(rows)

    ladder = base[base["window"].isin(["evaluation", "tuning"])][
        ["signal", "window", "construction", "gross_sharpe", "net_sharpe", "net_nw_t", "gross_return_ann",
         "net_return_ann", "cost_ann", "borrow_ann", "annual_turnover", "breakeven_cost_multiplier"]].copy()
    ladder["construction"] = pd.Categorical(ladder["construction"], ORDER, ordered=True)
    ladder = ladder.sort_values(["signal", "window", "construction"])
    ladder["net_return_ann_vs_previous"] = ladder.groupby(["signal", "window"], observed=True)["net_return_ann"].diff()
    ladder["net_sharpe_vs_previous"] = ladder.groupby(["signal", "window"], observed=True)["net_sharpe"].diff()
    ladder["window"] = ladder["window"].replace({"tuning": "tuning (in-sample for the lambdas)"})

    split = pd.Timestamp(pcfg["sample"]["split_date"])
    cost_rows, exp_rows, year_rows = [], [], []
    for (signal, construction), frame in daily.groupby(["signal", "construction"], sort=False):
        frame = frame.set_index("date").sort_index()
        h = int(frame["horizon"].iloc[0])
        for window, sub in (("evaluation", frame[frame.index >= split]), ("tuning", frame[frame.index < split])):
            first = summary.query("signal == @signal and construction == @construction and scenario == 'base' "
                                  "and window == @window")
            lo = pd.Timestamp(first["start"].iloc[0]) if len(first) and pd.notna(first["start"].iloc[0]) else sub.index[0]
            sub = sub[sub.index >= lo]
            label = "tuning (in-sample for the lambdas)" if window == "tuning" else window
            bp = {f"{c}_bp_per_year": sub[c].mean() * td * 1e4 for c in ("commission", "spread", "impact", "borrow")}
            total_cost = sum(bp.values())
            cost_rows.append({"signal": signal, "construction": construction, "window": label, **bp,
                              "total_cost_bp_per_year": total_cost,
                              "gross_bp_per_year": sub["gross_pnl"].mean() * td * 1e4,
                              "net_bp_per_year": sub["net_pnl"].mean() * td * 1e4,
                              "cost_per_unit_turnover_bp": (sub["cost"].sum() / sub["turnover"].sum() * 1e4 / 2)
                              if sub["turnover"].sum() > 0 else np.nan})
            reg = regression_beta(sub["net_pnl"].to_numpy(), sub["market_return"].to_numpy(),
                                  newey_west_lag(len(sub), h))
            fb = fallbacks.query("signal == @signal and construction == @construction and scenario == 'base'")
            fb_cols = [c for c in ("hold", "flat", "partial") if c in fb]
            exp_rows.append({"signal": signal, "construction": construction, "window": label, **reg,
                             "mean_gross_exposure": sub["gross_exposure"].mean(),
                             "mean_net_exposure": sub["net_exposure"].mean(),
                             "mean_abs_net_exposure": sub["net_exposure"].abs().mean(),
                             "mean_beta_ex_ante": sub["beta_ex_ante"].mean(),
                             "mean_abs_beta_ex_ante": sub["beta_ex_ante"].abs().mean(),
                             "rebalances_full_sample": int(fb["rebalances"].iloc[0]) if len(fb) else np.nan,
                             **{f"fallback_{c}_full_sample": int(fb[c].iloc[0]) for c in fb_cols}})
        start = summary.query("signal == @signal and construction == @construction and scenario == 'base' "
                              "and window == 'full'")["start"].iloc[0]
        frame = frame[frame.index >= pd.Timestamp(start)]
        for year, sub in frame.groupby(frame.index.year):
            year_rows.append({"signal": signal, "construction": construction, "year": year, "n_days": len(sub),
                              "net_sharpe": sharpe(sub["net_pnl"], td), "gross_sharpe": sharpe(sub["gross_pnl"], td),
                              "net_return": sub["net_pnl"].sum(), "gross_return": sub["gross_pnl"].sum(),
                              "window": "tuning (in-sample for the lambdas)" if year < split.year else "evaluation"})

    scenarios = summary[summary["window"].isin(["evaluation", "tuning"])][
        ["signal", "construction", "window", "scenario", "cost_multiplier", "aum_usd", "net_sharpe", "net_nw_t",
         "net_return_ann", "gross_return_ann", "cost_ann", "borrow_ann", "annual_turnover"]].copy()
    scenarios["window"] = scenarios["window"].replace({"tuning": "tuning (in-sample for the lambdas)"})

    tables = {
        "headline": headline,
        "ladder": ladder,
        "cost_breakdown": pd.DataFrame(cost_rows),
        "scenarios": scenarios,
        "exposures": pd.DataFrame(exp_rows),
        "by_year": pd.DataFrame(year_rows),
        "hyperparameters": chosen,
        "lambda_grid": grid,
        "summary": summary,
        "fallbacks": fallbacks,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    combined = {
        "runs": {k: {key: v.get(key) for key in ("config_commit", "config_sha256", "code_commit", "code_dirty",
                                                  "predictions_sha256", "panel_sha256", "selected_lambdas",
                                                  "evaluation_start", "last_engine_date", "runtime_seconds")}
                 for k, v in metas.items()},
        "tuning_runs": {k: {key: v.get(key) for key in ("config_sha256", "code_commit", "tuning_end", "runtime_seconds")}
                        for k, v in tune_metas.items()},
        "config_sha256": sorted({m["config_sha256"] for m in metas.values()}),
        "holdout_start": next(iter(metas.values()))["holdout_start"],
        "last_research_date": next(iter(metas.values()))["last_research_date"],
    }
    if len(combined["config_sha256"]) != 1 or {m["config_sha256"] for m in tune_metas.values()} != set(combined["config_sha256"]):
        raise AssertionError("runs were made with different portfolio configs")
    (out / "run_metadata.json").write_text(json.dumps(combined, indent=2, default=str) + "\n")
    return tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--tables", nargs="*", type=Path, default=None, help="run directories (one per signal)")
    parser.add_argument("--data", nargs="*", type=Path, default=None, help="data directories (same order)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.reconcile:
        print(reconcile(args.out).to_string(index=False))
    if args.tables:
        tables = build_tables(args.tables, args.data, args.out)
        print(tables["headline"].to_string(index=False))


if __name__ == "__main__":
    main()
