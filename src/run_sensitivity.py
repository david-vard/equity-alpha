"""Run the pre-registered sensitivity analysis of the recorded portfolio books.

Usage (from the repository root):
    python -m src.run_sensitivity        # -> reports/sensitivity/

Everything is computed from the outputs of ``src.run_portfolio``: no backtest is rerun and the
optimiser is not re-solved. The checks in ``configs/sensitivity.yaml`` run first, and if one fails
only ``checks.csv`` is written. A design config that is not committed is refused (see
``src.utils.provenance``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.metrics import sharpe
from src.portfolio import load_portfolio_config
from src.sensitivity import (
    SENSITIVITY_CONFIG_PATH,
    breakevens,
    load_sensitivity_config,
    offset_dispersion,
    offset_rows,
    percentile_interval,
    reprice,
    repricing_rows,
    resampled_sharpes,
    stationary_indices,
)
from src.signals import load_research_config, panel_path
from src.utils import REPO_ROOT, file_sha256, load_config, provenance

TUNING_LABEL = "tuning (in-sample for the lambdas)"
TOTAL = "total"
Book = tuple[str, str]
Windows = dict[Book, dict[str, tuple[pd.Timestamp, pd.Timestamp]]]


def load_books(cfg: dict[str, Any], pcfg: dict[str, Any], data_root: Path) -> dict[Book, pd.DataFrame]:
    """Daily base-case rows of every book: the total row and one row per tranche."""
    books = {}
    for signal in pcfg["signals"]:
        frame = pd.read_parquet(data_root / cfg["book_data"]["daily_returns"].format(signal=signal),
                                filters=[("scenario", "==", cfg["book_data"]["scenario"])])
        frame["date"] = pd.to_datetime(frame["date"])
        for construction, rows in frame.groupby("construction", sort=False):
            books[(signal, construction)] = rows.set_index("date").sort_index(kind="stable")
    return books


def book_windows(cfg: dict[str, Any], books: dict[Book, pd.DataFrame], summary: pd.DataFrame) -> Windows:
    """Each book's window dates, exactly as reported in ``reports/portfolio/summary.csv``."""
    out: Windows = {}
    for signal, construction in books:
        rows = summary[(summary["signal"] == signal) & (summary["construction"] == construction)
                       & (summary["scenario"] == cfg["book_data"]["scenario"])].set_index("window")
        out[(signal, construction)] = {
            w: (pd.Timestamp(rows.loc[w, "start"]), pd.Timestamp(rows.loc[w, "end"])) for w in cfg["windows"]}
    return out


def total_row(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["tranche"] == TOTAL]


def run_checks(cfg: dict[str, Any], base_y: float, base_borrow: float, books: dict[Book, pd.DataFrame], windows: Windows,
               summary: pd.DataFrame, holdout_start: pd.Timestamp, trading_days: int) -> pd.DataFrame:
    """Every pre-registered check, one row each; nothing else runs until they all pass."""
    tests = cfg["checks"]
    pair = cfg["bootstrap"]["paired_differences"]
    rows: list[dict[str, Any]] = []

    def add(check: str, book: Any, window: str, value: float, limit: float) -> None:
        rows.append({"check": check, "book": book, "window": window, "value": float(value), "limit": limit,
                     "passed": bool(value <= limit)})

    for book, frame in books.items():
        name = "/".join(book)
        total = total_row(frame)
        add("last_date_before_holdout", name, "all", 0.0 if frame.index.max() < holdout_start else 1.0, 0.0)
        parts = frame[frame["tranche"] != TOTAL].groupby(level="date")[["gross_pnl", "net_pnl"]].sum()
        add("tranches_sum", name, "all", (parts - total[["gross_pnl", "net_pnl"]]).abs().max().max(),
            tests["tranches_sum_tolerance"])
        add("reprice_base", name, "all", (reprice(total, base_y, base_borrow, base_y, base_borrow) - total["net_pnl"]).abs().max(),
            tests["reprice_base_tolerance"])
        recorded = summary[(summary["signal"] == book[0]) & (summary["construction"] == book[1])
                           & (summary["scenario"] == cfg["book_data"]["scenario"])].set_index("window")
        for window, (lo, hi) in windows[book].items():
            sub = total.loc[lo:hi]
            gap = max(abs(sharpe(sub[pnl], trading_days) - recorded.loc[window, column])
                      for pnl, column in (("gross_pnl", "gross_sharpe"), ("net_pnl", "net_sharpe")))
            add("reproduce_sharpes", name, window, gap, tests["reproduce_sharpes_tolerance"])
    for construction in cfg_constructions(books):
        for window in cfg["windows"]:
            a, b = (pair["minuend"], construction), (pair["subtrahend"], construction)
            same = total_row(books[a]).loc[slice(*windows[a][window])].index.equals(
                total_row(books[b]).loc[slice(*windows[b][window])].index)
            add("paired_dates", f"{pair['minuend']} - {pair['subtrahend']}, {construction}", window, 0.0 if same else 1.0, 0.0)
    return pd.DataFrame(rows)


def cfg_constructions(books: dict[Book, pd.DataFrame]) -> list[str]:
    return list(dict.fromkeys(construction for _, construction in books))


def bootstrap_tables(cfg: dict[str, Any], books: dict[Book, pd.DataFrame], windows: Windows,
                     trading_days: int) -> dict[str, pd.DataFrame]:
    boot = cfg["bootstrap"]
    pair = boot["paired_differences"]
    label = {"evaluation": "evaluation", "tuning": TUNING_LABEL}
    intervals, differences = [], []
    groups: dict[tuple[str, int], list[Book]] = {}
    for window in cfg["windows"]:
        for book in books:
            n = len(total_row(books[book]).loc[slice(*windows[book][window])])
            groups.setdefault((window, n), []).append(book)

    for (window, n), members in groups.items():
        for length in boot["mean_block_lengths"]:
            index = stationary_indices(n, boot["n_resamples"], length, boot["seed"])
            for pnl in boot["series"]:
                draws = {}
                for book in members:
                    x = total_row(books[book]).loc[slice(*windows[book][window])][pnl]
                    draws[book] = resampled_sharpes(x.to_numpy(), index, trading_days)
                    intervals.append({"signal": book[0], "construction": book[1], "series": pnl, "window": label[window],
                                      "mean_block_length": length, "n_days": n, "sharpe": sharpe(x, trading_days),
                                      **percentile_interval(draws[book], boot["confidence"])})
                for construction in cfg_constructions(books):
                    a, b = (pair["minuend"], construction), (pair["subtrahend"], construction)
                    if a in draws and b in draws:
                        point = intervals_point(intervals, a, pnl, label[window], length) - intervals_point(intervals, b, pnl, label[window], length)
                        differences.append({"minuend": a[0], "subtrahend": b[0], "construction": construction, "series": pnl,
                                            "window": label[window], "mean_block_length": length, "n_days": n, "difference": point,
                                            **percentile_interval(draws[a] - draws[b], boot["confidence"])})
    intervals_frame, differences_frame = pd.DataFrame(intervals), pd.DataFrame(differences)

    primary = cfg["primary_intervals"]
    length = boot["mean_block_lengths"][0]
    head = primary["headline"]
    one = intervals_frame[(intervals_frame["signal"] == head["signal"]) & (intervals_frame["construction"] == head["construction"])
                          & (intervals_frame["series"] == head["series"]) & (intervals_frame["window"] == head["window"])
                          & (intervals_frame["mean_block_length"] == length)].iloc[0]
    gap = primary["lightgbm_minus_b0"]
    two = differences_frame[(differences_frame["construction"] == gap["construction"]) & (differences_frame["series"] == gap["series"])
                            & (differences_frame["window"] == gap["window"]) & (differences_frame["mean_block_length"] == length)].iloc[0]
    keep = ["ci_low", "ci_high", "resample_sd", "share_above_zero", "contains_zero", "n_days", "mean_block_length"]
    primary_frame = pd.DataFrame([
        {"interval": "headline", "estimate": one["sharpe"], **one[keep].to_dict()},
        {"interval": "lightgbm_minus_b0", "estimate": two["difference"], **two[keep].to_dict()}])
    return {"bootstrap_intervals": intervals_frame, "paired_differences": differences_frame, "primary_intervals": primary_frame}


def intervals_point(rows: list[dict[str, Any]], book: Book, pnl: str, window: str, length: int) -> float:
    return next(r["sharpe"] for r in rows if (r["signal"], r["construction"]) == book and r["series"] == pnl
                and r["window"] == window and r["mean_block_length"] == length)


def repricing_tables(cfg: dict[str, Any], base_y: float, base_borrow: float, books: dict[Book, pd.DataFrame],
                     windows: Windows, trading_days: int) -> dict[str, pd.DataFrame]:
    label = {"evaluation": "evaluation", "tuning": TUNING_LABEL}
    grid, even = [], []
    for book, frame in books.items():
        total = total_row(frame)
        horizon = int(total["horizon"].iloc[0])
        for window, (lo, hi) in windows[book].items():
            sub = total.loc[lo:hi]
            ids = {"signal": book[0], "construction": book[1], "window": label[window]}
            grid += repricing_rows(sub, cfg, base_y, base_borrow, horizon, trading_days, ids)
            even.append({**ids, **breakevens(sub, base_y, base_borrow)})
    return {"repricing_grid": pd.DataFrame(grid), "repricing_breakeven": pd.DataFrame(even)}


def offset_tables(cfg: dict[str, Any], books: dict[Book, pd.DataFrame], windows: Windows,
                  trading_days: int) -> dict[str, pd.DataFrame]:
    label = {"evaluation": "evaluation", "tuning": TUNING_LABEL}
    series = cfg["offsets"]["series"]
    rows, totals = [], []
    for book, frame in books.items():
        for window, (lo, hi) in windows[book].items():
            sub = frame.loc[lo:hi]
            ids = {"signal": book[0], "construction": book[1], "window": label[window]}
            rows += offset_rows(sub[sub["tranche"] != TOTAL], trading_days, ids, series)
            total = total_row(sub)
            for pnl in series:
                totals.append({**ids, "series": pnl, "sharpe": sharpe(total[pnl], trading_days),
                               "return_ann": float(total[pnl].mean() * trading_days),
                               "turnover_ann": float(total["turnover"].mean() * trading_days)})
    offsets, keys = pd.DataFrame(rows), ["signal", "construction", "window", "series"]
    return {"offsets": offsets, "offset_dispersion": offset_dispersion(offsets, pd.DataFrame(totals), keys)}


def run(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=SENSITIVITY_CONFIG_PATH)
    parser.add_argument("--allow-dirty-config", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_sensitivity_config(args.config)
    fcfg = load_config(REPO_ROOT / cfg["features_config"])
    rcfg = load_research_config(REPO_ROOT / cfg["research_config"], fcfg)
    pcfg = load_portfolio_config(REPO_ROOT / cfg["portfolio_config"], fcfg, rcfg)
    meta = provenance(args.config, REPO_ROOT / cfg["features_config"], panel_path(fcfg), args.allow_dirty_config)
    portfolio_meta = json.loads((REPO_ROOT / cfg["book_data"]["portfolio_metadata"]).read_text())
    if portfolio_meta["config_sha256"] != [file_sha256(REPO_ROOT / cfg["portfolio_config"])]:
        raise SystemExit("the recorded books were not made with the current configs/portfolio.yaml")

    books = load_books(cfg, pcfg, REPO_ROOT)
    summary = pd.read_csv(REPO_ROOT / cfg["book_data"]["summary_table"])
    windows = book_windows(cfg, books, summary)
    base_y, base_borrow = float(pcfg["costs"]["impact"]["y"]), float(pcfg["costs"]["borrow_bps_per_year"])
    trading_days = rcfg["trading_days"]

    out = REPO_ROOT / cfg["outputs"]["reports_dir"]
    out.mkdir(parents=True, exist_ok=True)
    checks = run_checks(cfg, base_y, base_borrow, books, windows, summary, pd.Timestamp(rcfg["holdout_start"]), trading_days)
    checks.to_csv(out / "checks.csv", index=False)
    if not checks["passed"].all():
        print(checks[~checks["passed"]].to_string(index=False))
        raise SystemExit("a pre-registered check failed: no results were written")

    tables = {**bootstrap_tables(cfg, books, windows, trading_days),
              **repricing_tables(cfg, base_y, base_borrow, books, windows, trading_days),
              **offset_tables(cfg, books, windows, trading_days)}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    meta.update({"holdout_start": rcfg["holdout_start"], "portfolio_config_sha256": file_sha256(REPO_ROOT / cfg["portfolio_config"]),
                 "book_files_sha256": {path: file_sha256(REPO_ROOT / path) for path in sorted(
                     cfg["book_data"]["daily_returns"].format(signal=s) for s in pcfg["signals"])}})
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(tables["primary_intervals"].T.to_string())
    return {"checks": checks, **tables}


if __name__ == "__main__":
    run()
