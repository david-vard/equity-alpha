"""Run the pre-registered risk and attribution analysis of the recorded portfolio books.

Usage (from the repository root):
    python -m src.run_attribution --download   # factor files -> data/raw/factors/, with a manifest
    python -m src.run_attribution --run        # -> reports/attribution/

``--download`` keeps a file that already exists, so one vintage backs every run. ``--run`` never
touches the network. It stops before writing results if a check in ``configs/attribution.yaml``
fails, and it refuses a design config that is not committed (see ``src.utils.provenance``). No
backtest is rerun: books are read from the outputs of ``src.run_portfolio``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.attribution import (
    ATTRIBUTION_CONFIG_PATH,
    conditional_beta,
    drawdown_episodes,
    factor_tables,
    load_attribution_config,
    load_factors,
    moments,
    nested_models,
    ols_newey_west,
    primary_outcome,
    source_filename,
    up_down_beta,
    var_es,
    weights_exposure,
    worst_periods,
)
from src.features import rolling_beta
from src.metrics import newey_west_lag, sharpe
from src.signals import load_research_config, load_research_panel, panel_path
from src.utils import REPO_ROOT, file_sha256, load_config, provenance

MANIFEST = "manifest.json"
TUNING_LABEL = "tuning (in-sample for the lambdas)"
BETA_COLUMN = "beta_252d"
OWN_REVERSAL = "own_reversal_book"
PROXIES = {"market_return": "equal_weighted", "mkt_rf": "value_weighted"}


def download(cfg: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    """Fetch each factor file once and record its source, hash and download time."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / MANIFEST
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for source in cfg["factors"]["sources"].values():
        name = source_filename(source)
        if (raw_dir / name).exists():
            continue
        response = requests.get(source["url"], timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        (raw_dir / name).write_bytes(response.content)
        manifest[name] = {"url": source["url"], "bytes": len(response.content),
                          "sha256": file_sha256(raw_dir / name),
                          "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verified_manifest(cfg: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    manifest = json.loads((raw_dir / MANIFEST).read_text())
    for source in cfg["factors"]["sources"].values():
        name = source_filename(source)
        if name not in manifest or manifest[name]["sha256"] != file_sha256(raw_dir / name):
            raise SystemExit(f"{name} does not match {raw_dir / MANIFEST}")
    return manifest


def load_books(cfg: dict[str, Any], data_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Daily base-case rows and rebalance weights of every declared book."""
    book_data = cfg["book_data"]
    daily, weights = {}, {}
    for name, spec in cfg["books"].items():
        filters = [("construction", "==", spec["construction"])]
        frame = pd.read_parquet(
            data_root / book_data["daily_returns"].format(signal=spec["signal"]),
            filters=filters + [("tranche", "==", book_data["tranche"]), ("scenario", "==", book_data["scenario"])])
        frame["date"] = pd.to_datetime(frame["date"])
        daily[name] = frame.set_index("date").sort_index()
        held = pd.read_parquet(data_root / book_data["weights"].format(signal=spec["signal"]), filters=filters,
                               columns=["score_date", "trade_date", "tranche", "ticker", "weight"])
        held["score_date"], held["trade_date"] = pd.to_datetime(held["score_date"]), pd.to_datetime(held["trade_date"])
        weights[name] = held
    return daily, weights


def book_windows(cfg: dict[str, Any], summary: pd.DataFrame) -> dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp]]]:
    """Each book's window dates, exactly as reported in ``reports/portfolio/summary.csv``."""
    out = {}
    for name, spec in cfg["books"].items():
        rows = summary[(summary["signal"] == spec["signal"]) & (summary["construction"] == spec["construction"])
                       & (summary["scenario"] == cfg["book_data"]["scenario"])].set_index("window")
        out[name] = {w: (pd.Timestamp(rows.loc[w, "start"]), pd.Timestamp(rows.loc[w, "end"])) for w in cfg["windows"]}
    return out


def beta_frames(cfg: dict[str, Any], panel_file: Path, rcfg: dict[str, Any], fcfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Stock betas known at each date: the panel's ``beta_252d`` and the short-window beta."""
    panel = load_research_panel(["ret_1d", BETA_COLUMN], panel_file, rcfg, fcfg)
    returns = panel["ret_1d"].unstack("ticker").astype("float64")
    stale = cfg["beta_analysis"]["explanations"]["stale_beta"]
    short = rolling_beta(returns, returns.mean(axis=1), stale["short_beta_window"], stale["min_obs_frac"])
    return {"exposure_short_beta": short, "exposure_252d_weights": panel[BETA_COLUMN].unstack("ticker").astype("float64")}


def analyse(cfg: dict[str, Any], daily: dict[str, pd.DataFrame], exposures: dict[str, dict[str, pd.Series]],
            factors: pd.DataFrame, windows: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp]]],
            exposures_table: pd.DataFrame, trading_days: int) -> dict[str, pd.DataFrame]:
    """Every result table, plus ``checks``; pure, so it can run on synthetic books."""
    series = [cfg["series"]["primary"], cfg["series"]["secondary"]]
    models = nested_models(cfg)
    label = {"evaluation": "evaluation", "tuning": TUNING_LABEL}
    tests = cfg["checks"]
    rows: dict[str, list[dict]] = {k: [] for k in (
        "checks", "factor_models", "factor_loadings", "risk_statistics", "var_es", "drawdowns",
        "beta_comparison", "beta_conditional", "beta_up_down")}

    all_dates = pd.DatetimeIndex(sorted({d for name in daily for lo, hi in windows[name].values()
                                         for d in daily[name].loc[lo:hi].index}))
    covered = factors.reindex(all_dates)
    missing = int(covered.isna().any(axis=1).sum())
    rows["checks"].append({"check": "coverage", "book": "all", "window": "all", "value": missing, "limit": 0,
                           "passed": missing == 0})
    if missing:
        return {"checks": pd.DataFrame(rows["checks"])}

    market = next(iter(daily.values()))["market_return"].reindex(all_dates)
    corr = {k: covered["mkt_rf"].corr(market.shift(k)) for k in (-1, 0, 1)}
    rows["checks"].append({"check": "date_alignment", "book": "all", "window": "all", "value": corr[0],
                           "limit": max(corr[-1], corr[1]), "passed": bool(corr[0] > max(corr[-1], corr[1]))})

    for name, spec in cfg["books"].items():
        h = int(daily[name]["horizon"].iloc[0])
        for window, (lo, hi) in windows[name].items():
            frame = daily[name].loc[lo:hi]
            fac = factors.loc[frame.index]
            lag = newey_west_lag(len(frame), h)
            ids = {"book": name, "signal": spec["signal"], "construction": spec["construction"], "window": label[window]}
            proxy = pd.DataFrame({"market_return": frame["market_return"], "mkt_rf": fac["mkt_rf"]})

            published = exposures_table[(exposures_table["signal"] == spec["signal"])
                                        & (exposures_table["construction"] == spec["construction"])
                                        & (exposures_table["window"] == label[window])]
            # The universe return does not exist on the days after the research cutoff, when open tranches
            # run off; exposures.csv leaves those days out, and so do the regressions on that return.
            observed = frame[frame["market_return"].notna()]
            fit = ols_newey_west(observed["net_pnl"].to_numpy(), observed[["market_return"]].to_numpy(), ["market"], lag)
            gap = max(abs(fit["coef"]["market"] - published["realised_beta"].iloc[0]),
                      abs(fit["t"]["market"] - published["realised_beta_t"].iloc[0]))
            rows["checks"].append({"check": "reproduce_exposures", "book": name, "window": label[window], "value": gap,
                                   "limit": tests["reproduce_exposures_tolerance"],
                                   "passed": bool(gap <= tests["reproduce_exposures_tolerance"])})

            for pnl in series:
                y = frame[pnl]
                sid = {**ids, "series": pnl}
                model_rows, loading_rows = factor_tables(y, fac, models, h, trading_days, sid)
                rows["factor_models"] += model_rows
                rows["factor_loadings"] += loading_rows
                gap = max(abs(r["decomposition_gap"]) for r in model_rows)
                rows["checks"].append({"check": "decomposition", "book": name, "window": f"{label[window]}, {pnl}",
                                       "value": gap, "limit": tests["decomposition_tolerance"],
                                       "passed": bool(gap <= tests["decomposition_tolerance"])})

                vol = float(y.std(ddof=1))
                rows["risk_statistics"].append({
                    **sid, "n_days": len(y), "return_ann": float(y.mean() * trading_days),
                    "vol_ann": vol * np.sqrt(trading_days), "sharpe": sharpe(y, trading_days),
                    **moments(y, h), **worst_periods(y)})
                for confidence in cfg["risk"]["var_es"]["confidence"]:
                    for days in cfg["risk"]["var_es"]["horizons_days"]:
                        loss = var_es(y, confidence, days)
                        rows["var_es"].append({**sid, "confidence": confidence, "horizon_days": days, **loss,
                                               "var_in_daily_vols": loss["var"] / vol, "es_in_daily_vols": loss["es"] / vol})
                for rank, episode in enumerate(drawdown_episodes(y, cfg["risk"]["drawdown_episodes"]), 1):
                    rows["drawdowns"].append({**sid, "rank": rank, **episode})

                for proxy_name, proxy_label in PROXIES.items():
                    m = proxy[proxy_name].dropna()
                    y_m = y.loc[m.index]
                    fit = ols_newey_west(y_m.to_numpy(), m.to_frame().to_numpy(), ["market"], lag)
                    beta, se = float(fit["coef"]["market"]), float(fit["se"]["market"])
                    low, high = beta - 1.96 * se, beta + 1.96 * se
                    ex_ante = float(frame["beta_ex_ante"].mean())
                    held = exposures[name]
                    rows["beta_comparison"].append({
                        **sid, "market_proxy": proxy_label, "realised_beta": beta, "realised_beta_t": beta / se,
                        "ci95_low": low, "ci95_high": high, "mean_beta_ex_ante": ex_ante,
                        "ci_contains_ex_ante": bool(low <= ex_ante <= high),
                        "mean_exposure_short_beta": float(held["exposure_short_beta"].loc[lo:hi].mean()),
                        "mean_exposure_252d_weights": float(held["exposure_252d_weights"].loc[lo:hi].mean())})
                    rows["beta_up_down"].append({**sid, "market_proxy": proxy_label, **up_down_beta(y_m, m, lag)})
                    for measure, exposure in (("beta_ex_ante", daily[name]["beta_ex_ante"]),
                                              ("exposure_short_beta", held["exposure_short_beta"])):
                        rows["beta_conditional"].append({
                            **sid, "market_proxy": proxy_label, "exposure": measure,
                            **conditional_beta(y_m, m, exposure.reindex(daily[name].index), lag)})

    tables = {k: pd.DataFrame(v) for k, v in rows.items()}
    loadings = tables["factor_loadings"]
    tables["beta_factor_controls"] = loadings[loadings["term"] == "mkt_rf"].reset_index(drop=True)

    test = cfg["primary_test"]
    keep = (loadings["book"] == test["book"]) & (loadings["series"] == test["series"]) & (
        loadings["window"] == label[test["window"]]) & (loadings["model"] == test["model"])
    chosen = loadings[keep].set_index("term")
    summary = tables["factor_models"]
    fitted = summary[(summary["book"] == test["book"]) & (summary["series"] == test["series"])
                     & (summary["window"] == label[test["window"]]) & (summary["model"] == test["model"])].iloc[0]
    factor = test["factor"]
    tables["primary_test"] = pd.DataFrame([{
        "book": test["book"], "series": test["series"], "window": test["window"], "model": test["model"],
        "n_days": fitted["n_days"], "nw_lag": fitted["nw_lag"], "mean_return_ann": fitted["mean_return_ann"],
        "alpha_ann": fitted["alpha_ann"], "alpha_t": fitted["alpha_t"], "reversal_beta": chosen.loc[factor, "coef"],
        "reversal_t": chosen.loc[factor, "t"], "reversal_contribution_bp_per_year": chosen.loc[factor, "contribution_bp_per_year"],
        "t_threshold": test["t_threshold"],
        "outcome": primary_outcome(chosen.loc[factor, "coef"], chosen.loc[factor, "t"], fitted["alpha_t"], test["t_threshold"]),
    }])

    span = cfg["own_reversal_spanning"]
    lo, hi = windows[span["book"]][span["window"]]
    frame = daily[span["book"]].loc[lo:hi]
    joined = factors.loc[frame.index].assign(**{OWN_REVERSAL: daily[span["regressor_book"]][span["series"]].reindex(frame.index)})
    h = int(frame["horizon"].iloc[0])
    wide = {span["model"]: models[span["model"]] + [OWN_REVERSAL]}
    ids = {"book": span["book"], "regressor_book": span["regressor_book"], "series": span["series"], "window": span["window"]}
    model_rows, loading_rows = factor_tables(frame[span["series"]], joined, wide, h, trading_days, ids)
    tables["own_reversal_spanning"] = pd.DataFrame(loading_rows).assign(
        alpha_ann=model_rows[0]["alpha_ann"], alpha_t=model_rows[0]["alpha_t"], r2=model_rows[0]["r2"],
        mean_return_ann=model_rows[0]["mean_return_ann"])
    return tables


def run(argv: list[str] | None = None) -> dict[str, pd.DataFrame] | None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ATTRIBUTION_CONFIG_PATH)
    parser.add_argument("--allow-dirty-config", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_attribution_config(args.config)
    raw_dir = REPO_ROOT / cfg["factors"]["raw_dir"]
    if args.download:
        for name, item in download(cfg, raw_dir).items():
            print(f"{name}: {item['bytes']} bytes, {item['sha256'][:12]}, {item['downloaded_at']}")
        return None

    fcfg = load_config(REPO_ROOT / cfg["features_config"])
    rcfg = load_research_config(REPO_ROOT / cfg["research_config"], fcfg)
    panel_file = panel_path(fcfg)
    meta = provenance(args.config, REPO_ROOT / cfg["features_config"], panel_file, args.allow_dirty_config)
    portfolio_meta = json.loads((REPO_ROOT / cfg["book_data"]["portfolio_metadata"]).read_text())
    if portfolio_meta["config_sha256"] != [file_sha256(REPO_ROOT / cfg["portfolio_config"])]:
        raise SystemExit("the recorded books were not made with the current configs/portfolio.yaml")

    manifest = verified_manifest(cfg, raw_dir)
    holdout = pd.Timestamp(rcfg["holdout_start"])
    factors = load_factors(cfg, raw_dir, holdout)
    print(f"factors: {factors.index.min().date()} to {factors.index.max().date()}, {len(factors)} rows")
    daily, weights = load_books(cfg, REPO_ROOT)
    reports = REPO_ROOT / Path(cfg["book_data"]["portfolio_metadata"]).parent
    windows = book_windows(cfg, pd.read_csv(reports / "summary.csv"))
    betas = beta_frames(cfg, panel_file, rcfg, fcfg)
    exposures = {
        name: {key: weights_exposure(weights[name], frame, daily[name].index, int(daily[name]["horizon"].iloc[0]))
               for key, frame in betas.items()}
        for name in cfg["books"]}
    tables = analyse(cfg, daily, exposures, factors, windows, pd.read_csv(REPO_ROOT / cfg["book_data"]["exposures_table"]),
                     rcfg["trading_days"])

    out = REPO_ROOT / cfg["outputs"]["reports_dir"]
    out.mkdir(parents=True, exist_ok=True)
    checks = tables["checks"]
    checks.to_csv(out / "checks.csv", index=False)
    if not checks["passed"].all():
        print(checks[~checks["passed"]].to_string(index=False))
        raise SystemExit("a pre-registered check failed: no results were written")
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    meta.update({"factor_files": manifest, "last_factor_date": str(factors.index.max().date()),
                 "holdout_start": rcfg["holdout_start"], "portfolio_config_sha256": file_sha256(REPO_ROOT / cfg["portfolio_config"]),
                 "book_files_sha256": {str(p): file_sha256(REPO_ROOT / p) for p in sorted({
                     cfg["book_data"]["daily_returns"].format(signal=s["signal"]) for s in cfg["books"].values()})}})
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(tables["primary_test"].T.to_string())
    return tables


if __name__ == "__main__":
    run()
