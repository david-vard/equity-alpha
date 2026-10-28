"""Modelling: walk-forward out-of-sample forecasts from baselines, Ridge, Elastic Net and LightGBM.

Usage (from the repository root):
    python -m src.run_models [--config configs/models.yaml] [--panel PATH] [--out DIR]
                             [--predictions PATH] [--models ridge,elastic_net,...] [--lightgbm-ablation]

Reads only research-sample rows through ``src.signals.load_research_panel``. The design is fixed in
``configs/models.yaml``: the run refuses to start unless it (and ``configs/research.yaml``) is
committed and unmodified, unless ``--allow-dirty-config`` is given. Writes tidy CSVs, figures and
``run_metadata.json`` to ``reports/models/`` and the out-of-sample predictions to
``data/processed/oof_predictions.parquet`` (gitignored).
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")  # headless: figures are only saved

from src import model_plots as plots  # noqa: E402
from src.metrics import adjust_pvalues  # noqa: E402
from src.model_evaluation import (  # noqa: E402
    decile_evaluation,
    ic_row,
    mean_daily_ic,
    oos_r2,
    paired_ic_test,
    permutation_importance,
    shap_summary,
)
from src.models import (  # noqa: E402
    BASELINES,
    MODEL_NAMES,
    MODELS_CONFIG_PATH,
    FamaMacBethForecast,
    LinearScore,
    baseline_weights,
    build_model,
    coefficients,
    family_features,
    hyperparameter_grid,
    lightgbm_available,
    load_models_config,
    primary_variant,
    rank_target,
)
from src.run_signal_research import provenance  # noqa: E402
from src.signals import (  # noqa: E402
    RESEARCH_CONFIG_PATH,
    Grid,
    cross_sectional_ic,
    exclude_terminal,
    ic_decay,
    ic_stability,
    load_research_config,
    load_research_panel,
    panel_path,
    research_cutoff,
)
from src.utils import REPO_ROOT, load_config  # noqa: E402
from src.validation import Fold, position_mask, thin, walk_forward_folds  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "models"
PREDICTIONS_PATH = REPO_ROOT / "data" / "processed" / "oof_predictions.parquet"
REFERENCE_DIR = REPO_ROOT / "reports" / "signal_research"
BETA = "beta_252d"


def column_name(predictor: str, h: int, variant: str | None) -> str:
    return f"{predictor}_h{h}" + (f"_{variant}" if variant else "")


class Data:
    """Row arrays of the research panel shared by all fits (features, targets, labels, date ids)."""

    def __init__(self, panel: pd.DataFrame, features: list[str], horizons: list[int]):
        self.index = panel.index
        self.grid = Grid.from_index(panel.index)
        self.dates = self.grid.dates
        self.date_pos = self.grid.rows
        self.X = panel[features].to_numpy(dtype="float32")
        self.label = {h: panel[f"fwd_ret_{h}d"].to_numpy(dtype="float64") for h in horizons}
        self.xs = {h: panel[f"fwd_ret_{h}d_xs"].to_numpy(dtype="float64") for h in horizons}
        self.terminal = {h: panel[f"fwd_terminal_{h}d"].to_numpy() for h in horizons}
        self.ret_1d = panel["fwd_ret_1d"].to_numpy(dtype="float64")
        self.beta = panel[BETA].to_numpy(dtype="float64")
        self.targets = {
            h: {"raw": self.xs[h], "rank": rank_target(self.xs[h], self.date_pos)} for h in horizons
        }


def walk_forward(
    name: str, data: Data, columns: np.ndarray, target: np.ndarray, h: int, folds: list[Fold],
    mcfg: dict[str, Any], lag: int, min_names: int, fixed: dict[int, dict[str, Any]] | None = None,
    collect: Callable[[Fold, Any, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Out-of-sample predictions of one model on ``data.X[:, columns]`` over all folds.

    Per fold: every hyperparameter combination is fitted on the inner training dates and scored by
    the mean daily Spearman IC (prediction vs the raw evaluation label) on the validation block; the
    best one (ties: first in config order) is refitted on the whole purged training window, LightGBM
    with its early-stopped number of trees. ``fixed`` skips the search (fold id -> parameters).
    Returns the predictions (NaN outside the test years), the chosen parameters and the grid scores.
    """
    n_dates, y_eval = len(data.dates), data.label[h]
    stride = mcfg["models"][name]["train_date_stride"]
    X = data.X[:, columns]
    has_target = ~np.isnan(target)
    pred = np.full(len(target), np.nan)
    chosen, scores = [], []
    for fold in folds:
        inner = position_mask(data.date_pos, thin(fold.inner_train, stride), n_dates) & has_target
        val = position_mask(data.date_pos, fold.validation, n_dates) & has_target
        train = position_mask(data.date_pos, thin(fold.train, stride), n_dates) & has_target
        test = position_mask(data.date_pos, fold.test, n_dates)
        if data.date_pos[train].max() + lag + h >= data.date_pos[test].min():
            raise AssertionError(f"fold {fold.fold_id}: training rows leak into the test year")
        grid = [fixed[fold.fold_id]] if fixed else hyperparameter_grid(name, mcfg)
        best, best_score, best_rounds = None, -np.inf, None
        for params in grid:
            model = build_model(name, params, mcfg)
            model.fit(X[inner], target[inner], X_val=X[val], y_val=target[val])
            score = mean_daily_ic(model.predict(X[val]), y_eval[val], data.date_pos[val], min_names)
            rounds = getattr(getattr(model, "model", model), "best_iteration_", None)
            scores.append({"fold_id": fold.fold_id, **params, "validation_ic": score, "n_rounds": rounds})
            if np.isfinite(score) and score > best_score or best is None:
                best, best_score, best_rounds = params, score, rounds
        final = build_model(name, best, mcfg, n_rounds=best_rounds)
        final.fit(X[train], target[train])
        pred[test] = final.predict(X[test])
        chosen.append({"fold_id": fold.fold_id, **best, "validation_ic": best_score, "n_rounds": best_rounds,
                       "train_rows": int(train.sum()), "test_rows": int(test.sum())})
        if collect is not None:
            collect(fold, final, test)
    return pred, chosen, scores


def baseline_predictions(
    name: str, data: Data, features: list[str], target: np.ndarray | None, folds: list[Fold],
    mcfg: dict[str, Any], min_names: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """B0 / B1 score every test row; B2 averages per-date slopes over each fold's purged training dates."""
    pred = np.full(len(data.X), np.nan)
    n_dates = len(data.dates)
    rows = []
    for fold in folds:
        test = position_mask(data.date_pos, fold.test, n_dates)
        if name == "B2_fama_macbeth":
            train = position_mask(data.date_pos, fold.train, n_dates) & ~np.isnan(target)
            model = FamaMacBethForecast(min_names).fit(data.X[train], target[train], data.date_pos[train])
            rows.extend({"fold_id": fold.fold_id, "feature": f, "coef": c} for f, c in zip(features, model.coef_))
        else:
            model = LinearScore(baseline_weights(name, features, mcfg))
        pred[test] = model.predict(data.X[test])
    return pred, rows


def run(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=MODELS_CONFIG_PATH)
    parser.add_argument("--research-config", type=Path, default=RESEARCH_CONFIG_PATH)
    parser.add_argument("--features-config", type=Path, default=None)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--reference", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--models", default=",".join(MODEL_NAMES), help="subset of the pre-registered models")
    parser.add_argument("--lightgbm-ablation", action="store_true", help="also run the LightGBM family ablation")
    parser.add_argument("--allow-dirty-config", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    fcfg = load_config(args.features_config)
    rcfg = load_research_config(args.research_config, fcfg)
    mcfg = load_models_config(args.config, rcfg)
    panel_file = args.panel or panel_path(fcfg)
    identity = provenance(args.config, args.features_config, panel_file, args.allow_dirty_config)
    research_identity = provenance(args.research_config, args.features_config, panel_file, args.allow_dirty_config)
    models = [m for m in args.models.split(",") if m]
    if set(models) - set(MODEL_NAMES):
        raise SystemExit(f"--models must be a subset of {MODEL_NAMES}")
    if "lightgbm" in models and not lightgbm_available():
        raise SystemExit("lightgbm cannot be imported (on macOS: brew install libomp)")
    out, fig_dir = args.out, args.out / "figures"
    for d in (out, fig_dir, args.predictions.parent):
        d.mkdir(parents=True, exist_ok=True)

    horizons = mcfg["targets"]["horizons"]
    variants = list(mcfg["targets"]["variants"])
    primary = primary_variant(mcfg)
    features = mcfg["features"]["columns"]
    min_names, q, td = rcfg["min_names_per_date"], rcfg["quantiles"], rcfg["trading_days"]
    lag, offset = fcfg["labels"]["lag"], rcfg["portfolios"]["offset"]
    wf, ev = mcfg["walk_forward"], mcfg["evaluation"]
    cutoff = research_cutoff(panel_file, rcfg, fcfg)
    columns = list(features) + [f"fwd_ret_{h}d{s}" for h in horizons for s in ("", "_xs")]
    columns += [f"fwd_terminal_{h}d" for h in horizons] + ["fwd_ret_1d", BETA]
    panel = load_research_panel(list(dict.fromkeys(columns)), panel_file, rcfg, fcfg, cutoff)
    data = Data(panel, features, horizons)
    del panel
    dates = data.dates
    print(f"research sample: {dates[0].date()} to {dates[-1].date()} ({len(dates)} dates, {len(data.X):,} rows)")

    folds = {h: walk_forward_folds(dates, h, lag, wf["first_test_year"], wf["inner_validation"]["years"],
                                   dates[-1]) for h in horizons}
    first_test = dates[folds[horizons[0]][0].test[0]]
    oos_pos = np.flatnonzero(dates >= first_test)
    all_cols = np.arange(len(features))

    # ------------------------------------------------------------------ out-of-sample predictions
    preds: dict[str, tuple[str, int, str | None, np.ndarray]] = {}
    chosen_rows, grid_rows, coef_rows, perm_rows, shap_rows = [], [], [], [], []
    lgb_params: dict[tuple[int, str], dict[int, dict[str, Any]]] = {}
    interp = ev["interpretation"]
    for h in horizons:
        for name in ("B0_reversal", "B1_composite"):
            pred, _ = baseline_predictions(name, data, features, None, folds[h], mcfg, min_names)
            preds[column_name(name, h, None)] = (name, h, None, pred)
        for variant in variants:
            target = data.targets[h][variant]
            pred, rows = baseline_predictions("B2_fama_macbeth", data, features, target, folds[h], mcfg, min_names)
            preds[column_name("B2_fama_macbeth", h, variant)] = ("B2_fama_macbeth", h, variant, pred)
            coef_rows.extend({"model": "B2_fama_macbeth", "horizon": h, "variant": variant, **r} for r in rows)
            for name in models:
                tag = {"model": name, "horizon": h, "variant": variant}

                def collect(fold: Fold, model: Any, test: np.ndarray, name=name, tag=tag) -> None:
                    coef = coefficients(model)
                    if coef is not None:
                        coef_rows.extend({**tag, "fold_id": fold.fold_id, "feature": f, "coef": float(c)}
                                         for f, c in zip(features, coef))
                    if name == "lightgbm":
                        ok = test & ~np.isnan(data.label[h])
                        perm = permutation_importance(
                            model, data.X[ok], data.label[h][ok], data.date_pos[ok], features,
                            interp["lightgbm_permutation_importance"]["repeats"], min_names, mcfg["models"]["seed"])
                        perm_rows.extend({**tag, "fold_id": fold.fold_id, **r} for r in perm)
                        test_dates = np.unique(data.date_pos[test])
                        n_pick = min(interp["lightgbm_shap"]["dates_per_fold"], len(test_dates))
                        picked = test_dates[np.linspace(0, len(test_dates) - 1, n_pick).round().astype(int)]
                        sample = test & np.isin(data.date_pos, picked)
                        shap = shap_summary(getattr(model, "model", model).booster_, data.X[sample], features)
                        shap_rows.extend({**tag, "fold_id": fold.fold_id, **r} for r in shap)

                pred, chosen, scores = walk_forward(name, data, all_cols, target, h, folds[h], mcfg, lag,
                                                    min_names, collect=collect)
                preds[column_name(name, h, variant)] = (name, h, variant, pred)
                chosen_rows.extend({**tag, **r} for r in chosen)
                grid_rows.extend({**tag, **r} for r in scores)
                if name == "lightgbm":
                    keys = ("num_leaves", "min_data_in_leaf", "lambda_l2", "learning_rate", "feature_fraction")
                    lgb_params[(h, variant)] = {r["fold_id"]: {k: r[k] for k in keys} for r in chosen}
                print(f"  h={h} {variant:<4} {name:<12} done ({time.time() - started:.0f}s)")

    # every test row has a prediction and nothing precedes the first test date
    oos_rows = data.date_pos >= oos_pos[0]
    for col, (_, _, _, pred) in preds.items():
        if np.isfinite(pred[~oos_rows]).any():
            raise AssertionError(f"{col} has predictions before the first test date")
    fold_id = dates[data.date_pos].year.to_numpy() - wf["first_test_year"]
    oof = pd.DataFrame({c: p[3][oos_rows].astype("float32") for c, p in preds.items()},
                       index=data.index[oos_rows])
    oof["fold_id"] = fold_id[oos_rows].astype("int16")
    oof.to_parquet(args.predictions)

    # ------------------------------------------------------------------ evaluation on OOS dates
    grid = data.grid
    oos_dates = dates[oos_pos]
    label_w = {h: grid.wide(data.label[h]).loc[oos_dates] for h in horizons}
    beta_w = grid.wide(data.beta).loc[oos_dates]
    ret1_w = grid.wide(data.ret_1d).loc[oos_dates]
    stab_cfg = copy.deepcopy(rcfg)
    stab_cfg["periods"] = {**ev["stability"]["periods"], "halves": ev["stability"]["halves"]}

    ic_rows, daily_ic, spread_rows, decile_rows, to_rows, stab_rows, decay_rows = [], {}, [], [], [], [], []
    for col, (name, h, variant, pred) in preds.items():
        tag = {"predictor": name, "horizon": h, "variant": variant or "none", "column": col}
        wide = grid.wide(pred).loc[oos_dates]
        ic = cross_sectional_ic(wide, label_w[h], min_names)
        daily_ic[col] = ic
        row = {**tag, **ic_row(ic, h, label_w[h], rcfg)}
        if variant is not None:  # prediction in target units: B2 and models
            ok = oos_rows & np.isfinite(pred)
            row["oos_r2_vs_zero"] = oos_r2(pred[ok], data.targets[h][variant][ok])
            row["oos_r2_target"] = f"fwd_ret_{h}d_xs" + ("_rank" if variant == "rank" else "")
        ic_rows.append(row)
        spreads, deciles, to = decile_evaluation(wide, label_w[h], beta_w, h, q, min_names, td, offset)
        spread_rows.extend({**tag, **r} for r in spreads)
        decile_rows.extend({**tag, **r} for r in deciles)
        to_rows.append({**tag, **to})
        stab_rows.append(ic_stability(ic, h, stab_cfg).assign(**tag))
        decay = ic_decay(wide, ret1_w, ev["stability"]["decay_days"], min_names, rcfg)
        decay_rows.append(decay.assign(**tag, half_life=decay.attrs["half_life"]))
        print(f"  evaluated {col:<34} ({time.time() - started:.0f}s)")

    alpha = ev["multiple_testing"]["alpha"]
    ic_table = pd.DataFrame(ic_rows)
    ic_table["p_value_holm"] = adjust_pvalues(ic_table["p_value"].to_numpy(), "holm")
    ic_table["reject_holm"] = ic_table["p_value_holm"] < alpha

    comp_rows = []
    for h in horizons:
        for variant in variants:
            for a, b in ev["comparisons"]:
                if not {a, b} <= set(BASELINES) | set(models):
                    continue
                ca = column_name(a, h, None if a in ("B0_reversal", "B1_composite") else variant)
                cb = column_name(b, h, None if b in ("B0_reversal", "B1_composite") else variant)
                comp_rows.append({"model": a, "baseline": b, "horizon": h, "variant": variant,
                                  **paired_ic_test(daily_ic[ca], daily_ic[cb], h)})
    comparisons = pd.DataFrame(comp_rows)
    if len(comparisons):
        comparisons["p_value_holm"] = adjust_pvalues(comparisons["p_value"].to_numpy(), "holm")
        comparisons["reject_holm"] = comparisons["p_value_holm"] < alpha

    # ------------------------------------------------------------------ family ablation
    families = family_features(features, rcfg)
    ablation_specs = []
    for family, cols in families.items():
        ablation_specs.append(("leave_one_family_out", family, [f for f in features if f not in cols]))
        ablation_specs.append(("single_family", family, cols))
    ablation_models = ["ridge"] if "ridge" in models else []
    if args.lightgbm_ablation and "lightgbm" in models:
        ablation_models.append("lightgbm")
    abl_rows = []
    for name in ablation_models:
        abl_variants = variants if name == "ridge" else [primary]
        for h in horizons:
            for variant in abl_variants:
                full = ic_table[ic_table["column"] == column_name(name, h, variant)].iloc[0]
                abl_rows.append({"model": name, "horizon": h, "variant": variant, "kind": "all_features",
                                 "family": "all", "n_features": len(features),
                                 "ic_mean": full["ic_mean"], "nw_t": full["nw_t"]})
                for kind, family, cols in ablation_specs:
                    idx = np.array([features.index(c) for c in cols])
                    fixed = lgb_params[(h, variant)] if name == "lightgbm" else None
                    pred, _, _ = walk_forward(name, data, idx, data.targets[h][variant], h, folds[h], mcfg,
                                              lag, min_names, fixed=fixed)
                    ic = cross_sectional_ic(grid.wide(pred).loc[oos_dates], label_w[h], min_names)
                    stats = ic_row(ic, h, label_w[h], rcfg)
                    abl_rows.append({"model": name, "horizon": h, "variant": variant, "kind": kind,
                                     "family": family, "n_features": len(cols),
                                     "ic_mean": stats["ic_mean"], "nw_t": stats["nw_t"]})
                print(f"  ablation {name} h={h} {variant} done ({time.time() - started:.0f}s)")
    ablation = pd.DataFrame(abl_rows)
    if len(ablation):
        ablation["ic_change_vs_all"] = ablation["ic_mean"] - ablation.groupby(
            ["model", "horizon", "variant"])["ic_mean"].transform("first")

    # ------------------------------------------------------------------ terminal-label sensitivity
    term_rows = []
    model_ics = ic_table[ic_table["predictor"].isin(models) & (ic_table["variant"] == primary)]
    for h in horizons:
        sub = model_ics[model_ics["horizon"] == h]
        if not len(sub):
            continue
        best = sub.loc[sub["ic_mean"].idxmax()]
        wide = grid.wide(preds[best["column"]][3]).loc[oos_dates]
        term = grid.wide(data.terminal[h]).loc[oos_dates]
        ic_ex = cross_sectional_ic(wide, exclude_terminal(label_w[h], term), min_names)
        ex = ic_row(ic_ex, h, label_w[h], rcfg)
        term_rows.append({"horizon": h, "best_model": best["predictor"], "variant": primary,
                          "ic_mean_all": best["ic_mean"], "nw_t_all": best["nw_t"],
                          "ic_mean_ex_terminal": ex["ic_mean"], "nw_t_ex_terminal": ex["nw_t"],
                          "terminal_rows_oos": int(np.nansum(term.to_numpy() == 1))})

    # ------------------------------------------------------------------ outputs
    coefs = pd.DataFrame(coef_rows)
    tables = {
        "folds": pd.DataFrame([dict(horizon=h, **f.summary(dates)) for h in horizons for f in folds[h]]),
        "ic_summary": ic_table,
        "model_comparison": comparisons,
        "decile_spreads": pd.DataFrame(spread_rows),
        "decile_means": pd.DataFrame(decile_rows),
        "prediction_turnover": pd.DataFrame(to_rows),
        "ic_stability": pd.concat(stab_rows, ignore_index=True),
        "ic_decay": pd.concat(decay_rows, ignore_index=True),
        "hyperparameters": pd.DataFrame(chosen_rows),
        "hyperparameter_grid_scores": pd.DataFrame(grid_rows),
        "coefficients": coefs,
        "coefficient_signs": coefficient_signs(coefs, features, rcfg, args.reference),
        "lightgbm_permutation_importance": pd.DataFrame(perm_rows),
        "lightgbm_shap": pd.DataFrame(shap_rows),
        "family_ablation": ablation,
        "terminal_sensitivity": pd.DataFrame(term_rows),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "cache").mkdir(exist_ok=True)
    pd.DataFrame(daily_ic).to_parquet(out / "cache" / "daily_ic.parquet")

    for h in horizons:
        plots.plot_cumulative_ic(pd.DataFrame(daily_ic), tables["ic_summary"], h, primary,
                                 fig_dir / f"cumulative_ic_h{h}.png")
        plots.plot_ic_by_year(tables["ic_stability"], h, primary, fig_dir / f"ic_by_year_h{h}.png")
    if len(coefs):
        plots.plot_coefficient_paths(coefs, primary, fig_dir / "linear_coefficient_paths.png")
    if perm_rows:
        plots.plot_lightgbm_importance(tables["lightgbm_permutation_importance"], tables["lightgbm_shap"],
                                       primary, fig_dir / "lightgbm_importance.png")

    assert dates[-1] <= cutoff < pd.Timestamp(rcfg["holdout_start"]), "output reaches the holdout"
    metadata = {
        **identity,
        **{f"research_{k}": v for k, v in research_identity.items() if k.startswith("config")},
        "holdout_start": rcfg["holdout_start"],
        "first_research_date": str(dates[0].date()),
        "last_research_date": str(dates[-1].date()),
        "first_oos_date": str(first_test.date()),
        "last_oos_date": str(dates[-1].date()),
        "n_research_rows": len(data.X),
        "n_oos_rows": int(oos_rows.sum()),
        "models_run": models,
        "lightgbm_ablation": bool(args.lightgbm_ablation and "lightgbm" in models),
        "primary_variant": primary,
        "seed": mcfg["models"]["seed"],
        "num_threads": mcfg["models"]["num_threads"],
        "folds": {str(h): [f.summary(dates) for f in folds[h]] for h in horizons},
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": _version("lightgbm") if "lightgbm" in models else None,
        "shap": _version("shap") if "lightgbm" in models else None,
        "runtime_seconds": round(time.time() - started, 1),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "folds"}, indent=2, default=str))
    return tables


def _version(module: str) -> str:
    return __import__(module).__version__


def coefficient_signs(coefs: pd.DataFrame, features: list[str], rcfg: dict[str, Any], ref_dir: Path) -> pd.DataFrame:
    """Mean standardised coefficient per feature against the research priors and Fama-MacBeth signs.

    ``prior_sign`` is the expected IC sign in ``research.yaml``; ``signal_research_ic_sign`` and
    ``fm_multivariate_sign`` come from the signal-research run (same horizon) when its CSVs exist.
    """
    if not len(coefs):
        return pd.DataFrame()
    stats = (coefs.groupby(["model", "horizon", "variant", "feature"])["coef"]
             .agg(coef_mean="mean", share_positive=lambda c: float((c > 0).mean()),
                  share_zero=lambda c: float((c == 0).mean()))
             .reset_index())
    stats["prior_sign"] = [rcfg["signals"][f.removeprefix("z_")]["expected_sign"] for f in stats["feature"]]
    ic_file, fm_file = ref_dir / "ic_summary.csv", ref_dir / "fama_macbeth_multivariate.csv"
    if ic_file.exists():
        ic = pd.read_csv(ic_file).assign(feature=lambda d: "z_" + d["feature"])
        stats = stats.merge(ic[["feature", "horizon", "ic_mean"]].rename(columns={"ic_mean": "signal_research_ic"}),
                            on=["feature", "horizon"], how="left")
        stats["signal_research_ic_sign"] = np.sign(stats["signal_research_ic"])
    if fm_file.exists():
        fm = pd.read_csv(fm_file).query("regressor != 'const'")
        fm = fm.assign(feature="z_" + fm["regressor"].str.removeprefix("z_"))
        stats = stats.merge(fm[["feature", "horizon", "coef_mean"]].rename(columns={"coef_mean": "fm_multivariate_coef"}),
                            on=["feature", "horizon"], how="left")
        stats["fm_multivariate_sign"] = np.sign(stats["fm_multivariate_coef"])
    stats["coef_sign"] = np.sign(stats["coef_mean"])
    stats["matches_prior"] = np.where(stats["prior_sign"] != 0, stats["coef_sign"] == stats["prior_sign"], np.nan)
    return stats


if __name__ == "__main__":
    run()
