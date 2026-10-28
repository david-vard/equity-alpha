"""Baselines and pooled panel models with one fit / predict interface, plus the modelling config.

Every model takes ``X`` (rows x features, z-scores with NaN) and returns a score per row. Linear
models and baselines fill NaN with 0 (the cross-sectional mean of a z-score); LightGBM takes NaN
natively. Fitting is deterministic: closed-form Ridge, cyclic coordinate descent for Elastic Net,
seeded single-configuration LightGBM with fixed threads.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.utils import REPO_ROOT

MODELS_CONFIG_PATH = REPO_ROOT / "configs" / "models.yaml"
BASELINES = ("B0_reversal", "B1_composite", "B2_fama_macbeth")
MODEL_NAMES = ("ridge", "elastic_net", "lightgbm")
GRID_KEYS = {
    "ridge": ("alpha",),
    "elastic_net": ("alpha", "l1_ratio"),
    "lightgbm": ("num_leaves", "min_data_in_leaf", "lambda_l2", "learning_rate", "feature_fraction"),
}


# --------------------------------------------------------------------------- config


def load_models_config(path: str | Path | None = None, rcfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load ``configs/models.yaml`` and reject anything the code does not implement."""
    with open(path or MODELS_CONFIG_PATH) as handle:
        cfg = yaml.safe_load(handle)
    targets, wf, ev = cfg["targets"], cfg["walk_forward"], cfg["evaluation"]
    if rcfg is not None and not set(targets["horizons"]) <= set(rcfg["horizons"]):
        raise ValueError("model horizons must be research horizons")
    if set(targets["variants"]) != {"rank", "raw"} or sum(v["primary"] for v in targets["variants"].values()) != 1:
        raise ValueError("target variants must be rank and raw with exactly one primary")
    fixed = {
        "targets.label": (targets["label"], "fwd_ret_{h}d_xs"),
        "targets.evaluation_label": (targets["evaluation_label"], "fwd_ret_{h}d"),
        "walk_forward.scheme": (wf["scheme"], "expanding"),
        "walk_forward.refit": (wf["refit"], "every_january"),
        "walk_forward.purge": (wf["purge"], "label_end_before_first_test_date"),
        "walk_forward.inner_validation.purge": (wf["inner_validation"]["purge"], "label_end_before_first_validation_date"),
        "walk_forward.inner_validation.selection_metric": (wf["inner_validation"]["selection_metric"], "mean_daily_spearman_ic"),
        "evaluation.ic": (ev["ic"], "spearman"),
        "evaluation.multiple_testing.method": (ev["multiple_testing"]["method"], "holm"),
        "features.missing.linear": (cfg["features"]["missing"]["linear"], "zero_fill"),
        "features.missing.lightgbm": (cfg["features"]["missing"]["lightgbm"], "native"),
        "models.lightgbm.objective": (cfg["models"]["lightgbm"]["objective"], "regression"),
    }
    for key, (value, expected) in fixed.items():
        if value != expected:
            raise ValueError(f"{key}: only {expected!r} is implemented")
    features = cfg["features"]["columns"]
    if len(set(features)) != len(features):
        raise ValueError("duplicate feature columns")
    for name in ("B0_reversal", "B1_composite"):
        if not set(cfg["baselines"][name]["weights"]) <= set(features):
            raise ValueError(f"{name} uses a feature outside features.columns")
    for name, keys in GRID_KEYS.items():
        for key in keys:
            if not cfg["models"][name][key]:
                raise ValueError(f"models.{name}.{key} needs at least one value")
    for a, b in ev["comparisons"]:
        if {a, b} - set(BASELINES) - set(MODEL_NAMES):
            raise ValueError(f"unknown comparison {a} vs {b}")
    if not wf["inner_validation"]["refit_on_full_window"]:
        raise ValueError("only refitting on the full window is implemented")
    return cfg


def primary_variant(cfg: dict[str, Any]) -> str:
    return next(name for name, spec in cfg["targets"]["variants"].items() if spec["primary"])


def hyperparameter_grid(name: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product of the pre-registered values, in config order (the tie-break order)."""
    spec = cfg["models"][name]
    keys = GRID_KEYS[name]
    return [dict(zip(keys, values)) for values in itertools.product(*(spec[k] for k in keys))]


def family_features(features: list[str], rcfg: dict[str, Any]) -> dict[str, list[str]]:
    """Model features grouped by their research-config family (``z_<name>`` -> family of ``name``)."""
    out: dict[str, list[str]] = {}
    for column in features:
        out.setdefault(rcfg["signals"][column.removeprefix("z_")]["family"], []).append(column)
    return out


# --------------------------------------------------------------------------- targets


def rank_target(y: np.ndarray, date_ids: np.ndarray) -> np.ndarray:
    """Within-date uniform ranks mapped to [-0.5, 0.5] (ties averaged); NaN stays NaN."""
    s = pd.Series(np.asarray(y, dtype="float64"))
    groups = pd.Series(date_ids)
    rank = s.groupby(groups).rank()
    n = s.notna().groupby(groups).transform("sum")
    return ((rank - 0.5) / n - 0.5).to_numpy()


def zero_fill(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype="float64")
    return np.where(np.isnan(X), 0.0, X)


# --------------------------------------------------------------------------- models


class LinearScore:
    """Fixed linear score ``X @ w`` (B0 and B1); nothing is estimated."""

    def __init__(self, weights: np.ndarray):
        self.coef_ = np.asarray(weights, dtype="float64")

    def fit(self, X, y=None, **_: Any) -> "LinearScore":
        return self

    def predict(self, X) -> np.ndarray:
        return zero_fill(X) @ self.coef_


class FamaMacBethForecast:
    """Mean of per-date OLS slopes over the training dates, applied to current features (B2)."""

    def __init__(self, min_names: int):
        self.min_names = min_names

    def fit(self, X, y, date_ids: np.ndarray, **_: Any) -> "FamaMacBethForecast":
        X, y = zero_fill(X), np.asarray(y, dtype="float64")
        keep = ~np.isnan(y)
        X, y, d = X[keep], y[keep], np.asarray(date_ids)[keep]
        order = np.argsort(d, kind="stable")
        X, y, d = X[order], y[order], d[order]
        starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
        ends = np.r_[starts[1:], len(d)]
        coefs = []
        for s, e in zip(starts, ends):
            if e - s < max(self.min_names, X.shape[1] + 2):
                continue
            design = np.column_stack([np.ones(e - s), X[s:e]])
            coefs.append(np.linalg.lstsq(design, y[s:e], rcond=None)[0])
        if not coefs:
            raise ValueError("no training date has enough names for a cross-sectional regression")
        coefs = np.asarray(coefs)
        self.n_dates_ = len(coefs)
        self.intercept_, self.coef_ = float(coefs[:, 0].mean()), coefs[:, 1:].mean(axis=0)
        return self

    def predict(self, X) -> np.ndarray:
        return zero_fill(X) @ self.coef_ + self.intercept_


class RidgeModel:
    """Pooled Ridge: minimise ``mean((y - a - X b)^2) + alpha * ||b||^2``, closed form in float64."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, X, y, **_: Any) -> "RidgeModel":
        X, y = zero_fill(X), np.asarray(y, dtype="float64")
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc = X - x_mean
        gram = Xc.T @ Xc / len(y)
        self.coef_ = np.linalg.solve(gram + self.alpha * np.eye(X.shape[1]), Xc.T @ (y - y_mean) / len(y))
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X) -> np.ndarray:
        return zero_fill(X) @ self.coef_ + self.intercept_


class ElasticNetModel:
    """Pooled Elastic Net (sklearn objective), centred data and a precomputed Gram matrix."""

    def __init__(self, alpha: float, l1_ratio: float, max_iter: int, tol: float, seed: int):
        self.alpha, self.l1_ratio, self.max_iter, self.tol, self.seed = alpha, l1_ratio, max_iter, tol, seed

    def fit(self, X, y, **_: Any) -> "ElasticNetModel":
        from sklearn.linear_model import ElasticNet

        X, y = zero_fill(X), np.asarray(y, dtype="float64")
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc = X - x_mean
        model = ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=False, precompute=Xc.T @ Xc,
            max_iter=self.max_iter, tol=self.tol, selection="cyclic", random_state=self.seed,
        )
        model.fit(Xc, y - y_mean)
        self.coef_ = model.coef_.copy()
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        self.n_iter_ = int(model.n_iter_)
        return self

    def predict(self, X) -> np.ndarray:
        return zero_fill(X) @ self.coef_ + self.intercept_


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):  # OSError: the OpenMP runtime is missing
        return False
    return True


class LightGBMModel:
    """Seeded, deterministic LightGBM regressor; early stopping on a validation block if given.

    With ``n_rounds`` set the model trains exactly that many trees (the refit on the full window);
    otherwise it needs ``X_val`` / ``y_val`` and keeps the best iteration on validation l2.
    """

    def __init__(self, params: dict[str, Any], lgb_cfg: dict[str, Any], seed: int, num_threads: int,
                 n_rounds: int | None = None):
        self.params = {
            "objective": "regression", "metric": "l2", "verbosity": -1,
            "num_leaves": params["num_leaves"], "min_data_in_leaf": params["min_data_in_leaf"],
            "lambda_l2": params["lambda_l2"], "learning_rate": params["learning_rate"],
            "feature_fraction": params["feature_fraction"], "bagging_fraction": lgb_cfg["bagging_fraction"],
            "max_depth": lgb_cfg["max_depth"], "seed": seed, "data_random_seed": seed,
            "feature_fraction_seed": seed, "bagging_seed": seed, "num_threads": num_threads,
            "deterministic": True, "force_col_wise": True,
        }
        self.max_rounds = lgb_cfg["n_estimators"]["max"]
        self.early_stopping = lgb_cfg["n_estimators"]["early_stopping_rounds"]
        self.n_rounds = n_rounds

    def fit(self, X, y, X_val=None, y_val=None, **_: Any) -> "LightGBMModel":
        import lightgbm as lgb

        train = lgb.Dataset(np.asarray(X, dtype="float32"), label=np.asarray(y, dtype="float64"),
                            free_raw_data=True)
        if self.n_rounds is not None:
            self.booster_ = lgb.train(self.params, train, num_boost_round=self.n_rounds)
            self.best_iteration_ = self.n_rounds
            return self
        if X_val is None:
            raise ValueError("early stopping needs a validation block")
        valid = lgb.Dataset(np.asarray(X_val, dtype="float32"), label=np.asarray(y_val, dtype="float64"),
                            reference=train)
        self.booster_ = lgb.train(
            self.params, train, num_boost_round=self.max_rounds, valid_sets=[valid],
            callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
        )
        self.best_iteration_ = int(self.booster_.best_iteration or self.max_rounds)
        return self

    def predict(self, X) -> np.ndarray:
        return self.booster_.predict(np.asarray(X, dtype="float32"), num_iteration=self.best_iteration_)


@dataclass
class ScaledTarget:
    """Fit on ``y / sd(y)`` (pooled over the training rows) and return predictions in target units."""

    model: Any
    scale: float = field(default=float("nan"))

    def fit(self, X, y, X_val=None, y_val=None, **kwargs: Any) -> "ScaledTarget":
        y = np.asarray(y, dtype="float64")
        self.scale = float(np.std(y)) or 1.0
        y_val = None if y_val is None else np.asarray(y_val, dtype="float64") / self.scale
        self.model.fit(X, y / self.scale, X_val=X_val, y_val=y_val, **kwargs)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X) * self.scale


def build_model(name: str, params: dict[str, Any], cfg: dict[str, Any], n_rounds: int | None = None):
    """A model with the given hyperparameters, wrapped in ``ScaledTarget`` if the config asks for it."""
    m = cfg["models"]
    if name == "ridge":
        model = RidgeModel(params["alpha"])
    elif name == "elastic_net":
        en = m["elastic_net"]
        model = ElasticNetModel(params["alpha"], params["l1_ratio"], en["max_iter"], en["tol"], m["seed"])
    elif name == "lightgbm":
        model = LightGBMModel(params, m["lightgbm"], m["seed"], m["num_threads"], n_rounds)
    else:
        raise ValueError(f"unknown model {name!r}")
    return ScaledTarget(model) if cfg["targets"]["standardise_in_training"] else model


def baseline_weights(name: str, features: list[str], cfg: dict[str, Any]) -> np.ndarray:
    weights = cfg["baselines"][name]["weights"]
    return np.array([float(weights.get(f, 0.0)) for f in features])


def coefficients(model: Any) -> np.ndarray | None:
    """Slopes in standardised units (features are z-scores; the target is scaled to unit sd)."""
    inner = model.model if isinstance(model, ScaledTarget) else model
    return getattr(inner, "coef_", None)
