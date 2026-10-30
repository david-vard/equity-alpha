"""Holdout: the ``configs/holdout.yaml`` loader, the one-shot lock and the outcome rules.

The design is fixed in the config before any row on or after ``holdout_start`` is read. The loader
is strict: an unknown or missing key, a wrong type, a method the design does not define, a book that
differs from the frozen ones, or an input whose hash has changed is an error, never silently
ignored. ``load_holdout_panel`` is the only reader of holdout rows, and it refuses unless the
one-shot lock has already been written.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.models import load_models_config
from src.portfolio import _check_schema, _get, load_portfolio_config
from src.second_round import load_second_round_config
from src.sensitivity import load_sensitivity_config
from src.signals import load_research_config
from src.utils import REPO_ROOT, file_sha256, load_config

HOLDOUT_CONFIG_PATH = REPO_ROOT / "configs" / "holdout.yaml"
BOOK_ROLES = {"headline": "primary", "benchmark": "comparator", "candidate": "second_round_candidate"}
HEADLINE_LABELS = {"at_or_above": "holds_up", "at_or_below_minus": "fails", "between": "inconclusive"}
DIFFERENCE_LABELS = {"above_zero": "lightgbm_better", "below_zero": "b0_better", "contains_zero": "no_difference_shown"}
STATISTICS = {"net_sharpe", "net_nw_t", "net_return_ann", "net_vol_ann", "gross_sharpe", "annual_turnover",
              "breakeven_cost_multiplier", "cost_ann", "borrow_ann", "max_drawdown", "mean_gross_exposure"}
BOOTSTRAP_KEYS = ("method", "n_resamples", "seed", "confidence", "interval")

NUM = (int, float)
BOOK_SCHEMA = {"signal": str, "construction": str, "role": str, "lambdas": list, "liquid_fraction": NUM}
OUTCOME_T_SCHEMA = {"t_threshold": NUM, "labels": dict}
OUTCOME_INTERVAL_SCHEMA = {"interval": str, "labels": dict}

# Expected keys and leaf types. A dict is a section whose keys must match exactly.
SCHEMA: dict[str, Any] = {
    "portfolio_config": str,
    "research_config": str,
    "features_config": str,
    "models_config": str,
    "second_round_config": str,
    "pinned": {
        "panel_sha256": str, "models_config_sha256": str, "portfolio_config_sha256": str,
        "second_round_config_sha256": str, "portfolio_lambdas_sha256": str, "model_hyperparameters_sha256": str,
        "holdout_candidate_sha256": str,
    },
    "inputs": {
        "portfolio_lambdas": str, "model_hyperparameters": str, "holdout_candidate": str, "predictions": str,
        "recorded_daily_returns": str, "recorded_candidate_returns": str,
    },
    "books": {"headline": BOOK_SCHEMA, "benchmark": BOOK_SCHEMA, "candidate": BOOK_SCHEMA},
    "window": {"start": str, "last_panel_date": str, "statistics_start": str, "last_score_date": str, "runoff": str},
    "model_refit": {
        "model": str, "horizon": int, "target_variant": str, "scheme": str, "test_years": list, "settings": str,
        "grinold_ic": str, "benchmark_score": str,
    },
    "continuity": {"run_from": str, "december_2021": str, "reporting_starts": str, "engine_guard": str},
    "outcomes": {"headline": OUTCOME_T_SCHEMA, "lightgbm_minus_b0": OUTCOME_INTERVAL_SCHEMA, "candidate": OUTCOME_T_SCHEMA},
    "reporting": {"statistics": list, "by_year": bool, "breakevens": list, "model_ic": str, "realised_beta": str},
    "bootstrap": {"method": str, "n_resamples": int, "seed": int, "mean_block_length": int, "confidence": NUM, "interval": str},
    "one_shot": {"requires_flag": str, "requires_clean_tree": bool, "lock_file": str, "failed_check_policy": str},
    "checks": {"fold_2021_prediction_tolerance": NUM, "recorded_series_sharpe_tolerance": NUM, "fold_leakage": str},
    "outputs": {"reports_dir": str, "figures_dir": str, "data_dir": str, "notebook": str},
}

# Method choices that the design defines; anything else is rejected.
FIXED = {
    "window.start": "first_holdout_date",
    "window.statistics_start": "first_holdout_date",
    "window.last_score_date": "label_ends_by_last_panel_date",
    "window.runoff": "to_label_end",
    "model_refit.model": "lightgbm",
    "model_refit.target_variant": "rank",
    "model_refit.scheme": "continue_walk_forward",
    "model_refit.settings": "from_models_config",
    "model_refit.grinold_ic": "fold_validation_block",
    "model_refit.benchmark_score": "fixed_formula",
    "continuity.run_from": "first_oos_date",
    "continuity.december_2021": "fold_2021_model",
    "continuity.reporting_starts": "first_holdout_date",
    "continuity.engine_guard": "pass_first_date_after_panel",
    "outcomes.lightgbm_minus_b0.interval": "bootstrap_excludes_zero",
    "reporting.model_ic": "secondary",
    "reporting.realised_beta": "equal_weighted_universe",
    "one_shot.requires_flag": "--open-holdout",
    "one_shot.failed_check_policy": "stop_and_report",
    "checks.fold_leakage": "none",
}


def _check_pins(cfg: dict[str, Any]) -> None:
    pins, inputs = cfg["pinned"], cfg["inputs"]
    files = {
        "models_config_sha256": cfg["models_config"],
        "portfolio_config_sha256": cfg["portfolio_config"],
        "second_round_config_sha256": cfg["second_round_config"],
        "portfolio_lambdas_sha256": inputs["portfolio_lambdas"],
        "model_hyperparameters_sha256": inputs["model_hyperparameters"],
        "holdout_candidate_sha256": inputs["holdout_candidate"],
    }
    for key, path in files.items():
        if file_sha256(REPO_ROOT / path) != pins[key]:
            raise ValueError(f"pinned.{key}: {path} has changed since the design was committed")


def _check_books(cfg: dict[str, Any], pcfg: dict[str, Any]) -> None:
    books, head = cfg["books"], pcfg["headline"]
    for name, role in BOOK_ROLES.items():
        if books[name]["role"] != role:
            raise ValueError(f"books.{name}.role must be {role!r}")
    if (books["headline"]["signal"], books["headline"]["construction"]) != (head["signal"], head["construction"]):
        raise ValueError("books.headline must be the portfolio config's headline book")
    if (books["benchmark"]["signal"], books["benchmark"]["construction"]) != (head["benchmark"], head["construction"]):
        raise ValueError("books.benchmark must be the headline benchmark under the headline construction")
    if books["headline"]["liquid_fraction"] != 1 or books["benchmark"]["liquid_fraction"] != 1:
        raise ValueError("the headline and benchmark books use the whole universe")
    recorded = pd.read_csv(REPO_ROOT / cfg["inputs"]["portfolio_lambdas"]).set_index("signal")
    for name in ("headline", "benchmark"):
        signal, lambdas = books[name]["signal"], books[name]["lambdas"]
        if lambdas != [float(recorded.loc[signal, "lambda_risk"]), float(recorded.loc[signal, "lambda_tc"])]:
            raise ValueError(f"books.{name}.lambdas must be the recorded first-round choice")
    chosen = pd.read_csv(REPO_ROOT / cfg["inputs"]["holdout_candidate"])
    candidate = books["candidate"]
    if len(chosen) != 1:
        raise ValueError("inputs.holdout_candidate must hold exactly one book")
    row = chosen.iloc[0]
    if (candidate["signal"], candidate["construction"], candidate["liquid_fraction"]) != (
            row["signal"], row["construction"], float(row["fraction"])):
        raise ValueError("books.candidate must be the frozen second-round candidate")
    if candidate["lambdas"] != books["headline"]["lambdas"]:
        raise ValueError("books.candidate keeps its signal's recorded lambdas")


def _check_window_and_refit(cfg: dict[str, Any], fcfg: dict[str, Any], rcfg: dict[str, Any], mcfg: dict[str, Any]) -> None:
    window, refit = cfg["window"], cfg["model_refit"]
    if window["last_panel_date"] != fcfg["data"]["end"]:
        raise ValueError("window.last_panel_date must equal data.end in configs/features.yaml")
    if refit["horizon"] != 5 or refit["horizon"] not in mcfg["targets"]["horizons"]:
        raise ValueError("model_refit.horizon must be 5")
    first, last = pd.Timestamp(rcfg["holdout_start"]).year, pd.Timestamp(window["last_panel_date"]).year
    if refit["test_years"] != list(range(first, last + 1)):
        raise ValueError(f"model_refit.test_years must run from {first} to {last}, every year")
    if mcfg["walk_forward"]["refit"] != "every_january":
        raise ValueError("model_refit continues the first-round annual refit")


def _check_outcomes_and_bootstrap(cfg: dict[str, Any], sensitivity: dict[str, Any]) -> None:
    outcomes = cfg["outcomes"]
    for name in ("headline", "candidate"):
        if outcomes[name]["t_threshold"] != 2.0 or outcomes[name]["labels"] != HEADLINE_LABELS:
            raise ValueError(f"outcomes.{name} must use t = 2 and labels {HEADLINE_LABELS}")
    if outcomes["lightgbm_minus_b0"]["labels"] != DIFFERENCE_LABELS:
        raise ValueError(f"outcomes.lightgbm_minus_b0 must use labels {DIFFERENCE_LABELS}")
    reporting = cfg["reporting"]
    if set(reporting["statistics"]) != STATISTICS or not reporting["by_year"]:
        raise ValueError(f"reporting must cover {sorted(STATISTICS)} and every year")
    if set(reporting["breakevens"]) != {"impact_y", "borrow_bps_per_year"}:
        raise ValueError("reporting.breakevens must be impact_y and borrow_bps_per_year")
    boot, other = cfg["bootstrap"], sensitivity["bootstrap"]
    for key in BOOTSTRAP_KEYS:
        if boot[key] != other[key]:
            raise ValueError(f"bootstrap.{key} must equal configs/sensitivity.yaml")
    if boot["mean_block_length"] != other["mean_block_lengths"][0]:
        raise ValueError("bootstrap.mean_block_length must be the primary length of configs/sensitivity.yaml")


def _check_one_shot_and_outputs(cfg: dict[str, Any]) -> None:
    shot, checks, out = cfg["one_shot"], cfg["checks"], cfg["outputs"]
    if not shot["requires_clean_tree"]:
        raise ValueError("one_shot.requires_clean_tree must stay on")
    if Path(shot["lock_file"]).parent != Path(out["reports_dir"]):
        raise ValueError("one_shot.lock_file must be inside outputs.reports_dir")
    if not 0 < checks["fold_2021_prediction_tolerance"] <= 1e-4 or not 0 < checks["recorded_series_sharpe_tolerance"] <= 1e-4:
        raise ValueError("checks tolerances must be positive and at most 1e-4")
    if Path(out["reports_dir"]) != Path("reports/holdout") or Path(out["figures_dir"]).parent != Path(out["reports_dir"]):
        raise ValueError("outputs.reports_dir must be reports/holdout and figures_dir inside it")
    if Path(out["data_dir"]).parent != Path("data/processed"):
        raise ValueError("outputs.data_dir must be in data/processed (gitignored)")


def load_holdout_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``configs/holdout.yaml`` and reject anything outside the pre-registered design.

    The frozen inputs are read and their hashes compared. No return, panel row or prediction is read.
    """
    with open(path or HOLDOUT_CONFIG_PATH) as handle:
        cfg = yaml.safe_load(handle)
    _check_schema(cfg, SCHEMA, "")
    for key, expected in FIXED.items():
        if _get(cfg, key) != expected:
            raise ValueError(f"{key}: only {expected!r} is defined")
    _check_pins(cfg)
    fcfg = load_config(REPO_ROOT / cfg["features_config"])
    rcfg = load_research_config(REPO_ROOT / cfg["research_config"], fcfg)
    mcfg = load_models_config(REPO_ROOT / cfg["models_config"], rcfg)
    pcfg = load_portfolio_config(REPO_ROOT / cfg["portfolio_config"], fcfg, rcfg, mcfg)
    load_second_round_config(REPO_ROOT / cfg["second_round_config"], pcfg, fcfg, rcfg)
    _check_books(cfg, pcfg)
    _check_window_and_refit(cfg, fcfg, rcfg, mcfg)
    _check_outcomes_and_bootstrap(cfg, load_sensitivity_config(pcfg=pcfg, fcfg=fcfg, rcfg=rcfg))
    _check_one_shot_and_outputs(cfg)
    return cfg


# --------------------------------------------------------------------------- one shot


def open_lock(lock_file: Path, identity: dict[str, Any]) -> None:
    """Record the opening before the first holdout row is read; a second opening refuses."""
    if lock_file.exists():
        raise SystemExit(f"the holdout has already been opened: {lock_file}")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    stamp = {"opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **identity}
    lock_file.write_text(json.dumps(stamp, indent=2, default=str) + "\n")


def load_holdout_panel(columns: list[str], panel_file: str | Path, lock_file: Path) -> pd.DataFrame:
    """Every panel row through the last date, as float32; the only reader of holdout rows."""
    if not Path(lock_file).exists():
        raise SystemExit("the holdout panel can only be read after the one-shot lock is written")
    panel = pd.read_parquet(panel_file, columns=columns)
    float_cols = [c for c in panel.columns if panel[c].dtype != np.float32 and panel[c].dtype.kind == "f"]
    return panel.astype({c: "float32" for c in float_cols}) if float_cols else panel


# --------------------------------------------------------------------------- dates and outcomes


def last_score_date(calendar: pd.DatetimeIndex, lag: int, horizon: int) -> pd.Timestamp:
    """The last date whose position can still be closed at its label end inside the panel."""
    return pd.DatetimeIndex(sorted(set(calendar)))[-1 - lag - horizon]


def t_outcome(t: float, threshold: float, labels: dict[str, str]) -> str:
    """The declared label of a net t-statistic; a missing t is inconclusive."""
    if t >= threshold:
        return labels["at_or_above"]
    return labels["at_or_below_minus"] if t <= -threshold else labels["between"]


def interval_outcome(low: float, high: float, labels: dict[str, str]) -> str:
    """The declared label of a paired difference, from its bootstrap interval."""
    if low > 0:
        return labels["above_zero"]
    return labels["below_zero"] if high < 0 else labels["contains_zero"]
