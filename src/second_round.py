"""Second round: the ``configs/second_round.yaml`` loader.

The design is fixed in the config before any variant is simulated. The loader is strict: an unknown
or missing key, a wrong type, a method the design does not define, a grid without its recorded point,
or a lambda point that doesn't extend outward from the recorded choice is an error, never silently
ignored. Every evaluation-window result of this round is a second look.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.models import load_models_config
from src.portfolio import _check_schema, _get, _is_type, _positive_numbers, load_portfolio_config
from src.sensitivity import load_sensitivity_config
from src.signals import load_research_config
from src.utils import REPO_ROOT, load_config

SECOND_ROUND_CONFIG_PATH = REPO_ROOT / "configs" / "second_round.yaml"
BAND_CONSTRUCTION = "neutral_volscaled"
LIQUID_CONSTRUCTIONS = ("neutral_volscaled", "optimised")
EVALUATION_STATISTICS = {"net_sharpe", "net_nw_t", "net_return_ann", "gross_sharpe", "annual_turnover",
                         "breakeven_cost_multiplier"}
BOOTSTRAP_KEYS = ("method", "n_resamples", "seed", "confidence", "interval")

NUM = (int, float)

# Expected keys and leaf types. A dict is a section whose keys must match exactly; ``dict`` as a
# leaf is a mapping whose entries the checks below validate.
SCHEMA: dict[str, Any] = {
    "portfolio_config": str,
    "research_config": str,
    "features_config": str,
    "models_config": str,
    "recorded": {"hyperparameters": str, "lambda_grid": str, "summary": str, "portfolio_metadata": str, "daily_returns": str},
    "signals": {"primary": str, "benchmark": str},
    "no_trade_band": {
        "construction": str, "bands": list, "band_unit": str, "trade_rule": str, "unscored_names": str,
        "neutralise": str, "scale": str,
    },
    "liquid_universe": {
        "constructions": list, "fractions": list, "rank_feature": str, "rank_date": str, "rescore": str,
        "held_outside": str, "lambdas": str, "validation_ic": str,
    },
    "wider_lambda_grid": {"construction": str, "points": dict},
    "selection": {
        "metric": str, "window": str, "ties": str, "per": str, "recorded_point_competes": bool, "grid_edge": str,
    },
    "holdout_candidate": {"signal": str, "metric": str, "window": str, "must_exceed": str, "at_most": int},
    "evaluation": {"runs": str, "scenario": str, "evidence_label": str, "confirmatory_tests": str, "statistics": list},
    "bootstrap": {
        "method": str, "n_resamples": int, "seed": int, "mean_block_length": int, "confidence": NUM, "interval": str,
        "paired_against": str,
    },
    "checks": {
        "recorded_book_tolerance": NUM, "recorded_optimised_sharpe_tolerance": NUM, "tuning_isolation": str,
        "last_date": str, "evaluation_needs_matching_tuning_config": bool,
    },
    "outputs": {"reports_dir": str, "figures_dir": str, "data_dir": str, "notebook": str},
}
POINT_SCHEMA = {"extends_from": list, "added": list}

# Method choices that the design defines; anything else is rejected.
FIXED = {
    "no_trade_band.construction": BAND_CONSTRUCTION,
    "no_trade_band.band_unit": "mean_abs_target_weight",
    "no_trade_band.trade_rule": "to_band_edge",
    "no_trade_band.unscored_names": "sold_in_full",
    "no_trade_band.neutralise": "traded_names_only",
    "no_trade_band.scale": "down_only",
    "liquid_universe.rank_feature": "dollar_volume_21d",
    "liquid_universe.rank_date": "score_date",
    "liquid_universe.rescore": "within_kept_names",
    "liquid_universe.held_outside": "sold_at_rebalance",
    "liquid_universe.lambdas": "frozen_first_round",
    "liquid_universe.validation_ic": "full_universe",
    "wider_lambda_grid.construction": "optimised",
    "selection.metric": "net_sharpe",
    "selection.window": "tuning",
    "selection.ties": "first_in_config_order",
    "selection.per": "variant_signal_construction",
    "selection.grid_edge": "flag_and_stop",
    "holdout_candidate.metric": "net_sharpe",
    "holdout_candidate.window": "tuning",
    "holdout_candidate.must_exceed": "headline_tuning_net_sharpe",
    "evaluation.runs": "chosen_variants_once",
    "evaluation.scenario": "base",
    "evaluation.evidence_label": "second_look",
    "evaluation.confirmatory_tests": "none",
    "bootstrap.paired_against": "recorded_book",
    "checks.tuning_isolation": "no_date_after_tuning_end",
    "checks.last_date": "before_holdout_start",
}


def _check_signals(cfg: dict[str, Any], pcfg: dict[str, Any]) -> None:
    head, signals = pcfg["headline"], cfg["signals"]
    if signals["primary"] != head["signal"] or signals["benchmark"] != head["benchmark"]:
        raise ValueError("signals must be the headline signal and its benchmark")
    if cfg["holdout_candidate"]["signal"] != signals["primary"]:
        raise ValueError("holdout_candidate.signal must be the primary signal")
    if cfg["holdout_candidate"]["at_most"] != 1:
        raise ValueError("holdout_candidate.at_most must be 1")


def _check_band_and_liquid(cfg: dict[str, Any]) -> None:
    bands, liquid = cfg["no_trade_band"]["bands"], cfg["liquid_universe"]
    _positive_numbers("no_trade_band.bands", bands, allow_zero=True)
    if bands[0] != 0:
        raise ValueError("no_trade_band.bands must start at 0, the recorded book")
    _positive_numbers("liquid_universe.fractions", liquid["fractions"])
    if liquid["fractions"][-1] != 1.0 or liquid["fractions"][0] <= 0:
        raise ValueError("liquid_universe.fractions must end at 1, the recorded book")
    if tuple(liquid["constructions"]) != LIQUID_CONSTRUCTIONS:
        raise ValueError(f"liquid_universe.constructions must be {LIQUID_CONSTRUCTIONS}")


def _check_lambda_points(cfg: dict[str, Any], pcfg: dict[str, Any], hyperparameters: pd.DataFrame) -> None:
    """Added points are new, positive and outward from the recorded choice, on the edges it reached."""
    points = cfg["wider_lambda_grid"]["points"]
    if set(points) != set(cfg["signals"].values()):
        raise ValueError("wider_lambda_grid.points must cover exactly the two signals")
    opt = pcfg["constructions"]["optimised"]
    grid_risk, grid_tc = opt["lambda_risk"], opt["lambda_tc"]
    chosen = hyperparameters.set_index("signal")
    for signal, spec in points.items():
        _check_schema(spec, POINT_SCHEMA, f"wider_lambda_grid.points.{signal}")
        risk0, tc0 = spec["extends_from"]
        if (risk0, tc0) != (float(chosen.loc[signal, "lambda_risk"]), float(chosen.loc[signal, "lambda_tc"])):
            raise ValueError(f"{signal}: extends_from must be the recorded choice")
        added = spec["added"]
        if not added or len({tuple(p) for p in added}) != len(added):
            raise ValueError(f"{signal}: added must be distinct points")
        for point in added:
            if not (len(point) == 2 and all(_is_type(v, NUM) and v > 0 for v in point)):
                raise ValueError(f"{signal}: each added point is [lambda_risk, lambda_tc], both positive")
            risk, tc = point
            if risk in grid_risk and tc in grid_tc:
                raise ValueError(f"{signal}: {point} is already on the recorded grid")
            outward = ((risk0 == max(grid_risk) and risk > risk0) or (risk0 == min(grid_risk) and risk < risk0)
                       or (tc0 == max(grid_tc) and tc > tc0) or (tc0 == min(grid_tc) and tc < tc0))
            if not outward:
                raise ValueError(f"{signal}: {point} does not extend outward past a grid edge of the recorded choice")


def _check_evaluation_and_bootstrap(cfg: dict[str, Any], sensitivity: dict[str, Any]) -> None:
    if set(cfg["evaluation"]["statistics"]) != EVALUATION_STATISTICS:
        raise ValueError(f"evaluation.statistics must be {sorted(EVALUATION_STATISTICS)}")
    boot, other = cfg["bootstrap"], sensitivity["bootstrap"]
    for key in BOOTSTRAP_KEYS:
        if boot[key] != other[key]:
            raise ValueError(f"bootstrap.{key} must equal configs/sensitivity.yaml")
    if boot["mean_block_length"] != other["mean_block_lengths"][0]:
        raise ValueError("bootstrap.mean_block_length must be the primary length of configs/sensitivity.yaml")


def _check_checks_and_outputs(cfg: dict[str, Any]) -> None:
    checks, out = cfg["checks"], cfg["outputs"]
    if not 0 < checks["recorded_book_tolerance"] <= checks["recorded_optimised_sharpe_tolerance"] < 1e-3:
        raise ValueError("checks tolerances must be positive, the closed-form book's the tighter, and below 1e-3")
    if not checks["evaluation_needs_matching_tuning_config"]:
        raise ValueError("checks.evaluation_needs_matching_tuning_config must stay on")
    if Path(out["reports_dir"]).parts[:1] != ("reports",) or Path(out["figures_dir"]).parent != Path(out["reports_dir"]):
        raise ValueError("outputs.reports_dir must be under reports/ and figures_dir inside it")
    if out["reports_dir"] in ("reports/portfolio", "reports/attribution", "reports/sensitivity"):
        raise ValueError("outputs.reports_dir must not overwrite earlier results")
    if Path(out["data_dir"]).parent != Path("data/processed"):
        raise ValueError("outputs.data_dir must be in data/processed (gitignored)")


def load_second_round_config(
    path: str | Path | None = None,
    pcfg: dict[str, Any] | None = None,
    fcfg: dict[str, Any] | None = None,
    rcfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load ``configs/second_round.yaml`` and reject anything outside the pre-registered design.

    The other configs default to the paths named here. The recorded choices are read from
    ``recorded.hyperparameters``; no return, panel row or prediction is read.
    """
    with open(path or SECOND_ROUND_CONFIG_PATH) as handle:
        cfg = yaml.safe_load(handle)
    _check_schema(cfg, SCHEMA, "")
    for key, expected in FIXED.items():
        if _get(cfg, key) != expected:
            raise ValueError(f"{key}: only {expected!r} is defined")
    fcfg = fcfg or load_config(REPO_ROOT / cfg["features_config"])
    rcfg = rcfg or load_research_config(REPO_ROOT / cfg["research_config"], fcfg)
    mcfg = load_models_config(REPO_ROOT / cfg["models_config"], rcfg)
    pcfg = pcfg or load_portfolio_config(REPO_ROOT / cfg["portfolio_config"], fcfg, rcfg, mcfg)
    _check_signals(cfg, pcfg)
    _check_band_and_liquid(cfg)
    _check_lambda_points(cfg, pcfg, pd.read_csv(REPO_ROOT / cfg["recorded"]["hyperparameters"]))
    _check_evaluation_and_bootstrap(cfg, load_sensitivity_config(fcfg=fcfg, rcfg=rcfg, pcfg=pcfg))
    _check_checks_and_outputs(cfg)
    return cfg
