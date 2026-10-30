"""Second round: the ``configs/second_round.yaml`` loader, the variants and the selection rules.

The design is fixed in the config before any variant is simulated. The loader is strict: an unknown
or missing key, a wrong type, a method the design does not define, a grid without its recorded point,
or a lambda point that doesn't extend outward from the recorded choice is an error, never silently
ignored. Every evaluation-window result of this round is a second look.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.costs import CostModel
from src.models import load_models_config
from src.portfolio import (
    Target,
    _check_schema,
    _get,
    _is_type,
    _positive_numbers,
    _window,
    ledoit_wolf_risk,
    load_portfolio_config,
    volscaled_builder,
)
from src.sensitivity import load_sensitivity_config
from src.signals import load_research_config
from src.utils import REPO_ROOT, load_config

SECOND_ROUND_CONFIG_PATH = REPO_ROOT / "configs" / "second_round.yaml"
BAND_CONSTRUCTION = "neutral_volscaled"
LIQUID_CONSTRUCTIONS = ("neutral_volscaled", "optimised")
EVALUATION_STATISTICS = {"net_sharpe", "net_nw_t", "net_return_ann", "gross_sharpe", "annual_turnover",
                         "breakeven_cost_multiplier"}
BOOTSTRAP_KEYS = ("method", "n_resamples", "seed", "confidence", "interval")
MIN_TRADED = 3  # a dollar- and beta-neutral repair needs at least this many traded names

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


# --------------------------------------------------------------------------- variants


@dataclass(frozen=True)
class Variant:
    """One grid point. ``recorded`` marks the first-round book at that point."""

    family: str
    signal: str
    construction: str
    order: int
    recorded: bool = False
    band: float | None = None
    fraction: float | None = None
    lambdas: tuple[float, float] | None = None

    @property
    def name(self) -> str:
        if self.family == "no_trade_band":
            point = f"{self.band:g}"
        elif self.family == "liquid_universe":
            point = f"{self.fraction:g}"
        else:
            point = f"{self.lambdas[0]:g}_{self.lambdas[1]:g}"
        return f"{self.family}__{self.signal}__{self.construction}__{point}"

    def row(self) -> dict[str, Any]:
        risk, tc = self.lambdas if self.lambdas else (np.nan, np.nan)
        return {"variant": self.name, "family": self.family, "signal": self.signal, "construction": self.construction,
                "grid_order": self.order, "recorded": self.recorded, "band": self.band, "fraction": self.fraction,
                "lambda_risk": risk, "lambda_tc": tc}


def enumerate_variants(cfg: dict[str, Any], lambda_grid: pd.DataFrame) -> list[Variant]:
    """Every grid point in config order. Recorded lambda points are read, never rerun."""
    out: list[Variant] = []
    band, liquid = cfg["no_trade_band"], cfg["liquid_universe"]
    signals = [cfg["signals"]["primary"], cfg["signals"]["benchmark"]]
    for signal in signals:
        for order, value in enumerate(band["bands"]):
            out.append(Variant("no_trade_band", signal, band["construction"], order, value == 0, band=float(value)))
    for signal in signals:
        for construction in liquid["constructions"]:
            for order, value in enumerate(liquid["fractions"]):
                out.append(Variant("liquid_universe", signal, construction, order, value == 1, fraction=float(value)))
    for signal in signals:
        rows = lambda_grid[lambda_grid["signal"] == signal].sort_values("grid_order")
        points = [(float(r), float(c), True) for r, c in zip(rows["lambda_risk"], rows["lambda_tc"])]
        points += [(float(r), float(c), False) for r, c in cfg["wider_lambda_grid"]["points"][signal]["added"]]
        for order, (risk, tc, recorded) in enumerate(points):
            out.append(Variant("wider_lambda_grid", signal, cfg["wider_lambda_grid"]["construction"], order, recorded,
                               lambdas=(risk, tc)))
    return out


def liquid_scores(score: np.ndarray, adv: np.ndarray, fraction: float) -> np.ndarray:
    """Scores of the top ``fraction`` of each date's scored names by ``adv``; the rest are unscored.

    Each row uses only that date's values. A fraction of 1 returns the scores unchanged, so a name
    without a liquidity value stays scored, as in the recorded book.
    """
    if fraction == 1:
        return score
    out = np.full(score.shape, np.nan)
    for row in range(score.shape[0]):
        ok = np.flatnonzero(np.isfinite(score[row]) & np.isfinite(adv[row]))
        if len(ok):
            top = ok[np.argsort(-adv[row, ok], kind="stable")[: math.ceil(fraction * len(ok))]]
            out[row, top] = score[row, top]
    return out


def band_trade(current: np.ndarray, target: np.ndarray, band: float, beta: np.ndarray) -> np.ndarray | None:
    """Weights after a no-trade band of ``band`` times the mean |target weight|, or None to hold.

    A name moves only if its gap to target exceeds the band, and then only to the band's edge. A
    name the target doesn't hold is sold in full. Zero dollars and beta are restored by the
    least-change move of the names still traded towards a target. With fewer than ``MIN_TRADED`` of
    them the tranche holds. A band of 0 is the target itself.
    """
    held = target != 0
    if band == 0 or not held.any():
        return target.copy()
    gap = target - current
    trade = np.sign(gap) * np.clip(np.abs(gap) - band * np.abs(target[held]).mean(), 0.0, None)
    trade[~held] = -current[~held]
    traded = (trade != 0) & held
    if traded.sum() < MIN_TRADED:
        return None
    w = current + trade
    b = np.nan_to_num(beta, nan=0.0)
    excess = np.array([w.sum(), b @ w])
    move, *_ = np.linalg.lstsq(np.column_stack([np.ones(traded.sum()), b[traded]]).T, -excess, rcond=None)
    w[traded] += move
    return w


def scale_down(w: np.ndarray, ann_vol: float, target_vol_annual: float, max_gross: float) -> tuple[np.ndarray, bool]:
    """``w`` scaled down if its ex-ante vol or gross exceeds its limit; it is never scaled up."""
    gross = float(np.abs(w).sum())
    scale = min(1.0, target_vol_annual / ann_vol if ann_vol > 0 else 1.0, max_gross / gross if gross > 0 else 1.0)
    return (w * scale, True) if scale < 1 - 1e-12 else (w, False)


def band_builder(pcfg: dict[str, Any], band: float, min_names: int, trading_days: int, counts: Counter) -> Any:
    """``neutral_volscaled`` with a no-trade band. ``counts`` receives ``band_hold`` and ``band_scaled``."""
    base = volscaled_builder(pcfg, min_names, trading_days)
    spec, risk_cfg = pcfg["constructions"]["neutral_volscaled"], pcfg["risk_model"]

    def build(data: Any, s: int, current: np.ndarray, capital_usd: float, costs: CostModel) -> Target:
        target = base(data, s, current, capital_usd, costs).weights
        if band == 0 or not (target != 0).any():
            return Target(target)
        w = band_trade(current, target, band, data.beta[s])
        if w is None:
            counts["band_hold"] += 1
            return Target(current.copy(), "hold", "band_hold")
        names = np.flatnonzero(np.isfinite(data.score[s]))
        risk, _ = ledoit_wolf_risk(_window(data, s, names, risk_cfg["window"]), risk_cfg["min_obs_frac"])
        ann_vol = math.sqrt(max(risk.variance(w[names]), 0.0) * trading_days)
        w, scaled = scale_down(w, ann_vol, spec["target_vol_annual"], spec["max_gross"])
        counts["band_scaled"] += int(scaled)
        return Target(w)

    return build


# --------------------------------------------------------------------------- selection


def select_variants(grid: pd.DataFrame) -> pd.DataFrame:
    """One row per family, signal and construction: the highest tuning net Sharpe, ties to the first.

    ``on_grid_edge`` flags a choice at an extreme of its grid. No grid is extended because of it.
    """
    grid = grid.sort_values(["family", "signal", "construction", "grid_order"], kind="stable")
    chosen = []
    for _, group in grid.groupby(["family", "signal", "construction"], sort=False):
        row = group.loc[group["net_sharpe"].fillna(-np.inf).idxmax()].copy()
        edge = False
        for column in ("band", "fraction", "lambda_risk", "lambda_tc"):
            values = group[column].dropna()
            edge = edge or (len(values) > 0 and row[column] in (values.min(), values.max()))
        row["on_grid_edge"] = bool(edge)
        chosen.append(row)
    return pd.DataFrame(chosen).reset_index(drop=True)


def holdout_candidate(selected: pd.DataFrame, signal: str, headline_tuning_net_sharpe: float) -> pd.DataFrame:
    """At most one new book: the chosen variant of ``signal`` with the highest tuning net Sharpe, if it
    beats the headline's. A chosen recorded point is the first-round book, so it is not a candidate."""
    rows = selected[(selected["signal"] == signal) & ~selected["recorded"].astype(bool)]
    rows = rows[rows["net_sharpe"] > headline_tuning_net_sharpe]
    return rows.loc[[rows["net_sharpe"].idxmax()]].reset_index(drop=True) if len(rows) else rows.reset_index(drop=True)
