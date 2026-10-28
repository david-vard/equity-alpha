"""Transaction-cost model: commission, half-spread, square-root impact, borrow and the participation cap.

Pure, vectorised functions without state. Weights are per $ of the capital that holds them (one
tranche), so a weight change ``dw`` on capital ``C`` dollars trades ``|dw| * C`` dollars. Every
per-side rate applies to traded notional:

    cost $ = traded $ * m * (commission + half-spread + Y * sigma_daily * sqrt(traded $ / ADV $))

with ``m`` the scenario cost multiplier, which also scales borrow. Divided by ``C`` this is
``linear * |dw| + impact * |dw| ** 1.5`` (``cost_coefficients``); the optimiser's cost term and the
accounting both use those coefficients, so they price a trade identically.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

BPS = 1e-4
COMPONENTS = ("commission", "spread", "impact")


@dataclass(frozen=True)
class CostModel:
    """Cost parameters of one scenario (``multiplier`` and ``aum_usd`` vary across scenarios)."""

    commission_bps: float
    half_spread_bps: float
    impact_y: float
    borrow_bps_per_year: float
    aum_usd: float
    multiplier: float = 1.0
    trading_days: int = 252

    @classmethod
    def from_config(cls, pcfg: dict[str, Any], trading_days: int) -> "CostModel":
        costs = pcfg["costs"]
        return cls(
            commission_bps=float(costs["commission_bps"]),
            half_spread_bps=float(costs["half_spread_bps"]),
            impact_y=float(costs["impact"]["y"]),
            borrow_bps_per_year=float(costs["borrow_bps_per_year"]),
            aum_usd=float(costs["base_aum_usd"]),
            trading_days=trading_days,
        )

    def scenario(self, multiplier: float | None = None, aum_usd: float | None = None) -> "CostModel":
        """The same model with another cost multiplier and / or AUM."""
        return replace(
            self,
            multiplier=self.multiplier if multiplier is None else float(multiplier),
            aum_usd=self.aum_usd if aum_usd is None else float(aum_usd),
        )


def cost_coefficients(
    sigma: np.ndarray, adv_usd: np.ndarray, capital_usd: float, model: CostModel
) -> tuple[np.ndarray, np.ndarray]:
    """``(linear, impact)`` with cost / capital = linear * |dw| + impact * |dw| ** 1.5 per name.

    ``sigma`` is the daily return volatility and ``adv_usd`` the average daily dollar volume.
    """
    sigma, adv = np.asarray(sigma, dtype="float64"), np.asarray(adv_usd, dtype="float64")
    linear = np.full(sigma.shape, model.multiplier * (model.commission_bps + model.half_spread_bps) * BPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        impact = model.multiplier * model.impact_y * sigma * np.sqrt(capital_usd / adv)
    return linear, impact


def trade_costs(
    dw: np.ndarray, sigma: np.ndarray, adv_usd: np.ndarray, capital_usd: float, model: CostModel,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Cost of trading ``dw`` per name: traded dollars, each component and the total in dollars,
    and the total in bp of traded notional (NaN where nothing is traded)."""
    traded = np.abs(np.asarray(dw, dtype="float64")) * capital_usd
    sigma, adv = np.asarray(sigma, dtype="float64"), np.asarray(adv_usd, dtype="float64")
    m = model.multiplier
    with np.errstate(divide="ignore", invalid="ignore"):
        impact_rate = np.where(traded > 0, model.impact_y * sigma * np.sqrt(traded / adv), 0.0)
        out = pd.DataFrame(
            {
                "traded_usd": traded,
                "commission_usd": traded * m * model.commission_bps * BPS,
                "spread_usd": traded * m * model.half_spread_bps * BPS,
                "impact_usd": traded * m * impact_rate,
            },
            index=index,
        )
        out["total_usd"] = out[[f"{c}_usd" for c in COMPONENTS]].sum(axis=1, min_count=1)
        out["total_bps"] = np.where(traded > 0, out["total_usd"] / traded / BPS, np.nan)
    return out


def borrow_cost(short_usd: np.ndarray | float, model: CostModel) -> np.ndarray | float:
    """One day's borrow in dollars: the annual rate accrued daily on short notional (``>= 0``)."""
    return np.asarray(short_usd, dtype="float64") * model.multiplier * model.borrow_bps_per_year * BPS / model.trading_days


def max_trade_weight(adv_usd: np.ndarray, capital_usd: float, max_adv_frac: float) -> np.ndarray:
    """Largest |dw| a name may trade: ``max_adv_frac * ADV $ / capital $`` (0 where ADV is unknown)."""
    adv = np.asarray(adv_usd, dtype="float64")
    return np.where(np.isfinite(adv) & (adv > 0), max_adv_frac * adv / capital_usd, 0.0)


def apply_participation_cap(dw: np.ndarray, max_dw: np.ndarray) -> np.ndarray:
    """``dw`` clipped to ``[-max_dw, max_dw]`` name by name."""
    max_dw = np.asarray(max_dw, dtype="float64")
    return np.clip(np.asarray(dw, dtype="float64"), -max_dw, max_dw)
