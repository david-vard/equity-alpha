import numpy as np
import pytest
import yaml

from src.costs import (
    CostModel,
    apply_participation_cap,
    borrow_cost,
    cost_coefficients,
    max_trade_weight,
    trade_costs,
)
from src.portfolio import PORTFOLIO_CONFIG_PATH

BASE = CostModel.from_config(yaml.safe_load(open(PORTFOLIO_CONFIG_PATH)), trading_days=252)


def test_base_model_comes_from_the_config():
    assert (BASE.commission_bps, BASE.half_spread_bps, BASE.impact_y, BASE.borrow_bps_per_year) == (0.5, 2.0, 1.0, 25.0)
    assert BASE.aum_usd == 100_000_000 and BASE.multiplier == 1.0


def test_one_hand_computed_trade():
    # 1% of a $20m tranche = $200,000 traded in a name with 2% daily vol and $50m ADV
    tc = trade_costs(np.array([0.01]), np.array([0.02]), np.array([50e6]), 20e6, BASE).iloc[0]
    assert tc["traded_usd"] == pytest.approx(200_000)
    assert tc["commission_usd"] == pytest.approx(10.0)          # 0.5 bp
    assert tc["spread_usd"] == pytest.approx(40.0)              # 2 bp
    impact_rate = 1.0 * 0.02 * np.sqrt(200_000 / 50e6)          # 12.65 bp
    assert tc["impact_usd"] == pytest.approx(200_000 * impact_rate)
    assert tc["impact_usd"] == pytest.approx(252.98221, rel=1e-6)
    assert tc["total_usd"] == pytest.approx(302.98221, rel=1e-6)
    assert tc["total_bps"] == pytest.approx(15.1491, rel=1e-4)
    # borrow: $50m short for one day at 25 bp a year
    assert borrow_cost(50e6, BASE) == pytest.approx(50e6 * 0.0025 / 252)


def test_sells_cost_the_same_as_buys_and_no_trade_costs_nothing():
    tc = trade_costs(np.array([0.01, -0.01, 0.0]), np.full(3, 0.02), np.full(3, 50e6), 20e6, BASE)
    assert tc["total_usd"].iloc[0] == pytest.approx(tc["total_usd"].iloc[1])
    assert tc["total_usd"].iloc[2] == 0 and np.isnan(tc["total_bps"].iloc[2])


def test_optimiser_coefficients_price_a_trade_like_the_accounting():
    rng = np.random.default_rng(0)
    dw, sigma, adv = rng.normal(0, 0.01, 50), rng.uniform(0.01, 0.04, 50), rng.uniform(1e7, 1e9, 50)
    for model in (BASE, BASE.scenario(multiplier=2.5, aum_usd=1e9)):
        capital = model.aum_usd / 5
        linear, impact = cost_coefficients(sigma, adv, capital, model)
        from_coefficients = (linear * np.abs(dw) + impact * np.abs(dw) ** 1.5) * capital
        np.testing.assert_allclose(from_coefficients, trade_costs(dw, sigma, adv, capital, model)["total_usd"], rtol=1e-12)


def test_participation_cap_binds_exactly_at_the_adv_fraction():
    max_dw = max_trade_weight(np.array([50e6, 50e6, 50e6, np.nan]), 20e6, 0.05)
    np.testing.assert_allclose(max_dw, [0.125, 0.125, 0.125, 0.0])
    capped = apply_participation_cap(np.array([0.2, -0.2, 0.05, 0.01]), max_dw)
    np.testing.assert_allclose(capped, [0.125, -0.125, 0.05, 0.0])
    traded = np.abs(capped[:3]) * 20e6
    assert np.all(traded <= 0.05 * 50e6 + 1e-6) and traded[0] == pytest.approx(0.05 * 50e6)


@pytest.mark.parametrize("m", [0.0, 0.5, 2.0, 3.0])
def test_cost_multiplier_scales_every_component_linearly(m):
    args = (np.array([0.01, -0.03]), np.array([0.02, 0.03]), np.array([5e7, 2e8]), 2e7)
    base, scaled = trade_costs(*args, BASE), trade_costs(*args, BASE.scenario(multiplier=m))
    for c in ("commission_usd", "spread_usd", "impact_usd", "total_usd"):
        np.testing.assert_allclose(scaled[c], m * base[c], rtol=1e-12)
    np.testing.assert_allclose(scaled["traded_usd"], base["traded_usd"])
    assert borrow_cost(1e6, BASE.scenario(multiplier=m)) == pytest.approx(m * borrow_cost(1e6, BASE))


def test_aum_scenario_scales_linear_costs_and_impact_by_its_square_root():
    dw, sigma, adv = np.array([0.01]), np.array([0.02]), np.array([5e7])
    small, big = BASE.scenario(aum_usd=1e7), BASE.scenario(aum_usd=1e9)
    a = trade_costs(dw, sigma, adv, small.aum_usd / 5, small).iloc[0]
    b = trade_costs(dw, sigma, adv, big.aum_usd / 5, big).iloc[0]
    assert b["commission_usd"] / a["commission_usd"] == pytest.approx(100)
    assert b["impact_usd"] / a["impact_usd"] == pytest.approx(100 ** 1.5)   # $ x rate, rate ~ sqrt($)
    assert b["total_bps"] > a["total_bps"]
