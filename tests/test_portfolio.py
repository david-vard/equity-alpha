import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.covariance import LedoitWolf

from src.costs import CostModel, cost_coefficients, max_trade_weight
from src.portfolio import (
    PORTFOLIO_CONFIG_PATH,
    OptimiserSpec,
    check_constraints,
    decile_ls,
    fallback_weights,
    grinold_alpha,
    ledoit_wolf_risk,
    neutral_volscaled,
    optimised,
    zscore,
)
from src.signals import quantile_buckets

PCFG = yaml.safe_load(open(PORTFOLIO_CONFIG_PATH))
COSTS = CostModel.from_config(PCFG, 252)
SPEC = OptimiserSpec.from_config(PCFG, horizon=5, lambda_risk=10.0, lambda_tc=1.0)


# --------------------------------------------------------------------------- decile_ls


def test_decile_ls_longs_the_top_and_shorts_the_bottom_with_ties_and_nan():
    score = np.array([5, np.nan, 1, 1, 3, 9, 9, 7, 2, 4, 6, 8], dtype=float)
    w = decile_ls(score, quantiles=5, min_names=5)
    # 11 valid scores: the two tied 1s share the bottom bucket, the two tied 9s the top one
    expected = np.zeros(12)
    expected[[5, 6]] = 0.5
    expected[[2, 3]] = -0.5
    np.testing.assert_allclose(w, expected)
    assert w[1] == 0 and w.sum() == pytest.approx(0) and np.abs(w).sum() == pytest.approx(2)


def test_decile_ls_matches_the_research_buckets_and_needs_min_names():
    rng = np.random.default_rng(3)
    score = np.round(rng.normal(size=300), 1)   # many ties
    score[rng.choice(300, 40, replace=False)] = np.nan
    w = decile_ls(score, 10, 100)
    b = quantile_buckets(pd.DataFrame([score]), 10, 100).to_numpy()[0]
    assert set(np.flatnonzero(w > 0)) == set(np.flatnonzero(b == 9))
    assert set(np.flatnonzero(w < 0)) == set(np.flatnonzero(b == 0))
    assert w[w > 0].sum() == pytest.approx(1) and w[w < 0].sum() == pytest.approx(-1)
    assert not decile_ls(score, 10, 1000).any()


# --------------------------------------------------------------------------- risk model and neutral_volscaled


def test_ledoit_wolf_matches_sklearn_and_flags_short_histories():
    rng = np.random.default_rng(1)
    window = rng.normal(0, 0.02, (252, 30)) + rng.normal(0, 0.01, (252, 1))
    risk, covered = ledoit_wolf_risk(window, 0.8)
    np.testing.assert_allclose(risk.covariance(), LedoitWolf().fit(window).covariance_, rtol=1e-10, atol=1e-14)
    assert covered.all()
    w = rng.normal(size=30)
    assert risk.variance(w) == pytest.approx(w @ risk.covariance() @ w)
    window[:60, 0] = np.nan     # 76% observed: below 80%
    window[:40, 1] = np.nan     # 84% observed
    _, covered = ledoit_wolf_risk(window, 0.8)
    assert not covered[0] and covered[1]


def volscaled_inputs(n=80, seed=2):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    vol = rng.uniform(0.01, 0.04, n)
    beta = rng.normal(1, 0.3, n)
    window = rng.normal(0, 1, (252, n)) * vol + np.outer(rng.normal(0, 0.01, 252), beta)
    risk, _ = ledoit_wolf_risk(window, 0.8)
    return score, vol, beta, risk


def test_neutral_volscaled_hits_the_vol_target_ex_ante_and_is_dollar_and_beta_neutral():
    score, vol, beta, risk = volscaled_inputs()
    score[3] = np.nan
    w = neutral_volscaled(score, vol, beta, risk, 0.10, 2.0, 10, 252)
    assert np.sqrt(risk.variance(w) * 252) == pytest.approx(0.10, rel=1e-9)
    assert w.sum() == pytest.approx(0, abs=1e-12) and beta @ w == pytest.approx(0, abs=1e-12)
    assert np.abs(w).sum() < 2 and w[3] == 0


def test_neutral_volscaled_caps_gross_when_the_vol_target_needs_more():
    score, vol, beta, risk = volscaled_inputs()
    w = neutral_volscaled(score, vol, beta, risk, 0.90, 2.0, 10, 252)
    assert np.abs(w).sum() == pytest.approx(2.0)
    assert np.sqrt(risk.variance(w) * 252) < 0.90
    assert w.sum() == pytest.approx(0, abs=1e-12) and beta @ w == pytest.approx(0, abs=1e-12)


def test_neutral_volscaled_weights_follow_score_over_vol_before_the_projection():
    score, vol, beta, risk = volscaled_inputs()
    beta = np.ones_like(beta)   # projection then only demeans
    w = neutral_volscaled(score, vol, beta, risk, 0.10, 2.0, 10, 252)
    raw = zscore(score) / vol
    np.testing.assert_allclose(w / np.abs(w).sum(), (raw - raw.mean()) / np.abs(raw - raw.mean()).sum(), rtol=1e-10)


# --------------------------------------------------------------------------- optimised


def optimiser_case(seed, n=12, current_scale=0.01, adv=5e8, capital=2e7):
    rng = np.random.default_rng(seed)
    score, vol = rng.normal(size=n), rng.uniform(0.01, 0.03, n)
    beta = rng.normal(1, 0.3, n)
    window = rng.normal(0, 1, (60, n)) * vol + np.outer(rng.normal(0, 0.01, 60), beta)
    risk, covered = ledoit_wolf_risk(window, 0.8)
    current = rng.normal(0, current_scale, n)
    current -= current.mean()
    adv_usd = np.full(n, adv) * rng.uniform(0.5, 2, n)
    free = covered.copy()
    alpha = np.where(free, grinold_alpha(score, vol, 0.02, 5), 0.0)
    linear, impact = cost_coefficients(vol, adv_usd, capital, COSTS)
    max_dw = max_trade_weight(adv_usd, capital, SPEC.max_trade_adv_frac)
    return alpha, beta, current, max_dw, free, risk, linear, impact


@pytest.mark.parametrize("seed", range(4))
def test_optimised_output_satisfies_every_constraint(seed):
    alpha, beta, current, max_dw, free, risk, linear, impact = optimiser_case(seed)
    target = optimised(alpha, beta, current, max_dw, free, risk, linear, impact, SPEC)
    assert target.fallback is None
    assert all(check_constraints(target.weights, current, beta, max_dw, free, SPEC).values())
    assert alpha @ target.weights > 0


def test_optimised_respects_a_binding_adv_cap_and_sells_names_that_are_not_free():
    alpha, beta, current, max_dw, free, risk, linear, impact = optimiser_case(5, adv=2e6)  # 5% ADV ~ 0.5% of capital
    free[0] = False
    current[0] = 0.01
    current -= np.r_[0, np.full(len(current) - 1, current[0] / (len(current) - 1))]
    target = optimised(alpha, beta, current, max_dw, free, risk, linear, impact, SPEC)
    w = target.weights
    assert w[0] == 0
    if target.fallback != "partial":
        assert all(check_constraints(w, current, beta, max_dw, free, SPEC).values())
    assert np.all(np.abs(w - current)[free] <= max_dw[free] + 1e-6)


def test_solver_failure_falls_back_to_the_held_weights_inside_the_constraints():
    alpha, beta, current, max_dw, free, risk, linear, impact = optimiser_case(6, current_scale=0.03)
    broken = OptimiserSpec(**{**SPEC.__dict__, "solver": "NOT_A_SOLVER"})
    target = optimised(alpha, beta, current, max_dw, free, risk, linear, impact, broken)
    assert target.fallback == "hold" and target.status.startswith("error")
    assert all(check_constraints(target.weights, current, beta, max_dw, free, broken).values())
    assert np.abs(target.weights).max() <= broken.max_weight + 1e-12


def test_fallback_goes_flat_then_partial_when_the_adv_cap_forbids_holding():
    n = 6
    beta = np.ones(n)
    free = np.ones(n, dtype=bool)
    current = np.array([0.05, -0.05, 0.04, -0.04, 0.0, 0.0])   # above the 2% cap: holding needs a big trade
    roomy = np.full(n, 0.1)
    assert fallback_weights(current, beta, roomy, free, SPEC).fallback == "hold"
    tight = np.full(n, 0.01)                                     # cannot reach the cap or flat in one day
    target = fallback_weights(current, beta, tight, free, SPEC)
    assert target.fallback == "partial" and target.status == "fallback_constraints_violated"
    np.testing.assert_allclose(target.weights, [0.04, -0.04, 0.03, -0.03, 0.0, 0.0])
    assert np.all(np.abs(target.weights - current) <= tight + 1e-12)
    # one long name left after drift: neutralising it would trade the other names beyond their cap,
    # selling it alone is allowed
    lone = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    caps = np.array([0.02, 0.001, 0.001, 0.001, 0.001, 0.001])
    target = fallback_weights(lone, beta, caps, free, SPEC)
    assert target.fallback == "flat" and not target.weights.any()
    assert all(check_constraints(target.weights, lone, beta, caps, free, SPEC).values())


def test_grinold_alpha_is_ic_times_horizon_vol_times_z():
    score, vol = np.array([1.0, 2.0, 3.0, np.nan]), np.array([0.01, 0.02, 0.03, 0.02])
    alpha = grinold_alpha(score, vol, 0.02, 5)
    np.testing.assert_allclose(alpha[:3], 0.02 * vol[:3] * np.sqrt(5) * np.array([-1.0, 0.0, 1.0]))
    assert np.isnan(alpha[3])
