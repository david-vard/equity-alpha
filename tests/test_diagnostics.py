import copy

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import (
    COMPLETE,
    DAILY_MISSING,
    INTERIOR_GAP,
    LEFT_PANEL,
    TERMINAL,
    beta_hedged_spread,
    breakeven_costs,
    compound_daily_labels,
    compounding_check,
    cross_sectional_corr,
    decile_medians,
    decomposition_summary,
    dispersion_weighted_ic,
    ic_decomposition,
    rank_by_date,
)
from src.signals import cross_sectional_ic, load_research_config, quantile_buckets, quantile_returns
from src.utils import winsorize_by_date

RCFG = load_research_config()


def wide(values, start="2000-01-03"):
    values = np.asarray(values, dtype="float64")
    return pd.DataFrame(
        values,
        index=pd.bdate_range(start, periods=len(values)),
        columns=[f"T{i}" for i in range(values.shape[1])],
    )


def spread(means: pd.DataFrame) -> pd.Series:
    return means.iloc[:, -1] - means.iloc[:, 0]


def test_compounded_daily_labels_telescope_to_the_multi_day_label():
    rng = np.random.default_rng(0)
    r = 0.02 * rng.standard_normal((40, 5))
    h = 5
    label = np.full_like(r, np.nan)
    for t in range(len(r) - h + 1):
        label[t] = np.prod(1 + r[t : t + h], axis=0) - 1
    compounded, log_sum = compound_daily_labels(wide(r), h)
    ok = ~np.isnan(label)
    assert np.allclose(compounded.to_numpy()[ok], label[ok], atol=1e-12)
    assert np.allclose(np.expm1(log_sum.to_numpy()[ok]), label[ok], atol=1e-12)
    assert compounded.iloc[-(h - 1) :].isna().all().all()  # window runs past the sample


def test_compounding_check_classifies_every_incomplete_window():
    rng = np.random.default_rng(1)
    n, h = 30, 5
    r = 0.01 * rng.standard_normal((n, 4))
    label = np.full_like(r, np.nan)
    for t in range(n - h + 1):
        label[t] = np.prod(1 + r[t : t + h], axis=0) - 1
    present = np.ones_like(r, dtype=bool)
    present[10:, 1] = False  # T1 leaves the index at row 10 (its label still exists before that)
    present[5, 2] = False  # T2 misses one day
    r[10:, 1] = np.nan
    r[5, 2] = np.nan
    label[10:, 1] = np.nan
    label[5, 2] = np.nan
    r[20, 3] = np.nan  # T3 has a row but no daily label on row 20
    terminal = np.zeros_like(r)
    terminal[0, 0] = 1.0
    stored = wide(label).astype("float32")
    table, _ = compounding_check(wide(r).astype("float32"), stored, wide(terminal), wide(present), h, 1e-5)
    table = table.set_index("category")
    assert table.loc[TERMINAL, "n_cells"] == 1
    assert table.loc[LEFT_PANEL, "n_cells"] == 5  # T1 rows 6..9, and T2 row 1 (its window ends on the gap)
    assert table.loc[INTERIOR_GAP, "n_cells"] == 3  # T2 rows 2..4
    assert table.loc[DAILY_MISSING, "n_cells"] == 5  # T3 rows 16..20
    assert table.loc[COMPLETE, "n_above_tol"] == 0
    assert table["n_cells"].sum() == int(np.isfinite(label).sum())
    bad = label.copy()
    bad[3, 3] += 0.01  # a wrong stored label is caught
    table, _ = compounding_check(wide(r), wide(bad), wide(terminal), wide(present), h, 1e-5)
    assert table.set_index("category").loc[COMPLETE, "n_above_tol"] == 1


def test_linear_ic_matches_numpy_on_each_date():
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal((4, 50)), rng.standard_normal((4, 50))
    y[1, :5] = np.nan
    ic = cross_sectional_corr(wide(x), wide(y), 10)
    for i in range(4):
        ok = ~np.isnan(y[i])
        assert ic.iloc[i] == pytest.approx(np.corrcoef(x[i][ok], y[i][ok])[0, 1], abs=1e-12)
    assert cross_sectional_corr(wide(x), wide(y), 51).isna().all()


def dispersion_planted(n_dates=600, n_names=300, h=10, seed=3):
    """Persistent feature; calm days have IC +0.05, one day in ``h`` is 5x as dispersed with IC -0.1."""
    rng = np.random.default_rng(seed)
    f = np.tile(rng.standard_normal(n_names), (n_dates, 1))
    loud = (np.arange(n_dates) % h) == 0
    rho = np.where(loud, -0.1, 0.05)[:, None]
    sigma = np.where(loud, 5.0, 1.0)[:, None]
    r = 0.01 * sigma * (rho * f + np.sqrt(1 - rho**2) * rng.standard_normal((n_dates, n_names)))
    return wide(f), wide(r), loud


def test_decomposition_is_exact_and_finds_the_dispersion_covariance():
    f, r, loud = dispersion_planted()
    parts = ic_decomposition(f, r, 10, 100)
    summary = decomposition_summary(parts)
    assert summary["max_identity_error"] < 1e-10
    assert summary["ic_total_mean"] == pytest.approx(summary["no_covariance"] + summary["covariance_term"], abs=1e-12)
    assert summary["ic_daily_mean"] == pytest.approx(0.9 * 0.05 - 0.1 * 0.1, abs=0.01)
    assert summary["weight_sum_mean"] > 1  # sd of a sum <= sum of sds
    # the loud day dominates every 10-day window, so the multi-day IC is negative
    assert summary["ic_total_mean"] < 0 < summary["ic_daily_mean"]
    assert summary["covariance_term"] < 0
    # independent days with equal dispersion: no covariance, IC grows like sqrt(h)
    rng = np.random.default_rng(4)
    g = np.tile(rng.standard_normal(300), (600, 1))
    q = 0.01 * (0.03 * g + rng.standard_normal((600, 300)))
    calm = decomposition_summary(ic_decomposition(wide(g), wide(q), 10, 100))
    assert abs(calm["covariance_term"]) < 0.01
    assert calm["ic_total_mean"] / calm["ic_daily_mean"] == pytest.approx(np.sqrt(10), rel=0.15)


def test_dispersion_weighting_moves_the_mean_ic_towards_loud_days():
    f, r, loud = dispersion_planted()
    ic = cross_sectional_ic(f, r, 100)
    out = dispersion_weighted_ic(ic, r.std(axis=1))
    assert out["ic_mean_dispersion_weight"] < out["ic_mean_equal_weight"]
    assert out["corr_ic_dispersion"] < 0
    assert out["ic_mean_top_dispersion"] == pytest.approx(-0.1, abs=0.02)


def test_rank_by_date_is_uniform_and_centred():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 40))
    x[0, :4] = np.nan
    ranks = rank_by_date(wide(x)).to_numpy()
    assert np.nanmax(ranks) < 0.5 and np.nanmin(ranks) > -0.5
    assert np.allclose(np.nanmean(ranks, axis=1), 0, atol=1e-12)
    assert np.isnan(ranks[0, :4]).all()


def test_demeaned_labels_leave_the_spread_unchanged():
    rng = np.random.default_rng(6)
    f, y = rng.standard_normal((20, 100)), 0.01 * rng.standard_normal((20, 100))
    y[:, :7] = np.nan
    b = quantile_buckets(wide(f), 10, 50)
    xs = wide(y).sub(wide(y).mean(axis=1), axis=0)
    assert np.allclose(spread(quantile_returns(b, wide(y), 10)), spread(quantile_returns(b, xs, 10)), atol=1e-12)


def test_beta_hedge_removes_a_planted_market_exposure():
    rng = np.random.default_rng(7)
    t, n = 200, 200
    beta = np.tile(rng.uniform(0.5, 1.5, n), (t, 1))
    market = 0.01 + 0.02 * rng.standard_normal(t)
    y = beta * market[:, None] + 0.002 * rng.standard_normal((t, n))
    b = quantile_buckets(wide(beta), 10, 50)
    means = quantile_returns(b, wide(y), 10)
    betas = quantile_returns(b, wide(beta), 10)
    raw = spread(means)
    hedged = beta_hedged_spread(means, betas, pd.Series(market, index=means.index))
    assert raw.mean() > 0.005
    assert abs(hedged.mean()) < 3e-4 and hedged.std() < 0.1 * raw.std()


def test_winsorised_and_median_spreads_ignore_a_planted_outlier():
    rng = np.random.default_rng(8)
    f, y = rng.standard_normal((50, 200)), 0.01 * rng.standard_normal((50, 200))
    b = quantile_buckets(wide(f), 10, 50)
    clean = spread(quantile_returns(b, wide(y), 10)).mean()
    top = int(np.argmax(f[3]))
    y[3, top] = 5.0  # one takeover-style jump in the top decile
    raw = spread(quantile_returns(b, wide(y), 10)).mean()
    wins = spread(quantile_returns(b, winsorize_by_date(wide(y), (0.01, 0.99)), 10)).mean()
    med = spread(decile_medians(b, wide(y), 10)).mean()
    assert raw - clean > 0.004
    assert abs(wins - clean) < 0.0005 and abs(med - clean) < 0.002


def test_rank_spread_follows_the_ic_when_the_mean_follows_the_tail():
    """Top-decile names have a lower typical return but rare huge jumps: mean says +, IC and ranks say -."""
    rng = np.random.default_rng(9)
    t, n = 200, 500
    f = rng.standard_normal((t, n))
    y = -0.3 * f + rng.standard_normal((t, n))
    y += np.where((f > 1.2816) & (rng.uniform(size=(t, n)) < 0.03), 50.0, 0.0)
    b = quantile_buckets(wide(f), 10, 50)
    assert cross_sectional_ic(wide(f), wide(y), 50).mean() < 0
    assert spread(quantile_returns(b, wide(y), 10)).mean() > 0
    assert spread(quantile_returns(b, rank_by_date(wide(y)), 10)).mean() < 0
    assert spread(decile_medians(b, wide(y), 10)).mean() < 0


def test_breakeven_on_a_hand_example():
    idx = pd.bdate_range("2000-01-03", periods=2)
    out = breakeven_costs(pd.Series([0.002, 0.004], index=idx),
                          pd.DataFrame({"D1": [0.5, 0.5], "D10": [0.25, 0.25]}, index=idx))
    # 30 bp per rebalance; buys + sells = 2 * 0.75 = 1.5 of a leg
    assert out["spread_bps"] == pytest.approx(30.0)
    assert out["breakeven_bps_per_side"] == pytest.approx(20.0)
    assert out["breakeven_bps_round_trip"] == pytest.approx(40.0)


def test_runner_smoke_on_a_tiny_synthetic_panel(tmp_path):
    import json

    import yaml

    from src.run_diagnostics import run

    rcfg = copy.deepcopy(RCFG)
    rcfg.update(holdout_start="2021-06-01", min_names_per_date=10, quantiles=5)
    cfg_path = tmp_path / "research.yaml"
    cfg_path.write_text(yaml.safe_dump(rcfg))

    rng = np.random.default_rng(10)
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    tickers = [f"T{i}" for i in range(30)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    daily = 0.01 * rng.standard_normal((len(dates), len(tickers)))
    cols = {}
    for name in RCFG["signals"]:
        cols[name] = rng.standard_normal(len(index)).astype("float32")
        cols[f"z_{name}"] = rng.standard_normal(len(index)).astype("float32")
    for h in RCFG["horizons"]:  # consistent labels: the h-day label compounds the daily ones
        lab = np.full_like(daily, np.nan)
        for t in range(len(dates) - h + 1):
            lab[t] = np.prod(1 + daily[t : t + h], axis=0) - 1
        cols[f"fwd_ret_{h}d"] = lab.ravel().astype("float32")
        with np.errstate(invalid="ignore"):
            xs = lab - lab.mean(axis=1, keepdims=True)  # NaN rows at the end stay NaN
        cols[f"fwd_ret_{h}d_xs"] = xs.ravel().astype("float32")
        cols[f"fwd_terminal_{h}d"] = np.zeros(len(index), dtype="float32")
    panel_file = tmp_path / "panel.parquet"
    pd.DataFrame(cols, index=index).to_parquet(panel_file)

    out = tmp_path / "out"
    tables = run(["--config", str(cfg_path), "--panel", str(panel_file), "--out", str(out),
                  "--reference", str(tmp_path / "none"), "--allow-dirty-config"])
    meta = json.loads((out / "run_metadata.json").read_text())
    assert pd.Timestamp(meta["last_research_date"]) < pd.Timestamp("2021-06-01")
    assert meta["compounding_mismatches"] == 0 and meta["reference_check"].startswith("skipped")
    assert set(tables["decile_spreads_robust"]["variant"]) >= {"raw", "beta_hedged", "rank", "median"}
    be = tables["breakeven_costs"]
    assert set(be["sample"]) == {"full", "first_half", "second_half"}
    assert np.allclose(be["breakeven_bps_round_trip"], 2 * be["breakeven_bps_per_side"], equal_nan=True)
    assert len(tables["ic_decomposition"]) == 4 * 2
    assert (out / "figures" / "risk_deciles_h21.png").exists()
