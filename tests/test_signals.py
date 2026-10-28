import copy

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.metrics import (
    adjust_pvalues,
    newey_west_lag,
    newey_west_tstat,
    non_overlapping_positions,
)
from src.signals import (
    Grid,
    add_multiple_testing,
    cross_sectional_ic,
    decile_summary,
    embargo_days,
    exclude_terminal,
    fama_macbeth,
    fm_summary,
    ic_decay,
    ic_summary,
    last_research_date,
    load_research_config,
    load_research_panel,
    long_short_stats,
    quantile_buckets,
    quantile_returns,
    turnover,
)
from src.utils import load_config

FCFG = load_config()
RCFG = load_research_config()


def wide(values, start="2000-01-03"):
    values = np.asarray(values, dtype="float32")
    return pd.DataFrame(
        values,
        index=pd.bdate_range(start, periods=len(values)),
        columns=[f"T{i}" for i in range(values.shape[1])],
    )


def planted(n_dates=300, n_names=150, rho=0.1, seed=0):
    """Feature and label with a known cross-sectional correlation ``rho`` on every date."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n_dates, n_names))
    y = rho * f + np.sqrt(1 - rho**2) * rng.standard_normal((n_dates, n_names))
    return wide(f), wide(y)


def test_planted_ic_is_recovered_and_noise_is_zero():
    f, y = planted(rho=0.1)
    ic = cross_sectional_ic(f, y, 100)
    assert abs(ic.mean() - 0.1) < 0.01
    f0, y0 = planted(rho=0.0, seed=1)
    assert abs(cross_sectional_ic(f0, y0, 100).mean()) < 0.01


def test_false_positive_rate_is_near_nominal():
    hits = 0
    seeds = 300
    for seed in range(seeds):
        f, y = planted(n_dates=250, n_names=100, rho=0.0, seed=seed)
        ic = cross_sectional_ic(f, y, 50).to_numpy()
        t = newey_west_tstat(ic, newey_west_lag(len(ic), 1))["t"]
        hits += abs(t) > 1.96
    assert 0.02 <= hits / seeds <= 0.10


def test_ic_matches_scipy_spearman_with_ties_and_nans():
    rng = np.random.default_rng(3)
    f = rng.integers(0, 6, size=(5, 40)).astype("float64")  # many ties
    y = f * 0.3 + rng.standard_normal((5, 40))
    y[:, 2::7] = np.nan
    f[1, 3] = np.nan
    f[3, 10:15] = np.nan
    y[4, :] = np.round(y[4, :])  # ties in the label too
    ic = cross_sectional_ic(wide(f), wide(y), 5)
    for i in range(5):
        ok = ~np.isnan(f[i]) & ~np.isnan(y[i])
        assert ic.iloc[i] == pytest.approx(stats.spearmanr(f[i][ok], y[i][ok])[0], abs=1e-6)


def test_ic_needs_min_names():
    f, y = planted(n_dates=3, n_names=20)
    assert cross_sectional_ic(f, y, 21).isna().all()
    assert cross_sectional_ic(f, y, 20).notna().all()


def test_newey_west_matches_statsmodels_hac():
    rng = np.random.default_rng(4)
    e = rng.standard_normal(600)
    x = 0.05 + e + 0.6 * np.roll(e, 1)  # autocorrelated
    for lags in (1, 5, 21):
        ours = newey_west_tstat(x, lags)
        res = sm.OLS(x, np.ones(len(x))).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
        )
        assert ours["t"] == pytest.approx(float(res.tvalues[0]), rel=1e-9)
        assert ours["se"] == pytest.approx(float(res.bse[0]), rel=1e-9)


@pytest.mark.parametrize("method,sm_method", [("holm", "holm"), ("bh", "fdr_bh"), ("by", "fdr_by")])
def test_pvalue_adjustment_matches_statsmodels(method, sm_method):
    p = np.random.default_rng(5).uniform(size=40) ** 3
    p[3] = np.nan
    ours = adjust_pvalues(p, method)
    assert np.isnan(ours[3])
    ok = ~np.isnan(p)
    assert ours[ok] == pytest.approx(multipletests(p[ok], method=sm_method)[1])


def test_effective_family_excludes_rank_duplicates():
    table = pd.DataFrame(
        {"feature": ["a", "b", "log_dollar_volume_21d"], "p_value": [0.01, 0.02, 0.02]}
    )
    out = add_multiple_testing(table, "p_value", ["holm"], 0.05)
    assert out["p_value_holm"].iloc[0] == pytest.approx(0.03)
    assert out["p_value_holm_eff"].iloc[0] == pytest.approx(0.02)
    assert np.isnan(out["p_value_holm_eff"].iloc[2]) and not out["reject_holm_eff"].iloc[2]
    assert list(out["reject_holm"]) == [True, True, True]  # adjusted 0.03, 0.04, 0.04 < 0.05


def test_deciles_are_assigned_by_rank_with_ties_and_nans():
    row = np.arange(1, 21, dtype="float32")[None, :]
    b = quantile_buckets(wide(row), 10, 5).to_numpy()[0]
    assert list(b) == [i // 2 for i in range(20)]
    tied = np.array([[1, 1, 1, 1, 5, 6, 7, 8, 9, 10, np.nan, np.nan]], dtype="float32")
    bt = quantile_buckets(wide(tied), 5, 5).to_numpy()[0]
    assert len(set(bt[:4])) == 1  # tied values share a bucket
    assert np.isnan(bt[10:]).all() and np.nanmax(bt) == 4
    assert np.isnan(quantile_buckets(wide(tied), 5, 11).to_numpy()).all()  # too few names


def test_long_short_sign_flips_with_the_feature_sign():
    f, y = planted(n_dates=60, n_names=100, rho=0.2)
    d_pos = quantile_returns(quantile_buckets(f, 10, 50), y, 10)
    d_neg = quantile_returns(quantile_buckets(-f, 10, 50), y, 10)
    pos = long_short_stats(d_pos, 1, 1, RCFG)
    neg = long_short_stats(d_neg, 1, -1, RCFG)
    assert pos["gross_raw_mean"] > 0 > neg["gross_raw_mean"]
    assert pos["gross_raw_mean"] == pytest.approx(-neg["gross_raw_mean"], rel=1e-5)
    assert pos["gross_oriented_mean"] == pytest.approx(neg["gross_oriented_mean"], rel=1e-5)  # orientation undoes the flip
    assert np.isnan(long_short_stats(d_pos, 1, 0, RCFG)["gross_oriented_mean"])
    assert decile_summary(d_pos, 1)["monotonicity"] > 0.8


def test_non_overlapping_sampling_picks_every_hth_date():
    assert list(non_overlapping_positions(12, 5)) == [0, 5, 10]
    assert list(non_overlapping_positions(12, 5, 2)) == [2, 7]
    ic = pd.Series(np.arange(12.0), index=pd.bdate_range("2000-01-03", periods=12))
    row = ic_summary(ic, 5, 1, RCFG)
    assert row["no_n"] == 3 and row["no_ic_mean"] == pytest.approx(5.0)
    assert row["no_ic_mean_all_offsets"] == pytest.approx(ic.mean())


def test_long_short_uses_only_non_overlapping_dates():
    idx = pd.bdate_range("2000-01-03", periods=11)
    d = pd.DataFrame({1: 0.0, 2: np.arange(11.0)}, index=idx)
    out = long_short_stats(d, 5, 1, RCFG)
    assert out["n_periods"] == 3 and out["gross_raw_mean"] == pytest.approx(5.0)


def test_turnover_on_a_hand_built_example():
    # 2 names per leg of 4 names; q=2 -> bottom bucket 0, top bucket 1
    b = pd.DataFrame(
        [[0, 0, 1, 1], [0, 1, 0, 1], [np.nan, 1, 0, 1], [1, 1, 0, 0]],
        columns=list("ABCD"),
        index=pd.bdate_range("2000-01-03", periods=4),
        dtype="float32",
    )
    t = turnover(b, 2, 1)
    assert list(t["D1"]) == pytest.approx([0.5, 0.5, 0.5])  # D1: AB -> AC -> C -> CD
    assert list(t["D2"]) == pytest.approx([0.5, 0.0, 0.5])  # D2: CD -> BD -> BD -> AB


def test_fama_macbeth_recovers_a_planted_premium():
    rng = np.random.default_rng(6)
    t, n = 300, 200
    x1, x2 = rng.standard_normal((t, n)), rng.standard_normal((t, n))
    x2 = 0.5 * x1 + np.sqrt(0.75) * x2
    y = 0.002 * x1 - 0.001 * x2 + 0.02 * rng.standard_normal((t, n))
    x1[5, :10] = np.nan  # listwise deletion
    res = fama_macbeth({"a": wide(x1), "b": wide(x2)}, wide(y), 100)
    summary = fm_summary(res, 1).set_index("regressor")
    assert summary.loc["a", "coef_mean"] == pytest.approx(0.002, abs=3e-4)
    assert summary.loc["b", "coef_mean"] == pytest.approx(-0.001, abs=3e-4)
    assert summary.loc["a", "nw_t"] > 5 and abs(summary.loc["const", "nw_t"]) < 4
    assert res.n.iloc[5] == n - 10
    assert summary.loc["a", "avg_vif"] == pytest.approx(1 / (1 - 0.25), rel=0.1)
    uni = fama_macbeth({"a": wide(x1)}, wide(y), 100)
    d = 10  # one date against statsmodels
    ok = ~np.isnan(x1[d])
    ref = sm.OLS(y[d][ok], sm.add_constant(x1[d][ok])).fit()
    assert uni.coefs["a"].iloc[d] == pytest.approx(ref.params[1], rel=1e-3)
    assert uni.r2.iloc[d] == pytest.approx(ref.rsquared, rel=1e-3)


def test_decay_of_a_signal_built_to_decay_is_monotone():
    rng = np.random.default_rng(7)
    t, n, K = 400, 300, 10
    f = rng.standard_normal((t, n))
    r = rng.standard_normal((t, n + 0))
    # return on day s carries the feature from s-k with weight 0.9**k (k = 0.. K), so IC decays
    ret = rng.standard_normal((t, n))
    for k in range(K + 1):
        ret[k:] += 0.12 * 0.75**k * f[: t - k]
    res = ic_decay(wide(f), wide(ret), K + 1, 100, RCFG)
    ic = res["ic_mean"].to_numpy()
    assert (np.diff(ic) < 0).all()
    assert res["cum_ic"].is_monotonic_increasing
    assert 0.5 < res.attrs["half_life"] < 4
    noise = ic_decay(wide(f), wide(r), K + 1, 100, RCFG)
    assert np.isnan(noise.attrs["half_life"]) or abs(noise["nw_t"].iloc[0]) >= 2


def test_excluding_terminal_rows_changes_only_those_dates():
    f, y = planted(n_dates=30, n_names=120, rho=0.1)
    flag = pd.DataFrame(0.0, index=y.index, columns=y.columns)
    flag.iloc[[4, 9], 7] = 1.0
    full = cross_sectional_ic(f, y, 50)
    cut = cross_sectional_ic(f, exclude_terminal(y, flag), 50)
    changed = ~np.isclose(full, cut)
    assert list(np.nonzero(changed)[0]) == [4, 9]
    untouched = exclude_terminal(y, flag).to_numpy() == y.to_numpy()
    assert (~untouched & ~np.isnan(y.to_numpy())).sum() == 2


def make_panel_file(tmp_path, n_dates=400, n_names=3):
    dates = pd.bdate_range("2019-01-01", periods=n_dates)
    index = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(n_names)]], names=["date", "ticker"])
    panel = pd.DataFrame(
        {"ret_1d": np.arange(len(index), dtype="float64"), "fwd_ret_1d": 1.0}, index=index
    )
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path)
    return path, dates


def test_loader_never_returns_holdout_or_embargoed_rows(tmp_path):
    path, dates = make_panel_file(tmp_path)
    rcfg = copy.deepcopy(RCFG)
    rcfg["holdout_start"] = str(dates[300].date())
    panel = load_research_panel(["ret_1d"], panel_file=path, rcfg=rcfg, features_cfg=FCFG)
    embargo = FCFG["labels"]["lag"] + max(max(rcfg["horizons"]), rcfg["decay"]["n_days"])
    assert embargo == 22
    last = panel.index.get_level_values("date").max()
    assert last == dates[300 - embargo - 1]
    assert last + pd.offsets.BDay(embargo) < pd.Timestamp(rcfg["holdout_start"])
    assert list(panel.columns) == ["ret_1d"] and panel["ret_1d"].dtype == np.float32
    assert last_research_date(dates, rcfg, FCFG) == last


def test_config_validation_rejects_bad_priors_and_representatives(tmp_path):
    import yaml

    def write(mutate):
        cfg = yaml.safe_load(open("configs/research.yaml"))
        mutate(cfg)
        path = tmp_path / "r.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    with pytest.raises(ValueError):
        load_research_config(write(lambda c: c["signals"]["ret_1d"].update(expected_sign=2)))
    with pytest.raises(ValueError):
        load_research_config(
            write(lambda c: c["fama_macbeth"]["multivariate_representatives"].update(risk="ret_1d"))
        )
    with pytest.raises(ValueError):
        load_research_config(write(lambda c: c.update(horizons=[1, 5])))
    with pytest.raises(ValueError):  # method choices fixed in code must not be silently ignored
        load_research_config(write(lambda c: c["newey_west"].update(lag_rule="andrews")))
    with pytest.raises(ValueError):
        load_research_config(write(lambda c: c["multiple_testing"].update(alpha=5)))
    assert set(RCFG["signals"]) == {
        "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d", "mom_12_1",
        "vol_21d", "vol_63d", "beta_252d", "dollar_volume_21d", "log_dollar_volume_21d",
        "dist_high_252d", "overnight_ret", "intraday_ret", "trend_50d", "trend_200d",
    }


def test_embargo_follows_the_longest_horizon_or_decay_window():
    assert embargo_days(RCFG, FCFG) == 22
    longer = copy.deepcopy(RCFG)
    longer["decay"]["n_days"] = 30
    assert embargo_days(longer, FCFG) == 31
    assert RCFG["signals"]["ret_21d"] == {"family": "reversal", "expected_sign": 0}


def test_grid_round_trips_a_long_series():
    index = pd.MultiIndex.from_tuples(
        [("2000-01-04", "B"), ("2000-01-03", "A"), ("2000-01-04", "A")], names=["date", "ticker"]
    )
    index = index.set_levels(pd.to_datetime(index.levels[0]), level=0)
    grid = Grid.from_index(index)
    w = grid.wide([1.0, 2.0, 3.0])
    assert w.loc["2000-01-04", "B"] == 1 and w.loc["2000-01-03", "A"] == 2 and np.isnan(w.loc["2000-01-03", "B"])


def test_runner_smoke_on_a_tiny_synthetic_panel(tmp_path):
    import json

    import yaml

    from src.run_signal_research import run

    rcfg = copy.deepcopy(RCFG)
    rcfg.update(holdout_start="2021-06-01", min_names_per_date=10, quantiles=5)
    rcfg["decay"]["n_days"] = 5
    cfg_path = tmp_path / "research.yaml"
    cfg_path.write_text(yaml.safe_dump(rcfg))

    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    index = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(30)]], names=["date", "ticker"])
    n = len(index)
    cols = {}
    for name in RCFG["signals"]:
        cols[name] = rng.standard_normal(n).astype("float32")
        cols[f"z_{name}"] = rng.standard_normal(n).astype("float32")
    for h in RCFG["horizons"]:
        cols[f"fwd_ret_{h}d"] = (0.01 * rng.standard_normal(n)).astype("float32")
        cols[f"fwd_ret_{h}d_xs"] = (0.01 * rng.standard_normal(n)).astype("float32")
        cols[f"fwd_terminal_{h}d"] = np.zeros(n, dtype="float32")
    panel_file = tmp_path / "panel.parquet"
    pd.DataFrame(cols, index=index).to_parquet(panel_file)

    out = tmp_path / "out"
    tables = run(["--config", str(cfg_path), "--panel", str(panel_file), "--out", str(out),
                  "--allow-dirty-config"])
    meta = json.loads((out / "run_metadata.json").read_text())
    assert pd.Timestamp(meta["last_research_date"]) < pd.Timestamp("2021-06-01")
    assert len(tables["ic_summary"]) == 17 * 3
    assert {"p_value_holm", "p_value_bh", "p_value_by"} <= set(tables["ic_summary"].columns)
    assert (out / "figures" / "ic_decay.png").exists()
    assert {"reject_holm", "reject_bh", "reject_by"} <= set(tables["ic_summary"].columns)
    assert not tables["ic_summary"]["reject_holm"].any()
    assert "gross_raw_sharpe" in tables["long_short"].columns
    assert {"code_commit", "panel_sha256", "features_config_sha256", "scipy"} <= set(meta)
    assert tables["ic_summary"]["nw_t"].abs().max() < 6  # noise panel: nothing is significant
