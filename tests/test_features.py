import copy

import numpy as np
import pandas as pd
import pytest

from src.features import (
    build_panel,
    compute_features,
    compute_labels,
    feature_names,
    fill_short_gaps,
    prepare_prices,
)
from src.utils import load_config, membership_to_mask


def small_cfg(**data_overrides):
    cfg = copy.deepcopy(load_config())
    cfg["features"].update(
        reversal_windows=[1, 3],
        momentum_windows=[5, 10],
        momentum_12_1={"lookback": 10, "skip": 4},
        volatility_windows=[5, 10],
        beta_window=10,
        dollar_volume_window=5,
        high_window=10,
        trend_windows=[5, 10],
        min_obs_frac=0.8,
    )
    cfg["universe"]["liquidity"].update(window=5, min_avg_dollar_volume=0)
    cfg["preprocessing"]["winsor_limits"] = [0.05, 0.95]
    cfg["labels"].update(horizons=[1, 3], lag=1)
    cfg["data"].update(max_gap_days=2, **data_overrides)
    return cfg


def make_prices(n_days=300, n_tickers=30, seed=0, hole=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    shape = (n_days, n_tickers)
    close = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, shape), axis=0))
    dividend_factor = np.linspace(0.9, 1.0, n_days)[:, None] * np.ones(shape)
    frames = {
        "close": close,
        "closeadj": close * dividend_factor,
        "open": close * (1 + rng.normal(0, 0.005, shape)),
        "volume": rng.integers(100_000, 1_000_000, shape).astype(float),
    }
    prices = {k: pd.DataFrame(v, index=dates, columns=tickers) for k, v in frames.items()}
    if hole:  # one fully missing day for one ticker
        for frame in prices.values():
            frame.iloc[150, 3] = np.nan
    mask = pd.DataFrame(True, index=dates, columns=tickers)
    return prices, mask


def truncate(prices, mask, last_row):
    return (
        {k: v.iloc[: last_row + 1] for k, v in prices.items()},
        mask.iloc[: last_row + 1],
    )


# --- the key test -----------------------------------------------------------------------


def test_features_do_not_look_ahead():
    cfg = small_cfg()
    prices, mask = make_prices()
    full = compute_features(prepare_prices(prices, 2), mask, cfg)
    assert list(full) == feature_names(cfg)
    for t in (40, 90, 150, 151, 152, 250):
        cut_prices, cut_mask = truncate(prices, mask, t)
        part = compute_features(prepare_prices(cut_prices, 2), cut_mask, cfg)
        for name in full:
            np.testing.assert_allclose(
                part[name].iloc[-1].to_numpy(),
                full[name].iloc[t].to_numpy(),
                rtol=1e-7,
                atol=1e-10,
                equal_nan=True,
                err_msg=f"{name} at row {t} depends on future data",
            )


def test_panel_rows_do_not_look_ahead_including_zscores_and_liquidity_filter():
    cfg = small_cfg()
    cfg["universe"]["liquidity"]["min_avg_dollar_volume"] = 2.5e7  # binds for some names
    prices, mask = make_prices()
    full = build_panel(prices, mask, cfg)
    feature_cols = [c for c in full.columns if not c.startswith("fwd_")]
    dates = full.index.get_level_values("date").unique()
    assert full.groupby(level="date").size().min() < len(mask.columns)  # filter does bind
    for date in (dates[20], dates[100], dates[-30]):
        t = prices["close"].index.get_loc(date)
        part = build_panel(*truncate(prices, mask, t), cfg)
        pd.testing.assert_frame_equal(
            part.xs(date, level="date")[feature_cols],
            full.xs(date, level="date")[feature_cols],
            rtol=1e-4,
            atol=1e-6,
        )


# --- labels -----------------------------------------------------------------------------


def test_labels_hand_computed_and_nan_tail():
    adj = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41, 161.051]})
    labels = dict(compute_labels(adj, [2], lag=1))
    assert labels["fwd_ret_2d"].iloc[0, 0] == pytest.approx(133.1 / 110 - 1)
    assert labels["fwd_ret_2d"].iloc[3:].isna().all().all()  # last lag + h rows
    assert dict(compute_labels(adj, [2], lag=0))["fwd_ret_2d"].iloc[0, 0] == pytest.approx(0.21)


@pytest.mark.parametrize("lag", [1, 2])
def test_labels_use_only_prices_after_signal_date(lag):
    rng = np.random.default_rng(1)
    adj = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (40, 3)), axis=0)))
    t = 20
    base = dict(compute_labels(adj, [1, 5], lag))
    changed = adj.copy()
    changed.iloc[: t + 1] *= 3.0  # rewrite the past and the signal-date price
    # a label at t is a ratio of two prices after t, so rewriting <= t must not matter
    for name, frame in dict(compute_labels(changed, [1, 5], lag)).items():
        assert frame.iloc[t].equals(base[name].iloc[t])
    changed_future = adj.copy()
    changed_future.iloc[t + lag + 1] *= 2.0  # a price after entry does move it
    assert not dict(compute_labels(changed_future, [1, 5], lag))["fwd_ret_1d"].iloc[t].equals(
        base["fwd_ret_1d"].iloc[t]
    )


# --- hand-computed features -------------------------------------------------------------


def test_hand_computed_returns_momentum_trend_and_high():
    cfg = small_cfg()
    n, t = 30, 20
    index = pd.bdate_range("2021-01-04", periods=n)
    px = pd.DataFrame({"A": 100 * 1.01 ** np.arange(n), "B": 50 * 1.02 ** np.arange(n)}, index=index)
    prices = {"open": px, "close": px, "closeadj": px, "volume": px * 0 + 1000.0}
    mask = pd.DataFrame(True, index=index, columns=px.columns)
    feats = compute_features(prices, mask, cfg)

    assert feats["ret_1d"].iloc[t]["A"] == pytest.approx(0.01)
    assert feats["ret_5d"].iloc[t]["A"] == pytest.approx(1.01**5 - 1)
    assert feats["mom_12_1"].iloc[t]["A"] == pytest.approx(1.01**6 - 1)  # P[t-4] / P[t-10] - 1
    assert feats["vol_5d"].iloc[t]["A"] == pytest.approx(0, abs=1e-12)
    assert feats["dist_high_10d"].iloc[t]["A"] == pytest.approx(0, abs=1e-12)
    assert feats["trend_5d"].iloc[t]["A"] == pytest.approx(
        px["A"].iloc[t] / px["A"].iloc[t - 4 : t + 1].mean() - 1
    )
    assert feats["dollar_volume_5d"].iloc[t]["A"] == pytest.approx(
        1000 * px["A"].iloc[t - 4 : t + 1].mean()
    )
    assert feats["log_dollar_volume_5d"].iloc[t]["A"] == pytest.approx(
        np.log(feats["dollar_volume_5d"].iloc[t]["A"])
    )
    assert feats["overnight_ret"].iloc[t]["A"] == pytest.approx(0.01)  # open == close
    assert feats["intraday_ret"].iloc[t]["A"] == pytest.approx(0, abs=1e-12)


def test_distance_from_high_after_a_drop():
    cfg = small_cfg()
    cfg["features"]["high_window"] = 3
    index = pd.bdate_range("2021-01-04", periods=6)
    px = pd.DataFrame({"A": [100, 110, 120, 90, 95, 100.0]}, index=index)
    prices = {"open": px, "close": px, "closeadj": px, "volume": px * 0 + 1.0}
    feats = compute_features(prices, pd.DataFrame(True, index=index, columns=px.columns), cfg)
    assert feats["dist_high_3d"].iloc[3]["A"] == pytest.approx(90 / 120 - 1)


def test_beta_recovers_known_slopes():
    cfg = small_cfg()
    rng = np.random.default_rng(2)
    market = rng.normal(0, 0.01, 80)
    betas = {"A": 0.5, "B": 1.0, "C": 1.5}  # mean 1.0, so the EW market return equals `market`
    index = pd.bdate_range("2021-01-04", periods=80)
    px = pd.DataFrame({k: 100 * np.cumprod(1 + b * market) for k, b in betas.items()}, index=index)
    prices = {"open": px, "close": px, "closeadj": px, "volume": px * 0 + 1.0}
    feats = compute_features(prices, pd.DataFrame(True, index=index, columns=px.columns), cfg)
    for ticker, beta in betas.items():
        assert feats["beta_10d"].iloc[-1][ticker] == pytest.approx(beta, rel=1e-6)


def test_overnight_and_intraday_compound_to_the_daily_return():
    cfg = small_cfg()
    prices, mask = make_prices(hole=False)
    feats = compute_features(prepare_prices(prices, 2), mask, cfg)
    compounded = (1 + feats["overnight_ret"]) * (1 + feats["intraday_ret"]) - 1
    np.testing.assert_allclose(
        compounded.iloc[1:].to_numpy(), feats["ret_1d"].iloc[1:].to_numpy(), rtol=1e-9, atol=1e-12
    )
    open_adj = prices["open"] * prices["closeadj"] / prices["close"]
    expected = open_adj.iloc[5, 0] / prices["closeadj"].iloc[4, 0] - 1
    assert feats["overnight_ret"].iloc[5, 0] == pytest.approx(expected)


# --- missing data -----------------------------------------------------------------------


def test_fill_short_gaps_is_capped_and_causal():
    frame = pd.DataFrame({"A": [1.0, np.nan, np.nan, np.nan, 5.0, np.nan]})
    filled = fill_short_gaps(frame, 2)["A"].tolist()
    assert filled[:3] == [1.0, 1.0, 1.0]
    assert np.isnan(filled[3]) and filled[4] == 5.0 and filled[5] == 5.0
    assert fill_short_gaps(frame, 0).equals(frame)
    # causal: the value at row k does not depend on later rows
    for k in range(len(frame)):
        assert fill_short_gaps(frame.iloc[: k + 1], 2)["A"].iloc[-1] == pytest.approx(
            filled[k], nan_ok=True
        )


@pytest.mark.parametrize("max_gap_days", [0, 2])
def test_one_missing_day_does_not_wipe_out_later_features(max_gap_days):
    cfg = small_cfg()
    prices, mask = make_prices()  # ticker T03 is missing on row 150
    feats = compute_features(prepare_prices(prices, max_gap_days), mask, cfg)
    for name in ("ret_10d", "vol_10d", "dist_high_10d", "trend_10d", "beta_10d"):
        assert feats[name].iloc[155]["T03"] == feats[name].iloc[155]["T03"], name  # not NaN


# --- universe ---------------------------------------------------------------------------


def test_panel_follows_point_in_time_membership():
    cfg = small_cfg()
    prices, _ = make_prices(hole=False)
    index, columns = prices["close"].index, prices["close"].columns
    removal, addition = index[120], index[180]
    membership = pd.concat(
        [
            pd.DataFrame({"date": index[index < removal], "ticker": "T00"}),  # removed on `removal`
            pd.DataFrame({"date": index[index >= addition], "ticker": "T01"}),  # added on `addition`
            pd.DataFrame({"date": index, "ticker": "T02"}),
        ]
    )
    mask = membership_to_mask(membership, index, columns)
    panel = build_panel(prices, mask, cfg)
    dates = panel.index.get_level_values("date")
    tickers = panel.index.get_level_values("ticker")

    assert dates[tickers == "T00"].max() < removal
    assert dates[tickers == "T01"].min() >= addition
    assert set(tickers) == {"T00", "T01", "T02"}
    assert mask.to_numpy()[
        index.get_indexer(dates), columns.get_indexer(tickers)
    ].all()  # every row is a member-day


def test_new_member_has_features_from_pre_membership_history():
    cfg = small_cfg()
    prices, _ = make_prices(hole=False)
    index, columns = prices["close"].index, prices["close"].columns
    joined = index[200]
    membership = pd.concat(
        [
            pd.DataFrame({"date": index, "ticker": [c for c in columns[:5]][0]}),
            pd.DataFrame({"date": index[index >= joined], "ticker": "T10"}),
        ]
    )
    panel = build_panel(prices, membership_to_mask(membership, index, columns), cfg)
    first_row = panel.xs("T10", level="ticker").loc[joined]
    assert first_row["ret_10d"] == first_row["ret_10d"]  # defined on the first member day


# --- standardisation --------------------------------------------------------------------


def test_each_feature_is_standardised_within_each_date():
    cfg = small_cfg()
    prices, mask = make_prices()
    panel = build_panel(prices, mask, cfg)
    z_cols = [f"z_{n}" for n in feature_names(cfg)]
    grouped = panel[z_cols].groupby(level="date")
    stats_mean, stats_std, count = grouped.mean(), grouped.std(), grouped.count()
    enough = count >= 10
    assert (stats_mean.where(enough).abs().max() < 1e-4).all()
    assert ((stats_std.where(enough) - 1).abs().max() < 1e-4).all()
    assert {"fwd_ret_1d", "fwd_ret_1d_xs", "fwd_ret_3d"} <= set(panel.columns)
    assert panel.index.get_level_values("date").min() > mask.index[9]  # warm-up dropped


def test_default_config_has_no_liquidity_filter_and_universe_is_members_that_traded():
    cfg = load_config()
    assert cfg["universe"]["liquidity"]["min_avg_dollar_volume"] == 0
    cfg = small_cfg()
    cfg["universe"]["liquidity"]["min_avg_dollar_volume"] = 0
    prices, mask = make_prices()  # ticker T03 has no price on row 150
    panel = build_panel(prices, mask, cfg)
    dates = panel.index.get_level_values("date")
    kept_dates = prices["close"].index[10:]  # first 10 rows are warm-up
    expected = int(prices["closeadj"].loc[kept_dates].notna().to_numpy().sum())
    assert len(panel) == expected
    assert ("T03" in panel.index.get_level_values("ticker")) and (
        prices["close"].index[150] not in dates[panel.index.get_level_values("ticker") == "T03"]
    )


# --- delisted stocks --------------------------------------------------------------------

from src.features import delisting_info  # noqa: E402

DELIST_CFG = {"enabled": True, "grace_days": 2, "haircut": {"default": 0.0, "bankruptcy": -0.3}}


def stopped_trading_prices():
    """A trades throughout, B stops after row 6, C has an interior hole at row 4."""
    index = pd.bdate_range("2021-01-04", periods=12)
    a = pd.Series(100.0 * 1.01 ** np.arange(12), index=index)
    b = pd.Series(50.0 * 1.02 ** np.arange(12), index=index)
    b.iloc[7:] = np.nan
    c = pd.Series(20.0 * 1.03 ** np.arange(12), index=index)
    c.iloc[4] = np.nan
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_delisted_stock_exits_at_last_price_and_is_flagged():
    adj = stopped_trading_prices()
    info = delisting_info(adj, None, DELIST_CFG)
    assert info["delisted"].to_dict() == {"A": False, "B": True, "C": False}
    labels = dict(compute_labels(adj, [3], lag=1, terminal=info))
    label, flag = labels["fwd_ret_3d"], labels["fwd_terminal_3d"]
    # t=4: entry row 5, exit row 8 is after B's last price (row 6) -> exit at P[6]
    assert label.loc[adj.index[4], "B"] == pytest.approx(adj["B"].iloc[6] / adj["B"].iloc[5] - 1)
    assert flag.loc[adj.index[4], "B"] == 1.0
    assert flag.loc[adj.index[2], "B"] == 0.0  # exit row 6 exists: a normal label
    assert label.loc[adj.index[2], "B"] == pytest.approx(adj["B"].iloc[6] / adj["B"].iloc[3] - 1)
    assert np.isnan(label.loc[adj.index[6], "B"])  # entry row 7 is after the last price
    assert flag.loc[adj.index[6], "B"] == 0.0


def test_terminal_labels_never_fill_gaps_or_the_end_of_the_sample():
    adj = stopped_trading_prices()
    info = delisting_info(adj, None, DELIST_CFG)
    labels = dict(compute_labels(adj, [3], lag=1, terminal=info))
    label, flag = labels["fwd_ret_3d"], labels["fwd_terminal_3d"]
    assert np.isnan(label.loc[adj.index[0], "C"])  # exit row 4 is an interior hole
    assert flag["C"].sum() == 0
    assert label["A"].iloc[-4:].isna().all() and flag["A"].sum() == 0  # still trading: truncated


def test_haircut_by_reason_is_applied_to_the_last_price():
    adj = stopped_trading_prices()
    delistings = pd.DataFrame(
        {"ticker": ["B"], "date": [adj.index[6] + pd.Timedelta(days=3)], "reason": ["bankruptcy"]}
    )
    info = delisting_info(adj, delistings, DELIST_CFG)
    assert info.loc["B", "reason"] == "bankruptcy" and info.loc["B", "haircut"] == -0.3
    label = dict(compute_labels(adj, [3], lag=1, terminal=info))["fwd_ret_3d"]
    assert label.loc[adj.index[4], "B"] == pytest.approx(
        adj["B"].iloc[6] * 0.7 / adj["B"].iloc[5] - 1
    )
    far = delistings.assign(date=adj.index[6] + pd.Timedelta(days=90))
    assert delisting_info(adj, far, DELIST_CFG).loc["B", "reason"] == "unknown"


def test_panel_uses_terminal_labels_only_when_enabled_and_leaves_features_alone():
    prices, mask = make_prices(hole=False)
    for frame in prices.values():
        frame.iloc[250:, 5] = np.nan  # T05 stops trading after row 249
    on, off = small_cfg(), small_cfg()
    on["labels"]["delisting"] = dict(DELIST_CFG, grace_days=5)
    off["labels"]["delisting"] = {"enabled": False}
    panel_on, panel_off = build_panel(prices, mask, on), build_panel(prices, mask, off)
    date = prices["close"].index[246]
    row_on = panel_on.xs("T05", level="ticker").loc[date]
    row_off = panel_off.xs("T05", level="ticker").loc[date]
    adj = prices["closeadj"]["T05"]
    assert row_on["fwd_terminal_3d"] == 1.0
    assert row_on["fwd_ret_3d"] == pytest.approx(adj.iloc[249] / adj.iloc[247] - 1, rel=1e-5)
    assert "fwd_terminal_3d" not in panel_off.columns
    # beyond the 2-day gap fill the old behaviour is a NaN label, the new one exits at the last price
    later = prices["close"].index[248]
    assert np.isnan(panel_off.xs("T05", level="ticker").loc[later, "fwd_ret_3d"])
    row_later = panel_on.xs("T05", level="ticker").loc[later]
    assert row_later["fwd_terminal_3d"] == 1.0 and row_later["fwd_ret_3d"] == pytest.approx(0.0)
    features = [c for c in panel_off.columns if not c.startswith("fwd_")]
    pd.testing.assert_frame_equal(panel_on[features], panel_off[features])
