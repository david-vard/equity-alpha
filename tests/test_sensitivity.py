import numpy as np
import pandas as pd
import pytest

from src import run_sensitivity
from src.metrics import newey_west_lag, newey_west_tstat, sharpe
from src.run_sensitivity import TOTAL, TUNING_LABEL, bootstrap_tables, offset_tables, repricing_tables, run_checks
from src.sensitivity import (
    breakevens,
    load_sensitivity_config,
    offset_dispersion,
    offset_rows,
    percentile_interval,
    reprice,
    repricing_rows,
    resampled_sharpes,
    stationary_indices,
)

HOLDOUT = pd.Timestamp("2022-01-01")


# --------------------------------------------------------------------------- bootstrap


def test_stationary_indices_are_reproducible_and_stay_in_range():
    a = stationary_indices(50, 200, 5, 1)
    assert a.shape == (200, 50) and a.min() >= 0 and a.max() < 50
    assert np.array_equal(a, stationary_indices(50, 200, 5, 1))
    assert not np.array_equal(a, stationary_indices(50, 200, 5, 2))
    assert not np.array_equal(a, stationary_indices(50, 200, 10, 1))


def test_stationary_blocks_have_the_declared_mean_length_and_wrap():
    index = stationary_indices(500, 200, 21, 0)
    follows = np.diff(index, axis=1) % 500 == 1
    assert 1 / (1 - follows.mean()) == pytest.approx(21, rel=0.05)
    assert ((index[:, :-1] == 499) & (index[:, 1:] == 0)).any()  # the circle closes


def test_stationary_bootstrap_visits_days_evenly():
    counts = np.bincount(stationary_indices(100, 2000, 7, 3).ravel(), minlength=100)
    assert counts.min() > 0.9 * counts.mean() and counts.max() < 1.1 * counts.mean()


def test_resampled_sharpes_match_the_project_sharpe():
    x = np.random.default_rng(0).normal(0.001, 0.01, 100)
    index = stationary_indices(100, 5, 5, 0)
    out = resampled_sharpes(x, index, 252)
    assert out == pytest.approx([sharpe(x[row], 252) for row in index])


def test_resampled_sharpe_of_a_constant_is_missing_not_infinite():
    assert np.isnan(resampled_sharpes(np.ones(10), stationary_indices(10, 3, 2, 0), 252)).all()


def test_longer_blocks_widen_the_interval_of_an_autocorrelated_series():
    rng = np.random.default_rng(1)
    x = np.convolve(rng.normal(size=1000), np.ones(10) / 10, mode="same") + 0.01
    short = resampled_sharpes(x, stationary_indices(1000, 2000, 2, 0), 252).std()
    long = resampled_sharpes(x, stationary_indices(1000, 2000, 40, 0), 252).std()
    assert long > 1.3 * short


def test_percentile_interval_by_hand():
    out = percentile_interval(np.arange(-500.0, 501.0), 0.95)
    assert (out["ci_low"], out["ci_high"]) == pytest.approx((-475.0, 475.0))
    assert out["contains_zero"] and out["share_above_zero"] == pytest.approx(500 / 1001)
    assert not percentile_interval(np.arange(1.0, 101.0), 0.95)["contains_zero"]


# --------------------------------------------------------------------------- repricing


def costed(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", periods=n)
    frame = pd.DataFrame({
        "gross_pnl": rng.normal(0.0004, 0.004, n), "commission": rng.uniform(0, 1e-5, n), "spread": rng.uniform(0, 4e-5, n),
        "impact": rng.uniform(0, 1e-4, n), "borrow": rng.uniform(0, 2e-5, n)}, index=dates)
    frame["cost"] = frame[["commission", "spread", "impact"]].sum(axis=1)
    frame["net_pnl"] = frame["gross_pnl"] - frame["cost"] - frame["borrow"]
    return frame


def test_reprice_at_the_base_point_is_the_recorded_net():
    frame = costed()
    assert np.abs(reprice(frame, 1.0, 25.0, 1.0, 25.0) - frame["net_pnl"]).max() < 1e-15


def test_reprice_scales_only_impact_and_borrow():
    frame = costed()
    out = reprice(frame, 2.0, 50.0, 1.0, 25.0)
    assert out.to_numpy() == pytest.approx((frame["net_pnl"] - frame["impact"] - frame["borrow"]).to_numpy())
    free = reprice(frame, 0.25, 0.0, 1.0, 25.0)
    assert free.to_numpy() == pytest.approx(
        (frame["gross_pnl"] - frame["commission"] - frame["spread"] - frame["impact"] / 4).to_numpy())


def test_reprice_needs_positive_base_values():
    with pytest.raises(ValueError, match="positive base"):
        reprice(costed(), 1.0, 25.0, 1.0, 0.0)


def test_breakevens_set_the_mean_net_return_to_zero():
    frame = costed()
    out = breakevens(frame, 1.0, 25.0)
    assert reprice(frame, out["breakeven_impact_y"], 25.0, 1.0, 25.0).mean() == pytest.approx(0.0, abs=1e-15)
    assert reprice(frame, 1.0, out["breakeven_borrow_bps_per_year"], 1.0, 25.0).mean() == pytest.approx(0.0, abs=1e-15)


def test_a_negative_breakeven_means_losing_with_that_cost_at_zero():
    frame = costed()
    frame["gross_pnl"] = -abs(frame["gross_pnl"]) - 1e-4
    assert breakevens(frame, 1.0, 25.0)["breakeven_impact_y"] < 0


def test_repricing_grid_is_the_full_cross_with_one_base_point():
    cfg = load_sensitivity_config()
    frame = costed()
    rows = pd.DataFrame(repricing_rows(frame, cfg, 1.0, 25.0, 5, 252, {"book": "x"}))
    assert len(rows) == 25 and rows["is_base"].sum() == 1
    base = rows[rows["is_base"]].iloc[0]
    assert base["net_sharpe"] == pytest.approx(sharpe(frame["net_pnl"], 252))
    assert base["net_nw_t"] == pytest.approx(newey_west_tstat(frame["net_pnl"].to_numpy(), newey_west_lag(300, 5))["t"])


# --------------------------------------------------------------------------- offsets


def tranche_frame(n_days=250, n=5):
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2014-01-02", periods=n_days)
    return pd.concat([pd.DataFrame({"tranche": str(k), "gross_pnl": rng.normal(1e-4, 1e-3, n_days),
                                    "net_pnl": rng.normal(5e-5, 1e-3, n_days), "turnover": rng.uniform(0, 0.01, n_days)},
                                   index=dates) for k in range(n)])


def test_offsets_scale_returns_to_the_tranche_but_not_sharpe():
    frame = tranche_frame()
    rows = pd.DataFrame(offset_rows(frame, 252, {"book": "x"}, ["net_pnl"]))
    assert list(rows["offset"]) == [0, 1, 2, 3, 4]
    one = frame[frame["tranche"] == "2"]
    row = rows[rows["offset"] == 2].iloc[0]
    assert row["sharpe"] == pytest.approx(sharpe(one["net_pnl"], 252))
    assert row["return_ann"] == pytest.approx(one["net_pnl"].mean() * 252 * 5)
    assert row["turnover_ann"] == pytest.approx(one["turnover"].mean() * 252 * 5)


def test_offset_dispersion_by_hand():
    offsets = pd.DataFrame({"book": "x", "offset": [0, 1, 2], "sharpe": [0.1, 0.3, 0.2], "return_ann": [1.0, 2.0, 3.0],
                            "turnover_ann": [5.0, 5.0, 5.0]})
    totals = pd.DataFrame({"book": ["x"], "sharpe": [0.25], "return_ann": [2.0], "turnover_ann": [5.0]})
    out = offset_dispersion(offsets, totals, ["book"]).set_index("statistic")
    assert out.loc["sharpe", "total"] == pytest.approx(0.25) and out.loc["sharpe", "range"] == pytest.approx(0.2)
    assert out.loc["sharpe", "sd"] == pytest.approx(0.1) and out.loc["turnover_ann", "range"] == 0.0
    assert out.loc["return_ann", "n_offsets"] == 3


# --------------------------------------------------------------------------- the whole analysis on synthetic books


def synthetic(n_days=320):
    """Twelve books with 5 or 21 tranches, consistent accounting, and a matching summary table."""
    cfg = load_sensitivity_config()
    cfg["bootstrap"]["n_resamples"] = 200
    rng = np.random.default_rng(4)
    signals = {"lightgbm_h5_rank": 5, "B0_reversal_h5": 5, "B1_composite_h5": 5, "lightgbm_h21_rank": 21}
    dates = pd.bdate_range("2013-08-01", periods=n_days)
    books, rows, windows = {}, [], {}
    split = pd.Timestamp("2014-01-01")
    for signal, h in signals.items():
        for construction in ("decile_ls", "neutral_volscaled", "optimised"):
            parts = []
            for k in range(h):
                part = pd.DataFrame({"tranche": str(k), "horizon": h, "gross_pnl": rng.normal(1e-4, 1e-3, n_days) / h,
                                     "commission": 1e-6 / h, "spread": 4e-6 / h, "impact": rng.uniform(0, 2e-5, n_days) / h,
                                     "borrow": 1e-6 / h, "turnover": 0.01 / h}, index=dates)
                part["cost"] = part["commission"] + part["spread"] + part["impact"]
                part["net_pnl"] = part["gross_pnl"] - part["cost"] - part["borrow"]
                parts.append(part)
            frame = pd.concat(parts)
            numeric = frame.drop(columns=["tranche", "horizon"]).groupby(level=0).sum()
            total = numeric.assign(tranche=TOTAL, horizon=h)
            frame = pd.concat([frame, total]).rename_axis("date").sort_index(kind="stable")
            book = (signal, construction)
            books[book] = frame
            windows[book] = {"evaluation": (dates[dates >= split][0], dates[-1]), "tuning": (dates[0], dates[dates < split][-1])}
            for window, (lo, hi) in windows[book].items():
                sub = total.loc[lo:hi]
                rows.append({"signal": signal, "construction": construction, "scenario": "base", "window": window,
                             "start": lo, "end": hi, "gross_sharpe": sharpe(sub["gross_pnl"], 252),
                             "net_sharpe": sharpe(sub["net_pnl"], 252)})
    return cfg, books, windows, pd.DataFrame(rows)


def test_checks_pass_on_consistent_books():
    cfg, books, windows, summary = synthetic()
    checks = run_checks(cfg, 1.0, 25.0, books, windows, summary, HOLDOUT, 252)
    assert checks["passed"].all()
    assert set(checks["check"]) == {"last_date_before_holdout", "tranches_sum", "reprice_base", "reproduce_sharpes", "paired_dates"}


@pytest.mark.parametrize("check, spoil", [
    ("reproduce_sharpes", lambda books, summary: summary.__setitem__("net_sharpe", summary["net_sharpe"] + 1e-3)),
    ("tranches_sum", lambda books, summary: books.update({("B0_reversal_h5", "optimised"): (
        lambda f: f.assign(gross_pnl=f["gross_pnl"].where(f["tranche"] != "0", 1.0)))(books[("B0_reversal_h5", "optimised")])})),
    ("reprice_base", lambda books, summary: books.update({("B1_composite_h5", "decile_ls"): (
        lambda f: f.assign(net_pnl=f["net_pnl"] + 1e-6))(books[("B1_composite_h5", "decile_ls")])})),
])
def test_a_spoiled_input_fails_its_check(check, spoil):
    cfg, books, windows, summary = synthetic()
    spoil(books, summary)
    checks = run_checks(cfg, 1.0, 25.0, books, windows, summary, HOLDOUT, 252)
    assert not checks.query("check == @check")["passed"].all()


def test_a_date_on_the_holdout_fails_the_check():
    cfg, books, windows, summary = synthetic()
    assert not run_checks(cfg, 1.0, 25.0, books, windows, summary, pd.Timestamp("2014-06-01"), 252).query(
        "check == 'last_date_before_holdout'")["passed"].all()


def test_paired_books_with_different_dates_fail_the_check():
    cfg, books, windows, summary = synthetic()
    book = ("B0_reversal_h5", "optimised")
    books[book] = books[book][books[book].index != books[book].index[-1]]
    checks = run_checks(cfg, 1.0, 25.0, books, windows, summary, HOLDOUT, 252)
    assert not checks.query("check == 'paired_dates'")["passed"].all()


def test_bootstrap_tables_cover_every_book_length_and_window():
    cfg, books, windows, _ = synthetic()
    out = bootstrap_tables(cfg, books, windows, 252)
    assert len(out["bootstrap_intervals"]) == 12 * 2 * 2 * 3
    assert len(out["paired_differences"]) == 3 * 2 * 2 * 3
    assert set(out["bootstrap_intervals"]["window"]) == {"evaluation", TUNING_LABEL}
    primary = out["primary_intervals"].set_index("interval")
    assert list(primary.index) == ["headline", "lightgbm_minus_b0"]
    assert (primary["mean_block_length"] == 21).all()
    frame = books[("lightgbm_h5_rank", "optimised")]
    total = frame[frame["tranche"] == TOTAL].loc[slice(*windows[("lightgbm_h5_rank", "optimised")]["evaluation"])]
    assert primary.loc["headline", "estimate"] == pytest.approx(sharpe(total["net_pnl"], 252))
    assert primary.loc["headline", "ci_low"] < primary.loc["headline", "ci_high"]


def test_the_difference_of_a_book_with_itself_is_exactly_zero():
    cfg, books, windows, _ = synthetic()
    books[("B0_reversal_h5", "optimised")] = books[("lightgbm_h5_rank", "optimised")]
    row = bootstrap_tables(cfg, books, windows, 252)["primary_intervals"].set_index("interval").loc["lightgbm_minus_b0"]
    assert row["estimate"] == 0.0 and row["ci_low"] == 0.0 and row["ci_high"] == 0.0


def test_bootstrap_tables_are_reproducible():
    cfg, books, windows, _ = synthetic()
    pd.testing.assert_frame_equal(bootstrap_tables(cfg, books, windows, 252)["paired_differences"],
                                  bootstrap_tables(cfg, books, windows, 252)["paired_differences"])


def test_repricing_and_offset_tables_cover_every_book():
    cfg, books, windows, _ = synthetic()
    rep = repricing_tables(cfg, 1.0, 25.0, books, windows, 252)
    assert len(rep["repricing_grid"]) == 12 * 2 * 25 and len(rep["repricing_breakeven"]) == 12 * 2
    off = offset_tables(cfg, books, windows, 252)
    assert off["offsets"].groupby("signal")["offset"].nunique().to_dict()["lightgbm_h21_rank"] == 21
    assert len(off["offset_dispersion"]) == 12 * 2 * 2 * 3


def test_run_refuses_a_failed_check_without_writing_results(tmp_path, monkeypatch):
    cfg, books, windows, summary = synthetic()
    summary["net_sharpe"] += 1.0
    out = tmp_path / "reports"
    cfg["outputs"]["reports_dir"] = str(out)
    monkeypatch.setattr(run_sensitivity, "load_sensitivity_config", lambda *_: cfg)
    pcfg = run_sensitivity.load_portfolio_config()
    monkeypatch.setattr(run_sensitivity, "load_portfolio_config", lambda *a: pcfg)
    monkeypatch.setattr(run_sensitivity, "provenance", lambda *a: {})
    monkeypatch.setattr(run_sensitivity, "load_books", lambda *a: books)
    monkeypatch.setattr(run_sensitivity, "book_windows", lambda *a: windows)
    monkeypatch.setattr(run_sensitivity.pd, "read_csv", lambda *a, **k: summary)
    monkeypatch.setattr(run_sensitivity.json, "loads", lambda *_: {"config_sha256": ["x"]})
    monkeypatch.setattr(run_sensitivity, "file_sha256", lambda *_: "x")
    with pytest.raises(SystemExit, match="no results were written"):
        run_sensitivity.run([])
    assert [p.name for p in out.iterdir()] == ["checks.csv"]
