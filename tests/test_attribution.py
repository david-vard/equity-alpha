import zipfile

import numpy as np
import openpyxl
import pandas as pd
import pytest
import statsmodels.api as sm

from src import run_attribution
from src.attribution import (
    FACTORS,
    conditional_beta,
    drawdown_episodes,
    load_attribution_config,
    load_factors,
    moments,
    ols_newey_west,
    parse_aqr_xlsx,
    parse_french_csv,
    primary_outcome,
    up_down_beta,
    var_es,
    weights_exposure,
)
from src.portfolio_report import regression_beta
from src.run_attribution import TUNING_LABEL, analyse

CUT = pd.Timestamp("2022-01-01")
FRENCH = """This file was created using the 202608 CRSP database.

,Mkt-RF,SMB
20211230,    0.10,   -0.20
20211231,  -99.99,    0.30
20220103,    5.00,    6.00

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


# --------------------------------------------------------------------------- factor files


def test_french_parser_scales_marks_missing_and_cuts_at_the_holdout():
    frame = parse_french_csv(FRENCH, {"Mkt-RF": "mkt_rf", "SMB": "smb"}, 0.01, CUT)
    assert list(frame.index) == [pd.Timestamp("2021-12-30"), pd.Timestamp("2021-12-31")]
    assert frame.loc["2021-12-30", "mkt_rf"] == pytest.approx(0.001)
    assert np.isnan(frame.loc["2021-12-31", "mkt_rf"]) and frame.loc["2021-12-31", "smb"] == pytest.approx(0.003)


def test_french_parser_never_converts_a_row_past_the_holdout():
    poisoned = FRENCH.replace("20220103,    5.00,    6.00", "20220103,  not_a_number,  x")
    assert len(parse_french_csv(poisoned, {"Mkt-RF": "mkt_rf", "SMB": "smb"}, 0.01, CUT)) == 2


def test_french_parser_needs_the_header():
    with pytest.raises(ValueError, match="no header"):
        parse_french_csv(FRENCH, {"Mom": "mom"}, 0.01, CUT)


def aqr_file(path, last_cell=0.5):
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "BAB Factors"
    for row in (["Title"], [], ["DATE", "AUS", "USA"], ["12/30/2021", 0.9, 0.01], ["12/31/2021", 0.9, None],
                ["01/03/2022", 0.9, last_cell]):
        sheet.append(row)
    book.save(path)


def test_aqr_parser_stops_at_the_holdout(tmp_path):
    aqr_file(tmp_path / "b.xlsx", last_cell="not_a_number")
    frame = parse_aqr_xlsx(tmp_path / "b.xlsx", "BAB Factors", {"USA": "bab"}, 1.0, CUT)
    assert list(frame.index) == [pd.Timestamp("2021-12-30"), pd.Timestamp("2021-12-31")]
    assert frame["bab"].iloc[0] == pytest.approx(0.01) and np.isnan(frame["bab"].iloc[1])


def test_load_factors_merges_every_source_before_the_holdout(tmp_path):
    cfg = load_attribution_config()
    for source in cfg["factors"]["sources"].values():
        name = source["url"].rsplit("/", 1)[-1]
        if source["format"] == "aqr_xlsx":
            aqr_file(tmp_path / name)
            continue
        cols = list(source["columns"])
        rows = "\n".join(f"{d},{','.join(['0.5'] * len(cols))}" for d in ("20211230", "20211231", "20220103"))
        with zipfile.ZipFile(tmp_path / name, "w") as archive:
            archive.writestr("data.csv", f"note\n\n,{','.join(cols)}\n{rows}\n\nCopyright\n")
    factors = load_factors(cfg, tmp_path, CUT)
    assert list(factors.columns) == list(FACTORS)
    assert factors.index.max() == pd.Timestamp("2021-12-31")
    assert factors.loc["2021-12-30", "st_rev"] == pytest.approx(0.005)


def test_download_keeps_an_existing_file(tmp_path, monkeypatch):
    cfg = load_attribution_config()
    calls = []

    class Response:
        content = b"file"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(run_attribution.requests, "get", lambda url, **kw: calls.append(url) or Response())
    first = run_attribution.download(cfg, tmp_path)
    assert len(calls) == len(first) == 4
    run_attribution.download(cfg, tmp_path)
    assert len(calls) == 4
    name = next(iter(first))
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(SystemExit, match="does not match"):
        run_attribution.verified_manifest(cfg, tmp_path)


# --------------------------------------------------------------------------- regressions


def test_ols_matches_statsmodels_hac_and_the_portfolio_beta():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = 0.1 + X @ [1.0, -0.5, 0.2] + rng.normal(size=400)
    fit = ols_newey_west(y, X, ["a", "b", "c"], 7)
    ref = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 7, "use_correction": False})
    assert np.allclose(fit["coef"], ref.params) and np.allclose(fit["se"], ref.bse)
    assert fit["r2"] == pytest.approx(ref.rsquared)
    one = ols_newey_west(y, X[:, :1], ["a"], 7)
    old = regression_beta(y, X[:, 0], 7)
    assert one["coef"]["a"] == pytest.approx(old["realised_beta"]) and one["t"]["a"] == pytest.approx(old["realised_beta_t"])


def test_ols_refuses_missing_values():
    with pytest.raises(ValueError, match="never dropped"):
        ols_newey_west(np.array([1.0, np.nan, 2.0, 3.0]), np.arange(4.0)[:, None], ["a"], 1)


@pytest.mark.parametrize("beta, rev_t, alpha_t, outcome", [
    (0.5, 3.0, 1.0, "reversal_explained"),
    (0.5, 3.0, -5.0, "reversal_explained"),
    (0.5, 3.0, 2.0, "reversal_partial"),
    (0.5, 1.0, 2.5, "not_reversal"),
    (-0.5, 3.0, 2.5, "not_reversal"),
    (0.5, 1.0, 1.0, "inconclusive"),
    (-0.5, 3.0, 1.0, "inconclusive"),
])
def test_primary_outcome_covers_each_case(beta, rev_t, alpha_t, outcome):
    assert primary_outcome(beta, rev_t, alpha_t, 2.0) == outcome


# --------------------------------------------------------------------------- risk


def test_drawdown_episodes_by_hand():
    dates = pd.bdate_range("2020-01-01", periods=7)
    pnl = pd.Series([1.0, -0.5, 0.6, -2.0, 0.5, 0.1, 0.1], dates)
    first, second = drawdown_episodes(pnl, 3)
    assert first["depth"] == pytest.approx(-2.0) and first["peak"] == dates[2] and first["trough"] == dates[3]
    assert first["recovery"] is None and first["days_under_water"] == 4
    assert second["depth"] == pytest.approx(-0.5) and second["recovery"] == dates[2]


def test_var_es_and_moments_by_hand():
    pnl = pd.Series(np.arange(-50.0, 50.0) / 100, pd.bdate_range("2020-01-01", periods=100))
    loss = var_es(pnl, 0.95, 1)
    assert loss["var"] == pytest.approx(-np.quantile(pnl, 0.05))
    assert loss["es"] == pytest.approx(0.48)
    rolled = var_es(pnl, 0.95, 5)
    assert rolled["var"] > loss["var"]
    m = moments(pnl, 7)
    assert m["skew_daily"] == pytest.approx(0.0, abs=1e-12) and m["excess_kurtosis_daily"] < 0


# --------------------------------------------------------------------------- beta


def test_weights_exposure_holds_each_tranche_to_its_next_rebalance():
    dates = pd.bdate_range("2020-01-01", periods=4)
    beta = pd.DataFrame({"A": [2.0] * 4, "B": [1.0] * 4}, index=dates)
    weights = pd.DataFrame({
        "score_date": [dates[0], dates[0], dates[1], dates[2]], "trade_date": [dates[1], dates[1], dates[2], dates[3]],
        "tranche": [0, 0, 1, 0], "ticker": ["A", "B", "A", "Z"], "weight": [1.0, -1.0, 0.5, 1.0]})
    out = weights_exposure(weights, beta, dates, 2)
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0, 0.5])  # Z has no beta and counts zero


def test_conditional_beta_recovers_a_true_exposure():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2015-01-01", periods=3000)
    m = pd.Series(rng.normal(size=3000), dates)
    x = pd.Series(rng.normal(size=3000), dates)
    y = x.shift(1).fillna(0.0) * m + rng.normal(scale=0.01, size=3000)
    out = conditional_beta(y, m, x, 5)
    assert out["b0"] == pytest.approx(0.0, abs=0.01) and out["b1"] == pytest.approx(1.0, abs=0.01)
    assert out["n_days"] == 2999


def test_up_down_beta_separates_the_two_sides():
    rng = np.random.default_rng(2)
    m = pd.Series(rng.normal(size=2000))
    y = pd.Series(np.where(m > 0, 0.2, 0.7) * m + rng.normal(scale=0.01, size=2000))
    out = up_down_beta(y, m, 5)
    assert out["beta_up"] == pytest.approx(0.2, abs=0.01) and out["beta_down"] == pytest.approx(0.7, abs=0.01)
    assert out["up_minus_down"] == pytest.approx(-0.5, abs=0.02)


# --------------------------------------------------------------------------- the whole analysis on synthetic books


def synthetic(seed=3):
    cfg = load_attribution_config()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", "2015-12-31")
    n = len(dates)
    factors = pd.DataFrame(rng.normal(scale=0.01, size=(n, len(FACTORS))), dates, columns=list(FACTORS))
    market = factors["mkt_rf"] + rng.normal(scale=0.002, size=n)
    daily, exposures, rows = {}, {}, []
    split = pd.Timestamp("2014-01-01")
    for name, spec in cfg["books"].items():
        gross = 0.5 * factors["st_rev"] + 0.0001 + rng.normal(scale=0.004, size=n)
        frame = pd.DataFrame({"gross_pnl": gross, "net_pnl": gross - 0.0001, "market_return": market,
                              "beta_ex_ante": rng.normal(scale=0.01, size=n), "horizon": 5}, index=dates)
        daily[name] = frame
        exposures[name] = {"exposure_short_beta": pd.Series(rng.normal(scale=0.01, size=n), dates),
                           "exposure_252d_weights": frame["beta_ex_ante"]}
        for window, sub in (("evaluation", frame[frame.index >= split]), ("tuning", frame[frame.index < split])):
            beta = ols_newey_west(sub["net_pnl"].to_numpy(), sub[["market_return"]].to_numpy(), ["m"],
                                  run_attribution.newey_west_lag(len(sub), 5))
            rows.append({"signal": spec["signal"], "construction": spec["construction"],
                         "window": window if window == "evaluation" else TUNING_LABEL,
                         "realised_beta": beta["coef"]["m"], "realised_beta_t": beta["t"]["m"]})
    windows = {name: {"evaluation": (split, dates[-1]), "tuning": (dates[0], pd.Timestamp("2013-12-31"))} for name in daily}
    return cfg, daily, exposures, factors, windows, pd.DataFrame(rows)


def test_analysis_passes_its_checks_and_finds_the_planted_reversal():
    cfg, daily, exposures, factors, windows, table = synthetic()
    out = analyse(cfg, daily, exposures, factors, windows, table, 252)
    assert out["checks"]["passed"].all()
    test = out["primary_test"].iloc[0]
    assert test["reversal_beta"] == pytest.approx(0.5, abs=0.05) and test["outcome"].startswith("reversal_")
    assert (out["factor_models"]["decomposition_gap"].abs() < 1e-12).all()
    assert set(out["factor_loadings"]["window"]) == {"evaluation", TUNING_LABEL}
    assert OWN_TERM in set(out["own_reversal_spanning"]["term"])
    assert set(out["beta_comparison"]["market_proxy"]) == {"equal_weighted", "value_weighted"}
    assert set(out["beta_factor_controls"]["term"]) == {"mkt_rf"}
    assert len(out["drawdowns"]) and len(out["var_es"]) == 4 * 2 * 2 * 2 * 2


OWN_TERM = run_attribution.OWN_REVERSAL


def test_days_after_the_research_cutoff_leave_only_the_universe_regressions():
    cfg, daily, exposures, factors, windows, table = synthetic()
    full = analyse(cfg, daily, exposures, factors, windows, table, 252)
    tail = daily[next(iter(daily))].index[-6:]
    for frame in daily.values():
        frame.loc[tail, "market_return"] = np.nan
    rows = []
    for name, spec in cfg["books"].items():
        sub = daily[name].loc[windows[name]["evaluation"][0]:]
        ok = sub[sub["market_return"].notna()]
        beta = regression_beta(sub["net_pnl"].to_numpy(), sub["market_return"].to_numpy(),
                               run_attribution.newey_west_lag(len(sub), 5))
        assert len(ok) == len(sub) - 6
        rows.append({"signal": spec["signal"], "construction": spec["construction"], "window": "evaluation", **beta})
    table = pd.concat([table[table["window"] == TUNING_LABEL], pd.DataFrame(rows)], ignore_index=True)
    out = analyse(cfg, daily, exposures, factors, windows, table, 252)
    assert out["checks"]["passed"].all()
    pd.testing.assert_frame_equal(out["factor_loadings"], full["factor_loadings"])  # factor regressions keep the days
    keys = out["beta_comparison"].query("window == 'evaluation'")
    assert keys.query("market_proxy == 'equal_weighted'")["realised_beta"].notna().all()
    value_weighted = lambda f: f.query("market_proxy == 'value_weighted'")["realised_beta"].tolist()
    assert value_weighted(out["beta_comparison"]) == pytest.approx(value_weighted(full["beta_comparison"]))


def test_a_missing_factor_day_stops_the_analysis():
    cfg, daily, exposures, factors, windows, table = synthetic()
    factors.iloc[100, 2] = np.nan
    out = analyse(cfg, daily, exposures, factors, windows, table, 252)
    assert list(out) == ["checks"] and not out["checks"]["passed"].all()


def test_a_shifted_factor_file_fails_date_alignment():
    cfg, daily, exposures, factors, windows, table = synthetic()
    factors["mkt_rf"] = factors["mkt_rf"].shift(1).bfill()
    out = analyse(cfg, daily, exposures, factors, windows, table, 252)
    row = out["checks"].query("check == 'date_alignment'").iloc[0]
    assert not row["passed"]


def test_a_different_exposures_table_fails_reconciliation():
    cfg, daily, exposures, factors, windows, table = synthetic()
    table["realised_beta"] += 1e-3
    out = analyse(cfg, daily, exposures, factors, windows, table, 252)
    assert not out["checks"].query("check == 'reproduce_exposures'")["passed"].any()


def test_run_refuses_a_failed_check_without_writing_results(tmp_path, monkeypatch):
    cfg, daily, exposures, factors, windows, table = synthetic()
    table["realised_beta"] += 1.0
    raw = tmp_path / "raw"
    out = tmp_path / "reports"
    cfg["factors"]["raw_dir"], cfg["outputs"]["reports_dir"] = str(raw), str(out)
    monkeypatch.setattr(run_attribution, "load_attribution_config", lambda *_: cfg)
    monkeypatch.setattr(run_attribution, "provenance", lambda *a: {})
    monkeypatch.setattr(run_attribution, "verified_manifest", lambda *a: {})
    monkeypatch.setattr(run_attribution, "load_factors", lambda *a: factors)
    monkeypatch.setattr(run_attribution, "load_books", lambda *a: (daily, {n: None for n in daily}))
    monkeypatch.setattr(run_attribution, "book_windows", lambda *a: windows)
    monkeypatch.setattr(run_attribution, "beta_frames", lambda *a: {})
    monkeypatch.setattr(run_attribution, "analyse", lambda *a: analyse(cfg, daily, exposures, factors, windows, table, 252))
    monkeypatch.setattr(run_attribution.pd, "read_csv", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(run_attribution.json, "loads", lambda *_: {"config_sha256": ["x"]})
    monkeypatch.setattr(run_attribution, "file_sha256", lambda *_: "x")
    with pytest.raises(SystemExit, match="no results were written"):
        run_attribution.run(["--run"])
    assert [p.name for p in out.iterdir()] == ["checks.csv"]
