import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import yaml
from test_backtest import COSTS, HOLDOUT, LAG, H, PCFG, runner_args, synthetic, write_runner_inputs

from src.backtest import run_backtest
from src.portfolio import volscaled_builder
from src.second_round import (
    SECOND_ROUND_CONFIG_PATH,
    band_builder,
    band_trade,
    enumerate_variants,
    holdout_candidate,
    liquid_scores,
    load_second_round_config,
    scale_down,
    select_variants,
)

# --------------------------------------------------------------------------- the variants


def grid_rows():
    return pd.DataFrame([
        {"signal": s, "grid_order": i, "lambda_risk": r, "lambda_tc": t}
        for s in ("lightgbm_h5_rank", "B0_reversal_h5")
        for i, (r, t) in enumerate((r, t) for r in (3.0, 10.0, 30.0, 100.0) for t in (0.5, 1.0, 2.0, 4.0))])


def test_every_grid_point_is_enumerated_once_with_its_recorded_book_marked():
    variants = enumerate_variants(load_second_round_config(), grid_rows())
    count = Counter(v.family for v in variants)
    assert count == {"no_trade_band": 10, "liquid_universe": 12, "wider_lambda_grid": 2 * 16 + 5 + 3}
    assert len({v.name for v in variants}) == len(variants)
    assert sum(v.recorded for v in variants if v.family == "no_trade_band") == 2
    assert sum(v.recorded for v in variants if v.family == "liquid_universe") == 4
    assert sum(v.recorded for v in variants if v.family == "wider_lambda_grid") == 32


def test_liquid_scores_keep_the_top_share_of_each_date():
    score = np.array([[1.0, 2.0, 3.0, np.nan], [4.0, 5.0, 6.0, 7.0]])
    adv = np.array([[10.0, 30.0, 20.0, 99.0], [1.0, 2.0, 3.0, 4.0]])
    out = liquid_scores(score, adv, 0.5)
    assert np.array_equal(np.isfinite(out), [[False, True, True, False], [False, False, True, True]])
    assert out[0, 1] == 2.0  # kept scores are unchanged; the construction standardises them afterwards


def test_liquid_scores_at_one_leave_every_name_scored():
    score = np.array([[1.0, 2.0]])
    assert liquid_scores(score, np.array([[np.nan, 1.0]]), 1.0) is score


def test_liquid_scores_use_only_their_own_date():
    rng = np.random.default_rng(0)
    score, adv = rng.normal(size=(5, 20)), rng.uniform(1, 2, (5, 20))
    later = adv.copy()
    later[3:] = rng.uniform(1, 2, (2, 20))
    np.testing.assert_array_equal(liquid_scores(score, adv, 0.25)[:3], liquid_scores(score, later, 0.25)[:3])


# --------------------------------------------------------------------------- the band


TARGET = np.array([0.04, -0.04, 0.02, -0.02, 0.0, 0.0])
BETA = np.array([1.0, 1.2, 0.8, 0.9, 1.1, 1.0])


def test_band_zero_is_the_target():
    current = np.array([0.03, -0.03, 0.01, -0.01, 0.0, 0.0])
    assert np.array_equal(band_trade(current, TARGET, 0.0, BETA), TARGET)


def test_band_moves_a_name_only_to_the_edge_of_the_band():
    target = np.array([0.04, -0.04, 0.04, -0.04, 0.04, -0.04])
    current = np.array([0.0, 0.0, 0.0, 0.0, 0.04, -0.04])
    w = band_trade(current, target, 0.25, np.ones(6))  # unit 0.04, width 0.01: moves 0.03, not 0.04
    assert w[:4] == pytest.approx([0.03, -0.03, 0.03, -0.03])
    assert w[4] == 0.04 and w[5] == -0.04


def test_band_sells_a_name_the_target_does_not_hold():
    target = np.array([0.04, -0.04, 0.04, -0.04, 0.0, 0.0])
    current = np.array([0.0, 0.0, 0.0, 0.0, 0.03, -0.03])
    w = band_trade(current, target, 0.25, np.ones(6))
    assert w[4] == 0.0 and w[5] == 0.0              # sold in full, even though the band is wider than nothing


def test_band_restores_zero_dollars_and_beta_when_enough_names_trade():
    rng = np.random.default_rng(1)
    target, current, beta = rng.normal(0, 0.01, 200), rng.normal(0, 0.01, 200), rng.normal(1, 0.3, 200)
    w = band_trade(current, target, 0.5, beta)
    assert w.sum() == pytest.approx(0.0, abs=1e-12) and beta @ w == pytest.approx(0.0, abs=1e-12)
    inside = np.abs(target - current) <= 0.5 * np.abs(target).mean()
    assert inside.any() and np.array_equal(w[inside], current[inside])


def test_too_few_traded_names_hold():
    current = TARGET.copy()
    current[0] += 0.05
    assert band_trade(current, TARGET, 1.0, BETA) is None


def test_scale_down_never_scales_up():
    w = np.array([0.5, -0.5])
    out, scaled = scale_down(w, 0.05, 0.10, 2.0)
    assert not scaled and out is w
    out, scaled = scale_down(w, 0.20, 0.10, 2.0)
    assert scaled and out == pytest.approx(w / 2)
    out, scaled = scale_down(w * 4, 0.05, 0.10, 2.0)
    assert scaled and abs(out).sum() == pytest.approx(2.0)


def neutral(build, data):
    return run_backtest(data, build, COSTS, H, LAG, H, HOLDOUT, 20, record_weights=False)


def test_band_zero_reproduces_the_recorded_construction():
    data, _ = synthetic(T=90)
    base = neutral(volscaled_builder(PCFG, 10, 252), data)
    band = neutral(band_builder(PCFG, 0.0, 10, 252, Counter()), data)
    pd.testing.assert_frame_equal(base.total, band.total)


def test_a_wider_band_trades_less_and_the_counts_are_kept():
    data, _ = synthetic(T=90)
    turnover, counts = [], Counter()
    for band in (0.0, 1.0, 4.0):
        turnover.append(neutral(band_builder(PCFG, band, 10, 252, counts), data).total["turnover"].sum())
    assert turnover[0] > turnover[1] > turnover[2]
    assert set(counts) <= {"band_hold", "band_scaled"}


def test_the_band_cannot_see_the_future():
    data, _ = synthetic(T=90)
    cut = 60
    later = data.__class__(data.dates, data.tickers, data.score.copy(), data.ret.copy(), data.label.copy(),
                           data.vol.copy(), data.beta.copy(), data.adv.copy())
    rng = np.random.default_rng(9)
    for name in ("score", "vol", "beta", "adv"):
        block = getattr(later, name)
        block[cut + 1:] = np.where(np.isfinite(block[cut + 1:]), rng.normal(1, 1, block[cut + 1:].shape), np.nan)
    later.ret[cut + 1:] = rng.normal(0, 1, later.ret[cut + 1:].shape)
    a = neutral(band_builder(PCFG, 1.0, 10, 252, Counter()), data)
    b = neutral(band_builder(PCFG, 1.0, 10, 252, Counter()), later)
    pd.testing.assert_frame_equal(a.total.iloc[:cut - 8], b.total.iloc[:cut - 8])


# --------------------------------------------------------------------------- selection


def grid_frame():
    base = {"construction": "neutral_volscaled", "fraction": np.nan, "lambda_risk": np.nan, "lambda_tc": np.nan}
    rows = [{"family": "no_trade_band", "signal": "s", "variant": f"b{b}", "grid_order": i, "recorded": b == 0, "band": b, "net_sharpe": n, **base}
            for i, (b, n) in enumerate([(0.0, -1.0), (1.0, 0.2), (2.0, 0.2), (4.0, 0.1)])]
    return pd.DataFrame(rows)


def test_selection_takes_the_highest_tuning_sharpe_and_breaks_ties_by_order():
    row = select_variants(grid_frame()).iloc[0]
    assert row["band"] == 1.0 and not row["on_grid_edge"]


def test_selection_flags_a_choice_on_the_edge():
    grid = grid_frame()
    grid.loc[grid["band"] == 4.0, "net_sharpe"] = 0.9
    assert select_variants(grid).iloc[0]["on_grid_edge"]


def test_the_recorded_point_can_win():
    grid = grid_frame()
    grid.loc[grid["band"] == 0.0, "net_sharpe"] = 0.9
    row = select_variants(grid).iloc[0]
    assert row["recorded"] and row["on_grid_edge"]


def test_a_missing_sharpe_never_wins():
    grid = grid_frame()
    grid.loc[grid["band"] == 1.0, "net_sharpe"] = np.nan
    assert select_variants(grid).iloc[0]["band"] == 2.0


def selected(rows):
    return pd.DataFrame([{"variant": v, "signal": s, "recorded": r, "net_sharpe": n} for v, s, r, n in rows])


def test_the_candidate_is_the_best_chosen_variant_of_the_primary_signal():
    out = holdout_candidate(selected([("a", "L", False, 0.4), ("b", "L", False, 0.5), ("c", "B", False, 0.9)]), "L", 0.345)
    assert out["variant"].tolist() == ["b"]


def test_there_is_no_candidate_below_the_headline():
    assert holdout_candidate(selected([("a", "L", False, 0.345), ("b", "L", False, 0.2)]), "L", 0.345).empty


def test_a_recorded_book_is_never_a_candidate():
    assert holdout_candidate(selected([("a", "L", True, 0.9)]), "L", 0.345).empty


# --------------------------------------------------------------------------- the runner on synthetic data


def recorded_inputs(root):
    """A first-round run on the synthetic panel, and a second-round config that points at it."""
    from src.run_portfolio import run

    args = ["--signals", "lightgbm_h5_rank,B0_reversal_h5"]
    run(runner_args(root, "--tune", *args))
    run(runner_args(root, "--evaluate", *args))
    reports = root / "reports"
    chosen = pd.read_csv(reports / "tuning" / "selected_hyperparameters.csv")
    chosen.to_csv(root / "hyperparameters.csv", index=False)
    meta = json.loads((reports / "run_metadata.json").read_text())
    meta["config_sha256"] = [meta["config_sha256"]]
    (root / "recorded_meta.json").write_text(json.dumps(meta))

    cfg = yaml.safe_load(open(SECOND_ROUND_CONFIG_PATH))
    cfg["recorded"] = {"hyperparameters": str(root / "hyperparameters.csv"),
                       "lambda_grid": str(reports / "tuning" / "lambda_grid.csv"),
                       "summary": str(reports / "summary.csv"),
                       "portfolio_metadata": str(root / "recorded_meta.json"),
                       "daily_returns": str(root / "processed" / "portfolio_daily_returns.parquet")}
    cfg["no_trade_band"]["bands"] = [0.0, 4.0]
    cfg["liquid_universe"]["fractions"] = [0.5, 1.0]
    cfg["liquid_universe"]["constructions"] = ["neutral_volscaled", "optimised"]
    cfg["bootstrap"]["n_resamples"] = 1000
    for signal, row in chosen.set_index("signal").iterrows():
        point = [float(row["lambda_risk"]), float(row["lambda_tc"])]
        cfg["wider_lambda_grid"]["points"][signal] = {"extends_from": point, "added": [[point[0] * 3, point[1]]]}
    (root / "second.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def second_args(root, *extra):
    return ["--config", str(root / "second.yaml"), "--portfolio-config", str(root / "portfolio.yaml"),
            "--research-config", str(root / "research.yaml"), "--panel", str(root / "panel.parquet"),
            "--predictions", str(root / "oof.parquet"), "--ic-table", str(root / "hyper.csv"),
            "--out", str(root / "second"), "--data-out", str(root / "second_data"), "--allow-dirty-config", *extra]


@pytest.fixture
def relaxed_loader(monkeypatch):
    """The synthetic config changes declared values, so the real loader's bootstrap equality is relaxed."""
    import src.run_second_round as runner
    import src.second_round as second

    monkeypatch.setattr(second, "_check_evaluation_and_bootstrap", lambda *a: None)
    return runner


def test_the_whole_round_runs_in_order_with_its_guards(tmp_path, relaxed_loader):
    runner = relaxed_loader
    root = write_runner_inputs(tmp_path / "round")
    recorded_inputs(root)

    with pytest.raises(SystemExit, match="--select"):
        runner.run(second_args(root, "--evaluate"))

    first = runner.run(second_args(root, "--tune", "--shard", "0/2"))
    second = runner.run(second_args(root, "--tune", "--shard", "1/2"))
    tuned = pd.concat([first["variant_grid"], second["variant_grid"]])
    assert not tuned["family"].eq("wider_lambda_grid").all() and pd.to_datetime(tuned["end"]).max() <= pd.Timestamp("2007-04-02")
    assert (tuned.loc[tuned["recorded"], "recorded_gap"] <= 1e-6).all()

    picked = runner.run(second_args(root, "--select"))
    assert picked["checks"]["passed"].all()
    assert {"recorded_book", "every_variant_tuned", "tuning_isolation"} == set(picked["checks"]["check"])
    assert len(picked["selected_variants"]) == 2 + 4 + 2
    assert len(picked["holdout_candidate"]) <= 1

    runner.run(second_args(root, "--evaluate"))
    out = runner.run(second_args(root, "--report"))["evaluation"]
    assert (out["evidence"] == "second_look").all() and len(out) == 8
    assert pd.to_datetime(out["end"]).max() < pd.Timestamp("2008-03-01")
    kept = out[out["recorded"]]
    assert (kept["difference"].abs() < 1e-12).all() and (kept["difference_ci_low"] == 0).all()
    assert (out["net_sharpe_ci_low"] < out["net_sharpe_ci_high"]).all()


def test_tuning_never_reads_past_the_tuning_window(tmp_path, relaxed_loader):
    runner = relaxed_loader
    tuning_end = pd.bdate_range("2007-01-01", "2007-04-01")[-1]
    clean = write_runner_inputs(tmp_path / "clean")
    poisoned = write_runner_inputs(tmp_path / "poisoned", poison_after=tuning_end)
    recorded_inputs(clean)
    cfg = yaml.safe_load(open(clean / "second.yaml"))
    (poisoned / "second.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (poisoned / "reports").mkdir()
    a = runner.run(second_args(clean, "--tune"))["variant_grid"]
    args = second_args(poisoned, "--tune")
    args[args.index("--panel") + 1] = str(poisoned / "panel.parquet")
    b = runner.run(args)["variant_grid"]
    columns = ["variant", "net_sharpe", "gross_sharpe", "annual_turnover"]
    pd.testing.assert_frame_equal(a[columns], b[columns])


def test_a_failed_check_writes_no_choices(tmp_path, relaxed_loader):
    runner = relaxed_loader
    root = write_runner_inputs(tmp_path / "failed")
    recorded_inputs(root)
    runner.run(second_args(root, "--tune"))
    shard = root / "second" / "tuning" / "shards" / "variant_grid_0_of_1.csv"
    frame = pd.read_csv(shard)
    frame.loc[frame["recorded"] & frame["recorded_gap"].notna(), "recorded_gap"] = 1.0
    frame.to_csv(shard, index=False)
    with pytest.raises(SystemExit, match="no choices were written"):
        runner.run(second_args(root, "--select"))
    assert (root / "second" / "checks.csv").exists() and not (root / "second" / "tuning" / "selected_variants.csv").exists()


def test_evaluation_refuses_a_selection_from_another_config(tmp_path, relaxed_loader):
    runner = relaxed_loader
    root = write_runner_inputs(tmp_path / "other")
    recorded_inputs(root)
    runner.run(second_args(root, "--tune"))
    runner.run(second_args(root, "--select"))
    cfg = yaml.safe_load(open(root / "second.yaml"))
    cfg["no_trade_band"]["bands"] = [0.0, 2.0]
    (root / "second.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(SystemExit, match="another second-round config"):
        runner.run(second_args(root, "--evaluate"))
