import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml
from test_backtest import COSTS, H, HOLDOUT, LAG, MCFG, PCFG, RCFG, runner_args, write_runner_inputs

from src import holdout, run_holdout
from src.backtest import run_backtest
from src.holdout import (
    HOLDOUT_CONFIG_PATH,
    interval_outcome,
    last_score_date,
    load_holdout_panel,
    open_lock,
    t_outcome,
)
from src.models import baseline_weights, zero_fill
from src.run_holdout import Paths, holdout_tables, recorded_parameters, refit_scores, run_opened
from src.run_models import Data
from src.run_portfolio import Inputs, builder_for, calendar_extension, fold_ics, market_data
from src.second_round import liquid_scores
from src.signals import load_research_panel
from src.utils import load_config

REAL = yaml.safe_load(open(HOLDOUT_CONFIG_PATH))
LABELS = REAL["outcomes"]["headline"]["labels"]
DIFF = REAL["outcomes"]["lightgbm_minus_b0"]["labels"]


# --------------------------------------------------------------------------- outcomes, dates and the lock


@pytest.mark.parametrize("t, label", [(2.0, "holds_up"), (3.1, "holds_up"), (1.99, "inconclusive"), (0.0, "inconclusive"),
                                      (-1.99, "inconclusive"), (-2.0, "fails"), (float("nan"), "inconclusive")])
def test_t_outcome_covers_each_case(t, label):
    assert t_outcome(t, 2.0, LABELS) == label


@pytest.mark.parametrize("low, high, label", [(0.1, 1.0, "lightgbm_better"), (-1.0, -0.1, "b0_better"),
                                              (-0.5, 0.5, "no_difference_shown"), (0.0, 1.0, "no_difference_shown")])
def test_interval_outcome_covers_each_case(low, high, label):
    assert interval_outcome(low, high, DIFF) == label


def test_the_last_score_date_leaves_room_to_close_the_position():
    calendar = pd.bdate_range("2026-01-01", periods=20)
    assert last_score_date(calendar, 1, 5) == calendar[13]


def test_the_lock_refuses_a_second_opening(tmp_path):
    lock = tmp_path / "reports" / "opened.json"
    open_lock(lock, {"code_commit": "abc"})
    stamp = json.loads(lock.read_text())
    assert stamp["code_commit"] == "abc" and "opened_at" in stamp
    with pytest.raises(SystemExit, match="already been opened"):
        open_lock(lock, {})


def test_holdout_rows_cannot_be_read_before_the_lock(tmp_path):
    with pytest.raises(SystemExit, match="after the one-shot lock"):
        load_holdout_panel(["a"], tmp_path / "panel.parquet", tmp_path / "opened.json")


def test_the_command_stays_closed_without_its_flag(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("nothing may load before the flag is checked")

    monkeypatch.setattr(run_holdout, "load_holdout_config", refuse)
    with pytest.raises(SystemExit, match="stays closed"):
        run_holdout.run([])


def stub_config(monkeypatch, tmp_path, identity):
    """The real entry point with its loaders stubbed: nothing here touches the real panel."""
    cfg = copy.deepcopy(REAL)
    cfg["one_shot"]["lock_file"] = str(tmp_path / "opened.json")
    monkeypatch.setattr(run_holdout, "load_holdout_config", lambda *_: cfg)
    monkeypatch.setattr(run_holdout, "load_portfolio_config", lambda *a: PCFG)
    monkeypatch.setattr(run_holdout, "provenance", lambda *a: identity)
    monkeypatch.setattr(run_holdout, "run_opened", lambda *a: pytest.fail("the holdout must not open"))
    return cfg


def test_the_command_refuses_uncommitted_code(tmp_path, monkeypatch):
    stub_config(monkeypatch, tmp_path, {"code_dirty": True, "panel_sha256": "x"})
    with pytest.raises(SystemExit, match="uncommitted"):
        run_holdout.run(["--open-holdout"])
    assert not (tmp_path / "opened.json").exists()


def test_the_command_refuses_a_changed_panel(tmp_path, monkeypatch):
    stub_config(monkeypatch, tmp_path, {"code_dirty": False, "panel_sha256": "other"})
    with pytest.raises(SystemExit, match="panel has changed"):
        run_holdout.run(["--open-holdout"])
    assert not (tmp_path / "opened.json").exists()


def test_the_command_cannot_run_twice(tmp_path, monkeypatch):
    cfg = stub_config(monkeypatch, tmp_path, {"code_dirty": False, "panel_sha256": REAL["pinned"]["panel_sha256"]})
    (tmp_path / "opened.json").write_text("{}")
    with pytest.raises(SystemExit, match="already been opened"):
        run_holdout.run(["--open-holdout"])
    assert cfg["one_shot"]["lock_file"]


def test_no_flag_for_a_dirty_tree_exists():
    assert "allow-dirty" not in open(run_holdout.__file__).read()


# --------------------------------------------------------------------------- the holdout tables


def book_frame(rng, n, drift):
    dates = pd.bdate_range("2022-01-03", periods=n)
    gross = rng.normal(drift, 0.002, n)
    frame = pd.DataFrame({"gross_pnl": gross, "commission": 1e-6, "spread": 4e-6, "impact": 2e-5, "borrow": 1e-6,
                          "gross_exposure": 1.5, "net_exposure": 0.0, "beta_ex_ante": 0.0, "turnover": 0.01,
                          "market_return": rng.normal(0, 0.01, n)}, index=dates)
    frame["cost"] = frame["commission"] + frame["spread"] + frame["impact"]
    frame["net_pnl"] = frame["gross_pnl"] - frame["cost"] - frame["borrow"]
    return frame


def small_config():
    cfg = copy.deepcopy(REAL)
    cfg["bootstrap"]["n_resamples"] = 500
    return cfg


def test_holdout_tables_label_each_book_and_the_difference():
    rng = np.random.default_rng(0)
    totals = {"headline": book_frame(rng, 1200, 0.001), "benchmark": book_frame(rng, 1200, 0.0),
              "candidate": book_frame(rng, 1200, -0.001)}
    out = holdout_tables(small_config(), totals, 1.0, 25.0, 5, 252)
    books = out["books"].set_index("book")
    assert books.loc["headline", "outcome"] == "holds_up" and books.loc["candidate", "outcome"] == "fails"
    assert pd.isna(books.loc["benchmark", "outcome"])  # the comparator has no declared outcome of its own
    assert out["lightgbm_minus_b0"]["outcome"].iloc[0] == "lightgbm_better"
    assert set(out["by_year"]["book"]) == set(totals) and len(out["by_year"]) >= 12
    assert {"breakeven_impact_y", "breakeven_borrow_bps_per_year", "realised_beta", "net_sharpe_ci_low"} <= set(books.columns)


def test_holdout_tables_refuse_books_on_different_dates():
    rng = np.random.default_rng(1)
    totals = {"headline": book_frame(rng, 100, 0.0), "benchmark": book_frame(rng, 99, 0.0), "candidate": book_frame(rng, 100, 0.0)}
    with pytest.raises(ValueError, match="share the holdout dates"):
        holdout_tables(small_config(), totals, 1.0, 25.0, 5, 252)


# --------------------------------------------------------------------------- the whole run on a synthetic panel


FAST = copy.deepcopy(MCFG)
FAST["models"]["lightgbm"].update(num_leaves=[3], min_data_in_leaf=[50], lambda_l2=[1.0])
FAST["models"]["lightgbm"]["n_estimators"] = {"max": 40, "early_stopping_rounds": 10}
FAST["models"]["num_threads"] = 1


def enrich(root):
    """A holdout of January to June 2008 on the synthetic panel, with the columns the models need."""
    panel = pd.read_parquet(root / "panel.parquet")
    dates = panel.index.get_level_values("date")
    label = panel["fwd_ret_5d"]
    panel["fwd_ret_5d_xs"] = label - label.groupby(dates).transform("mean")
    panel["fwd_terminal_5d"] = False
    panel["fwd_ret_1d"] = panel["ret_1d"]
    panel.to_parquet(root / "panel.parquet")
    rcfg = yaml.safe_load(open(root / "research.yaml"))
    rcfg["holdout_start"] = "2008-01-01"
    (root / "research.yaml").write_text(yaml.safe_dump(rcfg))
    return panel


def design(root, chosen):
    """The real design with the frozen lambdas, dates and test years of the synthetic panel."""
    cfg = copy.deepcopy(REAL)
    chosen = chosen.set_index("signal")
    for name in ("headline", "benchmark", "candidate"):
        signal = cfg["books"][name]["signal"]
        cfg["books"][name]["lambdas"] = [float(chosen.loc[signal, "lambda_risk"]), float(chosen.loc[signal, "lambda_tc"])]
    cfg["books"]["candidate"]["liquid_fraction"] = 0.5
    cfg["window"]["last_panel_date"] = "2008-06-30"
    cfg["model_refit"]["test_years"] = [2008]
    cfg["bootstrap"]["n_resamples"] = 500
    return cfg


def configs(root):
    fcfg = load_config()
    rcfg = copy.deepcopy(RCFG)
    rcfg.update(yaml.safe_load(open(root / "research.yaml")))
    pcfg = yaml.safe_load(open(root / "portfolio.yaml"))
    pcfg["constructions"]["optimised"]["alpha"]["ic_table"] = str(root / "hyper.csv")
    return pcfg, rcfg, fcfg


def build_recorded(root):
    """First-round outputs whose 2007 predictions come from the same refit the holdout repeats."""
    panel = enrich(root)
    pcfg, rcfg, fcfg = configs(root)
    h, lag = 5, 1
    cutoff = run_holdout.research_cutoff(root / "panel.parquet", rcfg, fcfg)
    params = pd.DataFrame([{"model": "lightgbm", "horizon": 5, "variant": "rank", "fold_id": f, "validation_ic": 0.02,
                            "num_leaves": 3, "min_data_in_leaf": 50, "lambda_l2": 1.0, "learning_rate": 0.05,
                            "feature_fraction": 0.7} for f in (0, 1)])
    params.to_csv(root / "model_hyper.csv", index=False)
    oof = pd.read_parquet(root / "oof.parquet")
    features = list(FAST["features"]["columns"])
    data = Data(panel, features, [h])
    cfg = design(root, pd.DataFrame({"signal": ["lightgbm_h5_rank", "B0_reversal_h5"], "lambda_risk": 10.0, "lambda_tc": 1.0}))
    last = run_holdout.last_score_date(data.dates, lag, h)
    target = data.targets[h]["rank"]
    from src.run_models import walk_forward
    from src.validation import walk_forward_folds
    folds = walk_forward_folds(data.dates, h, lag, 2007, 2, last)
    fixed = {folds[0].fold_id: recorded_parameters(params, 0, 5, "rank", FAST)}
    pred, _, _ = walk_forward("lightgbm", data, np.arange(data.X.shape[1]), target, h, folds[:1], FAST, lag,
                              rcfg["min_names_per_date"], fixed=fixed)
    frame = pd.DataFrame({"lightgbm": pred}, index=data.index).dropna().reindex(oof.index)
    oof["lightgbm_h5_rank"] = frame["lightgbm"].astype("float32").fillna(oof["lightgbm_h5_rank"])
    score = zero_fill(panel[features].to_numpy()) @ baseline_weights("B0_reversal", features, FAST)
    oof["B0_reversal_h5"] = pd.Series(score, index=panel.index).reindex(oof.index).astype("float32")
    oof.to_parquet(root / "oof.parquet")

    from src.run_portfolio import run
    args = ["--signals", "lightgbm_h5_rank,B0_reversal_h5", "--constructions", "optimised"]
    run(runner_args(root, "--tune", *args))
    run(runner_args(root, "--evaluate", *args))
    chosen = pd.read_csv(root / "reports" / "tuning" / "selected_hyperparameters.csv")
    cfg = design(root, chosen)

    research = load_research_panel(None, root / "panel.parquet", rcfg, fcfg, cutoff)
    preds = oof[oof.index.get_level_values("date") <= cutoff][["lightgbm_h5_rank", "B0_reversal_h5"]]
    inp = Inputs(research.drop(columns=["fwd_ret_21d"], errors="ignore"), preds, pcfg, features)
    extension = calendar_extension(root / "panel.parquet", cutoff, lag + h, rcfg["holdout_start"])
    data_c, start = market_data(inp, "lightgbm_h5_rank", h, lag, pcfg["risk_model"]["window"], extension)
    data_c = run_holdout.replace(data_c, score=liquid_scores(data_c.score, data_c.adv, 0.5))
    ics = fold_ics(inp, pcfg["signals"]["lightgbm_h5_rank"], h, lag, FAST, pd.read_csv(root / "hyper.csv"), 10, 0.0)
    build = builder_for("optimised", pcfg, rcfg, h, ics, data_c.dates, tuple(cfg["books"]["candidate"]["lambdas"]))
    res = run_backtest(data_c, build, COSTS, h, lag, h, rcfg["holdout_start"], start, record_weights=False)
    res.total.reset_index(names="date").to_parquet(root / "candidate.parquet")
    return cfg, pcfg, rcfg, fcfg


def paths_for(root):
    return Paths(root / "panel.parquet", root / "oof.parquet", root / "model_hyper.csv",
                 str(root / "processed" / "portfolio_daily_returns.parquet"), root / "candidate.parquet",
                 root / "holdout", root / "holdout_data", root / "holdout" / "opened.json")


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    root = write_runner_inputs(tmp_path_factory.mktemp("holdout") / "base")
    return root, *build_recorded(root)


def fresh(recorded, tmp_path):
    source, cfg, pcfg, rcfg, fcfg = recorded
    root = tmp_path / "run"
    shutil.copytree(source, root)
    paths = paths_for(root)
    open_lock(paths.lock, {"code_commit": "test"})
    return root, cfg, pcfg, rcfg, fcfg, paths


def test_the_whole_run_on_synthetic_data(recorded, tmp_path):
    root, cfg, pcfg, rcfg, fcfg, paths = fresh(recorded, tmp_path)
    out = run_opened(cfg, pcfg, rcfg, fcfg, FAST, paths, {"code_commit": "test"})
    assert out["checks"]["passed"].all()
    assert {"recorded_net_sharpe", "predictions_lightgbm_h5_rank", "predictions_B0_reversal_h5"} == set(out["checks"]["check"])
    books = out["books"].set_index("book")
    assert list(books.index) == ["headline", "benchmark", "candidate"]
    assert (pd.to_datetime(books["start"]) >= pd.Timestamp("2008-01-01")).all()
    assert (pd.to_datetime(books["end"]) == pd.Timestamp("2008-06-30")).all()
    assert books.loc[["headline", "candidate"], "outcome"].isin(["holds_up", "inconclusive", "fails"]).all()
    assert (books.loc["headline", "net_sharpe"] != books.loc["candidate", "net_sharpe"])
    assert out["lightgbm_minus_b0"]["outcome"].iloc[0] in DIFF.values()
    assert out["refit_hyperparameters"]["fold_id"].tolist() == [1]
    for name in ("books", "by_year", "lightgbm_minus_b0", "model_ic", "checks", "refit_hyperparameters"):
        assert (paths.out / f"{name}.csv").exists()
    assert json.loads((paths.out / "run_metadata.json").read_text())["last_score_date"] < "2008-06-30"


def test_a_failed_check_writes_no_holdout_statistic(recorded, tmp_path):
    root, cfg, pcfg, rcfg, fcfg, paths = fresh(recorded, tmp_path)
    oof = pd.read_parquet(paths.predictions)
    oof["lightgbm_h5_rank"] += 0.1
    oof.to_parquet(paths.predictions)
    with pytest.raises(SystemExit, match="no holdout statistic"):
        run_opened(cfg, pcfg, rcfg, fcfg, FAST, paths, {})
    assert sorted(p.name for p in paths.out.iterdir()) == ["checks.csv", "opened.json"]


def test_a_later_label_cannot_change_a_score(recorded, tmp_path):
    """Poison every label that ends in the test year: the model can't have trained on it."""
    root, cfg, pcfg, rcfg, fcfg, paths = fresh(recorded, tmp_path)
    panel = pd.read_parquet(paths.panel)
    dates = panel.index.get_level_values("date")
    calendar = dates.unique().sort_values()
    first = int(calendar.searchsorted(pd.Timestamp("2008-01-01")))
    late = dates >= calendar[first - 6]
    other = panel.copy()
    for column in ("fwd_ret_5d", "fwd_ret_5d_xs"):
        other.loc[late, column] = np.random.default_rng(3).normal(0, 100, late.sum()).astype("float32")
    recorded_table, oof = pd.read_csv(paths.model_hyperparameters), pd.read_parquet(paths.predictions)
    cutoff, last = pd.Timestamp("2007-11-30"), run_holdout.last_score_date(calendar, 1, 5)
    a, _, _ = refit_scores(Data(panel, list(FAST["features"]["columns"]), [5]), cfg, FAST, recorded_table, oof, cutoff, last, 1, 10)
    b, _, _ = refit_scores(Data(other, list(FAST["features"]["columns"]), [5]), cfg, FAST, recorded_table, oof, cutoff, last, 1, 10)
    pd.testing.assert_frame_equal(a, b)
