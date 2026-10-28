import copy

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import ElasticNet, Ridge

from src.diagnostics import rank_by_date
from src.model_evaluation import daily_ic_long, oos_r2, paired_ic_test, permutation_importance
from src.models import (
    ElasticNetModel,
    FamaMacBethForecast,
    LightGBMModel,
    LinearScore,
    RidgeModel,
    ScaledTarget,
    build_model,
    family_features,
    hyperparameter_grid,
    lightgbm_available,
    load_models_config,
    primary_variant,
    rank_target,
)
from src.signals import Grid, cross_sectional_ic, load_research_config
from src.validation import Fold, assert_no_leakage, purged, thin, walk_forward_folds

RCFG = load_research_config()
MCFG = load_models_config(rcfg=RCFG)
needs_lightgbm = pytest.mark.skipif(not lightgbm_available(), reason="lightgbm cannot be imported (libomp)")


def panel_rows(n_dates=60, n_names=80, seed=0):
    rng = np.random.default_rng(seed)
    date_ids = np.repeat(np.arange(n_dates), n_names)
    X = rng.standard_normal((n_dates * n_names, 3))
    return rng, date_ids, X


# --------------------------------------------------------------------------- walk-forward folds


def test_folds_are_purged_annual_and_expanding():
    dates = pd.bdate_range("1999-01-04", "2021-11-30")
    lag = 1
    for h in (5, 21):
        folds = walk_forward_folds(dates, h, lag, 2007, 2, dates[-1])
        assert [f.test_year for f in folds] == list(range(2007, 2022))
        for prev, fold in zip(folds, folds[1:]):
            assert len(fold.train) > len(prev.train)  # expanding
        for fold in folds:
            first_test = fold.test[0]
            assert dates[first_test] == dates[dates.year == fold.test_year][0]
            assert fold.train.max() + lag + h == first_test - 1  # the last label ends the day before
            assert fold.train[0] == 0
            assert dates[fold.validation[0]] >= pd.Timestamp(fold.test_year - 2, 1, 1)
            assert dates[fold.validation[0] - 1] < pd.Timestamp(fold.test_year - 2, 1, 1)
            assert fold.inner_train.max() + lag + h < fold.validation.min()
        assert folds[-1].test[-1] == len(dates) - 1


def test_leakage_assertion_catches_an_unpurged_fold():
    train = np.arange(0, 100)
    bad = Fold(0, 2007, train, np.arange(0, 70), np.arange(80, 100), np.arange(100, 120))
    with pytest.raises(AssertionError):
        assert_no_leakage(bad, 5, 1)
    good = Fold(0, 2007, purged(train, 100, 5, 1), np.arange(0, 70), np.arange(80, 94), np.arange(100, 120))
    assert_no_leakage(good, 5, 1)


def test_date_stride_keeps_the_same_dates_in_every_fold():
    a, b = thin(np.arange(0, 50), 5), thin(np.arange(0, 80), 5)
    assert set(a) <= set(b) and list(a) == list(range(0, 50, 5))
    assert list(thin(np.arange(3, 9), 1)) == list(range(3, 9))


# --------------------------------------------------------------------------- config and targets


def test_config_has_the_preregistered_shape():
    assert MCFG["targets"]["horizons"] == [5, 21]
    assert primary_variant(MCFG) == "rank"
    assert len(MCFG["features"]["columns"]) == 16 and "z_dollar_volume_21d" not in MCFG["features"]["columns"]
    assert len(hyperparameter_grid("ridge", MCFG)) == 6
    assert len(hyperparameter_grid("elastic_net", MCFG)) == 15
    assert len(hyperparameter_grid("lightgbm", MCFG)) == 8
    assert hyperparameter_grid("elastic_net", MCFG)[0] == {"alpha": 0.0001, "l1_ratio": 0.1}
    fam = family_features(MCFG["features"]["columns"], RCFG)
    assert fam["liquidity_size"] == ["z_log_dollar_volume_21d"] and len(fam) == 6


def test_config_rejects_choices_the_code_does_not_implement(tmp_path):
    def write(mutate):
        cfg = yaml.safe_load(open("configs/models.yaml"))
        mutate(cfg)
        path = tmp_path / "m.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    with pytest.raises(ValueError):
        load_models_config(write(lambda c: c["walk_forward"].update(scheme="rolling")), RCFG)
    with pytest.raises(ValueError):
        load_models_config(write(lambda c: c["targets"].update(horizons=[2])), RCFG)
    with pytest.raises(ValueError):
        load_models_config(write(lambda c: c["models"]["ridge"].update(alpha=[])), RCFG)
    with pytest.raises(ValueError):
        load_models_config(write(lambda c: c["evaluation"]["multiple_testing"].update(method="bh")), RCFG)


def test_rank_target_matches_the_wide_rank_transform():
    rng = np.random.default_rng(1)
    y = rng.standard_normal((5, 30))
    y[2, :4] = np.nan
    date_ids = np.repeat(np.arange(5), 30)
    long = rank_target(y.ravel(), date_ids).reshape(5, 30)
    wide = rank_by_date(pd.DataFrame(y)).to_numpy()
    assert np.allclose(long, wide, equal_nan=True)
    assert np.nanmax(long) < 0.5 and np.nanmin(long) > -0.5


# --------------------------------------------------------------------------- models


def test_ridge_matches_sklearn_with_a_per_observation_penalty():
    rng, _, X = panel_rows()
    y = X @ np.array([0.3, -0.2, 0.0]) + rng.standard_normal(len(X))
    X[::17, 1] = np.nan  # zero-filled
    ours = RidgeModel(0.5).fit(X, y)
    ref = Ridge(alpha=0.5 * len(y)).fit(np.nan_to_num(X), y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-10)
    assert np.allclose(ours.predict(X), ref.predict(np.nan_to_num(X)), atol=1e-10)


def test_elastic_net_matches_sklearn_and_zeroes_a_noise_feature():
    rng, _, X = panel_rows()
    y = X @ np.array([0.3, -0.2, 0.0]) + rng.standard_normal(len(X))
    ours = ElasticNetModel(0.05, 0.9, 10000, 1e-8, 42).fit(X, y)
    ref = ElasticNet(alpha=0.05, l1_ratio=0.9, tol=1e-8, max_iter=10000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-6)
    assert ours.coef_[2] == 0.0 and abs(ours.coef_[0]) > 0.1


def test_fama_macbeth_averages_per_date_slopes():
    rng, date_ids, X = panel_rows()
    slope = np.where(date_ids % 2 == 0, 0.4, 0.0)  # 0.4 on even dates, 0 on odd: mean 0.2
    y = slope * X[:, 0] + 0.1 * rng.standard_normal(len(X))
    model = FamaMacBethForecast(min_names=20).fit(X, y, date_ids)
    assert model.coef_[0] == pytest.approx(0.2, abs=0.01)
    assert abs(model.coef_[1]) < 0.02 and model.n_dates_ == 60
    few = FamaMacBethForecast(min_names=81)
    with pytest.raises(ValueError):
        few.fit(X, y, date_ids)


def test_linear_score_fills_missing_features_with_zero():
    X = np.array([[1.0, np.nan], [np.nan, 2.0]])
    assert list(LinearScore(np.array([-1.0, 1.0])).predict(X)) == [-1.0, 2.0]


def test_target_scaling_returns_predictions_in_target_units():
    rng, _, X = panel_rows()
    y = 0.01 * (X[:, 0] + rng.standard_normal(len(X)))
    scaled = ScaledTarget(RidgeModel(0.0)).fit(X, y)
    plain = RidgeModel(0.0).fit(X, y)
    assert np.allclose(scaled.predict(X), plain.predict(X), atol=1e-12)
    assert scaled.scale == pytest.approx(np.std(y))


@needs_lightgbm
def test_lightgbm_is_deterministic_and_early_stops():
    rng, _, X = panel_rows(n_dates=100)
    y = np.tanh(X[:, 0]) + rng.standard_normal(len(X))
    params = {"num_leaves": 7, "min_data_in_leaf": 50, "lambda_l2": 1.0, "learning_rate": 0.1, "feature_fraction": 0.7}
    cfg = copy.deepcopy(MCFG)
    cfg["models"]["lightgbm"]["n_estimators"].update(max=300, early_stopping_rounds=20)
    a = build_model("lightgbm", params, cfg).fit(X[:6000], y[:6000], X_val=X[6000:], y_val=y[6000:])
    b = build_model("lightgbm", params, cfg).fit(X[:6000], y[:6000], X_val=X[6000:], y_val=y[6000:])
    assert np.array_equal(a.predict(X), b.predict(X))
    rounds = a.model.best_iteration_
    assert 1 <= rounds < 300
    refit = build_model("lightgbm", params, cfg, n_rounds=rounds).fit(X, y)
    assert refit.model.booster_.num_trees() == rounds
    assert isinstance(refit.model, LightGBMModel)


# --------------------------------------------------------------------------- evaluation helpers


def test_long_ic_matches_the_wide_ic():
    rng, date_ids, X = panel_rows(n_dates=10, n_names=50)
    y = 0.2 * X[:, 0] + rng.standard_normal(len(X))
    y[::13] = np.nan
    index = pd.MultiIndex.from_arrays(
        [pd.bdate_range("2000-01-03", periods=10).repeat(50), np.tile([f"T{i}" for i in range(50)], 10)],
        names=["date", "ticker"],
    )
    grid = Grid.from_index(index)
    wide = cross_sectional_ic(grid.wide(X[:, 0]), grid.wide(y), 20)
    long = daily_ic_long(X[:, 0], y, date_ids, 20)
    assert np.allclose(wide.to_numpy(), long.to_numpy(), atol=1e-6)


def test_paired_test_and_oos_r2_on_hand_examples():
    idx = pd.bdate_range("2000-01-03", periods=300)
    rng = np.random.default_rng(3)
    a = pd.Series(0.02 + 0.01 * rng.standard_normal(300), index=idx)
    b = pd.Series(0.01 * rng.standard_normal(300), index=idx)
    out = paired_ic_test(a, b, 5)
    assert out["ic_diff_mean"] == pytest.approx((a - b).mean()) and out["nw_t"] > 5
    assert oos_r2(np.array([0.0, 0.0]), np.array([1.0, -1.0])) == 0.0
    assert oos_r2(np.array([1.0, -1.0]), np.array([1.0, -1.0])) == 1.0


def test_permutation_importance_ranks_the_signal_feature_first():
    rng, date_ids, X = panel_rows(n_dates=40, n_names=200)
    y = 0.5 * X[:, 0] + rng.standard_normal(len(X))
    model = RidgeModel(0.0).fit(X, y)
    rows = permutation_importance(model, X, y, date_ids, ["a", "b", "c"], 2, 20, 0)
    drops = {r["feature"]: r["ic_drop_mean"] for r in rows}
    assert drops["a"] > 0.3 and abs(drops["b"]) < 0.02 and abs(drops["c"]) < 0.02


# --------------------------------------------------------------------------- runner


def test_runner_smoke_on_a_tiny_synthetic_panel(tmp_path):
    import json

    from src.run_models import run

    rcfg = copy.deepcopy(RCFG)
    rcfg.update(min_names_per_date=10, quantiles=5)
    rpath = tmp_path / "research.yaml"
    rpath.write_text(yaml.safe_dump(rcfg))
    mcfg = yaml.safe_load(open("configs/models.yaml"))
    mcfg["models"]["ridge"]["alpha"] = [0.1, 10.0]
    mcfg["models"]["elastic_net"].update(alpha=[0.001, 0.01], l1_ratio=[0.5])
    mcfg["models"]["lightgbm"].update(num_leaves=[7], min_data_in_leaf=[50], lambda_l2=[1.0])
    mcfg["models"]["lightgbm"]["n_estimators"].update(max=50, early_stopping_rounds=10)
    mcfg["evaluation"]["interpretation"]["lightgbm_shap"]["dates_per_fold"] = 5
    mpath = tmp_path / "models.yaml"
    mpath.write_text(yaml.safe_dump(mcfg))

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2003-01-01", "2009-12-31")
    tickers = [f"T{i}" for i in range(40)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(index)
    cols = {name: rng.standard_normal(n).astype("float32") for name in RCFG["signals"]}
    for name in RCFG["signals"]:
        cols[f"z_{name}"] = rng.standard_normal(n).astype("float32")
    cols["z_mom_12_1"][::50] = np.nan
    signal = -cols["z_ret_5d"].astype("float64")  # planted reversal
    for h in RCFG["horizons"]:
        r = 0.002 * np.sqrt(h) * signal + 0.02 * np.sqrt(h) * rng.standard_normal(n)
        cols[f"fwd_ret_{h}d"] = r.astype("float32")
        cols[f"fwd_ret_{h}d_xs"] = (r - r.reshape(len(dates), -1).mean(axis=1).repeat(40)).astype("float32")
        cols[f"fwd_terminal_{h}d"] = np.zeros(n, dtype="float32")
    panel_file = tmp_path / "panel.parquet"
    pd.DataFrame(cols, index=index).to_parquet(panel_file)

    models = "ridge,elastic_net" + (",lightgbm" if lightgbm_available() else "")
    out, preds = tmp_path / "out", tmp_path / "oof.parquet"
    tables = run(["--config", str(mpath), "--research-config", str(rpath), "--panel", str(panel_file),
                  "--out", str(out), "--predictions", str(preds), "--models", models,
                  "--reference", str(tmp_path / "none"), "--allow-dirty-config"])
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["first_oos_date"] == "2007-01-01" and len(meta["folds"]["5"]) == 3
    oof = pd.read_parquet(preds)
    assert oof.index.get_level_values("date").min() == pd.Timestamp("2007-01-01")
    assert {"ridge_h5_rank", "B0_reversal_h21", "B2_fama_macbeth_h5_raw", "fold_id"} <= set(oof.columns)
    ic = tables["ic_summary"].set_index("column")
    assert ic.loc["ridge_h5_rank", "ic_mean"] > 0.05 and ic.loc["B0_reversal_h5", "ic_mean"] > 0.05
    assert np.isnan(ic.loc["B0_reversal_h5", "oos_r2_vs_zero"])
    assert {"p_value_holm", "reject_holm"} <= set(tables["model_comparison"].columns)
    abl = tables["family_ablation"]
    single_rev = abl.query("kind == 'single_family' and family == 'reversal' and horizon == 5 and variant == 'rank'")
    lofo_rev = abl.query("kind == 'leave_one_family_out' and family == 'reversal' and horizon == 5 and variant == 'rank'")
    assert single_rev["ic_mean"].iloc[0] > 0.05 > lofo_rev["ic_mean"].iloc[0]
    signs = tables["coefficient_signs"].query("model == 'ridge' and feature == 'z_ret_5d'")
    assert (signs["coef_mean"] < 0).all() and signs["matches_prior"].all()
    assert (out / "figures" / "cumulative_ic_h5.png").exists()
    assert len(tables["terminal_sensitivity"]) == 2
