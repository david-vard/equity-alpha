import subprocess

import pandas as pd
import pytest
import yaml

from src.portfolio import PORTFOLIO_CONFIG_PATH, evaluation_start, load_portfolio_config
from src.utils import provenance


def raw_config():
    return yaml.safe_load(open(PORTFOLIO_CONFIG_PATH))


def write(tmp_path, cfg, name="p.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg))
    return path


def predictions_file(tmp_path, columns):
    index = pd.MultiIndex.from_product([pd.bdate_range("2007-01-03", periods=2), ["A"]], names=["date", "ticker"])
    path = tmp_path / "oof.parquet"
    pd.DataFrame({c: [0.0, 0.0] for c in columns}, index=index).to_parquet(path)
    return path


SIGNAL_COLUMNS = ["lightgbm_h5_rank", "B0_reversal_h5", "B1_composite_h5", "lightgbm_h21_rank"]


# --------------------------------------------------------------------------- the real config


def test_real_config_loads_with_the_preregistered_design(tmp_path):
    cfg = load_portfolio_config(predictions=predictions_file(tmp_path, SIGNAL_COLUMNS + ["fold_id"]))
    roles = {c: s["role"] for c, s in cfg["signals"].items()}
    assert roles == {"lightgbm_h5_rank": "primary", "B0_reversal_h5": "benchmark",
                     "B1_composite_h5": "benchmark", "lightgbm_h21_rank": "low_turnover_variant"}
    assert cfg["timing"]["lag"] == 1 and cfg["sample"]["split_date"] == "2014-01-01"
    assert list(cfg["constructions"])[:3] == ["decile_ls", "neutral_volscaled", "optimised"]
    assert cfg["constructions"]["sector_neutral"]["enabled"] is False
    assert cfg["costs"]["scenarios"]["cost_multipliers"] == [0.0, 0.5, 1.0, 2.0, 3.0]
    assert cfg["costs"]["scenarios"]["aum_usd"] == [10_000_000, 100_000_000, 1_000_000_000]
    head = cfg["headline"]
    assert (head["signal"], head["construction"], head["benchmark"]) == ("lightgbm_h5_rank", "optimised", "B0_reversal_h5")


def test_real_config_loads_against_the_default_predictions_path():
    load_portfolio_config()  # checks the predictions schema only if the gitignored file exists


def test_evaluation_starts_on_the_first_trading_day_of_2014():
    dates = pd.bdate_range("2013-12-27", "2014-01-10").drop(pd.Timestamp("2014-01-01"))
    assert evaluation_start(raw_config(), dates) == pd.Timestamp("2014-01-02")


# --------------------------------------------------------------------------- each rule rejects a bad config


def set_path(cfg, path, value):
    *head, last = path.split(".")
    for key in head:
        cfg = cfg[key]
    cfg[last] = value


BAD = {
    # unknown and missing keys
    "unknown_top_level_key": lambda c: c.update(rebalance_every=5),
    "unknown_nested_key": lambda c: c["costs"].update(slippage_bps=1.0),
    "unknown_signal_field": lambda c: c["signals"]["B0_reversal_h5"].update(weight=1),
    "missing_section": lambda c: c.pop("headline"),
    "missing_nested_key": lambda c: c["costs"]["impact"].pop("y"),
    # wrong types
    "lag_as_string": lambda c: set_path(c, "timing.lag", "1"),
    "bool_as_number": lambda c: set_path(c, "costs.commission_bps", True),
    "list_as_number": lambda c: set_path(c, "constructions.optimised.lambda_risk", 10.0),
    "section_as_scalar": lambda c: set_path(c, "risk_model", "ledoit_wolf"),
    # methods the design does not define
    "other_impact_model": lambda c: set_path(c, "costs.impact.model", "linear"),
    "other_rebalancing": lambda c: set_path(c, "rebalancing.n_tranches", "1"),
    "other_risk_model": lambda c: set_path(c, "risk_model.estimator", "sample"),
    # referenced predictors
    "unknown_predictor": lambda c: c["signals"].update(xgboost_h5_rank={"predictor": "xgboost", "horizon": 5, "variant": "rank", "role": "benchmark"}),
    "column_name_mismatch": lambda c: c["signals"].update(lgbm_h5={"predictor": "lightgbm", "horizon": 5, "variant": "rank", "role": "benchmark"}),
    "baseline_with_variant": lambda c: c["signals"].update(B0_reversal_h5_rank={"predictor": "B0_reversal", "horizon": 5, "variant": "rank", "role": "benchmark"}),
    "model_without_variant": lambda c: c["signals"].update(lightgbm_h5={"predictor": "lightgbm", "horizon": 5, "variant": None, "role": "benchmark"}),
    "two_primaries": lambda c: set_path(c, "signals.B1_composite_h5.role", "primary"),
    # horizons and lag against features.yaml / research.yaml / models.yaml
    "horizon_not_modelled": lambda c: c["signals"].update(lightgbm_h1_rank={"predictor": "lightgbm", "horizon": 1, "variant": "rank", "role": "benchmark"}),
    "horizon_not_researched": lambda c: c["signals"].update(lightgbm_h10_rank={"predictor": "lightgbm", "horizon": 10, "variant": "rank", "role": "benchmark"}),
    "lag_differs_from_labels": lambda c: set_path(c, "timing.lag", 0),
    "split_inside_holdout": lambda c: set_path(c, "sample.split_date", "2022-06-01"),
    "split_before_oos": lambda c: set_path(c, "sample.split_date", "2005-01-01"),
    "benchmark_other_horizon": lambda c: (
        c["signals"].update(B0_reversal_h21={"predictor": "B0_reversal", "horizon": 21, "variant": None, "role": "benchmark"}),
        set_path(c, "headline.benchmark", "B0_reversal_h21"),
    ),
    "benchmark_not_a_benchmark": lambda c: set_path(c, "headline.benchmark", "lightgbm_h21_rank"),
    # features, constructions, costs, headline, outputs
    "feature_not_in_panel": lambda c: set_path(c, "costs.impact.adv_feature", "market_cap"),
    "beta_cap_too_wide": lambda c: set_path(c, "constructions.optimised.constraints.max_abs_beta", 1.5),
    "gross_above_two": lambda c: set_path(c, "constructions.optimised.constraints.max_gross", 3.0),
    "empty_lambda_grid": lambda c: set_path(c, "constructions.optimised.lambda_tc", []),
    "unsorted_lambda_grid": lambda c: set_path(c, "constructions.optimised.lambda_risk", [10.0, 3.0]),
    "sector_neutral_on": lambda c: set_path(c, "constructions.sector_neutral.enabled", True),
    "negative_cost": lambda c: set_path(c, "costs.half_spread_bps", -1.0),
    "scenarios_without_base": lambda c: set_path(c, "costs.scenarios.cost_multipliers", [0.0, 2.0]),
    "aum_scenarios_without_base": lambda c: set_path(c, "costs.scenarios.aum_usd", [10_000_000]),
    "headline_not_primary": lambda c: set_path(c, "headline.signal", "B1_composite_h5"),
    "headline_at_other_costs": lambda c: set_path(c, "headline.cost_multiplier", 2.0),
    "headline_unknown_construction": lambda c: set_path(c, "headline.construction", "risk_parity"),
    "headline_on_tuning_window": lambda c: set_path(c, "headline.window", "tuning"),
    "outputs_tracked_by_git": lambda c: set_path(c, "outputs.weights", "reports/portfolio/weights.parquet"),
}


@pytest.mark.parametrize("mutate", BAD.values(), ids=BAD.keys())
def test_each_rule_rejects_a_bad_config(tmp_path, mutate):
    cfg = raw_config()
    mutate(cfg)
    # the predictions file has every configured column, so only the rule under test can fire
    columns = list(cfg["signals"]) if isinstance(cfg.get("signals"), dict) else SIGNAL_COLUMNS
    with pytest.raises(ValueError):
        load_portfolio_config(write(tmp_path, cfg), predictions=predictions_file(tmp_path, columns))


def test_a_signal_missing_from_the_predictions_file_is_rejected(tmp_path):
    path = write(tmp_path, raw_config())
    with pytest.raises(ValueError, match="lightgbm_h21_rank"):
        load_portfolio_config(path, predictions=predictions_file(tmp_path, SIGNAL_COLUMNS[:3]))
    load_portfolio_config(path, predictions=predictions_file(tmp_path, SIGNAL_COLUMNS))


# --------------------------------------------------------------------------- provenance guard


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                   capture_output=True)


def test_provenance_blocks_an_uncommitted_or_modified_config(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    config, panel = repo / "configs" / "portfolio.yaml", tmp_path / "panel.parquet"
    config.write_text("a: 1\n")
    panel.write_bytes(b"panel")
    features = tmp_path / "features.yaml"
    features.write_text("x: 1\n")
    git(repo, "init", "-q")

    with pytest.raises(SystemExit, match="must be committed"):  # untracked
        provenance(config, features, panel, False, repo_root=repo)

    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "design")
    clean = provenance(config, features, panel, False, repo_root=repo)
    assert clean["config_commit"] and clean["config_dirty"] is False and clean["code_dirty"] is False

    config.write_text("a: 2\n")  # modified after the commit
    with pytest.raises(SystemExit, match="must be committed"):
        provenance(config, features, panel, False, repo_root=repo)
    dirty = provenance(config, features, panel, True, repo_root=repo)
    assert dirty["config_dirty"] is True and dirty["code_dirty"] is True
    assert dirty["config_sha256"] != clean["config_sha256"]


def test_provenance_blocks_a_config_outside_the_repository(tmp_path):
    config, panel = tmp_path / "research.yaml", tmp_path / "panel.parquet"
    config.write_text("a: 1\n")
    panel.write_bytes(b"panel")
    with pytest.raises(SystemExit):
        provenance(config, config, panel, False)
    assert provenance(config, config, panel, True)["config_commit"] is None


def test_runners_share_the_provenance_guard():
    from src import run_diagnostics, run_models, run_signal_research

    assert run_signal_research.provenance is run_models.provenance is run_diagnostics.provenance is provenance
