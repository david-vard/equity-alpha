import pytest
import yaml

from src.holdout import HOLDOUT_CONFIG_PATH, load_holdout_config


def raw_config():
    return yaml.safe_load(open(HOLDOUT_CONFIG_PATH))


def write(tmp_path, cfg):
    path = tmp_path / "h.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# --------------------------------------------------------------------------- the real config


def test_real_config_loads_with_the_preregistered_design():
    cfg = load_holdout_config()
    books = cfg["books"]
    assert list(books) == ["headline", "benchmark", "candidate"]
    assert (books["headline"]["signal"], books["headline"]["lambdas"]) == ("lightgbm_h5_rank", [100.0, 4.0])
    assert (books["benchmark"]["signal"], books["benchmark"]["lambdas"]) == ("B0_reversal_h5", [3.0, 1.0])
    assert (books["candidate"]["liquid_fraction"], books["candidate"]["lambdas"]) == (0.25, [100.0, 4.0])
    assert cfg["model_refit"]["test_years"] == [2022, 2023, 2024, 2025, 2026]
    assert cfg["one_shot"]["requires_flag"] == "--open-holdout" and cfg["one_shot"]["requires_clean_tree"] is True


def test_loading_the_design_reads_no_holdout_row(monkeypatch):
    import pandas as pd

    def refuse(*a, **k):
        raise AssertionError("the loader must not read a parquet file")

    monkeypatch.setattr(pd, "read_parquet", refuse)
    load_holdout_config()


# --------------------------------------------------------------------------- each rule rejects a bad config


def set_path(cfg, path, value):
    *head, last = path.split(".")
    for key in head:
        cfg = cfg[key]
    cfg[last] = value


BAD = {
    # unknown and missing keys
    "unknown_top_level_key": lambda c: c.update(band_book=True),
    "unknown_nested_key": lambda c: c["model_refit"].update(n_estimators=500),
    "fourth_book": lambda c: c["books"].update(band=dict(c["books"]["candidate"])),
    "missing_section": lambda c: c.pop("one_shot"),
    "missing_book": lambda c: c["books"].pop("benchmark"),
    "missing_nested_key": lambda c: c["outcomes"]["headline"].pop("t_threshold"),
    # wrong types
    "test_years_as_number": lambda c: set_path(c, "model_refit.test_years", 2022),
    "bool_as_number": lambda c: set_path(c, "outcomes.headline.t_threshold", True),
    "section_as_scalar": lambda c: set_path(c, "window", "2022"),
    # pinned inputs
    "other_portfolio_hash": lambda c: set_path(c, "pinned.portfolio_config_sha256", "0" * 64),
    "other_models_hash": lambda c: set_path(c, "pinned.models_config_sha256", "0" * 64),
    "other_second_round_hash": lambda c: set_path(c, "pinned.second_round_config_sha256", "0" * 64),
    "other_lambdas_hash": lambda c: set_path(c, "pinned.portfolio_lambdas_sha256", "0" * 64),
    "other_hyperparameters_hash": lambda c: set_path(c, "pinned.model_hyperparameters_sha256", "0" * 64),
    "other_candidate_hash": lambda c: set_path(c, "pinned.holdout_candidate_sha256", "0" * 64),
    # books
    "headline_other_signal": lambda c: set_path(c, "books.headline.signal", "lightgbm_h21_rank"),
    "headline_other_construction": lambda c: set_path(c, "books.headline.construction", "neutral_volscaled"),
    "headline_other_lambdas": lambda c: set_path(c, "books.headline.lambdas", [30.0, 4.0]),
    "headline_liquid": lambda c: set_path(c, "books.headline.liquid_fraction", 0.5),
    "headline_role_changed": lambda c: set_path(c, "books.headline.role", "comparator"),
    "benchmark_other_signal": lambda c: set_path(c, "books.benchmark.signal", "B1_composite_h5"),
    "benchmark_second_round_lambdas": lambda c: set_path(c, "books.benchmark.lambdas", [1.0, 1.0]),
    "candidate_swapped_for_band": lambda c: set_path(c, "books.candidate.construction", "neutral_volscaled"),
    "candidate_other_fraction": lambda c: set_path(c, "books.candidate.liquid_fraction", 0.5),
    "candidate_other_lambdas": lambda c: set_path(c, "books.candidate.lambdas", [30.0, 4.0]),
    # window and refit
    "later_start": lambda c: set_path(c, "window.start", "2023-01-01"),
    "other_last_date": lambda c: set_path(c, "window.last_panel_date", "2025-12-31"),
    "no_runoff": lambda c: set_path(c, "window.runoff", "none"),
    "other_model": lambda c: set_path(c, "model_refit.model", "ridge"),
    "other_horizon": lambda c: set_path(c, "model_refit.horizon", 21),
    "other_variant": lambda c: set_path(c, "model_refit.target_variant", "raw"),
    "single_frozen_model": lambda c: set_path(c, "model_refit.scheme", "fit_once"),
    "skipped_year": lambda c: set_path(c, "model_refit.test_years", [2022, 2024, 2025, 2026]),
    "missing_last_year": lambda c: set_path(c, "model_refit.test_years", [2022, 2023, 2024, 2025]),
    "hand_settings": lambda c: set_path(c, "model_refit.settings", "custom"),
    "other_ic": lambda c: set_path(c, "model_refit.grinold_ic", "holdout_ic"),
    # continuity
    "flat_start": lambda c: set_path(c, "continuity.run_from", "first_holdout_date"),
    "december_with_a_new_model": lambda c: set_path(c, "continuity.december_2021", "fold_2022_model"),
    "reporting_from_2021": lambda c: set_path(c, "continuity.reporting_starts", "2021-01-01"),
    # outcomes and reporting
    "lower_threshold": lambda c: set_path(c, "outcomes.headline.t_threshold", 1.5),
    "softer_label": lambda c: set_path(c, "outcomes.headline.labels.between", "promising"),
    "candidate_lower_threshold": lambda c: set_path(c, "outcomes.candidate.t_threshold", 1.0),
    "difference_by_t": lambda c: set_path(c, "outcomes.lightgbm_minus_b0.interval", "t_test"),
    "difference_label_changed": lambda c: set_path(c, "outcomes.lightgbm_minus_b0.labels.contains_zero", "similar"),
    "missing_statistic": lambda c: set_path(c, "reporting.statistics", ["net_sharpe"]),
    "no_years": lambda c: set_path(c, "reporting.by_year", False),
    "no_breakevens": lambda c: set_path(c, "reporting.breakevens", []),
    # bootstrap
    "other_seed": lambda c: set_path(c, "bootstrap.seed", 7),
    "other_block": lambda c: set_path(c, "bootstrap.mean_block_length", 63),
    "fewer_resamples": lambda c: set_path(c, "bootstrap.n_resamples", 1000),
    # one shot, checks, outputs
    "other_flag": lambda c: set_path(c, "one_shot.requires_flag", "--run"),
    "dirty_tree_allowed": lambda c: set_path(c, "one_shot.requires_clean_tree", False),
    "lock_elsewhere": lambda c: set_path(c, "one_shot.lock_file", "/tmp/opened.json"),
    "rerun_on_failure": lambda c: set_path(c, "one_shot.failed_check_policy", "rerun"),
    "loose_prediction_tolerance": lambda c: set_path(c, "checks.fold_2021_prediction_tolerance", 0.01),
    "loose_sharpe_tolerance": lambda c: set_path(c, "checks.recorded_series_sharpe_tolerance", 0.01),
    "leakage_allowed": lambda c: set_path(c, "checks.fold_leakage", "some"),
    "reports_elsewhere": lambda c: (set_path(c, "outputs.reports_dir", "reports/portfolio"),
                                    set_path(c, "outputs.figures_dir", "reports/portfolio/figures"),
                                    set_path(c, "one_shot.lock_file", "reports/portfolio/opened.json")),
    "figures_elsewhere": lambda c: set_path(c, "outputs.figures_dir", "reports/figures"),
    "data_tracked": lambda c: set_path(c, "outputs.data_dir", "reports/holdout/data"),
}


@pytest.mark.parametrize("name", list(BAD))
def test_bad_config_is_rejected(tmp_path, name):
    cfg = raw_config()
    BAD[name](cfg)
    with pytest.raises(ValueError):
        load_holdout_config(write(tmp_path, cfg))
