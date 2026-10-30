import pytest
import yaml

from src.second_round import SECOND_ROUND_CONFIG_PATH, load_second_round_config


def raw_config():
    return yaml.safe_load(open(SECOND_ROUND_CONFIG_PATH))


def write(tmp_path, cfg):
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# --------------------------------------------------------------------------- the real config


def test_real_config_loads_with_the_preregistered_design():
    cfg = load_second_round_config()
    assert cfg["no_trade_band"]["bands"][0] == 0.0 and cfg["liquid_universe"]["fractions"][-1] == 1.0
    assert cfg["selection"]["window"] == "tuning" and cfg["selection"]["recorded_point_competes"] is True
    assert cfg["evaluation"]["evidence_label"] == "second_look" and cfg["evaluation"]["confirmatory_tests"] == "none"
    assert cfg["holdout_candidate"]["at_most"] == 1
    points = cfg["wider_lambda_grid"]["points"]
    assert points["lightgbm_h5_rank"]["extends_from"] == [100.0, 4.0] and len(points["lightgbm_h5_rank"]["added"]) == 5
    assert points["B0_reversal_h5"]["extends_from"] == [3.0, 1.0] and len(points["B0_reversal_h5"]["added"]) == 3


# --------------------------------------------------------------------------- each rule rejects a bad config


def set_path(cfg, path, value):
    *head, last = path.split(".")
    for key in head:
        cfg = cfg[key]
    cfg[last] = value


LGBM = "wider_lambda_grid.points.lightgbm_h5_rank"
B0 = "wider_lambda_grid.points.B0_reversal_h5"

BAD = {
    # unknown and missing keys
    "unknown_top_level_key": lambda c: c.update(vol_target=0.1),
    "unknown_nested_key": lambda c: c["no_trade_band"].update(hysteresis=True),
    "missing_section": lambda c: c.pop("holdout_candidate"),
    "missing_nested_key": lambda c: c["selection"].pop("ties"),
    "unknown_point_field": lambda c: c["wider_lambda_grid"]["points"]["B0_reversal_h5"].update(note="x"),
    # wrong types
    "bands_as_number": lambda c: set_path(c, "no_trade_band.bands", 1.0),
    "seed_as_string": lambda c: set_path(c, "bootstrap.seed", "20260101"),
    "bool_as_number": lambda c: set_path(c, "bootstrap.confidence", True),
    "section_as_scalar": lambda c: set_path(c, "selection", "net_sharpe"),
    # methods and scope the design does not define
    "band_on_optimised": lambda c: set_path(c, "no_trade_band.construction", "optimised"),
    "band_in_bp": lambda c: set_path(c, "no_trade_band.band_unit", "basis_points"),
    "trade_to_target": lambda c: set_path(c, "no_trade_band.trade_rule", "to_target"),
    "band_keeps_unscored": lambda c: set_path(c, "no_trade_band.unscored_names", "held"),
    "neutralise_everything": lambda c: set_path(c, "no_trade_band.neutralise", "all_names"),
    "scale_up": lambda c: set_path(c, "no_trade_band.scale", "to_target"),
    "other_liquidity_rank": lambda c: set_path(c, "liquid_universe.rank_feature", "market_cap"),
    "rank_with_future_liquidity": lambda c: set_path(c, "liquid_universe.rank_date", "trade_date"),
    "retuned_lambdas": lambda c: set_path(c, "liquid_universe.lambdas", "retuned"),
    "selection_on_evaluation": lambda c: set_path(c, "selection.window", "evaluation"),
    "selection_on_gross": lambda c: set_path(c, "selection.metric", "gross_sharpe"),
    "extending_again": lambda c: set_path(c, "selection.grid_edge", "extend"),
    "candidate_on_evaluation": lambda c: set_path(c, "holdout_candidate.window", "evaluation"),
    "candidate_without_a_bar": lambda c: set_path(c, "holdout_candidate.must_exceed", "zero"),
    "two_candidates": lambda c: set_path(c, "holdout_candidate.at_most", 2),
    "candidate_on_b0": lambda c: set_path(c, "holdout_candidate.signal", "B0_reversal_h5"),
    "other_scenario": lambda c: set_path(c, "evaluation.scenario", "cost_x2"),
    "evaluation_not_second_look": lambda c: set_path(c, "evaluation.evidence_label", "confirmatory"),
    "evaluation_with_a_test": lambda c: set_path(c, "evaluation.confirmatory_tests", "net_sharpe"),
    "rerunning_all_variants": lambda c: set_path(c, "evaluation.runs", "all_variants"),
    # signals
    "other_primary": lambda c: set_path(c, "signals.primary", "lightgbm_h21_rank"),
    "other_benchmark": lambda c: set_path(c, "signals.benchmark", "B1_composite_h5"),
    # bands and fractions
    "band_grid_without_zero": lambda c: set_path(c, "no_trade_band.bands", [0.5, 1.0]),
    "unsorted_bands": lambda c: set_path(c, "no_trade_band.bands", [0.0, 2.0, 1.0]),
    "fractions_without_one": lambda c: set_path(c, "liquid_universe.fractions", [0.25, 0.5]),
    "zero_fraction": lambda c: set_path(c, "liquid_universe.fractions", [0.0, 1.0]),
    "liquid_on_one_construction": lambda c: set_path(c, "liquid_universe.constructions", ["optimised"]),
    # lambda points
    "other_extends_from": lambda c: set_path(c, f"{LGBM}.extends_from", [30.0, 4.0]),
    "no_added_points": lambda c: set_path(c, f"{LGBM}.added", []),
    "repeated_added_point": lambda c: set_path(c, f"{B0}.added", [[1.0, 1.0], [1.0, 1.0]]),
    "point_already_on_grid": lambda c: set_path(c, f"{LGBM}.added", [[30.0, 4.0]]),
    "inward_point": lambda c: set_path(c, f"{B0}.added", [[5.0, 1.0]]),
    "point_past_an_edge_it_did_not_reach": lambda c: set_path(c, f"{B0}.added", [[3.0, 8.0]]),
    "point_on_the_wrong_side": lambda c: set_path(c, f"{LGBM}.added", [[1.0, 0.25]]),
    "negative_lambda": lambda c: set_path(c, f"{LGBM}.added", [[-300.0, 4.0]]),
    "three_values": lambda c: set_path(c, f"{LGBM}.added", [[300.0, 4.0, 1.0]]),
    "extra_signal": lambda c: c["wider_lambda_grid"]["points"].update(
        B1_composite_h5={"extends_from": [3.0, 2.0], "added": [[1.0, 2.0]]}),
    # evaluation, bootstrap, checks, outputs
    "missing_statistic": lambda c: set_path(c, "evaluation.statistics", ["net_sharpe"]),
    "other_seed": lambda c: set_path(c, "bootstrap.seed", 1),
    "other_block_length": lambda c: set_path(c, "bootstrap.mean_block_length", 63),
    "fewer_resamples": lambda c: set_path(c, "bootstrap.n_resamples", 1000),
    "paired_against_b0": lambda c: set_path(c, "bootstrap.paired_against", "benchmark"),
    "loose_tolerance": lambda c: set_path(c, "checks.recorded_book_tolerance", 0.01),
    "tolerances_swapped": lambda c: set_path(c, "checks.recorded_optimised_sharpe_tolerance", 1.0e-12),
    "no_tuning_isolation": lambda c: set_path(c, "checks.tuning_isolation", "none"),
    "evaluation_without_tuning": lambda c: set_path(c, "checks.evaluation_needs_matching_tuning_config", False),
    "reports_outside_reports": lambda c: set_path(c, "outputs.reports_dir", "data/second_round"),
    "figures_elsewhere": lambda c: set_path(c, "outputs.figures_dir", "reports/figures"),
    "data_tracked": lambda c: set_path(c, "outputs.data_dir", "reports/second_round/data"),
    "overwrites_sensitivity": lambda c: (
        set_path(c, "outputs.reports_dir", "reports/sensitivity"),
        set_path(c, "outputs.figures_dir", "reports/sensitivity/figures"),
    ),
}


@pytest.mark.parametrize("name", list(BAD))
def test_bad_config_is_rejected(tmp_path, name):
    cfg = raw_config()
    BAD[name](cfg)
    with pytest.raises(ValueError):
        load_second_round_config(write(tmp_path, cfg))
