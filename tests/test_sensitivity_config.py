import pytest
import yaml

from src.sensitivity import SENSITIVITY_CONFIG_PATH, load_sensitivity_config


def raw_config():
    return yaml.safe_load(open(SENSITIVITY_CONFIG_PATH))


def write(tmp_path, cfg):
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# --------------------------------------------------------------------------- the real config


def test_real_config_loads_with_the_preregistered_design():
    cfg = load_sensitivity_config()
    boot = cfg["bootstrap"]
    assert (boot["method"], boot["n_resamples"], boot["mean_block_lengths"][0]) == ("stationary", 10000, 21)
    assert (boot["paired_differences"]["minuend"], boot["paired_differences"]["subtrahend"]) == (
        "lightgbm_h5_rank", "B0_reversal_h5")
    head = cfg["primary_intervals"]["headline"]
    assert (head["signal"], head["construction"], head["series"], head["window"]) == (
        "lightgbm_h5_rank", "optimised", "net_pnl", "evaluation")
    assert cfg["repricing"]["method"] == "base_trades_held_fixed"
    assert 1.0 in cfg["repricing"]["impact_y"] and 25.0 in cfg["repricing"]["borrow_bps_per_year"]
    assert cfg["offsets"]["source"] == "tranche_rows"


# --------------------------------------------------------------------------- each rule rejects a bad config


def set_path(cfg, path, value):
    *head, last = path.split(".")
    for key in head:
        cfg = cfg[key]
    cfg[last] = value


BAD = {
    # unknown and missing keys
    "unknown_top_level_key": lambda c: c.update(deflated_sharpe=True),
    "unknown_nested_key": lambda c: c["bootstrap"].update(studentised=True),
    "missing_section": lambda c: c.pop("primary_intervals"),
    "missing_nested_key": lambda c: c["bootstrap"].pop("seed"),
    # wrong types
    "resamples_as_string": lambda c: set_path(c, "bootstrap.n_resamples", "10000"),
    "bool_as_number": lambda c: set_path(c, "bootstrap.confidence", True),
    "section_as_scalar": lambda c: set_path(c, "offsets", "tranche_rows"),
    # methods and scope the design does not define
    "other_scenario": lambda c: set_path(c, "book_data.scenario", "cost_x2"),
    "fewer_signals": lambda c: set_path(c, "book_data.signals", "headline_only"),
    "tuning_as_primary": lambda c: set_path(c, "windows.tuning", "primary"),
    "other_bootstrap": lambda c: set_path(c, "bootstrap.method", "circular_block"),
    "other_statistic": lambda c: set_path(c, "bootstrap.statistic", "mean"),
    "other_interval": lambda c: set_path(c, "bootstrap.interval", "bca"),
    "resolving_the_optimiser": lambda c: set_path(c, "repricing.method", "resolve"),
    "partial_grid": lambda c: set_path(c, "repricing.grid", "one_at_a_time"),
    "other_offset_source": lambda c: set_path(c, "offsets.source", "rerun"),
    "offset_tests": lambda c: set_path(c, "offsets.inference", "bootstrap"),
    # bootstrap
    "net_only": lambda c: set_path(c, "bootstrap.series", ["net_pnl"]),
    "too_few_resamples": lambda c: set_path(c, "bootstrap.n_resamples", 100),
    "confidence_as_percent": lambda c: set_path(c, "bootstrap.confidence", 95),
    "no_block_lengths": lambda c: set_path(c, "bootstrap.mean_block_lengths", []),
    "repeated_block_length": lambda c: set_path(c, "bootstrap.mean_block_lengths", [21, 21]),
    "block_below_longest_horizon": lambda c: set_path(c, "bootstrap.mean_block_lengths", [10, 21]),
    "primary_block_not_shortest": lambda c: set_path(c, "bootstrap.mean_block_lengths", [42, 21]),
    "other_pair": lambda c: set_path(c, "bootstrap.paired_differences.subtrahend", "B1_composite_h5"),
    # primary intervals
    "headline_other_signal": lambda c: set_path(c, "primary_intervals.headline.signal", "B0_reversal_h5"),
    "headline_other_construction": lambda c: set_path(c, "primary_intervals.headline.construction", "neutral_volscaled"),
    "headline_on_gross": lambda c: set_path(c, "primary_intervals.headline.series", "gross_pnl"),
    "headline_on_tuning": lambda c: set_path(c, "primary_intervals.headline.window", "tuning"),
    "difference_other_construction": lambda c: set_path(
        c, "primary_intervals.lightgbm_minus_b0.construction", "neutral_volscaled"),
    # repricing
    "grid_without_base_y": lambda c: set_path(c, "repricing.impact_y", [0.5, 2.0]),
    "grid_without_base_borrow": lambda c: set_path(c, "repricing.borrow_bps_per_year", [0.0, 50.0]),
    "zero_y": lambda c: set_path(c, "repricing.impact_y", [0.0, 1.0]),
    "unsorted_y": lambda c: set_path(c, "repricing.impact_y", [2.0, 1.0]),
    "negative_borrow": lambda c: set_path(c, "repricing.borrow_bps_per_year", [-25.0, 25.0]),
    "no_breakevens": lambda c: set_path(c, "repricing.breakeven", []),
    "missing_repricing_statistic": lambda c: set_path(c, "repricing.statistics", ["net_sharpe"]),
    # offsets, checks and outputs
    "offsets_gross_only": lambda c: set_path(c, "offsets.series", ["gross_pnl"]),
    "offsets_unscaled": lambda c: set_path(c, "offsets.scale_to_tranche_capital", False),
    "missing_offset_statistic": lambda c: set_path(c, "offsets.statistics", ["sharpe"]),
    "missing_dispersion": lambda c: set_path(c, "offsets.dispersion", ["range"]),
    "loose_tolerance": lambda c: set_path(c, "checks.reprice_base_tolerance", 0.01),
    "zero_tolerance": lambda c: set_path(c, "checks.reproduce_sharpes_tolerance", 0.0),
    "paired_dates_relaxed": lambda c: set_path(c, "checks.paired_dates", "intersection"),
    "reports_outside_reports": lambda c: set_path(c, "outputs.reports_dir", "data/sensitivity"),
    "figures_elsewhere": lambda c: set_path(c, "outputs.figures_dir", "reports/figures"),
    "overwrites_attribution": lambda c: (
        set_path(c, "outputs.reports_dir", "reports/attribution"),
        set_path(c, "outputs.figures_dir", "reports/attribution/figures"),
    ),
    "book_data_tracked": lambda c: set_path(c, "book_data.daily_returns", "reports/{signal}/daily.parquet"),
}


@pytest.mark.parametrize("name", list(BAD))
def test_bad_config_is_rejected(tmp_path, name):
    cfg = raw_config()
    BAD[name](cfg)
    with pytest.raises(ValueError):
        load_sensitivity_config(write(tmp_path, cfg))
