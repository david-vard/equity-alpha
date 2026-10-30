import pytest
import yaml

from src.attribution import ATTRIBUTION_CONFIG_PATH, FACTORS, load_attribution_config


def raw_config():
    return yaml.safe_load(open(ATTRIBUTION_CONFIG_PATH))


def write(tmp_path, cfg):
    path = tmp_path / "a.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# --------------------------------------------------------------------------- the real config


def test_real_config_loads_with_the_preregistered_design():
    cfg = load_attribution_config()
    roles = {b["role"] for b in cfg["books"].values()}
    assert roles == {"headline", "primary_attribution", "benchmark"}
    assert list(cfg["models"]) == ["market", "ff5", "ff5_mom", "ff5_mom_strev", "ff5_mom_strev_bab"]
    used = [f for spec in cfg["models"].values() for f in spec["adds"]]
    assert sorted(used) == sorted(FACTORS)
    test = cfg["primary_test"]
    assert (test["book"], test["series"], test["window"], test["model"], test["factor"], test["t_threshold"]) == (
        "lightgbm_h5_neutral_volscaled", "gross_pnl", "evaluation", "ff5_mom_strev_bab", "st_rev", 2.0)
    assert cfg["factors"]["cut_before"] == "holdout_start"
    assert cfg["book_data"]["scenario"] == "base"


def test_model_order_is_kept_as_nested():
    assert list(raw_config()["models"])[0] == "market"
    assert list(raw_config()["models"])[-1] == "ff5_mom_strev_bab"


# --------------------------------------------------------------------------- each rule rejects a bad config


def set_path(cfg, path, value):
    *head, last = path.split(".")
    for key in head:
        cfg = cfg[key]
    cfg[last] = value


def drop_factor(c):
    del c["factors"]["sources"]["aqr_betting_against_beta"]


BAD = {
    # unknown and missing keys
    "unknown_top_level_key": lambda c: c.update(sectors=True),
    "unknown_nested_key": lambda c: c["regression"].update(robust=True),
    "unknown_book_field": lambda c: c["books"]["B0_h5_optimised"].update(weight=1),
    "unknown_source_field": lambda c: c["factors"]["sources"]["french_momentum"].update(sheet="x"),
    "missing_section": lambda c: c.pop("primary_test"),
    "missing_nested_key": lambda c: c["primary_test"].pop("factor"),
    # wrong types
    "threshold_as_string": lambda c: set_path(c, "primary_test.t_threshold", "2"),
    "bool_as_number": lambda c: set_path(c, "primary_test.t_threshold", True),
    "section_as_scalar": lambda c: set_path(c, "risk", "historical"),
    # methods and scope the design does not define
    "other_scenario": lambda c: set_path(c, "book_data.scenario", "cost_x2"),
    "net_as_primary": lambda c: set_path(c, "series.primary", "net_pnl"),
    "tuning_as_primary": lambda c: set_path(c, "windows.tuning", "primary"),
    "factors_past_the_holdout": lambda c: set_path(c, "factors.cut_before", "none"),
    "filling_missing_factors": lambda c: set_path(c, "factors.coverage", "forward_fill"),
    "other_estimator": lambda c: set_path(c, "regression.estimator", "ridge"),
    "other_standard_errors": lambda c: set_path(c, "regression.standard_errors", "white"),
    "parametric_var": lambda c: set_path(c, "risk.var_es.method", "gaussian"),
    "compounded_drawdown": lambda c: set_path(c, "risk.cumulative", "compounded"),
    "other_market_proxy": lambda c: set_path(c, "beta_analysis.market_proxies.value_weighted", "spy"),
    # books
    "no_books": lambda c: set_path(c, "books", {}),
    "unknown_signal": lambda c: set_path(c, "books.B0_h5_optimised.signal", "B9_h5"),
    "unknown_construction": lambda c: set_path(c, "books.B0_h5_optimised.construction", "black_litterman"),
    "unknown_role": lambda c: set_path(c, "books.B0_h5_optimised.role", "extra"),
    "headline_book_missing": lambda c: c["books"].pop("lightgbm_h5_optimised"),
    "benchmark_book_missing": lambda c: c["books"].pop("B0_h5_optimised"),
    "repeated_book": lambda c: c["books"].update(copy=dict(c["books"]["B0_h5_neutral_volscaled"])),
    "two_primary_books": lambda c: set_path(c, "books.B0_h5_neutral_volscaled.role", "primary_attribution"),
    # factors
    "factor_source_missing": drop_factor,
    "http_source": lambda c: set_path(
        c, "factors.sources.french_momentum.url", "http://mba.tuck.dartmouth.edu/x.zip"),
    "unknown_format": lambda c: set_path(c, "factors.sources.french_momentum.format", "csv"),
    "zero_scale": lambda c: set_path(c, "factors.sources.french_momentum.scale", 0),
    "aqr_without_sheet": lambda c: c["factors"]["sources"]["aqr_betting_against_beta"].pop("sheet"),
    "duplicated_factor": lambda c: set_path(c, "factors.sources.french_momentum.columns", {"Mom": "st_rev"}),
    "raw_dir_tracked": lambda c: set_path(c, "factors.raw_dir", "reports/factors"),
    # models and tests
    "model_adds_nothing": lambda c: set_path(c, "models.ff5_mom.adds", []),
    "factor_added_twice": lambda c: set_path(c, "models.ff5_mom.adds", ["smb"]),
    "extra_factor": lambda c: set_path(c, "models.ff5_mom.adds", ["mom", "lt_rev"]),
    "market_not_first": lambda c: set_path(c, "models", dict(reversed(list(c["models"].items())))),
    "test_on_smaller_model": lambda c: set_path(c, "primary_test.model", "ff5_mom_strev"),
    "test_on_net": lambda c: set_path(c, "primary_test.series", "net_pnl"),
    "test_on_tuning": lambda c: set_path(c, "primary_test.window", "tuning"),
    "test_on_other_book": lambda c: set_path(c, "primary_test.book", "lightgbm_h5_optimised"),
    "test_on_other_factor": lambda c: set_path(c, "primary_test.factor", "mom"),
    "non_positive_threshold": lambda c: set_path(c, "primary_test.t_threshold", 0.0),
    "spanning_on_itself": lambda c: set_path(c, "own_reversal_spanning.regressor_book", "lightgbm_h5_neutral_volscaled"),
    "spanning_across_constructions": lambda c: set_path(c, "own_reversal_spanning.regressor_book", "B0_h5_optimised"),
    # risk, beta and outputs
    "no_drawdown_episodes": lambda c: set_path(c, "risk.drawdown_episodes", 0),
    "confidence_outside_range": lambda c: set_path(c, "risk.var_es.confidence", [95]),
    "zero_day_horizon": lambda c: set_path(c, "risk.var_es.horizons_days", [0]),
    "one_var_unit": lambda c: set_path(c, "risk.var_es.units", ["per_dollar_of_capital"]),
    "one_moment_frequency": lambda c: set_path(c, "risk.moments.skew_excess_kurtosis_on", ["daily"]),
    "no_worst_month": lambda c: set_path(c, "risk.worst_periods", ["day"]),
    "beta_on_net_only": lambda c: set_path(c, "beta_analysis.series", ["net_pnl"]),
    "short_beta_too_long": lambda c: set_path(c, "beta_analysis.explanations.stale_beta.short_beta_window", 252),
    "other_min_obs": lambda c: set_path(c, "beta_analysis.explanations.stale_beta.min_obs_frac", 0.5),
    "reports_outside_reports": lambda c: set_path(c, "outputs.reports_dir", "data/attribution"),
    "figures_elsewhere": lambda c: set_path(c, "outputs.figures_dir", "reports/figures"),
    "overwrites_portfolio_reports": lambda c: (
        set_path(c, "outputs.reports_dir", "reports/portfolio"),
        set_path(c, "outputs.figures_dir", "reports/portfolio/figures"),
    ),
    "book_data_tracked": lambda c: set_path(c, "book_data.daily_returns", "reports/{signal}/daily.parquet"),
}


@pytest.mark.parametrize("name", list(BAD))
def test_bad_config_is_rejected(tmp_path, name):
    cfg = raw_config()
    BAD[name](cfg)
    with pytest.raises(ValueError):
        load_attribution_config(write(tmp_path, cfg))
