import copy

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    load_config,
    membership_to_mask,
    validate_config,
    warmup_days,
    winsorize_by_date,
    zscore_by_date,
)


def test_default_config_loads_and_warmup_matches_longest_window():
    cfg = load_config()
    assert warmup_days(cfg) == 252
    assert cfg["labels"]["lag"] == 1 and cfg["labels"]["horizons"] == [1, 5, 21]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["preprocessing"].update(winsor_limits=[0.9, 0.1]),
        lambda c: c["labels"].update(horizons=[0, 5]),
        lambda c: c["labels"].update(lag=-1),
        lambda c: c["features"].update(min_obs_frac=0),
        lambda c: c["features"]["momentum_12_1"].update(skip=300),
        lambda c: c["features"].update(trend_windows=[]),
        lambda c: c["data"].update(start="2026-01-01", end="2020-01-01"),
        lambda c: c.pop("labels"),
    ],
)
def test_invalid_config_is_rejected(mutate):
    cfg = copy.deepcopy(load_config())
    mutate(cfg)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_winsorize_clips_each_date_at_its_own_quantiles():
    values = np.arange(101, dtype=float)
    frame = pd.DataFrame([values, values * 10], columns=[f"S{i}" for i in range(101)])
    clipped = winsorize_by_date(frame, (0.01, 0.99))
    assert clipped.iloc[0].min() == pytest.approx(1.0) and clipped.iloc[0].max() == pytest.approx(99.0)
    assert clipped.iloc[1].max() == pytest.approx(990.0)  # scaled independently of row 0
    assert clipped.iloc[0, 50] == 50.0  # interior values untouched


def test_zscore_by_date_gives_zero_mean_unit_std_and_keeps_nans():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(5, 3, (4, 50)))
    frame.iloc[0, 0] = np.nan
    constant = pd.DataFrame([[2.0] * 5])
    z = zscore_by_date(frame)
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=1), 1, atol=1e-12)
    assert np.isnan(z.iloc[0, 0])
    assert zscore_by_date(constant).isna().all().all()


def test_membership_to_mask_ignores_non_trading_days_and_unknown_tickers():
    index = pd.DatetimeIndex(["2024-01-05", "2024-01-08"])  # Friday, Monday
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-08", "2024-01-08"]),
            "ticker": ["AAA", "AAA", "AAA", "ZZZ"],
        }
    )
    mask = membership_to_mask(membership, index, pd.Index(["AAA", "BBB"]))
    assert mask.to_numpy().tolist() == [[True, False], [True, False]]
