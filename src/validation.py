"""Walk-forward folds for the modelling: expanding window, annual refit, purge and inner validation.

Everything is in trading-date *positions* of the research calendar (consecutive trading days),
so a label of horizon ``h`` entered with lag ``lag`` at position ``p`` ends at ``p + lag + h``.
A training date is usable for a fold only if that end lies strictly before the fold's first test
date (purge); the inner validation block is purged the same way against the inner training rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """Date positions of one walk-forward fold (training positions before any date stride)."""

    fold_id: int
    test_year: int
    train: np.ndarray
    inner_train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def summary(self, dates: pd.DatetimeIndex) -> dict[str, object]:
        def span(pos: np.ndarray, prefix: str) -> dict[str, object]:
            return {
                f"{prefix}_start": dates[pos[0]].date() if len(pos) else None,
                f"{prefix}_end": dates[pos[-1]].date() if len(pos) else None,
                f"{prefix}_dates": len(pos),
            }

        return {
            "fold_id": self.fold_id,
            "test_year": self.test_year,
            **span(self.train, "train"),
            **span(self.inner_train, "inner_train"),
            **span(self.validation, "validation"),
            **span(self.test, "test"),
        }


def purged(positions: np.ndarray, first_blocked: int, horizon: int, lag: int) -> np.ndarray:
    """Positions whose label window ``p + lag + h`` ends strictly before ``first_blocked``."""
    return positions[positions + lag + horizon < first_blocked]


def walk_forward_folds(
    dates: pd.DatetimeIndex,
    horizon: int,
    lag: int,
    first_test_year: int,
    inner_years: int,
    last_test_date: pd.Timestamp | None = None,
) -> list[Fold]:
    """Expanding-window folds refit every January, from ``first_test_year`` to ``last_test_date``.

    Fold ``Y`` tests every date of calendar year ``Y`` (cut at ``last_test_date``) and trains on all
    earlier dates whose label ends before the first test date. Its inner validation block is the
    last ``inner_years`` calendar years of that training window; the inner training dates are purged
    against the first validation date.
    """
    dates = pd.DatetimeIndex(dates)
    last = pd.Timestamp(last_test_date) if last_test_date is not None else dates[-1]
    all_pos = np.arange(len(dates))
    folds = []
    for fold_id, year in enumerate(range(first_test_year, last.year + 1)):
        test = all_pos[(dates.year == year) & (dates <= last)]
        if not len(test):
            continue
        train = purged(all_pos[: test[0]], test[0], horizon, lag)
        val_start = int(dates.searchsorted(pd.Timestamp(year - inner_years, 1, 1)))
        validation = train[train >= val_start]
        inner_train = purged(train[train < val_start], val_start, horizon, lag)
        fold = Fold(fold_id, year, train, inner_train, validation, test)
        assert_no_leakage(fold, horizon, lag)
        folds.append(fold)
    return folds


def thin(positions: np.ndarray, stride: int) -> np.ndarray:
    """Every ``stride``-th date by absolute position, so the kept dates are the same in every fold."""
    return positions[positions % stride == 0] if stride > 1 else positions


def assert_no_leakage(fold: Fold, horizon: int, lag: int) -> None:
    """Fail if any training label reaches the test block or an inner label reaches validation."""
    if not len(fold.test) or not len(fold.train) or not len(fold.inner_train) or not len(fold.validation):
        raise AssertionError(f"fold {fold.fold_id} has an empty block")
    if fold.train.max() + lag + horizon >= fold.test.min():
        raise AssertionError(f"fold {fold.fold_id}: a training label reaches the test period")
    if fold.inner_train.max() + lag + horizon >= fold.validation.min():
        raise AssertionError(f"fold {fold.fold_id}: an inner training label reaches validation")
    if not np.isin(fold.validation, fold.train).all() or not np.isin(fold.inner_train, fold.train).all():
        raise AssertionError(f"fold {fold.fold_id}: inner blocks are not inside the training window")
    if np.intersect1d(fold.inner_train, fold.validation).size:
        raise AssertionError(f"fold {fold.fold_id}: inner training and validation overlap")
    if (np.diff(fold.test) != 1).any():
        raise AssertionError(f"fold {fold.fold_id}: test dates are not contiguous")


def position_mask(row_positions: np.ndarray, positions: np.ndarray, n_dates: int) -> np.ndarray:
    """Boolean row mask: rows whose date position is in ``positions``."""
    keep = np.zeros(n_dates, dtype=bool)
    keep[positions] = True
    return keep[row_positions]
