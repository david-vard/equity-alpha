"""Features and forward-return labels for the cross-sectional equity panel.

Conventions (see README "Features"):

* All windows count trading days (rows of the price frames), not calendar days.
* A feature in row ``t`` uses data up to and including the close of day ``t``.
* ``closeadj`` (split + dividend + spinoff adjusted) is used for every return.
  ``close``, ``open`` and ``volume`` are split adjusted only. The adjusted open is
  ``open * closeadj / close`` (the same factor as that day's close), so overnight and
  intraday returns are consistent with the close-to-close return:
  ``(1 + overnight) * (1 + intraday) == 1 + ret_1d``.
* Labels: the signal is known at the close of ``t``, the position is entered at the close
  of ``t + lag`` (default 1) and the return is measured over ``h`` trading days:
  ``fwd_ret_{h}d = P[t + lag + h] / P[t + lag] - 1``. Labels use prices strictly after ``t``
  and are NaN when that price does not exist (end of sample, or the stock stopped trading).
* Features and labels are computed on the dense price history of every ticker that was ever
  an index member, so a stock that joins the index already has its trailing history. Rows are
  restricted to point-in-time members (plus the optional liquidity filter) only afterwards.
* Missing values are never imputed: a price gap is forward-filled (causally, from the last
  observed price) for at most ``max_gap_days`` days, everything else stays NaN, and a NaN
  feature stays NaN after z-scoring. A row is only kept if the stock actually traded that day.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from src.utils import warmup_days, winsorize_by_date, zscore_by_date

PRICE_FIELDS = ("open", "close", "closeadj", "volume")


def _min_periods(window: int, min_obs_frac: float) -> int:
    return min(window, max(2, math.ceil(window * min_obs_frac)))


def fill_short_gaps(frame: pd.DataFrame, max_gap_days: int) -> pd.DataFrame:
    """Causally forward-fill at most ``max_gap_days`` consecutive missing days.

    Only the last observed price is carried forward (information known at the time), and
    never past ``max_gap_days`` days, so long gaps and the days after a delisting stay NaN.
    """
    return frame.ffill(limit=max_gap_days) if max_gap_days > 0 else frame


def prepare_prices(prices: dict[str, pd.DataFrame], max_gap_days: int) -> dict[str, pd.DataFrame]:
    """Gap-fill the price series used for returns; volume and open are left untouched."""
    prepared = dict(prices)
    for field in ("close", "closeadj"):
        prepared[field] = fill_short_gaps(prices[field], max_gap_days)
    return prepared


def n_day_return(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """``P[t] / P[t-n] - 1``; needs only the two end points to exist."""
    return prices / prices.shift(n) - 1


def rolling_beta(
    returns: pd.DataFrame, market: pd.Series, window: int, min_obs_frac: float
) -> pd.DataFrame:
    """Rolling OLS slope of each stock's return on the market return (cov / var)."""
    min_periods = _min_periods(window, min_obs_frac)
    covariance = returns.rolling(window, min_periods=min_periods).cov(market)
    variance = market.rolling(window, min_periods=min_periods).var()
    return covariance.div(variance.where(variance > 0), axis=0)


def average_dollar_volume(
    close: pd.DataFrame, volume: pd.DataFrame, window: int, min_obs_frac: float
) -> pd.DataFrame:
    """Trailing mean of ``close * volume`` (both split adjusted, so splits cancel)."""
    return (close * volume).rolling(window, min_periods=_min_periods(window, min_obs_frac)).mean()


def feature_names(cfg: dict[str, Any]) -> list[str]:
    """Names of the raw features produced by ``iter_features``, in output order."""
    f = cfg["features"]
    names = [f"ret_{n}d" for n in f["reversal_windows"]]
    names += [f"ret_{n}d" for n in f["momentum_windows"]]
    names += ["mom_12_1"]
    names += [f"vol_{w}d" for w in f["volatility_windows"]]
    names += [f"beta_{f['beta_window']}d"]
    names += [f"dollar_volume_{f['dollar_volume_window']}d"]
    names += [f"log_dollar_volume_{f['dollar_volume_window']}d"]
    names += [f"dist_high_{f['high_window']}d"]
    names += ["overnight_ret", "intraday_ret"]
    names += [f"trend_{w}d" for w in f["trend_windows"]]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate feature names from config windows: {duplicates}")
    return names


def iter_features(
    prices: dict[str, pd.DataFrame], member_mask: pd.DataFrame, cfg: dict[str, Any]
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(name, date x ticker frame)`` for every raw feature, one at a time.

    ``prices`` holds ``open, close, closeadj, volume`` (already gap-filled by
    ``prepare_prices``); ``member_mask`` is the boolean point-in-time index membership used
    only to build the equal-weighted market return for beta.
    """
    f = cfg["features"]
    frac = f["min_obs_frac"]
    adjusted = prices["closeadj"]
    daily_return = n_day_return(adjusted, 1)

    for n in f["reversal_windows"]:
        yield f"ret_{n}d", n_day_return(adjusted, n)
    for n in f["momentum_windows"]:
        yield f"ret_{n}d", n_day_return(adjusted, n)
    lookback, skip = f["momentum_12_1"]["lookback"], f["momentum_12_1"]["skip"]
    yield "mom_12_1", adjusted.shift(skip) / adjusted.shift(lookback) - 1

    for w in f["volatility_windows"]:
        yield f"vol_{w}d", daily_return.rolling(w, min_periods=_min_periods(w, frac)).std()

    market_return = daily_return.where(member_mask).mean(axis=1)
    w = f["beta_window"]
    yield f"beta_{w}d", rolling_beta(daily_return, market_return, w, frac)

    w = f["dollar_volume_window"]
    dollar_volume = average_dollar_volume(prices["close"], prices["volume"], w, frac)
    yield f"dollar_volume_{w}d", dollar_volume
    yield f"log_dollar_volume_{w}d", np.log(dollar_volume.where(dollar_volume > 0))

    w = f["high_window"]
    rolling_high = adjusted.rolling(w, min_periods=_min_periods(w, frac)).max()
    yield f"dist_high_{w}d", adjusted / rolling_high - 1

    adjusted_open = prices["open"] * adjusted / prices["close"]
    yield "overnight_ret", adjusted_open / adjusted.shift(1) - 1
    yield "intraday_ret", adjusted / adjusted_open - 1

    for w in f["trend_windows"]:
        moving_average = adjusted.rolling(w, min_periods=_min_periods(w, frac)).mean()
        yield f"trend_{w}d", adjusted / moving_average - 1


def compute_features(
    prices: dict[str, pd.DataFrame], member_mask: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """All raw features as a dict of date x ticker frames (prices must be prepared)."""
    return dict(iter_features(prices, member_mask, cfg))


def delisting_info(
    adjusted: pd.DataFrame, delistings: pd.DataFrame | None, delisting_cfg: dict[str, Any]
) -> pd.DataFrame:
    """Per-ticker terminal-event table: ``last_pos, last_price, delisted, reason, haircut``.

    A ticker counts as delisted when its last price is more than ``grace_days`` rows before
    the end of the price history (so stocks that are still trading, or whose latest rows are
    just late, are not). ``reason`` comes from the nearest ``delistings`` event within 30 days
    of the last price, else ``unknown``; ``haircut`` is the configured extra return applied to
    the last price for that reason.
    """
    n = len(adjusted)
    valid = adjusted.notna().to_numpy()
    has_any = valid.any(axis=0)
    last_pos = np.where(has_any, n - 1 - valid[::-1].argmax(axis=0), -1)
    last_price = np.where(has_any, adjusted.to_numpy()[np.maximum(last_pos, 0), np.arange(valid.shape[1])], np.nan)
    info = pd.DataFrame(
        {
            "last_pos": last_pos,
            "last_price": last_price,
            "delisted": has_any & (last_pos < n - 1 - delisting_cfg["grace_days"]),
            "reason": "unknown",
        },
        index=adjusted.columns,
    )
    if delistings is not None and not delistings.empty:
        last_date = pd.Series(
            np.where(has_any, adjusted.index[np.maximum(last_pos, 0)], pd.NaT), index=adjusted.columns
        )
        events = {ticker: group for ticker, group in delistings.groupby("ticker")}
        for ticker in info.index[info["delisted"]]:
            group = events.get(ticker)
            if group is None:
                continue
            distance = (group["date"] - last_date[ticker]).abs()
            if distance.min() <= pd.Timedelta(days=30):
                info.loc[ticker, "reason"] = group.loc[distance.idxmin(), "reason"]
    haircuts = delisting_cfg["haircut"]
    info["haircut"] = info["reason"].map(lambda r: haircuts.get(r, haircuts["default"]))
    return info


def compute_labels(
    adjusted: pd.DataFrame,
    horizons: list[int],
    lag: int,
    terminal: pd.DataFrame | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``fwd_ret_{h}d = P[t+lag+h] / P[t+lag] - 1`` for each horizon.

    With ``terminal`` (from ``delisting_info``), a stock that stopped trading before day
    ``t+lag+h`` is exited at its last price times ``1 + haircut`` instead of getting a NaN
    label, and ``fwd_terminal_{h}d`` flags those labels (1.0, else 0.0). Interior gaps and the
    end of the sample are never filled.
    """
    entry = adjusted.shift(-lag)
    if terminal is not None:
        info = terminal.reindex(adjusted.columns)
        last_pos = info["last_pos"].to_numpy()
        delisted = info["delisted"].fillna(False).to_numpy(dtype=bool)
        terminal_price = (info["last_price"] * (1 + info["haircut"])).to_numpy()
        rows = np.arange(len(adjusted))[:, None]
    for h in horizons:
        exit_price = adjusted.shift(-(lag + h))
        if terminal is None:
            yield f"fwd_ret_{h}d", exit_price / entry - 1
            continue
        stopped = delisted[None, :] & (rows + lag + h > last_pos[None, :]) & (
            rows + lag <= last_pos[None, :]
        )
        filled = exit_price.where(~stopped, pd.DataFrame(
            np.broadcast_to(terminal_price, adjusted.shape), index=adjusted.index, columns=adjusted.columns
        ))
        label = filled / entry - 1
        yield f"fwd_ret_{h}d", label
        flag = pd.DataFrame(stopped & label.notna().to_numpy(), index=adjusted.index, columns=adjusted.columns)
        yield f"fwd_terminal_{h}d", flag.astype("float32")


def universe_mask(
    prices: dict[str, pd.DataFrame], member_mask: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Index members that pass the liquidity filter (trailing mean dollar volume up to day t)."""
    liquidity = cfg["universe"]["liquidity"]
    mask = member_mask.copy()
    if liquidity["min_avg_dollar_volume"] > 0:
        adv = average_dollar_volume(
            prices["close"], prices["volume"], liquidity["window"], cfg["features"]["min_obs_frac"]
        )
        mask = mask & (adv >= liquidity["min_avg_dollar_volume"])
    return mask


def build_panel(
    raw_prices: dict[str, pd.DataFrame],
    member_mask: pd.DataFrame,
    cfg: dict[str, Any],
    delistings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the (date, ticker) modelling panel.

    Columns: raw features, ``z_<feature>`` (winsorised then z-scored within each date over
    the rows kept), labels ``fwd_ret_{h}d`` and the cross-sectionally demeaned
    ``fwd_ret_{h}d_xs``, plus ``fwd_terminal_{h}d`` flags for labels that exit a delisted stock
    at its last price (``cfg['labels']['delisting']``; ``delistings`` supplies the reasons).
    The first ``warmup_days`` dates are dropped.
    """
    prices = prepare_prices(raw_prices, cfg["data"]["max_gap_days"])
    traded_today = raw_prices["closeadj"].notna()
    mask = universe_mask(prices, member_mask, cfg) & traded_today
    mask.iloc[: warmup_days(cfg), :] = False

    rows, cols = np.nonzero(mask.to_numpy())
    index = pd.MultiIndex.from_arrays(
        [mask.index[rows], mask.columns[cols]], names=["date", "ticker"]
    )
    limits = tuple(cfg["preprocessing"]["winsor_limits"])

    def cells(frame: pd.DataFrame) -> np.ndarray:
        return frame.to_numpy()[rows, cols].astype("float32")

    raw, standardised, labels = {}, {}, {}
    for name, frame in iter_features(prices, member_mask, cfg):
        in_universe = frame.where(mask)
        raw[name] = cells(in_universe)
        standardised[f"z_{name}"] = cells(zscore_by_date(winsorize_by_date(in_universe, limits)))
    delisting_cfg = cfg["labels"].get("delisting", {"enabled": False})
    terminal = (
        delisting_info(raw_prices["closeadj"], delistings, delisting_cfg)  # true last print
        if delisting_cfg["enabled"]
        else None
    )
    for name, frame in compute_labels(
        prices["closeadj"], cfg["labels"]["horizons"], cfg["labels"]["lag"], terminal
    ):
        in_universe = frame.where(mask)
        labels[name] = cells(in_universe)
        if name.startswith("fwd_ret_"):
            labels[f"{name}_xs"] = cells(in_universe.sub(in_universe.mean(axis=1), axis=0))

    return pd.DataFrame({**raw, **standardised, **labels}, index=index)
