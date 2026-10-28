"""Tranche-based backtest engine: rebalancing, drift, label-consistent exits, costs and daily accounting.

Timing follows the labels. A score known at the close of date position ``s`` is traded at the close
of ``p = s + lag``. Tranche ``k`` of ``n`` (capital ``1 / n``) rebalances on score positions
``start + offset_k + j * h``, with ``offset_k = k * h // n``, so with ``n = h`` every day has
exactly one rebalance.

Between rebalances weights drift with the daily returns ``ret`` (``closeadj`` close to close, NaN
where the name has no panel row, which then earns 0 that day). On a tranche's exit day ``p + h``,
each held name also books the rest of its label, ``(1 + label) / prod(1 + observed returns) - 1``,
so the tranche's h-day return equals the labels exactly, including a delisted stock's exit at
its last price. At the same close the tranche trades from its drifted weights to the new target;
costs are charged on that trade with the vol and ADV known at ``s`` (carried forward for a held
name that has no row). A due tranche without usable scores goes flat, which at the end of the data
liquidates the book.

Everything is per $ of total capital (fixed, no compounding), so tranche rows add up to the total.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.costs import COMPONENTS, CostModel, borrow_cost, trade_costs
from src.metrics import newey_west_lag, newey_west_tstat, sharpe

REALISED_BETA_WINDOW = 63  # trading days of the rolling regression on the equal-weighted market
DAILY_COLUMNS = (
    "gross_pnl", "commission", "spread", "impact", "cost", "borrow", "net_pnl", "gross_exposure",
    "net_exposure", "beta_ex_ante", "beta_realised", "turnover",
)


@dataclass
class MarketData:
    """Date x ticker arrays on one calendar (float64; NaN where unknown).

    ``score``: the signal at the score date; ``ret``: the day's return from the panel row;
    ``label``: ``fwd_ret_{h}d`` at the score date; ``vol`` / ``beta`` / ``adv``: ``vol_63d``,
    ``beta_252d`` and ``dollar_volume_21d`` from the panel row.
    """

    dates: pd.DatetimeIndex
    tickers: pd.Index
    score: np.ndarray
    ret: np.ndarray
    label: np.ndarray
    vol: np.ndarray
    beta: np.ndarray
    adv: np.ndarray

    def __post_init__(self) -> None:
        shape = (len(self.dates), len(self.tickers))
        for name in ("score", "ret", "label", "vol", "beta", "adv"):
            value = np.asarray(getattr(self, name), dtype="float64")
            if value.shape != shape:
                raise ValueError(f"MarketData.{name} has shape {value.shape}, expected {shape}")
            setattr(self, name, value)
        if not self.dates.is_monotonic_increasing or self.dates.has_duplicates:
            raise ValueError("MarketData.dates must be strictly increasing")

    def check_holdout(self, holdout_start: str | pd.Timestamp) -> None:
        if len(self.dates) and self.dates.max() >= pd.Timestamp(holdout_start):
            raise ValueError(f"a date on or after holdout_start {holdout_start} reached the backtest engine")


@dataclass
class BacktestResult:
    tranches: pd.DataFrame            # one row per (date, tranche), DAILY_COLUMNS + rebalanced, fallback
    total: pd.DataFrame               # one row per date: tranche sums plus market_return
    weights: pd.DataFrame             # per rebalance and name: target weight, trade and its cost
    fallbacks: Counter = field(default_factory=Counter)
    first_full_date: pd.Timestamp | None = None   # first day on which every tranche holds its first target
    n_rebalances: int = 0


def ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaN down each column (causal: only earlier rows are used)."""
    return pd.DataFrame(a).ffill().to_numpy()


def cost_inputs(a: np.ndarray) -> np.ndarray:
    """Carried-forward values; still missing ones get that date's cross-sectional median."""
    filled = ffill(a)
    with np.errstate(all="ignore"):
        median = np.nanmedian(np.where(np.isfinite(filled), filled, np.nan), axis=1)
    return np.where(np.isfinite(filled), filled, median[:, None])


def run_backtest(
    data: MarketData,
    build: Any,
    costs: CostModel,
    horizon: int,
    lag: int,
    n_tranches: int,
    holdout_start: str | pd.Timestamp,
    start: int | None = None,
    record_weights: bool = True,
) -> BacktestResult:
    """Simulate ``n_tranches`` staggered tranches of a construction over ``data``.

    ``build(data, s, current, capital_usd, costs)`` returns the ``Target`` for score position
    ``s`` given the tranche's current (drifted) weights per $ of its capital. ``start`` is the
    first score position (default: the first date with any finite score).
    """
    data.check_holdout(holdout_start)
    T, N = data.score.shape
    if start is None:
        has_score = np.isfinite(data.score).any(axis=1)
        if not has_score.any():
            raise ValueError("no scores to trade")
        start = int(np.argmax(has_score))
    n, h, aum = n_tranches, horizon, costs.aum_usd
    cap = 1.0 / n
    offsets = np.array([k * h // n for k in range(n)])
    sigma_cost, adv_cost, beta_ff = cost_inputs(data.vol), cost_inputs(data.adv), ffill(data.beta)

    H = np.zeros((n, N))                 # holdings, $ per $ of total capital
    G = np.ones((n, N))                  # product of observed (1 + r) since the tranche's last trade
    L = np.full((n, N), np.nan)          # labels of the current holding window
    exit_at = np.full(n, -1)
    started = np.zeros(n, dtype=bool)
    out = {c: np.zeros((T, n)) for c in DAILY_COLUMNS if c != "beta_realised"}
    rebalanced = np.zeros((T, n), dtype=bool)
    fallback = np.full((T, n), "", dtype=object)
    counts: Counter = Counter()
    first_full, n_rebalances, records = None, 0, []

    for p in range(T):
        r = np.nan_to_num(data.ret[p], nan=0.0)
        gross = H @ r
        out["borrow"][p] = borrow_cost(np.clip(-H, 0, None).sum(axis=1) * aum, costs) / aum
        H *= 1 + r
        G *= 1 + r

        for k in np.flatnonzero(exit_at == p):   # book the rest of each label
            held = H[k] != 0
            with np.errstate(divide="ignore", invalid="ignore"):
                resid = np.where(held & np.isfinite(L[k]) & (G[k] > 0), (1 + L[k]) / G[k] - 1, 0.0)
            gross[k] += H[k] @ resid
            H[k] *= 1 + resid
            exit_at[k] = -1
        out["gross_pnl"][p] = gross

        s = p - lag
        for k in range(n):
            first = start + offsets[k]
            if s < first or (s - first) % h:
                continue
            current = H[k] / cap
            target = build(data, s, current, cap * aum, costs)
            w = np.asarray(target.weights, dtype="float64")
            dw = w - current
            traded = np.flatnonzero(dw != 0)
            if len(traded):
                tc = trade_costs(dw[traded], sigma_cost[s, traded], adv_cost[s, traded], cap * aum, costs)
                for c in COMPONENTS:
                    out[c][p, k] = tc[f"{c}_usd"].sum() / aum
                if record_weights:
                    records.append(pd.DataFrame({
                        "score_date": data.dates[s], "trade_date": data.dates[p], "tranche": k,
                        "ticker": data.tickers[traded], "weight": w[traded], "trade": dw[traded],
                        "cost_usd": tc["total_usd"].to_numpy(),
                    }))
                    kept = np.flatnonzero((dw == 0) & (w != 0))
                    if len(kept):
                        records.append(pd.DataFrame({
                            "score_date": data.dates[s], "trade_date": data.dates[p], "tranche": k,
                            "ticker": data.tickers[kept], "weight": w[kept], "trade": 0.0, "cost_usd": 0.0,
                        }))
            out["turnover"][p, k] = 0.5 * np.abs(dw).sum() * cap
            H[k], G[k], L[k] = w * cap, 1.0, data.label[s]
            exit_at[k] = p + h if (w != 0).any() else -1
            rebalanced[p, k] = True
            n_rebalances += 1
            if target.fallback:
                fallback[p, k] = target.fallback
                counts[target.fallback] += 1
            started[k] = True
            if first_full is None and started.all():
                first_full = p + 1

        out["gross_exposure"][p] = np.abs(H).sum(axis=1)
        out["net_exposure"][p] = H.sum(axis=1)
        out["beta_ex_ante"][p] = H @ np.nan_to_num(beta_ff[p], nan=0.0)

    out["cost"] = out["commission"] + out["spread"] + out["impact"]
    out["net_pnl"] = out["gross_pnl"] - out["cost"] - out["borrow"]
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = np.isfinite(data.ret)
        market = np.where(observed.any(axis=1), np.nansum(data.ret, axis=1) / observed.sum(axis=1), np.nan)
    out["beta_realised"] = realised_beta(out["gross_pnl"], market, data.dates)

    tranches = pd.concat(
        [pd.DataFrame({"date": data.dates, "tranche": k, **{c: out[c][:, k] for c in DAILY_COLUMNS},
                       "rebalanced": rebalanced[:, k], "fallback": fallback[:, k]}) for k in range(n)],
        ignore_index=True,
    ).sort_values(["date", "tranche"], ignore_index=True)
    total = pd.DataFrame({c: out[c].sum(axis=1) for c in DAILY_COLUMNS}, index=data.dates)
    total["market_return"] = market
    total["n_rebalanced"] = rebalanced.sum(axis=1)
    weights = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["score_date", "trade_date", "tranche", "ticker", "weight", "trade", "cost_usd"])
    return BacktestResult(
        tranches=tranches, total=total, weights=weights, fallbacks=counts,
        first_full_date=data.dates[first_full] if first_full is not None and first_full < T else None,
        n_rebalances=n_rebalances,
    )


def realised_beta(pnl: np.ndarray, market: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Rolling ``REALISED_BETA_WINDOW``-day OLS beta of each P&L column on the market return.

    Covariance is linear in the P&L, so tranche betas add up to the total's beta.
    """
    frame = pd.DataFrame(pnl, index=dates)
    m = pd.Series(market, index=dates)
    w = REALISED_BETA_WINDOW
    cov = frame.rolling(w, min_periods=w).cov(m)
    var = m.rolling(w, min_periods=w).var()
    return cov.div(var, axis=0).to_numpy()


def performance(
    total: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None, horizon: int, trading_days: int,
) -> dict[str, Any]:
    """Statistics of the daily total series between ``start`` and ``end`` (inclusive).

    Sharpe = mean / sd * sqrt(trading_days); NW t of the mean daily net return with
    ``max(h, plug-in)`` lags (tranches overlap); turnover is one-way per year; the break-even cost
    multiplier is mean gross / mean (trading cost + borrow), i.e. at this run's trades.
    """
    sub = total.loc[start:end]
    net, gross = sub["net_pnl"].to_numpy(), sub["gross_pnl"].to_numpy()
    cost = (sub["cost"] + sub["borrow"]).to_numpy()
    n = len(sub)
    nw = newey_west_tstat(net, newey_west_lag(n, horizon))
    equity = np.cumsum(net)
    drawdown = equity - np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] if n else np.array([np.nan])
    mean_cost = float(cost.mean()) if n else np.nan
    return {
        "start": sub.index[0].date() if n else None,
        "end": sub.index[-1].date() if n else None,
        "n_days": n,
        "net_return_ann": float(net.mean() * trading_days) if n else np.nan,
        "net_vol_ann": float(net.std(ddof=1) * np.sqrt(trading_days)) if n > 1 else np.nan,
        "net_sharpe": sharpe(net, trading_days),
        "net_nw_t": nw["t"],
        "nw_lag": newey_west_lag(n, horizon),
        "gross_return_ann": float(gross.mean() * trading_days) if n else np.nan,
        "gross_sharpe": sharpe(gross, trading_days),
        "cost_ann": float(sub["cost"].mean() * trading_days) if n else np.nan,
        "borrow_ann": float(sub["borrow"].mean() * trading_days) if n else np.nan,
        "annual_turnover": float(sub["turnover"].mean() * trading_days) if n else np.nan,
        "breakeven_cost_multiplier": float(gross.mean() / mean_cost) if n and mean_cost > 0 else np.nan,
        "max_drawdown": float(drawdown.min()) if n else np.nan,
        "mean_gross_exposure": float(sub["gross_exposure"].mean()) if n else np.nan,
        "mean_abs_beta_ex_ante": float(sub["beta_ex_ante"].abs().mean()) if n else np.nan,
    }
