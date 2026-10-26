# equity-alpha

Cross-Sectional Equity Alpha & Portfolio Construction

Research question: can cross-sectional price/volume signals predict relative US equity returns out-of-sample, and do the forecasts survive realistic transaction costs?

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env   # then set SHARADAR_API_KEY
pytest
```

Run notebooks and scripts from the repository root, or add it to `sys.path`, so that
`from src.data import get_universe_prices` resolves. See [`data/README.md`](data/README.md)
for the data layer (fields, adjustments, survivorship, known limitations).

## Build the modelling dataset

```bash
python -m src.build_panel            # parameters: configs/features.yaml
```

Writes `data/processed/panel.parquet` (gitignored) indexed by `(date, ticker)` and a short
summary to `reports/panel_summary.txt` (date range, tickers per date, rows, missingness).
The first run downloads from the Sharadar API; later runs read the cache.

Columns: the raw features below, `z_<feature>` (winsorised at the 1st/99th percentile, then
z-scored, both **within each date** over the rows kept), and the labels `fwd_ret_{h}d` plus the
cross-sectionally demeaned `fwd_ret_{h}d_xs`, and the flags `fwd_terminal_{h}d`. Rows are S&P 500 members on that date (point in
time) that traded that day; the first 252 trading days are dropped as warm-up. There is no
liquidity filter by default (an optional dollar-volume cut-off exists in `configs/features.yaml`; see
[`data/README.md`](data/README.md) for why it is off).
Features are computed on each ticker's full price history, so a stock that joins the index
already has its trailing windows.

### Timing convention

Features in row `t` use data up to and including the close of `t`. The signal is known at the
close of `t`, the position is entered at the close of `t + lag` (default `lag = 1`) and the
label is `P[t+lag+h] / P[t+lag] - 1` on `closeadj`, for `h` in {1, 5, 21} trading days. The
last `lag + h` dates of the sample have a NaN label. A stock that stops trading before the label
date (94% are takeovers) is exited at its last price and flagged in `fwd_terminal_{h}d` instead
of being dropped; see [`data/README.md`](data/README.md). 5d and 21d labels overlap across dates,
so they are autocorrelated.

### Missing values

No imputation. Prices are forward-filled (causally) for at most 2 trading days; return-style
features only need their two end-point prices; rolling statistics need at least 80% of the
window; a row needs an actual price that day. Anything else is NaN and stays NaN after
z-scoring.

### Sanity-check figures

Produced by [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) after a
panel build; regenerate them only when the panel or config changes.

- [`universe_size.png`](reports/figures/universe_size.png): S&P 500 members per date in the panel.
- [`coverage_by_year.png`](reports/figures/coverage_by_year.png): share of non-missing values by
  feature and label, per year.
- [`feature_correlations.png`](reports/figures/feature_correlations.png): pooled correlations of
  the standardised features, showing which ones overlap.

## Features

All windows are trading days (21 = 1 month, 63 = 3, 126 = 6, 252 = 12). `P` is `closeadj`,
`r_t = P_t / P_{t-1} - 1`. Few features, each tied to a documented effect:

| Feature | Formula | Economic rationale |
|---|---|---|
| `ret_1d`, `ret_5d` | `P_t / P_{t-n} - 1` | Short-term reversal: liquidity provision and overreaction; losers of the last days tend to rebound |
| `ret_21d`, `ret_63d`, `ret_126d`, `ret_252d` | `P_t / P_{t-n} - 1` | Momentum across horizons: underreaction to news, slow information diffusion |
| `mom_12_1` | `P_{t-21} / P_{t-252} - 1` | Classic 12-1 momentum; skipping the last month removes the short-term reversal |
| `vol_21d`, `vol_63d` | std of `r` over the window | Total-risk / lottery-demand effect: high-vol stocks are often overpriced (low-volatility anomaly) |
| `beta_252d` | `cov(r_i, r_m) / var(r_m)`, `r_m` = mean return of that day's S&P 500 members | Market risk (betting-against-beta); equal-weighted proxy because SPY is not in the data bundle |
| `dollar_volume_21d`, `log_dollar_volume_21d` | mean of `close * volume` over 21 days (and its log) | Size / liquidity premium and a proxy for trading cost |
| `dist_high_252d` | `P_t / max(P_{t-251..t}) - 1` | 52-week-high anchoring: investors anchor on the high and underreact to news near it |
| `overnight_ret` | `open_adj_t / P_{t-1} - 1` | Overnight vs intraday split: overnight drift reflects informed/institutional flow, intraday reflects retail trading and reversal |
| `intraday_ret` | `P_t / open_adj_t - 1` | See above; `(1+overnight)(1+intraday) = 1 + r_t` |
| `trend_50d`, `trend_200d` | `P_t / SMA_n(P) - 1` | Trend following: price relative to its moving average captures medium-term momentum |

`open_adj = open * closeadj / close` puts the split-adjusted open on the same dividend basis as
`closeadj`. `ret_21d` and `ret_252d` overlap with `mom_12_1`, and `ret_1d`/`ret_5d` with the
overnight/intraday split; the features are deliberately correlated, so evaluate them one at a
time or after orthogonalisation.

## Tests

`pytest` runs offline (HTTP is mocked, features use synthetic prices). The key test checks that
every feature at date `t` computed from data truncated at `t` equals the value computed from the
full dataset (including z-scores and the optional liquidity filter). Others cover labels and lag, per-date
standardisation, point-in-time membership, hand-computed returns / momentum / beta, and missing
data handling.
