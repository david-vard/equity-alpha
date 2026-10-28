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

## Signal research

Univariate baselines before any model. 

- H0: a feature has **zero** mean cross-sectional Spearman IC with `fwd_ret_{h}d`
- H1: a feature has **non-zero** mean cross-sectional Spearman IC with `fwd_ret_{h}d` 

The design is fixed in [`configs/research.yaml`](configs/research.yaml)
and was committed before any result was generated.

```bash
python -m src.run_signal_research     # tables, figures and run_metadata.json -> reports/signal_research/
```

The run does not start unless `configs/research.yaml` is committed and unmodified.
`run_metadata.json` records the config commit and hash, code revision (and whether `src`/`configs`
were dirty), hashes of the feature config and the panel, the research cutoff and package versions.
Plots and narrative: [`notebooks/02_signal_research.ipynb`](notebooks/02_signal_research.ipynb).

### Design

- **Holdout and embargo.** `holdout_start = 2022-01-01`. The research sample ends
  `lag + max(max horizon, decay.n_days) = 22` trading days before the first holdout date, so every
  label and shifted return used ends strictly before the holdout. The last research date is derived
  from the panel's trading calendar and written to `run_metadata.json`.
  `src.signals.load_research_panel` is the only way research code reads the panel and never returns
  later rows.
- **Expected signs and families.** Each feature has a family and a sign prior for its IC
  (+1 / -1, 0 = none) in the config. One sign per feature across horizons, so priors are loose
  where the economics differ by horizon (e.g. `trend_50d` mixes reversal and momentum). Sign
  consistency is a descriptive column and not part of any test. `ret_21d` is the monthly-reversal
  signal (no prior); `beta_252d` and the dollar-volume features carry no prior because raw-return
  evidence is mixed. Dollar volume proxies size and trading activity (the panel has no market cap).
- **Primary test.** Newey-West t of the mean daily IC over all research dates: Bartlett kernel,
  `max(h, floor(4 (T/100)^(2/9)))` lags because h-day labels overlap, two-sided normal p-values
  (slightly anti-conservative on the ~270-point non-overlapping series). Statistics on
  non-overlapping series (every h-th date, offset 0 and averaged over all h offsets) use the
  plug-in lag only.
- **Multiple testing.** Holm, Benjamini-Hochberg and Benjamini-Yekutieli (BH assumes positive
  dependence, BY does not) at `alpha = 0.05`, plus a flag for |t| > 3. Three separate families of
  17 x 3 = 51 tests, each adjusted on its own and not pooled: (1) the IC t-tests (primary),
  (2) the gross long-short spreads, (3) the univariate Fama-MacBeth slopes. The multivariate
  Fama-MacBeth is not adjusted. `dollar_volume_21d` and `log_dollar_volume_21d` are rank-identical,
  so the effective family has 48 tests; `*_eff` columns give that adjustment.
- **Annualisation.** Sharpe ratios and IC IR of non-overlapping h-day series are scaled by
  `sqrt(252 / h)`. `ic_ir` in `ic_summary.csv` is per date and not annualised. Fama-MacBeth
  coefficients are per h-day return per 1 SD of the feature, not annualised.
- **Portfolios.** Equal-weight deciles on the raw feature; tied values share a bucket (so
  `dist_high_252d`, which is exactly 0 at a new high, gives unequal deciles) and NaN is excluded.
  Rebalance every h-th date. Long-short columns are prefixed `gross_` (no costs). Turnover is the
  one-way turnover of the bottom (D1) and top (D10) legs by feature value, so for a negative
  expected sign the long leg is D1.
- **Decay.** IC of the feature at `t` against the 1-day return on holding day `k + 1`, for
  `k = 0..20`: 21 daily returns that make up the 21-day holding window (position entered at the
  close of `t + 1`, so `k = 0` is the first tradable day). Built from a shifted wide frame, so
  returns exist only for names still in the index at `t + k` (a mild survivorship effect). The
  half-life is the first lag where the 3-point smoothed IC falls to half its `k = 0` value,
  linearly interpolated; it is reported only if |t| of the `k = 0` IC is at least 2, and is NaN if
  the IC does not halve within the window. Cumulative IC is a sum of daily correlations, not the IC
  of the cumulative return.
- **Fama-MacBeth.** Per-date OLS of `fwd_ret_{h}d_xs` on `z_` features, univariate for each feature
  and multivariate with one pre-specified representative per family, with VIFs and condition
  numbers. Newey-West t as above.
- **Stability.** IC by calendar year and sub-period (the post-2020 window covers only 2020-01 to
  2021-11), first vs second half, and with / without `fwd_terminal` rows.

Statistical significance (`nw_t`, adjusted p, `reject_*`), economic magnitude (IC, spread, Fama-MacBeth
slope), gross portfolio performance and turnover are separate columns / tables and should be read
together: a significant IC says nothing about whether the spread survives turnover.

## Tests

`pytest` runs offline (HTTP is mocked, features use synthetic prices). The key test checks that
every feature at date `t` computed from data truncated at `t` equals the value computed from the
full dataset (including z-scores and the optional liquidity filter). Others cover labels and lag, per-date
standardisation, point-in-time membership, hand-computed returns / momentum / beta, and missing
data handling.
