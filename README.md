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

### Results (research sample 1999-01-04 to 2021-11-30, 5,766 dates)

Of 51 IC tests, 11 are rejected by Holm, 14 by Benjamini-Yekutieli and 18 by Benjamini-Hochberg at 5%
(13 have |t| > 3). Short-term reversal (`ret_1d`, `ret_5d`, `intraday_ret`) is the only effect that is
significant under every correction and, at short horizons, in the long-short and Fama-MacBeth tests; momentum and low volatility
have small, persistent ICs at the one-day horizon whose long-short spreads are not significant; the remaining
features are null. All long-short figures are gross (no costs). IC is per date, NW t uses `max(h, plug-in)`
lags, p-values are adjusted over the 51 IC tests, Sharpe is the annualised raw D10 - D1 spread on
non-overlapping dates (negative for reversal: it pays to buy D1 and sell D10), and turnover is the mean one-way leg turnover per rebalance. Narrative:
[`notebooks/02_signal_research.ipynb`](notebooks/02_signal_research.ipynb).

Rejected by Benjamini-Hochberg (Holm-rejected rows have Holm p < 0.05):

| Feature | h | IC mean | NW t | Holm p | BH p | Gross L/S Sharpe | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ret_1d` | 1 | -0.0092 | -4.12 | 0.0017 | 0.0003 | -0.81 | 0.85 |
| `ret_1d` | 5 | -0.0119 | -5.60 | <0.0001 | <0.0001 | -0.60 | 0.86 |
| `ret_1d` | 21 | -0.0073 | -3.45 | 0.0227 | 0.0026 | -0.14 | 0.87 |
| `ret_5d` | 1 | -0.0104 | -4.93 | <0.0001 | <0.0001 | -0.86 | 0.40 |
| `ret_5d` | 5 | -0.0155 | -4.23 | 0.0011 | 0.0002 | -0.69 | 0.87 |
| `ret_5d` | 21 | -0.0115 | -2.69 | 0.2610 | 0.0231 | -0.17 | 0.86 |
| `ret_126d` | 1 | +0.0083 | +3.09 | 0.0788 | 0.0084 | -0.13 | 0.08 |
| `ret_252d` | 1 | +0.0113 | +3.87 | 0.0046 | 0.0006 | -0.12 | 0.06 |
| `mom_12_1` | 1 | +0.0134 | +4.74 | 0.0001 | <0.0001 | +0.05 | 0.06 |
| `mom_12_1` | 5 | +0.0151 | +2.57 | 0.3608 | 0.0309 | +0.01 | 0.14 |
| `vol_21d` | 1 | -0.0118 | -3.97 | 0.0032 | 0.0005 | +0.15 | 0.09 |
| `vol_63d` | 1 | -0.0122 | -3.87 | 0.0046 | 0.0006 | +0.13 | 0.04 |
| `beta_252d` | 1 | -0.0100 | -2.73 | 0.2375 | 0.0218 | +0.13 | 0.02 |
| `dist_high_252d` | 1 | +0.0089 | +2.97 | 0.1140 | 0.0109 | -0.33 | 0.16 |
| `trend_200d` | 1 | +0.0070 | +2.53 | 0.3933 | 0.0328 | -0.19 | 0.08 |
| `intraday_ret` | 1 | -0.0087 | -4.19 | 0.0013 | 0.0002 | -0.75 | 0.85 |
| `intraday_ret` | 5 | -0.0106 | -5.37 | <0.0001 | <0.0001 | -0.63 | 0.86 |
| `intraday_ret` | 21 | -0.0062 | -3.05 | 0.0892 | 0.0090 | -0.02 | 0.86 |

<details><summary>All 51 feature x horizon tests</summary>

| Feature | h | IC mean | NW t | Holm p | BH p | Gross L/S Sharpe | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ret_1d` | 1 | -0.0092 | -4.12 | 0.0017 | 0.0003 | -0.81 | 0.85 |
| `ret_1d` | 5 | -0.0119 | -5.60 | <0.0001 | <0.0001 | -0.60 | 0.86 |
| `ret_1d` | 21 | -0.0073 | -3.45 | 0.0227 | 0.0026 | -0.14 | 0.87 |
| `ret_5d` | 1 | -0.0104 | -4.93 | <0.0001 | <0.0001 | -0.86 | 0.40 |
| `ret_5d` | 5 | -0.0155 | -4.23 | 0.0011 | 0.0002 | -0.69 | 0.87 |
| `ret_5d` | 21 | -0.0115 | -2.69 | 0.2610 | 0.0231 | -0.17 | 0.86 |
| `ret_21d` | 1 | -0.0039 | -1.77 | 1.0000 | 0.1297 | -0.45 | 0.20 |
| `ret_21d` | 5 | -0.0095 | -2.18 | 0.9075 | 0.0711 | -0.26 | 0.44 |
| `ret_21d` | 21 | -0.0154 | -2.24 | 0.7950 | 0.0634 | -0.21 | 0.87 |
| `ret_63d` | 1 | +0.0009 | +0.39 | 1.0000 | 0.7902 | -0.39 | 0.12 |
| `ret_63d` | 5 | -0.0031 | -0.64 | 1.0000 | 0.6356 | -0.25 | 0.26 |
| `ret_63d` | 21 | -0.0089 | -1.02 | 1.0000 | 0.4108 | -0.17 | 0.52 |
| `ret_126d` | 1 | +0.0083 | +3.09 | 0.0788 | 0.0084 | -0.13 | 0.08 |
| `ret_126d` | 5 | +0.0063 | +1.15 | 1.0000 | 0.3735 | -0.09 | 0.19 |
| `ret_126d` | 21 | -0.0010 | -0.11 | 1.0000 | 0.9380 | -0.10 | 0.38 |
| `ret_252d` | 1 | +0.0113 | +3.87 | 0.0046 | 0.0006 | -0.12 | 0.06 |
| `ret_252d` | 5 | +0.0116 | +1.91 | 1.0000 | 0.1029 | -0.13 | 0.13 |
| `ret_252d` | 21 | +0.0026 | +0.24 | 1.0000 | 0.8743 | -0.14 | 0.27 |
| `mom_12_1` | 1 | +0.0134 | +4.74 | 0.0001 | <0.0001 | +0.05 | 0.06 |
| `mom_12_1` | 5 | +0.0151 | +2.57 | 0.3608 | 0.0309 | +0.01 | 0.14 |
| `mom_12_1` | 21 | +0.0063 | +0.59 | 1.0000 | 0.6573 | -0.09 | 0.28 |
| `vol_21d` | 1 | -0.0118 | -3.97 | 0.0032 | 0.0005 | +0.15 | 0.09 |
| `vol_21d` | 5 | -0.0079 | -1.28 | 1.0000 | 0.3176 | +0.13 | 0.25 |
| `vol_21d` | 21 | -0.0039 | -0.35 | 1.0000 | 0.8043 | +0.16 | 0.57 |
| `vol_63d` | 1 | -0.0122 | -3.87 | 0.0046 | 0.0006 | +0.13 | 0.04 |
| `vol_63d` | 5 | -0.0079 | -1.19 | 1.0000 | 0.3596 | +0.11 | 0.10 |
| `vol_63d` | 21 | -0.0048 | -0.40 | 1.0000 | 0.7902 | +0.13 | 0.25 |
| `beta_252d` | 1 | -0.0100 | -2.73 | 0.2375 | 0.0218 | +0.13 | 0.02 |
| `beta_252d` | 5 | -0.0054 | -0.71 | 1.0000 | 0.5933 | +0.11 | 0.04 |
| `beta_252d` | 21 | -0.0010 | -0.07 | 1.0000 | 0.9447 | +0.11 | 0.11 |
| `dollar_volume_21d` | 1 | -0.0028 | -1.98 | 1.0000 | 0.0930 | -0.34 | 0.02 |
| `dollar_volume_21d` | 5 | -0.0032 | -1.08 | 1.0000 | 0.3974 | -0.39 | 0.07 |
| `dollar_volume_21d` | 21 | -0.0114 | -1.97 | 1.0000 | 0.0930 | -0.46 | 0.18 |
| `log_dollar_volume_21d` | 1 | -0.0028 | -1.98 | 1.0000 | 0.0930 | -0.34 | 0.02 |
| `log_dollar_volume_21d` | 5 | -0.0032 | -1.08 | 1.0000 | 0.3974 | -0.39 | 0.07 |
| `log_dollar_volume_21d` | 21 | -0.0114 | -1.97 | 1.0000 | 0.0930 | -0.46 | 0.18 |
| `dist_high_252d` | 1 | +0.0089 | +2.97 | 0.1140 | 0.0109 | -0.33 | 0.16 |
| `dist_high_252d` | 5 | +0.0049 | +0.79 | 1.0000 | 0.5641 | -0.25 | 0.31 |
| `dist_high_252d` | 21 | -0.0012 | -0.10 | 1.0000 | 0.9380 | -0.19 | 0.46 |
| `trend_50d` | 1 | -0.0030 | -1.32 | 1.0000 | 0.3055 | -0.47 | 0.16 |
| `trend_50d` | 5 | -0.0096 | -2.12 | 1.0000 | 0.0785 | -0.30 | 0.35 |
| `trend_50d` | 21 | -0.0180 | -2.35 | 0.6167 | 0.0502 | -0.19 | 0.69 |
| `trend_200d` | 1 | +0.0070 | +2.53 | 0.3933 | 0.0328 | -0.19 | 0.08 |
| `trend_200d` | 5 | +0.0043 | +0.75 | 1.0000 | 0.5743 | -0.12 | 0.17 |
| `trend_200d` | 21 | -0.0023 | -0.22 | 1.0000 | 0.8743 | -0.09 | 0.36 |
| `overnight_ret` | 1 | -0.0022 | -1.03 | 1.0000 | 0.4108 | -0.15 | 0.84 |
| `overnight_ret` | 5 | -0.0038 | -2.05 | 1.0000 | 0.0885 | -0.20 | 0.86 |
| `overnight_ret` | 21 | -0.0031 | -1.80 | 1.0000 | 0.1275 | -0.24 | 0.87 |
| `intraday_ret` | 1 | -0.0087 | -4.19 | 0.0013 | 0.0002 | -0.75 | 0.85 |
| `intraday_ret` | 5 | -0.0106 | -5.37 | <0.0001 | <0.0001 | -0.63 | 0.86 |
| `intraday_ret` | 21 | -0.0062 | -3.05 | 0.0892 | 0.0090 | -0.02 | 0.86 |

</details>

## Pre-model diagnostics

```bash
python -m src.run_diagnostics         # tables and figures -> reports/diagnostics/
```

Three checks run before any model was specified. Summary and recommendation:
[`reports/diagnostics/SUMMARY.md`](reports/diagnostics/SUMMARY.md).

- **Why IC shrinks with horizon** (momentum: 0.0134 at 1 day, 0.0063 at 21 days). Compounded daily labels reproduce
  the stored 21-day labels exactly, so this is not a bug. It is a real feature of the data. Momentum and low-vol ICs
  flip sign on the most dispersed days (momentum IC -0.033 on the top 10% of days, +0.018 on the rest), and those days
  dominate a multi-day return. An exact decomposition of the 21-day IC shows this effect removes about 70% of the IC that
  sqrt(h) scaling would predict.
- **Why volatility deciles disagree with the IC.** The high-vol decile has a beta about 1 higher than the low-vol
  decile. Hedged for beta, measured by medians or in return ranks, the vol spread has the IC's sign. Momentum, by
  contrast, works in ranks and medians but not in mean returns (rebounds among the losers).
- **Break-even costs.** Since mid-2010 one-day reversal breaks even at 0-2 bp per side, and five-day reversal at about
  4 bp per side.

## Modelling

Do Ridge, Elastic Net and LightGBM forecast cross-sectional returns better out of sample than simple baselines?
The design is fixed in [`configs/models.yaml`](configs/models.yaml) and was committed before any model was fitted.

```bash
python -m src.run_models --lightgbm-ablation   # tables, figures, run_metadata.json -> reports/models/
```

Like the signal research, the run refuses to start unless `configs/models.yaml` and `configs/research.yaml` are
committed and unmodified. Out-of-sample predictions go to `data/processed/oof_predictions.parquet` (gitignored).
Plots and narrative: [`notebooks/03_modelling.ipynb`](notebooks/03_modelling.ipynb). On macOS, LightGBM needs an
OpenMP runtime; if Homebrew's `libomp` is not installed, the copy bundled with scikit-learn works:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs"
```

### Design

- **Targets.** `fwd_ret_5d_xs` and `fwd_ret_21d_xs`; h = 1 is excluded because its break-even cost is near zero. The
  primary variant ranks the target within each date into [-0.5, 0.5]; the raw target is the alternative. Evaluation
  always uses the raw label `fwd_ret_{h}d`.
- **Features.** The 16 `z_` features (`z_dollar_volume_21d` is dropped as rank-identical to its log). Linear models fill
  missing values with 0, the cross-sectional mean; LightGBM handles NaN natively.
- **Baselines.**
  - B0: reversal, `-z_ret_5d`.
  - B1: equal-weight composite of `ret_5d` (-), `mom_12_1` (+), `vol_63d` (-) and `dist_high_252d` (+).
  - B2: expanding-window Fama-MacBeth, the mean of past per-date slopes on all 16 features.
- **Models.** Pooled Ridge, Elastic Net and LightGBM. The pre-registered grids are small, the trees shallow and
  heavily regularised, and LightGBM's number of trees comes from early stopping. Seeds and thread counts are fixed.
  Each training target is scaled to unit standard deviation so that one penalty grid serves both targets.
- **Walk-forward.**
  - Expanding window, refit every January; the first test year is 2007, so the dot-com era is training data only.
  - Training rows whose label window reaches the first test date are purged.
  - Hyperparameters are chosen by mean daily Spearman IC on the last two years of each training window, purged the
    same way, then refit on the full window. No test date influences its own fold.
- **Evaluation.**
  - Daily rank IC, with Newey-West t as in the signal research and a dispersion-weighted IC alongside.
  - Paired tests of the daily IC difference between models and baselines.
  - Holm adjustment, run separately over the 20 IC tests and over the 36 comparisons.
  - Gross deciles on non-overlapping dates, with the beta-hedged spread as the primary one.
  - Also: stability by sub-period, decay, coefficient paths, LightGBM permutation importance and SHAP, a family
    ablation and a terminal-label check.

### Results (out of sample 2007-01-03 to 2021-11-30, 1.89m stock-days)

**No step in complexity improves the forecast significantly. None of the 36 paired comparisons is rejected by Holm.**
LightGBM on the 5-day rank target is the only predictor with a Holm-significant IC. Its edge over the one-feature
reversal baseline (+0.005 IC, t 1.0) is not distinguishable from zero. At 21 days no predictor is significant, and the
linear models lose in 2007-09, when momentum crashed.

| Predictor (rank target) | IC h=5 | NW t | Holm p | Beta-hedged D10-D1 (bp / 5d) | IC h=21 | NW t |
|---|---:|---:|---:|---:|---:|---:|
| B0 reversal | 0.0123 | 2.54 | 0.20 | 22.3 | 0.0103 | 1.84 |
| B1 composite | 0.0161 | 2.08 | 0.64 | 7.2 | 0.0091 | 0.64 |
| B2 Fama-MacBeth | 0.0109 | 1.75 | 1.00 | 1.3 | -0.0051 | -0.51 |
| Ridge | 0.0120 | 1.73 | 1.00 | 0.5 | -0.0103 | -0.90 |
| Elastic Net | 0.0129 | 2.00 | 0.73 | 4.7 | -0.0037 | -0.43 |
| LightGBM | **0.0174** | **3.50** | **0.010** | 13.2 | 0.0060 | 0.93 |

- **Every complexity step adds little or nothing.** Ridge vs B2 and Elastic Net vs Ridge are about zero. LightGBM vs
  Ridge is positive in all four horizon x target cases (+0.004 to +0.016), but t is only 1.2-1.6.
- **The rank target beats the raw target for every model at 5 days.** Out-of-sample R² against a zero forecast is
  about 0, as expected for cross-sectional forecasts.
- **Short-term reversal carries the result.** On its own it nearly matches all 16 features at 5 days (Ridge 0.0112,
  LightGBM 0.0142). At 21 days momentum on its own is negative and drags the full models down. LightGBM relies on
  `ret_5d` (negative) and `mom_12_1` (positive), consistent with the priors and the Fama-MacBeth results.
- **Excluding delisting exits** leaves the best model's IC unchanged.
- All figures are gross. Transaction costs and portfolio construction are not modelled yet.

## Data attribution

Data from Sharadar: Core US Equities Bundle (SEP prices, S&P 500 constituents,corporate actions). 
Licensed data is not redistributed; `data/raw` and `data/processed` are gitignored.

## Tests

`pytest` runs offline (HTTP is mocked, features use synthetic prices). The key test checks that
every feature at date `t` computed from data truncated at `t` equals the value computed from the
full dataset (including z-scores and the optional liquidity filter). Others cover labels and lag, per-date
standardisation, point-in-time membership, hand-computed returns / momentum / beta, and missing
data handling.

The modelling tests check that every walk-forward fold is purged (no training label reaches the test year, no inner
training label reaches validation), that Ridge and Elastic Net match scikit-learn, that the Fama-MacBeth forecast
averages per-date slopes, and that LightGBM is deterministic. A runner smoke test on a synthetic panel recovers a
planted reversal signal. The diagnostics tests cover the label-compounding check, the exact IC decomposition, the
beta-hedged and robust spreads, and a hand-computed break-even example.
