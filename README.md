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

## Portfolio construction and costs: design

Do the out-of-sample forecasts survive realistic portfolio construction and transaction costs? The design is fixed in
[`configs/portfolio.yaml`](configs/portfolio.yaml) and was committed before any portfolio return was computed.
`src.portfolio.load_portfolio_config` rejects unknown or missing keys, wrong types, methods the design does not define,
signal columns the predictions file lacks, and horizons or a lag that disagree with `configs/features.yaml`,
`configs/research.yaml` and `configs/models.yaml`. The future runner will use the same guard as the other runners
(`src.utils.provenance`): it does not start unless the config is committed and unmodified.

### Why turnover is the central question

- **The forecast is weak and not better than reversal.** LightGBM on the 5-day rank target is the only Holm-significant
  predictor (IC 0.0174, NW t 3.50), but its edge over B0 reversal (IC 0.0123) is +0.005 with t 1.0, and none of the 36
  paired comparisons is significant.
- **The gross spreads are small.** Beta-hedged D10 - D1 per 5 days: LightGBM 13.2 bp (t 1.4), B0 22.3 bp (t 1.9).
- **Turnover is high.** One-way leg turnover per rebalance: LightGBM 0.54 (D1) / 0.75 (D10), B0 0.88 / 0.86.
- **So break-even is low.** Spread / (2 x both legs' turnover) gives about 5 bp per side for LightGBM and 6 bp for
  B0, ignoring the hedge's trading. The diagnostics found about 4 bp per side for 5-day reversal since mid-2010.
  Commission plus half-spread alone (2.5 bp) uses about half of that before any market impact. A construction
  that trades less is therefore worth more than a better forecast.

### Design

- **Signals** (columns of `data/processed/oof_predictions.parquet`):
  - Primary: `lightgbm_h5_rank`.
  - Benchmarks, with identical construction and costs: `B0_reversal_h5` and `B1_composite_h5`.
  - Low-turnover variant: `lightgbm_h21_rank` (IC 0.006, t 0.93, so not significant). It tests whether slower
    trading pays for a weaker signal.
- **Sample and tuning.**
  - OOS predictions run from 2007-01-03 to the last research date (2021-11-30).
  - Portfolio hyperparameters (λ_risk, λ_tc) are chosen on 2007-2013 only.
  - The evaluation window runs from 2014-01-02 to the research end and is reported as the clean result.
  - The 2022+ holdout is untouched. `oof_predictions.parquet` ends at the research end, so evaluating the holdout
    later will need a model refit.
  - The return series starts once every tranche holds positions.
- **Timing, as in the labels.**
  - The score is known at the close of `t`; the trade happens at the close of `t + 1`.
  - Daily P&L uses `ret_1d` (`closeadj`), with weights drifting between rebalances.
  - The panel only has rows for index members, so a held name that leaves the index (or stops trading) earns 0 on
    the days it has no row. The rest of its return, from its `fwd_ret_{h}d` label, is booked on the tranche's exit day.
    Each tranche's h-day return therefore equals the labels exactly, delisting exits at the last price included,
    using only `load_research_panel`.
  - A held name with no score at its tranche's next rebalance is sold.
- **Rebalancing.** h staggered tranches (5 for h=5, 21 for h=21), each with 1/h of capital and its own offset, so the
  result does not depend on the offset (the diagnostics found h=21 spreads moving a lot with the offset). Exactly one
  tranche trades per day.
- **Construction ladder.** Every rung is dollar-neutral, with $1 long and $1 short per $1 of capital.
  - `decile_ls`: equal weight, long the top decile and short the bottom decile. It is not beta-hedged, so its gross
    returns are not directly comparable to the beta-hedged spreads above.
  - `neutral_volscaled`:
    - Weights are proportional to z(score) / `vol_63d`, with each leg set by the sign of the score.
    - The weights are then projected onto {Σw = 0, β'w = 0} with `beta_252d`. Scaling the two legs separately cannot
      make dollar and beta exposure zero together unless both legs have the same beta.
    - Finally the book is scaled to a 10% annualised ex-ante vol, with gross capped at 2.
  - `optimised`: maximise α'w − λ_risk·h·w'Σw − λ_tc·cost(trades).
    - Constraints: dollar-neutral, |β| ≤ 0.05, |w_i| ≤ 2%, gross ≤ 2, and each trade ≤ 5% of `dollar_volume_21d`.
    - Alpha is Grinold's IC·σ_i·z_i. The IC is the mean daily IC on each fold's purged two-year validation block,
      which is known before the test year, floored at 0. A single IC per fold only rescales alpha against the λ
      grid, so only its change from year to year matters.
    - Σ is Ledoit-Wolf on 252 trailing daily returns. A name needs 80% of that window, so recent index joiners cannot
      be held that day.
    - (λ_risk, λ_tc) come from a 4 x 4 grid, chosen per signal by net Sharpe on 2007-2013 and kept fixed in every
      cost and AUM scenario.
    - Solved with cvxpy / Clarabel, because the square-root impact term makes trading cost scale with trade size to
      the power 1.5.
  - Sector neutrality is declared as a robustness variant but switched off. The panel has no sector field, and
    Sharadar's sector is today's classification, not point in time.
- **Costs, per side, as a share of traded notional.**
  - Commission 0.5 bp and half-spread 2 bp (there is no quote data).
  - Square-root impact Y·σ_daily·sqrt(trade $ / ADV $), with Y = 1, σ_daily = `vol_63d` and
    ADV = `dollar_volume_21d`.
  - Borrow of 25 bp a year on short notional. Cash earns 0, because returns are excess returns.
  - Base AUM is a constant nominal $100m. Member dollar volume grew about 10x over the sample, so early years are
    relatively harder to trade.
  - Scenarios: cost multipliers {0, 0.5, 1, 2, 3} and AUM {$10m, $100m, $1bn}. `optimised` is re-run with the
    scaled costs in its objective.
- **Statistics.** Daily net and gross returns per $ of capital. Sharpe is annualised by sqrt(252). NW t uses
  `max(h, plug-in)` lags because the tranches overlap. Turnover is one-way per year.

### Headline result (declared before any portfolio return exists)

The annualised net Sharpe and NW t of `optimised` on `lightgbm_h5_rank`, at base costs and $100m, over 2014-01-02 to
the research end. Reported alongside it:

- the same for `B0_reversal_h5` under identical settings;
- the gross Sharpe;
- annual turnover;
- the break-even cost multiplier: the multiplier at which the mean net return is zero, computed after the fact at
  the base-case trades.

A null or negative result is reported as such.

**Outputs.** The runner writes tidy CSVs, `figures/` and `run_metadata.json` to `reports/portfolio/`, and daily
returns and weights to `data/processed/` (gitignored).

### Running the portfolio backtest

Run the two modes in order:

```bash
python -m src.run_portfolio --tune       # (lambda_risk, lambda_tc) per signal on 2007-2013 -> reports/portfolio/tuning/
python -m src.run_portfolio --evaluate   # every signal x construction x cost scenario, frozen lambdas -> reports/portfolio/
```

- **`--tune`** reads no panel row, prediction or label later than the last trading day before the split date. It
  writes:
  - `lambda_grid.csv` (every grid point, with its fallback counts);
  - `selected_hyperparameters.csv`;
  - its own `run_metadata.json`.
- **`--evaluate`** refuses to start if there is no tuning output, or if the tuning output was made with another
  version of `configs/portfolio.yaml`. It runs the whole sample and writes:
  - `summary.csv`: statistics per signal, construction, cost scenario and window (tuning, evaluation, full);
  - `headline.csv`: the pre-declared headline rows;
  - `fallbacks.csv`: optimiser fallbacks per run;
  - `figures/cumulative_net_<signal>.png`;
  - `run_metadata.json`;
  - `data/processed/portfolio_daily_returns.parquet`: per-tranche and total daily accounting;
  - `data/processed/portfolio_weights.parquet`: target weights, trades and their costs per rebalance.
- **Both modes** refuse to start unless `configs/portfolio.yaml`, `configs/research.yaml` and the feature config are
  committed and unmodified (`--allow-dirty-config` overrides). `--signals` and `--constructions` run a subset.
- **Runtime.** `optimised` solves one cvxpy / Clarabel problem per rebalance. In practice a 2007-2013 tuning run of
  one grid point takes about 17 minutes, and the evaluate run of one signal takes 4-6 hours. The recorded results
  were run one process per signal (`--signals S --out reports/portfolio/runs/S --data-out data/processed/portfolio/S`)
  and merged with `python -m src.portfolio_report --tables reports/portfolio/runs/* --data data/processed/portfolio/*`.

The engine ([`src/backtest.py`](src/backtest.py)) uses the constructions in [`src/portfolio.py`](src/portfolio.py)
and the cost model in [`src/costs.py`](src/costs.py). The optimiser's cost term and the accounting use the same cost
coefficients. If the solver fails, returns an infeasible problem or gives a solution that breaks a constraint, the
tranche falls back to one of the following, and every fallback is logged and counted:
1. its current weights, neutralised and scaled into the caps, if the ADV cap allows that trade;
2. otherwise flat;
3. otherwise each name moved towards zero as far as the ADV cap allows, flagged as breaking the constraints.

### Results (evaluation window 2014-01-02 to 2021-12-08, frozen λs)

**No net edge after costs.** The engine reconciles with the h=5 decile spread from the models report: mean difference
-7e-6 per period, explained entirely by 6 periods with missing leg labels.

The headline, `optimised` on `lightgbm_h5_rank` at $100m and base costs, has a net Sharpe of -0.00 (NW t -0.00) against
-0.03 (t -0.07) for B0. It trades 1.2× a year, has a gross Sharpe of 0.14 and a break-even cost multiplier of 0.99.
Every tuned λ landed on the edge of the grid.

`neutral_volscaled` has the best gross Sharpe (0.80 for `lightgbm_h5_rank`), but at 54× turnover, mostly impact, it
loses net. LightGBM does not beat B0 after costs. None of the 12 signal × construction books has a net t above 1.

Full write-up, with cost breakdown, AUM scenarios, exposures, limitations and a not-pre-registered list:
[`reports/portfolio/SUMMARY.md`](reports/portfolio/SUMMARY.md). Notebook:
[`notebooks/04_portfolio.ipynb`](notebooks/04_portfolio.ipynb).

## Risk and attribution: design

What is the gross return of the portfolio books made of? In particular, is the 0.80 gross Sharpe of `lightgbm_h5_rank`
under `neutral_volscaled` just the published short-term reversal factor? The design is fixed in
[`configs/attribution.yaml`](configs/attribution.yaml) and is committed before any factor file is downloaded or any
regression is run. `src.attribution.load_attribution_config` rejects unknown or missing keys, wrong types, methods the
design does not define, books the portfolio config does not define, and a confirmatory test that differs from the one
below. The runner will use the same guard as the other runners (`src.utils.provenance`).

This design was written after the results in `reports/portfolio/` were seen. It describes the recorded base-case
books. No backtest is rerun, no book is chosen, and nothing in `configs/portfolio.yaml` changes.

### Books, series and windows

- **Books:** `optimised` and `neutral_volscaled`, each on `lightgbm_h5_rank` and `B0_reversal_h5`, at $100m and base
  costs. The first is the headline book and the third its benchmark.
- **Series:** gross P&L is primary. Net P&L is reported but labelled as not like for like, because the published factors
  are gross of their own trading costs.
- **Windows:** the evaluation window (2014-01-02 to 2021-12-08) is primary. The tuning window (2007-2013) is secondary
  and keeps its in-sample label.

### Factors

- **Models:** nested OLS regressions of daily book returns on a constant and the factors: market, then Fama-French 5,
  then momentum, then short-term reversal, then betting-against-beta. Books are self-financing and earn no cash, so no
  risk-free rate is subtracted.
- **Inference:** Newey-West with the research lag rule, `max(h, plug-in)`.
- **Decomposition:** mean return = alpha + the sum of beta × mean factor return, reported in bp a year per factor.
- **Sources:** daily files from the Ken French data library, and the AQR daily BAB file (US column). The URLs are in the
  config, and each file's sha256 and download date go into `run_metadata.json`. The files stay in `data/raw/factors/`,
  which is gitignored.
- **Holdout rule:** the published files run past 2022. The parser drops every row on or after `holdout_start` before
  it returns any value, and a later date is an error. A book date with no factor value is also an error, never dropped
  or filled.
- **Own reversal book:** the published factor is a one-month, all-cap, value-weighted sort, so it can miss a five-day
  reversal inside the S&P 500. One further regression adds the gross return of B0 `neutral_volscaled` to the largest
  model for the LightGBM `neutral_volscaled` book.

### The one confirmatory test

Book LightGBM `neutral_volscaled`, gross, evaluation window, largest model. Reversal counts as present when the
`st_rev` beta is positive with NW t ≥ 2.

| Reversal | Alpha NW t | Outcome | Reading |
|---|---|---|---|
| present | < 2 | `reversal_explained` | The gross return is the published reversal premium. |
| present | ≥ 2 | `reversal_partial` | The reversal factor explains part of it, and alpha remains. |
| absent | ≥ 2 | `not_reversal` | The alpha is not explained by these factors. |
| absent | < 2 | `inconclusive` | Neither is distinguishable from zero in this window. |

Everything else in this design is descriptive and gets no multiple-testing correction. The window is about 8 years,
so the test has limited power, and `inconclusive` is a real possibility.

### Risk statistics

For every book, gross and net, per window: the three deepest drawdowns on additive P&L (fixed capital, as in the
backtest), historical VaR and ES at 95% and 99% over 1 and 5 days, skew and excess kurtosis on daily and on
non-overlapping h-day returns, and the worst day and month. VaR and ES are shown per $ of capital and as multiples of
daily vol, because book vols differ. All of these are point estimates.

### Why realised beta differs from ex-ante

`beta_ex_ante` is the holdings after drift times `beta_252d`, so drift between rebalances is already inside it. The
design reports each explanation and doesn't pick a winner in advance:

- **Benchmark:** `beta_252d` is measured on the equal-weighted universe. Each book is regressed on that return and on
  Mkt-RF, which stands in for SPY because SPY is not in the data bundle.
- **Estimation error:** does the realised beta's 95% Newey-West interval contain the mean ex-ante beta?
- **Stale beta:** exposure recomputed from rebalance weights with a 63-day beta known at the score date.
- **Conditional beta:** `r_t = a + b0 m_t + b1 X_(t-1) m_t`, for the ex-ante exposure and the 63-day exposure. An unbiased
  exposure measure gives `b0 = 0` and `b1 = 1`.
- **Asymmetry:** separate betas on up and down market days.
- **Factor controls:** the market beta in each nested model.

### Checks and outputs

The run stops before writing results if any of these fails:
- the net regression on the universe return must reproduce `reports/portfolio/exposures.csv`;
- the correlation of Mkt-RF with the universe return must peak at lag 0 across lags -1, 0 and +1, which catches a date
  shift in a factor file;
- the mean decomposition must hold exactly;
- every book date must have a value for every factor.

The results will go to `reports/attribution/`, with a summary there and `notebooks/05_attribution.ipynb`. If a factor
file's layout differs from the config, for example the AQR sheet or column name, the correction is committed before any
regression runs and recorded in the summary. A source that can't be obtained stops the run: the largest model isn't
replaced with another one.

### Running the attribution

```bash
python -m src.run_attribution --download   # factor files -> data/raw/factors/, with a manifest of hashes
python -m src.run_attribution --run        # -> reports/attribution/
```

`--download` keeps a file that already exists, so one vintage backs every run. `--run` never touches the network,
refuses a factor file that doesn't match the manifest, and writes only `checks.csv` if a check fails. It reads the
recorded books from `data/processed/portfolio/` and reruns no backtest.

### Results (evaluation window, gross, largest model)

**The 0.80 gross Sharpe of LightGBM `neutral_volscaled` is not simply the published reversal factor.** The pre-registered
rule returns `reversal_explained`: the `st_rev` beta has t 10.4 and alpha has t 1.89. That label is stronger than the numbers,
because the factor accounts for 110 of the 592 bp a year and alpha for 355 bp. The outcome also sits close to the
threshold, with alpha t between 1.57 and 2.06 across the nested models.

Adding the project's own B0 book removes the `st_rev` loading (t 0.08). B0 takes 195 bp, about a third of the mean, and
alpha falls to 285 bp (t 1.65). The B0 books are mostly the reversal factor. Momentum is LightGBM's largest risk (beta
0.28, t 15.8) but earns little. Realised betas aren't noise, a benchmark effect or a stale-beta effect. The strongest
evidence is that they shrink once `st_rev` is in the model.

Full write-up: [`reports/attribution/SUMMARY.md`](reports/attribution/SUMMARY.md). Notebook:
[`notebooks/05_attribution.ipynb`](notebooks/05_attribution.ipynb).

## Sensitivity: design

How much of the portfolio result is sampling noise, and how far does it depend on the two assumed cost inputs and on the
choice of rebalance offset? The design is fixed in [`configs/sensitivity.yaml`](configs/sensitivity.yaml) and is committed
before any of these statistics is computed. `src.sensitivity.load_sensitivity_config` rejects unknown or missing keys,
wrong types, methods the design does not define, a grid without its base point, and primary intervals that differ from the
ones below. The runner will use the same guard as the other runners (`src.utils.provenance`).

This design was written after the results in `reports/portfolio/` and `reports/attribution/` were seen. It re-reads the
recorded base-case books for all 12 signal × construction pairs. No backtest is rerun, the optimiser is not re-solved, no
book is chosen, and nothing in `configs/portfolio.yaml` changes. The evaluation window is primary. The tuning window is
secondary and keeps its in-sample label.

### Bootstrap

- **Method:** stationary bootstrap of daily returns with 10,000 resamples and a fixed seed, giving 95% percentile intervals
  for the annualised Sharpe, gross and net.
- **Block length:** a mean of 21 trading days is primary, which reaches the longest horizon so overlapping tranches stay
  together. Means of 42 and 63 days show how much that choice matters.
- **Paired differences:** LightGBM minus B0 in each construction, with both books using the same resampled days, so their
  shared risk stays in the comparison.

Two intervals are declared in advance, both net and on the evaluation window: the headline Sharpe, and LightGBM minus B0
under `optimised`. Everything else is descriptive and gets no multiple-testing correction. An interval that contains zero
says the data can't separate the number from zero. It doesn't show the number is zero.

### Y and borrow repricing

`net = gross - commission - spread - impact × (Y / 1) - borrow × (rate / 25 bp)`, day by day, over a full cross of Y in
{0.25, 0.5, 1, 1.5, 2} and borrow in {0, 25, 50, 100, 200} bp a year. Impact and borrow are linear in these inputs, so this
is exact for the recorded trades. Each book also gets a break-even for each input, with the other held at base. A negative
break-even means the book loses money even with that cost at zero.

**The trades are held fixed.** The optimiser saw base costs, and it would trade differently under other costs. So these
numbers answer "what would this book have cost?", not "what would the best book have been?". They differ on purpose from
`reports/portfolio/scenarios.csv`, which re-solves.

### Offsets

Each tranche is one rebalance offset, so the recorded tranche rows already hold each offset's book. Returns are scaled to the
tranche's own capital, which leaves Sharpe unchanged. Each book reports Sharpe, return and turnover per offset, gross and net,
with the minimum, maximum, range and standard deviation across offsets next to the total. This is descriptive, with no test
declared, because offsets share most of their risk.

### Checks and outputs

The run stops before writing results if any of these fails:
- point Sharpes must reproduce `reports/portfolio/summary.csv`;
- the base point of the grid must reproduce the recorded net series, day by day;
- tranche rows must sum to the total row, day by day;
- both books of a paired difference must have identical dates;
- the last date of every book must fall before `holdout_start`.

The results will go to `reports/sensitivity/`, with a summary there and `notebooks/06_sensitivity.ipynb`.

### Running the sensitivity analysis

```bash
python -m src.run_sensitivity   # -> reports/sensitivity/
```

The checks run first, and if one fails only `checks.csv` is written. The run reads the recorded books from
`data/processed/portfolio/`, reruns no backtest and doesn't re-solve the optimiser.

### Results (evaluation window, base costs, mean block 21 days)

**The null headline holds, but it's a weak null.** The headline net Sharpe is -0.00 with a 95% interval of -0.60 to +0.62, and
LightGBM minus B0 under `optimised` is +0.02 (-0.96 to +1.07). The data rule out a large net edge but can't separate zero from
a net Sharpe of about 0.5.

Of the 24 gross and net intervals, three exclude zero: gross LightGBM `neutral_volscaled` (0.20 to 1.42), and net B0 `decile_ls`
and `neutral_volscaled`, which are negative. With trades held fixed, no book reaches a net t of 2 anywhere on the Y and borrow
grid. The headline depends on borrow (break-even 24.6 bp, against 25 bp assumed), and LightGBM `neutral_volscaled` depends on
impact (break-even Y 0.61). Reversal depends most on the rebalance offset: B0 `optimised` runs from -0.27 to +0.21 net.

Full write-up: [`reports/sensitivity/SUMMARY.md`](reports/sensitivity/SUMMARY.md). Notebook:
[`notebooks/06_sensitivity.ipynb`](notebooks/06_sensitivity.ipynb).

## Second round: design

Do three declared ways of trading less, or trading only liquid names, change the portfolio result? The design is fixed in
[`configs/second_round.yaml`](configs/second_round.yaml) and is committed before any variant is simulated.
`src.second_round.load_second_round_config` rejects unknown or missing keys, wrong types, methods the design does not
define, grids without their recorded point, and lambda points that don't extend outward from the recorded choice. The
runner will use the same guard as the other runners (`src.utils.provenance`).

**Every evaluation-window number in this round is a second look.** The ideas came from seeing the evaluation window, so
those numbers are weaker evidence than the pre-registered headline and can't replace it. Every such row carries the label
`second_look`, and no confirmatory test is declared. Holdout rules, lag, costs and $100m AUM are unchanged.
`configs/portfolio.yaml` and the engine do not change.

### Variants

All use `lightgbm_h5_rank` and `B0_reversal_h5`. Each grid includes the recorded book, so keeping it is an allowed outcome.

- **No-trade band, `neutral_volscaled`.** Bands are 0, 0.5, 1, 2 and 4 times the mean |target weight| at each rebalance.
  - A name moves only if its gap to target exceeds the band, and then only to the band's edge.
  - A name without a score is sold in full, as in the recorded book.
  - Zero dollars and beta are restored by moving only the traded names. With fewer than 3 traded names, the tranche holds.
  - The book is scaled down if gross or ex-ante vol exceed the limits, and never scaled up, which would trade every name.
    So realised vol can fall below the 10% target. It is reported.
- **Liquid universe, `neutral_volscaled` and `optimised`.** Each date keeps the top 25%, 50% or 100% of scored names by
  `dollar_volume_21d` at the score date, and scores are standardised after that cut.
  - Lambdas stay at their recorded values, and the Grinold IC isn't re-estimated on the smaller universe.
  - `min_names_per_date` stays at 100. At 25% of about 500 names that margin is thin, and a date below it holds no position
    from that rebalance. Those dates are counted and reported.
- **Wider lambda grid, `optimised`.** New points only extend past the grid edge that the recorded choice reached:
  - LightGBM, from (100, 4): (100, 8), (100, 16), (300, 4), (300, 8), (300, 16).
  - B0, from (3, 1): (1, 0.5), (1, 1), (1, 2).

### Choosing on 2007-2013 only

Each variant, signal and construction gets one choice: the highest net Sharpe on the tuning window, at base costs and AUM,
with ties going to the first in config order. Nothing after the last tuning date is read. A choice on a grid edge is flagged,
and no grid is extended again.

**A holdout candidate is fixed before evaluation runs.** It's the chosen LightGBM variant with the highest tuning net Sharpe,
and only if that exceeds the headline's tuning figure of 0.35. Otherwise there is no candidate. This round can send at most
one book to the holdout.

### Evaluation

Each chosen variant runs once, at base costs and AUM. A chosen recorded point isn't rerun. Each row reports net Sharpe, NW t, net
return, gross Sharpe, turnover and the break-even cost multiplier, with a stationary-bootstrap interval using the settings of
`configs/sensitivity.yaml`, and a paired interval for the variant minus its recorded book.

### Checks and outputs

The run stops before writing results if any of these fails:
- band 0 and fraction 1 of `neutral_volscaled` must reproduce the recorded daily net series;
- fraction 1 of `optimised` must reproduce the recorded tuning net Sharpe, within a wider tolerance because the solver reruns;
- a tuning run must not read past the last tuning date, and no date may reach `holdout_start`;
- evaluation refuses a tuning output made with a different config.

The results will go to `reports/second_round/`, with a summary there and `notebooks/07_second_round.ipynb`.

### Running the second round

Run these in order. `--shard i/n` runs every n-th variant from i, so a run can be split across n processes:

```bash
python -m src.run_second_round --tune --shard 0/4       # variants on 2007-2013 only
python -m src.run_second_round --select                 # checks, choices and the holdout candidate
python -m src.run_second_round --evaluate --shard 0/2   # the chosen variants, once
python -m src.run_second_round --report                 # -> reports/second_round/evaluation.csv
```

`--select` writes only `checks.csv` if a check fails. It writes the holdout candidate before any evaluation-window return of a
variant exists. `--evaluate` refuses a selection made with a different config.

### Results (second look: evaluation window, base costs)

**The headline doesn't change, and every number here is weaker evidence than it.** The holdout candidate, chosen on 2007-2013, is
LightGBM `optimised` on the top 25% of names. It reaches 0.38 net (t 1.09, interval -0.29 to 1.07) on a very small book: mean gross
0.25 and about 30 bp a year.

LightGBM `neutral_volscaled` with band 4 reaches 0.71 net (t 2.04, interval 0.09 to 1.35) and is positive in all 8 years. It's the
most interesting result and the least understood. It trades about 1.5× a year, so it's no longer a five-day book, its holdings
depend on its 2007 start, and the same rule fails on B0. It isn't the candidate, because its tuning net Sharpe of 0.32 was below the
headline's 0.35. The liquid universe didn't hold up out of sample, and the wider lambda grid added nothing.

Full write-up: [`reports/second_round/SUMMARY.md`](reports/second_round/SUMMARY.md). Notebook:
[`notebooks/07_second_round.ipynb`](notebooks/07_second_round.ipynb).

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
beta-hedged and robust spreads, and a hand-computed break-even example. The portfolio-config tests load the real
`configs/portfolio.yaml`, check that each validation rule rejects a bad config, and check in a temporary git repository
that the provenance guard blocks an uncommitted or modified config.

The cost tests check a hand-computed trade (commission, spread, impact and borrow), the participation cap, and that
cost scenarios scale linearly. The construction tests check decile selection with ties and NaN, the ex-ante vol target,
the optimiser constraints and each fallback path. The backtest tests run on small synthetic panels and check:
- the accounting identities, and that tranches sum to the total;
- exact drift between rebalances;
- terminal exits at the last price, charged once;
- no lookahead: inputs after `t` cannot change anything up to `t`;
- that zero-cost `decile_ls` reproduces the research D10 - D1 spread;
- the holdout guard;
- that `--tune` gives identical results when everything after the tuning window is poisoned.

The attribution-config tests load the real `configs/attribution.yaml` and check that the loader rejects each way of
leaving the pre-registered design, including a change to the confirmatory test or to the holdout cut on factor files.

The sensitivity-config tests do the same for `configs/sensitivity.yaml`, including a change to a primary interval, a grid
without its base point, and a block length shorter than the longest horizon.

The sensitivity tests check that the stationary bootstrap is reproducible and has the declared block length, and that
resampled Sharpes match the project's Sharpe. They also check that repricing at the base point returns the recorded net
series, that break-evens set the mean net return to zero, that a book minus itself gives exactly zero, and that each spoiled
input fails its check with only `checks.csv` written.

The second-round config tests do the same for `configs/second_round.yaml`, including selection outside the tuning window, a
second holdout candidate, and a lambda point that doesn't extend past a grid edge the recorded choice reached.

The second-round tests check the band by hand: a name moves only to the band's edge, a name without a target is sold in full,
and zero dollars and beta are restored. Band 0 reproduces the recorded construction, and the band can't see the future. They
also run the whole round on a synthetic panel, including tuning on data poisoned after the tuning window.
