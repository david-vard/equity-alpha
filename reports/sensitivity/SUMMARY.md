# Sensitivity

Design from `configs/sensitivity.yaml`, committed before any of these statistics was computed. Produced by
`python -m src.run_sensitivity` at commit 62b3c1a with `code_dirty: false`. Notebook: `notebooks/06_sensitivity.ipynb`.

This re-reads the recorded base-case books ($100m, ×1 costs) for all 12 signal × construction pairs. No backtest was rerun,
the optimiser wasn't re-solved, and no book was chosen. The design was written after the portfolio and attribution results
were seen. Unless stated otherwise, numbers are for the evaluation window 2014-01-02 to the end of each book.

**The null headline holds, but it's a weak null.** The headline net Sharpe is -0.00 with a 95% interval of -0.60 to +0.62.
LightGBM doesn't beat B0 under `optimised` (+0.02, interval -0.96 to +1.07). Across the whole Y and borrow grid, no book reaches a
net t of 2. The data rule out a large net edge, but they can't separate zero from a net Sharpe of about 0.5.

## 1. Checks (`checks.csv`)

All 66 passed:
- Point Sharpes reproduce `reports/portfolio/summary.csv` to 1e-16.
- The base grid point reproduces the recorded net series, day by day, to 4e-17.
- Tranche rows sum to the total to 3e-17.
- Paired books have identical dates, and every date falls before `holdout_start`.

## 2. The two declared intervals (`primary_intervals.csv`)

Stationary bootstrap, 10,000 resamples, mean block 21 days, net, evaluation window, 1,999 days:

| Interval | Estimate | 95% interval | Resample sd | Share above zero |
|---|---:|---:|---:|---:|
| Headline net Sharpe | -0.00 | -0.60 to +0.62 | 0.31 | 49% |
| LightGBM minus B0, `optimised` | +0.02 | -0.96 to +1.07 | 0.52 | 52% |

Both contain zero. That doesn't show either number is zero. The sd of 0.31 is about what 8 years of data allows (1/√8 = 0.35),
so this is a limit of the sample, not of the method.

## 3. All 12 books (`bootstrap_intervals.csv`)

Sharpe with its 95% interval, mean block 21 days:

| Book | Gross | Net |
|---|---:|---:|
| LightGBM h=5 `decile_ls` | 0.20 (-0.43, 0.90) | -0.55 (-1.15, 0.11) |
| LightGBM h=5 `neutral_volscaled` | **0.80 (0.20, 1.42)** | -0.26 (-0.87, 0.34) |
| LightGBM h=5 `optimised` | 0.14 (-0.47, 0.76) | -0.00 (-0.60, 0.62) |
| B0 `decile_ls` | 0.24 (-0.55, 1.00) | **-1.10 (-2.05, -0.24)** |
| B0 `neutral_volscaled` | 0.51 (-0.20, 1.23) | **-1.13 (-1.93, -0.38)** |
| B0 `optimised` | 0.26 (-0.49, 0.95) | -0.03 (-0.79, 0.65) |
| B1 `decile_ls` | -0.04 (-0.67, 0.66) | -0.38 (-1.00, 0.29) |
| B1 `neutral_volscaled` | 0.68 (-0.05, 1.43) | 0.28 (-0.44, 1.03) |
| B1 `optimised` | -0.01 (-0.62, 0.64) | -0.06 (-0.68, 0.59) |
| LightGBM h=21 `decile_ls` | 0.01 (-0.59, 0.60) | -0.21 (-0.81, 0.38) |
| LightGBM h=21 `neutral_volscaled` | 0.11 (-0.42, 0.67) | -0.13 (-0.65, 0.42) |
| LightGBM h=21 `optimised` | 0.03 (-0.53, 0.61) | -0.06 (-0.62, 0.52) |

Three of the 24 intervals exclude zero (in bold). They are descriptive, with no multiple-testing correction.
- **Gross LightGBM `neutral_volscaled` is the only gross book that excludes zero.** This agrees with the attribution only
  in part: the raw Sharpe excludes zero, but alpha after factors has t 1.89.
- **No net interval excludes zero on the positive side.** The two that exclude zero are clearly negative.

**Block length barely matters.** Mean resample sd is 0.34 at 21 days, 0.32 at 42 and 0.31 at 63. It falls slightly as blocks get
longer, so the 21-day block wasn't missing positive serial dependence. The headline interval is -0.54 to +0.59 at 63 days.

## 4. LightGBM minus B0 (`paired_differences.csv`)

| Construction | Gross difference (95% interval) | Net difference (95% interval) |
|---|---:|---:|
| `decile_ls` | -0.04 (-0.75, 0.76) | +0.55 (-0.13, 1.42) |
| `neutral_volscaled` | +0.29 (-0.49, 1.08) | **+0.87 (0.15, 1.68)** |
| `optimised` | -0.13 (-1.10, 0.91) | +0.02 (-0.96, 1.07) |

The `neutral_volscaled` net difference excludes zero at all three block lengths, but it's not evidence of forecasting skill.
Both books lose money net, and the gross difference doesn't exclude zero. LightGBM loses less, in large part because it
trades 54× a year against B0's 72×. In the tuning window the same difference is +0.80 (0.05, 1.61).

## 5. Y and borrow, with trades held fixed (`repricing_grid.csv`, `repricing_breakeven.csv`)

Net Sharpe of the headline book, and of LightGBM `neutral_volscaled`:

| Y | Headline, 0 bp | Headline, 25 bp | Headline, 200 bp | `neutral_volscaled`, 0 bp | `neutral_volscaled`, 25 bp | `neutral_volscaled`, 200 bp |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.11 | 0.02 | -0.61 | 0.27 | 0.24 | 0.00 |
| 1 | 0.09 | -0.00 | -0.62 | -0.23 | -0.26 | -0.50 |
| 2 | 0.06 | -0.02 | -0.65 | -0.89 | -0.92 | -1.16 |

Break-evens, with the other input at its base value:

| Book | Break-even Y (base 1) | Break-even borrow, bp (base 25) |
|---|---:|---:|
| LightGBM h=5 `optimised` | 0.94 | 24.6 |
| LightGBM h=5 `neutral_volscaled` | 0.61 | -170 |
| B0 `optimised` | 0.86 | 6.4 |
| B0 `neutral_volscaled` | -0.02 | -798 |
| B1 `neutral_volscaled` | 2.30 | 298 |
| LightGBM h=21 `optimised` | -1.17 | -7.5 |

A negative break-even means the book loses money even with that cost at zero.

- **The headline is a bet on borrow, not impact.** Its Sharpe hardly reacts to Y. Each extra 25 bp of borrow costs it about 0.09
  of Sharpe. Its net Sharpe at 0 bp is 0.09, which follows directly from the known cost breakdown and isn't a finding.
- **LightGBM `neutral_volscaled` is the opposite.** Its result depends on Y. It turns positive if Y is below 0.61, which is
  plausible, because Y = 1 isn't calibrated. Even at Y = 0.25 and no borrow, its net Sharpe is 0.27.
- **B0 `neutral_volscaled` can't be saved by cheaper impact:** its break-even Y is about zero. Commission, spread and borrow
  alone absorb its gross return (384 bp a year against 366 bp).
- **Nothing reaches significance on the grid.** The largest net t in any cell is 1.30, for B1 `neutral_volscaled` at Y = 0.25 and
  0 bp borrow. At base costs, only the two B0 books above are significant, and both are negative.
- **B1 `neutral_volscaled` is the most robust point estimate:** positive in 22 of 25 cells. It's a benchmark, and this is an
  evaluation-window observation. Its tuning-window net Sharpe was -0.45, so this isn't a basis for picking it.

## 6. Offsets (`offsets.csv`, `offset_dispersion.csv`)

Net Sharpe across a book's rebalance offsets:

| Book | Total | Lowest | Highest | Range |
|---|---:|---:|---:|---:|
| LightGBM h=5 `optimised` | -0.00 | -0.09 | +0.09 | 0.18 |
| LightGBM h=5 `neutral_volscaled` | -0.26 | -0.34 | -0.17 | 0.17 |
| B0 `optimised` | -0.03 | -0.27 | +0.21 | 0.48 |
| B0 `decile_ls` | -1.10 | -1.14 | -0.63 | 0.51 |
| B1 `neutral_volscaled` | 0.28 | 0.24 | 0.33 | 0.09 |
| LightGBM h=21 `decile_ls` | -0.21 | -0.44 | +0.22 | 0.66 |

- **Reversal depends most on the offset among the h=5 signals.** B0's range is about 0.3-0.5, against 0.1-0.2 for LightGBM and B1.
  With a single offset, B0 `optimised` could look positive or negative, which is why the design uses staggered tranches.
- **LightGBM h=5 is stable,** and no single offset of the headline book stands out.
- **The h=21 ranges are wider partly because there are 21 offsets,** and each holds 1/21 of the capital. Their sd is 0.09-0.15.
- **Turnover hardly changes across offsets.** The differences above are in returns, not trading.
- **A total can sit near the edge of its offsets.** B0 `neutral_volscaled` is at -1.13 with offsets from -1.14 to -0.84. Averaging
  offsets cuts volatility, which scales the Sharpe further from zero in either direction.

Tranches exit on slightly different days at the end of the sample, so a few closing days have zero returns for some offsets.
The effect on 2,000-day statistics is negligible.

## Limitations

- **One 8-year path.** A bootstrap measures sampling noise within this path. It can't allow for a change in regime.
- **Repricing holds trades fixed.** The optimiser would trade differently under other costs, so the `optimised` rows show what
  this book would have cost, not what the best book would have earned. `reports/portfolio/scenarios.csv` is the re-solved view.
- **Percentile intervals are simple.** They aren't adjusted for bias or skew, though Sharpes near zero don't strain them much.
- **Y and borrow ranges are assumptions.** There is no fill or borrow data behind them. Borrow is flat across names, so a real
  hard-to-borrow short side could be worse than any point on this grid.
- **Half-spread wasn't varied.** It stays at the flat 2 bp.
- **Written after seeing results.** The design chose no book, but the headline's cost breakdown was known.

## Not pre-registered (listed, not run)

- **The deflated Sharpe ratio** across all books and λ grid points.
- **Half-spread sensitivity**, at the fixed trades.
- **Re-solving `optimised` under other Y and borrow values.** This takes hours and isn't a repricing.
- **A calendar test of the B0 offsets.** Five-day offsets line up loosely with weekdays, so a weekday effect could explain B0's
  offset range. This hasn't been tested.
- **A joint bootstrap of the Y and borrow break-evens.**
- **Intervals for factor alpha** from the attribution.
