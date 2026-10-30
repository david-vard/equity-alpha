# Holdout

Design from `configs/holdout.yaml`, committed (acbf5a9) before any row on or after `holdout_start` was read. The runner was
committed (6f731f9) and tested on synthetic data only. The holdout was opened once, at 14:45 UTC on 2026-10-30, with
`code_dirty: false`. The run took 1 hour 51 minutes. Notebook: `notebooks/08_holdout.ipynb`.

Window: 2022-01-03 to 2026-10-23, 1,207 trading days, about 4.8 years. Base costs, $100m AUM.

**The holdout doesn't confirm the headline, and it doesn't refute it.** Both primary outcomes are null. The headline is
`inconclusive` at 0.21 net (t 0.48), and LightGBM wasn't shown to beat B0. The candidate is also `inconclusive`. Nearly all of the
LightGBM return arrives in 2026.

## 1. Declared outcomes

| Outcome | Role | Result | 95% interval | Label |
|---|---|---:|---:|---|
| Headline: LightGBM `optimised`, net Sharpe | primary | **0.21 (t 0.48)** | -0.58 to 1.01 | `inconclusive` |
| Headline minus B0, net Sharpe | primary | **+0.83** | -0.33 to 2.02 | `no_difference_shown` |
| Candidate: LightGBM `optimised`, top 25%, net Sharpe | secondary | 0.25 (t 0.53) | -0.63 to 1.06 | `inconclusive` |

B0 is the comparator and has no outcome of its own: -0.61 net (t -1.41), interval -1.43 to 0.17.

`inconclusive` doesn't show the books have no edge. With about 4.8 years, a t of 2 needs a net Sharpe near 0.9, so this sample
can't separate zero from a modest edge. The headline's interval contains both zero and its tuning figure of 0.35. For the
difference, 92% of resamples were above zero, but the declared test is whether the interval excludes zero, and it doesn't.

## 2. Checks (`checks.csv`)

All 5 passed. The refitted 2021 model matches the recorded predictions, with a largest gap of 2e-9 (B0: 0). Each book reproduces its
recorded net Sharpe to the last research date, within 3e-17. `assert_no_leakage` ran on every fold inside `walk_forward_folds`. It
stops the run on failure, so it has no row in the file.

## 3. The books (`books.csv`)

| | Headline | B0 | Candidate |
|---|---:|---:|---:|
| Net Sharpe | 0.21 | -0.61 | 0.25 |
| Gross Sharpe | 0.34 | -0.30 | 0.31 |
| Net return a year | 36 bp | -463 bp | 15 bp |
| Net vol | 1.7% | 7.6% | 0.6% |
| Trading costs a year | 7 bp | 209 bp | 2 bp |
| Borrow a year | 14 bp | 25 bp | 2 bp |
| Turnover | 0.85× | 13.6× | 0.28× |
| Break-even cost multiplier | 2.7 | negative | 4.6 |
| Break-even impact Y | 12.6 | negative | 20.8 |
| Break-even borrow, bp a year | 89 | negative | 213 |
| Max drawdown | -3.6% | -33.9% | -1.0% |
| Mean gross exposure | 1.13 | 1.97 | 0.16 |
| Realised beta (t) | -0.00 (-0.1) | 0.13 (6.6) | 0.00 (0.4) |

Across the three windows, net Sharpe runs:

| Book | Tuning | Evaluation | Holdout |
|---|---:|---:|---:|
| Headline | 0.35 | -0.00 | 0.21 |
| B0 | 0.45 | -0.03 | -0.61 |
| Candidate | 0.62 | 0.38 (second look) | 0.25 |

- **Headline.** It covers its costs in this window, and the break-evens are comfortable. But it's a small return on a low-exposure
  book. Mean gross fell to 1.13, from 1.70 in the evaluation window.
- **B0.** It loses before costs, so no cost assumption rescues it. Its realised beta of 0.13 is close to the earlier windows (0.10
  and 0.13), so its ex-ante neutrality doesn't hold in practice. That beta isn't what sank it: the universe rose in 4 of the 5 years.
- **Candidate.** It's positive, but it's the smallest book by far, at 15 bp a year on 0.16 of gross. It can't replace the headline
  under the design, and these numbers give no reason to wish it could.

## 4. Year by year (`by_year.csv`)

| Year | Headline net Sharpe | Net return | B0 net Sharpe | Net return | Candidate net Sharpe | Net return |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | -0.00 | -0.0% | -0.12 | -1.0% | -0.58 | -0.3% |
| 2023 | -0.06 | -0.1% | -1.05 | -8.4% | -0.72 | -0.3% |
| 2024 | 1.55 | +2.4% | -1.27 | -6.6% | 1.33 | +0.6% |
| 2025 | -1.54 | -2.3% | 0.15 | +0.7% | -1.02 | -0.5% |
| 2026, to 23 Oct | 0.81 | +1.6% | -0.81 | -6.9% | 1.44 | +1.2% |

**The positive LightGBM results rest on 2026.** This is descriptive. It wasn't declared, and it reads only the recorded daily
series (`notebooks/08_holdout.ipynb`):

| Book | Total net return | To end of 2025 | 2026 | Best month | Sharpe without it |
|---|---:|---:|---:|---|---:|
| Headline | +1.74% | +0.10% | +1.64% | July 2026, +1.5% | 0.03 |
| Candidate | +0.70% | -0.50% | +1.20% | June 2026, +0.7% | 0.00 |

Dropping a book's best month after the fact is a harsh cut, and it would lower any book's Sharpe. Still, the headline's first 48
months, of 58, add up to +0.10%. Its +1.5% in July 2026 was followed by -1.3% in August. B0's worst stretch came early: -11.8%
over the 42 days to 2023-03-23.

## 5. The model (`model_ic.csv`, `refit_hyperparameters.csv`)

The score kept its ranking power. Its daily rank IC is **0.018 (t 2.25)** over 1,201 days, close to the 0.017 (t 3.50) of 2007-2021.
This is a secondary statistic. So the forecast didn't vanish on new data, but the book still can't turn it into a return this sample
can distinguish from zero. That's the same gap the first portfolio results showed.

The five January refits chose by the unchanged rule. Validation ICs range from 0.010 to 0.023, and the 2022 fold stopped after 5
boosting rounds, which is a very thin model for that year.

## Limitations

- **One opening.** There's no second sample. The data can't be reused for confirmation, and these numbers now count as seen.
- **Low power.** Only an edge near 0.9 net would have reached t 2 here. The tuning figure of 0.35 wouldn't have.
- **One path, one start.** The books run continuously from 2007, so their state on 2022-01-03 depends on that start. The bootstrap
  treats the sample as given.
- **Costs are the same assumptions as before.** Trades are small at this turnover, which is where the square-root impact model is
  least tested. Borrow is a flat 25 bp.
- **The end of the window is run-off.** Open tranches liquidate in the last five days, as declared, and those days are in the statistics.
- **No book-level attribution.** The run recorded totals only, so nothing here says which names made July 2026.

## Run notes

- **Wording of the checks.** The config says checks run "before any holdout return is computed". A continuous run simulates holdout
  days in memory before it can compare the pre-holdout rows. The checks read only pre-holdout data, and no holdout statistic was
  computed or written until they passed. The README states it this way.
- **One optimiser fallback.** The log shows one `hold`, and the recorded headline book has exactly one pre-holdout `hold`, so no
  holdout rebalance fell back. The run didn't save fallback dates, so this rests on that count.
- **No bug, no rerun.** No code or config changed after the lock was written.

## Not pre-registered (listed, not run)

- **Name-level attribution of 2026,** especially July. It would need a further run with weights recorded, so a committed amendment.
- **Factor attribution of the holdout series,** on the factors in `configs/attribution.yaml`, extended past 2021.
- **LightGBM band 4 on the holdout.** It was the most interesting second-look result and wasn't a declared book. Any run now would
  be a look at a spent window, not a test.
- **B0's IC over the holdout.**
- **A Sharpe pooled across the evaluation window and the holdout.**
- **The deflated Sharpe ratio** across everything tried.
