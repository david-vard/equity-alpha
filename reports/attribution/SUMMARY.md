# Risk and attribution

Design from `configs/attribution.yaml`, committed before any factor file was downloaded. Produced by
`python -m src.run_attribution --run` at commit 02e0444 with `code_dirty: false`. Provenance, factor-file hashes and
download times are in `run_metadata.json`. Notebook: `notebooks/05_attribution.ipynb`.

This describes the recorded base-case books ($100m, ×1 costs). No backtest was rerun, no book was chosen, and
`configs/portfolio.yaml` did not change. The design was written after the portfolio results were seen. Unless stated
otherwise, numbers are gross, for the evaluation window 2014-01-02 to 2021-12-08, on the largest model (Fama-French 5 +
momentum + short-term reversal + BAB).

**The 0.80 gross Sharpe is not simply the published reversal factor.** The pre-registered rule returns
`reversal_explained`, but alpha t is 1.89, just under the threshold of 2, and the factor accounts for only 110 of the 592 bp a
year. Most of the rest is alpha that can't be told apart from zero. The project's own five-day reversal book explains
more than the published factor does. After costs, nothing remains.

## 1. Checks (`checks.csv`)

All passed:
- The net regression on the universe return reproduces `reports/portfolio/exposures.csv` for all 4 books and both
  windows, to 4e-15.
- Mkt-RF correlates 0.98 with the universe return at lag 0, against at most -0.11 at lags ±1, so the dates line up.
- The mean decomposition holds to 3e-19.
- Every book date has a value for every factor.

## 2. The confirmatory test (`primary_test.csv`)

LightGBM `neutral_volscaled`, gross, evaluation window, largest model, 1,999 days, 7 NW lags:

| | Value |
|---|---:|
| mean return | 592 bp a year |
| alpha | 355 bp a year, **t 1.89** |
| `st_rev` beta | 0.166, **t 10.4** |
| `st_rev` contribution | 110 bp a year |
| **Pre-registered outcome** | **`reversal_explained`** |

**How to read the label.** The rule asks whether reversal is present and whether alpha reaches t = 2. Both answers are clear
as stated, but the label is stronger than the numbers:
- The README's reading for this outcome is "the gross return is the published reversal premium". That is not what the
  decomposition shows: the factor contributes 19% of the mean and alpha 60%.
- The outcome turns on a small gap. Alpha t runs from 1.57 to 2.06 across the nested models, and one of the five clears 2.
- A better statement: reversal is clearly in the book, but alpha isn't distinguishable from zero *and* isn't shown to be
  zero. The 95% interval on alpha is about -14 to +723 bp a year.

The rule and the README wording are left as committed.

## 3. Where the gross return comes from (`factor_loadings.csv`, bp per year)

| Book | Mean | Alpha (t) | `st_rev` | `mkt_rf` | `hml` | Others | R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM `optimised` (headline) | 32 | 26 (0.3) | 10 | 3 | -19 | 13 | 0.08 |
| LightGBM `neutral_volscaled` | 592 | 355 (1.9) | 110 | 61 | -45 | 111 | 0.50 |
| B0 `optimised` | 182 | 95 (0.5) | 158 | -32 | -16 | -22 | 0.49 |
| B0 `neutral_volscaled` | 366 | 130 (0.6) | 205 | 18 | -16 | 29 | 0.48 |

- **The B0 books are mostly the reversal factor.** `st_rev` t is 20.3 and 9.3, and R² jumps from 0.11 to 0.48 for
  `neutral_volscaled` when it enters. Their alphas are insignificant.
- **Momentum is the LightGBM book's largest risk, not its return.** The beta is 0.28 (t 15.8), and R² rises from 0.13 to
  0.40 when momentum enters. It adds only 21 bp a year, because the factor's mean was small in this window.
- **LightGBM also carries value and quality tilts:** HML t 4.4, RMW t 3.8 and CMA t -3.6. Together they contribute about
  25 bp a year.
- **The headline book explains almost nothing.** It has R² 0.08 and alpha t 0.3, because it holds so little risk that
  there is little to attribute.
- **BAB adds nothing** to any book. Its largest t is 1.1.

The tuning window gives a similar picture for LightGBM `neutral_volscaled`: mean 444 bp, alpha 272 bp (t 1.40), `st_rev`
123 bp. Its alpha t never clears 2 once `st_rev` is in the model.

## 4. The project's own reversal book (`own_reversal_spanning.csv`)

Adding the B0 `neutral_volscaled` gross return to the largest model for the LightGBM `neutral_volscaled` book:

| | Without B0 | With B0 |
|---|---:|---:|
| alpha | 355 bp (t 1.89) | 285 bp (t 1.65) |
| `st_rev` beta | 0.166 (t 10.4) | 0.001 (t 0.08) |
| B0 book beta | | 0.534 (t 14.2) |
| B0 book contribution | | 195 bp |
| R² | 0.50 | 0.64 |

The published factor's loading disappears. The reversal in this book is the five-day reversal inside the S&P 500, which
accounts for about a third of the mean return. Momentum stays large (beta 0.29, t 17.9).

## 5. Net of costs (`factor_models.csv`)

Net alpha isn't like for like, because the factors are gross of their own costs, so these numbers are secondary.

| Book | Net alpha, bp a year | t |
|---|---:|---:|
| LightGBM `optimised` | -7 | -0.1 |
| LightGBM `neutral_volscaled` | -430 | -2.3 |
| B0 `optimised` | -103 | -0.6 |
| B0 `neutral_volscaled` | -1,049 | -5.1 |

## 6. Risk (`risk_statistics.csv`, `drawdowns.csv`, `var_es.csv`)

These are point estimates, with no confidence intervals.

| Book | Vol | Skew, daily | Excess kurtosis | Max drawdown, gross | Max drawdown, net | 99% 1-day ES | Worst month |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM `optimised` | 2.4% | 0.0 | 2.2 | -5.2% | -5.5% | 0.49% | -1.9% (2015-12) |
| LightGBM `neutral_volscaled` | 7.4% | 0.0 | 6.7 | -10.9% | -25.6% | 1.80% | -5.7% (2015-04) |
| B0 `optimised` | 6.9% | 1.8 | 23.2 | -19.3% | -28.6% | 1.60% | -4.0% (2019-08) |
| B0 `neutral_volscaled` | 7.2% | 0.5 | 15.7 | -24.0% | -70.5% | 1.80% | -4.9% (2020-03) |

- **Reversal has positive skew here, and very fat tails.** The large days are often gains. In March 2020 the B0 books hit
  their gross troughs on 2020-03-18, then recovered to their earlier peaks within weeks.
- **Costs drive the long declines.** B0 `neutral_volscaled` is under water net on 1,995 of 1,999 days. Its gross drawdown
  lasted 179 days.
- **Nothing is hiding in the tails of the LightGBM book** that its Sharpe conceals: its daily skew is about zero and
  its 5-day skew is positive (0.9).
- **Recent weakness:** the LightGBM `neutral_volscaled` gross drawdown runs from 2020-11-06 to the end of the window, unrecovered.

## 7. Why realised beta differs from ex-ante (`beta_*.csv`)

Regressions on the universe return use 1,993 days, because that return doesn't exist on the six days after the research
cutoff. Mkt-RF regressions use all 1,999.

| Book | Mean ex-ante | Realised, equal-weighted (t) | Realised, Mkt-RF (t) | 63-day exposure |
|---|---:|---:|---:|---:|
| LightGBM `optimised` | -0.005 | 0.007 (1.5) | 0.009 (2.1) | -0.009 |
| LightGBM `neutral_volscaled` | 0.001 | 0.059 (2.2) | 0.090 (4.7) | -0.028 |
| B0 `optimised` | -0.005 | 0.100 (5.0) | 0.054 (3.0) | 0.001 |
| B0 `neutral_volscaled` | 0.002 | 0.124 (4.9) | 0.108 (4.3) | 0.013 |

The design declared the explanations without picking a winner:
- **Estimation error: not it.** All 16 realised-beta 95% intervals exclude the mean ex-ante beta.
- **Benchmark: not it.** Moving to Mkt-RF raises one book's beta and lowers another's, and doesn't close the gap.
- **Stale beta: not it.** The 63-day exposure stays small, and has the opposite sign for LightGBM `neutral_volscaled`.
  It also fails the conditional test: `b1` is rejected against 1 in 7 of 8 cases.
- **Conditional beta:** with the ex-ante measure, `b1` is consistent with 1 in 7 of 8 gross cases, and `b0` stays
  significantly positive in 5 of 8. That points to a steady offset, not a mismeasured exposure.
- **Up/down asymmetry: not it.** The largest t on the difference is 1.7.
- **Factor controls: the strongest evidence.** For the B0 books, Mkt-RF beta falls once `st_rev` enters (0.084 to 0.014, and
  0.031 to -0.022), so much of the realised beta comes along with reversal itself. LightGBM `neutral_volscaled` keeps 0.042.

This is descriptive. It says where the beta sits in a factor model, not what mechanism causes it.

## Limitations

- **One 8-year path.** Alpha t of 1.89 can't separate zero from a sizeable alpha.
- **Published factors aren't S&P 500 factors.** They are all-cap, and Mkt-RF stands in for SPY.
- **The confirmatory rule has a sharp edge.** It uses one threshold, so a small change in alpha t flips the label.
- **Gross comparison is generous.** Factors are gross of their costs, and so are the books, but the books' costs were
  shown to be large.
- **Written after seeing results.** The rule chooses no book, but the book's 2.29 gross t was known when the threshold
  was set.
- **Factor vintage.** Published factor files are revised. Hashes pin the version used.

## Corrections during the run

One bug in `src/run_attribution.py` stopped the first run before any result existed: the universe return is missing on
the six days after the research cutoff. The fix is in its own commit (02e0444) with a test. Config and engine were not touched.

## Not pre-registered (listed, not run)

- **A value-weighted S&P 500 market proxy** built from the panel, in place of Mkt-RF.
- **Attribution on a shorter reversal factor** built on the full CRSP-style universe.
- **Rolling factor loadings**, to see whether momentum exposure is stable through time.
- **A characteristic-level look at why LightGBM loads on momentum**, for example from SHAP values.
- **A decomposition of the alpha by long and short leg.**
- **The B1 and h=21 books.** They were not declared here.
