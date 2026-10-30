# Second round

Design from `configs/second_round.yaml`, committed before any variant was simulated. Tuning ran at commit a488291, and the
choices and holdout candidate were committed (a621f2d) before any evaluation-window return of a variant existed. All runs
recorded `code_dirty: false`. Notebook: `notebooks/07_second_round.ipynb`.

**Every evaluation-window number here is a second look.** The ideas came from seeing that window, so these numbers are weaker
evidence than the pre-registered headline and can't replace it. No confirmatory test was declared. Base costs, $100m AUM.

**The headline doesn't change, and the round leaves one open question.** The holdout candidate, LightGBM `optimised` on the top
25% of names, reaches 0.38 net (t 1.09), with an interval that contains zero, on a very small book. LightGBM `neutral_volscaled`
with band 4 reaches 0.71 net (t 2.04), and its interval sits above zero. That's the most interesting result of the round and the
least understood, and it isn't the candidate.

## 1. Checks (`checks.csv`)

All 8 passed. Rerun recorded books match the first round: 7e-18 per day for `neutral_volscaled`, and an exact tuning Sharpe for
`optimised`. Every variant was tuned, and no tuning row passes the last tuning date.

## 2. Choices on 2007-2013 (`tuning/`, 62 grid points tried, 32 of them recorded)

| Variant | Book | Choice | Tuning net Sharpe | Recorded book | On a grid edge |
|---|---|---|---:|---:|---|
| Liquid universe | LightGBM `optimised` | top 25% | **0.62** | 0.35 | yes |
| Liquid universe | LightGBM `neutral_volscaled` | top 25% | -0.47 | -0.98 | yes |
| Liquid universe | B0 `optimised` | keep recorded | 0.45 | 0.45 | yes |
| Liquid universe | B0 `neutral_volscaled` | top 25% | -1.15 | -1.77 | yes |
| No-trade band | LightGBM `neutral_volscaled` | band 4 | 0.32 | -0.98 | yes |
| No-trade band | B0 `neutral_volscaled` | band 4 | -0.06 | -1.77 | yes |
| Wider λ grid | LightGBM `optimised` | keep (100, 4) | 0.35 | 0.35 | no |
| Wider λ grid | B0 `optimised` | (1, 1) | 0.47 | 0.45 | yes |

No grid was extended again. The wider grid found nothing for LightGBM: none of its 5 new points beat (100, 4), so the first-round
corner choice is no longer on an edge.

**Holdout candidate, fixed before evaluation:** LightGBM `optimised`, top 25%, tuning net Sharpe 0.62 (t 1.68).

## 3. The second look (`evaluation.csv`, evaluation window, label `second_look`)

| Variant | Tuning net | **Net Sharpe (t)** | 95% interval | Recorded net | Difference (95% interval) | Turnover | Net vol |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM band 4 | 0.32 | **0.71 (2.0)** | 0.09 to 1.35 | -0.26 | **+0.97 (0.29, 1.69)** | 1.5× | 2.4% |
| LightGBM top 25%, `optimised` (candidate) | 0.62 | **0.38 (1.1)** | -0.29 to 1.07 | -0.00 | +0.39 (-0.35, 1.13) | 0.5× | 0.8% |
| LightGBM top 25%, `neutral_volscaled` | -0.47 | -0.57 (-1.7) | -1.20 to 0.06 | -0.26 | -0.30 (-0.70, 0.07) | 55× | 8.3% |
| LightGBM (100, 4), recorded | 0.35 | -0.00 (-0.0) | -0.60 to 0.62 | -0.00 | 0 | 1.2× | 2.4% |
| B0 (1, 1) | 0.47 | -0.06 (-0.2) | -0.84 to 0.62 | -0.03 | -0.03 (-0.15, 0.08) | 11× | 8.6% |
| B0 top 25%, `neutral_volscaled` | -1.15 | -1.01 (-2.6) | -1.73 to -0.33 | -1.13 | +0.12 (-0.26, 0.55) | 72× | 8.3% |
| B0 band 4 | -0.06 | -1.17 (-3.3) | -1.89 to -0.50 | -1.13 | -0.03 (-0.99, 0.94) | 0.7× | 1.5% |
| B0 `optimised`, recorded | 0.45 | -0.03 (-0.1) | -0.79 to 0.65 | -0.03 | 0 | 10× | 6.9% |

These are descriptive, with no multiple-testing correction across the variants.

**The candidate.** Its interval and its gap from the recorded book both contain zero. It's a very small book: mean gross 0.25
of a possible 2, net return about 30 bp a year, 37 bp gross. Its low exposure was visible in the tuning window. It goes
forward because the committed rule selects it, not because this look supports it.

**Band 4 on LightGBM.** Net return is about 169 bp a year: 204 bp gross, less 19 bp of trading costs and 17 bp of borrow. Its
Sharpe rose out of sample, from 0.32 to 0.71, and its break-even cost multiplier is 5.8.
- **The accounting looks sound.** Recorded columns show it stays neutral (largest |net exposure| 1%, mean |ex-ante beta| 0.002,
  universe beta 0.012), and it's positive in all 8 evaluation years, though only 0.17-0.18 in 2016 and 2021.
- **It isn't a five-day book any more.** At 1.5× turnover, positions last months. What it earns from is unknown: it could be
  slow-moving signal content, drift, or a factor tilt. This round doesn't test that.
- **Its holdings depend on its start date.** It started flat in 2007, and the band kept most names out, so gross exposure took
  years to build: 0.72 on average in the tuning window, 1.33 in the evaluation window, peaking near 2 in 2018. Most of its
  return arrives in 2016-2019.
- **The same rule fails on B0.** Band 4 on B0 has a gross Sharpe of -1.02, and it's negative in all 8 years. So a wide band doesn't
  help in general.
- **Selection still applies.** This is the best of six variants run, chosen at the edge of its grid.
- **It isn't the holdout candidate.** Its tuning figure of 0.32 is below the headline's 0.35. Swapping it in now would mean
  choosing on the evaluation window, which is what this design rules out.

**Liquidity didn't hold up as an idea.** In the tuning window, the top 25% improved both `neutral_volscaled` books. In the
evaluation window, LightGBM got worse and B0 barely moved. The candidate's fall from 0.62 to 0.38 fits the tuning curve's jump
(0.35, 0.26, 0.62) being partly noise.

**The wider grid added nothing.** B0 at (1, 1) moved from 0.45 to 0.47 in tuning, and it's slightly worse than the recorded book
in the second look.

## Limitations

- **Second look.** These numbers can only be suggestive. The window has been seen several times, and 62 grid points were tried.
- **One 8-year path, and for band 4 one start date.** The bootstrap treats the sample as given, so it doesn't capture how the
  book's state depends on starting in 2007.
- **Band unit.** Band 4 is close to no trading at all, and the grid stops there. Bands 1 and 2 weren't run in this window,
  because they weren't chosen.
- **The liquid `optimised` books keep the full-universe IC and first-round λs,** which is the likely cause of the candidate's low
  exposure, though this hasn't been tested.
- **Costs are the same assumptions as before.** At band 4 and 25% of names, trades are small, which is where the square-root
  impact model is least tested.

## Not pre-registered (listed, not run)

- **Attribution of LightGBM band 4,** on the factors in `configs/attribution.yaml`. This is the first thing worth doing.
- **Band 4 from other start dates,** to measure how much depends on starting in 2007.
- **The evaluation-window results of bands 1 and 2.**
- **A liquid-universe `optimised` book with a re-estimated IC or re-tuned λs.**
- **Band combined with the liquid universe.**
- **The deflated Sharpe ratio** across all 62 grid points. `n_variants_tried` is recorded for it.
