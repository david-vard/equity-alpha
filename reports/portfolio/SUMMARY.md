# Portfolio backtest: gross vs net

Design from `configs/portfolio.yaml` (config sha256 `6e2e5c1a…`, the same in every run). Produced by
`python -m src.run_portfolio --tune` and `--evaluate`, run once per signal, then merged by `python -m src.portfolio_report`.
Inputs were read only through `load_research_panel` and `data/processed/oof_predictions.parquet`.

**Windows:**
- **Tuning:** 2007-01-03 to 2013-12-31. The optimiser's λs were chosen here, so every tuning-window number is in-sample for the λs.
- **Evaluation:** 2014-01-02 to 2021-12-08. It was run once, with the frozen λs. Open tranches run to the end of their labels and then
  liquidate at cost. Every date is before `holdout_start` (2022-01-01).

Base case: $100m of capital, ×1 costs, lag 1, h staggered tranches. Notebook: `notebooks/04_portfolio.ipynb`.

**Headline: no net edge.** In the evaluation window, `optimised` on `lightgbm_h5_rank` has a net Sharpe of **-0.00 (NW t -0.00)**,
and B0 has -0.03 (t -0.07). After costs, none of the 12 signal × construction books is significantly different from zero.

## 1. Reconciliation (`reconciliation.csv`)

The test runs `decile_ls` on `B0_reversal_h5` with zero costs, one tranche and offset 0, against the raw h=5 D10-D1 spread in
`reports/models/decile_spreads.csv` (751 non-overlapping periods):

| Source | Mean per period | Vol | Sharpe | NW t |
|---|---:|---:|---:|---:|
| models report | 0.002904 | 0.035048 | 0.5883 | 2.176 |
| spread recomputed from labels | 0.002904 | 0.035048 | 0.5883 | 2.176 |
| engine | 0.002898 | 0.035003 | 0.5877 | 2.176 |

**Passes.** The mean differs by -7e-6. All of the difference comes from 6 periods in which a leg holds a name with no label.
The models report drops that name, while the engine holds it at the last price. Without those 6 periods, the largest difference
is 1e-16. At zero costs, gross equals net exactly, with costs summing to 0, for `decile_ls` and `neutral_volscaled` over the full
sample and for `optimised` over its first 60 out-of-sample days.

## 2. Tuning (`lambda_grid.csv`, `hyperparameters.csv`)

The λ grid is λ_risk ∈ {3, 10, 30, 100} × λ_tc ∈ {0.5, 1, 2, 4}. The choice maximises net Sharpe in the tuning window.

| Signal | λ_risk | λ_tc | Tuning net Sharpe (in-sample) | On the grid edge |
|---|---:|---:|---:|---|
| lightgbm_h5_rank | 100 | 4 | 0.35 | **both λs (corner)** |
| B0_reversal_h5 | 3 | 1 | 0.45 | λ_risk |
| B1_composite_h5 | 3 | 2 | -0.14 | λ_risk |
| lightgbm_h21_rank | 3 | 4 | 0.48 | both λs |

**Every choice is on the edge of the grid.** The grid was not extended.

For `lightgbm_h5_rank`, net Sharpe broadly rises along the λ_tc axis at every λ_risk. At λ_tc=0.5 the book trades 24-31× a year and
goes negative net (gross Sharpe 0.38-0.57). At λ_tc=4 it trades 1.6-2.6× a year. The grid's preferred direction is "trade less"
and it stops at the boundary.

## 3. Headline (`headline.csv`), evaluation window

| | lightgbm_h5_rank | B0_reversal_h5 |
|---|---:|---:|
| net Sharpe | -0.00 | -0.03 |
| net NW t (lags 7) | -0.00 | -0.07 |
| net return, annualised | -0.00% | -0.18% |
| net vol, annualised | 2.4% | 6.9% |
| gross Sharpe | 0.14 | 0.26 |
| annual one-way turnover | 1.2× | 9.9× |
| break-even cost multiplier | 0.99 | 0.91 |
| *tuning-window net Sharpe (in-sample λs)* | *0.35 (t 0.94)* | *0.45 (t 1.10)* |

The frozen λs fell from 0.35 and 0.45 in-sample to about zero out of sample. The headline book runs at about 2.4% vol because
λ_risk=100 shrinks its risk sharply. The optimised book has no volatility target.

## 4. What each construction rung adds net of costs (`ladder.csv`, evaluation window)

| Signal | decile_ls net SR | neutral_volscaled net SR | optimised net SR | Turnover/yr (d / n / o) |
|---|---:|---:|---:|---|
| lightgbm_h5_rank | -0.55 (t -1.6) | -0.26 (t -0.8) | -0.00 (t -0.0) | 64 / 54 / 1.2 |
| B0_reversal_h5 | -1.10 (t -2.6) | -1.13 (t -2.8) | -0.03 (t -0.1) | 88 / 72 / 9.9 |
| B1_composite_h5 | -0.38 (t -1.1) | 0.28 (t 0.8) | -0.06 (t -0.2) | 42 / 29 / 2.4 |
| lightgbm_h21_rank | -0.21 (t -0.7) | -0.13 (t -0.4) | -0.06 (t -0.2) | 15 / 11 / 2.4 |

- **`decile_ls` to `neutral_volscaled`:** risk scaling and neutralisation raise gross Sharpe for every signal. For
  `lightgbm_h5_rank`, gross Sharpe goes from 0.20 to 0.80, the highest gross figure anywhere. Turnover stays at 11-72× a year,
  though, so net Sharpe is still negative except for B1.
- **`neutral_volscaled` to `optimised`:** turnover falls 5-45×, and net Sharpe moves toward zero. This happens mainly because both
  the gross return and the costs shrink, not because a net edge appears. Gross Sharpe for `lightgbm_h5_rank` falls from 0.80 to
  0.14.
- **The only positive net book is B1 `neutral_volscaled`,** at 0.28 (t 0.78, break-even multiplier 1.7). It is not significant.

## 5. Where the costs go (`cost_breakdown.csv`, bp per year on capital, evaluation window)

| Book | commission | spread | impact | borrow | total | gross | net |
|---|---:|---:|---:|---:|---:|---:|---:|
| lightgbm_h5 decile_ls | 64 | 257 | 1138 | 25 | 1484 | 400 | -1084 |
| lightgbm_h5 neutral_volscaled | 54 | 215 | 493 | 25 | 786 | 592 | -194 |
| lightgbm_h5 optimised | 1 | 5 | 6 | 21 | 33 | 32 | -0 |
| B0 optimised | 10 | 40 | 126 | 24 | 200 | 182 | -18 |

- Impact is the largest trading cost in every high-turnover book, at 1-5× the spread cost.
- In the headline book trading costs are only 11 bp a year, and **borrow (21 bp) is the largest single item.** At about 1.2×
  turnover, the 25 bp short-borrow assumption costs about as much as the whole gross return.
- Cost per unit of one-way turnover is 11-13 bp for `decile_ls` and 5-9 bp for the other constructions.

## 6. How AUM and the cost multiplier change the picture (`scenarios.csv`)

The scenarios vary one dimension at a time around the base case. For `decile_ls` and `neutral_volscaled` the trades do not depend
on costs, so net Sharpe falls steadily as the multiplier rises. For `lightgbm_h5_rank` `neutral_volscaled`, net Sharpe is 0.80 at
×0, 0.27 at ×0.5 and -0.26 at ×1, and its break-even multiplier is 0.75.

| AUM (base costs) | lightgbm_h5 decile_ls | lightgbm_h5 neutral_volscaled | lightgbm_h5 optimised | B0 optimised |
|---|---:|---:|---:|---:|
| $10m | -0.15 | 0.19 | -0.25 | -0.14 |
| $100m | -0.55 | -0.26 | -0.00 | -0.03 |
| $1bn | -1.78 | -1.69 | -0.02 | -0.07 |

- **Fixed-trade constructions:** impact scales with sqrt(trade size), so net Sharpe falls fast with AUM. At $1bn both lose 1.7-1.8
  Sharpe a year. At $10m, `neutral_volscaled` on `lightgbm_h5_rank` turns positive (0.19, not significant).
- **`optimised`:** the book re-solves under each scenario's costs, so it is not the base trades repriced, and its curve is not
  monotone. With zero costs it trades 15-82× a year and earns a gross Sharpe of -0.25 (lightgbm_h21) to 0.57 (B0). At $10m it trades
  more than at $100m and does worse (-0.25). At $1bn it trades very little and stays near zero.
- With costs applied (multiplier > 0), no (signal, construction, scenario) cell in the evaluation window has a net NW t above
  1.33. The only t above 2 is zero-cost `lightgbm_h5_rank` `neutral_volscaled` (t 2.29), which is a gross number.

## 7. Does LightGBM beat B0 after costs?

**No.** Under `optimised`, the headline net Sharpe is -0.00 against -0.03 for B0, a difference of 0.02 in Sharpe with both t-stats
near zero. LightGBM trades about 8× less than B0, but it also earns about half B0's gross Sharpe (0.14 vs 0.26). Under the two
fixed-trade constructions, LightGBM loses less than B0 (decile -0.55 vs -1.10, neutral -0.26 vs -1.13), but both are negative net.

## 8. Exposures (`exposures.csv`, evaluation window)

- Ex ante the neutral books hold their constraints: mean |β'w| ≤ 0.04, and |net exposure| ≤ 0.6% of capital.
- **Realised** beta, from a regression of net returns on the equal-weighted universe return, is small but often significant:
  - B0 `optimised` 0.10 (t 5.0);
  - B0 `neutral_volscaled` 0.12 (t 4.8);
  - the headline book 0.007 (t 1.5).
- `decile_ls` is not beta-neutral by construction. B1's decile book runs a realised beta of -0.64 (t -6.4).
- Fallbacks are rare: at most 8 holds in 3,776 rebalances for any optimised scenario, 1 flat (B1 base) and 7 partials (B1 ×0).
  None occurs in the fixed-trade constructions.

## 9. By year (`by_year.csv`)

Net Sharpe for the headline book swings from -1.2 (2020) to +1.7 (2018) across the evaluation years. In the tuning years it ranged
from -2.3 (2009) to +2.1 (2007). The yearly signs are mixed for every signal, and no single year drives the near-zero total.

## Limitations

- **No quote data.** The half-spread is a flat 2 bp, not a measured bid-ask. Small and illiquid names are probably undercharged.
- **Assumed impact coefficient.** Y=1 in Y·σ·sqrt(trade/ADV) is not calibrated to fills. Impact is the largest trading cost in most
  books, so the results are sensitive to it. The cost-multiplier scenarios bound this only crudely.
- **Borrow is flat.** It is 25 bp a year on all shorts, with no hard-to-borrow names and no locate failures.
- **No sector or industry neutrality.** Only dollar and beta are neutralised, so residual industry bets are not controlled.
- **A single path.** The evaluation window is one 8-year path, with no bootstrap. The λs were chosen on another single 7-year path,
  and every choice was on the grid edge.
- **Provenance.** `run_metadata.json` records `code_dirty: true` only because `src/portfolio_report.py` was untracked during the
  runs. The engine, constructions and runner are unchanged from commit 3f967a2.

## Not pre-registered (listed, not run)

- **A wider λ grid**, for example λ_tc > 4 or λ_risk < 3 or > 100. Every chosen point is on an edge.
- **The full cost-multiplier × AUM cross.** The design varies one dimension at a time.
- **A volatility target for `optimised`**, matching the 10% target of `neutral_volscaled`. This would make the gross and net
  numbers comparable in size across rungs.
- **A no-trade band or a longer rebalance interval** for `neutral_volscaled`. It has the highest gross Sharpe but loses net only
  through turnover.
- **Sector-neutral constraints** in `optimised`.
- **Bootstrap confidence intervals** for the differences in net Sharpe, for example LightGBM vs B0.
