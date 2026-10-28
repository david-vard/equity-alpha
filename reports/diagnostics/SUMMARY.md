# Pre-model diagnostics

Research sample 1999-01-04 to 2021-11-30 (5,766 dates, 2.89M rows), read only through
`load_research_panel`. Design from `configs/research.yaml`. Produced by
`python -m src.run_diagnostics`. Provenance is in `run_metadata.json`.

**No bug found.** Compounded daily labels reproduce `fwd_ret_{5,21}d` on every comparable cell. The
recomputed raw spreads, ICs and turnover match `reports/signal_research/` exactly, and `fwd_ret_xs`
leaves the D10-D1 spread unchanged (max difference 6e-9).

## 1. Why the h=21 IC is smaller than the h=1 IC

**Label check** (`label_compounding_check.csv`). For h=21, 99.1% of labels have all 21 daily
labels. On those 2.86M cells the product of the daily labels equals the stored label: max relative error 2.8e-6
(float32 storage), 0 cells above 1e-5. The other 0.9% cannot be compared and are explained by category: 0.54% left
the panel before the window ended, 0.22% are terminal exits (the 484 terminal cells that could be compared
also match), and 0.15% have an interior gap. h=5 gives the same picture.

**IC by horizon** (`ic_horizon_reconciliation.csv`, rank IC; the IC computed from compounded labels equals the IC from stored labels to 1e-7):

| Feature | IC h=1 | IC h=21 | ratio | sqrt(21) |
|---|---:|---:|---:|---:|
| mom_12_1 | +0.0134 | +0.0063 | 0.47 | 4.58 |
| ret_252d | +0.0113 | +0.0026 | 0.23 | 4.58 |
| vol_63d | -0.0122 | -0.0048 | 0.39 | 4.58 |
| ret_5d | -0.0104 | -0.0115 | 1.11 | 4.58 |

Restricting to names with a complete 21-day window barely moves anything (mom_12_1 h=1: 0.0130 vs 0.0134). Survivorship
in the decay frame is therefore not the explanation.

**Mechanism** (`ic_decomposition.csv`, exact identity, max error 2e-15). For log returns, on every date
`IC_21 = sum_k IC_k * sd(r_k) / sd(R_21)`. Averaged over dates, this splits into a no-covariance part and a covariance part:

| Feature (z, linear) | daily IC | sum of weights | no-covariance (sqrt-h) prediction | covariance term | actual IC_21 |
|---|---:|---:|---:|---:|---:|
| mom_12_1 | +0.0075 | 4.67 | +0.0348 | **-0.0239** | +0.0109 |
| ret_252d | +0.0075 | 4.67 | +0.0349 | **-0.0257** | +0.0092 |
| vol_63d | -0.0070 | 4.67 | -0.0328 | **+0.0105** | -0.0223 |
| ret_5d | -0.0003 | 4.67 | -0.0012 | -0.0044 | -0.0056 |

The weights sum to 4.67, about sqrt(21). Daily cross-sections are close to independent, so sqrt(h) scaling *would* hold
if the daily IC did not depend on the day. It does depend on the day: the IC is negative on the most dispersed days, and those
days dominate a 21-day return. For momentum, the covariance term removes about 70% of the sqrt(h) prediction. Direct evidence
(`ic_dispersion.csv`, `figures/ic_by_dispersion_decile_h1.png`): on the 10% most dispersed days the mom_12_1
daily IC is -0.033, against +0.018 on the other days. The dispersion-weighted mean daily IC is 0.0049, not 0.0134. vol_63d shows the
mirror image: on loud days high-vol names win (+0.023).

**Conclusion: a real feature of the data, exposed by the estimator.** The equal-weighted mean of daily ICs
treats a crash/rebound day like a quiet day. A multi-day holder's return is dominated by the loud days, where
momentum and low-vol lose. The h=1 IC therefore overstates what these signals earn over a holding period. Reversal
(ret_5d) behaves the other way: it is stronger on dispersed days.

## 2. Decile spreads: beta-hedged and robust (`decile_spreads_robust.csv`, `figures/risk_deciles_h21.png`)

Spread = D10 - D1 on non-overlapping dates. NW t in brackets. rank = mean within-date return rank in percentile points.

| Feature, h | IC | raw (bp) | beta-hedged (bp) | winsorised (bp) | median (bp) | rank (pp) |
|---|---:|---:|---:|---:|---:|---:|
| vol_63d, 1 | -0.012 | +1.9 (0.7) | -2.5 (-1.2) | +0.4 (0.2) | -5.3 (-2.1) | -1.2 (-3.9) |
| vol_63d, 21 | -0.005 | +39.9 (0.7) | -49.0 (-1.2) | +24.8 (0.5) | -13.8 (-0.3) | +0.0 (0.0) |
| beta_252d, 21 | -0.001 | +34.1 (0.6) | -90.1 (-2.1) | +26.4 (0.5) | +3.7 (0.1) | +0.2 (0.1) |
| mom_12_1, 1 | +0.013 | +0.6 (0.2) | +1.0 (0.4) | +1.8 (0.7) | +5.4 (2.3) | +1.5 (5.5) |
| mom_12_1, 5 | +0.015 | +0.7 (0.1) | +1.5 (0.1) | +3.3 (0.3) | +7.0 (0.6) | +1.9 (3.1) |
| ret_252d, 1 | +0.011 | -1.5 (-0.5) | -1.1 (-0.5) | -0.2 (-0.1) | +3.0 (1.2) | +1.3 (4.4) |
| ret_5d, 1 | -0.010 | -8.5 (-4.1) | -6.8 (-3.7) | -8.3 (-4.4) | -7.7 (-4.1) | -0.8 (-3.9) |
| ret_5d, 5 | -0.015 | -31.7 (-3.4) | -24.2 (-3.0) | -29.7 (-3.4) | -25.8 (-2.9) | -1.1 (-2.6) |

- **Volatility and beta: the contradiction is a beta effect.** The top vol decile has a beta about 0.98 higher than the
  bottom (beta_252d: 1.42), so in a rising equal-weighted market its raw mean wins. Beta-hedged, the vol spread has the IC's
  sign at every horizon (t -1.2 to -1.4), beta_252d becomes a betting-against-beta spread (t about -2.0 at all h), and the
  rank and median spreads agree with the IC at h=1. Winsorising alone does not fix it: the cause is market exposure,
  not outliers. The decile profile is hump-shaped in ranks and medians (decile 10 is the worst; raw means are lifted by its right tail).
- **Momentum, 12-month return, 52-week high, trend_200d: a "typical stock" effect.** In rank space they are strongly
  positive at h=1 (mom_12_1 t 5.5, ret_252d 4.4, ret_126d 3.6, dist_high 3.1, trend_200d 3.2). In mean space they are
  zero or negative under every variant, including beta-hedged. The loser decile's mean is carried by large rebounds,
  the same mechanism as item 1. The contradiction disappears for ranks and medians but **not** for anything a dollar-weighted
  portfolio earns.
- **Reversal (ret_1d, ret_5d, intraday_ret) is robust.** The spread is the same sign and significant under every
  variant at h=1 and h=5. Nothing is significant at h=21.

## 3. Break-even costs (`breakeven_costs.csv`)

Assumptions: dollar-neutral, $1 per leg, equal weight, rebalanced every h-th date (offset 0), no drift between
rebalances, an index exit counts as traded. One-way leg turnover TO = 0.5·Σ|Δw| is the amount sold; the same amount is bought,
so traded notional per rebalance = 2·(TO_D1 + TO_D10). **Per-side break-even = spread / (2·(TO_D1 + TO_D10))**,
the cost per unit traded (each buy and each sell). Round trip is twice that, which equals spread / (TO_D1 + TO_D10), i.e. the
formula in the brief. Spreads are oriented by the prior sign, or by the full-sample IC sign where there is no prior
(descriptive only). The hedge's own trading is ignored. Halves split at 2010-06-21.

Reversal, raw spread, per-side break-even in bp (spread t in brackets):

| Signal, h | turnover (both legs) | full | first half | second half |
|---|---:|---:|---:|---:|
| ret_1d, 1 | 1.70 | 2.1 (3.7) | 4.2 (4.3) | 0.0 (0.0) |
| ret_5d, 1 | 0.80 | 5.3 (4.1) | 8.7 (4.4) | 1.8 (1.1) |
| intraday_ret, 1 | 1.70 | 1.9 (3.5) | 4.0 (4.3) | -0.2 (-0.4) |
| ret_1d, 5 | 1.72 | 8.4 (2.8) | 13.3 (3.0) | 3.7 (0.9) |
| ret_5d, 5 | 1.73 | 9.1 (3.4) | 14.2 (3.8) | 4.1 (1.0) |
| intraday_ret, 5 | 1.72 | 8.0 (2.9) | 16.2 (3.6) | 0.0 (0.0) |
| ret_5d, 21 | 1.72 | 10.3 (0.8) | 11.4 (0.6) | 9.2 (0.6) |

- **h=1 does not survive.** Since 2010 the daily reversal spread is 0-3 bp per day against ~1.7 units of turnover, a
  break-even of 0-2 bp per side, below any realistic S&P 500 cost.
- **h=5 is marginal**: about 4 bp per side (8 bp round trip) after 2010 and not significant on its own (t about 1).
- Slow signals (momentum, vol, trend, 52-week high) turn over 0.1-0.6 per rebalance, but their raw spreads are ≤ 0, so a
  break-even is not meaningful. Beta-hedged, vol_63d (h=5: 36 bp per side, t 1.4) and beta_252d (122 bp, t 2.0) have room
  to pay for costs, because their turnover is very low.
- dollar_volume (small names do better) is a first-half effect: t 1.8 → 0.1.
- Caution: h=21 uses 274 rebalances at offset 0, and the means move a lot with the offset (mom_12_1: -23.7 bp at
  offset 0 vs -9.5 bp averaged over all offsets; see `spread_bps_all_offsets`).

## Recommendation for modelling

1. **Targets: h=5 and h=21; drop h=1** (break-even ≈ 0-2 bp per side since 2010). h=5 carries the only robust
   signal, and h=21 is where the slow, low-turnover signals could pay.
2. **Make the rank-transformed target the primary variant, not a robustness check.** Items 1 and 2 show that raw-return
   losses are dominated by high-dispersion days and right tails, where momentum and low-vol flip sign. Pooled MSE on
   `fwd_ret_xs` will be driven by those days. Keep the raw target as the variant.
3. **Evaluate deciles beta-hedged, with the rank-space spread next to it.** The raw spread mixes in market beta (vol, beta)
   and tail rebounds (momentum). Report the raw spread too, but only as a secondary column.
4. **Keep all 16 features.** Reversal (ret_5d, ret_1d, intraday_ret) is the expected driver. Momentum/trend/52-week high
   carry a rank-space signal that a rank target or trees may exploit but a mean spread will not reward. vol/beta
   are informative only once beta-neutral. Expect the low-vol and momentum signs in models to depend on the target choice.
5. **Report the dispersion-weighted IC next to the equal-weighted IC**, so that a model which only wins on quiet days
   is visible.
6. **B1 composite as specified** is fine, but its vol_63d component loses money unhedged. Evaluate it beta-hedged like the models.
