# Data

This folder holds data only. The access code lives in [`src/data.py`](../src/data.py).
Everything under `data/raw/` and `data/processed/` is gitignored (licensed market data).

## Setup

1. Copy `env.example` to `.env` in the repository root and set `SHARADAR_API_KEY`.
   `.env` is gitignored. The key can also come from the environment.
2. Use it:

```python
from src.data import get_universe_prices, get_sp500_membership

prices = get_universe_prices("1998-01-01", "2026-10-23")   # date x ticker, closeadj
members = get_sp500_membership("1998-01-01", "2026-10-23")  # long: date, ticker
```

`get_universe_prices(start, end, price_field="closeadj", refresh=False)` returns a wide frame
indexed by trading date with one column per ticker. A price is kept only if the ticker was an
S&P 500 member on that date, otherwise it is `NaN`. `price_field` is `closeadj` (splits, dividends,
spinoffs), `close` (splits only) or `closeunadj`.

### Several fields at once

```python
from src.data import get_universe_panel

panel = get_universe_panel("1998-01-01", "2026-10-23", fields=("closeadj", "close", "volume"))
returns = panel["closeadj"].pct_change()
dollar_volume = panel["close"] * panel["volume"]
```

`get_universe_panel` returns a dict of wide frames, one per field, all with identical index and
columns and the same membership filter. Available fields: `open`, `high`, `low`, `close`,
`closeadj`, `closeunadj`, `volume`. `volume` is split adjusted, so use `close` (not `closeadj`,
which also adjusts for dividends) when computing dollar volume. Each distinct field list is a
separate cache entry, so decide the field list before pulling a long history.

## Cache

Raw CSV responses are cached under `data/raw/sharadar/<table>/`, one file per request. Delete the
folder, or pass `refresh=True`, to re-download. The `sp500` request includes today's date, so it
refreshes once a day.

## S&P 500 membership

Rebuilt from the `sp500` table: the `current` constituents are the anchor, `added`/`removed`
changes (effective dates; removals applied before additions on the same day) are unwound back to
the start date and replayed forward. Quarterly `historical` snapshot rows are ignored. This
reproduces all 115 quarterly snapshots (1998-03-31 to 2026-09-30) exactly.

## Fields used and how they are adjusted

Checked on the cached `stocks` (SEP) data and on AAPL's 4-for-1 split (2020-08-31):

| Field | Adjustment | Used for |
|---|---|---|
| `closeadj` | splits, dividends and spinoffs (back-adjusted to today) | every return, momentum, volatility, beta, trend, labels |
| `close` | splits only (124.81 on 2020-08-28, continuous across the split) | dollar volume, adjusted open |
| `open` | splits only, same basis as `close` | overnight and intraday returns |
| `volume` | splits only, same basis as `close` | dollar volume (`close * volume`) |
| `closeunadj` | none | not used |

`closeadj / close` is constant between dividend ex-dates, so an adjusted open is
`open * closeadj / close`. Adjusted levels are back-adjusted with future dividends, so only
ratios within a window (returns, price / moving average, price / rolling high) are used, never
levels. Ratios are unaffected because the back-adjustment factor cancels.

`get_universe_panel(..., apply_membership=False)` returns the full price history of every ticker
that was ever a member (no membership NaNs). The feature pipeline uses this so that a stock that
joins the index already has its trailing history; it restricts to members afterwards.

## Survivorship and known limitations

- **Survivorship:** the universe is the point-in-time S&P 500 (delisted and acquired names are
  in the data, 1,171 tickers in total), so there is no survivorship bias from membership.
- **Delisted stocks and labels:** 360 of the 668 index exits (54%) are stocks that stop trading
  within 5 trading days of leaving the index. They are overwhelmingly takeovers, not
  underperformers: median trailing 252-day return +29%, 70% had a positive trailing 21-day return
  and only 1 of 360 had a last close under $1. Names dropped from the index for weak performance
  mostly keep trading, so their labels already continue. A forward label that needs a price after
  the last trade is not dropped: the stock is exited at its **last price** (times `1 + haircut`,
  default 0, see `labels.delisting` in `configs/features.yaml`) and flagged in `fwd_terminal_{h}d`.
  Sharadar prices run to the last OTC trade, so the last price already holds most of a bankruptcy
  loss; an extra CRSP-style -30% would double count, but can be set per reason for sensitivity
  runs. `get_delistings` reads the reason from the `actions` table: of the 7,091 terminal 21-day
  labels, 6,673 are acquisitions (+0.8% mean raw 21d return, +0.23% vs the cross-section), 378
  "other" (-2.8% vs cross-section) and 40 bankruptcies (-84% mean). The mean cross-sectional label
  moves by under 0.001% whether or not these rows are kept, but the 40 bankruptcies are heavy left-tail
  observations that were silently dropped before. Interior gaps and the end of the sample are
  never filled.
- **Extreme values:** they are real events spread over many names, not one stock or year. 97
  one-day moves above 50% across 62 tickers (43 of them in 2008, e.g. Lehman's +340% penny-stock
  bounce on 2008-09-19 while still an index member); the 377 rows with a 252-day return above
  +1,000% come from 12 tickers (Qualcomm 1999, dot-com names such as Conexant and Power-One,
  2009-10 bank recoveries such as Fifth Third and Genworth, PENN in 2021, and a 2026 memory /
  optics rally in SNDK, MU, WDC and LITE that could not be verified independently but is
  internally consistent; adjusted and split-adjusted closes agree). No large move reverses the
  next day, so there is no sign of price glitches. The only clear data errors are two bad
  `open` prints from ticker reuse or a bad quote (GAP 2002-11-01: open 23.54 vs closes near 12;
  EVHC 2016-12-02: open 22.58 vs closes near 68), 2 rows in 3.5M affecting `overnight_ret` and
  `intraday_ret` only. The z-scores winsorise them; the raw columns keep them.
- **Missing prices:** 721 of 3.6M member-days (0.02%) have no row in SEP, mostly single-day vendor
  gaps (worst day 2002-10-08, 12 names). They stay NaN; features forward-fill at most
  `max_gap_days` (2) trading days, causally.
- **Ticker reuse:** membership is keyed by ticker. Sharadar disambiguates reused symbols with a
  numeric suffix (e.g. `CBB1`); pre-2002 `MAR` has no prices at all, so it is simply missing.
- **Zero volume:** 999 rows have `volume == 0`; they are kept as zero in dollar volume.
- **Liquidity filter (off by default):** the universe is the point-in-time S&P 500, which is already
  a strong size / liquidity screen. A fixed $5m trailing dollar-volume cut-off would drop 6.4% of
  member-days in 1999 (about 32 names a date, e.g. Wendy's, Brown-Forman), 2.3% in 2002, 0.4% in
  2005 and nothing after 2010, because member dollar volume grew about 10x (median $33m in 1999,
  $316m in 2025) and prices about 2x. A constant nominal threshold therefore changes the universe
  definition over time. Liquidity and capacity are handled in the cost / portfolio stage;
  `universe.liquidity.min_avg_dollar_volume` in `configs/features.yaml` re-enables a cut-off for
  robustness checks.
- **Sample start:** 1998-01 is the start of both SEP and the S&P 500 table.

## API quirks (direct Sharadar API, `api.sharadar.com`)

- `ticker` accepts at most 30 tickers and 200 characters per request (HTTP 400 otherwise);
  requests are batched accordingly.
- Responses are limited to 10,000 rows, so requests are paged with `skip`.
- The `actions` table silently returns only the first 10 years of a longer date range (and `skip`
  cannot reach the rest), so `get_delistings` requests 5-year windows. `sp500` and `stocks` do not
  have this limit.
- `from` defaults to one year ago, so history must be requested explicitly. For `sp500`, a `from`
  earlier than 1998 combined with `sort=date.asc` returns zero rows.
- Rate limits are undocumented; requests are spaced 0.5 seconds apart.
- Membership is keyed by ticker only, so a reused ticker symbol can be confused with a different
  company.
