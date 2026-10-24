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

## API quirks (direct Sharadar API, `api.sharadar.com`)

- `ticker` accepts at most 30 tickers and 200 characters per request (HTTP 400 otherwise);
  requests are batched accordingly.
- Responses are limited to 10,000 rows, so requests are paged with `skip`.
- `from` defaults to one year ago, so history must be requested explicitly. For `sp500`, a `from`
  earlier than 1998 combined with `sort=date.asc` returns zero rows.
- Rate limits are undocumented; requests are spaced 0.5 seconds apart.
- Membership is keyed by ticker only, so a reused ticker symbol can be confused with a different
  company.
