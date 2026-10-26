"""Sharadar data access for the equity-alpha project.

Configure ``SHARADAR_API_KEY`` in the environment or in a repository-root
``.env`` file. Raw API responses are cached in ``data/raw/sharadar``; delete
them or pass ``refresh=True`` to re-download.

Example:
    from src.data import get_universe_prices
    prices = get_universe_prices(start="2012-01-01", end="2024-12-31")
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_RAW_CACHE_ROOT = _REPOSITORY_ROOT / "data" / "raw" / "sharadar"
_SHARADAR_DATA_URL = "https://api.sharadar.com/v1.0/data"
_API_PAGE_SIZE = 10_000
# The API answers HTTP 400 if ``ticker`` exceeds either limit.
_MAX_TICKER_PARAMETER_CHARS = 200
_MAX_TICKERS_PER_REQUEST = 30
_MIN_SECONDS_BETWEEN_REQUESTS = 0.5
# The API defaults ``from`` to one year ago, so history must be requested.
# Docs say the table starts Jan 1998.
_SP500_HISTORY_START = "1998-01-01"
_ALLOWED_PRICE_FIELDS = {"close", "closeadj", "closeunadj"}
_ALLOWED_PANEL_FIELDS = _ALLOWED_PRICE_FIELDS | {"open", "high", "low", "volume"}
_MIN_EXPECTED_CONSTITUENTS = 400
# Highest priority first: the reason recorded when a ticker has several delisting rows.
_DELISTING_REASONS = {
    "bankruptcyliquidation": "bankruptcy",
    "regulatorydelisting": "regulatory",
    "voluntarydelisting": "voluntary",
    "acquisitionby": "acquisition",
    "delisted": "other",
}
_KNOWN_SP500_ACTIONS = {"current", "historical", "added", "removed"}

load_dotenv(_REPOSITORY_ROOT / ".env")
_last_request_time = 0.0


def _get_api_key() -> str:
    """Return the configured Sharadar API key or raise an actionable error."""
    api_key = os.environ.get("SHARADAR_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SHARADAR_API_KEY is missing. Set it in your environment or add it "
            f"to {_REPOSITORY_ROOT / '.env'} (see env.example)."
        )
    return api_key


def _cache_path(table_name: str, query_parameters: dict[str, Any]) -> Path:
    """Build a deterministic cache path without including the API key."""
    canonical_query = json.dumps(query_parameters, sort_keys=True, separators=(",", ":"))
    query_digest = hashlib.sha256(canonical_query.encode("utf-8")).hexdigest()
    return _RAW_CACHE_ROOT / table_name / f"{query_digest}.csv"


def _request_csv(table_name: str, query_parameters: dict[str, Any]) -> bytes:
    """GET one CSV page; errors never include the URL, which carries the key."""
    global _last_request_time
    wait = _MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()

    request_parameters = {
        **query_parameters,
        "api_key": _get_api_key(),
        "format": "csv",
    }
    try:
        response = requests.get(
            f"{_SHARADAR_DATA_URL}/{table_name}",
            params=request_parameters,
            timeout=(10, 120),
        )
    except requests.RequestException as error:
        raise RuntimeError(
            f"Sharadar request for '{table_name}' failed: {type(error).__name__}"
        ) from None
    if not response.ok:
        detail = ""
        try:
            detail = f": {response.json().get('description', '')}"
        except (ValueError, AttributeError):
            pass
        raise RuntimeError(
            f"Sharadar returned HTTP {response.status_code} for '{table_name}'{detail}"
        )
    return response.content


def _read_csv_bytes(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(content))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fetch_api_page(
    table_name: str, query_parameters: dict[str, Any], refresh: bool = False
) -> pd.DataFrame:
    """Read one CSV page from cache or retrieve and cache it from Sharadar."""
    cache_path = _cache_path(table_name, query_parameters)
    if not refresh and cache_path.exists():
        return _read_csv_bytes(cache_path.read_bytes())

    content = _request_csv(table_name, query_parameters)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(cache_path)
    return _read_csv_bytes(content)


def _fetch_table(table_name: str, refresh: bool = False, **filters: Any) -> pd.DataFrame:
    """Fetch all rows matching filters, following Sharadar's page limit."""
    result_pages: list[pd.DataFrame] = []
    row_offset = 0

    while True:
        page_parameters = {
            **filters,
            "limit": _API_PAGE_SIZE,
            "skip": row_offset,
            "sort": "date.asc",
        }
        page = _fetch_api_page(table_name, page_parameters, refresh)
        if page.empty:
            break

        result_pages.append(page)
        if len(page) < _API_PAGE_SIZE:
            break
        row_offset += len(page)

    if not result_pages:
        return pd.DataFrame()
    return pd.concat(result_pages, ignore_index=True).drop_duplicates()


def _parse_date_range(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse and validate an inclusive date range."""
    try:
        start_date = pd.Timestamp(start).normalize()
        end_date = pd.Timestamp(end).normalize()
    except (TypeError, ValueError) as error:
        raise ValueError("start and end must be valid dates in YYYY-MM-DD format") from error

    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError("start and end must be valid dates in YYYY-MM-DD format")
    if start_date > end_date:
        raise ValueError("start must be on or before end")
    return start_date, end_date


def _batch_tickers(tickers: list[str]) -> list[list[str]]:
    """Group tickers so each comma-joined batch fits the API's length limit."""
    batches: list[list[str]] = []
    batch_length = 0
    for ticker in tickers:
        added_length = len(ticker) + (1 if batches and batches[-1] else 0)
        if (
            not batches
            or batch_length + added_length > _MAX_TICKER_PARAMETER_CHARS
            or len(batches[-1]) >= _MAX_TICKERS_PER_REQUEST
        ):
            batches.append([ticker])
            batch_length = len(ticker)
        else:
            batches[-1].append(ticker)
            batch_length += added_length
    return batches


def _load_sp500_records(refresh: bool = False) -> pd.DataFrame:
    """Load and clean the full S&P 500 action table (history, not just last year)."""
    sp500_records = _fetch_table(
        "sp500",
        refresh,
        **{
            "from": _SP500_HISTORY_START,
            "to": pd.Timestamp.today().strftime("%Y-%m-%d"),
            "fields": "date,action,ticker",
        },
    )
    missing_columns = {"date", "action", "ticker"}.difference(sp500_records.columns)
    if missing_columns:
        raise RuntimeError(
            "Sharadar returned an unexpected sp500 schema; missing columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
    return sp500_records


def build_sp500_membership(
    sp500_records: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Reconstruct daily ``date, ticker`` membership from raw sp500 records.

    ``current`` rows anchor today's constituents. ``added``/``removed`` rows are
    treated as effective-dated changes (a ticker added on D is a member on D, one
    removed on D is not); those on/after ``start_date`` are reversed from the
    anchor, then replayed forward. ``historical`` snapshot rows are ignored.
    Removals are applied before additions on the same date.
    """
    records = sp500_records[["date", "action", "ticker"]].copy()
    records["action"] = records["action"].astype(str).str.strip().str.lower()
    records["ticker"] = records["ticker"].astype(str).str.strip()
    records["date"] = pd.to_datetime(records["date"], errors="coerce").dt.normalize()
    records = records.dropna(subset=["date"])

    unknown_actions = set(records["action"]) - _KNOWN_SP500_ACTIONS
    if unknown_actions:
        warnings.warn(f"Ignoring unrecognised sp500 actions: {sorted(unknown_actions)}")

    constituents = set(records.loc[records["action"].eq("current"), "ticker"])
    if len(constituents) < _MIN_EXPECTED_CONSTITUENTS:
        warnings.warn(
            f"Only {len(constituents)} current S&P 500 constituents returned; the "
            "subscription may be a restricted sample, not the full index."
        )
    dated_changes = records.loc[
        records["action"].isin(["added", "removed"]) & records["date"].ge(start_date)
    ]
    changes_by_date = {
        change_date: (
            set(changes.loc[changes["action"].eq("added"), "ticker"]),
            set(changes.loc[changes["action"].eq("removed"), "ticker"]),
        )
        for change_date, changes in dated_changes.groupby("date")
    }

    for change_date in sorted(changes_by_date, reverse=True):
        added, removed = changes_by_date[change_date]
        constituents = (constituents - added) | removed

    membership_frames = []
    for calendar_date in pd.date_range(start_date, end_date, freq="D"):
        if calendar_date in changes_by_date:
            added, removed = changes_by_date[calendar_date]
            constituents = (constituents - removed) | added
        membership_frames.append(
            pd.DataFrame({"date": calendar_date, "ticker": sorted(constituents)})
        )
    return pd.concat(membership_frames, ignore_index=True)


def get_sp500_membership(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Return daily point-in-time S&P 500 membership as ``date, ticker`` rows."""
    start_date, end_date = _parse_date_range(start, end)
    return build_sp500_membership(_load_sp500_records(refresh), start_date, end_date)


def _year_windows(start: str, end: str, years: int) -> list[tuple[str, str]]:
    """Split ``[start, end]`` into consecutive, non-overlapping windows of ``years`` years."""
    windows, window_start, last = [], pd.Timestamp(start), pd.Timestamp(end)
    while window_start <= last:
        next_start = window_start + pd.DateOffset(years=years)
        window_end = min(next_start - pd.Timedelta(days=1), last)
        windows.append((window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
        window_start = next_start
    return windows


def get_delistings(end: str, refresh: bool = False) -> pd.DataFrame:
    """Return one ``ticker, date, reason`` row per delisting event in the ``actions`` table.

    ``reason`` is ``bankruptcy``, ``regulatory``, ``voluntary``, ``acquisition`` or ``other``
    (a plain ``delisted`` row). Sharadar does not document the action codes; they were read
    from sample rows (e.g. ``acquisitionby`` carries the acquirer in ``contraticker``). When
    several rows share a ticker and date, the highest-priority reason wins. ``end`` is explicit
    so the cache key is stable.

    The ``actions`` table silently returns only the first 10 years of a longer date range (and
    ``skip`` cannot reach the rest), so each action is requested in 5-year windows.
    """
    end_date = pd.Timestamp(end).strftime("%Y-%m-%d")
    frames = []
    for priority, action in enumerate(_DELISTING_REASONS):
        for window_start, window_end in _year_windows(_SP500_HISTORY_START, end_date, 5):
            rows = _fetch_table(
                "actions",
                refresh,
                action=action,
                **{"from": window_start, "to": window_end, "fields": "date,action,ticker"},
            )
            if not rows.empty:
                rows = rows.assign(priority=priority, reason=_DELISTING_REASONS[action])
                frames.append(rows[["ticker", "date", "reason", "priority"]])
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "reason"])
    events = pd.concat(frames, ignore_index=True)
    events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.normalize()
    events = events.dropna(subset=["date", "ticker"])
    events = events.sort_values("priority").drop_duplicates(["ticker", "date"])
    return events[["ticker", "date", "reason"]].sort_values(["date", "ticker"], ignore_index=True)


def get_universe_panel(
    start: str,
    end: str,
    fields: tuple[str, ...] | list[str] = ("open", "close", "closeadj", "volume"),
    refresh: bool = False,
    apply_membership: bool = True,
) -> dict[str, pd.DataFrame]:
    """Return one wide date x ticker frame per field, kept to S&P 500 members.

    Prices are only kept for dates on which the ticker was an index member;
    other cells are ``NaN``. All frames share the same index and columns.
    With ``apply_membership=False`` the full price history of every ticker that
    was a member at any point in ``[start, end]`` is returned instead, so that
    trailing windows can use pre-membership history (filter to members afterwards).
    ``closeadj`` is adjusted for splits, dividends and spinoffs, ``close`` for
    splits only, ``closeunadj`` is unadjusted, and ``volume`` is split adjusted
    (so pair it with ``close``, not ``closeadj``, for dollar volume).
    """
    fields = list(fields)
    unknown_fields = sorted(set(fields) - _ALLOWED_PANEL_FIELDS)
    if not fields or unknown_fields or len(set(fields)) != len(fields):
        valid_fields = ", ".join(sorted(_ALLOWED_PANEL_FIELDS))
        raise ValueError(
            f"fields must be a non-empty list of unique names from: {valid_fields}"
            + (f" (got unknown: {', '.join(unknown_fields)})" if unknown_fields else "")
        )

    start_date, end_date = _parse_date_range(start, end)
    membership = get_sp500_membership(start, end, refresh)
    empty_panel = {
        field: pd.DataFrame(index=pd.DatetimeIndex([], name="date")) for field in fields
    }
    if membership.empty:
        return empty_panel

    universe_tickers = sorted(membership["ticker"].unique())
    price_batches: list[pd.DataFrame] = []
    for ticker_batch in _batch_tickers(universe_tickers):
        batch_prices = _fetch_table(
            "stocks",
            refresh,
            ticker=",".join(ticker_batch),
            **{
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "fields": ",".join(["date", "ticker", *fields]),
            },
        )
        if not batch_prices.empty:
            price_batches.append(batch_prices)

    if not price_batches:
        return empty_panel

    daily_prices = pd.concat(price_batches, ignore_index=True)
    daily_prices["date"] = pd.to_datetime(daily_prices["date"], errors="coerce").dt.normalize()
    for field in fields:
        daily_prices[field] = pd.to_numeric(daily_prices[field], errors="coerce")
    daily_prices = daily_prices.dropna(subset=["date", "ticker"])
    daily_prices = daily_prices.drop_duplicates(subset=["date", "ticker"])

    if apply_membership:
        point_in_time_prices = daily_prices.merge(membership, on=["date", "ticker"], how="inner")
    else:
        point_in_time_prices = daily_prices
    panel: dict[str, pd.DataFrame] = {}
    for field in fields:
        wide_field = point_in_time_prices.pivot(index="date", columns="ticker", values=field)
        wide_field.index.name = "date"
        wide_field.columns.name = "ticker"
        panel[field] = wide_field.sort_index()
    return panel


def get_universe_prices(
    start: str,
    end: str,
    price_field: str = "closeadj",
    refresh: bool = False,
) -> pd.DataFrame:
    """Return S&P 500 EOD prices filtered to the constituents on each date."""
    if price_field not in _ALLOWED_PRICE_FIELDS:
        valid_fields = ", ".join(sorted(_ALLOWED_PRICE_FIELDS))
        raise ValueError(f"price_field must be one of: {valid_fields}")
    return get_universe_panel(start, end, (price_field,), refresh)[price_field]
