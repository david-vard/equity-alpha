import pandas as pd
import pytest

from src import data as data_api

sharadar = data_api
build = data_api.build_sp500_membership
TS = pd.Timestamp

pytestmark = pytest.mark.filterwarnings("ignore:Only .* current S&P 500")


def _records(rows):
    return pd.DataFrame(rows, columns=["date", "action", "ticker"])


def _members(frame, day):
    return set(frame.loc[frame.date == TS(day), "ticker"])


def test_add_and_remove_effective_dates():
    records = _records(
        [
            ("2024-01-01", "current", "AAA"),
            ("2024-01-01", "current", "NEW"),
            ("2024-01-03", "added", "NEW"),
            ("2024-01-03", "removed", "OLD"),
        ]
    )
    m = build(records, TS("2024-01-01"), TS("2024-01-05"))
    assert _members(m, "2024-01-02") == {"AAA", "OLD"}
    assert _members(m, "2024-01-03") == {"AAA", "NEW"}
    assert _members(m, "2024-01-05") == {"AAA", "NEW"}


def test_changes_before_start_are_not_unwound():
    records = _records(
        [
            ("2024-01-01", "current", "AAA"),
            ("2023-06-01", "added", "AAA"),
            ("2023-07-01", "removed", "OLD"),
        ]
    )
    m = build(records, TS("2024-01-01"), TS("2024-01-02"))
    assert _members(m, "2024-01-01") == {"AAA"}


def test_same_day_remove_and_readd_stays_member():
    records = _records(
        [
            ("2024-01-01", "current", "X"),
            ("2024-01-02", "removed", "X"),
            ("2024-01-02", "added", "X"),
        ]
    )
    m = build(records, TS("2024-01-01"), TS("2024-01-03"))
    for day in ("2024-01-01", "2024-01-02", "2024-01-03"):
        assert _members(m, day) == {"X"}


def test_historical_rows_ignored():
    records = _records([("2024-01-01", "current", "AAA"), ("1999-01-01", "historical", "ZZZ")])
    m = build(records, TS("2024-01-01"), TS("2024-01-01"))
    assert _members(m, "2024-01-01") == {"AAA"}


def test_bad_inputs():
    with pytest.raises(ValueError):
        data_api.get_universe_prices("2024-02-01", "2024-01-01")
    with pytest.raises(ValueError):
        data_api.get_universe_prices("2024-01-01", "2024-02-01", price_field="open")


class _FakeResponse:
    def __init__(self, content=b"", status_code=200):
        self.content, self.status_code = content, status_code
        self.ok = status_code < 400


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    monkeypatch.setattr(sharadar, "_RAW_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(sharadar, "_MIN_SECONDS_BETWEEN_REQUESTS", 0)
    monkeypatch.setenv("SHARADAR_API_KEY", "SECRET123")
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, dict(params)))
        if url.endswith("/sp500"):
            return _FakeResponse(
                b"date,action,ticker\n2024-01-01,current,AAA\n2024-01-03,added,AAA\n"
            )
        values = {"closeadj": (10, 11), "close": (20, 22), "volume": (1000, 2000)}
        columns = params["fields"].split(",")
        rows = [
            ",".join([day, "AAA", *(str(values[c][i]) for c in columns[2:])])
            for i, day in enumerate(["2024-01-02", "2024-01-03"])
        ]
        return _FakeResponse(("\n".join([params["fields"], *rows]) + "\n").encode())

    monkeypatch.setattr(sharadar.requests, "get", fake_get)
    return calls


def test_sp500_requests_full_history_and_cache_hits(fake_api):
    data_api.get_sp500_membership("2024-01-01", "2024-01-03")
    _, params = fake_api[0]
    assert params["from"] == "1998-01-01"
    assert "to" in params and params["format"] == "csv"
    n_calls = len(fake_api)
    data_api.get_sp500_membership("2024-01-01", "2024-01-03")
    assert len(fake_api) == n_calls


def test_universe_prices_filtered_to_membership(fake_api):
    prices = data_api.get_universe_prices("2024-01-01", "2024-01-03")
    # AAA is added on 01-03, so unwound out of the 01-02 membership.
    assert list(prices.index) == [TS("2024-01-03")]
    assert prices.loc[TS("2024-01-03"), "AAA"] == 11


def test_http_error_does_not_leak_key(tmp_path, monkeypatch):
    monkeypatch.setattr(sharadar, "_RAW_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(sharadar, "_MIN_SECONDS_BETWEEN_REQUESTS", 0)
    monkeypatch.setenv("SHARADAR_API_KEY", "SECRET123")
    monkeypatch.setattr(
        sharadar.requests, "get", lambda *a, **k: _FakeResponse(status_code=403)
    )
    with pytest.raises(RuntimeError) as error:
        data_api.get_sp500_membership("2024-01-01", "2024-01-02")
    assert "SECRET123" not in str(error.value)
    assert not list(tmp_path.rglob("*.csv"))


def test_ticker_batches_fit_api_limit():
    tickers = [f"T{i:03d}" for i in range(300)]
    batches = sharadar._batch_tickers(tickers)
    assert [t for b in batches for t in b] == tickers
    assert all(len(",".join(b)) <= 200 and len(b) <= 30 for b in batches)


def test_small_universe_warns():
    records = _records([("2024-01-01", "current", "AAA")])
    with pytest.warns(UserWarning, match="restricted sample"):
        build(records, TS("2024-01-01"), TS("2024-01-01"))


def test_panel_fields_aligned_and_filtered_to_membership(fake_api):
    panel = data_api.get_universe_panel("2024-01-01", "2024-01-03")
    assert list(panel) == ["closeadj", "close", "volume"]
    sent_fields = [p["fields"] for url, p in fake_api if url.endswith("/stocks")]
    assert sent_fields == ["date,ticker,closeadj,close,volume"]
    for frame in panel.values():
        assert frame.index.equals(panel["close"].index)
        assert frame.columns.equals(panel["close"].columns)
        assert list(frame.index) == [TS("2024-01-03")]
    assert panel["volume"].loc[TS("2024-01-03"), "AAA"] == 2000
    assert panel["close"].loc[TS("2024-01-03"), "AAA"] == 22


def test_single_field_panel_equals_universe_prices(fake_api):
    panel = data_api.get_universe_panel("2024-01-01", "2024-01-03", ("close",))
    prices = data_api.get_universe_prices("2024-01-01", "2024-01-03", "close")
    pd.testing.assert_frame_equal(panel["close"], prices)


@pytest.mark.parametrize("fields", [[], ["bogus"], ["close", "close"], ["close", "lastupdated"]])
def test_panel_rejects_bad_fields(fields):
    with pytest.raises(ValueError):
        data_api.get_universe_panel("2024-01-01", "2024-01-03", fields)
