import time
import requests
import pandas as pd


# ============================================================
# NextCandle AI — BYBIT DATA ENGINE V2
# ============================================================

BASE_URL = "https://api.bybit.com"

CATEGORY = "linear"

DEFAULT_SYMBOL = "CYSUSDT"

INTERVALS = {
    "Min15": "15",
    "Hour4": "240",
}


def _interval_seconds(interval):
    mapping = {
        "Min15": 15 * 60,
        "Hour4": 4 * 60 * 60,
    }

    if interval not in mapping:
        raise ValueError(
            f"Unsupported interval: {interval}"
        )

    return mapping[interval]


def _request(url, params, retries=3):

    last_error = None

    for attempt in range(retries):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=20,
            )

            response.raise_for_status()

            payload = response.json()

            if payload.get("retCode") != 0:

                raise RuntimeError(
                    f"Bybit API error "
                    f"{payload.get('retCode')}: "
                    f"{payload.get('retMsg', 'Unknown error')}"
                )

            return payload

        except Exception as exc:

            last_error = exc

            if attempt < retries - 1:
                time.sleep(
                    1.5 * (attempt + 1)
                )

    raise RuntimeError(
        f"Bybit request failed after "
        f"{retries} attempts: {last_error}"
    )


def _fetch_page(
    symbol,
    interval,
    start_ms,
    end_ms,
    limit=1000,
):

    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": interval,
        "start": start_ms,
        "end": end_ms,
        "limit": limit,
    }

    payload = _request(
        f"{BASE_URL}/v5/market/kline",
        params,
    )

    return payload.get(
        "result",
        {}
    ).get(
        "list",
        []
    )


def fetch_klines(
    symbol=DEFAULT_SYMBOL,
    interval="Min15",
    total=1000,
):

    symbol = (
        symbol
        .upper()
        .strip()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
    )

    if interval not in INTERVALS:
        raise ValueError(
            f"Unsupported interval: {interval}"
        )

    if total < 100:
        raise ValueError(
            "total must be at least 100."
        )

    bybit_interval = INTERVALS[interval]

    candle_seconds = _interval_seconds(
        interval
    )

    # --------------------------------------------------------
    # EXCLUDE CURRENTLY FORMING CANDLE
    # --------------------------------------------------------

    now_seconds = int(time.time())

    current_period_start = (
        now_seconds // candle_seconds
    ) * candle_seconds

    end_ms = (
        current_period_start * 1000
    ) - 1

    start_ms = (
        end_ms
        - total * candle_seconds * 1000
    )

    all_rows = []

    cursor_end = end_ms

    # --------------------------------------------------------
    # PAGINATE BACKWARDS
    # --------------------------------------------------------

    while len(all_rows) < total:

        remaining = total - len(all_rows)

        limit = min(
            1000,
            remaining + 20,
        )

        rows = _fetch_page(
            symbol=symbol,
            interval=bybit_interval,
            start_ms=start_ms,
            end_ms=cursor_end,
            limit=limit,
        )

        if not rows:
            break

        all_rows.extend(rows)

        oldest_timestamp = min(
            int(row[0])
            for row in rows
        )

        next_end = oldest_timestamp - 1

        if next_end >= cursor_end:
            break

        cursor_end = next_end

        if cursor_end < start_ms:
            break

        time.sleep(0.15)

    if not all_rows:
        raise RuntimeError(
            "No Bybit candle data returned."
        )

    # --------------------------------------------------------
    # PARSE BYBIT KLINES
    # --------------------------------------------------------

    records = []

    for row in all_rows:

        if len(row) < 7:
            continue

        records.append(
            {
                "timestamp": pd.to_datetime(
                    int(row[0]),
                    unit="ms",
                    utc=True,
                ),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "turnover": float(row[6]),
            }
        )

    if not records:
        raise RuntimeError(
            "Bybit returned unusable candle data."
        )

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # CLEAN + SORT
    # --------------------------------------------------------

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]

    df[numeric_columns] = (
        df[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    df = (
        df
        .replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )
        .dropna(
            subset=numeric_columns
        )
    )

    # --------------------------------------------------------
    # OHLC SANITY CHECK
    # --------------------------------------------------------

    valid = (
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    )

    df = df[valid].copy()

    # --------------------------------------------------------
    # SECOND LIVE-CANDLE PROTECTION
    # --------------------------------------------------------

    current_period_start = pd.to_datetime(
        current_period_start * 1000,
        unit="ms",
        utc=True,
    )

    df = df[
        df["timestamp"]
        < current_period_start
    ]

    df = (
        df
        .sort_values("timestamp")
        .tail(total)
        .reset_index(drop=True)
    )

    if len(df) < 100:
        raise RuntimeError(
            "Not enough completed Bybit candles. "
            f"Received {len(df)}."
        )

    # ========================================================
    # CRITICAL:
    #
    # Features.py expects timestamps to be the index.
    # ========================================================

    df = df.set_index(
        "timestamp"
    )

    df.index.name = "timestamp"

    return df


def get_latest_price(
    symbol=DEFAULT_SYMBOL,
):

    symbol = (
        symbol
        .upper()
        .strip()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
    )

    params = {
        "category": CATEGORY,
        "symbol": symbol,
    }

    payload = _request(
        f"{BASE_URL}/v5/market/tickers",
        params,
    )

    rows = (
        payload
        .get("result", {})
        .get("list", [])
    )

    if not rows:
        raise RuntimeError(
            "No Bybit ticker data returned."
        )

    price = rows[0].get(
        "lastPrice"
    )

    if price is None:
        raise RuntimeError(
            "Bybit did not return last price."
        )

    return float(price)
