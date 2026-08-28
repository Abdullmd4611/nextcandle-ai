import time
import requests
import pandas as pd


# ============================================================
# NextCandle AI — Bybit Market Data Engine V2
# Primary market: Bybit USDT Perpetual
# Primary timeframe: 15 minutes
# Higher timeframe: 4 hours
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
    """
    Reliable Bybit HTTP request with retry handling.
    """

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
    """
    Fetch one Bybit kline page.

    Bybit returns newest candles first.
    """

    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": interval,
        "start": start_ms,
        "end": end_ms,
        "limit": limit,
    }

    url = (
        f"{BASE_URL}/v5/market/kline"
    )

    payload = _request(
        url,
        params,
    )

    result = payload.get(
        "result",
        {}
    )

    rows = result.get(
        "list",
        []
    )

    return rows


def fetch_klines(
    symbol=DEFAULT_SYMBOL,
    interval="Min15",
    total=1000,
):
    """
    Download completed Bybit USDT perpetual candles.

    Parameters
    ----------
    symbol:
        Bybit symbol, e.g. CYSUSDT.

    interval:
        Min15 or Hour4.

    total:
        Approximate number of completed candles required.

    Returns
    -------
    pandas.DataFrame
        Columns:

        timestamp
        open
        high
        low
        close
        volume
        turnover
    """

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

    bybit_interval = INTERVALS[
        interval
    ]

    candle_seconds = _interval_seconds(
        interval
    )

    now_ms = int(
        time.time() * 1000
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # End at the beginning of the currently
    # forming candle so the live candle is excluded.
    # --------------------------------------------------------

    current_period_start_ms = (
        int(time.time())
        // candle_seconds
    ) * candle_seconds * 1000

    end_ms = (
        current_period_start_ms - 1
    )

    start_ms = (
        end_ms
        - (
            total
            * candle_seconds
            * 1000
        )
    )

    all_rows = []

    cursor_end = end_ms

    # Bybit allows a maximum of 1000 candles/page.
    #
    # We continue backwards until we have enough data.
    while len(all_rows) < total:

        rows = _fetch_page(
            symbol=symbol,
            interval=bybit_interval,
            start_ms=start_ms,
            end_ms=cursor_end,
            limit=min(
                1000,
                total - len(all_rows) + 50,
            ),
        )

        if not rows:
            break

        all_rows.extend(rows)

        # Bybit returns newest first.
        timestamps = [
            int(row[0])
            for row in rows
        ]

        oldest_timestamp = min(
            timestamps
        )

        next_end = (
            oldest_timestamp - 1
        )

        if next_end >= cursor_end:
            break

        cursor_end = next_end

        # Safety stop.
        if cursor_end < start_ms:
            break

        time.sleep(0.15)

    if not all_rows:
        raise RuntimeError(
            "No Bybit candle data returned."
        )

    # --------------------------------------------------------
    # BYBIT KLINE FORMAT
    #
    # [0] startTime
    # [1] open
    # [2] high
    # [3] low
    # [4] close
    # [5] volume
    # [6] turnover
    # --------------------------------------------------------

    parsed = []

    for row in all_rows:

        if len(row) < 7:
            continue

        parsed.append(
            {
                "timestamp": pd.to_datetime(
                    int(row[0]),
                    unit="ms",
                    utc=True,
                ),
                "open": pd.to_numeric(
                    row[1],
                    errors="coerce",
                ),
                "high": pd.to_numeric(
                    row[2],
                    errors="coerce",
                ),
                "low": pd.to_numeric(
                    row[3],
                    errors="coerce",
                ),
                "close": pd.to_numeric(
                    row[4],
                    errors="coerce",
                ),
                "volume": pd.to_numeric(
                    row[5],
                    errors="coerce",
                ),
                "turnover": pd.to_numeric(
                    row[6],
                    errors="coerce",
                ),
            }
        )

    if not parsed:
        raise RuntimeError(
            "Bybit returned unusable candle data."
        )

    df = pd.DataFrame(parsed)

    # --------------------------------------------------------
    # CLEAN DATA
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
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # VALIDATE OHLC
    # --------------------------------------------------------

    valid_ohlc = (
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

    df = df[
        valid_ohlc
    ].copy()

    # --------------------------------------------------------
    # FINAL PROTECTION AGAINST LIVE CANDLE
    # --------------------------------------------------------

    current_period_start = pd.to_datetime(
        current_period_start_ms,
        unit="ms",
        utc=True,
    )

    df = df[
        df["timestamp"]
        < current_period_start
    ].copy()

    # Keep exactly the requested amount,
    # using the newest completed candles.
    df = (
        df
        .sort_values("timestamp")
        .tail(total)
        .reset_index(drop=True)
    )

    if len(df) < 100:
        raise RuntimeError(
            "Not enough completed Bybit candle "
            f"data returned. Got {len(df)}."
        )

    return df


def get_latest_price(
    symbol=DEFAULT_SYMBOL,
):
    """
    Get the latest Bybit traded price.

    This is informational only.
    The prediction engine should use completed candles.
    """

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

    url = (
        f"{BASE_URL}/v5/market/tickers"
    )

    payload = _request(
        url,
        params,
    )

    result = payload.get(
        "result",
        {}
    )

    rows = result.get(
        "list",
        []
    )

    if not rows:
        raise RuntimeError(
            "No Bybit ticker data returned."
        )

    last_price = rows[0].get(
        "lastPrice"
    )

    if last_price is None:
        raise RuntimeError(
            "Bybit did not return last price."
        )

    return float(last_price)
