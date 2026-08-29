import time
import requests
import pandas as pd


# =========================================================
# MEXC FUTURES DATA ENGINE
# =========================================================
#
# NextCandle AI
#
# Target:
#   Predict the NEXT completed 15-minute candle.
#
# Evidence:
#   1M + 5M + 15M + 4H + historical data
#
# IMPORTANT:
#   Only COMPLETED candles are returned.
#   The currently forming candle is never used.
# =========================================================


BASE_URL = "https://api.mexc.com"


INTERVAL_SECONDS = {
    "Min1": 60,
    "Min5": 5 * 60,
    "Min15": 15 * 60,
    "Hour4": 4 * 60 * 60,
}


# =========================================================
# SESSION
# =========================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "NextCandleAI/2.0",
    "Accept": "application/json",
})


# =========================================================
# HELPERS
# =========================================================

def _validate_interval(interval):

    if interval not in INTERVAL_SECONDS:

        raise ValueError(
            f"Unsupported interval: {interval}. "
            f"Supported intervals: "
            f"{', '.join(INTERVAL_SECONDS.keys())}"
        )


def _request(
    symbol,
    interval,
    start,
    end,
    retries=3
):

    url = (
        f"{BASE_URL}/api/v1/contract/kline/{symbol}"
    )

    params = {
        "interval": interval,
        "start": int(start),
        "end": int(end),
    }

    last_error = None

    for attempt in range(retries):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=20,
            )

            response.raise_for_status()

            payload = response.json()

            if not payload.get("success"):

                raise RuntimeError(
                    f"MEXC returned error "
                    f"{payload.get('code')}: "
                    f"{payload.get('message', 'Unknown error')}"
                )

            data = payload.get("data")

            if not data:

                return pd.DataFrame()

            if not data.get("time"):

                return pd.DataFrame()

            frame = pd.DataFrame({
                "timestamp": data["time"],
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"],
                "volume": data["vol"],
                "turnover": data["amount"],
            })

            return frame

        except Exception as exc:

            last_error = exc

            if attempt < retries - 1:

                time.sleep(
                    1.5 * (attempt + 1)
                )

    raise RuntimeError(
        f"Unable to download {symbol} "
        f"{interval} candles: {last_error}"
    )


# =========================================================
# NORMALIZE
# =========================================================

def _normalize(frame):

    if frame is None or frame.empty:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ]
        )

    x = frame.copy()

    x["timestamp"] = pd.to_datetime(
        x["timestamp"],
        unit="s",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]

    for column in numeric_columns:

        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    x = (
        x
        .sort_values("timestamp")
        .drop_duplicates(
            "timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    x = (
        x
        .replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )
        .dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .reset_index(drop=True)
    )

    return x


# =========================================================
# REMOVE FORMING CANDLE
# =========================================================

def _remove_forming_candle(
    frame,
    interval,
):

    if frame.empty:

        return frame

    seconds = INTERVAL_SECONDS[
        interval
    ]

    now = int(time.time())

    current_period_start = (
        now // seconds
    ) * seconds

    current_period_start = pd.to_datetime(
        current_period_start,
        unit="s",
        utc=True,
    )

    return frame[
        frame["timestamp"]
        < current_period_start
    ].copy().reset_index(drop=True)


# =========================================================
# FETCH HISTORICAL CANDLES
# =========================================================

def fetch_klines(
    symbol="CYS_USDT",
    interval="Min15",
    total=2500,
):

    symbol = (
        symbol
        .upper()
        .strip()
    )

    _validate_interval(interval)

    if total < 100:

        raise ValueError(
            "total must be at least 100 candles."
        )

    seconds = INTERVAL_SECONDS[
        interval
    ]

    now = int(time.time())

    # -----------------------------------------------------
    # We request in chunks instead of assuming MEXC will
    # return thousands of candles in one request.
    # -----------------------------------------------------

    max_per_request = 2000

    target_start = (
        now
        - (total * seconds)
    )

    chunks = []

    cursor_end = now

    while cursor_end > target_start:

        cursor_start = max(
            target_start,
            cursor_end
            - (
                max_per_request
                * seconds
            ),
        )

        chunk = _request(
            symbol=symbol,
            interval=interval,
            start=cursor_start,
            end=cursor_end,
        )

        if not chunk.empty:

            chunks.append(chunk)

        # Move backwards.
        #
        # Subtract one second so the boundary candle
        # isn't repeatedly requested.

        cursor_end = (
            cursor_start - 1
        )

        if len(chunks) > 100:

            raise RuntimeError(
                "Historical download required "
                "an unexpectedly large number "
                "of requests."
            )

        time.sleep(0.05)

    if not chunks:

        raise RuntimeError(
            f"No {interval} candles returned "
            f"for {symbol}."
        )

    frame = pd.concat(
        chunks,
        ignore_index=True,
    )

    frame = _normalize(frame)

    frame = _remove_forming_candle(
        frame,
        interval,
    )

    # Keep only requested amount.

    if len(frame) > total:

        frame = frame.tail(
            total
        ).reset_index(drop=True)

    if len(frame) < 100:

        raise RuntimeError(
            f"Only {len(frame)} completed "
            f"{interval} candles were returned "
            f"for {symbol}."
        )

    return frame


# =========================================================
# MULTI-TIMEFRAME DOWNLOAD
# =========================================================

def fetch_multi_timeframe(
    symbol="CYS_USDT",
    history_15m=2500,
):

    symbol = (
        symbol
        .upper()
        .strip()
    )

    # -----------------------------------------------------
    # We deliberately collect more lower-timeframe candles
    # than the 15M target requires.
    #
    # This gives the feature engine enough information to
    # study how each 15M candle actually developed.
    # -----------------------------------------------------

    candles_15m = int(
        history_15m
    )

    candles_5m = int(
        candles_15m * 3 + 300
    )

    candles_1m = int(
        candles_15m * 15 + 1500
    )

    candles_4h = max(
        300,
        int(candles_15m / 16) + 100,
    )

    data = {}

    data["1m"] = fetch_klines(
        symbol=symbol,
        interval="Min1",
        total=candles_1m,
    )

    data["5m"] = fetch_klines(
        symbol=symbol,
        interval="Min5",
        total=candles_5m,
    )

    data["15m"] = fetch_klines(
        symbol=symbol,
        interval="Min15",
        total=candles_15m,
    )

    data["4h"] = fetch_klines(
        symbol=symbol,
        interval="Hour4",
        total=candles_4h,
    )

    return data


# =========================================================
# DATA QUALITY CHECK
# =========================================================

def validate_timeframe_data(
    data,
):

    required = {
        "1m",
        "5m",
        "15m",
        "4h",
    }

    missing = (
        required
        - set(data.keys())
    )

    if missing:

        raise ValueError(
            f"Missing timeframes: "
            f"{sorted(missing)}"
        )

    for timeframe in required:

        frame = data[timeframe]

        if frame is None or frame.empty:

            raise ValueError(
                f"{timeframe} data is empty."
            )

        if not frame[
            "timestamp"
        ].is_monotonic_increasing:

            raise ValueError(
                f"{timeframe} timestamps "
                f"are not sorted."
            )

        if frame[
            "timestamp"
        ].duplicated().any():

            raise ValueError(
                f"{timeframe} contains "
                f"duplicate timestamps."
            )

        if (
            frame[
                ["open", "high", "low", "close"]
            ]
            <= 0
        ).any().any():

            raise ValueError(
                f"{timeframe} contains "
                f"invalid non-positive prices."
            )

    return True
