import time
import requests
import pandas as pd


# =========================================================
# NextCandle AI — BYBIT FUTURES DATA ENGINE V4.0
#
# Market:
#     Bybit USDT Perpetual
#
# Primary:
#     CYSUSDT
#
# Prediction:
#     NEXT 15-MINUTE CANDLE
#
# Evidence:
#     1M + 5M + 15M + 4H
#
# Bybit API:
#     V5 Public Market API
#     category = linear
#
# IMPORTANT:
#     Only COMPLETED candles are returned.
#
#     The currently forming candle is NEVER included.
# =========================================================


BASE_URL = "https://api.bybit.com"

CATEGORY = "linear"


# =========================================================
# INTERVALS
# =========================================================

INTERVAL_SECONDS = {

    "Min1": 60,

    "Min5": 5 * 60,

    "Min15": 15 * 60,

    "Hour4": 4 * 60 * 60,
}


BYBIT_INTERVALS = {

    "Min1": "1",

    "Min5": "5",

    "Min15": "15",

    "Hour4": "240",
}


# =========================================================
# SESSION
# =========================================================

SESSION = requests.Session()

SESSION.headers.update({

    "User-Agent":
        "NextCandleAI/4.0",

    "Accept":
        "application/json",
})


# =========================================================
# VALIDATE INTERVAL
# =========================================================

def _validate_interval(interval):

    if interval not in INTERVAL_SECONDS:

        raise ValueError(
            f"Unsupported interval: {interval}. "
            f"Supported intervals: "
            f"{', '.join(INTERVAL_SECONDS.keys())}"
        )


# =========================================================
# NORMALIZE SYMBOL
# =========================================================

def _normalize_symbol(symbol):

    if symbol is None:

        raise ValueError(
            "Symbol cannot be None."
        )

    symbol = (
        str(symbol)
        .upper()
        .strip()
    )

    # -----------------------------------------------------
    # Allow users to type:
    #
    # CYSUSDT
    # CYSUSDT.P
    # CYS_USDT
    #
    # Bybit linear uses:
    #
    # CYSUSDT
    # -----------------------------------------------------

    symbol = symbol.replace(
        ".P",
        "",
    )

    symbol = symbol.replace(
        "_",
        "",
    )

    if not symbol:

        raise ValueError(
            "Symbol cannot be empty."
        )

    return symbol


# =========================================================
# BYBIT V5 REQUEST
# =========================================================

def _request(
    symbol,
    interval,
    start,
    end,
    retries=3,
):

    url = (
        f"{BASE_URL}/v5/market/kline"
    )

    params = {

        "category":
            CATEGORY,

        "symbol":
            symbol,

        "interval":
            BYBIT_INTERVALS[
                interval
            ],

        "start":
            int(start),

        "end":
            int(end),

        "limit":
            1000,
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

            ret_code = payload.get(
                "retCode"
            )

            if ret_code != 0:

                raise RuntimeError(
                    "Bybit returned error "
                    f"{ret_code}: "
                    f"{payload.get('retMsg', 'Unknown error')}"
                )

            result = payload.get(
                "result",
                {},
            )

            data = result.get(
                "list",
                [],
            )

            if not data:

                return pd.DataFrame()

            rows = []

            for candle in data:

                if len(candle) < 7:

                    continue

                rows.append({

                    "timestamp":
                        candle[0],

                    "open":
                        candle[1],

                    "high":
                        candle[2],

                    "low":
                        candle[3],

                    "close":
                        candle[4],

                    "volume":
                        candle[5],

                    "turnover":
                        candle[6],
                })

            if not rows:

                return pd.DataFrame()

            return pd.DataFrame(
                rows
            )

        except Exception as exc:

            last_error = exc

            if attempt < retries - 1:

                time.sleep(
                    1.5 * (attempt + 1)
                )

    raise RuntimeError(
        f"Unable to download "
        f"{symbol} {interval} candles: "
        f"{last_error}"
    )


# =========================================================
# NORMALIZE DATA
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

    # -----------------------------------------------------
    # Bybit V5 timestamps are milliseconds.
    # -----------------------------------------------------

    x["timestamp"] = pd.to_datetime(
        pd.to_numeric(
            x["timestamp"],
            errors="coerce",
        ),
        unit="ms",
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

    if frame is None or frame.empty:

        return frame

    seconds = INTERVAL_SECONDS[
        interval
    ]

    now = int(
        time.time()
    )

    # -----------------------------------------------------
    # Start of the CURRENT candle.
    # -----------------------------------------------------

    current_period_start = (
        now // seconds
    ) * seconds

    current_period_start = pd.to_datetime(
        current_period_start,
        unit="s",
        utc=True,
    )

    # -----------------------------------------------------
    # ONLY candles that started before the current candle
    # are completed.
    # -----------------------------------------------------

    completed = frame[
        frame["timestamp"]
        < current_period_start
    ].copy()

    return (
        completed
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# =========================================================
# FETCH HISTORICAL CANDLES
# =========================================================

def fetch_klines(
    symbol="CYSUSDT",
    interval="Min15",
    total=2500,
):

    symbol = _normalize_symbol(
        symbol
    )

    _validate_interval(
        interval
    )

    if total < 100:

        raise ValueError(
            "total must be at least "
            "100 candles."
        )

    seconds = INTERVAL_SECONDS[
        interval
    ]

    now = int(
        time.time()
    )

    # -----------------------------------------------------
    # Bybit maximum kline page size is 1000.
    # -----------------------------------------------------

    max_per_request = 1000

    # -----------------------------------------------------
    # Ask for slightly more history than necessary because
    # the current forming candle will be removed later.
    # -----------------------------------------------------

    target_start = (
        now
        - (
            (total + 2)
            * seconds
        )
    )

    target_start = (
        target_start // seconds
    ) * seconds

    cursor_end = now

    chunks = []

    request_count = 0

    while cursor_end > target_start:

        cursor_start = max(
            target_start,
            cursor_end
            - (
                max_per_request
                * seconds
            )
            + 1,
        )

        chunk = _request(

            symbol=symbol,

            interval=interval,

            start=cursor_start,

            end=cursor_end,
        )

        request_count += 1

        if not chunk.empty:

            chunks.append(
                chunk
            )

        # -------------------------------------------------
        # Move backwards.
        # -------------------------------------------------

        cursor_end = (
            cursor_start - 1
        )

        if request_count > 100:

            raise RuntimeError(
                "Historical download required "
                "an unexpectedly large number "
                "of requests."
            )

        time.sleep(
            0.05
        )

    if not chunks:

        raise RuntimeError(
            f"No {interval} candles returned "
            f"for {symbol}."
        )

    frame = pd.concat(
        chunks,
        ignore_index=True,
    )

    frame = _normalize(
        frame
    )

    # -----------------------------------------------------
    # CRITICAL:
    #
    # Remove the currently forming candle.
    # -----------------------------------------------------

    frame = _remove_forming_candle(
        frame,
        interval,
    )

    # -----------------------------------------------------
    # Keep only requested number.
    # -----------------------------------------------------

    if len(frame) > total:

        frame = (
            frame
            .tail(total)
            .reset_index(drop=True)
        )

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
    symbol="CYSUSDT",
    history_15m=2500,
):

    symbol = _normalize_symbol(
        symbol
    )

    candles_15m = int(
        history_15m
    )

    # -----------------------------------------------------
    # 5M:
    #
    # 3 x 5M candles = 1 x 15M candle.
    # -----------------------------------------------------

    candles_5m = int(
        candles_15m * 3
        + 300
    )

    # -----------------------------------------------------
    # 1M:
    #
    # 15 x 1M candles = 1 x 15M candle.
    # -----------------------------------------------------

    candles_1m = int(
        candles_15m * 15
        + 1500
    )

    # -----------------------------------------------------
    # 4H:
    #
    # 16 x 15M candles = 1 x 4H candle.
    # -----------------------------------------------------

    candles_4h = max(
        300,
        int(
            candles_15m / 16
        )
        + 100,
    )

    data = {}

    # =====================================================
    # 1 MINUTE
    # =====================================================

    data["1m"] = fetch_klines(

        symbol=symbol,

        interval="Min1",

        total=candles_1m,
    )

    # =====================================================
    # 5 MINUTE
    # =====================================================

    data["5m"] = fetch_klines(

        symbol=symbol,

        interval="Min5",

        total=candles_5m,
    )

    # =====================================================
    # 15 MINUTE
    # =====================================================

    data["15m"] = fetch_klines(

        symbol=symbol,

        interval="Min15",

        total=candles_15m,
    )

    # =====================================================
    # 4 HOUR
    # =====================================================

    data["4h"] = fetch_klines(

        symbol=symbol,

        interval="Hour4",

        total=candles_4h,
    )

    validate_timeframe_data(
        data
    )

    return data


# =========================================================
# DATA QUALITY CHECK
# =========================================================

def validate_timeframe_data(
    data,
):

    if data is None:

        raise ValueError(
            "Market data is None."
        )

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
            "Missing timeframes: "
            f"{sorted(missing)}"
        )

    required_columns = [

        "timestamp",

        "open",

        "high",

        "low",

        "close",

        "volume",
    ]

    for timeframe in required:

        frame = data[
            timeframe
        ]

        if frame is None or frame.empty:

            raise ValueError(
                f"{timeframe} data is empty."
            )

        missing_columns = [

            column
            for column in required_columns
            if column not in frame.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{timeframe} is missing "
                f"columns: "
                f"{missing_columns}"
            )

        # -------------------------------------------------
        # Timestamp validation.
        # -------------------------------------------------

        if not pd.api.types.is_datetime64_any_dtype(
            frame["timestamp"]
        ):

            raise ValueError(
                f"{timeframe} timestamp column "
                f"is not datetime."
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

        # -------------------------------------------------
        # Price validation.
        # -------------------------------------------------

        price_columns = [

            "open",

            "high",

            "low",

            "close",
        ]

        if (
            frame[
                price_columns
            ]
            <= 0
        ).any().any():

            raise ValueError(
                f"{timeframe} contains "
                f"invalid non-positive prices."
            )

        # -------------------------------------------------
        # OHLC validation.
        # -------------------------------------------------

        invalid_ohlc = (

            (frame["high"] < frame["low"])

            | (frame["high"] < frame["open"])

            | (frame["high"] < frame["close"])

            | (frame["low"] > frame["open"])

            | (frame["low"] > frame["close"])
        )

        if invalid_ohlc.any():

            raise ValueError(
                f"{timeframe} contains "
                f"invalid OHLC relationships."
            )

        # -------------------------------------------------
        # Volume validation.
        # -------------------------------------------------

        if (
            frame["volume"]
            < 0
        ).any():

            raise ValueError(
                f"{timeframe} contains "
                f"negative volume."
            )

    # -----------------------------------------------------
    # Verify latest candle is completed.
    # -----------------------------------------------------

    now = int(
        time.time()
    )

    seconds_map = {

        "1m": 60,

        "5m": 5 * 60,

        "15m": 15 * 60,

        "4h": 4 * 60 * 60,
    }

    for timeframe, frame in data.items():

        seconds = seconds_map[
            timeframe
        ]

        current_period_start = (
            now // seconds
        ) * seconds

        current_period_start = pd.to_datetime(
            current_period_start,
            unit="s",
            utc=True,
        )

        latest_timestamp = frame[
            "timestamp"
        ].iloc[-1]

        if latest_timestamp >= current_period_start:

            raise ValueError(
                f"{timeframe} contains "
                f"a currently forming candle."
            )

    return True


# =========================================================
# LATEST COMPLETED CANDLE
# =========================================================

def get_latest_completed_candle(
    symbol="CYSUSDT",
    interval="Min15",
):

    frame = fetch_klines(

        symbol=symbol,

        interval=interval,

        total=200,
    )

    if frame.empty:

        raise RuntimeError(
            "No completed candles available."
        )

    return frame.iloc[-1].copy()


# =========================================================
# LATEST BYBIT PRICE
# =========================================================

def get_latest_price(
    symbol="CYSUSDT",
):

    symbol = _normalize_symbol(
        symbol
    )

    url = (
        f"{BASE_URL}/v5/market/tickers"
    )

    params = {

        "category":
            CATEGORY,

        "symbol":
            symbol,
    }

    last_error = None

    for attempt in range(3):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=15,
            )

            response.raise_for_status()

            payload = response.json()

            ret_code = payload.get(
                "retCode"
            )

            if ret_code != 0:

                raise RuntimeError(
                    "Bybit ticker request failed: "
                    f"{ret_code} — "
                    f"{payload.get('retMsg', 'Unknown error')}"
                )

            result = payload.get(
                "result",
                {},
            )

            data = result.get(
                "list",
                [],
            )

            if not data:

                raise RuntimeError(
                    "Bybit returned empty ticker data."
                )

            price = data[0].get(
                "lastPrice"
            )

            if price is None:

                raise RuntimeError(
                    "Bybit ticker response "
                    "does not contain lastPrice."
                )

            return float(
                price
            )

        except Exception as exc:

            last_error = exc

            if attempt < 2:

                time.sleep(
                    1.0 * (attempt + 1)
                )

    raise RuntimeError(
        f"Unable to get latest price "
        f"for {symbol}: {last_error}"
    )
