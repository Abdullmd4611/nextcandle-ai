import time
import requests
import pandas as pd


# =========================================================
# NextCandle AI — MEXC FUTURES DATA ENGINE V3.1
#
# Market:
#     CYSUSDT.P
#
# MEXC contract symbol:
#     CYS_USDT
#
# Prediction:
#     NEXT 15-MINUTE CANDLE
#
# Evidence:
#     1M + 5M + 15M + 4H
#
# IMPORTANT:
#     Only COMPLETED candles are returned.
#
#     The currently forming candle is NEVER included.
# =========================================================


BASE_URL = "https://contract.mexc.com"


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

    "User-Agent":
        "NextCandleAI/3.1",

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
# REQUEST MEXC FUTURES DATA
# =========================================================

def _request(
    symbol,
    interval,
    start,
    end,
    retries=3,
):

    url = (
        f"{BASE_URL}/api/v1/contract/kline/"
        f"{symbol}"
    )

    params = {

        "interval":
            interval,

        "start":
            int(start),

        "end":
            int(end),
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
                    "MEXC returned error "
                    f"{payload.get('code')}: "
                    f"{payload.get('message', 'Unknown error')}"
                )

            data = payload.get("data")

            if not data:

                return pd.DataFrame()

            if not data.get("time"):

                return pd.DataFrame()

            frame = pd.DataFrame({

                "timestamp":
                    data["time"],

                "open":
                    data["open"],

                "high":
                    data["high"],

                "low":
                    data["low"],

                "close":
                    data["close"],

                "volume":
                    data["vol"],

                "turnover":
                    data["amount"],
            })

            return frame

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
    # MEXC contract timestamps are Unix seconds.
    # -----------------------------------------------------

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

    if frame is None or frame.empty:

        return frame

    seconds = INTERVAL_SECONDS[
        interval
    ]

    now = int(
        time.time()
    )

    # -----------------------------------------------------
    # Beginning of the CURRENT candle.
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
    # Keep ONLY candles that started before the
    # currently forming candle.
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
    symbol="CYS_USDT",
    interval="Min15",
    total=2500,
):

    symbol = (
        symbol
        .upper()
        .strip()
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

    max_per_request = 2000

    # -----------------------------------------------------
    # Request enough history to obtain the desired number
    # of COMPLETED candles.
    # -----------------------------------------------------

    target_start = (
        now
        - (
            total * seconds
        )
        - seconds
    )

    chunks = []

    cursor_end = now

    request_count = 0

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

        request_count += 1

        if not chunk.empty:

            chunks.append(
                chunk
            )

        # -------------------------------------------------
        # Move backwards.
        #
        # Subtract one second to avoid repeatedly requesting
        # the same boundary candle.
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
    symbol="CYS_USDT",
    history_15m=2500,
):

    symbol = (
        symbol
        .upper()
        .strip()
    )

    candles_15m = int(
        history_15m
    )

    # -----------------------------------------------------
    # 5M:
    #
    # Three 5M candles per 15M candle.
    # Extra history is added for indicators.
    # -----------------------------------------------------

    candles_5m = int(
        candles_15m * 3
        + 300
    )

    # -----------------------------------------------------
    # 1M:
    #
    # Fifteen 1M candles per 15M candle.
    # Extra history is added for indicators.
    # -----------------------------------------------------

    candles_1m = int(
        candles_15m * 15
        + 1500
    )

    # -----------------------------------------------------
    # 4H:
    #
    # Sixteen 15M candles per 4H candle.
    # At least 300 candles are requested so that the
    # 100-period 4H indicators have sufficient history.
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

        # -------------------------------------------------
        # Required columns.
        # -------------------------------------------------

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
        # OHLC relationship validation.
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
    # Make sure the latest candle returned for each
    # timeframe is actually completed.
    # -----------------------------------------------------

    now = int(
        time.time()
    )

    for timeframe, frame in data.items():

        if timeframe == "1m":

            seconds = 60

        elif timeframe == "5m":

            seconds = 5 * 60

        elif timeframe == "15m":

            seconds = 15 * 60

        elif timeframe == "4h":

            seconds = 4 * 60 * 60

        else:

            continue

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
    symbol="CYS_USDT",
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
# LATEST PRICE
# =========================================================

def get_latest_price(
    symbol="CYS_USDT",
):

    symbol = (
        symbol
        .upper()
        .strip()
    )

    url = (
        f"{BASE_URL}/api/v1/contract/ticker"
    )

    params = {
        "symbol": symbol,
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

            if not payload.get("success"):

                raise RuntimeError(
                    "MEXC ticker request failed: "
                    f"{payload.get('message', 'Unknown error')}"
                )

            data = payload.get(
                "data"
            )

            if not data:

                raise RuntimeError(
                    "MEXC returned empty ticker data."
                )

            price = data.get(
                "lastPrice"
            )

            if price is None:

                raise RuntimeError(
                    "MEXC ticker response "
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
