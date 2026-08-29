import time
import requests
import pandas as pd


# ============================================================
# NextCandle AI — MEXC FUTURES DATA ENGINE
#
# MARKET:
#     CYS_USDT
#
# PRIMARY:
#     NEXT 15-MINUTE CANDLE
#
# TIMEFRAMES:
#     1M + 5M + 15M + 4H
#
# MEXC FUTURES API
# ============================================================


BASE_URL = "https://api.mexc.com"

INTERVAL_SECONDS = {
    "Min1": 60,
    "Min5": 300,
    "Min15": 900,
    "Hour4": 14400,
}

MEXC_INTERVALS = {
    "Min1": "Min1",
    "Min5": "Min5",
    "Min15": "Min15",
    "Hour4": "Hour4",
}


SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
})


# ============================================================
# VALIDATE INTERVAL
# ============================================================

def _validate_interval(interval):

    if interval not in INTERVAL_SECONDS:

        raise ValueError(
            f"Unsupported interval: {interval}"
        )


# ============================================================
# NORMALIZE SYMBOL
# ============================================================

def _normalize_symbol(symbol):

    if symbol is None:

        raise ValueError(
            "Symbol cannot be None."
        )

    symbol = str(symbol).upper().strip()

    symbol = symbol.replace(".P", "")
    symbol = symbol.replace("_", "")
    symbol = symbol.replace("-", "")

    if not symbol:

        raise ValueError(
            "Symbol cannot be empty."
        )

    # --------------------------------------------------------
    # MEXC Futures uses CYS_USDT
    # --------------------------------------------------------

    if symbol == "CYSUSDT":

        return "CYS_USDT"

    if symbol.endswith("USDT"):

        base = symbol[:-4]

        return f"{base}_USDT"

    return symbol


# ============================================================
# REQUEST
# ============================================================

def _request(
    symbol,
    interval,
    start,
    end,
    limit=1000,
    retries=5,
):

    url = (
        f"{BASE_URL}/api/v1/contract/kline/"
        f"{symbol}"
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
                timeout=30,
            )

            response.raise_for_status()

            payload = response.json()

            # ------------------------------------------------
            # MEXC can return data directly or inside success
            # ------------------------------------------------

            if isinstance(payload, dict):

                if payload.get("success") is False:

                    raise RuntimeError(
                        "MEXC returned an error: "
                        f"{payload}"
                    )

                data = payload.get(
                    "data",
                    payload,
                )

            else:

                data = payload

            # ------------------------------------------------
            # MEXC futures kline format
            #
            # {
            #   time: [],
            #   open: [],
            #   close: [],
            #   high: [],
            #   low: [],
            #   vol: [],
            #   amount: []
            # }
            # ------------------------------------------------

            if not isinstance(data, dict):

                return pd.DataFrame()

            timestamps = data.get(
                "time",
                [],
            )

            opens = data.get(
                "open",
                [],
            )

            closes = data.get(
                "close",
                [],
            )

            highs = data.get(
                "high",
                [],
            )

            lows = data.get(
                "low",
                [],
            )

            volumes = data.get(
                "vol",
                [],
            )

            amounts = data.get(
                "amount",
                [],
            )

            if not timestamps:

                return pd.DataFrame()

            records = []

            count = min(
                len(timestamps),
                len(opens),
                len(closes),
                len(highs),
                len(lows),
                len(volumes),
            )

            for i in range(count):

                try:

                    records.append({
                        "timestamp": int(
                            timestamps[i]
                        ),
                        "open": float(
                            opens[i]
                        ),
                        "high": float(
                            highs[i]
                        ),
                        "low": float(
                            lows[i]
                        ),
                        "close": float(
                            closes[i]
                        ),
                        "volume": float(
                            volumes[i]
                        ),
                        "turnover": (
                            float(amounts[i])
                            if i < len(amounts)
                            else 0.0
                        ),
                    })

                except (
                    ValueError,
                    TypeError,
                ):

                    continue

            if not records:

                return pd.DataFrame()

            return pd.DataFrame(records)

        except Exception as exc:

            last_error = exc

            if attempt < retries - 1:

                time.sleep(
                    2 + attempt * 2
                )

    raise RuntimeError(
        f"Unable to download "
        f"{symbol} {interval} candles: "
        f"{last_error}"
    )


# ============================================================
# NORMALIZE DATA
# ============================================================

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

    # --------------------------------------------------------
    # MEXC timestamps are Unix seconds.
    # --------------------------------------------------------

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


# ============================================================
# REMOVE CURRENTLY FORMING CANDLE
# ============================================================

def _remove_forming_candle(
    frame,
    interval,
):

    if frame is None or frame.empty:

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

    completed = frame[
        frame["timestamp"]
        < current_period_start
    ].copy()

    return (
        completed
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# FETCH KLINES
# ============================================================

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

    total = int(total)

    if total < 100:

        raise ValueError(
            "total must be at least 100 candles."
        )

    seconds = INTERVAL_SECONDS[
        interval
    ]

    now = int(time.time())

    # --------------------------------------------------------
    # MEXC futures API supports historical time ranges.
    # We request chunks to build the required history.
    # --------------------------------------------------------

    max_per_request = 1000

    target_end = now

    target_start = (
        now
        - (
            total
            * seconds
        )
        - seconds
    )

    chunks = []

    cursor_end = target_end

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
            interval=MEXC_INTERVALS[
                interval
            ],
            start=cursor_start,
            end=cursor_end,
            limit=max_per_request,
        )

        request_count += 1

        if not chunk.empty:

            chunks.append(
                chunk
            )

        cursor_end = (
            cursor_start - 1
        )

        if request_count > 100:

            raise RuntimeError(
                "Historical download required "
                "too many requests."
            )

        time.sleep(0.15)

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

    frame = _remove_forming_candle(
        frame,
        interval,
    )

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


# ============================================================
# FETCH ALL TIMEFRAMES
# ============================================================

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

    candles_5m = (
        candles_15m * 3
        + 300
    )

    candles_1m = (
        candles_15m * 15
        + 1500
    )

    candles_4h = max(
        300,
        int(
            candles_15m / 16
        ) + 100,
    )

    data = {}

    print(
        "[NextCandle AI] "
        "Downloading 1M candles..."
    )

    data["1m"] = fetch_klines(
        symbol=symbol,
        interval="Min1",
        total=candles_1m,
    )

    print(
        "[NextCandle AI] "
        "Downloading 5M candles..."
    )

    data["5m"] = fetch_klines(
        symbol=symbol,
        interval="Min5",
        total=candles_5m,
    )

    print(
        "[NextCandle AI] "
        "Downloading 15M candles..."
    )

    data["15m"] = fetch_klines(
        symbol=symbol,
        interval="Min15",
        total=candles_15m,
    )

    print(
        "[NextCandle AI] "
        "Downloading 4H candles..."
    )

    data["4h"] = fetch_klines(
        symbol=symbol,
        interval="Hour4",
        total=candles_4h,
    )

    validate_timeframe_data(
        data
    )

    return data


# ============================================================
# VALIDATE TIMEFRAME DATA
# ============================================================

def validate_timeframe_data(
    data
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
            f"Missing timeframes: "
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
                f"columns: {missing_columns}"
            )

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

        price_columns = [
            "open",
            "high",
            "low",
            "close",
        ]

        if (
            frame[price_columns]
            <= 0
        ).any().any():

            raise ValueError(
                f"{timeframe} contains "
                f"invalid non-positive prices."
            )

        invalid_ohlc = (
            (frame["high"] < frame["low"])
            | (
                frame["high"]
                < frame["open"]
            )
            | (
                frame["high"]
                < frame["close"]
            )
            | (
                frame["low"]
                > frame["open"]
            )
            | (
                frame["low"]
                > frame["close"]
            )
        )

        if invalid_ohlc.any():

            raise ValueError(
                f"{timeframe} contains "
                f"invalid OHLC relationships."
            )

        if (
            frame["volume"] < 0
        ).any():

            raise ValueError(
                f"{timeframe} contains "
                f"negative volume."
            )

    # --------------------------------------------------------
    # Verify no currently-forming candles
    # --------------------------------------------------------

    now = int(time.time())

    timeframe_intervals = {
        "1m": "Min1",
        "5m": "Min5",
        "15m": "Min15",
        "4h": "Hour4",
    }

    for timeframe, frame in data.items():

        seconds = INTERVAL_SECONDS[
            timeframe_intervals[
                timeframe
            ]
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

        if (
            latest_timestamp
            >= current_period_start
        ):

            raise ValueError(
                f"{timeframe} contains "
                f"a currently forming candle."
            )

    return True


# ============================================================
# LATEST COMPLETED CANDLE
# ============================================================

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

    return frame.iloc[
        -1
    ].copy()


# ============================================================
# LATEST PRICE
# ============================================================

def get_latest_price(
    symbol="CYSUSDT",
):

    symbol = _normalize_symbol(
        symbol
    )

    url = (
        f"{BASE_URL}/api/v1/contract/ticker"
    )

    params = {
        "symbol": symbol,
    }

    last_error = None

    for attempt in range(5):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=20,
            )

            response.raise_for_status()

            payload = response.json()

            if (
                isinstance(payload, dict)
                and payload.get("success")
                is False
            ):

                raise RuntimeError(
                    "MEXC ticker request failed: "
                    f"{payload}"
                )

            data = payload.get(
                "data",
                payload,
            )

            if isinstance(
                data,
                list,
            ):

                if not data:

                    raise RuntimeError(
                        "MEXC returned an empty "
                        "ticker response."
                    )

                data = data[0]

            if not isinstance(
                data,
                dict,
            ):

                raise RuntimeError(
                    "Invalid MEXC ticker response."
                )

            price = (
                data.get("lastPrice")
                or data.get("last")
            )

            if price is None:

                raise RuntimeError(
                    "MEXC ticker response "
                    "does not contain a last price."
                )

            return float(price)

        except Exception as exc:

            last_error = exc

            if attempt < 4:

                time.sleep(
                    2 + attempt
                )

    raise RuntimeError(
        f"Unable to get latest price "
        f"for {symbol}: {last_error}"
    )
