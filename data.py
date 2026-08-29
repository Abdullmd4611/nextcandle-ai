import time
import requests
import pandas as pd


BASE_URL = "https://api.bytick.com"
CATEGORY = "linear"


INTERVAL_SECONDS = {
    "Min1": 60,
    "Min5": 300,
    "Min15": 900,
    "Hour4": 14400,
}


BYBIT_INTERVALS = {
    "Min1": "1",
    "Min5": "5",
    "Min15": "15",
    "Hour4": "240",
}


SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": "NextCandleAI/5.0",
    "Accept": "application/json",
})


def _validate_interval(interval):

    if interval not in INTERVAL_SECONDS:
        raise ValueError(
            f"Unsupported interval: {interval}"
        )


def _normalize_symbol(symbol):

    if symbol is None:
        raise ValueError("Symbol cannot be None.")

    symbol = str(symbol).upper().strip()

    symbol = symbol.replace("_", "")
    symbol = symbol.replace(".P", "")

    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    return symbol


def _request(
    symbol,
    interval,
    start,
    end,
    limit=1000,
    retries=5,
):

    url = f"{BASE_URL}/v5/market/kline"

    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": interval,
        "start": int(start),
        "end": int(end),
        "limit": int(limit),
    }

    last_error = None

    for attempt in range(retries):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=30,
            )

            if response.status_code == 403:

                raise RuntimeError(
                    "Bybit returned HTTP 403 Forbidden. "
                    "The API endpoint or runner IP may be blocked."
                )

            response.raise_for_status()

            payload = response.json()

            if payload.get("retCode") != 0:

                raise RuntimeError(
                    "Bybit returned error "
                    f"{payload.get('retCode')}: "
                    f"{payload.get('retMsg', 'Unknown error')}"
                )

            rows = (
                payload
                .get("result", {})
                .get("list", [])
            )

            if not rows:
                return pd.DataFrame()

            records = []

            for row in rows:

                if len(row) < 7:
                    continue

                records.append({
                    "timestamp": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "turnover": float(row[6]),
                })

            if not records:
                return pd.DataFrame()

            return pd.DataFrame(records)

        except Exception as exc:

            last_error = exc

            if attempt < retries - 1:
                time.sleep(2 + attempt * 2)

    raise RuntimeError(
        f"Unable to download "
        f"{symbol} {interval} candles: "
        f"{last_error}"
    )


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


def _remove_forming_candle(
    frame,
    interval,
):

    if frame is None or frame.empty:
        return frame

    seconds = INTERVAL_SECONDS[interval]

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
        frame["timestamp"] < current_period_start
    ].copy()

    return (
        completed
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def fetch_klines(
    symbol="CYSUSDT",
    interval="Min15",
    total=2500,
):

    symbol = _normalize_symbol(symbol)

    _validate_interval(interval)

    total = int(total)

    if total < 100:
        raise ValueError(
            "total must be at least 100 candles."
        )

    seconds = INTERVAL_SECONDS[interval]

    now = int(time.time())

    max_per_request = 1000

    interval_ms = seconds * 1000

    target_end = now * 1000

    target_start = (
        now
        - (total * seconds)
        - seconds
    ) * 1000

    chunks = []

    cursor_end = target_end

    request_count = 0

    while cursor_end > target_start:

        cursor_start = max(
            target_start,
            cursor_end
            - (
                max_per_request
                * interval_ms
            ),
        )

        chunk = _request(
            symbol=symbol,
            interval=BYBIT_INTERVALS[interval],
            start=cursor_start,
            end=cursor_end,
            limit=max_per_request,
        )

        request_count += 1

        if not chunk.empty:
            chunks.append(chunk)

        cursor_end = cursor_start - 1

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

    frame = _normalize(frame)

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


def fetch_multi_timeframe(
    symbol="CYSUSDT",
    history_15m=2500,
):

    symbol = _normalize_symbol(symbol)

    candles_15m = int(history_15m)

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
        int(candles_15m / 16) + 100,
    )

    data = {}

    print(
        "[NextCandle AI] Downloading 1M candles..."
    )

    data["1m"] = fetch_klines(
        symbol=symbol,
        interval="Min1",
        total=candles_1m,
    )

    print(
        "[NextCandle AI] Downloading 5M candles..."
    )

    data["5m"] = fetch_klines(
        symbol=symbol,
        interval="Min5",
        total=candles_5m,
    )

    print(
        "[NextCandle AI] Downloading 15M candles..."
    )

    data["15m"] = fetch_klines(
        symbol=symbol,
        interval="Min15",
        total=candles_15m,
    )

    print(
        "[NextCandle AI] Downloading 4H candles..."
    )

    data["4h"] = fetch_klines(
        symbol=symbol,
        interval="Hour4",
        total=candles_4h,
    )

    validate_timeframe_data(data)

    return data


def validate_timeframe_data(data):

    if data is None:
        raise ValueError("Market data is None.")

    required = {
        "1m",
        "5m",
        "15m",
        "4h",
    }

    missing = required - set(data.keys())

    if missing:
        raise ValueError(
            f"Missing timeframes: {sorted(missing)}"
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

        frame = data[timeframe]

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
            frame[price_columns] <= 0
        ).any().any():

            raise ValueError(
                f"{timeframe} contains "
                f"invalid non-positive prices."
            )

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

        if (
            frame["volume"] < 0
        ).any():

            raise ValueError(
                f"{timeframe} contains "
                f"negative volume."
            )

    now = int(time.time())

    timeframe_intervals = {
        "1m": "Min1",
        "5m": "Min5",
        "15m": "Min15",
        "4h": "Hour4",
    }

    for timeframe, frame in data.items():

        seconds = INTERVAL_SECONDS[
            timeframe_intervals[timeframe]
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


def get_latest_price(
    symbol="CYSUSDT",
):

    symbol = _normalize_symbol(symbol)

    url = (
        f"{BASE_URL}/v5/market/tickers"
    )

    params = {
        "category": CATEGORY,
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

            if response.status_code == 403:

                raise RuntimeError(
                    "Bybit returned HTTP 403 Forbidden."
                )

            response.raise_for_status()

            payload = response.json()

            if payload.get("retCode") != 0:

                raise RuntimeError(
                    "Bybit ticker request failed: "
                    f"{payload.get('retMsg', 'Unknown error')}"
                )

            ticker_list = (
                payload
                .get("result", {})
                .get("list", [])
            )

            if not ticker_list:

                raise RuntimeError(
                    "Bybit returned an empty "
                    "ticker response."
                )

            price = ticker_list[0].get(
                "lastPrice"
            )

            if price is None:

                raise RuntimeError(
                    "Bybit ticker response "
                    "does not contain lastPrice."
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
