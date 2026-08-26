import time
import requests
import pandas as pd


BASE = "https://contract.mexc.com"


def fetch_klines(
    symbol="ACE_USDT",
    interval="Min15",
    total=1000
):
    seconds_per_candle = {
        "Min15": 15 * 60,
        "Hour4": 4 * 60 * 60,
    }

    if interval not in seconds_per_candle:
        raise ValueError(
            "Unsupported interval"
        )

    now = int(time.time())

    candle_seconds = seconds_per_candle[interval]

    end = now

    start = end - (
        total * candle_seconds
    )

    params = {
        "interval": interval,
        "start": start,
        "end": end,
    }

    url = (
        f"{BASE}/api/v1/contract/kline/{symbol}"
    )

    r = requests.get(
        url,
        params=params,
        timeout=20
    )

    r.raise_for_status()

    payload = r.json()

    if not payload.get("success"):
        raise RuntimeError(
            f"MEXC error {payload.get('code')}: "
            f"{payload.get('message', 'Unknown error')}"
        )

    data = payload.get("data")

    if not data or not data.get("time"):
        raise RuntimeError(
            "No MEXC candle data returned."
        )

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            data["time"],
            unit="s",
            utc=True
        ),
        "open": pd.to_numeric(
            data["open"],
            errors="coerce"
        ),
        "high": pd.to_numeric(
            data["high"],
            errors="coerce"
        ),
        "low": pd.to_numeric(
            data["low"],
            errors="coerce"
        ),
        "close": pd.to_numeric(
            data["close"],
            errors="coerce"
        ),
        "volume": pd.to_numeric(
            data["vol"],
            errors="coerce"
        ),
        "turnover": pd.to_numeric(
            data["amount"],
            errors="coerce"
        ),
    })

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # =========================================================
    # REMOVE CURRENTLY FORMING CANDLE
    # =========================================================

    current_period_start = (
        now // candle_seconds
    ) * candle_seconds

    current_period_start = pd.to_datetime(
        current_period_start,
        unit="s",
        utc=True
    )

    df = df[
        df["timestamp"] < current_period_start
    ].copy()

    df = (
        df
        .replace(
            [float("inf"), float("-inf")],
            pd.NA
        )
        .dropna()
        .reset_index(drop=True)
    )

    if len(df) < 100:
        raise RuntimeError(
            "Not enough completed candle data returned."
        )

    return df
