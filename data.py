import time
import requests
import pandas as pd

BASE = "https://api.mexc.com"


def fetch_klines(symbol="ACE_USDT", interval="Min15", total=1000):
    end = int(time.time())
    start = end - (total * 15 * 60)

    params = {
        "interval": interval,
        "start": start,
        "end": end,
    }

    url = f"{BASE}/api/v1/contract/kline/{symbol}"

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    payload = r.json()

    if not payload.get("success"):
        raise RuntimeError(
            payload.get("message", "MEXC API error")
        )

    data = payload["data"]

    if not data or not data.get("time"):
        raise RuntimeError("No MEXC candle data returned.")

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            data["time"], unit="s", utc=True
        ),
        "open": pd.to_numeric(data["open"]),
        "high": pd.to_numeric(data["high"]),
        "low": pd.to_numeric(data["low"]),
        "close": pd.to_numeric(data["close"]),
        "volume": pd.to_numeric(data["vol"]),
        "turnover": pd.to_numeric(data["amount"]),
    })

    return (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
