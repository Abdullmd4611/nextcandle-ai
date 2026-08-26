import time
import requests
import pandas as pd

BASE = "https://api.binance.com"

def fetch_klines(symbol="BTCUSDT", interval="5", total=5000):
    rows = []
    end = int(time.time() * 1000)
    remaining = total

    while remaining > 0:
        limit = min(1000, remaining)

        params = {
            "symbol": symbol,
            "interval": interval,
            "endTime": end,
            "limit": limit,
        }

        r = requests.get(
            f"{BASE}/api/v3/klines",
            params=params,
            timeout=20,
        )

        r.raise_for_status()

        batch = r.json()

        if not batch:
            break

        rows.extend(batch)

        oldest = min(int(x[0]) for x in batch)
        end = oldest - 1
        remaining -= len(batch)

        if len(batch) < limit:
            break

    if not rows:
        raise RuntimeError("No candle data was returned.")

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["turnover"] = pd.to_numeric(
        df["quote_volume"], errors="coerce"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"].astype("int64"),
        unit="ms",
        utc=True,
    )

    df = (
        df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "turnover",
            ]
        ]
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    return df
