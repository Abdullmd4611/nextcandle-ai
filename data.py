import time
import requests
import pandas as pd

BASE = "https://api.bybit.com"

def fetch_klines(symbol="BTCUSDT", interval="5", total=5000):
    rows = []
    end = int(time.time() * 1000)
    remaining = total

    while remaining > 0:
        limit = min(1000, remaining)
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "end": end,
            "limit": limit,
        }
        r = requests.get(f"{BASE}/v5/market/kline", params=params, timeout=20)
        r.raise_for_status()
        payload = r.json()
        if payload.get("retCode") != 0:
            raise RuntimeError(payload.get("retMsg", "Bybit API error"))
        batch = payload["result"]["list"]
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(x[0]) for x in batch)
        end = oldest - 1
        remaining -= len(batch)
        if len(batch) < limit:
            break

    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df
