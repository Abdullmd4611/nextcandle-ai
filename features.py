import numpy as np
import pandas as pd

def make_features(df):
    x = df.copy()

    x["ret1"] = x.close.pct_change()
    x["ret3"] = x.close.pct_change(3)
    x["ret5"] = x.close.pct_change(5)
    x["ret10"] = x.close.pct_change(10)

    x["body"] = (x.close - x.open) / x.open
    x["range"] = (x.high - x.low) / x.open
    x["upper_wick"] = (x.high - x[["open","close"]].max(axis=1)) / x.open
    x["lower_wick"] = (x[["open","close"]].min(axis=1) - x.low) / x.open

    x["vol_change"] = x.volume.pct_change()
    x["vol_z20"] = (x.volume - x.volume.rolling(20).mean()) / x.volume.rolling(20).std()

    for n in [5, 10, 20, 50]:
        x[f"volatility_{n}"] = x["ret1"].rolling(n).std()
        x[f"dist_high_{n}"] = x.close / x.high.rolling(n).max() - 1
        x[f"dist_low_{n}"] = x.close / x.low.rolling(n).min() - 1

    for n in [9, 20, 50, 200]:
        ema = x.close.ewm(span=n, adjust=False).mean()
        x[f"ema_gap_{n}"] = x.close / ema - 1

    delta = x.close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = 100 - (100 / (1 + rs))

    # Simple candle-context features; all use only information available at the current row.
    x["green"] = (x.close > x.open).astype(int)
    x["prev_green"] = x["green"].shift(1)
    x["three_green"] = x["green"].rolling(3).sum()
    x["three_red"] = (1-x["green"]).rolling(3).sum()

    # Target: direction of the NEXT candle.
    future_ret = x.close.shift(-1) / x.close - 1
    x["target"] = np.select(
        [future_ret > 0.0001, future_ret < -0.0001],
        [1, -1],
        default=0
    )

    return x.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
