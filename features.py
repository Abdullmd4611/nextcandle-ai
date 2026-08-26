import numpy as np
import pandas as pd


def make_features(df, htf_df=None):
    x = df.copy()

    # =========================
    # 15-MINUTE FEATURES
    # =========================

    x["ret1"] = x.close.pct_change()
    x["ret3"] = x.close.pct_change(3)
    x["ret5"] = x.close.pct_change(5)
    x["ret10"] = x.close.pct_change(10)

    x["body"] = (x.close - x.open) / x.open
    x["range"] = (x.high - x.low) / x.open

    x["upper_wick"] = (
        x.high - x[["open", "close"]].max(axis=1)
    ) / x.open

    x["lower_wick"] = (
        x[["open", "close"]].min(axis=1) - x.low
    ) / x.open

    x["vol_change"] = x.volume.pct_change()

    vol_mean = x.volume.rolling(20).mean()
    vol_std = x.volume.rolling(20).std()

    x["vol_z20"] = (
        (x.volume - vol_mean) / vol_std
    )

    for n in [5, 10, 20, 50]:
        x[f"volatility_{n}"] = (
            x["ret1"].rolling(n).std()
        )

        x[f"dist_high_{n}"] = (
            x.close / x.high.rolling(n).max() - 1
        )

        x[f"dist_low_{n}"] = (
            x.close / x.low.rolling(n).min() - 1
        )

    for n in [9, 20, 50, 200]:
        ema = x.close.ewm(
            span=n,
            adjust=False
        ).mean()

        x[f"ema_gap_{n}"] = (
            x.close / ema - 1
        )

    # RSI
    delta = x.close.diff()

    gain = (
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
    ).rolling(14).mean()

    rs = gain / loss.replace(0, np.nan)

    x["rsi14"] = 100 - (
        100 / (1 + rs)
    )

    # Candle context
    x["green"] = (
        x.close > x.open
    ).astype(int)

    x["prev_green"] = x["green"].shift(1)

    x["three_green"] = (
        x["green"].rolling(3).sum()
    )

    x["three_red"] = (
        (1 - x["green"])
        .rolling(3)
        .sum()
    )


    # =========================
    # 4-HOUR HIGHER-TIMEFRAME BIAS
    # =========================

    if htf_df is not None and len(htf_df) > 0:

        h = htf_df.copy()

        h["htf_ema20"] = (
            h.close.ewm(
                span=20,
                adjust=False
            ).mean()
        )

        h["htf_ema50"] = (
            h.close.ewm(
                span=50,
                adjust=False
            ).mean()
        )

        h["htf_ema_gap20"] = (
            h.close / h.htf_ema20 - 1
        )

        h["htf_ema_gap50"] = (
            h.close / h.htf_ema50 - 1
        )

        h["htf_body"] = (
            h.close - h.open
        ) / h.open

        h["htf_range"] = (
            h.high - h.low
        ) / h.open

        h["htf_green"] = (
            h.close > h.open
        ).astype(int)

        # Only use COMPLETED 4H candles.
        # The newest candle may still be forming.
        if len(h) > 1:
            h = h.iloc[:-1].copy()

        h = h[
            [
                "timestamp",
                "close",
                "htf_ema_gap20",
                "htf_ema_gap50",
                "htf_body",
                "htf_range",
                "htf_green",
            ]
        ].rename(
            columns={
                "close": "htf_close"
            }
        )

        h = h.sort_values("timestamp")
        x = x.sort_values("timestamp")

        # Attach the latest completed 4H information
        # to each 15M candle.
        x = pd.merge_asof(
            x,
            h,
            on="timestamp",
            direction="backward"
        )


    # =========================
    # TARGET
    # =========================

    # Direction of the NEXT 15-minute candle.
    future_ret = (
        x.close.shift(-1) / x.close - 1
    )

    x["target"] = np.select(
        [
            future_ret > 0.0001,
            future_ret < -0.0001
        ],
        [
            1,
            -1
        ],
        default=0
    )

    return (
        x
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
