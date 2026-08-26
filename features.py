import numpy as np
import pandas as pd


def make_features(df, htf_df=None):
    x = df.copy()

    # =========================================================
    # CLEAN + SORT
    # =========================================================

    x["timestamp"] = pd.to_datetime(x["timestamp"])
    x = x.sort_values("timestamp").reset_index(drop=True)

    # =========================================================
    # 15-MINUTE PRICE FEATURES
    # =========================================================

    x["ret1"] = x["close"].pct_change()
    x["ret3"] = x["close"].pct_change(3)
    x["ret5"] = x["close"].pct_change(5)
    x["ret10"] = x["close"].pct_change(10)

    # Candle body and range
    x["body"] = (
        (x["close"] - x["open"]) / x["open"]
    )

    x["range"] = (
        (x["high"] - x["low"]) / x["open"]
    )

    # Body relative to candle range
    x["body_to_range"] = (
        x["body"].abs()
        / x["range"].replace(0, np.nan)
    )

    # Upper/lower wick
    candle_top = x[["open", "close"]].max(axis=1)
    candle_bottom = x[["open", "close"]].min(axis=1)

    x["upper_wick"] = (
        (x["high"] - candle_top)
        / x["open"]
    )

    x["lower_wick"] = (
        (candle_bottom - x["low"])
        / x["open"]
    )

    # Candle direction
    x["green"] = (
        x["close"] > x["open"]
    ).astype(int)

    x["prev_green"] = x["green"].shift(1)

    x["three_green"] = (
        x["green"].rolling(3).sum()
    )

    x["three_red"] = (
        (1 - x["green"]).rolling(3).sum()
    )

    # =========================================================
    # EMA / TREND
    # =========================================================

    for n in [9, 20, 50, 100, 200]:

        ema = x["close"].ewm(
            span=n,
            adjust=False
        ).mean()

        x[f"ema_gap_{n}"] = (
            x["close"] / ema - 1
        )

    # EMA relationships
    ema9 = x["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    ema20 = x["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = x["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    x["ema9_20_gap"] = (
        ema9 / ema20 - 1
    )

    x["ema20_50_gap"] = (
        ema20 / ema50 - 1
    )

    x["ema9_slope"] = (
        ema9.pct_change(3)
    )

    x["ema20_slope"] = (
        ema20.pct_change(3)
    )

    # =========================================================
    # RSI
    # =========================================================

    delta = x["close"].diff()

    gain = (
        delta.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    x["rsi14"] = (
        100 - (100 / (1 + rs))
    )

    # RSI distance from midpoint
    x["rsi_mid_gap"] = (
        x["rsi14"] - 50
    ) / 50

    # =========================================================
    # VOLATILITY
    # =========================================================

    for n in [5, 10, 20, 50]:

        x[f"volatility_{n}"] = (
            x["ret1"].rolling(n).std()
        )

    # ATR-like normalized range
    previous_close = x["close"].shift(1)

    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - previous_close).abs(),
            (x["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    x["atr14_pct"] = (
        true_range.rolling(14).mean()
        / x["close"]
    )

    # =========================================================
    # RECENT PRICE LOCATION
    # =========================================================

    for n in [5, 10, 20, 50]:

        rolling_high = (
            x["high"].rolling(n).max()
        )

        rolling_low = (
            x["low"].rolling(n).min()
        )

        x[f"dist_high_{n}"] = (
            x["close"] / rolling_high - 1
        )

        x[f"dist_low_{n}"] = (
            x["close"] / rolling_low - 1
        )

    # =========================================================
    # VOLUME BEHAVIOR
    # =========================================================

    x["vol_change"] = (
        x["volume"].pct_change()
    )

    vol_mean20 = (
        x["volume"].rolling(20).mean()
    )

    vol_std20 = (
        x["volume"].rolling(20).std()
    )

    x["vol_z20"] = (
        (x["volume"] - vol_mean20)
        / vol_std20.replace(0, np.nan)
    )

    x["volume_ratio20"] = (
        x["volume"] / vol_mean20
    )

    # Price move + volume confirmation
    x["volume_confirm"] = (
        x["ret1"] * x["vol_z20"]
    )

    # =========================================================
    # 4-HOUR HIGHER-TIMEFRAME BIAS
    # =========================================================

    if htf_df is not None and len(htf_df) > 0:

        h = htf_df.copy()

        h["timestamp"] = pd.to_datetime(
            h["timestamp"]
        )

        h = (
            h.sort_values("timestamp")
            .reset_index(drop=True)
        )

        # 4H EMAs
        h["htf_ema20"] = (
            h["close"].ewm(
                span=20,
                adjust=False
            ).mean()
        )

        h["htf_ema50"] = (
            h["close"].ewm(
                span=50,
                adjust=False
            ).mean()
        )

        h["htf_ema200"] = (
            h["close"].ewm(
                span=200,
                adjust=False
            ).mean()
        )

        h["htf_ema_gap20"] = (
            h["close"] / h["htf_ema20"] - 1
        )

        h["htf_ema_gap50"] = (
            h["close"] / h["htf_ema50"] - 1
        )

        h["htf_ema20_50_gap"] = (
            h["htf_ema20"]
            / h["htf_ema50"]
            - 1
        )

        h["htf_ema50_200_gap"] = (
            h["htf_ema50"]
            / h["htf_ema200"]
            - 1
        )

        h["htf_ema20_slope"] = (
            h["htf_ema20"].pct_change(3)
        )

        h["htf_ret1"] = (
            h["close"].pct_change()
        )

        h["htf_ret3"] = (
            h["close"].pct_change(3)
        )

        h["htf_body"] = (
            (h["close"] - h["open"])
            / h["open"]
        )

        h["htf_range"] = (
            (h["high"] - h["low"])
            / h["open"]
        )

        h["htf_green"] = (
            h["close"] > h["open"]
        ).astype(int)

        # -----------------------------------------------------
        # CRITICAL:
        # Only completed 4H candles can influence prediction.
        # The newest 4H candle may still be forming.
        # -----------------------------------------------------

        if len(h) > 1:
            h = h.iloc[:-1].copy()

        h = h[
            [
                "timestamp",
                "close",
                "htf_ema_gap20",
                "htf_ema_gap50",
                "htf_ema20_50_gap",
                "htf_ema50_200_gap",
                "htf_ema20_slope",
                "htf_ret1",
                "htf_ret3",
                "htf_body",
                "htf_range",
                "htf_green"
            ]
        ].rename(
            columns={
                "close": "htf_close"
            }
        )

        x = x.sort_values("timestamp")

        x = pd.merge_asof(
            x,
            h,
            on="timestamp",
            direction="backward"
        )

    # =========================================================
    # 4H BIAS SCORE
    # =========================================================

    if "htf_ema_gap20" in x.columns:

        bullish_conditions = (
            (x["htf_ema_gap20"] > 0).astype(int)
            + (x["htf_ema20_50_gap"] > 0).astype(int)
            + (x["htf_ema50_200_gap"] > 0).astype(int)
            + (x["htf_ema20_slope"] > 0).astype(int)
            + (x["htf_ret3"] > 0).astype(int)
        )

        bearish_conditions = (
            (x["htf_ema_gap20"] < 0).astype(int)
            + (x["htf_ema20_50_gap"] < 0).astype(int)
            + (x["htf_ema50_200_gap"] < 0).astype(int)
            + (x["htf_ema20_slope"] < 0).astype(int)
            + (x["htf_ret3"] < 0).astype(int)
        )

        x["htf_bias_score"] = (
            bullish_conditions
            - bearish_conditions
        )

    # =========================================================
    # TARGET: NEXT 15-MINUTE CANDLE
    # =========================================================

    future_ret = (
        x["close"].shift(-1)
        / x["close"]
        - 1
    )

    # Adaptive neutral zone:
    # approximately one recent volatility unit.
    recent_vol = (
        x["ret1"]
        .rolling(20)
        .std()
    )

    neutral_threshold = (
        recent_vol * 0.50
    ).clip(
        lower=0.0001
    )

    x["target"] = np.select(
        [
            future_ret > neutral_threshold,
            future_ret < -neutral_threshold
        ],
        [
            1,
            -1
        ],
        default=0
    )

    # =========================================================
    # CLEAN DATA
    # =========================================================

    x = (
        x
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .reset_index(drop=True)
    )

    return x
