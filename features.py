import numpy as np
import pandas as pd


def make_features(df, htf_df=None):

    x = df.copy()

    # =========================================================
    # CLEAN + SORT
    # =========================================================

    x["timestamp"] = pd.to_datetime(
        x["timestamp"],
        utc=True
    )

    x = (
        x
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # =========================================================
    # 15M RETURNS
    # =========================================================

    x["ret1"] = (
        x["close"].pct_change()
    )

    x["ret3"] = (
        x["close"].pct_change(3)
    )

    x["ret5"] = (
        x["close"].pct_change(5)
    )

    x["ret10"] = (
        x["close"].pct_change(10)
    )

    # =========================================================
    # CANDLE STRUCTURE
    # =========================================================

    x["body"] = (
        (x["close"] - x["open"])
        / x["open"]
    )

    x["range"] = (
        (x["high"] - x["low"])
        / x["open"]
    )

    x["body_to_range"] = (
        x["body"].abs()
        / x["range"].replace(0, np.nan)
    )

    candle_top = x[
        ["open", "close"]
    ].max(axis=1)

    candle_bottom = x[
        ["open", "close"]
    ].min(axis=1)

    x["upper_wick"] = (
        (x["high"] - candle_top)
        / x["open"]
    )

    x["lower_wick"] = (
        (candle_bottom - x["low"])
        / x["open"]
    )

    x["green"] = (
        x["close"] > x["open"]
    ).astype(int)

    x["prev_green"] = (
        x["green"].shift(1)
    )

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

    rs = (
        gain
        / loss.replace(0, np.nan)
    )

    x["rsi14"] = (
        100
        - (100 / (1 + rs))
    )

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

    previous_close = (
        x["close"].shift(1)
    )

    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (
                x["high"]
                - previous_close
            ).abs(),
            (
                x["low"]
                - previous_close
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    x["atr14_pct"] = (
        true_range.rolling(14).mean()
        / x["close"]
    )

    # =========================================================
    # PRICE LOCATION
    # =========================================================

    for n in [5, 10, 20, 50]:

        rolling_high = (
            x["high"].rolling(n).max()
        )

        rolling_low = (
            x["low"].rolling(n).min()
        )

        x[f"dist_high_{n}"] = (
            x["close"]
            / rolling_high
            - 1
        )

        x[f"dist_low_{n}"] = (
            x["close"]
            / rolling_low
            - 1
        )

    # =========================================================
    # VOLUME
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
        (
            x["volume"]
            - vol_mean20
        )
        / vol_std20.replace(0, np.nan)
    )

    x["volume_ratio20"] = (
        x["volume"]
        / vol_mean20
    )

    x["volume_confirm"] = (
        x["ret1"]
        * x["vol_z20"]
    )

    # =========================================================
    # 4H CONTEXT
    # =========================================================

    if htf_df is not None and len(htf_df) > 0:

        h = htf_df.copy()

        h["timestamp"] = pd.to_datetime(
            h["timestamp"],
            utc=True
        )

        h = (
            h
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

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
            h["close"]
            / h["htf_ema20"]
            - 1
        )

        h["htf_ema_gap50"] = (
            h["close"]
            / h["htf_ema50"]
            - 1
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

        # Remove newest 4H candle if necessary.
        # Only completed HTF candles should influence
        # the 15M model.

        now = pd.Timestamp.now(
            tz="UTC"
        )

        four_hour_start = (
            now.floor("4h")
        )

        h = h[
            h["timestamp"]
            < four_hour_start
        ].copy()

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
                "htf_green",
            ]
        ].rename(
            columns={
                "close": "htf_close"
            }
        )

        x = pd.merge_asof(
            x.sort_values("timestamp"),
            h.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

    # =========================================================
    # 4H BIAS SCORE
    # =========================================================

    if "htf_ema_gap20" in x.columns:

        bullish_conditions = (
            (x["htf_ema_gap20"] > 0).astype(int)
            + (
                x["htf_ema20_50_gap"] > 0
            ).astype(int)
            + (
                x["htf_ema50_200_gap"] > 0
            ).astype(int)
            + (
                x["htf_ema20_slope"] > 0
            ).astype(int)
            + (
                x["htf_ret3"] > 0
            ).astype(int)
        )

        bearish_conditions = (
            (x["htf_ema_gap20"] < 0).astype(int)
            + (
                x["htf_ema20_50_gap"] < 0
            ).astype(int)
            + (
                x["htf_ema50_200_gap"] < 0
            ).astype(int)
            + (
                x["htf_ema20_slope"] < 0
            ).astype(int)
            + (
                x["htf_ret3"] < 0
            ).astype(int)
        )

        x["htf_bias_score"] = (
            bullish_conditions
            - bearish_conditions
        )

    # =========================================================
    # NEXT CANDLE DIRECTION TARGET
    # =========================================================

    future_return = (
        x["close"].shift(-1)
        / x["close"]
        - 1
    )

    recent_vol = (
        x["ret1"].rolling(20).std()
    )

    neutral_threshold = (
        recent_vol * 0.50
    ).clip(
        lower=0.0001
    )

    x["target"] = np.select(
        [
            future_return > neutral_threshold,
            future_return < -neutral_threshold,
        ],
        [
            1,
            -1,
        ],
        default=0
    )

    # =========================================================
    # NEXT CANDLE OHLC REGRESSION TARGETS
    # =========================================================

    # Next OPEN relative to current close
    x["future_open_return"] = (
        x["open"].shift(-1)
        / x["close"]
        - 1
    )

    # Next HIGH relative to current close
    x["future_high_return"] = (
        x["high"].shift(-1)
        / x["close"]
        - 1
    )

    # Next LOW relative to current close
    x["future_low_return"] = (
        x["low"].shift(-1)
        / x["close"]
        - 1
    )

    # Next CLOSE relative to current close
    x["future_close_return"] = (
        x["close"].shift(-1)
        / x["close"]
        - 1
    )

    # Compatibility name
    x["future_return"] = (
        x["future_close_return"]
    )

    x["future_close"] = (
        x["close"].shift(-1)
    )

    # =========================================================
    # CLEAN
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
