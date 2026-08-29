import numpy as np
import pandas as pd


# ============================================================
# NextCandle AI — Feature Engineering V3
#
# Primary prediction:
#     NEXT 15-MINUTE CANDLE
#
# Inputs:
#     1-minute  -> immediate/micro behavior
#     5-minute  -> short-term behavior
#     15-minute -> primary candle/market structure
#     4-hour    -> higher-timeframe context
#
# IMPORTANT:
# Features at time T may only use information available at T.
# The NEXT 15-minute candle is the prediction target.
# ============================================================


EPS = 1e-12


# ============================================================
# BASIC UTILITIES
# ============================================================

def _safe_div(a, b):
    return a / (b.replace(0, np.nan) + EPS)


def _atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        period,
        min_periods=period,
    ).mean()


def _rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / (avg_loss + EPS)

    return 100 - (100 / (1 + rs))


def _ema(series, period):
    return series.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def _rolling_zscore(series, period):
    mean = series.rolling(
        period,
        min_periods=period,
    ).mean()

    std = series.rolling(
        period,
        min_periods=period,
    ).std()

    return (series - mean) / (std + EPS)


# ============================================================
# DATA VALIDATION
# ============================================================

def _validate_ohlcv(df, name):
    if df is None or df.empty:
        raise ValueError(f"{name} data is empty.")

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} data missing columns: {missing}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            f"{name} index must be a pandas DatetimeIndex."
        )


def _clean_ohlcv(df):
    df = df.copy()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    df = df.sort_index()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric_columns
    )

    # Reject impossible OHLC relationships.
    df = df[
        (df["high"] >= df["low"])
        & (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
    ]

    return df


# ============================================================
# CANDLE ANATOMY
# ============================================================

def _candle_features(df, prefix=""):
    out = pd.DataFrame(index=df.index)

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    candle_range = (
        h - l
    ).clip(lower=EPS)

    body = (
        c - o
    ).abs()

    upper_wick = (
        h
        - pd.concat(
            [o, c],
            axis=1,
        ).max(axis=1)
    ).clip(lower=0)

    lower_wick = (
        pd.concat(
            [o, c],
            axis=1,
        ).min(axis=1)
        - l
    ).clip(lower=0)

    out[f"{prefix}body_pct"] = (
        body / candle_range
    )

    out[f"{prefix}upper_wick_pct"] = (
        upper_wick / candle_range
    )

    out[f"{prefix}lower_wick_pct"] = (
        lower_wick / candle_range
    )

    out[f"{prefix}body_direction"] = (
        (c - o) / candle_range
    )

    out[f"{prefix}range_pct"] = (
        candle_range
        / (c.shift(1).abs() + EPS)
    )

    out[f"{prefix}close_position"] = (
        (c - l) / candle_range
    )

    out[f"{prefix}open_position"] = (
        (o - l) / candle_range
    )

    out[f"{prefix}bullish"] = (
        c > o
    ).astype(int)

    out[f"{prefix}bearish"] = (
        c < o
    ).astype(int)

    out[f"{prefix}doji_like"] = (
        body / candle_range <= 0.10
    ).astype(int)

    # Geometry useful for future candle-shape classification.
    out[f"{prefix}lower_wick_body_ratio"] = (
        lower_wick / (body + EPS)
    )

    out[f"{prefix}upper_wick_body_ratio"] = (
        upper_wick / (body + EPS)
    )

    out[f"{prefix}range_body_ratio"] = (
        candle_range / (body + EPS)
    )

    # Soft pattern geometry.
    out[f"{prefix}hammer_geometry"] = (
        (lower_wick >= body * 2.0)
        & (upper_wick <= body)
        & (body / candle_range <= 0.45)
    ).astype(int)

    out[f"{prefix}shooting_star_geometry"] = (
        (upper_wick >= body * 2.0)
        & (lower_wick <= body)
        & (body / candle_range <= 0.45)
    ).astype(int)

    out[f"{prefix}long_lower_rejection"] = (
        lower_wick / candle_range >= 0.50
    ).astype(int)

    out[f"{prefix}long_upper_rejection"] = (
        upper_wick / candle_range >= 0.50
    ).astype(int)

    return out


# ============================================================
# MOMENTUM
# ============================================================

def _momentum_features(df, prefix=""):
    out = pd.DataFrame(index=df.index)

    close = df["close"]

    for period in [
        1,
        2,
        3,
        4,
        5,
        8,
        12,
        20,
        32,
        48,
    ]:
        out[f"{prefix}return_{period}"] = (
            close / close.shift(period) - 1
        )

    out[f"{prefix}rsi_7"] = _rsi(
        close,
        7,
    )

    out[f"{prefix}rsi_14"] = _rsi(
        close,
        14,
    )

    out[f"{prefix}rsi_28"] = _rsi(
        close,
        28,
    )

    ema9 = _ema(close, 9)
    ema21 = _ema(close, 21)
    ema50 = _ema(close, 50)
    ema100 = _ema(close, 100)

    out[f"{prefix}ema9_distance"] = _safe_div(
        close - ema9,
        close,
    )

    out[f"{prefix}ema21_distance"] = _safe_div(
        close - ema21,
        close,
    )

    out[f"{prefix}ema50_distance"] = _safe_div(
        close - ema50,
        close,
    )

    out[f"{prefix}ema100_distance"] = _safe_div(
        close - ema100,
        close,
    )

    out[f"{prefix}ema9_21_spread"] = _safe_div(
        ema9 - ema21,
        close,
    )

    out[f"{prefix}ema21_50_spread"] = _safe_div(
        ema21 - ema50,
        close,
    )

    out[f"{prefix}ema50_100_spread"] = _safe_div(
        ema50 - ema100,
        close,
    )

    # Momentum acceleration.
    return_1 = close.pct_change(1)
    return_3 = close.pct_change(3)

    out[f"{prefix}momentum_acceleration"] = (
        return_1
        - return_1.shift(3)
    )

    out[f"{prefix}momentum_slope"] = (
        return_3
        - return_3.shift(3)
    )

    return out


# ============================================================
# VOLATILITY
# ============================================================

def _volatility_features(df, prefix=""):
    out = pd.DataFrame(index=df.index)

    close = df["close"]

    atr14 = _atr(
        df,
        14,
    )

    atr28 = _atr(
        df,
        28,
    )

    out[f"{prefix}atr_pct"] = _safe_div(
        atr14,
        close,
    )

    out[f"{prefix}atr_ratio"] = _safe_div(
        atr14,
        atr28,
    )

    returns = close.pct_change()

    for period in [
        5,
        8,
        20,
        48,
    ]:
        out[
            f"{prefix}volatility_{period}"
        ] = returns.rolling(
            period,
            min_periods=period,
        ).std()

    out[
        f"{prefix}volatility_zscore"
    ] = _rolling_zscore(
        returns.rolling(
            20,
            min_periods=20,
        ).std(),
        48,
    )

    return out


# ============================================================
# VOLUME
# ============================================================

def _volume_features(df, prefix=""):
    out = pd.DataFrame(index=df.index)

    volume = df["volume"]

    mean20 = volume.rolling(
        20,
        min_periods=20,
    ).mean()

    std20 = volume.rolling(
        20,
        min_periods=20,
    ).std()

    out[f"{prefix}volume_ratio"] = (
        volume / (mean20 + EPS)
    )

    out[f"{prefix}volume_zscore"] = (
        (volume - mean20)
        / (std20 + EPS)
    )

    out[f"{prefix}volume_change_1"] = (
        volume.pct_change(1)
    )

    out[f"{prefix}volume_change_4"] = (
        volume.pct_change(4)
    )

    candle_range = (
        df["high"] - df["low"]
    ).clip(lower=EPS)

    close_location = (
        (df["close"] - df["low"])
        / candle_range
    )

    out[f"{prefix}volume_pressure"] = (
        (close_location - 0.5) * 2
    ) * out[f"{prefix}volume_ratio"]

    return out


# ============================================================
# STRUCTURE
# ============================================================

def _structure_features(df, prefix=""):
    out = pd.DataFrame(index=df.index)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_high = high.shift(1)
    previous_low = low.shift(1)

    for period in [
        5,
        10,
        20,
        48,
    ]:

        rolling_high = high.shift(1).rolling(
            period,
            min_periods=period,
        ).max()

        rolling_low = low.shift(1).rolling(
            period,
            min_periods=period,
        ).min()

        out[
            f"{prefix}break_high_{period}"
        ] = (
            close > rolling_high
        ).astype(int)

        out[
            f"{prefix}break_low_{period}"
        ] = (
            close < rolling_low
        ).astype(int)

        out[
            f"{prefix}distance_high_{period}"
        ] = _safe_div(
            close - rolling_high,
            close,
        )

        out[
            f"{prefix}distance_low_{period}"
        ] = _safe_div(
            close - rolling_low,
            close,
        )

    direction = np.sign(
        close.diff()
    )

    for period in [
        3,
        6,
        12,
        24,
    ]:

        out[
            f"{prefix}direction_{period}_sum"
        ] = direction.rolling(
            period,
            min_periods=period,
        ).sum()

    out[f"{prefix}higher_high"] = (
        high > previous_high
    ).astype(int)

    out[f"{prefix}lower_low"] = (
        low < previous_low
    ).astype(int)

    # Expansion of current range relative to recent ranges.
    current_range = (
        high - low
    )

    average_range = current_range.rolling(
        20,
        min_periods=20,
    ).mean()

    out[f"{prefix}range_expansion"] = (
        current_range
        / (average_range + EPS)
    )

    return out


# ============================================================
# COMPLETE TIMEFRAME FEATURES
# ============================================================

def _base_features(df, prefix):
    parts = [
        _candle_features(
            df,
            prefix,
        ),
        _momentum_features(
            df,
            prefix,
        ),
        _volatility_features(
            df,
            prefix,
        ),
        _volume_features(
            df,
            prefix,
        ),
        _structure_features(
            df,
            prefix,
        ),
    ]

    return pd.concat(
        parts,
        axis=1,
    )


# ============================================================
# LOWER-TIMEFRAME AGGREGATION
# ============================================================

def _aggregate_lower_timeframe(
    df,
    target_index,
    prefix,
):
    """
    Convert completed lower-timeframe candles into features
    aligned to each 15-minute prediction timestamp.

    The aggregation uses only lower-timeframe candles whose
    timestamps are strictly before the prediction timestamp.

    This prevents the lower timeframe from leaking information
    from the future 15-minute candle.
    """

    if df is None or df.empty:
        return pd.DataFrame(
            index=target_index
        )

    df = _clean_ohlcv(df)

    features = _base_features(
        df,
        prefix,
    )

    # Add recent lower-timeframe state.
    close = df["close"]

    features[
        f"{prefix}short_return"
    ] = close.pct_change(3)

    features[
        f"{prefix}medium_return"
    ] = close.pct_change(12)

    features[
        f"{prefix}direction_ratio_12"
    ] = (
        np.sign(
            close.diff()
        )
        .rolling(
            12,
            min_periods=12,
        )
        .mean()
    )

    # Candle rejection statistics.
    lower_rejection = features[
        f"{prefix}long_lower_rejection"
    ]

    upper_rejection = features[
        f"{prefix}long_upper_rejection"
    ]

    features[
        f"{prefix}lower_rejection_rate"
    ] = lower_rejection.rolling(
        15,
        min_periods=15,
    ).mean()

    features[
        f"{prefix}upper_rejection_rate"
    ] = upper_rejection.rolling(
        15,
        min_periods=15,
    ).mean()

    # We require STRICTLY earlier observations.
    #
    # merge_asof with allow_exact_matches=False ensures that
    # a candle beginning exactly at the prediction timestamp
    # is not included.
    features = features.reset_index()

    feature_time = features.columns[0]

    target_frame = pd.DataFrame(
        {
            feature_time:
                target_index
        }
    )

    target_frame = target_frame.sort_values(
        feature_time
    )

    features = features.sort_values(
        feature_time
    )

    merged = pd.merge_asof(
        target_frame,
        features,
        on=feature_time,
        direction="backward",
        allow_exact_matches=False,
    )

    merged = merged.set_index(
        feature_time
    )

    return merged.drop(
        columns=[],
        errors="ignore",
    )


# ============================================================
# 1M MICROSTRUCTURE
# ============================================================

def _microstructure_1m(
    df_1m,
    target_index,
):
    """
    Extra 1-minute information.

    This describes what happened immediately before each
    15-minute prediction point.
    """

    if df_1m is None or df_1m.empty:
        return pd.DataFrame(
            index=target_index
        )

    df = _clean_ohlcv(
        df_1m
    )

    out = pd.DataFrame(
        index=df.index
    )

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Very recent momentum.
    for period in [
        1,
        3,
        5,
        10,
        15,
    ]:

        out[
            f"m1_recent_return_{period}"
        ] = (
            close
            / close.shift(period)
            - 1
        )

    # Immediate acceleration.
    one_minute_return = (
        close.pct_change()
    )

    out["m1_acceleration"] = (
        one_minute_return
        - one_minute_return.shift(3)
    )

    # Recent high/low pressure.
    recent_high = high.shift(1).rolling(
        15,
        min_periods=15,
    ).max()

    recent_low = low.shift(1).rolling(
        15,
        min_periods=15,
    ).min()

    out["m1_distance_recent_high"] = (
        _safe_div(
            close - recent_high,
            close,
        )
    )

    out["m1_distance_recent_low"] = (
        _safe_div(
            close - recent_low,
            close,
        )
    )

    # Volume pressure.
    volume_mean = volume.rolling(
        15,
        min_periods=15,
    ).mean()

    out["m1_volume_ratio_15"] = (
        volume
        / (volume_mean + EPS)
    )

    # Number of bullish/bearish candles recently.
    bullish = (
        df["close"] > df["open"]
    ).astype(float)

    bearish = (
        df["close"] < df["open"]
    ).astype(float)

    out["m1_bullish_ratio_15"] = (
        bullish.rolling(
            15,
            min_periods=15,
        ).mean()
    )

    out["m1_bearish_ratio_15"] = (
        bearish.rolling(
            15,
            min_periods=15,
        ).mean()
    )

    # Net directional pressure.
    out["m1_directional_pressure"] = (
        out["m1_bullish_ratio_15"]
        - out["m1_bearish_ratio_15"]
    )

    # Volatility immediately before prediction.
    out["m1_volatility_15"] = (
        one_minute_return.rolling(
            15,
            min_periods=15,
        ).std()
    )

    return _align_features(
        out,
        target_index,
    )


# ============================================================
# ALIGN FEATURES
# ============================================================

def _align_features(
    features,
    target_index,
):
    if features is None or features.empty:
        return pd.DataFrame(
            index=target_index
        )

    features = features.copy()

    features = features[
        ~features.index.duplicated(
            keep="last"
        )
    ].sort_index()

    result = pd.DataFrame(
        {
            "_target_time":
                target_index
        }
    )

    left_time = "_target_time"

    right = features.reset_index()

    right_time = right.columns[0]

    result = result.sort_values(
        left_time
    )

    right = right.sort_values(
        right_time
    )

    merged = pd.merge_asof(
        result,
        right,
        left_on=left_time,
        right_on=right_time,
        direction="backward",
        allow_exact_matches=False,
    )

    merged = merged.set_index(
        "_target_time"
    )

    if right_time in merged.columns:
        merged = merged.drop(
            columns=[right_time]
        )

    return merged


# ============================================================
# 4H CONTEXT
# ============================================================

def _prepare_4h_context(
    df_15m,
    df_4h,
):
    """
    Attach only COMPLETED 4H information.

    The currently building 4H candle is never used.
    """

    if df_4h is None or df_4h.empty:
        return pd.DataFrame(
            index=df_15m.index
        )

    htf = _clean_ohlcv(
        df_4h
    )

    features = pd.DataFrame(
        index=htf.index
    )

    close = htf["close"]

    ema20 = _ema(
        close,
        20,
    )

    ema50 = _ema(
        close,
        50,
    )

    ema100 = _ema(
        close,
        100,
    )

    features["htf_return_1"] = (
        close.pct_change(1)
    )

    features["htf_return_3"] = (
        close.pct_change(3)
    )

    features["htf_return_6"] = (
        close.pct_change(6)
    )

    features["htf_rsi"] = _rsi(
        close,
        14,
    )

    features["htf_ema20_distance"] = (
        _safe_div(
            close - ema20,
            close,
        )
    )

    features["htf_ema50_distance"] = (
        _safe_div(
            close - ema50,
            close,
        )
    )

    features["htf_ema100_distance"] = (
        _safe_div(
            close - ema100,
            close,
        )
    )

    features["htf_ema20_50_spread"] = (
        _safe_div(
            ema20 - ema50,
            close,
        )
    )

    features["htf_trend_up"] = (
        (close > ema20)
        & (ema20 > ema50)
        & (ema50 > ema100)
    ).astype(int)

    features["htf_trend_down"] = (
        (close < ema20)
        & (ema20 < ema50)
        & (ema50 < ema100)
    ).astype(int)

    htf_atr = _atr(
        htf,
        14,
    )

    features["htf_atr_pct"] = (
        _safe_div(
            htf_atr,
            close,
        )
    )

    # Use ONLY the previous completed 4H candle.
    features = features.shift(1)

    return _align_features(
        features,
        df_15m.index,
    )


# ============================================================
# 15M PRIMARY FEATURES
# ============================================================

def _primary_15m_features(df_15m):
    return _base_features(
        df_15m,
        "m15_",
    )


# ============================================================
# CROSS-TIMEFRAME FEATURES
# ============================================================

def _cross_timeframe_features(
    df_1m,
    df_5m,
    df_15m,
):
    """
    Compare momentum across timeframes.

    These features attempt to capture whether lower timeframe
    movement agrees with the primary 15-minute structure.
    """

    target_index = df_15m.index

    out = pd.DataFrame(
        index=target_index
    )

    if (
        df_1m is not None
        and not df_1m.empty
    ):

        m1 = _align_features(
            pd.DataFrame(
                {
                    "m1_close":
                        df_1m["close"]
                }
            ),
            target_index,
        )

        out["cross_m1_price"] = (
            m1["m1_close"]
        )

    if (
        df_5m is not None
        and not df_5m.empty
    ):

        m5 = _align_features(
            pd.DataFrame(
                {
                    "m5_close":
                        df_5m["close"]
                }
            ),
            target_index,
        )

        out["cross_m5_price"] = (
            m5["m5_close"]
        )

    primary_close = df_15m["close"]

    if "cross_m1_price" in out.columns:
        out["m1_vs_m15"] = _safe_div(
            out["cross_m1_price"]
            - primary_close,
            primary_close,
        )

    if "cross_m5_price" in out.columns:
        out["m5_vs_m15"] = _safe_div(
            out["cross_m5_price"]
            - primary_close,
            primary_close,
        )

    return out


# ============================================================
# MAIN FEATURE BUILDER
# ============================================================

def build_features(
    df_15m,
    df_4h=None,
    df_1m=None,
    df_5m=None,
):
    """
    Build the complete multi-timeframe feature matrix.

    Parameters
    ----------
    df_15m:
        Primary 15-minute OHLCV data.

    df_4h:
        Completed/historical 4-hour OHLCV data.

    df_1m:
        1-minute OHLCV data.

    df_5m:
        5-minute OHLCV data.

    Returns
    -------
    pandas.DataFrame
        Feature matrix indexed by 15-minute timestamps.
    """

    _validate_ohlcv(
        df_15m,
        "15m",
    )

    df_15m = _clean_ohlcv(
        df_15m
    )

    if df_4h is not None:
        _validate_ohlcv(
            df_4h,
            "4h",
        )

        df_4h = _clean_ohlcv(
            df_4h
        )

    if df_1m is not None:
        _validate_ohlcv(
            df_1m,
            "1m",
        )

        df_1m = _clean_ohlcv(
            df_1m
        )

    if df_5m is not None:
        _validate_ohlcv(
            df_5m,
            "5m",
        )

        df_5m = _clean_ohlcv(
            df_5m
        )

    target_index = df_15m.index

    parts = []

    # --------------------------------------------------------
    # PRIMARY 15M FEATURES
    # --------------------------------------------------------

    parts.append(
        _primary_15m_features(
            df_15m
        )
    )

    # --------------------------------------------------------
    # 1M FEATURES
    # --------------------------------------------------------

    if df_1m is not None:

        parts.append(
            _aggregate_lower_timeframe(
                df_1m,
                target_index,
                "m1_",
            )
        )

        parts.append(
            _microstructure_1m(
                df_1m,
                target_index,
            )
        )

    # --------------------------------------------------------
    # 5M FEATURES
    # --------------------------------------------------------

    if df_5m is not None:

        parts.append(
            _aggregate_lower_timeframe(
                df_5m,
                target_index,
                "m5_",
            )
        )

    # --------------------------------------------------------
    # 4H CONTEXT
    # --------------------------------------------------------

    if df_4h is not None:

        parts.append(
            _prepare_4h_context(
                df_15m,
                df_4h,
            )
        )

    # --------------------------------------------------------
    # CROSS-TIMEFRAME
    # --------------------------------------------------------

    parts.append(
        _cross_timeframe_features(
            df_1m,
            df_5m,
            df_15m,
        )
    )

    features = pd.concat(
        parts,
        axis=1,
    )

    # Remove duplicate columns defensively.
    features = features.loc[
        :,
        ~features.columns.duplicated()
    ]

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return features


# ============================================================
# NEXT 15M DIRECTION TARGET
# ============================================================

def build_target(
    df_15m,
    neutral_threshold=0.0015,
):
    """
    Direction of the NEXT 15-minute candle.

    0 = bearish
    1 = neutral
    2 = bullish
    """

    _validate_ohlcv(
        df_15m,
        "15m",
    )

    close = df_15m["close"]

    next_return = (
        close.shift(-1)
        / close
        - 1
    )

    target = pd.Series(
        np.nan,
        index=df_15m.index,
        dtype="float64",
    )

    target[
        next_return < -neutral_threshold
    ] = 0

    target[
        next_return.abs()
        <= neutral_threshold
    ] = 1

    target[
        next_return > neutral_threshold
    ] = 2

    return target.rename(
        "target"
    )


# ============================================================
# NEXT CANDLE STRUCTURE TARGET
# ============================================================

def build_structure_target(
    df_15m,
):
    """
    Describes the ACTUAL next 15-minute candle.

    This is deliberately calculated from the next candle,
    making it a training target rather than an input feature.

    Classes:

        0 = OTHER
        1 = HAMMER_LIKE
        2 = SHOOTING_STAR_LIKE
        3 = DOJI_LIKE
        4 = STRONG_BULLISH
        5 = STRONG_BEARISH
        6 = HANGING_MAN_LIKE

    These labels are geometry-based approximations.
    The model will learn their historical conditions rather
    than being told that the pattern guarantees direction.
    """

    _validate_ohlcv(
        df_15m,
        "15m",
    )

    future = df_15m.shift(-1)

    o = future["open"]
    h = future["high"]
    l = future["low"]
    c = future["close"]

    candle_range = (
        h - l
    ).clip(lower=EPS)

    body = (
        c - o
    ).abs()

    upper_wick = (
        h
        - pd.concat(
            [o, c],
            axis=1,
        ).max(axis=1)
    ).clip(lower=0)

    lower_wick = (
        pd.concat(
            [o, c],
            axis=1,
        ).min(axis=1)
        - l
    ).clip(lower=0)

    body_pct = (
        body / candle_range
    )

    upper_pct = (
        upper_wick / candle_range
    )

    lower_pct = (
        lower_wick / candle_range
    )

    bullish = c > o
    bearish = c < o

    structure = pd.Series(
        0,
        index=df_15m.index,
        dtype="float64",
    )

    # Doji-like.
    doji = (
        body_pct <= 0.10
    )

    structure[
        doji
    ] = 3

    # Hammer-like.
    hammer = (
        (lower_wick >= body * 2.0)
        & (upper_wick <= body)
        & (body_pct <= 0.45)
        & (lower_pct >= 0.45)
    )

    structure[
        hammer
    ] = 1

    # Shooting-star-like.
    shooting_star = (
        (upper_wick >= body * 2.0)
        & (lower_wick <= body)
        & (body_pct <= 0.45)
        & (upper_pct >= 0.45)
    )

    structure[
        shooting_star
    ] = 2

    # Strong directional candle.
    strong_bullish = (
        bullish
        & (body_pct >= 0.70)
    )

    strong_bearish = (
        bearish
        & (body_pct >= 0.70)
    )

    structure[
        strong_bullish
    ] = 4

    structure[
        strong_bearish
    ] = 5

    # Hanging-man-like:
    # small-ish body, large lower wick, bearish close.
    hanging_man = (
        bearish
        & (lower_wick >= body * 2.0)
        & (upper_wick <= body)
        & (body_pct <= 0.45)
        & (lower_pct >= 0.45)
    )

    structure[
        hanging_man
    ] = 6

    # The final row has no future candle.
    structure.iloc[-1] = np.nan

    return structure.rename(
        "structure_target"
    )


# ============================================================
# COMPLETE TRAINING DATA
# ============================================================

def prepare_training_data(
    df_15m,
    df_4h=None,
    df_1m=None,
    df_5m=None,
    neutral_threshold=0.0015,
):
    """
    Complete training-data preparation.

    Returns:

        X
            Multi-timeframe feature matrix.

        y
            Direction target.

        structure_y
            Next-candle structure target.
    """

    X = build_features(
        df_15m=df_15m,
        df_4h=df_4h,
        df_1m=df_1m,
        df_5m=df_5m,
    )

    y = build_target(
        df_15m,
        neutral_threshold,
    )

    structure_y = build_structure_target(
        df_15m
    )

    data = X.copy()

    data["target"] = y

    data["structure_target"] = (
        structure_y
    )

    # Last candle has no known future candle.
    data = data.iloc[:-1]

    # Features must exist before training.
    data = data.dropna(
        subset=X.columns
    )

    data = data.dropna(
        subset=[
            "target",
            "structure_target",
        ]
    )

    y = data[
        "target"
    ].astype(int)

    structure_y = data[
        "structure_target"
    ].astype(int)

    X = data.drop(
        columns=[
            "target",
            "structure_target",
        ]
    )

    return (
        X,
        y,
        structure_y,
    )
