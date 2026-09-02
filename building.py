# ============================================================
# NextCandle AI — CURRENT BUILDING CANDLE ENGINE
#
# PURPOSE:
#   Predict the CURRENT 15-minute candle before it finishes.
#
# Example:
#   10:00 -> predict 10:00-10:15 candle
#   10:15 -> compare prediction with actual candle
#
# IMPORTANT:
#   This module is leakage-safe for the OPEN-stage prediction.
#   It only uses information available BEFORE the target 15M
#   candle starts.
#
# Existing features.py is NOT modified.
# ============================================================

import numpy as np
import pandas as pd

from features import (
    _validate_ohlcv,
    _clean_ohlcv,
    _primary_15m_features,
    _aggregate_lower_timeframe,
    _microstructure_1m,
    _prepare_4h_context,
    _align_features,
)


# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-12

BEARISH = 0
NEUTRAL = 1
BULLISH = 2


# ============================================================
# HELPERS
# ============================================================

def _safe_div(a, b):
    return a / (b + EPS)


def _normalise_index(df):
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.set_index("time")
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or time column.")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df.sort_index()


# ============================================================
# CURRENT CANDLE TARGET
# ============================================================

def build_current_candle_target(
    df_15m,
    neutral_threshold=0.0015,
):
    """
    Build the target for the SAME 15M candle.

    Unlike the old target:

        close.shift(-1) / close

    this target measures:

        current_candle_close / current_candle_open - 1

    Therefore:

        0 = bearish
        1 = neutral
        2 = bullish
    """

    df = _normalise_index(df_15m)
    df = _clean_ohlcv(df)

    if neutral_threshold <= 0:
        raise ValueError("neutral_threshold must be greater than zero.")

    candle_return = (
        df["close"] / (df["open"] + EPS)
    ) - 1.0

    target = pd.Series(
        NEUTRAL,
        index=df.index,
        dtype="int64",
        name="target",
    )

    target.loc[candle_return < -neutral_threshold] = BEARISH
    target.loc[candle_return > neutral_threshold] = BULLISH

    return target


# ============================================================
# CURRENT CANDLE STRUCTURE TARGET
# ============================================================

def build_current_structure_target(df_15m):
    """
    Classifies the SAME 15M candle after it has completed.

    0 OTHER
    1 HAMMER_LIKE
    2 SHOOTING_STAR_LIKE
    3 DOJI_LIKE
    4 STRONG_BULLISH
    5 STRONG_BEARISH
    6 HANGING_MAN_LIKE
    """

    df = _normalise_index(df_15m)
    df = _clean_ohlcv(df)

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    candle_range = (h - l).clip(lower=EPS)
    body = (c - o).abs()

    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    body_ratio = body / candle_range
    upper_ratio = upper_wick / candle_range
    lower_ratio = lower_wick / candle_range

    out = pd.Series(
        0,
        index=df.index,
        dtype="int64",
        name="structure_target",
    )

    # DOJI
    out.loc[body_ratio <= 0.10] = 3

    # HAMMER
    hammer = (
        (lower_ratio >= 0.50)
        & (upper_ratio <= 0.25)
        & (body_ratio <= 0.40)
    )
    out.loc[hammer] = 1

    # SHOOTING STAR
    shooting_star = (
        (upper_ratio >= 0.50)
        & (lower_ratio <= 0.25)
        & (body_ratio <= 0.40)
    )
    out.loc[shooting_star] = 2

    # STRONG BULLISH
    strong_bullish = (
        (c > o)
        & (body_ratio >= 0.70)
    )
    out.loc[strong_bullish] = 4

    # STRONG BEARISH
    strong_bearish = (
        (c < o)
        & (body_ratio >= 0.70)
    )
    out.loc[strong_bearish] = 5

    # HANGING MAN
    hanging_man = (
        (lower_ratio >= 0.50)
        & (upper_ratio <= 0.25)
        & (body_ratio <= 0.40)
        & (c < o)
    )
    out.loc[hanging_man] = 6

    return out


# ============================================================
# CROSS-TIMEFRAME FEATURES
# ============================================================

def _build_open_cross_features(
    df_15m,
    df_5m=None,
    df_1m=None,
):
    """
    Build lower-timeframe relationships for prediction at the
    START of the target 15M candle.

    Critical rule:

        Target 15M close is NEVER used.

    Lower timeframe information is shifted so that only completed
    lower-timeframe candles are available at the target 15M start.
    """

    target_index = df_15m.index

    previous_15m_close = df_15m["close"].shift(1)

    frames = []

    # --------------------------------------------------------
    # 1M
    # --------------------------------------------------------

    if df_1m is not None and not df_1m.empty:
        m1 = _clean_ohlcv(df_1m)

        m1_close = m1["close"].shift(1)

        m1_aligned = _align_features(
            target_index,
            m1_close.to_frame("m1_close"),
            "m1_close",
        )

        frames.append(m1_aligned)

    # --------------------------------------------------------
    # 5M
    # --------------------------------------------------------

    if df_5m is not None and not df_5m.empty:
        m5 = _clean_ohlcv(df_5m)

        m5_close = m5["close"].shift(1)

        m5_aligned = _align_features(
            target_index,
            m5_close.to_frame("m5_close"),
            "m5_close",
        )

        frames.append(m5_aligned)

    if not frames:
        return pd.DataFrame(index=target_index)

    result = pd.concat(frames, axis=1)

    # --------------------------------------------------------
    # Relationships to PREVIOUS completed 15M candle
    # --------------------------------------------------------

    if "m1_close" in result.columns:
        result["m1_vs_prev_15m"] = (
            result["m1_close"]
            / (previous_15m_close + EPS)
        ) - 1.0

    if "m5_close" in result.columns:
        result["m5_vs_prev_15m"] = (
            result["m5_close"]
            / (previous_15m_close + EPS)
        ) - 1.0

    if "m1_close" in result.columns and "m5_close" in result.columns:
        result["m1_vs_m5"] = (
            result["m1_close"]
            / (result["m5_close"] + EPS)
        ) - 1.0

    return result


# ============================================================
# OPEN-STAGE FEATURE BUILDER
# ============================================================

def build_open_features(
    df_15m,
    df_5m=None,
    df_1m=None,
    df_4h=None,
):
    """
    Build features available at the START of every 15M candle.

    For target candle T:

        15M features -> previous completed 15M candle
        5M features  -> latest completed 5M information
        1M features  -> latest completed 1M information
        4H features  -> previous completed 4H context

    The target candle's final OHLC is NOT used.
    """

    df_15m = _normalise_index(df_15m)
    df_15m = _clean_ohlcv(df_15m)

    if not _validate_ohlcv(df_15m):
        raise ValueError("Invalid 15M OHLCV data.")

    target_index = df_15m.index

    feature_parts = []

    # ========================================================
    # 15M
    # ========================================================

    primary = _primary_15m_features(df_15m)

    # IMPORTANT:
    #
    # The current target candle is not allowed here.
    #
    # Move previous completed 15M features onto the target
    # candle's timestamp.
    #
    primary = primary.shift(1)

    feature_parts.append(primary)

    # ========================================================
    # 5M
    # ========================================================

    if df_5m is not None and not df_5m.empty:
        df_5m = _normalise_index(df_5m)
        df_5m = _clean_ohlcv(df_5m)

        five_features = _aggregate_lower_timeframe(
            df_5m,
            target_index,
            prefix="m5",
        )

        feature_parts.append(five_features)

    # ========================================================
    # 1M
    # ========================================================

    if df_1m is not None and not df_1m.empty:
        df_1m = _normalise_index(df_1m)
        df_1m = _clean_ohlcv(df_1m)

        one_features = _microstructure_1m(
            df_1m,
            target_index,
        )

        feature_parts.append(one_features)

    # ========================================================
    # 4H
    # ========================================================

    if df_4h is not None and not df_4h.empty:
        df_4h = _normalise_index(df_4h)
        df_4h = _clean_ohlcv(df_4h)

        four_hour_features = _prepare_4h_context(
            df_4h,
            target_index,
        )

        feature_parts.append(four_hour_features)

    # ========================================================
    # CROSS TIMEFRAME
    # ========================================================

    cross = _build_open_cross_features(
        df_15m=df_15m,
        df_5m=df_5m,
        df_1m=df_1m,
    )

    if not cross.empty:
        feature_parts.append(cross)

    # ========================================================
    # COMBINE
    # ========================================================

    features = pd.concat(
        feature_parts,
        axis=1,
    )

    # Remove duplicate columns
    features = features.loc[
        :,
        ~features.columns.duplicated()
    ]

    # Replace invalid values
    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return features


# ============================================================
# TRAINING DATA
# ============================================================

def prepare_open_training_data(
    df_15m,
    df_5m=None,
    df_1m=None,
    df_4h=None,
    neutral_threshold=0.0015,
):
    """
    Prepare leakage-safe training data.

    Each row means:

        "At the START of this 15M candle,
         what will THIS candle eventually become?"

    X:
        Information available before/during the start of
        the target candle.

    y:
        Final direction of the SAME candle.
    """

    features = build_open_features(
        df_15m=df_15m,
        df_5m=df_5m,
        df_1m=df_1m,
        df_4h=df_4h,
    )

    target = build_current_candle_target(
        df_15m=df_15m,
        neutral_threshold=neutral_threshold,
    )

    structure_target = build_current_structure_target(
        df_15m=df_15m,
    )

    # Align everything
    common_index = (
        features.index
        .intersection(target.index)
        .intersection(structure_target.index)
    )

    features = features.loc[common_index]
    target = target.loc[common_index]
    structure_target = structure_target.loc[common_index]

    # Remove rows with no usable feature information
    valid_rows = features.notna().any(axis=1)

    features = features.loc[valid_rows]
    target = target.loc[valid_rows]
    structure_target = structure_target.loc[valid_rows]

    # Drop rows containing NaN feature values.
    #
    # This is important because the earliest rows do not have
    # enough historical candles for all indicators.
    complete_rows = features.notna().all(axis=1)

    features = features.loc[complete_rows]
    target = target.loc[complete_rows]
    structure_target = structure_target.loc[complete_rows]

    if features.empty:
        raise ValueError(
            "No usable training rows remain after feature construction."
        )

    return (
        features,
        target,
        structure_target,
    )


# ============================================================
# INFORMATION REPORT
# ============================================================

def describe_open_training_data(
    X,
    y,
    structure_y=None,
):
    """
    Return a compact diagnostic dictionary.
    """

    result = {
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "start": str(X.index.min()) if len(X) else None,
        "end": str(X.index.max()) if len(X) else None,
        "class_distribution": {
            "bearish": int((y == BEARISH).sum()),
            "neutral": int((y == NEUTRAL).sum()),
            "bullish": int((y == BULLISH).sum()),
        },
    }

    if structure_y is not None:
        result["structure_distribution"] = {
            "other": int((structure_y == 0).sum()),
            "hammer_like": int((structure_y == 1).sum()),
            "shooting_star_like": int((structure_y == 2).sum()),
            "doji_like": int((structure_y == 3).sum()),
            "strong_bullish": int((structure_y == 4).sum()),
            "strong_bearish": int((structure_y == 5).sum()),
            "hanging_man_like": int((structure_y == 6).sum()),
        }

    return result
