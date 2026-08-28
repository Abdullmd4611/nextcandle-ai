import numpy as np
import pandas as pd


# ============================================================
# NextCandle AI — Feature Engineering V2
# Primary timeframe: 15 minutes
# Higher timeframe context: 4 hours
# Target: next 15-minute candle behavior
# ============================================================


EPS = 1e-12


def _safe_div(a, b):
    """Division that prevents zero/invalid values."""
    return a / (b.replace(0, np.nan) + EPS)


def _atr(df, period=14):
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period, min_periods=period).mean()


def _rsi(series, period=14):
    """Wilder-style RSI."""
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


def _candle_features(df):
    """
    Features describing the actual shape/behavior of each candle.
    """

    out = pd.DataFrame(index=df.index)

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    candle_range = (h - l).clip(lower=EPS)
    body = (c - o).abs()

    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

    out["body_pct"] = body / candle_range
    out["upper_wick_pct"] = upper_wick / candle_range
    out["lower_wick_pct"] = lower_wick / candle_range

    out["bullish_candle"] = (c > o).astype(int)
    out["bearish_candle"] = (c < o).astype(int)

    out["range_pct"] = candle_range / (c.shift(1).abs() + EPS)

    out["body_direction"] = (c - o) / candle_range

    out["close_position"] = (c - l) / candle_range

    out["open_position"] = (o - l) / candle_range

    return out


def _momentum_features(df):
    """
    Price momentum and trend features.
    """

    out = pd.DataFrame(index=df.index)

    close = df["close"]

    for period in [1, 2, 3, 4, 8, 12, 20, 32, 48]:

        out[f"return_{period}"] = (
            close / close.shift(period) - 1
        )

    out["rsi_7"] = _rsi(close, 7)
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_28"] = _rsi(close, 28)

    ema_9 = _ema(close, 9)
    ema_21 = _ema(close, 21)
    ema_50 = _ema(close, 50)
    ema_100 = _ema(close, 100)

    out["ema9_distance"] = _safe_div(close - ema_9, close)
    out["ema21_distance"] = _safe_div(close - ema_21, close)
    out["ema50_distance"] = _safe_div(close - ema_50, close)
    out["ema100_distance"] = _safe_div(close - ema_100, close)

    out["ema9_21_spread"] = _safe_div(
        ema_9 - ema_21,
        close,
    )

    out["ema21_50_spread"] = _safe_div(
        ema_21 - ema_50,
        close,
    )

    out["ema50_100_spread"] = _safe_div(
        ema_50 - ema_100,
        close,
    )

    return out


def _volatility_features(df):
    """
    Volatility and expansion/compression measurements.
    """

    out = pd.DataFrame(index=df.index)

    close = df["close"]

    atr14 = _atr(df, 14)
    atr28 = _atr(df, 28)

    out["atr_pct"] = _safe_div(atr14, close)
    out["atr_ratio"] = _safe_div(atr14, atr28)

    returns = close.pct_change()

    out["volatility_8"] = returns.rolling(
        8,
        min_periods=8,
    ).std()

    out["volatility_20"] = returns.rolling(
        20,
        min_periods=20,
    ).std()

    out["volatility_48"] = returns.rolling(
        48,
        min_periods=48,
    ).std()

    out["volatility_zscore"] = _rolling_zscore(
        returns.rolling(20).std(),
        48,
    )

    rolling_high = df["high"].rolling(
        20,
        min_periods=20,
    ).max()

    rolling_low = df["low"].rolling(
        20,
        min_periods=20,
    ).min()

    out["range_20_position"] = _safe_div(
        close - rolling_low,
        rolling_high - rolling_low,
    )

    return out


def _volume_features(df):
    """
    Volume behavior.
    """

    out = pd.DataFrame(index=df.index)

    volume = df["volume"]

    volume_mean_20 = volume.rolling(
        20,
        min_periods=20,
    ).mean()

    volume_std_20 = volume.rolling(
        20,
        min_periods=20,
    ).std()

    out["volume_ratio"] = _safe_div(
        volume,
        volume_mean_20,
    )

    out["volume_zscore"] = (
        volume - volume_mean_20
    ) / (volume_std_20 + EPS)

    out["volume_change_1"] = volume.pct_change()
    out["volume_change_4"] = volume.pct_change(4)

    # Money-flow-style directional pressure.
    candle_range = (
        df["high"] - df["low"]
    ).clip(lower=EPS)

    close_location = (
        (df["close"] - df["low"]) / candle_range
    )

    out["volume_pressure"] = (
        (close_location - 0.5) * 2
    ) * out["volume_ratio"]

    return out


def _structure_features(df):
    """
    Market structure features.
    """

    out = pd.DataFrame(index=df.index)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_high = high.shift(1)
    previous_low = low.shift(1)

    rolling_high_10 = high.shift(1).rolling(
        10,
        min_periods=10,
    ).max()

    rolling_low_10 = low.shift(1).rolling(
        10,
        min_periods=10,
    ).min()

    rolling_high_20 = high.shift(1).rolling(
        20,
        min_periods=20,
    ).max()

    rolling_low_20 = low.shift(1).rolling(
        20,
        min_periods=20,
    ).min()

    # Break of previous local structure.
    out["break_high_10"] = (
        close > rolling_high_10
    ).astype(int)

    out["break_low_10"] = (
        close < rolling_low_10
    ).astype(int)

    out["break_high_20"] = (
        close > rolling_high_20
    ).astype(int)

    out["break_low_20"] = (
        close < rolling_low_20
    ).astype(int)

    # Distance from recent structure.
    out["distance_high_10"] = _safe_div(
        close - rolling_high_10,
        close,
    )

    out["distance_low_10"] = _safe_div(
        close - rolling_low_10,
        close,
    )

    # Candle-to-candle directional persistence.
    direction = np.sign(close.diff())

    out["direction_3_sum"] = direction.rolling(
        3,
        min_periods=3,
    ).sum()

    out["direction_6_sum"] = direction.rolling(
        6,
        min_periods=6,
    ).sum()

    out["direction_12_sum"] = direction.rolling(
        12,
        min_periods=12,
    ).sum()

    # Higher highs / lower lows.
    out["higher_high"] = (
        high > previous_high
    ).astype(int)

    out["lower_low"] = (
        low < previous_low
    ).astype(int)

    return out


def _time_features(df):
    """
    Cyclical time-of-day/week information.
    """

    out = pd.DataFrame(index=df.index)

    if not isinstance(df.index, pd.DatetimeIndex):
        return out

    minutes = (
        df.index.hour * 60
        + df.index.minute
    )

    day_fraction = minutes / 1440.0

    out["time_sin"] = np.sin(
        2 * np.pi * day_fraction
    )

    out["time_cos"] = np.cos(
        2 * np.pi * day_fraction
    )

    weekday_fraction = df.index.dayofweek / 7.0

    out["weekday_sin"] = np.sin(
        2 * np.pi * weekday_fraction
    )

    out["weekday_cos"] = np.cos(
        2 * np.pi * weekday_fraction
    )

    return out


def _prepare_4h_context(df_15m, df_4h):
    """
    Attach ONLY completed 4H information to each 15M candle.

    Important:
    The 4H candle currently building must NEVER be used.

    We therefore shift the 4H feature set by one completed candle
    before using merge_asof.
    """

    if df_4h is None or df_4h.empty:
        return pd.DataFrame(index=df_15m.index)

    htf = df_4h.copy()

    htf = htf.sort_index()

    htf_features = pd.DataFrame(index=htf.index)

    close = htf["close"]

    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    ema100 = _ema(close, 100)

    htf_features["htf_return_1"] = close.pct_change(1)
    htf_features["htf_return_3"] = close.pct_change(3)
    htf_features["htf_return_6"] = close.pct_change(6)

    htf_features["htf_rsi"] = _rsi(close, 14)

    htf_features["htf_ema20_distance"] = _safe_div(
        close - ema20,
        close,
    )

    htf_features["htf_ema50_distance"] = _safe_div(
        close - ema50,
        close,
    )

    htf_features["htf_ema100_distance"] = _safe_div(
        close - ema100,
        close,
    )

    htf_features["htf_ema20_50_spread"] = _safe_div(
        ema20 - ema50,
        close,
    )

    htf_features["htf_trend_up"] = (
        (close > ema20)
        & (ema20 > ema50)
        & (ema50 > ema100)
    ).astype(int)

    htf_features["htf_trend_down"] = (
        (close < ema20)
        & (ema20 < ema50)
        & (ema50 < ema100)
    ).astype(int)

    htf_atr = _atr(htf, 14)

    htf_features["htf_atr_pct"] = _safe_div(
        htf_atr,
        close,
    )

    # CRITICAL:
    # Only the previous COMPLETED 4H candle is allowed.
    htf_features = htf_features.shift(1)

    htf_features = htf_features.reset_index()

    left = df_15m.reset_index()

    time_column = left.columns[0]

    merged = pd.merge_asof(
        left.sort_values(time_column),
        htf_features.sort_values(time_column),
        on=time_column,
        direction="backward",
        allow_exact_matches=True,
    )

    merged = merged.set_index(time_column)

    return merged[
        htf_features.columns.drop(time_column)
    ]


def build_features(
    df_15m,
    df_4h=None,
):
    """
    Main feature-building function.

    Parameters
    ----------
    df_15m : pandas.DataFrame
        15-minute OHLCV data.

    df_4h : pandas.DataFrame, optional
        4-hour OHLCV data.

    Returns
    -------
    pandas.DataFrame
        Feature matrix.
    """

    required_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df_15m.columns
    ]

    if missing:
        raise ValueError(
            f"15m data missing columns: {missing}"
        )

    df = df_15m.copy()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    df = df.sort_index()

    features = pd.DataFrame(index=df.index)

    features = pd.concat(
        [
            features,
            _candle_features(df),
            _momentum_features(df),
            _volatility_features(df),
            _volume_features(df),
            _structure_features(df),
            _time_features(df),
        ],
        axis=1,
    )

    if df_4h is not None:
        htf_features = _prepare_4h_context(
            df,
            df_4h,
        )

        features = pd.concat(
            [
                features,
                htf_features,
            ],
            axis=1,
        )

    # Replace infinities generated by unusual market data.
    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Do NOT forward-fill feature values here.
    #
    # Forward filling can accidentally carry information across
    # invalid/missing observations. The training pipeline should
    # explicitly decide how to handle missing rows.
    return features


def build_target(
    df_15m,
    neutral_threshold=0.0015,
):
    """
    Build the target for the NEXT 15-minute candle.

    Classes:
        0 = bearish
        1 = neutral
        2 = bullish

    The target is based ONLY on the next candle's close
    relative to the current candle's close.

    neutral_threshold:
        Minimum next-candle percentage movement required
        before calling it bullish/bearish.

    Example:
        +0.20% -> bullish
        -0.30% -> bearish
        +0.05% -> neutral
    """

    if "close" not in df_15m.columns:
        raise ValueError(
            "15m data must contain 'close'."
        )

    close = df_15m["close"]

    next_return = (
        close.shift(-1) / close - 1
    )

    target = pd.Series(
        np.nan,
        index=df_15m.index,
        dtype="float64",
    )

    target[next_return < -neutral_threshold] = 0
    target[
        next_return.abs() <= neutral_threshold
    ] = 1
    target[next_return > neutral_threshold] = 2

    return target.rename("target")


def prepare_training_data(
    df_15m,
    df_4h=None,
    neutral_threshold=0.0015,
):
    """
    Complete training-data preparation.

    Returns:
        X = features
        y = next-candle target
    """

    X = build_features(
        df_15m=df_15m,
        df_4h=df_4h,
    )

    y = build_target(
        df_15m=df_15m,
        neutral_threshold=neutral_threshold,
    )

    data = X.copy()
    data["target"] = y

    # The last 15m candle has no known NEXT candle yet.
    data = data.iloc[:-1]

    # Remove rows where required features aren't available.
    data = data.dropna(
        subset=X.columns
    )

    y = data["target"].astype(int)

    X = data.drop(
        columns=["target"]
    )

    return X, y
