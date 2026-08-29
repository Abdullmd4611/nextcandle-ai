import numpy as np
import pandas as pd


# ============================================================
# NextCandle AI — Feature Engineering V3.1
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
#
# V3.1 FIX:
#     Automatically normalizes candle timestamps into a
#     pandas DatetimeIndex before feature engineering.
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

    return (series - mean) / (
