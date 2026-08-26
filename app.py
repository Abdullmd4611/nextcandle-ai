import streamlit as st
import pandas as pd
import numpy as np

from data import fetch_klines
from features import make_features
from model import train_model, predict_next, walk_forward_backtest


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="NextCandle AI V2",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NextCandle AI — V2")

st.caption(
    "ACEUSDT 15-minute next-candle prediction "
    "with 4-hour higher-timeframe bias. "
    "Research/paper-trading only."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    symbol = st.text_input(
        "Trading pair",
        "ACE_USDT"
    ).upper().strip()

    st.selectbox(
        "Prediction timeframe",
        ["15 minutes"],
        index=0
    )

    interval = "Min15"

    history = st.slider(
        "15M historical candles",
        1000,
        10000,
        5000,
        step=500
    )

    threshold = st.slider(
        "Minimum probability for signal",
        0.50,
        0.90,
        0.65,
        0.01
    )

    run = st.button(
        "🚀 Run / Refresh",
        type="primary"
    )


# =========================================================
# BUILD MODEL
# =========================================================

if run or "result" not in st.session_state:

    with st.spinner(
        "Downloading ACEUSDT 15M + 4H data and training V2..."
    ):

        try:

            # -------------------------------------------------
            # 15M DATA
            # -------------------------------------------------

            raw_15m = fetch_klines(
                symbol,
                "Min15",
                history
            )

            # -------------------------------------------------
            # 4H DATA
            # -------------------------------------------------

            raw_4h = fetch_klines(
                symbol,
                "Hour4",
                500
            )

            # -------------------------------------------------
            # FEATURES
            # -------------------------------------------------

            df = make_features(
                raw_15m,
                raw_4h
            )

            if len(df) < 300:

                raise ValueError(
                    "Not enough clean historical data "
                    "after feature calculation."
                )

            # -------------------------------------------------
            # TRAIN
            # -------------------------------------------------

            model, feature_cols, metrics = train_model(
                df
            )

            # -------------------------------------------------
            # NEXT 15M PREDICTION
            # -------------------------------------------------

            probs, signal = predict_next(
                model,
                df,
                feature_cols,
                threshold
            )

            # -------------------------------------------------
            # BACKTEST
            # -------------------------------------------------

            bt = walk_forward_backtest(
                df,
                feature_cols,
                signal_threshold=threshold
            )

            st.session_state.result = (
                df,
                probs,
                signal,
                metrics,
                bt
            )

        except Exception as e:

            st.error(
                f"❌ Could not build V2 model: {e}"
            )

            st.stop()


# =========================================================
# RESULTS
# =========================================================

df, probs, signal, metrics, bt = (
    st.session_state.result
)


latest = df.iloc[-1]


# =========================================================
# DETERMINE 4H BIAS
# =========================================================

if "htf_bias_score" in df.columns:

    htf_score = latest["htf_bias_score"]

    if htf_score >= 2:

        htf_bias = "BULLISH"
        htf_icon = "🟢"

    elif htf_score <= -2:

        htf_bias = "BEARISH"
        htf_icon = "🔴"

    else:

        htf_bias = "NEUTRAL"
        htf_icon = "⚪"

else:

    htf_bias = "UNKNOWN"
    htf_icon = "⚪"


# =========================================================
# MAIN PREDICTION
# =========================================================

st.divider()

st.subheader("🎯 NEXT 15M CANDLE PREDICTION")


if signal == "BULLISH":

    confidence = probs["bullish"]

    st.success(
        f"🟢 NEXT 15M: BULLISH"
    )

elif signal == "BEARISH":

    confidence = probs["bearish"]

    st.error(
        f"🔴 NEXT 15M: BEARISH"
    )

else:

    best_direction = max(
        probs,
        key=probs.get
    )

    confidence = probs[best_direction]

    st.warning(
        "⚪ NEXT 15M: NO EDGE — WAIT"
    )


# =========================================================
# KEY METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Confidence",
    f"{confidence * 100:.1f}%"
)

c2.metric(
    "🧭 4H Bias",
    f"{htf_icon} {htf_bias}"
)

c3.metric(
    "🟢 Bullish",
    f"{probs['bullish'] * 100:.1f}%"
)

c4.metric(
    "🔴 Bearish",
    f"{probs['bearish'] * 100:.1f}%"
)


# =========================================================
# PROBABILITY BREAKDOWN
# =========================================================

st.subheader("📊 Probability Breakdown")

p1, p2, p3 = st.columns(3)

with p1:

    st.metric(
        "🟢 Bullish",
        f"{probs['bullish'] * 100:.2f}%"
    )

with p2:

    st.metric(
        "⚪ Neutral",
        f"{probs['neutral'] * 100:.2f}%"
    )

with p3:

    st.metric(
        "🔴 Bearish",
        f"{probs['bearish'] * 100:.2f}%"
    )


# =========================================================
# SUPPORTING FACTORS
# =========================================================

st.divider()

st.subheader("🔎 Supporting Factors")


factors = []


# ---------------------------------------------------------
# 4H TREND
# ---------------------------------------------------------

if htf_bias == "BULLISH":

    factors.append(
        "🟢 4H trend is bullish"
    )

elif htf_bias == "BEARISH":

    factors.append(
        "🔴 4H trend is bearish"
    )

else:

    factors.append(
        "⚪ 4H trend is mixed/neutral"
    )


# ---------------------------------------------------------
# 15M MOMENTUM
# ---------------------------------------------------------

ret5 = latest.get(
    "ret5",
    np.nan
)

if pd.notna(ret5):

    if ret5 > 0:

        factors.append(
            "🟢 15M momentum is bullish"
        )

    elif ret5 < 0:

        factors.append(
            "🔴 15M momentum is bearish"
        )

    else:

        factors.append(
            "⚪ 15M momentum is flat"
        )


# ---------------------------------------------------------
# EMA STRUCTURE
# ---------------------------------------------------------

ema_gap = latest.get(
    "ema9_20_gap",
    np.nan
)

if pd.notna(ema_gap):

    if ema_gap > 0:

        factors.append(
            "🟢 EMA 9 is above EMA 20"
        )

    elif ema_gap < 0:

        factors.append(
            "🔴 EMA 9 is below EMA 20"
        )

    else:

        factors.append(
            "⚪ EMA 9 / EMA 20 are neutral"
        )


# ---------------------------------------------------------
# RSI
# ---------------------------------------------------------

rsi = latest.get(
    "rsi14",
    np.nan
)

if pd.notna(rsi):

    if rsi >= 55:

        factors.append(
            f"🟢 RSI supports bullish momentum ({rsi:.1f})"
        )

    elif rsi <= 45:

        factors.append(
            f"🔴 RSI supports bearish momentum ({rsi:.1f})"
        )

    else:

        factors.append(
            f"⚪ RSI is neutral ({rsi:.1f})"
        )


# ---------------------------------------------------------
# VOLUME
# ---------------------------------------------------------

volume_ratio = latest.get(
    "volume_ratio20",
    np.nan
)

if pd.notna(volume_ratio):

    if volume_ratio > 1.20:

        factors.append(
            f"🟢 Volume is above average ({volume_ratio:.2f}x)"
        )

    elif volume_ratio < 0.80:

        factors.append(
            f"⚪ Volume is below average ({volume_ratio:.2f}x)"
        )

    else:

        factors.append(
            f"⚪ Volume is near average ({volume_ratio:.2f}x)"
        )


# ---------------------------------------------------------
# CANDLE STRUCTURE
# ---------------------------------------------------------

body = latest.get(
    "body",
    np.nan
)

if pd.notna(body):

    if body > 0:

        factors.append(
            "🟢 Latest 15M candle closed bullish"
        )

    elif body < 0:

        factors.append(
            "🔴 Latest 15M candle closed bearish"
        )

    else:

        factors.append(
            "⚪ Latest 15M candle closed neutral"
        )


for factor in factors:

    st.write(factor)


# =========================================================
# 4H DETAILS
# =========================================================

st.divider()

st.subheader("🧭 4-Hour Market Bias")


if "htf_ema_gap20" in df.columns:

    htf_gap = latest["htf_ema_gap20"] * 100

    st.write(
        f"4H EMA20 distance: **{htf_gap:.2f}%**"
    )

    st.write(
        f"4H bias score: **{htf_score:.0f}**"
    )

else:

    st.warning(
        "4H bias data unavailable."
    )


# =========================================================
# MODEL VALIDATION
# =========================================================

left, right = st.columns(2)


with left:

    st.subheader("🧪 Model Validation")

    st.write(
        f"Holdout accuracy: "
        f"**{metrics['holdout_accuracy'] * 100:.2f}%**"
    )

    st.write(
        f"Holdout samples: "
        f"**{metrics['holdout_samples']:,}**"
    )

    st.write(
        f"Training samples: "
        f"**{metrics['train_samples']:,}**"
    )

    st.caption(
        "Chronological validation is used. "
        "The data is not randomly shuffled."
    )


with right:

    st.subheader("🔄 Walk-Forward Backtest")

    st.write(
        f"Predictions: "
        f"**{bt['predictions']:,}**"
    )

    st.write(
        f"Direction accuracy: "
        f"**{bt['accuracy'] * 100:.2f}%**"
    )

    st.write(
        f"Signals ≥ selected threshold: "
        f"**{bt['signals']:,}**"
    )

    if bt["signals"]:

        st.write(
            f"Signal accuracy: "
            f"**{bt['signal_accuracy'] * 100:.2f}%**"
        )

    else:

        st.write(
            "No high-confidence signals "
            "in the backtest window."
        )


# =========================================================
# RECENT CANDLES
# =========================================================

st.divider()

st.subheader("🕯️ Recent ACEUSDT 15M Candles")

display_columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

available_columns = [
    c
    for c in display_columns
    if c in df.columns
]

st.dataframe(
    df[
        available_columns
    ].tail(20),
    use_container_width=True
)


# =========================================================
# DISCLAIMER
# =========================================================

st.caption(
    "V2 does not place orders. Model probabilities are "
    "estimates based on historical data and are not guarantees "
    "of future price movement."
)
