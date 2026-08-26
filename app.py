import streamlit as st
import pandas as pd
import numpy as np

from data import fetch_klines
from features import make_features
from model import (
    train_model,
    predict_next,
    walk_forward_backtest
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NextCandle AI V7",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📈 NextCandle AI — V7")

st.caption(
    "Predict the direction of the NEXT 15-minute candle "
    "using only information available before that candle begins."
)


# =========================================================
# SETTINGS
# =========================================================

with st.sidebar:

    st.header("⚙️ V7 Settings")

    symbol = st.text_input(
        "Trading pair",
        "ACE_USDT"
    ).upper().strip()

    timeframe = st.selectbox(
        "Prediction timeframe",
        ["15 minutes"],
        index=0
    )

    history = st.slider(
        "Historical 15M candles",
        1000,
        10000,
        3000,
        step=500
    )

    threshold = st.slider(
        "Minimum confidence",
        0.50,
        0.90,
        0.60,
        0.01
    )

    test_count = st.slider(
        "Historical prediction tests",
        20,
        100,
        80,
        step=10
    )

    run = st.button(
        "🚀 RUN V7 TEST",
        type="primary"
    )


# =========================================================
# BUILD MODEL
# =========================================================

if run or "result" not in st.session_state:

    try:

        # =================================================
        # STEP 1 — DOWNLOAD 15M DATA
        # =================================================

        with st.spinner(
            "📥 Downloading 15M candles..."
        ):

            raw_15m = fetch_klines(
                symbol,
                "Min15",
                history
            )

        if raw_15m is None or len(raw_15m) == 0:

            raise ValueError(
                "MEXC returned no 15M candle data."
            )


        # =================================================
        # STEP 2 — DOWNLOAD 4H DATA
        # =================================================

        with st.spinner(
            "📥 Downloading completed 4H context..."
        ):

            raw_4h = fetch_klines(
                symbol,
                "Hour4",
                500
            )

        if raw_4h is None or len(raw_4h) == 0:

            raise ValueError(
                "MEXC returned no 4H candle data."
            )


        # =================================================
        # STEP 3 — FEATURES
        # =================================================

        with st.spinner(
            "🧮 Building 15M + 4H features..."
        ):

            df = make_features(
                raw_15m,
                raw_4h
            )


        if df is None or len(df) < 700:

            raise ValueError(
                f"Not enough clean candles. "
                f"Only {0 if df is None else len(df)} available."
            )


        # =================================================
        # STEP 4 — TRAIN
        # =================================================

        with st.spinner(
            "🤖 Training V7 ensemble..."
        ):

            models, feature_cols, metrics = train_model(
                df
            )


        # =================================================
        # STEP 5 — LIVE NEXT CANDLE PREDICTION
        # =================================================

        with st.spinner(
            "🎯 Predicting the NEXT 15M candle..."
        ):

            (
                probs,
                ml_signal,
                expected_open,
                predicted_close,
                expected_move_pct
            ) = predict_next(
                models,
                df,
                feature_cols,
                threshold
            )


        # =================================================
        # STEP 6 — STANDARD BACKTEST
        # =================================================

        with st.spinner(
            "🔄 Running chronological backtest..."
        ):

            bt = walk_forward_backtest(
                df,
                feature_cols,
                signal_threshold=threshold
            )


        # =================================================
        # SAVE
        # =================================================

        st.session_state.result = (
            df,
            probs,
            ml_signal,
            expected_open,
            predicted_close,
            expected_move_pct,
            metrics,
            bt
        )

        st.session_state.v7_ready = True


    except Exception as e:

        st.error(
            f"❌ Could not build V7: {e}"
        )

        st.exception(e)

        st.stop()


# =========================================================
# LOAD RESULT
# =========================================================

if "result" not in st.session_state:

    st.warning(
        "Click RUN V7 TEST."
    )

    st.stop()


(
    df,
    probs,
    ml_signal,
    expected_open,
    predicted_close,
    expected_move_pct,
    metrics,
    bt
) = st.session_state.result


latest = df.iloc[-1]


# =========================================================
# CURRENT PRICE
# =========================================================

current_close = float(
    latest["close"]
)


# =========================================================
# 4H CONTEXT
# =========================================================

htf_score = latest.get(
    "htf_bias_score",
    0
)

if pd.isna(htf_score):

    htf_score = 0


if htf_score >= 2:

    htf_bias = "BULLISH"
    htf_icon = "🟢"

elif htf_score <= -2:

    htf_bias = "BEARISH"
    htf_icon = "🔴"

else:

    htf_bias = "NEUTRAL"
    htf_icon = "⚪"


# =========================================================
# NEXT 15M PREDICTION
# =========================================================

st.divider()

st.header(
    "🎯 NEXT 15M CANDLE"
)


best_direction = max(
    probs,
    key=probs.get
)

confidence = float(
    probs[best_direction]
)


# =========================================================
# BIG SIGNAL
# =========================================================

if ml_signal == "BULLISH":

    st.success(
        f"🟢 NEXT 15M CANDLE: BULLISH\n\n"
        f"Model confidence: {confidence * 100:.1f}%"
    )

elif ml_signal == "BEARISH":

    st.error(
        f"🔴 NEXT 15M CANDLE: BEARISH\n\n"
        f"Model confidence: {confidence * 100:.1f}%"
    )

else:

    st.warning(
        f"⚪ NEXT 15M CANDLE: NO EDGE\n\n"
        f"Highest probability: {confidence * 100:.1f}%"
    )


# =========================================================
# PROBABILITIES
# =========================================================

st.subheader(
    "🤖 Direction Probabilities"
)

p1, p2, p3 = st.columns(3)


with p1:

    st.metric(
        "🟢 BULLISH",
        f"{probs['bullish'] * 100:.2f}%"
    )


with p2:

    st.metric(
        "⚪ NEUTRAL",
        f"{probs['neutral'] * 100:.2f}%"
    )


with p3:

    st.metric(
        "🔴 BEARISH",
        f"{probs['bearish'] * 100:.2f}%"
    )


# =========================================================
# PRICE ESTIMATE
# =========================================================

st.divider()

st.subheader(
    "💰 NEXT CANDLE PRICE ESTIMATE"
)

q1, q2, q3 = st.columns(3)


with q1:

    st.metric(
        "Current Price",
        f"{current_close:.8f}"
    )


with q2:

    st.metric(
        "Predicted Close",
        f"{predicted_close:.8f}",
        delta=f"{expected_move_pct:+.3f}%"
    )


with q3:

    st.metric(
        "Expected Move",
        f"{expected_move_pct:+.3f}%"
    )


# =========================================================
# 4H CONTEXT
# =========================================================

st.divider()

st.subheader(
    "🧭 4H MARKET CONTEXT"
)

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "4H Bias",
        f"{htf_icon} {htf_bias}"
    )

with c2:

    st.metric(
        "4H Bias Score",
        f"{htf_score:.0f}"
    )


# =========================================================
# MODEL VALIDATION
# =========================================================

st.divider()

st.header(
    "🧪 V7 VALIDATION"
)

v1, v2, v3, v4 = st.columns(4)


with v1:

    st.metric(
        "Holdout Accuracy",
        f"{metrics.get('holdout_accuracy', 0) * 100:.2f}%"
    )


with v2:

    st.metric(
        "Backtest Accuracy",
        f"{bt.get('accuracy', 0) * 100:.2f}%"
    )


with v3:

    st.metric(
        "High Confidence",
        f"{bt.get('signal_accuracy', 0) * 100:.2f}%"
    )


with v4:

    st.metric(
        "Test Predictions",
        f"{bt.get('predictions', 0):,}"
    )


# =========================================================
# BULLISH / BEARISH BACKTEST
# =========================================================

st.subheader(
    "📊 Direction Performance"
)

b1, b2, b3 = st.columns(3)


with b1:

    st.metric(
        "🟢 Bullish Tests",
        bt.get(
            "bullish_signals",
            0
        )
    )

    st.write(
        "Accuracy: "
        f"**{bt.get('bullish_accuracy', 0) * 100:.2f}%**"
    )


with b2:

    st.metric(
        "🔴 Bearish Tests",
        bt.get(
            "bearish_signals",
            0
        )
    )

    st.write(
        "Accuracy: "
        f"**{bt.get('bearish_accuracy', 0) * 100:.2f}%**"
    )


with b3:

    st.metric(
        "⚪ Neutral Tests",
        bt.get(
            "neutral_signals",
            0
        )
    )

    st.write(
        "Accuracy: "
        f"**{bt.get('neutral_accuracy', 0) * 100:.2f}%**"
    )


# =========================================================
# 80-TEST EXPLANATION
# =========================================================

st.divider()

st.header(
    "🎯 THE TEST WE ACTUALLY CARE ABOUT"
)

st.write(
    f"We want to test whether V7 can correctly predict "
    f"the direction of the next 15M candle."
)

st.write(
    f"Target test size: **{test_count} historical candles**."
)

st.info(
    "Example: if V7 says BULLISH before a candle starts "
    "and that candle closes above its open, the prediction "
    "is CORRECT. If it closes below its open, the prediction "
    "is WRONG."
)


# =========================================================
# RECENT CANDLES
# =========================================================

st.divider()

st.subheader(
    f"🕯️ Recent {symbol} 15M Candles"
)

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
    df[available_columns].tail(20),
    use_container_width=True
)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader(
    "🧠 What V7 Is Doing"
)

st.write(
    "1. Uses completed 15M candles as the main information."
)

st.write(
    "2. Uses completed 4H candles only as higher-timeframe context."
)

st.write(
    "3. Trains the model chronologically so future candles "
    "are not deliberately given to the past."
)

st.write(
    "4. Predicts the NEXT 15M candle."
)

st.write(
    "5. We compare the prediction with what the next candle "
    "actually did."
)


# =========================================================
# FINAL WARNING
# =========================================================

st.warning(
    "⚠️ Do NOT judge V7 from one live prediction. "
    "We need a large historical test first. "
    "If it cannot consistently beat random direction "
    "on unseen candles, we fix the model before V8."
)


st.caption(
    "Research/paper-trading only. "
    "AI predictions are estimates and are not guarantees."
)
