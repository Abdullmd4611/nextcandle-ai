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

    st.selectbox(
        "Prediction timeframe",
        ["15 minutes"],
        index=0
    )

    history = st.slider(
        "Historical 15M candles",
        min_value=1000,
        max_value=10000,
        value=3000,
        step=500
    )

    threshold = st.slider(
        "Minimum confidence",
        min_value=0.50,
        max_value=0.90,
        value=0.60,
        step=0.01
    )

    # =====================================================
    # FIXED TEST SIZE
    # =====================================================

    test_count = 80

    st.info(
        "🎯 Historical test size: 80 candles"
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
        # STEP 3 — BUILD FEATURES
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
        # STEP 4 — TRAIN LIVE V7 MODEL
        # =================================================

        with st.spinner(
            "🤖 Training V7 ensemble..."
        ):

            models, feature_cols, metrics = train_model(
                df
            )


        # =================================================
        # STEP 5 — LIVE NEXT 15M PREDICTION
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
        # STEP 6 — EXACT 80-CANDLE WALK-FORWARD TEST
        # =================================================

        with st.spinner(
            "🔄 Testing the last 80 historical candles..."
        ):

            bt = walk_forward_backtest(
                df,
                feature_cols,
                n_tests=test_count,
                min_train=1000,
                signal_threshold=threshold
            )


        # =================================================
        # SAVE RESULT
        # =================================================

        st.session_state.result = (
            df,
            probs,
            ml_signal,
            expected_open,
            predicted_close,
            expected_move_pct,
            metrics,
            bt,
            symbol,
            threshold
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
        "Click 🚀 RUN V7 TEST."
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
    bt,
    result_symbol,
    result_threshold
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
# 4H MARKET CONTEXT
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
# V7 VALIDATION
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
        "80-Test Accuracy",
        f"{bt.get('accuracy', 0) * 100:.2f}%"
    )


with v3:

    st.metric(
        "High-Confidence Accuracy",
        f"{bt.get('signal_accuracy', 0) * 100:.2f}%"
    )


with v4:

    st.metric(
        "Tests Completed",
        f"{bt.get('predictions', 0):,}"
    )


# =========================================================
# ACTUAL 80-TEST RESULT
# =========================================================

st.divider()

st.header(
    "🎯 THE 80-CANDLE TEST"
)

predictions_done = bt.get(
    "predictions",
    0
)

overall_accuracy = bt.get(
    "accuracy",
    0
)

correct_predictions = round(
    predictions_done * overall_accuracy
)

wrong_predictions = (
    predictions_done
    - correct_predictions
)


st.write(
    f"V7 tested **{predictions_done} historical candles**."
)

st.write(
    f"✅ Correct predictions: "
    f"**{correct_predictions}**"
)

st.write(
    f"❌ Wrong predictions: "
    f"**{wrong_predictions}**"
)

st.write(
    f"📊 Accuracy: "
    f"**{overall_accuracy * 100:.2f}%**"
)


# =========================================================
# VERDICT
# =========================================================

if predictions_done >= 80:

    if overall_accuracy >= 0.70:

        st.success(
            "🟢 V7 PASSED the 80-candle test "
            "with at least 70% accuracy."
        )

    elif overall_accuracy >= 0.55:

        st.warning(
            "🟡 V7 is above random direction, "
            "but the edge is not strong enough yet."
        )

    else:

        st.error(
            "🔴 V7 FAILED the 80-candle test. "
            "We should improve the model before V8."
        )

else:

    st.warning(
        f"⚠️ Only {predictions_done} tests completed. "
        "We need 80 completed tests before judging V7."
    )


# =========================================================
# DIRECTION PERFORMANCE
# =========================================================

st.divider()

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
        f"**{bt.get('neutral
