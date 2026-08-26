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
        0.65,
        0.01
    )

    test_count = st.slider(
        "Historical prediction tests",
        80,
        80,
        80
    )

    run = st.button(
        "🚀 RUN V7 TEST",
        type="primary",
        use_container_width=True
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
            f"📥 Downloading {symbol} 15M candles..."
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


        if df is None or len(df) < 1100:

            raise ValueError(
                f"Not enough clean candles. "
                f"Only {0 if df is None else len(df)} available."
            )


        # =================================================
        # STEP 4 — TRAIN LIVE V7
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
            "🧪 Running the 80 historical candle reality test..."
        ):

            bt = walk_forward_backtest(
                df,
                feature_cols,
                n_tests=80,
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
# 80-TEST SCORE
# =========================================================

st.divider()

st.header(
    "🎯 THE 80-CANDLE REALITY TEST"
)

total_tests = int(
    bt.get(
        "predictions",
        0
    )
)

overall_accuracy = float(
    bt.get(
        "accuracy",
        0
    )
)

correct_tests = int(
    round(
        total_tests
        * overall_accuracy
    )
)

wrong_tests = (
    total_tests
    - correct_tests
)


r1, r2, r3, r4 = st.columns(4)


with r1:

    st.metric(
        "🧪 Tests",
        total_tests
    )


with r2:

    st.metric(
        "✅ Correct",
        correct_tests
    )


with r3:

    st.metric(
        "❌ Wrong",
        wrong_tests
    )


with r4:

    st.metric(
        "🎯 Accuracy",
        f"{overall_accuracy * 100:.2f}%"
    )


# =========================================================
# PASS / FAIL
# =========================================================

if total_tests >= 80:

    if correct_tests >= 70:

        st.success(
            f"🔥 V7 PASSES the 80-candle test: "
            f"{correct_tests}/80 correct "
            f"({overall_accuracy * 100:.2f}%)."
        )

    elif correct_tests >= 64:

        st.warning(
            f"⚠️ V7 is promising but not strong enough yet: "
            f"{correct_tests}/80 correct "
            f"({overall_accuracy * 100:.2f}%)."
        )

    else:

        st.error(
            f"❌ V7 FAILS the current target: "
            f"{correct_tests}/80 correct "
            f"({overall_accuracy * 100:.2f}%)."
        )

else:

    st.warning(
        f"Only {total_tests} tests completed. "
        "We need all 80 tests before judging V7."
    )


# =========================================================
# HIGH-CONFIDENCE SIGNAL TEST
# =========================================================

st.subheader(
    "🔥 HIGH-CONFIDENCE SIGNAL TEST"
)

signals = int(
    bt.get(
        "signals",
        0
    )
)

signal_accuracy = float(
    bt.get(
        "signal_accuracy",
        0
    )
)

s1, s2, s3 = st.columns(3)


with s1:

    st.metric(
        "High-Confidence Signals",
        signals
    )


with s2:

    st.metric(
        "Signal Accuracy",
        f"{signal_accuracy * 100:.2f}%"
    )


with s3:

    if signals > 0:

        signal_correct = int(
            round(
                signals
                * signal_accuracy
            )
        )

    else:

        signal_correct = 0

    st.metric(
        "Correct Signals",
        signal_correct
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

    bullish_count = int(
        bt.get(
            "bullish_signals",
            0
        )
    )

    bullish_accuracy = float(
        bt.get(
            "bullish_accuracy",
            0
        )
    )

    st.metric(
        "🟢 Bullish Tests",
        bullish_count
    )

    st.write(
        f"Accuracy: "
        f"**{bullish_accuracy * 100:.2f}%**"
    )


with b2:

    bearish_count = int(
        bt.get(
            "bearish_signals",
            0
        )
    )

    bearish_accuracy = float(
        bt.get(
            "bearish_accuracy",
            0
        )
    )

    st.metric(
        "🔴 Bearish Tests",
        bearish_count
    )

    st.write(
        f"Accuracy: "
        f"**{bearish_accuracy * 100:.2f}%**"
    )


with b3:

    neutral_count = int(
        bt.get(
            "neutral_signals",
            0
        )
    )

    neutral_accuracy = float(
        bt.get(
            "neutral_accuracy",
            0
        )
    )

    st.metric(
        "⚪ Neutral Tests",
        neutral_count
    )

    st.write(
        f"Accuracy: "
        f"**{neutral_accuracy * 100:.2f}%**"
    )


# =========================================================
# INDIVIDUAL 80 TESTS
# =========================================================

st.divider()

st.header(
    "🧾 INDIVIDUAL 80-CANDLE RESULTS"
)

test_results = bt.get(
    "test_results",
    []
)


if test_results:

    results_df = pd.DataFrame(
        test_results
    )

    # -----------------------------------------------------
    # Friendly display columns
    # -----------------------------------------------------

    display_results = results_df.copy()

    if "timestamp" in display_results.columns:

        display_results["timestamp"] = (
            pd.to_datetime(
                display_results["timestamp"],
                utc=True,
                errors="coerce"
            )
            .dt.strftime(
                "%Y-%m-%d %H:%M UTC"
            )
        )

    if "confidence" in display_results.columns:

        display_results["confidence"] = (
            display_results["confidence"]
            * 100
        ).round(2)

        display_results = display_results.rename(
            columns={
                "confidence": "confidence_%"
            }
        )

    if "correct" in display_results.columns:

        display_results["correct"] = (
            display_results["correct"]
            .map({
                True: "✅",
                False: "❌"
            })
        )

    if "signal_correct" in display_results.columns:

        display_results["signal_correct"] = (
            display_results["signal_correct"]
            .map({
                True: "✅",
                False: "❌"
            })
        )

    display_results = display_results.rename(
        columns={
            "test": "Test",
            "timestamp": "Prediction Time",
            "prediction": "V7 Prediction",
            "confidence_%": "Confidence %",
            "signal": "Signal",
            "agreement": "Models Agree",
            "actual": "Actual Candle",
            "correct": "Prediction",
            "signal_correct": "Signal Result",
            "next_open": "Next Open",
            "next_close": "Next Close"
        }
    )

    preferred_columns = [
        "Test",
        "Prediction Time",
        "V7 Prediction",
        "Confidence %",
        "Signal",
        "Models Agree",
        "Actual Candle",
        "Prediction",
        "Next Open",
        "Next Close"
    ]

    final_columns = [
        c
        for c in preferred_columns
        if c in display_results.columns
    ]

    st.dataframe(
        display_results[final_columns],
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "No individual test results were returned."
    )


# =========================================================
# TEST INTERPRETATION
# =========================================================

st.divider()

st.header(
    "🧠 HOW WE JUDGE THE TEST"
)

st.write(
    "For every historical test, V7 is placed at a point "
    "before the next 15M candle begins."
)

st.write(
    "V7 makes its prediction using information available "
    "at that time."
)

st.write(
    "Then we look at the NEXT candle:"
)

st.write(
    "🟢 Next candle CLOSE > OPEN = BULLISH"
)

st.write(
    "🔴 Next candle CLOSE < OPEN = BEARISH"
)

st.write(
    "⚪ Next candle CLOSE = OPEN = NEUTRAL"
)

st.write(
    "The prediction is correct only when V7's predicted "
    "direction matches what that next candle actually did."
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
# WHAT V7 DOES
# =========================================================

st.divider()

st.subheader(
    "🧠 What V7 Is Doing"
)

st.write(
    "1. Uses completed 15M candles as the main information."
)

st.write(
    "2. Uses completed 4H candles as higher-timeframe context."
)

st.write(
    "3. Uses three machine-learning models as an ensemble."
)

st.write(
    "4. Predicts the NEXT 15M candle."
)

st.write(
    "5. Runs a chronological 80-candle walk-forward test."
)

st.write(
    "6. Compares every prediction with the actual next candle."
)


# =========================================================
# FINAL WARNING
# =========================================================

st.warning(
    "⚠️ Do NOT use the 80-test result as proof that V7 "
    "will make money in live trading. It is a historical "
    "research test. We use it to decide whether the model "
    "deserves further development."
)


st.caption(
    "NextCandle AI V7 — research/paper-trading only. "
    "AI predictions are estimates and are not guarantees."
)
