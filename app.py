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
    "Analyze the NEXT 15-minute candle using completed "
    "historical market information."
)


# =========================================================
# SETTINGS
# =========================================================

st.sidebar.header("⚙️ SETTINGS")

symbol = st.sidebar.text_input(
    "Trading pair",
    value="ACE_USDT"
).upper().strip()

history = st.sidebar.slider(
    "Historical 15M candles",
    min_value=1000,
    max_value=10000,
    value=3000,
    step=500
)

threshold = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.50,
    max_value=0.90,
    value=0.60,
    step=0.01
)


# =========================================================
# TEST SETTINGS
# =========================================================

TEST_COUNT = 80
MIN_TRAIN = 1000

st.sidebar.info(
    "🎯 The system will test exactly 80 historical "
    "15-minute candles."
)


# =========================================================
# RUN BUTTON
# =========================================================

run_test = st.sidebar.button(
    "🚀 RUN 80-CANDLE TEST",
    key="run_v7_test",
    type="primary",
    use_container_width=True
)


# =========================================================
# INITIAL SCREEN
# =========================================================

if (
    not run_test
    and "result" not in st.session_state
):

    st.info(
        "👈 Select your settings and press "
        "**🚀 RUN 80-CANDLE TEST**."
    )

    st.subheader(
        "🎯 What this test does"
    )

    st.write(
        "The system examines historical ACEUSDT "
        "15-minute candles."
    )

    st.write(
        "It makes a direction decision for the "
        "next 15-minute candle."
    )

    st.write(
        "The prediction is then compared with what "
        "the next candle actually did."
    )

    st.write(
        "The final 80 candles are kept outside the "
        "training data so they can be used as an "
        "unseen historical test."
    )

    st.warning(
        "⚠️ Historical results do not guarantee "
        "future trading performance."
    )

    st.stop()


# =========================================================
# RUN TEST
# =========================================================

if run_test:

    try:

        # Clear previous result

        st.session_state.pop(
            "result",
            None
        )

        # =================================================
        # STEP 1 — MARKET DATA
        # =================================================

        with st.status(
            "📥 Downloading market data...",
            expanded=True
        ) as status:

            st.write(
                f"Downloading {symbol} 15M candles..."
            )

            raw_15m = fetch_klines(
                symbol,
                "Min15",
                history
            )

            if (
                raw_15m is None
                or len(raw_15m) == 0
            ):

                raise ValueError(
                    "MEXC returned no 15M candles."
                )

            st.write(
                f"✅ Received "
                f"{len(raw_15m):,} 15M candles."
            )

            # ---------------------------------------------
            # 4H DATA
            # ---------------------------------------------

            st.write(
                "Downloading completed 4H candles..."
            )

            raw_4h = fetch_klines(
                symbol,
                "Hour4",
                500
            )

            if (
                raw_4h is None
                or len(raw_4h) == 0
            ):

                raise ValueError(
                    "MEXC returned no 4H candles."
                )

            st.write(
                f"✅ Received "
                f"{len(raw_4h):,} 4H candles."
            )

            status.update(
                label="✅ Market data downloaded",
                state="complete"
            )


        # =================================================
        # STEP 2 — FEATURES
        # =================================================

        with st.status(
            "🧮 Building features...",
            expanded=True
        ) as status:

            df = make_features(
                raw_15m,
                raw_4h
            )

            if (
                df is None
                or len(df) < 700
            ):

                raise ValueError(
                    "Not enough clean candles. "
                    f"Only "
                    f"{0 if df is None else len(df)} "
                    f"available."
                )

            st.write(
                f"✅ Feature dataset contains "
                f"{len(df):,} candles."
            )

            status.update(
                label="✅ Features ready",
                state="complete"
            )


        # =================================================
        # STEP 3 — TRAIN LIVE MODEL
        # =================================================

        with st.status(
            "🤖 Training AI model...",
            expanded=True
        ) as status:

            models, feature_cols, metrics = train_model(
                df
            )

            st.write(
                f"✅ Using "
                f"{len(feature_cols)} "
                f"model features."
            )

            st.write(
                "✅ Three classification models trained."
            )

            st.write(
                "✅ Regression model trained."
            )

            status.update(
                label="✅ AI model trained",
                state="complete"
            )


        # =================================================
        # STEP 4 — CURRENT NEXT CANDLE
        # =================================================

        with st.status(
            "🎯 Analyzing the next 15M candle...",
            expanded=True
        ) as status:

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

            status.update(
                label="✅ Current analysis ready",
                state="complete"
            )


        # =================================================
        # STEP 5 — 80 CANDLE TEST
        # =================================================

        with st.status(
            "🧪 Running 80-candle historical test...",
            expanded=True
        ) as status:

            st.write(
                "Training the test model once, then "
                "checking the final 80 unseen candles."
            )

            progress_text = st.empty()

            progress_bar = st.progress(
                0
            )

            # The backtest itself is optimized and
            # processes the test window in one operation.

            bt = walk_forward_backtest(
                df,
                feature_cols,
                n_tests=TEST_COUNT,
                min_train=MIN_TRAIN,
                signal_threshold=threshold
            )

            completed = int(
                bt.get(
                    "predictions",
                    0
                )
            )

            progress_bar.progress(
                100
            )

            progress_text.write(
                f"✅ Completed "
                f"{completed}/{TEST_COUNT} "
                f"historical tests."
            )

            status.update(
                label="✅ 80-candle test complete",
                state="complete"
            )


        # =================================================
        # SAVE RESULT
        # =================================================

        st.session_state.result = {

            "df":
                df,

            "probs":
                probs,

            "signal":
                ml_signal,

            "expected_open":
                expected_open,

            "predicted_close":
                predicted_close,

            "expected_move_pct":
                expected_move_pct,

            "metrics":
                metrics,

            "bt":
                bt,

            "symbol":
                symbol,

            "threshold":
                threshold
        }

        st.success(
            "🎉 80-candle test completed successfully!"
        )

    except Exception as e:

        st.error(
            f"❌ Test failed: {e}"
        )

        st.exception(e)

        st.stop()


# =========================================================
# LOAD RESULT
# =========================================================

if "result" not in st.session_state:

    st.stop()


result = st.session_state.result

df = result["df"]

probs = result["probs"]

ml_signal = result["signal"]

expected_open = result["expected_open"]

predicted_close = result["predicted_close"]

expected_move_pct = result["expected_move_pct"]

metrics = result["metrics"]

bt = result["bt"]

symbol = result["symbol"]

threshold = result["threshold"]


# =========================================================
# LATEST CANDLE
# =========================================================

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
# NEXT 15M CANDLE
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


if ml_signal == "BULLISH":

    st.success(
        f"🟢 NEXT 15M CANDLE: BULLISH\n\n"
        f"Confidence: "
        f"{confidence * 100:.1f}%"
    )

elif ml_signal == "BEARISH":

    st.error(
        f"🔴 NEXT 15M CANDLE: BEARISH\n\n"
        f"Confidence: "
        f"{confidence * 100:.1f}%"
    )

else:

    st.warning(
        f"⚪ NEXT 15M CANDLE: NO EDGE\n\n"
        f"Highest probability: "
        f"{confidence * 100:.1f}%"
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
# VALIDATION
# =========================================================

st.divider()

st.header(
    "🧪 MODEL VALIDATION"
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
        "Signal Accuracy",
        f"{bt.get('signal_accuracy', 0) * 100:.2f}%"
    )


with v4:

    st.metric(
        "Tests Completed",
        str(
            bt.get(
                "predictions",
                0
            )
        )
    )


# =========================================================
# 80 TEST RESULT
# =========================================================

st.divider()

st.header(
    "🎯 THE 80-CANDLE TEST RESULT"
)

predictions_done = int(
    bt.get(
        "predictions",
        0
    )
)

accuracy = float(
    bt.get(
        "accuracy",
        0
    )
)

correct = int(
    bt.get(
        "correct",
        round(
            predictions_done
            * accuracy
        )
    )
)

wrong = int(
    bt.get(
        "wrong",
        predictions_done - correct
    )
)


r1, r2, r3 = st.columns(3)


with r1:

    st.metric(
        "Tests",
        predictions_done
    )


with r2:

    st.metric(
        "Correct",
        correct
    )


with r3:

    st.metric(
        "Wrong",
        wrong
    )


st.metric(
    "🎯 Overall Accuracy",
    f"{accuracy * 100:.2f}%"
)


# =========================================================
# VERDICT
# =========================================================

if predictions_done >= TEST_COUNT:

    if accuracy >= 0.70:

        st.success(
            "🟢 The model reached the 70% "
            "historical accuracy target."
        )

    elif accuracy >= 0.55:

        st.warning(
            "🟡 The model is above the "
            "50% baseline but below the "
            "70% target."
        )

    else:

        st.error(
            "🔴 The model did not reach "
            "the required target."
        )

else:

    st.warning(
        f"Only {predictions_done} tests completed. "
        f"We need {TEST_COUNT} before judging the result."
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
        int(
            bt.get(
                "bullish_signals",
                0
            )
        )
    )

    st.write(
        "Accuracy: "
        f"{float(bt.get('bullish_accuracy', 0)) * 100:.2f}%"
    )


with b2:

    st.metric(
        "🔴 Bearish Tests",
        int(
            bt.get(
                "bearish_signals",
                0
            )
        )
    )

    st.write(
        "Accuracy: "
        f"{float(bt.get('bearish_accuracy', 0)) * 100:.2f}%"
    )


with b3:

    st.metric(
        "⚪ Neutral Tests",
        int(
            bt.get(
                "neutral_signals",
                0
            )
        )
    )

    st.write(
        "Accuracy: "
        f"{float(bt.get('neutral_accuracy', 0)) * 100:.2f}%"
    )


# =========================================================
# HIGH CONFIDENCE
# =========================================================

st.divider()

st.subheader(
    "🔥 HIGH-CONFIDENCE TEST"
)

st.write(
    f"Signals: **"
    f"{int(bt.get('signals', 0))}"
    f"**"
)

st.write(
    "Accuracy: **"
    f"{float(bt.get('signal_accuracy', 0)) * 100:.2f}%"
    "**"
)

st.caption(
    f"Minimum confidence: "
    f"{threshold * 100:.0f}%"
)


# =========================================================
# INDIVIDUAL TESTS
# =========================================================

st.divider()

st.header(
    "🧾 HISTORICAL TEST RESULTS"
)

test_results = bt.get(
    "test_results",
    []
)

if test_results:

    test_df = pd.DataFrame(
        test_results
    )

    if "correct" in test_df.columns:

        test_df["correct"] = (
            test_df["correct"]
            .map(
                {
                    True: "✅ CORRECT",
                    False: "❌ WRONG"
                }
            )
        )

    if "signal_correct" in test_df.columns:

        test_df["signal_correct"] = (
            test_df["signal_correct"]
            .map(
                {
                    True: "✅",
                    False: "❌"
                }
            )
        )

    st.dataframe(
        test_df,
        use_container_width=True,
        height=600
    )

else:

    st.warning(
        "No individual test results available."
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
    column
    for column in display_columns
    if column in df.columns
]

st.dataframe(
    df[
        available_columns
    ].tail(20),
    use_container_width=True
)


# =========================================================
# FINAL
# =========================================================

st.divider()

st.info(
    "🧠 Do not move to another version yet. "
    "First examine the 80 historical results and "
    "determine whether the system genuinely has an edge."
)

st.caption(
    "Research/paper-trading only. "
    "Historical accuracy does not guarantee future performance."
)
