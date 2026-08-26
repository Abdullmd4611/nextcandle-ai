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

st.title("📈 NextCandle AI — V7")

st.caption(
    "AI prediction of the NEXT 15-minute candle. "
    "Direction + estimated next-candle price. "
    "Research/paper-trading only."
)


# =========================================================
# SETTINGS
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

    history = st.slider(
        "15M historical candles",
        1000,
        10000,
        5000,
        step=500
    )

    threshold = st.slider(
        "Minimum confidence",
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
# BUILD V7
# =========================================================

if run or "result" not in st.session_state:

    with st.spinner(
        "Downloading candles and training V7..."
    ):

        try:

            raw_15m = fetch_klines(
                symbol,
                "Min15",
                history
            )

            raw_4h = fetch_klines(
                symbol,
                "Hour4",
                500
            )

            df = make_features(
                raw_15m,
                raw_4h
            )

            if len(df) < 500:
                raise ValueError(
                    "Not enough clean historical candles."
                )

            models, feature_cols, metrics = train_model(
                df
            )

            # IMPORTANT:
            # Current model.py returns FIVE values.
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

            bt = walk_forward_backtest(
                df,
                feature_cols,
                signal_threshold=threshold
            )

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

        except Exception as e:

            st.error(
                f"❌ Could not build V7 model: {e}"
            )

            st.stop()


# =========================================================
# LOAD RESULT
# =========================================================

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
# CURRENT PRICE
# =========================================================

current_close = float(
    latest["close"]
)


# =========================================================
# MAIN PREDICTION
# =========================================================

st.divider()

st.subheader(
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
        f"🟢 NEXT 15M: BULLISH — "
        f"{confidence * 100:.1f}%"
    )

elif ml_signal == "BEARISH":

    st.error(
        f"🔴 NEXT 15M: BEARISH — "
        f"{confidence * 100:.1f}%"
    )

else:

    st.warning(
        f"⚪ NEXT 15M: WAIT — "
        f"highest confidence "
        f"{confidence * 100:.1f}%"
    )


# =========================================================
# TOP METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "ML Confidence",
        f"{confidence * 100:.1f}%"
    )

with c2:
    st.metric(
        "🧭 4H Context",
        f"{htf_icon} {htf_bias}"
    )

with c3:
    st.metric(
        "🤖 Ensemble",
        f"{metrics.get('ensemble_models', 3)} models"
    )

with c4:
    st.metric(
        "🎯 Threshold",
        f"{threshold * 100:.0f}%"
    )


# =========================================================
# NEXT CANDLE PRICE
# =========================================================

st.divider()

st.subheader(
    "💰 NEXT 15M PRICE ESTIMATE"
)

st.caption(
    "The regression model estimates the next candle's "
    "opening reference, closing price and percentage move."
)


q1, q2, q3 = st.columns(3)


with q1:

    st.metric(
        "Current / Expected Open",
        f"{expected_open:.8f}"
    )


with q2:

    if expected_move_pct > 0:

        delta_text = (
            f"+{expected_move_pct:.3f}%"
        )

    elif expected_move_pct < 0:

        delta_text = (
            f"{expected_move_pct:.3f}%"
        )

    else:

        delta_text = "0.000%"

    st.metric(
        "Predicted Close",
        f"{predicted_close:.8f}",
        delta=delta_text
    )


with q3:

    st.metric(
        "Current Price",
        f"{current_close:.8f}"
    )


# =========================================================
# EXPECTED MOVE
# =========================================================

if expected_move_pct > 0:

    move_icon = "🟢"
    move_direction = "UP"

elif expected_move_pct < 0:

    move_icon = "🔴"
    move_direction = "DOWN"

else:

    move_icon = "⚪"
    move_direction = "FLAT"


st.info(
    f"{move_icon} Model expects the NEXT 15M candle "
    f"to move approximately "
    f"**{expected_move_pct:.3f}% {move_direction}**."
)


price_difference = (
    predicted_close - current_close
)

st.write(
    f"Estimated price difference: "
    f"**{price_difference:+.8f}**"
)


# =========================================================
# PROBABILITY BREAKDOWN
# =========================================================

st.divider()

st.subheader(
    "🤖 V7 Probability Breakdown"
)


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
# V7 ENGINE
# =========================================================

st.divider()

st.subheader(
    "🧠 V7 Prediction Engine"
)

st.write(
    "V7 predicts the direction of the NEXT 15-minute candle."
)

st.write(
    "The ensemble uses three machine-learning "
    "classification models."
)

st.write(
    "A separate regression model estimates the "
    "next-candle percentage return."
)

st.write(
    "The 4H timeframe provides higher-timeframe context."
)


# =========================================================
# PREDICTION STATUS
# =========================================================

if ml_signal == "BULLISH":

    st.success(
        f"🎯 V7 prediction: NEXT 15M candle is "
        f"most likely **BULLISH** with "
        f"{confidence * 100:.1f}% model confidence."
    )

elif ml_signal == "BEARISH":

    st.error(
        f"🎯 V7 prediction: NEXT 15M candle is "
        f"most likely **BEARISH** with "
        f"{confidence * 100:.1f}% model confidence."
    )

else:

    st.warning(
        f"🎯 V7 prediction: **WAIT**. "
        f"The highest model probability is only "
        f"{confidence * 100:.1f}%."
    )


# =========================================================
# 4H CONTEXT
# =========================================================

st.divider()

st.subheader(
    "🧭 4H Context Used By The Model"
)


htf_gap = latest.get(
    "htf_ema_gap20",
    np.nan
)

if pd.notna(htf_gap):

    st.write(
        f"4H EMA20 distance: "
        f"**{htf_gap * 100:.2f}%**"
    )


st.write(
    f"4H bias score: **{htf_score:.0f}**"
)

st.caption(
    "4H data is used as context for the NEXT 15M prediction."
)


# =========================================================
# MODEL VALIDATION
# =========================================================

st.divider()

left, right = st.columns(2)


with left:

    st.subheader(
        "🧪 Chronological Validation"
    )

    st.write(
        f"Holdout accuracy: "
        f"**{metrics.get('holdout_accuracy', 0) * 100:.2f}%**"
    )

    st.write(
        f"Training samples: "
        f"**{metrics.get('train_samples', 0):,}**"
    )

    st.write(
        f"Holdout samples: "
        f"**{metrics.get('holdout_samples', 0):,}**"
    )

    st.write(
        f"Close-return MAE: "
        f"**{metrics.get('close_mae_pct', 0):.4f}%**"
    )

    st.caption(
        "Training data comes before validation data."
    )


with right:

    st.subheader(
        "🔄 Walk-Forward Backtest"
    )

    st.write(
        f"Predictions: "
        f"**{bt.get('predictions', 0):,}**"
    )

    st.write(
        f"Overall accuracy: "
        f"**{bt.get('accuracy', 0) * 100:.2f}%**"
    )

    st.write(
        f"High-confidence signals: "
        f"**{bt.get('signals', 0):,}**"
    )

    st.write(
        f"Signal accuracy: "
        f"**{bt.get('signal_accuracy', 0) * 100:.2f}%**"
    )


# =========================================================
# DIRECTION BACKTEST
# =========================================================

st.subheader(
    "📊 Direction Backtest"
)


b1, b2, b3 = st.columns(3)


with b1:

    st.metric(
        "🟢 Bullish",
        bt.get("bullish_signals", 0)
    )

    st.write(
        f"Accuracy: "
        f"**{bt.get('bullish_accuracy', 0) * 100:.2f}%**"
    )


with b2:

    st.metric(
        "🔴 Bearish",
        bt.get("bearish_signals", 0)
    )

    st.write(
        f"Accuracy: "
        f"**{bt.get('bearish_accuracy', 0) * 100:.2f}%**"
    )


with b3:

    st.metric(
        "⚪ Neutral",
        bt.get("neutral_signals", 0)
    )

    st.write(
        f"Accuracy: "
        f"**{bt.get('neutral_accuracy', 0) * 100:.2f}%**"
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
# DISCLAIMER
# =========================================================

st.caption(
    "V7 is a research/paper-trading model. "
    "Predictions, probabilities and price estimates "
    "are not guarantees of future market movement."
)
