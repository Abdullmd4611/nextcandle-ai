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


st.set_page_config(
    page_title="NextCandle AI V5",
    page_icon="📈",
    layout="wide"
)


st.title("📈 NextCandle AI — V5")

st.caption(
    "AI ensemble prediction of the NEXT 15-minute candle. "
    "Bullish / Neutral / Bearish. "
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
# BUILD V5
# =========================================================

if run or "result" not in st.session_state:

    with st.spinner(
        "Downloading candles and training V5 ensemble..."
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

            probs, ml_signal = predict_next(
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
                metrics,
                bt
            )

        except Exception as e:

            st.error(
                f"❌ Could not build V5 model: {e}"
            )

            st.stop()


df, probs, ml_signal, metrics, bt = (
    st.session_state.result
)


latest = df.iloc[-1]


# =========================================================
# 4H BIAS — CONTEXT ONLY
# =========================================================

htf_score = latest.get(
    "htf_bias_score",
    0
)

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
# MAIN NEXT-CANDLE PREDICTION
# =========================================================

st.divider()

st.subheader(
    "🎯 NEXT 15M CANDLE"
)


best_direction = max(
    probs,
    key=probs.get
)

confidence = probs[
    best_direction
]


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


c1.metric(
    "ML Confidence",
    f"{confidence * 100:.1f}%"
)


c2.metric(
    "🧭 4H Context",
    f"{htf_icon} {htf_bias}"
)


c3.metric(
    "🤖 Ensemble",
    f"{metrics['ensemble_models']} models"
)


c4.metric(
    "🎯 Threshold",
    f"{threshold * 100:.0f}%"
)


# =========================================================
# PROBABILITY BREAKDOWN
# =========================================================

st.subheader(
    "🤖 V5 Probability Breakdown"
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
# SIMPLE PREDICTION EXPLANATION
# =========================================================

st.divider()

st.subheader(
    "🧠 V5 Prediction Engine"
)

st.write(
    "V5 combines three different machine-learning models "
    "and averages their probabilities."
)

st.write(
    "The final prediction is only accepted when the "
    "highest probability reaches your selected threshold "
    "and at least two of the three models agree."
)

if ml_signal == "BULLISH":

    st.success(
        "The V5 ensemble currently favors the NEXT 15M "
        "candle closing bullish."
    )

elif ml_signal == "BEARISH":

    st.error(
        "The V5 ensemble currently favors the NEXT 15M "
        "candle closing bearish."
    )

else:

    st.warning(
        "V5 does not have enough confidence/agreement "
        "to issue a directional prediction."
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
    "The 4H information is used as model context for "
    "the next 15M prediction. It is not a separate "
    "trade-analysis system."
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
        f"**{metrics['holdout_accuracy'] * 100:.2f}%**"
    )

    st.write(
        f"Training samples: "
        f"**{metrics['train_samples']:,}**"
    )

    st.write(
        f"Holdout samples: "
        f"**{metrics['holdout_samples']:,}**"
    )

    st.caption(
        "No random shuffle. Training data always comes "
        "before the validation data."
    )


with right:

    st.subheader(
        "🔄 Walk-Forward Backtest"
    )

    st.write(
        f"Predictions: "
        f"**{bt['predictions']:,}**"
    )

    st.write(
        f"Overall accuracy: "
        f"**{bt['accuracy'] * 100:.2f}%**"
    )

    st.write(
        f"High-confidence signals: "
        f"**{bt['signals']:,}**"
    )

    st.write(
        f"Signal accuracy: "
        f"**{bt['signal_accuracy'] * 100:.2f}%**"
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
        bt["bullish_signals"]
    )

    st.write(
        f"Accuracy: "
        f"**{bt['bullish_accuracy'] * 100:.2f}%**"
    )


with b2:

    st.metric(
        "🔴 Bearish",
        bt["bearish_signals"]
    )

    st.write(
        f"Accuracy: "
        f"**{bt['bearish_accuracy'] * 100:.2f}%**"
    )


with b3:

    st.metric(
        "⚪ Neutral",
        bt["neutral_signals"]
    )

    st.write(
        f"Accuracy: "
        f"**{bt['neutral_accuracy'] * 100:.2f}%**"
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


st.caption(
    "V5 predicts only the direction of the NEXT 15-minute "
    "candle. Probabilities are estimates, not guarantees."
)
