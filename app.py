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
    page_title="NextCandle AI V6",
    page_icon="📈",
    layout="wide"
)


st.title("📈 NextCandle AI — V6")

st.caption(
    "AI prediction of the NEXT 15-minute candle. "
    "Direction + estimated next-candle price range. "
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
# BUILD V6
# =========================================================

if run or "result" not in st.session_state:

    with st.spinner(
        "Downloading candles and training V6..."
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

            # -------------------------------------------------
            # COMPATIBILITY WITH CURRENT MODEL.PY
            # -------------------------------------------------

            prediction_result = predict_next(
                models,
                df,
                feature_cols,
                threshold
            )

            if isinstance(
                prediction_result,
                tuple
            ) and len(prediction_result) >= 2:

                probs = prediction_result[0]
                ml_signal = prediction_result[1]

            else:

                raise ValueError(
                    "predict_next() returned an unexpected result."
                )

            # -------------------------------------------------
            # WALK-FORWARD BACKTEST
            # -------------------------------------------------

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
                f"❌ Could not build V6 model: {e}"
            )

            st.stop()


df, probs, ml_signal, metrics, bt = (
    st.session_state.result
)

latest = df.iloc[-1]


# =========================================================
# 4H CONTEXT
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
# NEXT CANDLE PRICE ESTIMATION
# =========================================================

current_close = float(
    latest["close"]
)

atr_pct = latest.get(
    "atr14_pct",
    np.nan
)

if not pd.notna(atr_pct) or atr_pct <= 0:

    atr_pct = (
        df["ret1"]
        .rolling(20)
        .std()
        .iloc[-1]
    )

if not pd.notna(atr_pct) or atr_pct <= 0:

    atr_pct = 0.002


# Probability-weighted direction

direction_score = (
    probs.get("bullish", 0.0)
    - probs.get("bearish", 0.0)
)


expected_move_pct = (
    direction_score * atr_pct
)


estimated_close = (
    current_close
    * (1 + expected_move_pct)
)


range_pct = max(
    atr_pct,
    0.0005
)


estimated_high = (
    current_close
    * (1 + range_pct)
)


estimated_low = (
    current_close
    * (1 - range_pct)
)


# =========================================================
# MAIN NEXT 15M PREDICTION
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


c1.metric(
    "ML Confidence",
    f"{confidence * 100:.1f}%"
)


c2.metric(
    "🧭 4H Context",
    f"{htf_icon} {htf_bias}"
)


ensemble_count = metrics.get(
    "ensemble_models",
    3
)

c3.metric(
    "🤖 Ensemble",
    f"{ensemble_count} models"
)


c4.metric(
    "🎯 Threshold",
    f"{threshold * 100:.0f}%"
)


# =========================================================
# NEXT 15M PRICE ESTIMATE
# =========================================================

st.divider()

st.subheader(
    "💰 NEXT 15M PRICE ESTIMATE"
)

st.caption(
    "Model-based estimates for the next candle. "
    "They are not guaranteed future prices."
)


q1, q2, q3, q4 = st.columns(4)


with q1:

    st.metric(
        "Expected Open",
        f"{current_close:.8f}"
    )


with q2:

    st.metric(
        "Estimated Close",
        f"{estimated_close:.8f}"
    )


with q3:

    st.metric(
        "Estimated High",
        f"{estimated_high:.8f}"
    )


with q4:

    st.metric(
        "Estimated Low",
        f"{estimated_low:.8f}"
    )


# =========================================================
# EXPECTED MOVE
# =========================================================

if estimated_close > current_close:

    move_icon = "🟢"

elif estimated_close < current_close:

    move_icon = "🔴"

else:

    move_icon = "⚪"


st.info(
    f"{move_icon} Estimated NEXT 15M move: "
    f"**{expected_move_pct * 100:.3f}%**"
)


# =========================================================
# PROBABILITY BREAKDOWN
# =========================================================

st.subheader(
    "🤖 V6 Probability Breakdown"
)


p1, p2, p3 = st.columns(3)


with p1:

    st.metric(
        "🟢 Bullish",
        f"{probs.get('bullish', 0) * 100:.2f}%"
    )


with p2:

    st.metric(
        "⚪ Neutral",
        f"{probs.get('neutral', 0) * 100:.2f}%"
    )


with p3:

    st.metric(
        "🔴 Bearish",
        f"{probs.get('bearish', 0) * 100:.2f}%"
    )


# =========================================================
# V6 ENGINE
# =========================================================

st.divider()

st.subheader(
    "🧠 V6 Prediction Engine"
)

st.write(
    "V6 is focused specifically on predicting the "
    "NEXT 15-minute candle."
)

st.write(
    "The ensemble combines multiple machine-learning "
    "models and uses their probabilities to determine "
    "the strongest direction."
)

st.write(
    "The 4H timeframe is used only as higher-timeframe "
    "context for the next 15M prediction."
)

st.write(
    "V6 also provides an estimated next-candle close "
    "and volatility-based high/low range."
)


# =========================================================
# PREDICTION STATUS
# =========================================================

if ml_signal == "BULLISH":

    st.success(
        f"🎯 V6 prediction: "
        f"NEXT 15M candle is most likely "
        f"**BULLISH** with "
        f"{confidence * 100:.1f}% model confidence."
    )

elif ml_signal == "BEARISH":

    st.error(
        f"🎯 V6 prediction: "
        f"NEXT 15M candle is most likely "
        f"**BEARISH** with "
        f"{confidence * 100:.1f}% model confidence."
    )

else:

    st.warning(
        f"🎯 V6 prediction: **WAIT**. "
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
    "4H data is model context only. "
    "The app remains focused on the NEXT 15M candle."
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

    st.caption(
        "No random shuffle. Training data always comes "
        "before validation data."
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
       
