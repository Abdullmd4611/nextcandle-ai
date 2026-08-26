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
    "AI prediction focused ONLY on the NEXT 15-minute candle. "
    "Direction + estimated open + close + price range. "
    "Research/paper-trading only."
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
        1000,
        10000,
        5000,
        step=500
    )

    threshold = st.slider(
        "Minimum confidence",
        0.50,
        0.95,
        0.65,
        0.01
    )

    run = st.button(
        "🚀 Predict NEXT 15M",
        type="primary"
    )


# =========================================================
# BUILD V7
# =========================================================

if run or "result" not in st.session_state:

    with st.spinner(
        "Downloading candles and building V7 NEXT-CANDLE model..."
    ):

        try:

            # -------------------------------------------------
            # 15M candles
            # -------------------------------------------------

            raw_15m = fetch_klines(
                symbol,
                "Min15",
                history
            )

            # -------------------------------------------------
            # 4H context used internally by the model
            # -------------------------------------------------

            raw_4h = fetch_klines(
                symbol,
                "Hour4",
                500
            )

            # -------------------------------------------------
            # Features
            # -------------------------------------------------

            df = make_features(
                raw_15m,
                raw_4h
            )

            if len(df) < 500:

                raise ValueError(
                    "Not enough clean historical candles."
                )

            # -------------------------------------------------
            # Train
            # -------------------------------------------------

            models, feature_cols, metrics = train_model(
                df
            )

            # -------------------------------------------------
            # IMPORTANT:
            # Current V6 model.py returns FIVE values
            # -------------------------------------------------

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

            # -------------------------------------------------
            # Backtest
            # -------------------------------------------------

            bt = walk_forward_backtest(
                df,
                feature_cols,
                signal_threshold=threshold
            )

            # -------------------------------------------------
            # Save
            # -------------------------------------------------

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
# CURRENT PRICE
# =========================================================

current_price = float(
    latest["close"]
)


# =========================================================
# VOLATILITY FOR ESTIMATED RANGE
# =========================================================

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


atr_pct = float(
    max(
        atr_pct,
        0.0005
    )
)


# =========================================================
# ESTIMATED NEXT CANDLE RANGE
# =========================================================

estimated_high = (
    max(
        expected_open,
        predicted_close
    )
    * (1 + atr_pct)
)


estimated_low = (
    min(
        expected_open,
        predicted_close
    )
    * (1 - atr_pct)
)


# =========================================================
# BEST DIRECTION
# =========================================================

best_direction = max(
    probs,
    key=probs.get
)

confidence = float(
    probs[best_direction]
)


# =========================================================
# MAIN NEXT 15M PREDICTION
# =========================================================

st.divider()

st.subheader(
    "🎯 NEXT 15M CANDLE"
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
        "🎯 Direction Confidence",
        f"{confidence * 100:.1f}%"
    )


with c2:

    st.metric(
        "🤖 Ensemble",
        f"{metrics.get('ensemble_models', 3)} models"
    )


with c3:

    st.metric(
        "📊 Current Price",
        f"{current_price:.8f}"
    )


with c4:

    st.metric(
        "🎯 Required Confidence",
        f"{threshold * 100:.0f}%"
    )


# =========================================================
# NEXT CANDLE PRICE PREDICTION
# =========================================================

st.divider()

st.subheader(
    "💰 NEXT 15M CANDLE PRICE PREDICTION"
)

st.caption(
    "Estimated prices for the candle immediately after "
    "the current 15-minute candle."
)


q1, q2, q3, q4 = st.columns(4)


with q1:

    st.metric(
        "🔓 Expected Open",
        f"{expected_open:.8f}"
    )


with q2:

    close_delta = (
        predicted_close
        - current_price
    )

    st.metric(
        "🔒 Predicted Close",
        f"{predicted_close:.8f}",
        delta=f"{close_delta:+.8f}"
    )


with q3:

    st.metric(
        "📈 Estimated High",
        f"{estimated_high:.8f}"
    )


with q4:

    st.metric(
        "📉 Estimated Low",
        f"{estimated_low:.8f}"
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
    f"{move_icon} NEXT 15M expected move: "
    f"**{expected_move_pct:+.3f}% {move_direction}**"
)


price_difference = (
    predicted_close
    - current_price
)


st.write(
    f"Estimated close difference from current price: "
    f"**{price_difference:+.8f}**"
)


# =========================================================
# PROBABILITY BREAKDOWN
# =========================================================

st.divider()

st.subheader(
    "🤖 V7 NEXT-CANDLE PROBABILITY"
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
# V7 DECISION
# =========================================================

st.divider()

st.subheader(
    "🧠 V7 DECISION"
)


if ml_signal == "BULLISH":

    st.success(
        f"🎯 **NEXT 15M: 🟢 BULLISH — "
        f"{confidence * 100:.1f}%**"
    )

    st.write(
        "The V7 ensemble currently gives the highest "
        "probability to a bullish close for the NEXT "
        "15-minute candle."
    )

elif ml_signal == "BEARISH":

    st.error(
        f"🎯 **NEXT 15M: 🔴 BEARISH — "
        f"{confidence * 100:.1f}%**"
    )

    st.write(
        "The V7 ensemble currently gives the highest "
        "probability to a bearish close for the NEXT "
        "15-minute candle."
    )

else:

    st.warning(
        f"🎯 **NEXT 15M: ⚪ WAIT**"
    )

    st.write(
        f"The strongest probability is only "
        f"**{confidence * 100:.1f}%**, below the required "
        f"{threshold * 100:.0f}% confidence."
    )


# =========================================================
# MODEL AGREEMENT
# =========================================================

st.subheader(
    "🤖 Model Confidence"
)

st.write(
    "V7 uses an ensemble of three classification models "
    "to estimate the direction of the NEXT 15-minute candle."
)

st.write(
    "A directional signal is displayed only when the "
    "confidence reaches the selected threshold and the "
    "ensemble has sufficient agreement."
)


# =========================================================
# PRICE MODEL
# =========================================================

st.subheader(
    "💰 Next-Candle Price Engine"
)

st.write(
    "A separate regression model estimates the expected "
    "percentage return of the NEXT 15-minute candle."
)

st.write(
    f"Expected return: **{expected_move_pct:+.3f}%**"
)

st.write(
    f"Predicted close: **{predicted_close:.8f}**"
)

st.write(
    f"Estimated high: **{estimated_high:.8f}**"
)

st.write(
    f"Estimated low: **{estimated_low:.8f}**"
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

    st.write(
        f"Close-return MAE: "
        f"**{metrics.get('close_mae_pct', 0):.4f}%**"
    )

    st.caption(
        "Chronological validation. "
        "No random shuffling."
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
    "📊 NEXT-CANDLE Direction Backtest"
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


# =========================================================
# FINAL DISCLAIMER
# =========================================================

st.caption(
    "V7 is designed to predict ONLY the NEXT 15-minute "
    "candle. Direction and price outputs are statistical "
    "model estimates, not guaranteed future prices."
)
