import streamlit as st
import pandas as pd
import numpy as np

from datetime import datetime, timezone

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


st.title(
    "📈 NextCandle AI — V7"
)

st.caption(
    "AI prediction of the NEXT 15-minute candle. "
    "Direction + predicted OHLC + expected move. "
    "Research/paper-trading only."
)


# =========================================================
# SETTINGS
# =========================================================

with st.sidebar:

    st.header(
        "⚙️ Settings"
    )

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

if (
    run
    or "result" not in st.session_state
    or st.session_state.get(
        "last_symbol"
    ) != symbol
):

    with st.spinner(
        "Downloading completed candles and training V7..."
    ):

        try:

            # -------------------------------------------------
            # 15M
            # -------------------------------------------------

            raw_15m = fetch_klines(
                symbol,
                "Min15",
                history
            )

            # -------------------------------------------------
            # 4H
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

            if len(df) < 500:

                raise ValueError(
                    "Not enough clean historical candles."
                )

            # -------------------------------------------------
            # TRAIN
            # -------------------------------------------------

            (
                models,
                feature_cols,
                metrics
            ) = train_model(
                df
            )

            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            (
                probs,
                ml_signal,
                expected_open,
                predicted_close,
                expected_move_pct,
                predicted_high,
                predicted_low,
                agreement_count
            ) = predict_next(
                models,
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

            # -------------------------------------------------
            # SAVE
            # -------------------------------------------------

            st.session_state.result = (
                df,
                probs,
                ml_signal,
                expected_open,
                predicted_close,
                expected_move_pct,
                predicted_high,
                predicted_low,
                agreement_count,
                metrics,
                bt
            )

            st.session_state.last_symbol = symbol

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
    predicted_high,
    predicted_low,
    agreement_count,
    metrics,
    bt
) = st.session_state.result


latest = df.iloc[-1]


current_close = float(
    latest["close"]
)


# =========================================================
# 4H CONTEXT
# =========================================================

htf_score = float(
    latest.get(
        "htf_bias_score",
        0
    )
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
# NEXT CANDLE TIME
# =========================================================

last_timestamp = pd.to_datetime(
    latest["timestamp"],
    utc=True
)

next_candle_time = (
    last_timestamp
    + pd.Timedelta(minutes=15)
)

now_utc = pd.Timestamp.now(
    tz="UTC"
)

seconds_remaining = max(
    0,
    int(
        (
            next_candle_time
            - now_utc
        ).total_seconds()
    )
)

minutes_remaining = (
    seconds_remaining // 60
)

seconds_part = (
    seconds_remaining % 60
)

next_time_text = (
    next_candle_time
    .strftime(
        "%H:%M UTC"
    )
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
        f"⚪ NEXT 15M: WAIT / NO EDGE — "
        f"highest confidence "
        f"{confidence * 100:.1f}%"
    )


# =========================================================
# CANDLE TIMER
# =========================================================

t1, t2, t3 = st.columns(3)


with t1:

    st.metric(
        "🕐 Next Candle Opens",
        next_time_text
    )


with t2:

    st.metric(
        "⏳ Time Until Open",
        f"{minutes_remaining}m "
        f"{seconds_part:02d}s"
    )


with t3:

    st.metric(
        "🤖 Model Agreement",
        f"{agreement_count}/3"
    )


st.caption(
    "The countdown is based on the timestamp of the "
    "latest completed 15-minute candle."
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
# PREDICTED OHLC
# =========================================================

st.divider()

st.subheader(
    "💰 V7 PREDICTED NEXT-CANDLE OHLC"
)

st.caption(
    "These prices are produced by separate regression "
    "models trained on historical completed candles."
)


q1, q2, q3, q4 = st.columns(4)


with q1:

    st.metric(
        "Predicted Open",
        f"{expected_open:.8f}"
    )


with q2:

    st.metric(
        "Predicted High",
        f"{predicted_high:.8f}"
    )


with q3:

    st.metric(
        "Predicted Low",
        f"{predicted_low:.8f}"
    )


with q4:

    delta_text = (
        f"{expected_move_pct:+.3f}%"
    )

    st.metric(
        "Predicted Close",
        f"{predicted_close:.8f}",
        delta=delta_text
    )


# =========================================================
# CURRENT PRICE
# =========================================================

st.write(
    f"Current completed-candle close: "
    f"**{current_close:.8f}**"
)


price_difference = (
    predicted_close
    - current_close
)


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
    f"{move_icon} V7 expects the NEXT 15M "
    f"close to move approximately "
    f"**{expected_move_pct:+.3f}% {move_direction}** "
    f"from the current completed-candle close."
)


st.write(
    f"Estimated close difference: "
    f"**{price_difference:+.8f}**"
)


# =========================================================
# PROBABILITIES
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
# ENGINE
# =========================================================

st.divider()

st.subheader(
    "🧠 V7 Prediction Engine"
)

st.write(
    "V7 uses three classification models to estimate "
    "the probability that the NEXT 15-minute candle "
    "will be bullish, neutral or bearish."
)

st.write(
    "Four separate regression models estimate the "
    "NEXT candle's Open, High, Low and Close."
)

st.write(
    "Only completed 15-minute candles are used for "
    "training and the latest prediction."
)

st.write(
    "The 4H timeframe provides higher-timeframe context."
)

if ml_signal == "BULLISH":

    st.success(
        f"🎯 V7 currently favors a BULLISH next candle "
        f"with {confidence * 100:.1f}% model confidence "
        f"and {agreement_count}/3 model agreement."
    )

elif ml_signal == "BEARISH":

    st.error(
        f"🎯 V7 currently favors a BEARISH next candle "
        f"with {confidence * 100:.1f}% model confidence "
        f"and {agreement_count}/3 model agreement."
    )

else:

    st.warning(
        f"🎯 V7 says NO EDGE. "
        f"Highest probability: "
        f"{confidence * 100:.1f}%. "
        f"Model agreement: {agreement_count}/3."
    )


# =========================================================
# 4H CONTEXT
# =========================================================

st.divider()

st.subheader(
    "🧭 4H Context"
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
    f"4H bias score: "
    f"**{htf_score:.0f}**"
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
        f"Open MAE: "
        f"**{metrics.get('open_mae_pct', 0):.4f}%**"
    )

    st.write(
        f"High MAE: "
        f"**{metrics.get('high_mae_pct', 0):.4f}%**"
    )

    st.write(
        f"Low MAE: "
        f"**{metrics.get('low_mae_pct', 0):.4f}%**"
    )

    st.write(
        f"Close MAE: "
        f"**{metrics.get('close_mae_pct', 0):.4f}%**"
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


# =========================================================
# DISCLAIMER
# =========================================================

st.caption(
    "V7 is a statistical research/paper-trading system. "
    "Predicted prices, probabilities and directions are "
    "estimates, not guarantees of future market movement."
)
