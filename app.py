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
    page_title="NextCandle AI V4",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NextCandle AI — V4")

st.caption(
    "ACEUSDT 15-minute next-candle prediction with "
    "4-hour higher-timeframe confirmation and "
    "confidence calibration. Research/paper-trading only."
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

    history = st.slider(
        "15M historical candles",
        1000,
        10000,
        5000,
        step=500
    )

    threshold = st.slider(
        "Minimum ML probability",
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
        "Downloading ACEUSDT 15M + 4H data and "
        "training V4..."
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
                    "Not enough clean historical data."
                )

            model, feature_cols, metrics = train_model(
                df
            )

            probs, ml_signal = predict_next(
                model,
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
                f"❌ Could not build V4 model: {e}"
            )

            st.stop()


df, probs, ml_signal, metrics, bt = (
    st.session_state.result
)

latest = df.iloc[-1]


# =========================================================
# 4H BIAS
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
# MARKET CONFIRMATION
# =========================================================

confirmation_score = 0
confirmation_reasons = []


# 4H
if htf_bias == "BULLISH":

    confirmation_score += 2

    confirmation_reasons.append(
        "🟢 4H trend supports bullish direction"
    )

elif htf_bias == "BEARISH":

    confirmation_score -= 2

    confirmation_reasons.append(
        "🔴 4H trend supports bearish direction"
    )

else:

    confirmation_reasons.append(
        "⚪ 4H trend is neutral"
    )


# 15M momentum
ret5 = latest.get(
    "ret5",
    0
)

if ret5 > 0:

    confirmation_score += 1

    confirmation_reasons.append(
        "🟢 15M momentum is bullish"
    )

elif ret5 < 0:

    confirmation_score -= 1

    confirmation_reasons.append(
        "🔴 15M momentum is bearish"
    )


# EMA
ema_gap = latest.get(
    "ema9_20_gap",
    0
)

if ema_gap > 0:

    confirmation_score += 1

    confirmation_reasons.append(
        "🟢 EMA 9 is above EMA 20"
    )

elif ema_gap < 0:

    confirmation_score -= 1

    confirmation_reasons.append(
        "🔴 EMA 9 is below EMA 20"
    )


# RSI
rsi = latest.get(
    "rsi14",
    50
)

if rsi >= 55:

    confirmation_score += 1

    confirmation_reasons.append(
        f"🟢 RSI supports bullish momentum ({rsi:.1f})"
    )

elif rsi <= 45:

    confirmation_score -= 1

    confirmation_reasons.append(
        f"🔴 RSI supports bearish momentum ({rsi:.1f})"
    )

else:

    confirmation_reasons.append(
        f"⚪ RSI is neutral ({rsi:.1f})"
    )


# Volume
volume_ratio = latest.get(
    "volume_ratio20",
    1
)

if volume_ratio >= 1.20:

    if ret5 > 0:

        confirmation_score += 1

        confirmation_reasons.append(
            f"🟢 Bullish move has above-average volume "
            f"({volume_ratio:.2f}x)"
        )

    elif ret5 < 0:

        confirmation_score -= 1

        confirmation_reasons.append(
            f"🔴 Bearish move has above-average volume "
            f"({volume_ratio:.2f}x)"
        )

else:

    confirmation_reasons.append(
        f"⚪ Volume confirmation is weak "
        f"({volume_ratio:.2f}x)"
    )


# =========================================================
# FINAL SIGNAL
# =========================================================

if ml_signal == "BULLISH":

    ml_direction = 1

elif ml_signal == "BEARISH":

    ml_direction = -1

else:

    ml_direction = 0


if ml_direction == 1:

    if confirmation_score >= 2:

        final_signal = "BULLISH"
        final_icon = "🟢"
        signal_quality = "STRONG CONFIRMATION"

    elif confirmation_score >= 0:

        final_signal = "BULLISH"
        final_icon = "🟢"
        signal_quality = "MIXED CONFIRMATION"

    else:

        final_signal = "WAIT"
        final_icon = "⚪"
        signal_quality = "MODEL / MARKET DISAGREEMENT"


elif ml_direction == -1:

    if confirmation_score <= -2:

        final_signal = "BEARISH"
        final_icon = "🔴"
        signal_quality = "STRONG CONFIRMATION"

    elif confirmation_score <= 0:

        final_signal = "BEARISH"
        final_icon = "🔴"
        signal_quality = "MIXED CONFIRMATION"

    else:

        final_signal = "WAIT"
        final_icon = "⚪"
        signal_quality = "MODEL / MARKET DISAGREEMENT"


else:

    final_signal = "WAIT"
    final_icon = "⚪"
    signal_quality = "NO ML EDGE"


# =========================================================
# MAIN PREDICTION
# =========================================================

st.divider()

st.subheader(
    "🎯 NEXT 15M CANDLE"
)

if final_signal == "BULLISH":

    confidence = probs["bullish"]

    st.success(
        "🟢 NEXT 15M: BULLISH"
    )

elif final_signal == "BEARISH":

    confidence = probs["bearish"]

    st.error(
        "🔴 NEXT 15M: BEARISH"
    )

else:

    confidence = max(
        probs.values()
    )

    st.warning(
        "⚪ NEXT 15M: WAIT — INSUFFICIENT CONFIRMATION"
    )


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "ML Confidence",
    f"{confidence * 100:.1f}%"
)

c2.metric(
    "🧭 4H Bias",
    f"{htf_icon} {htf_bias}"
)

c3.metric(
    "Confirmation",
    f"{confirmation_score:+d}"
)

c4.metric(
    "Signal Quality",
    signal_quality
)


# =========================================================
# PROBABILITIES
# =========================================================

st.subheader(
    "🤖 ML Probability"
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
# V4 CONFIDENCE CALIBRATION
# =========================================================

st.divider()

st.subheader(
    "🎯 V4 Confidence Calibration"
)

st.write(
    "This compares the model's confidence with how often "
    "similar predictions were actually correct historically."
)

calibration_table = metrics.get(
    "calibration_table",
    []
)

if calibration_table:

    calibration_df = pd.DataFrame(
        calibration_table
    )

    if not calibration_df.empty:

        calibration_df[
            "average_confidence"
        ] = (
            calibration_df[
                "average_confidence"
            ] * 100
        ).round(2)

        calibration_df[
            "actual_accuracy"
        ] = (
            calibration_df[
                "actual_accuracy"
            ] * 100
        ).round(2)

        calibration_df = calibration_df.rename(
            columns={
                "bucket":
                    "Confidence range",

                "samples":
                    "Samples",

                "average_confidence":
                    "Avg confidence %",

                "actual_accuracy":
                    "Actual accuracy %"
            }
        )

        st.dataframe(
            calibration_df,
            use_container_width=True,
            hide_index=True
        )

else:

    st.info(
        "Not enough calibration data yet."
    )


calibration_error = metrics.get(
    "average_calibration_error",
    0.0
)

st.write(
    f"Average calibration error: "
    f"**{calibration_error * 100:.2f}%**"
)


# Find current confidence bucket
current_bucket = None

if confidence < 0.55:
    current_bucket = "50-55%"

elif confidence < 0.60:
    current_bucket = "55-60%"

elif confidence < 0.65:
    current_bucket = "60-65%"

elif confidence < 0.70:
    current_bucket = "65-70%"

elif confidence < 0.75:
    current_bucket = "70-75%"

elif confidence < 0.80:
    current_bucket = "75-80%"

else:
    current_bucket = "80%+"


current_bucket_accuracy = None
current_bucket_samples = 0

for row in calibration_table:

    if row["bucket"] == current_bucket:

        current_bucket_accuracy = (
            row["actual_accuracy"]
        )

        current_bucket_samples = (
            row["samples"]
        )

        break


if current_bucket_accuracy is not None:

    st.info(
        f"Current ML confidence falls in the "
        f"**{current_bucket}** historical bucket. "
        f"That bucket had **"
        f"{current_bucket_accuracy * 100:.1f}%** "
        f"actual accuracy across "
        f"**{current_bucket_samples}** predictions."
    )

else:

    st.warning(
        f"No reliable historical sample for the "
        f"current {current_bucket} confidence bucket."
    )


# =========================================================
# MARKET CONFIRMATION
# =========================================================

st.divider()

st.subheader(
    "🔎 Market Confirmation"
)

for reason in confirmation_reasons:

    st.write(reason)


# =========================================================
# EXPLANATION
# =========================================================

if ml_direction == 1 and confirmation_score < 0:

    st.warning(
        "⚠️ ML is bullish, but current market factors "
        "are mostly bearish. Model/market disagreement."
    )

elif ml_direction == -1 and confirmation_score > 0:

    st.warning(
        "⚠️ ML is bearish, but current market factors "
        "are mostly bullish. Model/market disagreement."
    )

elif final_signal in ["BULLISH", "BEARISH"]:

    st.info(
        "The ML prediction and current market confirmation "
        "are pointing in the same direction."
    )


# =========================================================
# 4H DETAILS
# =========================================================

st.divider()

st.subheader(
    "🧭 4-Hour Market Bias"
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


# =========================================================
# MODEL VALIDATION
# =========================================================

left, right = st.columns(2)


with left:

    st.subheader(
        "🧪 Model Validation"
    )

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

    st.write(
        f"Calibration samples: "
        f"**{metrics['calibration_samples']:,}**"
    )

    st.caption(
        "70% training → 10% calibration → "
        "20% final holdout. No random shuffling."
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

    st.write(
        f"Calibration error: "
        f"**{bt.get('calibration_error', 0) * 100:.2f}%**"
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
        "🟢 Bullish signals",
        bt["bullish_signals"]
    )

    st.write(
        f"Accuracy: "
        f"**{bt['bullish_accuracy'] * 100:.2f}%**"
    )


with b2:

    st.metric(
        "🔴 Bearish signals",
        bt["bearish_signals"]
    )

    st.write(
        f"Accuracy: "
        f"**{bt['bearish_accuracy'] * 100:.2f}%**"
    )


with b3:

    st.metric(
        "⚪ Neutral signals",
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
    "🕯️ Recent ACEUSDT 15M Candles"
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
    df[
        available_columns
    ].tail(20),
    use_container_width=True
)


st.caption(
    "V4 does not place orders. Probabilities are model "
    "estimates, not guarantees of future price movement."
)
