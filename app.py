import streamlit as st
import pandas as pd
import numpy as np

from data import fetch_klines
from features import make_features
from model import train_model, predict_next, walk_forward_backtest


st.set_page_config(
    page_title="NextCandle AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NextCandle AI — V1")
st.caption(
    "Probability engine for the direction of the next completed candle. "
    "Research/paper-trading only."
)


with st.sidebar:
    st.header("Settings")

    symbol = st.text_input(
        "Pair",
        "ACE_USDT"
    ).upper().strip()

    timeframe = st.selectbox(
        "Timeframe",
        ["15 minutes", "4 hours"],
        index=0
    )

    interval = (
        "Min15"
        if timeframe == "15 minutes"
        else "Hour4"
    )

    history = st.slider(
        "Historical candles",
        1000,
        10000,
        5000,
        step=500
    )

    threshold = st.slider(
        "Minimum probability for a signal",
        0.50,
        0.90,
        0.65,
        0.01
    )

    run = st.button(
        "Run / Refresh",
        type="primary"
    )


if run or "result" not in st.session_state:

    with st.spinner(
        "Downloading candles and training V1..."
    ):

        try:

            raw = fetch_klines(
                symbol,
                interval,
                history
            )

            df = make_features(raw)

            model, feature_cols, metrics = train_model(
                df
            )

            probs, signal = predict_next(
                model,
                df,
                feature_cols,
                threshold
            )

            bt = walk_forward_backtest(
                df,
                feature_cols
            )

            st.session_state.result = (
                df,
                probs,
                signal,
                metrics,
                bt
            )

        except Exception as e:

            st.error(
                f"Could not build the model: {e}"
            )

            st.stop()


df, probs, signal, metrics, bt = (
    st.session_state.result
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "🟢 Bullish",
    f"{probs['bullish'] * 100:.1f}%"
)

c2.metric(
    "⚪ Neutral",
    f"{probs['neutral'] * 100:.1f}%"
)

c3.metric(
    "🔴 Bearish",
    f"{probs['bearish'] * 100:.1f}%"
)

c4.metric(
    "Model accuracy",
    f"{metrics['holdout_accuracy'] * 100:.1f}%"
)


st.divider()


if signal == "BULLISH":

    st.success(
        f"NEXT CANDLE BIAS: 🟢 BULLISH — "
        f"{probs['bullish'] * 100:.1f}%"
    )

elif signal == "BEARISH":

    st.error(
        f"NEXT CANDLE BIAS: 🔴 BEARISH — "
        f"{probs['bearish'] * 100:.1f}%"
    )

else:

    st.warning(
        "NEXT CANDLE: ⚪ NO EDGE — WAIT"
    )


left, right = st.columns(2)


with left:

    st.subheader("Model validation")

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
        "The holdout is chronological: future "
        "candles are not shuffled into training."
    )


with right:

    st.subheader("Walk-forward backtest")

    st.write(
        f"Predictions: "
        f"**{bt['predictions']:,}**"
    )

    st.write(
        f"Direction accuracy: "
        f"**{bt['accuracy'] * 100:.2f}%**"
    )

    st.write(
        f"Signals ≥ 65%: "
        f"**{bt['signals']:,}**"
    )

    if bt["signals"]:

        st.write(
            f"Signal accuracy: "
            f"**{bt['signal_accuracy'] * 100:.2f}%**"
        )

    else:

        st.write(
            "No high-confidence signals "
            "in test window."
        )


st.subheader("Recent candles")


st.dataframe(
    df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ].tail(20),
    use_container_width=True
)


st.caption(
    "V1 intentionally does not place orders. "
    "Probability is an estimate, not a guarantee."
)
