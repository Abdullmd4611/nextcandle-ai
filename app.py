import streamlit as st
import pandas as pd

from data import fetch_klines
from features import make_features
from model import (
    train_model,
    predict_next
)


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="NextCandle AI",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📈 NextCandle AI")

st.caption(
    "15-minute market analysis using completed candles "
    "and higher-timeframe market context."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "⚙️ SETTINGS"
)

symbol = st.sidebar.text_input(
    "Trading Pair",
    value="ACE_USDT"
).upper().strip()

history = st.sidebar.slider(
    "Historical 15M Candles",
    min_value=1000,
    max_value=5000,
    value=2500,
    step=500
)

run_analysis = st.sidebar.button(
    "🚀 RUN ANALYSIS",
    type="primary",
    use_container_width=True
)


# =========================================================
# INITIAL SCREEN
# =========================================================

if (
    not run_analysis
    and "analysis_result" not in st.session_state
):

    st.info(
        "👈 Choose your trading pair and press "
        "**🚀 RUN ANALYSIS**."
    )

    st.subheader(
        "🎯 What this version does"
    )

    st.write(
        "The system analyzes completed 15-minute candles."
    )

    st.write(
        "It also uses completed 4-hour market context."
    )

    st.write(
        "It then estimates whether the NEXT 15-minute "
        "candle is more likely to be bullish or bearish."
    )

    st.warning(
        "This is an analysis tool, not a guarantee of "
        "future price movement."
    )

    st.stop()


# =========================================================
# RUN ANALYSIS
# =========================================================

if run_analysis:

    try:

        # =================================================
        # DOWNLOAD 15M
        # =================================================

        with st.spinner(
            "📥 Downloading completed 15M candles..."
        ):

            raw_15m = fetch_klines(
                symbol,
                "Min15",
                history
            )

        if raw_15m is None or len(raw_15m) == 0:

            raise ValueError(
                "No 15M market data returned."
            )

        # =================================================
        # DOWNLOAD 4H
        # =================================================

        with st.spinner(
            "📥 Downloading completed 4H candles..."
        ):

            raw_4h = fetch_klines(
                symbol,
                "Hour4",
                300
            )

        if raw_4h is None or len(raw_4h) == 0:

            raise ValueError(
                "No 4H market data returned."
            )

        # =================================================
        # FEATURES
        # =================================================

        with st.spinner(
            "🧮 Analyzing market structure..."
        ):

            df = make_features(
                raw_15m,
                raw_4h
            )

        if df is None or len(df) < 500:

            raise ValueError(
                f"Not enough usable data after feature "
                f"calculation. Only {len(df)} candles."
            )

        # =================================================
        # TRAIN
        # =================================================

        with st.spinner(
            "🤖 Running AI analysis..."
        ):

            models, feature_cols, metrics = (
                train_model(df)
            )

        # =================================================
        # NEXT CANDLE
        # =================================================

        result = predict_next(
            models,
            df,
            feature_cols
        )

        # =================================================
        # SAVE
        # =================================================

        st.session_state.analysis_result = {

            "symbol": symbol,

            "df": df,

            "result": result,

            "metrics": metrics,

            "feature_count":
                len(feature_cols)
        }

    except Exception as e:

        st.error(
            f"❌ Analysis failed: {e}"
        )

        st.exception(e)

        st.stop()


# =========================================================
# LOAD RESULT
# =========================================================

result_data = (
    st.session_state.analysis_result
)

symbol = result_data["symbol"]

df = result_data["df"]

result = result_data["result"]

metrics = result_data["metrics"]


latest = df.iloc[-1]

current_price = float(
    latest["close"]
)


# =========================================================
# MAIN RESULT
# =========================================================

st.divider()

st.header(
    "🎯 NEXT 15-MINUTE CANDLE"
)


direction = result["direction"]

confidence = (
    result["confidence"]
)

bullish_probability = (
    result["bullish_probability"]
)

bearish_probability = (
    result["bearish_probability"]
)


# =========================================================
# BIG DIRECTION
# =========================================================

if direction == "BULLISH":

    st.success(
        f"🟢 NEXT 15M CANDLE: BULLISH\n\n"
        f"Confidence: {confidence * 100:.1f}%"
    )

else:

    st.error(
        f"🔴 NEXT 15M CANDLE: BEARISH\n\n"
        f"Confidence: {confidence * 100:.1f}%"
    )


# =========================================================
# PROBABILITIES
# =========================================================

st.subheader(
    "🤖 AI Direction"
)

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "🟢 BULLISH",
        f"{bullish_probability * 100:.2f}%"
    )

with c2:

    st.metric(
        "🔴 BEARISH",
        f"{bearish_probability * 100:.2f}%"
    )


# =========================================================
# MARKET
# =========================================================

st.divider()

st.subheader(
    "💰 MARKET"
)

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Pair",
        symbol
    )

with m2:

    st.metric(
        "Current Price",
        f"{current_price:.8f}"
    )

with m3:

    st.metric(
        "Model Agreement",
        f"{result['agreement']}/3"
    )


# =========================================================
# 4H CONTEXT
# =========================================================

st.divider()

st.subheader(
    "🧭 4H MARKET CONTEXT"
)

htf_score = latest.get(
    "htf_bias_score",
    0
)

if pd.isna(htf_score):

    htf_score = 0


if htf_score > 0:

    htf_bias = "🟢 BULLISH"

elif htf_score < 0:

    htf_bias = "🔴 BEARISH"

else:

    htf_bias = "⚪ NEUTRAL"


h1, h2 = st.columns(2)

with h1:

    st.metric(
        "4H Bias",
        htf_bias
    )

with h2:

    st.metric(
        "4H Bias Score",
        f"{htf_score:.0f}"
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader(
    "🧠 MODEL INFORMATION"
)

i1, i2, i3 = st.columns(3)

with i1:

    st.metric(
        "Historical Candles",
        f"{len(df):,}"
    )

with i2:

    st.metric(
        "Model Features",
        metrics["features"]
    )

with i3:

    st.metric(
        "Holdout Accuracy",
        f"{metrics['holdout_accuracy'] * 100:.2f}%"
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

available = [
    c
    for c in display_columns
    if c in df.columns
]

st.dataframe(
    df[available].tail(20),
    use_container_width=True
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "⚠️ The result is a statistical model output, "
    "not a guarantee. Always validate the system "
    "before using real capital."
)
