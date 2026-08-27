import streamlit as st
import pandas as pd

from data import fetch_klines
from features import make_features
from model import train_model, predict_next


# =========================================================
# PAGE CONFIG
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
    "AI market-direction analysis for the next completed 15-minute candle."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ SETTINGS")

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
# START SCREEN
# =========================================================

if (
    not run_analysis
    and "analysis_result" not in st.session_state
):

    st.info(
        "👈 Select your pair and press "
        "**🚀 RUN ANALYSIS**."
    )

    st.subheader("🎯 How it works")

    st.write(
        "The app uses completed 15-minute candles."
    )

    st.write(
        "It analyzes price action, trend, momentum, "
        "volatility, volume and completed 4H context."
    )

    st.write(
        "It then gives the most likely direction "
        "for the next 15-minute candle."
    )

    st.stop()


# =========================================================
# RUN ANALYSIS
# =========================================================

if run_analysis:

    try:

        # -------------------------------------------------
        # DOWNLOAD 15M
        # -------------------------------------------------

        with st.spinner(
            "📥 Downloading completed 15M candles..."
        ):

            raw_15m = fetch_klines(
                symbol,
                "Min15",
                history
            )

        if raw_15m is None or len(raw_15m) < 500:

            raise ValueError(
                "Not enough 15M candle data returned."
            )


        # -------------------------------------------------
        # DOWNLOAD 4H
        # -------------------------------------------------

        with st.spinner(
            "📥 Downloading completed 4H candles..."
        ):

            raw_4h = fetch_klines(
                symbol,
                "Hour4",
                300
            )

        if raw_4h is None or len(raw_4h) < 50:

            raise ValueError(
                "Not enough 4H candle data returned."
            )


        # -------------------------------------------------
        # FEATURES
        # -------------------------------------------------

        with st.spinner(
            "🧮 Analyzing market structure..."
        ):

            df = make_features(
                raw_15m,
                raw_4h
            )


        if df is None or len(df) < 500:

            raise ValueError(
                "Not enough usable data after feature calculation."
            )


        # -------------------------------------------------
        # TRAIN
        # -------------------------------------------------

        with st.spinner(
            "🤖 Running AI analysis..."
        ):

            models, feature_cols, metrics = train_model(
                df
            )


        # -------------------------------------------------
        # NEXT CANDLE
        # -------------------------------------------------

        result = predict_next(
            models,
            df,
            feature_cols
        )


        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        st.session_state.analysis_result = {

            "symbol": symbol,

            "df": df,

            "result": result,

            "metrics": metrics,

            "feature_count": len(feature_cols)
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

if "analysis_result" not in st.session_state:

    st.stop()


saved = st.session_state.analysis_result

symbol = saved["symbol"]

df = saved["df"]

result = saved["result"]

metrics = saved["metrics"]


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

confidence = float(
    result["confidence"]
)

bullish_probability = float(
    result["bullish_probability"]
)

bearish_probability = float(
    result["bearish_probability"]
)


# =========================================================
# DIRECTION
# =========================================================

if direction == "BULLISH":

    st.success(
        f"🟢 NEXT 15M CANDLE: BULLISH\n\n"
        f"Model confidence: {confidence * 100:.1f}%"
    )

else:

    st.error(
        f"🔴 NEXT 15M CANDLE: BEARISH\n\n"
        f"Model confidence: {confidence * 100:.1f}%"
    )


# =========================================================
# PROBABILITIES
# =========================================================

st.subheader(
    "🤖 Direction"
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
        saved["feature_count"]
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
    c for c in display_columns
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
    "⚠️ This is a statistical market-analysis system. "
    "It does not guarantee the direction or profitability "
    "of any future candle."
)
