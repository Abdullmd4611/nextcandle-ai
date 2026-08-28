import streamlit as st
import pandas as pd
import numpy as np

from data import fetch_klines, get_latest_price
from features import prepare_training_data
from model import NextCandleModel


# ============================================================
# NEXTCANDLE AI V2
# Bybit CYSUSDT Perpetual
# Primary prediction: next 15-minute candle
# Higher timeframe: completed 4-hour context
# ============================================================


st.set_page_config(
    page_title="NextCandle AI V2",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_SYMBOL = "CYSUSDT"

PRIMARY_INTERVAL = "Min15"
HIGHER_INTERVAL = "Hour4"


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# ============================================================
# HEADER
# ============================================================

st.title("📈 NextCandle AI V2")

st.caption(
    "Bybit CYSUSDT perpetual • "
    "Next 15-minute candle analysis"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ MARKET SETTINGS")

symbol = st.sidebar.text_input(
    "Bybit Perpetual Symbol",
    value=DEFAULT_SYMBOL,
).upper().strip()


history = st.sidebar.slider(
    "15M Historical Candles",
    min_value=1000,
    max_value=5000,
    value=2500,
    step=500,
)


neutral_threshold_pct = st.sidebar.slider(
    "Neutral Threshold (%)",
    min_value=0.05,
    max_value=0.50,
    value=0.15,
    step=0.05,
)


confidence_threshold = st.sidebar.slider(
    "Minimum Confidence (%)",
    min_value=50,
    max_value=90,
    value=55,
    step=1,
)


edge_threshold = st.sidebar.slider(
    "Minimum Probability Edge (%)",
    min_value=5,
    max_value=40,
    value=10,
    step=1,
)


run_analysis = st.sidebar.button(
    "🚀 RUN ANALYSIS",
    type="primary",
    use_container_width=True,
)


# ============================================================
# INTRO
# ============================================================

if (
    st.session_state.analysis_result is None
    and not run_analysis
):

    st.info(
        "Select the market settings and press "
        "**🚀 RUN ANALYSIS**."
    )

    st.subheader("🧠 What this system does")

    st.write(
        "• Uses Bybit perpetual-market data."
    )

    st.write(
        "• Uses completed 15-minute candles."
    )

    st.write(
        "• Uses completed 4-hour candles as "
        "higher-timeframe context."
    )

    st.write(
        "• Learns three outcomes: "
        "BEARISH, NEUTRAL and BULLISH."
    )

    st.write(
        "• Refuses weak predictions instead of "
        "forcing a BUY/SELL direction."
    )

    st.warning(
        "This is a statistical prediction system. "
        "No model can guarantee the next candle."
    )

    st.stop()


# ============================================================
# ANALYSIS
# ============================================================

if run_analysis:

    try:

        # ----------------------------------------------------
        # VALIDATE SYMBOL
        # ----------------------------------------------------

        if not symbol:

            raise ValueError(
                "Please enter a Bybit symbol."
            )


        # ----------------------------------------------------
        # DOWNLOAD 15M DATA
        # ----------------------------------------------------

        with st.spinner(
            "📥 Downloading completed Bybit 15M candles..."
        ):

            raw_15m = fetch_klines(
                symbol=symbol,
                interval=PRIMARY_INTERVAL,
                total=history,
            )


        if (
            raw_15m is None
            or len(raw_15m) < 500
        ):

            raise ValueError(
                "Not enough completed 15M candles."
            )


        # ----------------------------------------------------
        # DOWNLOAD 4H DATA
        # ----------------------------------------------------

        with st.spinner(
            "📥 Downloading completed Bybit 4H candles..."
        ):

            raw_4h = fetch_klines(
                symbol=symbol,
                interval=HIGHER_INTERVAL,
                total=300,
            )


        if (
            raw_4h is None
            or len(raw_4h) < 50
        ):

            raise ValueError(
                "Not enough completed 4H candles."
            )


        # ----------------------------------------------------
        # PREPARE FEATURES + TARGET
        # ----------------------------------------------------

        with st.spinner(
            "🧮 Building market features..."
        ):

            X, y = prepare_training_data(
                df_15m=raw_15m,
                df_4h=raw_4h,
                neutral_threshold=(
                    neutral_threshold_pct / 100
                ),
            )


        if len(X) < 500:

            raise ValueError(
                "Not enough usable training samples "
                "after feature engineering."
            )


        # ----------------------------------------------------
        # CHECK CLASS BALANCE
        # ----------------------------------------------------

        class_counts = (
            y.value_counts()
            .reindex(
                [0, 1, 2],
                fill_value=0,
            )
        )


        if (class_counts == 0).any():

            raise ValueError(
                "Training data does not contain all "
                "three classes. Increase history or "
                "adjust the neutral threshold."
            )


        # ----------------------------------------------------
        # TIME-ORDERED TRAIN / VALIDATION SPLIT
        # ----------------------------------------------------

        split_index = int(
            len(X) * 0.80
        )


        X_train = X.iloc[
            :split_index
        ]

        y_train = y.iloc[
            :split_index
        ]

        X_test = X.iloc[
            split_index:
        ]

        y_test = y.iloc[
            split_index:
        ]


        if len(X_test) < 100:

            raise ValueError(
                "Holdout dataset is too small."
            )


        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Training NextCandle AI V2..."
        ):

            model = NextCandleModel()

            model.fit(
                X_train,
                y_train,
            )


        # ----------------------------------------------------
        # HOLDOUT EVALUATION
        # ----------------------------------------------------

        with st.spinner(
            "🧪 Testing model on unseen candles..."
        ):

            metrics = model.evaluate(
                X_test,
                y_test,
            )


        # ----------------------------------------------------
        # LATEST FEATURES
        # ----------------------------------------------------

        latest_features = X.iloc[
            [-1]
        ]


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        result_list = model.signal(
            latest_features,
            minimum_confidence=(
                confidence_threshold / 100
            ),
            minimum_edge=(
                edge_threshold / 100
            ),
        )


        result = result_list[0]


        # ----------------------------------------------------
        # CURRENT PRICE
        # ----------------------------------------------------

        with st.spinner(
            "💰 Getting latest Bybit price..."
        ):

            current_price = get_latest_price(
                symbol
            )


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        st.session_state.analysis_result = {

            "symbol": symbol,

            "raw_15m": raw_15m,

            "raw_4h": raw_4h,

            "X": X,

            "y": y,

            "model": model,

            "result": result,

            "metrics": metrics,

            "class_counts": class_counts,

            "current_price": current_price,

        }


    except Exception as e:

        st.error(
            f"❌ Analysis failed: {e}"
        )

        st.exception(e)

        st.stop()


# ============================================================
# LOAD RESULT
# ============================================================

saved = st.session_state.analysis_result

if saved is None:
    st.stop()


symbol = saved["symbol"]

raw_15m = saved["raw_15m"]

raw_4h = saved["raw_4h"]

X = saved["X"]

y = saved["y"]

model = saved["model"]

result = saved["result"]

metrics = saved["metrics"]

class_counts = saved["class_counts"]

current_price = saved["current_price"]


# ============================================================
# MAIN SIGNAL
# ============================================================

st.divider()

st.header(
    "🎯 NEXT 15-MINUTE CANDLE"
)


signal = result["signal"]

prediction = result["prediction"]

confidence = result["confidence"]

edge = result["edge"]


if signal == "BULLISH":

    st.success(
        f"🟢 BULLISH\n\n"
        f"Confidence: {confidence * 100:.2f}%\n\n"
        f"Probability edge: {edge * 100:.2f}%"
    )


elif signal == "BEARISH":

    st.error(
        f"🔴 BEARISH\n\n"
        f"Confidence: {confidence * 100:.2f}%\n\n"
        f"Probability edge: {edge * 100:.2f}%"
    )


else:

    st.warning(
        f"⚪ WAIT / NO STRONG EDGE\n\n"
        f"Model's strongest class: {prediction}\n\n"
        f"Confidence: {confidence * 100:.2f}%\n\n"
        f"Probability edge: {edge * 100:.2f}%"
    )


# ============================================================
# PROBABILITIES
# ============================================================

st.subheader(
    "🤖 Model Probability"
)


p1, p2, p3 = st.columns(3)


with p1:

    st.metric(
        "🔴 BEARISH",
        f"{result['bearish_probability'] * 100:.2f}%",
    )


with p2:

    st.metric(
        "⚪ NEUTRAL",
        f"{result['neutral_probability'] * 100:.2f}%",
    )


with p3:

    st.metric(
        "🟢 BULLISH",
        f"{result['bullish_probability'] * 100:.2f}%",
    )


# ============================================================
# MARKET
# ============================================================

st.divider()

st.subheader(
    "💰 BYBIT MARKET"
)


m1, m2, m3 = st.columns(3)


with m1:

    st.metric(
        "Symbol",
        symbol,
    )


with m2:

    st.metric(
        "Latest Price",
        f"{current_price:.8f}",
    )


with m3:

    last_candle = raw_15m.iloc[-1]

    candle_change = (
        float(last_candle["close"])
        / float(last_candle["open"])
        - 1
    )

    st.metric(
        "Last 15M Change",
        f"{candle_change * 100:.3f}%",
    )


# ============================================================
# 4H CONTEXT
# ============================================================

st.divider()

st.subheader(
    "🧭 4-HOUR MARKET CONTEXT"
)


htf_close = raw_4h["close"]

htf_ema20 = (
    htf_close
    .ewm(
        span=20,
        adjust=False,
    )
    .mean()
)

htf_ema50 = (
    htf_close
    .ewm(
        span=50,
        adjust=False,
    )
    .mean()
)


latest_4h_close = float(
    htf_close.iloc[-1]
)

latest_ema20 = float(
    htf_ema20.iloc[-1]
)

latest_ema50 = float(
    htf_ema50.iloc[-1]
)


if (
    latest_4h_close > latest_ema20
    and latest_ema20 > latest_ema50
):

    htf_bias = "🟢 BULLISH"


elif (
    latest_4h_close < latest_ema20
    and latest_ema20 < latest_ema50
):

    htf_bias = "🔴 BEARISH"


else:

    htf_bias = "⚪ MIXED / NEUTRAL"


h1, h2, h3 = st.columns(3)


with h1:

    st.metric(
        "4H Bias",
        htf_bias,
    )


with h2:

    st.metric(
        "4H Close",
        f"{latest_4h_close:.8f}",
    )


with h3:

    st.metric(
        "4H Candles",
        f"{len(raw_4h):,}",
    )


# ============================================================
# MODEL QUALITY
# ============================================================

st.divider()

st.subheader(
    "🧪 MODEL QUALITY — UNSEEN DATA"
)


q1, q2, q3 = st.columns(3)


with q1:

    st.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%",
    )


with q2:

    st.metric(
        "Balanced Accuracy",
        f"{metrics['balanced_accuracy'] * 100:.2f}%",
    )


with q3:

    st.metric(
        "Log Loss",
        f"{metrics['log_loss']:.4f}",
    )


st.caption(
    "The holdout set is the latest 20% of the "
    "historical samples and is not used to train "
    "the model."
)


# ============================================================
# TRAINING CLASS DISTRIBUTION
# ============================================================

st.subheader(
    "📊 Training Target Distribution"
)


d1, d2, d3 = st.columns(3)


with d1:

    st.metric(
        "BEARISH Samples",
        f"{int(class_counts[0]):,}",
    )


with d2:

    st.metric(
        "NEUTRAL Samples",
        f"{int(class_counts[1]):,}",
    )


with d3:

    st.metric(
        "BULLISH Samples",
        f"{int(class_counts[2]):,}",
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

with st.expander(
    "📋 Detailed classification report"
):

    st.text(
        metrics["classification_report"]
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

with st.expander(
    "🔢 Confusion matrix"
):

    matrix = np.array(
        metrics["confusion_matrix"]
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual BEARISH",
            "Actual NEUTRAL",
            "Actual BULLISH",
        ],
        columns=[
            "Predicted BEARISH",
            "Predicted NEUTRAL",
            "Predicted BULLISH",
        ],
    )

    st.dataframe(
        matrix_df,
        use_container_width=True,
    )


# ============================================================
# RECENT 15M CANDLES
# ============================================================

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
    "volume",
]


available_columns = [
    column
    for column in display_columns
    if column in raw_15m.columns
]


st.dataframe(
    raw_15m[
        available_columns
    ].tail(20),
    use_container_width=True,
)


# ============================================================
# RECENT 4H CANDLES
# ============================================================

st.subheader(
    "🕯️ Recent 4H Candles"
)


st.dataframe(
    raw_4h[
        available_columns
    ].tail(10),
    use_container_width=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.warning(
    "⚠️ IMPORTANT: This system produces statistical "
    "probabilities, not guaranteed predictions. "
    "Crypto perpetual markets can move rapidly, "
    "and leverage can amplify losses."
)

st.caption(
    "NextCandle AI V2 • Bybit data • "
    "15M prediction • 4H context"
)
