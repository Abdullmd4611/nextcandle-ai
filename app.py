import streamlit as st
import pandas as pd
import numpy as np

from data import fetch_klines, get_latest_price
from features import build_features, prepare_training_data
from model import NextCandleModel


# ============================================================
# NEXTCANDLE AI V2
# ============================================================
# Market:
#     Bybit USDT Perpetual
#
# Primary:
#     CYSUSDT
#
# Prediction:
#     NEXT 15-MINUTE CANDLE
#
# Higher timeframe:
#     COMPLETED 4-HOUR CONTEXT
#
# IMPORTANT:
#     Training data and live prediction data are kept separate.
# ============================================================


st.set_page_config(
    page_title="NextCandle AI V2",
    page_icon="📈",
    layout="wide",
)


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
    "Bybit CYSUSDT Perpetual • "
    "Next 15-minute candle prediction"
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
    "Historical 15M Candles",
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
# START SCREEN
# ============================================================

if (
    st.session_state.analysis_result is None
    and not run_analysis
):

    st.info(
        "Select your settings and press "
        "**🚀 RUN ANALYSIS**."
    )

    st.subheader("🎯 System")

    st.write(
        "The AI learns from historical completed "
        "15-minute candles."
    )

    st.write(
        "It uses completed 4-hour candles for "
        "higher-timeframe context."
    )

    st.write(
        "It predicts the direction of the "
        "NEXT 15-minute candle."
    )

    st.write(
        "Weak predictions can produce WAIT instead "
        "of forcing a trade direction."
    )

    st.warning(
        "No AI model can guarantee the next candle."
    )

    st.stop()


# ============================================================
# RUN ANALYSIS
# ============================================================

if run_analysis:

    try:

        # ====================================================
        # VALIDATION
        # ====================================================

        if not symbol:

            raise ValueError(
                "Please enter a Bybit symbol."
            )


        # ====================================================
        # DOWNLOAD COMPLETED 15M DATA
        # ====================================================

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


        # ====================================================
        # DOWNLOAD COMPLETED 4H DATA
        # ====================================================

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


        # ====================================================
        # BUILD TRAINING DATA
        # ====================================================

        with st.spinner(
            "🧮 Building historical training data..."
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
                "Not enough usable training samples."
            )


        # ====================================================
        # CLASS CHECK
        # ====================================================

        class_counts = (
            y.value_counts()
            .reindex(
                [0, 1, 2],
                fill_value=0,
            )
        )


        if (class_counts == 0).any():

            raise ValueError(
                "One or more target classes are missing. "
                "Increase history or adjust the neutral "
                "threshold."
            )


        # ====================================================
        # TIME-ORDERED TRAIN / HOLDOUT SPLIT
        # ====================================================

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


        # ====================================================
        # TRAIN MODEL
        # ====================================================

        with st.spinner(
            "🤖 Training NextCandle AI..."
        ):

            model = NextCandleModel()

            model.fit(
                X_train,
                y_train,
            )


        # ====================================================
        # UNSEEN DATA EVALUATION
        # ====================================================

        with st.spinner(
            "🧪 Testing on unseen historical candles..."
        ):

            metrics = model.evaluate(
                X_test,
                y_test,
            )


        # ====================================================
        # LIVE FEATURE CALCULATION
        # ====================================================
        #
        # IMPORTANT:
        #
        # X is the training dataset.
        #
        # We DO NOT use X.iloc[-1] for the live prediction.
        #
        # Instead we calculate features from the complete
        # raw 15M dataset and take the latest completed candle.
        #
        # This candle does not have a target yet because the
        # NEXT candle has not happened.
        #
        # That is exactly what we want to predict.
        # ====================================================

        with st.spinner(
            "🔎 Preparing the latest completed candle..."
        ):

            live_features = build_features(
                df_15m=raw_15m,
                df_4h=raw_4h,
            )


        if live_features.empty:

            raise ValueError(
                "Unable to calculate live features."
            )


        # ----------------------------------------------------
        # Remove rows with unusable features.
        #
        # We only need the latest valid completed candle.
        # ----------------------------------------------------

        valid_live = live_features.dropna(
            how="all"
        )


        if valid_live.empty:

            raise ValueError(
                "No valid live feature row available."
            )


        latest_live_features = (
            valid_live.iloc[[-1]]
        )


        # ====================================================
        # LIVE PREDICTION
        # ====================================================

        with st.spinner(
            "🎯 Predicting the NEXT 15M candle..."
        ):

            result = model.signal(
                latest_live_features,
                minimum_confidence=(
                    confidence_threshold / 100
                ),
                minimum_edge=(
                    edge_threshold / 100
                ),
            )[0]


        # ====================================================
        # LATEST BYBIT PRICE
        # ====================================================

        with st.spinner(
            "💰 Getting latest Bybit price..."
        ):

            current_price = get_latest_price(
                symbol
            )


        # ====================================================
        # STORE
        # ====================================================

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

            "latest_prediction_candle":
                raw_15m.iloc[-1]["timestamp"],

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

result = saved["result"]

metrics = saved["metrics"]

class_counts = saved["class_counts"]

current_price = saved["current_price"]

prediction_candle = (
    saved["latest_prediction_candle"]
)


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
        f"Confidence: "
        f"{confidence * 100:.2f}%\n\n"
        f"Probability edge: "
        f"{edge * 100:.2f}%"
    )


elif signal == "BEARISH":

    st.error(
        f"🔴 BEARISH\n\n"
        f"Confidence: "
        f"{confidence * 100:.2f}%\n\n"
        f"Probability edge: "
        f"{edge * 100:.2f}%"
    )


else:

    st.warning(
        f"⚪ WAIT\n\n"
        f"Strongest model class: "
        f"{prediction}\n\n"
        f"Confidence: "
        f"{confidence * 100:.2f}%\n\n"
        f"Probability edge: "
        f"{edge * 100:.2f}%"
    )


# ============================================================
# PROBABILITIES
# ============================================================

st.subheader(
    "🤖 NEXT-CANDLE PROBABILITY"
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
# PREDICTION CANDLE
# ============================================================

st.subheader(
    "🕯️ PREDICTION BASIS"
)

st.write(
    "The prediction is based on the latest "
    "completed 15-minute candle:"
)

st.info(
    str(prediction_candle)
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

    latest_candle = raw_15m.iloc[-1]

    candle_change = (
        float(latest_candle["close"])
        / float(latest_candle["open"])
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

    htf_bias = "⚪ MIXED"


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


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

st.subheader(
    "📊 Historical Target Distribution"
)


d1, d2, d3 = st.columns(3)


with d1:

    st.metric(
        "🔴 BEARISH",
        f"{int(class_counts[0]):,}",
    )


with d2:

    st.metric(
        "⚪ NEUTRAL",
        f"{int(class_counts[1]):,}",
    )


with d3:

    st.metric(
        "🟢 BULLISH",
        f"{int(class_counts[2]):,}",
    )


# ============================================================
# DETAILED REPORT
# ============================================================

with st.expander(
    "📋 Classification report"
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
# RECENT 15M
# ============================================================

st.divider()

st.subheader(
    f"🕯️ Recent {symbol} 15M Candles"
)


columns_15m = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


available_15m = [
    column
    for column in columns_15m
    if column in raw_15m.columns
]


st.dataframe(
    raw_15m[
        available_15m
    ].tail(20),
    use_container_width=True,
)


# ============================================================
# RECENT 4H
# ============================================================

st.subheader(
    "🕯️ Recent 4H Candles"
)


available_4h = [
    column
    for column in columns_15m
    if column in raw_4h.columns
]


st.dataframe(
    raw_4h[
        available_4h
    ].tail(10),
    use_container_width=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.warning(
    "⚠️ This system provides statistical probabilities, "
    "not guaranteed future outcomes. Perpetual futures "
    "and leverage involve substantial risk."
)

st.caption(
    "NextCandle AI V2 • Bybit • CYSUSDT • "
    "15M prediction • 4H context"
)
