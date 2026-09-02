import json
import numpy as np

from data import (
    fetch_multi_timeframe,
    validate_timeframe_data,
)

from building import (
    prepare_open_training_data,
    describe_open_training_data,
)

from model import (
    NextCandleModel,
)


# ============================================================
# NextCandle AI - CURRENT BUILDING CANDLE TRAINING ENGINE
#
# Market:
#     CYSUSDT.P
#
# API symbol:
#     CYS_USDT
#
# PREDICTION:
#     CURRENT 15-MINUTE CANDLE
#
# Example:
#     10:00 candle starts
#     AI predicts the 10:00-10:15 candle
#     10:15 candle closes
#     prediction is compared with actual result
#
# Evidence:
#     1M + 5M + 15M + 4H
#
# Direction target:
#     0 = BEARISH
#     1 = NEUTRAL
#     2 = BULLISH
#
# Training:
#     Chronological 80/20 split
#
# IMPORTANT:
#     The target candle's final close/high/low are NEVER
#     used as prediction features.
# ============================================================


SYMBOL = "CYS_USDT"

DISPLAY_SYMBOL = "CYSUSDT.P"

HISTORY_15M = 2500

NEUTRAL_THRESHOLD = 0.0015

TRAIN_RATIO = 0.80

# ------------------------------------------------------------
# IMPORTANT:
# Use a different filename from the old next-candle model.
# This prevents us from accidentally overwriting your old model.
# ------------------------------------------------------------

MODEL_PATH = "current_candle_model.joblib"

METRICS_PATH = "current_candle_training_metrics.json"


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(f"[NextCandle AI] {message}")


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_training_dataset(
    X,
    y,
    structure_y=None,
):

    if X is None or X.empty:

        raise ValueError(
            "Feature dataset is empty."
        )

    if y is None or len(y) == 0:

        raise ValueError(
            "Direction target dataset is empty."
        )

    if len(X) != len(y):

        raise ValueError(
            "Feature/target mismatch: "
            f"{len(X)} != {len(y)}"
        )

    # --------------------------------------------------------
    # Timestamp checks
    # --------------------------------------------------------

    if X.index.duplicated().any():

        raise ValueError(
            "Feature dataset contains "
            "duplicate timestamps."
        )

    if not X.index.is_monotonic_increasing:

        raise ValueError(
            "Feature timestamps are not "
            "chronologically ordered."
        )

    # --------------------------------------------------------
    # Target checks
    # --------------------------------------------------------

    if y.isna().any():

        raise ValueError(
            "Direction target contains "
            "missing values."
        )

    y_classes = sorted(
        y.astype(int)
        .unique()
        .tolist()
    )

    invalid_classes = [
        value
        for value in y_classes
        if value not in [0, 1, 2]
    ]

    if invalid_classes:

        raise ValueError(
            "Invalid direction target classes: "
            f"{invalid_classes}"
        )

    if len(y_classes) < 2:

        raise ValueError(
            "Training data contains fewer "
            "than two direction classes."
        )

    # --------------------------------------------------------
    # Structure target checks
    # --------------------------------------------------------

    if structure_y is not None:

        if len(structure_y) != len(X):

            raise ValueError(
                "Feature/structure-target mismatch: "
                f"{len(X)} != {len(structure_y)}"
            )

        if structure_y.isna().any():

            raise ValueError(
                "Structure target contains "
                "missing values."
            )

    return True


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    X,
    y,
    train_ratio=TRAIN_RATIO,
):

    if not 0.5 <= train_ratio < 1.0:

        raise ValueError(
            "train_ratio must be between "
            "0.5 and 1.0."
        )

    split_index = int(
        len(X) * train_ratio
    )

    if split_index <= 0:

        raise ValueError(
            "Training split is empty."
        )

    if split_index >= len(X):

        raise ValueError(
            "Validation split is empty."
        )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# DIRECTION CLASS DISTRIBUTION
# ============================================================

def class_distribution(y):

    counts = (
        y.astype(int)
        .value_counts()
        .sort_index()
    )

    total = len(y)

    names = {
        0: "BEARISH",
        1: "NEUTRAL",
        2: "BULLISH",
    }

    result = {}

    for class_id in [0, 1, 2]:

        count = int(
            counts.get(
                class_id,
                0,
            )
        )

        result[
            names[class_id]
        ] = {

            "count": count,

            "percentage":
                round(
                    count / total * 100,
                    2,
                )
                if total
                else 0.0,
        }

    return result


# ============================================================
# STRUCTURE CLASS DISTRIBUTION
# ============================================================

def structure_class_distribution(
    structure_y,
):

    counts = (
        structure_y.astype(int)
        .value_counts()
        .sort_index()
    )

    total = len(structure_y)

    names = {

        0: "OTHER",

        1: "HAMMER_LIKE",

        2: "SHOOTING_STAR_LIKE",

        3: "DOJI_LIKE",

        4: "STRONG_BULLISH",

        5: "STRONG_BEARISH",

        6: "HANGING_MAN_LIKE",
    }

    result = {}

    for class_id, name in names.items():

        count = int(
            counts.get(
                class_id,
                0,
            )
        )

        result[name] = {

            "count": count,

            "percentage":
                round(
                    count / total * 100,
                    2,
                )
                if total
                else 0.0,
        }

    return result


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    log(
        "================================================"
    )

    log(
        "STARTING CURRENT-BUILDING CANDLE TRAINING"
    )

    log(
        "================================================"
    )

    log(
        f"Market: {DISPLAY_SYMBOL}"
    )

    log(
        f"API symbol: {SYMBOL}"
    )

    log(
        "Prediction: CURRENT 15-minute candle"
    )

    log(
        "Evidence: 1M + 5M + 15M + 4H"
    )

    log(
        f"Neutral threshold: "
        f"{NEUTRAL_THRESHOLD * 100:.2f}%"
    )

    # ========================================================
    # DOWNLOAD HISTORICAL DATA
    # ========================================================

    log(
        f"Downloading approximately "
        f"{HISTORY_15M} historical 15M candles..."
    )

    market_data = fetch_multi_timeframe(
        symbol=SYMBOL,
        history_15m=HISTORY_15M,
    )

    log(
        "Historical market data downloaded."
    )

    # ========================================================
    # VALIDATE RAW DATA
    # ========================================================

    log(
        "Validating downloaded market data..."
    )

    validate_timeframe_data(
        market_data
    )

    for timeframe, frame in market_data.items():

        log(
            f"{timeframe}: "
            f"{len(frame)} completed candles"
        )

    # ========================================================
    # BUILD CURRENT-CANDLE TRAINING DATA
    # ========================================================

    log(
        "Building CURRENT-CANDLE features..."
    )

    log(
        "The model will learn:"
    )

    log(
        "START OF CANDLE -> FINAL RESULT OF SAME CANDLE"
    )

    (
        X,
        y,
        structure_y,
    ) = prepare_open_training_data(

        df_15m=
            market_data["15m"],

        df_5m=
            market_data["5m"],

        df_1m=
            market_data["1m"],

        df_4h=
            market_data["4h"],

        neutral_threshold=
            NEUTRAL_THRESHOLD,
    )

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    info = describe_open_training_data(
        X,
        y,
        structure_y,
    )

    log(
        "Current-candle training dataset created."
    )

    log(
        f"Training rows available: "
        f"{info['rows']}"
    )

    log(
        f"Feature count: "
        f"{info['features']}"
    )

    log(
        f"Dataset start: "
        f"{info['start']}"
    )

    log(
        f"Dataset end: "
        f"{info['end']}"
    )

    # ========================================================
    # VALIDATE TRAINING DATASET
    # ========================================================

    validate_training_dataset(
        X,
        y,
        structure_y,
    )

    log(
        "Training dataset validation PASSED."
    )

    # ========================================================
    # DIRECTION DISTRIBUTION
    # ========================================================

    distribution = class_distribution(
        y
    )

    log(
        "CURRENT 15M CANDLE direction distribution:"
    )

    for name, values in distribution.items():

        log(
            f"  {name}: "
            f"{values['count']} "
            f"({values['percentage']}%)"
        )

    # ========================================================
    # STRUCTURE DISTRIBUTION
    # ========================================================

    structure_distribution = (
        structure_class_distribution(
            structure_y
        )
    )

    log(
        "CURRENT 15M CANDLE structure distribution:"
    )

    for name, values in (
        structure_distribution.items()
    ):

        log(
            f"  {name}: "
            f"{values['count']} "
            f"({values['percentage']}%)"
        )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = chronological_split(

        X,
        y,
        TRAIN_RATIO,
    )

    log(
        f"Training samples: "
        f"{len(X_train)}"
    )

    log(
        f"Validation samples: "
        f"{len(X_test)}"
    )

    log(
        "Chronological 80/20 split confirmed."
    )

    # ========================================================
    # CHECK CLASSES
    # ========================================================

    train_classes = sorted(
        y_train.astype(int)
        .unique()
        .tolist()
    )

    test_classes = sorted(
        y_test.astype(int)
        .unique()
        .tolist()
    )

    log(
        f"Training direction classes: "
        f"{train_classes}"
    )

    log(
        f"Validation direction classes: "
        f"{test_classes}"
    )

    if len(train_classes) < 2:

        raise ValueError(
            "Training period contains fewer "
            "than two direction classes."
        )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    log(
        "Creating current-candle prediction model..."
    )

    model = NextCandleModel(
        model_path=MODEL_PATH
    )

    # ========================================================
    # TRAIN
    # ========================================================

    log(
        "Training gradient boosting model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    log(
        "Model training completed."
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    log(
        "Evaluating on unseen chronological data..."
    )

    metrics = model.evaluate(
        X_test,
        y_test,
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = float(
        metrics["accuracy"]
    )

    balanced_accuracy = float(
        metrics["balanced_accuracy"]
    )

    log(
        "================================================"
    )

    log(
        "VALIDATION RESULTS"
    )

    log(
        "================================================"
    )

    log(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    log(
        f"Balanced accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    log(
        f"Log loss: "
        f"{metrics['log_loss']:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    log(
        "Confusion matrix:"
    )

    print(
        np.array(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    log(
        "Classification report:"
    )

    print(
        metrics[
            "classification_report"
        ]
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save(
        MODEL_PATH
    )

    log(
        f"Current-candle model saved to:"
    )

    log(
        MODEL_PATH
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    output = {

        "project":
            "NextCandle AI",

        "version":
            "current-building-v1",

        "symbol":
            SYMBOL,

        "display_symbol":
            DISPLAY_SYMBOL,

        "prediction":
            "CURRENT_15M_CANDLE",

        "prediction_definition":
            "Information available at the start of "
            "a 15-minute candle is used to predict "
            "the final result of that same candle.",

        "primary_timeframe":
            "15m",

        "lower_timeframes":
            [
                "1m",
                "5m",
            ],

        "higher_timeframe":
            "4h",

        "neutral_threshold":
            NEUTRAL_THRESHOLD,

        "history_15m":
            HISTORY_15M,

        "train_ratio":
            TRAIN_RATIO,

        "training_samples":
            len(X_train),

        "validation_samples":
            len(X_test),

        "feature_count":
            X.shape[1],

        "direction_target":
            {
                "0": "BEARISH",
                "1": "NEUTRAL",
                "2": "BULLISH",
            },

        "structure_target":
            {
                "0": "OTHER",
                "1": "HAMMER_LIKE",
                "2": "SHOOTING_STAR_LIKE",
                "3": "DOJI_LIKE",
                "4": "STRONG_BULLISH",
                "5": "STRONG_BEARISH",
                "6": "HANGING_MAN_LIKE",
            },

        "direction_distribution":
            distribution,

        "structure_distribution":
            structure_distribution,

        "metrics":
            metrics,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            default=str,
        )

    log(
        f"Training metrics saved to:"
    )

    log(
        METRICS_PATH
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    log(
        "================================================"
    )

    log(
        "CURRENT-BUILDING CANDLE TRAINING COMPLETED"
    )

    log(
        "================================================"
    )

    log(
        f"Model: {MODEL_PATH}"
    )

    log(
        f"Metrics: {METRICS_PATH}"
    )

    log(
        f"Validation accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    log(
        f"Balanced accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    log(
        "================================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
