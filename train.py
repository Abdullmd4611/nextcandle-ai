import json
import numpy as np

from data import (
    fetch_multi_timeframe,
    validate_timeframe_data,
)

from features import (
    prepare_training_data,
)

from model import (
    NextCandleModel,
)


# ============================================================
# NextCandle AI - TRAINING ENGINE
#
# Market:
#     CYSUSDT.P
#
# MEXC API symbol:
#     CYS_USDT
#
# Prediction:
#     NEXT 15-MINUTE CANDLE
#
# Evidence:
#     1M + 5M + 15M + 4H
#
# Direction target:
#     0 = BEARISH
#     1 = NEUTRAL
#     2 = BULLISH
#
# Training is chronological.
# No random train/test split.
# ============================================================


SYMBOL = "CYS_USDT"

DISPLAY_SYMBOL = "CYSUSDT.P"

HISTORY_15M = 2500

NEUTRAL_THRESHOLD = 0.0015

TRAIN_RATIO = 0.80

MODEL_PATH = "nextcandle_model.joblib"

METRICS_PATH = "training_metrics.json"


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(f"[NextCandle AI] {message}")


# ============================================================
# DATA QUALITY
# ============================================================

def validate_training_dataset(X, y, structure_y=None):

    if X is None or X.empty:
        raise ValueError("Feature dataset is empty.")

    if y is None or len(y) == 0:
        raise ValueError("Target dataset is empty.")

    if len(X) != len(y):
        raise ValueError(
            f"Feature/target mismatch: {len(X)} != {len(y)}"
        )

    if X.index.duplicated().any():
        raise ValueError(
            "Feature dataset contains duplicate timestamps."
        )

    if not X.index.is_monotonic_increasing:
        raise ValueError(
            "Feature timestamps are not chronologically ordered."
        )

    if y.isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    classes = sorted(
        y.astype(int).unique().tolist()
    )

    if len(classes) < 2:
        raise ValueError(
            "Training data contains fewer than two target classes."
        )

    if structure_y is not None:

        if len(structure_y) != len(X):
            raise ValueError(
                "Feature/structure-target mismatch: "
                f"{len(X)} != {len(structure_y)}"
            )

        if structure_y.isna().any():
            raise ValueError(
                "Structure target contains missing values."
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
            "train_ratio must be between 0.5 and 1.0."
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

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

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

        result[names[class_id]] = {
            "count": count,
            "percentage": (
                round(
                    count / total * 100,
                    2,
                )
                if total
                else 0.0
            ),
        }

    return result


# ============================================================
# STRUCTURE CLASS DISTRIBUTION
# ============================================================

def structure_class_distribution(structure_y):

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
            "percentage": (
                round(
                    count / total * 100,
                    2,
                )
                if total
                else 0.0
            ),
        }

    return result


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    log("================================================")
    log("Starting NextCandle AI training pipeline.")
    log("================================================")

    log(f"Market: {DISPLAY_SYMBOL}")
    log(f"MEXC symbol: {SYMBOL}")
    log("Target: next 15-minute candle")
    log("Evidence: 1M + 5M + 15M + 4H")

    # --------------------------------------------------------
    # DOWNLOAD HISTORICAL DATA
    # --------------------------------------------------------

    log(
        f"Downloading approximately "
        f"{HISTORY_15M} historical 15m candles..."
    )

    market_data = fetch_multi_timeframe(
        symbol=SYMBOL,
        history_15m=HISTORY_15M,
    )

    log("Historical data downloaded.")

    # --------------------------------------------------------
    # VALIDATE RAW DATA
    # --------------------------------------------------------

    validate_timeframe_data(
        market_data
    )

    for timeframe, frame in market_data.items():

        log(
            f"{timeframe}: "
            f"{len(frame)} completed candles"
        )

    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    log(
        "Building multi-timeframe features..."
    )

    (
        X,
        y,
        structure_y,
    ) = prepare_training_data(
        df_15m=market_data["15m"],
        df_4h=market_data["4h"],
        df_1m=market_data["1m"],
        df_5m=market_data["5m"],
        neutral_threshold=NEUTRAL_THRESHOLD,
    )

    log(
        f"Feature matrix: "
        f"{X.shape[0]} rows x "
        f"{X.shape[1]} features"
    )

    # --------------------------------------------------------
    # VALIDATE DATASET
    # --------------------------------------------------------

    validate_training_dataset(
        X,
        y,
        structure_y,
    )

    log("Dataset validation passed.")

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    distribution = class_distribution(y)

    log(
        f"Direction target distribution: "
        f"{distribution}"
    )

    structure_distribution = (
        structure_class_distribution(
            structure_y
        )
    )

    log(
        f"Structure target distribution: "
        f"{structure_distribution}"
    )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

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
        "Chronological split confirmed."
    )

    # --------------------------------------------------------
    # CHECK TRAINING CLASSES
    # --------------------------------------------------------

    train_classes = sorted(
        y_train.astype(int)
        .unique()
        .tolist()
    )

    if len(train_classes) < 2:
        raise ValueError(
            "Training period contains fewer than "
            "two direction classes."
        )

    log(
        f"Training direction classes: "
        f"{train_classes}"
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = NextCandleModel(
        model_path=MODEL_PATH
    )

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

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    log(
        "Evaluating on unseen chronological "
        "validation data..."
    )

    metrics = model.evaluate(
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # PRINT METRICS
    # --------------------------------------------------------

    accuracy = metrics["accuracy"]

    balanced_accuracy = metrics[
        "balanced_accuracy"
    ]

    log(
        f"Validation accuracy: "
        f"{accuracy:.4f}"
    )

    log(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    log(
        f"Log loss: "
        f"{metrics['log_loss']:.4f}"
    )

    log("Confusion matrix:")

    print(
        np.array(
            metrics["confusion_matrix"]
        )
    )

    log("Classification report:")

    print(
        metrics["classification_report"]
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    log(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    output = {
        "project": "NextCandle AI",
        "version": "training-v3",
        "symbol": SYMBOL,
        "display_symbol": DISPLAY_SYMBOL,
        "prediction": "next_15m_candle",
        "evidence": [
            "1m",
            "5m",
            "15m",
            "4h",
        ],
        "history_15m": HISTORY_15M,
        "neutral_threshold": NEUTRAL_THRESHOLD,
        "train_ratio": TRAIN_RATIO,
        "training_samples": len(X_train),
        "validation_samples": len(X_test),
        "feature_count": X.shape[1],
        "target_distribution": distribution,
        "structure_target_distribution": structure_distribution,
        "metrics": metrics,
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
        f"Metrics saved to: "
        f"{METRICS_PATH}"
    )

    log("================================================")
    log("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    log("================================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
