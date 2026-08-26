import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# COLUMNS THAT MUST NEVER BE USED AS MODEL FEATURES
# =========================================================

EXCLUDE = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "target",
}


# =========================================================
# FEATURE SELECTION
# =========================================================

def feature_columns(df):

    return [
        c
        for c in df.columns
        if c not in EXCLUDE
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(df):

    cols = feature_columns(df)

    if not cols:
        raise ValueError(
            "No usable model features were found."
        )

    if "target" not in df.columns:
        raise ValueError(
            "Target column is missing."
        )

    data = df.dropna(
        subset=cols + ["target"]
    ).copy()

    if len(data) < 300:
        raise ValueError(
            "Not enough clean historical candles "
            "to train the model."
        )

    X = data[cols]

    y = data["target"].map({
        -1: 0,   # BEARISH
         0: 1,   # NEUTRAL
         1: 2    # BULLISH
    }).astype(int)

    # ---------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # ---------------------------------------------------------

    split = int(len(data) * 0.80)

    if split <= 0 or split >= len(data):
        raise ValueError(
            "Invalid chronological train/test split."
        )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    metrics = {
        "holdout_accuracy": float(
            accuracy
        ),
        "holdout_samples": int(
            len(y_test)
        ),
        "train_samples": int(
            len(y_train)
        )
    }

    return model, cols, metrics


# =========================================================
# NEXT 15-MINUTE PREDICTION
# =========================================================

def predict_next(
    model,
    df,
    cols,
    threshold=0.65
):

    if len(df) == 0:
        raise ValueError(
            "No candle data available for prediction."
        )

    latest = df[cols].iloc[[-1]]

    if latest.isnull().any().any():
        raise ValueError(
            "Latest candle contains missing features."
        )

    probabilities = model.predict_proba(
        latest
    )[0]

    mapping = {
        int(k): float(v)
        for k, v in zip(
            model.classes_,
            probabilities
        )
    }

    probs = {
        "bearish": mapping.get(
            0,
            0.0
        ),
        "neutral": mapping.get(
            1,
            0.0
        ),
        "bullish": mapping.get(
            2,
            0.0
        ),
    }

    best = max(
        probs,
        key=probs.get
    )

    confidence = float(
        probs[best]
    )

    if confidence >= threshold:

        signal = best.upper()

    else:

        signal = "NO EDGE"

    return probs, signal


# =========================================================
# WALK-FORWARD BACKTEST
# =========================================================

def walk_forward_backtest(
    df,
    cols,
    min_train=1000,
    step=50,
    signal_threshold=0.65
):

    data = df.dropna(
        subset=cols + ["target"]
    ).copy()

    if len(data) <= min_train + step:

        return {
            "predictions": 0,
            "accuracy": 0.0,
            "signals": 0,
            "signal_accuracy": 0.0
        }

    preds = []
    actuals = []
    signal_preds = []

    # ---------------------------------------------------------
    # WALK FORWARD
    #
    # At every step:
    #   train ONLY on candles before the test window.
    #
    # No future candles are placed into training.
    # ---------------------------------------------------------

    for i in range(
        min_train,
        len(data) - 1,
        step
    ):

        train = data.iloc[:i]

        test = data.iloc[
            i:min(
                i + step,
                len(data) - 1
            )
        ]

        if len(test) == 0:
            continue

        ytrain = train["target"].map({
            -1: 0,
             0: 1,
             1: 2
        }).astype(int)

        # -----------------------------------------------------
        # Skip windows where training contains only one class.
        # -----------------------------------------------------

        if ytrain.nunique() < 2:
            continue

        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=42
        )

        model.fit(
            train[cols],
            ytrain
        )

        p = model.predict_proba(
            test[cols]
        )

        pred = model.predict(
            test[cols]
        )

        preds.extend(
            pred.tolist()
        )

        actual = (
            test["target"]
            .map({
                -1: 0,
                 0: 1,
                 1: 2
            })
            .astype(int)
            .tolist()
        )

        actuals.extend(
            actual
        )

        confidence = p.max(
            axis=1
        )

        for pp, yy, cc in zip(
            pred,
            actual,
            confidence
        ):

            if cc >= signal_threshold:

                signal_preds.append(
                    (
                        int(pp),
                        int(yy)
                    )
                )

    # =========================================================
    # OVERALL ACCURACY
    # =========================================================

    accuracy = (
        float(
            np.mean(
                np.array(preds)
                == np.array(actuals)
            )
        )
        if preds
        else 0.0
    )

    # =========================================================
    # HIGH-CONFIDENCE SIGNAL ACCURACY
    # =========================================================

    signal_accuracy = (
        float(
            np.mean([
                p == y
                for p, y
                in signal_preds
            ])
        )
        if signal_preds
        else 0.0
    )

    return {
        "predictions": int(
            len(preds)
        ),
        "accuracy": accuracy,
        "signals": int(
            len(signal_preds)
        ),
        "signal_accuracy": signal_accuracy
    }
