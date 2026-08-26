import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.metrics import accuracy_score


# =========================================================
# FEATURES TO EXCLUDE
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


def feature_columns(df):
    return [
        c
        for c in df.columns
        if c not in EXCLUDE
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# =========================================================
# MODEL FACTORY
# =========================================================

def _make_models():

    models = [

        HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=1.5,
            random_state=42
        ),

        RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),

        ExtraTreesClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    ]

    return models


# =========================================================
# PROBABILITY ALIGNMENT
# =========================================================

def _aligned_probability(model, X):

    probabilities = model.predict_proba(X)

    result = np.zeros(
        (len(X), 3),
        dtype=float
    )

    for j, cls in enumerate(model.classes_):

        result[:, int(cls)] = probabilities[:, j]

    return result


# =========================================================
# ENSEMBLE PREDICTION
# =========================================================

def _ensemble_probability(models, X):

    all_probs = []

    for model in models:

        all_probs.append(
            _aligned_probability(
                model,
                X
            )
        )

    return np.mean(
        all_probs,
        axis=0
    )


# =========================================================
# TRAIN V5
# =========================================================

def train_model(df):

    cols = feature_columns(df)

    if not cols:
        raise ValueError(
            "No usable model features were found."
        )

    data = df.dropna(
        subset=cols + ["target"]
    ).copy()

    if len(data) < 500:
        raise ValueError(
            "Not enough clean historical candles."
        )

    X = data[cols]

    y = data["target"].map({
        -1: 0,
        0: 1,
        1: 2
    }).astype(int)

    split = int(
        len(data) * 0.80
    )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    models = _make_models()

    trained_models = []

    for model in models:

        model.fit(
            X_train,
            y_train
        )

        trained_models.append(model)

    probabilities = _ensemble_probability(
        trained_models,
        X_test
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    metrics = {

        "holdout_accuracy": float(
            accuracy_score(
                y_test,
                predictions
            )
        ),

        "holdout_samples": int(
            len(y_test)
        ),

        "train_samples": int(
            len(y_train)
        ),

        "ensemble_models": int(
            len(trained_models)
        )
    }

    return (
        trained_models,
        cols,
        metrics
    )


# =========================================================
# NEXT 15-MINUTE PREDICTION
# =========================================================

def predict_next(
    models,
    df,
    cols,
    threshold=0.65
):

    latest = df[cols].iloc[[-1]]

    if latest.isnull().any().any():

        raise ValueError(
            "Latest candle contains missing features."
        )

    probabilities = _ensemble_probability(
        models,
        latest
    )[0]

    probs = {

        "bearish": float(
            probabilities[0]
        ),

        "neutral": float(
            probabilities[1]
        ),

        "bullish": float(
            probabilities[2]
        )
    }

    best_index = int(
        np.argmax(probabilities)
    )

    names = [
        "BEARISH",
        "NEUTRAL",
        "BULLISH"
    ]

    best = names[best_index]

    confidence = float(
        probabilities[best_index]
    )

    # -----------------------------------------------------
    # MODEL AGREEMENT
    # -----------------------------------------------------

    individual_predictions = []

    for model in models:

        p = _aligned_probability(
            model,
            latest
        )[0]

        individual_predictions.append(
            int(np.argmax(p))
        )

    agreement = (
        sum(
            p == best_index
            for p in individual_predictions
        )
        / len(individual_predictions)
    )

    # -----------------------------------------------------
    # FINAL SIGNAL
    # -----------------------------------------------------

    # Require both sufficient probability
    # and model agreement.

    if (
        confidence >= threshold
        and agreement >= 2 / 3
        and best != "NEUTRAL"
    ):

        signal = best

    else:

        signal = "NO EDGE"

    return (
        probs,
        signal
    )


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
            "signal_accuracy": 0.0,

            "bullish_signals": 0,
            "bullish_accuracy": 0.0,

            "bearish_signals": 0,
            "bearish_accuracy": 0.0,

            "neutral_signals": 0,
            "neutral_accuracy": 0.0
        }

    preds = []
    actuals = []

    signal_preds = []

    bullish = []
    bearish = []
    neutral = []

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

        if ytrain.nunique() < 2:
            continue

        models = _make_models()

        trained_models = []

        for model in models:

            model.fit(
                train[cols],
                ytrain
            )

            trained_models.append(model)

        probabilities = _ensemble_probability(
            trained_models,
            test[cols]
        )

        predictions = np.argmax(
            probabilities,
            axis=1
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

        preds.extend(
            predictions.tolist()
        )

        actuals.extend(
            actual
        )

        confidence = probabilities.max(
            axis=1
        )

        # -------------------------------------------------
        # SIGNALS
        # -------------------------------------------------

        for j in range(
            len(predictions)
        ):

            prediction = int(
                predictions[j]
            )

            real = int(
                actual[j]
            )

            conf = float(
                confidence[j]
            )

            if conf >= signal_threshold:

                signal_preds.append(
                    (prediction, real)
                )

                if prediction == 2:

                    bullish.append(
                        prediction == real
                    )

                elif prediction == 0:

                    bearish.append(
                        prediction == real
                    )

                else:

                    neutral.append(
                        prediction == real
                    )

    # =====================================================
    # ACCURACY
    # =====================================================

    if preds:

        accuracy = float(
            np.mean(
                np.array(preds)
                == np.array(actuals)
            )
        )

    else:

        accuracy = 0.0

    # =====================================================
    # SIGNAL ACCURACY
    # =====================================================

    if signal_preds:

        signal_accuracy = float(
            np.mean([
                p == y
                for p, y
                in signal_preds
            ])
        )

    else:

        signal_accuracy = 0.0

    # =====================================================
    # DIRECTION ACCURACY
    # =====================================================

    bullish_accuracy = (
        float(np.mean(bullish))
        if bullish
        else 0.0
    )

    bearish_accuracy = (
        float(np.mean(bearish))
        if bearish
        else 0.0
    )

    neutral_accuracy = (
        float(np.mean(neutral))
        if neutral
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

        "signal_accuracy": signal_accuracy,

        "bullish_signals": int(
            len(bullish)
        ),

        "bullish_accuracy":
            bullish_accuracy,

        "bearish_signals": int(
            len(bearish)
        ),

        "bearish_accuracy":
            bearish_accuracy,

        "neutral_signals": int(
            len(neutral)
        ),

        "neutral_accuracy":
            neutral_accuracy
    }
