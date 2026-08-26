import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.metrics import accuracy_score, mean_absolute_error


EXCLUDE = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "target",
    "future_return",
    "future_close",
}


def feature_columns(df):

    return [
        c
        for c in df.columns
        if c not in EXCLUDE
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _classifier():

    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=42
    )


def _regressor():

    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=42
    )


def train_model(df):

    cols = feature_columns(df)

    if not cols:
        raise ValueError(
            "No usable model features were found."
        )

    data = df.copy()

    # Next-candle return target
    data["future_return"] = (
        data["close"].shift(-1)
        / data["close"]
        - 1
    )

    data["future_close"] = (
        data["close"].shift(-1)
    )

    data = data.dropna(
        subset=cols + [
            "target",
            "future_return",
            "future_close"
        ]
    ).copy()

    if len(data) < 500:
        raise ValueError(
            "Not enough clean historical candles."
        )

    X = data[cols]

    # Direction target
    y_direction = (
        data["target"]
        .map({
            -1: 0,
            0: 1,
            1: 2
        })
        .astype(int)
    )

    # Regression target = next candle return
    y_return = data["future_return"]

    split = int(len(data) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y_direction.iloc[:split]
    y_test = y_direction.iloc[split:]

    r_train = y_return.iloc[:split]
    r_test = y_return.iloc[split:]

    # =====================================================
    # THREE CLASSIFICATION MODELS
    # =====================================================

    classifiers = [
        _classifier(),
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=12,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=123
        ),
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=10,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=321
        )
    ]

    for model in classifiers:
        model.fit(
            X_train,
            y_train
        )

    test_probabilities = []

    for model in classifiers:

        test_probabilities.append(
            model.predict_proba(
                X_test
            )
        )

    avg_probabilities = np.mean(
        test_probabilities,
        axis=0
    )

    pred = np.argmax(
        avg_probabilities,
        axis=1
    )

    # =====================================================
    # NEXT-CLOSE REGRESSION MODEL
    # =====================================================

    regressor = _regressor()

    regressor.fit(
        X_train,
        r_train
    )

    predicted_return = regressor.predict(
        X_test
    )

    metrics = {
        "holdout_accuracy": float(
            accuracy_score(
                y_test,
                pred
            )
        ),
        "train_samples": int(
            len(X_train)
        ),
        "holdout_samples": int(
            len(X_test)
        ),
        "ensemble_models": 3,
        "close_mae_pct": float(
            mean_absolute_error(
                r_test,
                predicted_return
            ) * 100
        )
    }

    return (
        {
            "classifiers": classifiers,
            "regressor": regressor
        },
        cols,
        metrics
    )


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

    # =====================================================
    # ENSEMBLE DIRECTION
    # =====================================================

    probabilities = []

    for model in models["classifiers"]:

        probabilities.append(
            model.predict_proba(
                latest
            )[0]
        )

    avg_probability = np.mean(
        probabilities,
        axis=0
    )

    mapping = {
        int(k): float(v)
        for k, v in zip(
            models["classifiers"][0].classes_,
            avg_probability
        )
    }

    probs = {
        "bearish": mapping.get(0, 0.0),
        "neutral": mapping.get(1, 0.0),
        "bullish": mapping.get(2, 0.0)
    }

    best = max(
        probs,
        key=probs.get
    )

    confidence = probs[best]

    # =====================================================
    # MODEL AGREEMENT
    # =====================================================

    individual_predictions = []

    for model in models["classifiers"]:

        p = model.predict_proba(
            latest
        )[0]

        individual_predictions.append(
            int(
                np.argmax(p)
            )
        )

    agreement_count = (
        individual_predictions.count(
            int(
                {"bearish": 0,
                 "neutral": 1,
                 "bullish": 2}[best]
            )
        )
    )

    # Require confidence AND majority agreement
    if (
        confidence >= threshold
        and agreement_count >= 2
    ):

        signal = best.upper()

    else:

        signal = "NO EDGE"

    # =====================================================
    # EXPECTED NEXT CANDLE CLOSE
    # =====================================================

    predicted_return = float(
        models["regressor"].predict(
            latest
        )[0]
    )

    current_close = float(
        df["close"].iloc[-1]
    )

    expected_open = current_close

    predicted_close = (
        current_close
        * (1 + predicted_return)
    )

    expected_move_pct = (
        predicted_return * 100
    )

    return (
        probs,
        signal,
        expected_open,
        predicted_close,
        expected_move_pct
    )


def walk_forward_backtest(
    df,
    cols,
    min_train=1000,
    step=50,
    signal_threshold=0.65
):

    data = df.copy()

    data["future_return"] = (
        data["close"].shift(-1)
        / data["close"]
        - 1
    )

    data = data.dropna(
        subset=cols + [
            "target",
            "future_return"
        ]
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

        ytrain = (
            train["target"]
            .map({
                -1: 0,
                0: 1,
                1: 2
            })
            .astype(int)
        )

        if ytrain.nunique() < 2:
            continue

        models = [
            _classifier(),
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.03,
                max_leaf_nodes=12,
                min_samples_leaf=30,
                l2_regularization=2.0,
                random_state=123
            ),
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=10,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=321
            )
        ]

        probabilities = []

        for model in models:

            model.fit(
                train[cols],
                ytrain
            )

            probabilities.append(
                model.predict_proba(
                    test[cols]
                )
            )

        avg_p = np.mean(
            probabilities,
            axis=0
        )

        pred = np.argmax(
            avg_p,
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
            pred.tolist()
        )

        actuals.extend(
            actual
        )

        confidence = avg_p.max(
            axis=1
        )

        for pp, yy, cc in zip(
            pred,
            actual,
            confidence
        ):

            pp = int(pp)
            yy = int(yy)
            cc = float(cc)

            if cc >= signal_threshold:

                signal_preds.append(
                    (pp, yy)
                )

                if pp == 2:
                    bullish.append(
                        pp == yy
                    )

                elif pp == 0:
                    bearish.append(
                        pp == yy
                    )

                else:
                    neutral.append(
                        pp == yy
                    )

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

    signal_accuracy = (
        float(
            np.mean([
                p == y
                for p, y in signal_preds
            ])
        )
        if signal_preds
        else 0.0
    )

    return {
        "predictions": len(preds),
        "accuracy": accuracy,
        "signals": len(signal_preds),
        "signal_accuracy": signal_accuracy,

        "bullish_signals": len(bullish),
        "bullish_accuracy": (
            float(np.mean(bullish))
            if bullish else 0.0
        ),

        "bearish_signals": len(bearish),
        "bearish_accuracy": (
            float(np.mean(bearish))
            if bearish else 0.0
        ),

        "neutral_signals": len(neutral),
        "neutral_accuracy": (
            float(np.mean(neutral))
            if neutral else 0.0
        )
    }
