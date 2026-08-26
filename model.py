import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


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
        c for c in df.columns
        if c not in EXCLUDE
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def train_model(df):

    cols = feature_columns(df)

    X = df[cols]
    y = df["target"].map({
        -1: 0,
        0: 1,
        1: 2
    }).astype(int)

    split = int(len(df) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=12,
        l2_regularization=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    metrics = {
        "holdout_accuracy": accuracy_score(
            y_test,
            pred
        ),
        "holdout_samples": len(y_test),
        "train_samples": len(y_train)
    }

    return model, cols, metrics


def predict_next(
    model,
    df,
    cols,
    threshold=0.65
):

    latest = df[cols].iloc[[-1]]

    p = model.predict_proba(latest)[0]

    mapping = {
        int(k): float(v)
        for k, v in zip(
            model.classes_,
            p
        )
    }

    probs = {
        "bearish": mapping.get(0, 0.0),
        "neutral": mapping.get(1, 0.0),
        "bullish": mapping.get(2, 0.0),
    }

    best = max(
        probs,
        key=probs.get
    )

    signal = (
        best.upper()
        if probs[best] >= threshold
        else "NO EDGE"
    )

    return probs, signal


def walk_forward_backtest(
    df,
    cols,
    min_train=1000,
    step=50
):

    if len(df) <= min_train + step:

        return {
            "predictions": 0,
            "accuracy": 0.0,
            "signals": 0,
            "signal_accuracy": 0.0
        }

    preds = []
    actuals = []
    signal_preds = []

    for i in range(
        min_train,
        len(df) - 1,
        step
    ):

        train = df.iloc[:i]

        test = df.iloc[
            i:min(i + step, len(df) - 1)
        ]

        ytrain = train["target"].map({
            -1: 0,
            0: 1,
            1: 2
        }).astype(int)

        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.06,
            max_leaf_nodes=12,
            l2_regularization=1.0,
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

        actuals.extend(
            test["target"]
            .map({
                -1: 0,
                0: 1,
                1: 2
            })
            .astype(int)
            .tolist()
        )

        confidence = p.max(axis=1)

        for pp, yy, cc in zip(
            pred,
            test["target"]
            .map({
                -1: 0,
                0: 1,
                1: 2
            }),
            confidence
        ):

            if cc >= 0.65:

                signal_preds.append(
                    (pp, int(yy))
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
                for p, y
                in signal_preds
            ])
        )
        if signal_preds
        else 0.0
    )

    return {
        "predictions": len(preds),
        "accuracy": accuracy,
        "signals": len(signal_preds),
        "signal_accuracy": signal_accuracy
    }
