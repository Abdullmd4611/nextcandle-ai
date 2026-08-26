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
        c
        for c in df.columns
        if c not in EXCLUDE
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _make_model():
    return HistGradientBoostingClassifier(
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

    data = df.dropna(
        subset=cols + ["target"]
    ).copy()

    if len(data) < 300:
        raise ValueError(
            "Not enough clean historical candles."
        )

    X = data[cols]

    y = data["target"].map({
        -1: 0,
        0: 1,
        1: 2
    }).astype(int)

    split = int(len(data) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = _make_model()

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    metrics = {
        "holdout_accuracy": float(
            accuracy_score(y_test, pred)
        ),
        "holdout_samples": int(
            len(y_test)
        ),
        "train_samples": int(
            len(y_train)
        )
    }

    return model, cols, metrics


def predict_next(
    model,
    df,
    cols,
    threshold=0.65
):

    latest = df[cols].iloc[[-1]]

    if latest.isnull().any().any():
        raise ValueError(
            "Latest candle contains missing features."
        )

    p = model.predict_proba(
        latest
    )[0]

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

    confidence = probs[best]

    signal = (
        best.upper()
        if confidence >= threshold
        else "NO EDGE"
    )

    return probs, signal


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

        model = _make_model()

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

        confidence = p.max(
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

                elif pp == 1:
                    neutral.append(
                        pp == yy
                    )

    if preds:

        accuracy = float(
            np.mean(
                np.array(preds)
                == np.array(actuals)
            )
        )

    else:

        accuracy = 0.0

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
        "bullish_accuracy": bullish_accuracy,

        "bearish_signals": int(
            len(bearish)
        ),
        "bearish_accuracy": bearish_accuracy,

        "neutral_signals": int(
            len(neutral)
        ),
        "neutral_accuracy": neutral_accuracy
    }
