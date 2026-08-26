import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error
)


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

    "future_open_return",
    "future_high_return",
    "future_low_return",
    "future_close_return",
}


# =========================================================
# FEATURE COLUMNS
# =========================================================

def feature_columns(df):

    return [
        c
        for c in df.columns
        if c not in EXCLUDE
        and pd.api.types.is_numeric_dtype(
            df[c]
        )
    ]


# =========================================================
# MODELS
# =========================================================

def _classifier(
    seed=42,
    max_iter=250,
    learning_rate=0.04,
    max_leaf_nodes=15,
    min_samples_leaf=25,
    l2=1.5
):

    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2,
        random_state=seed
    )


def _regressor(seed=42):

    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=2.0,
        random_state=seed
    )


def _make_classifiers():

    return [
        _classifier(
            seed=42,
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2=1.5
        ),

        _classifier(
            seed=123,
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=12,
            min_samples_leaf=30,
            l2=2.0
        ),

        _classifier(
            seed=321,
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=10,
            min_samples_leaf=20,
            l2=1.0
        )
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

    required = [
        "target",
        "future_open_return",
        "future_high_return",
        "future_low_return",
        "future_close_return"
    ]

    data = df.copy()

    data = data.dropna(
        subset=cols + required
    ).copy()

    if len(data) < 500:
        raise ValueError(
            "Not enough clean historical candles."
        )

    X = data[cols]

    # =====================================================
    # DIRECTION TARGET
    # =====================================================

    y_direction = (
        data["target"]
        .map({
            -1: 0,
            0: 1,
            1: 2
        })
        .astype(int)
    )

    # =====================================================
    # REGRESSION TARGETS
    # =====================================================

    y_open = data[
        "future_open_return"
    ]

    y_high = data[
        "future_high_return"
    ]

    y_low = data[
        "future_low_return"
    ]

    y_close = data[
        "future_close_return"
    ]

    split = int(
        len(data) * 0.80
    )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y_direction.iloc[:split]
    y_test = y_direction.iloc[split:]

    open_train = y_open.iloc[:split]
    open_test = y_open.iloc[split:]

    high_train = y_high.iloc[:split]
    high_test = y_high.iloc[split:]

    low_train = y_low.iloc[:split]
    low_test = y_low.iloc[split:]

    close_train = y_close.iloc[:split]
    close_test = y_close.iloc[split:]

    # =====================================================
    # CLASSIFICATION ENSEMBLE
    # =====================================================

    classifiers = _make_classifiers()

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
    # OHLC REGRESSION MODELS
    # =====================================================

    open_model = _regressor(101)
    high_model = _regressor(202)
    low_model = _regressor(303)
    close_model = _regressor(404)

    open_model.fit(
        X_train,
        open_train
    )

    high_model.fit(
        X_train,
        high_train
    )

    low_model.fit(
        X_train,
        low_train
    )

    close_model.fit(
        X_train,
        close_train
    )

    pred_open = open_model.predict(
        X_test
    )

    pred_high = high_model.predict(
        X_test
    )

    pred_low = low_model.predict(
        X_test
    )

    pred_close = close_model.predict(
        X_test
    )

    # =====================================================
    # METRICS
    # =====================================================

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

        "open_mae_pct": float(
            mean_absolute_error(
                open_test,
                pred_open
            ) * 100
        ),

        "high_mae_pct": float(
            mean_absolute_error(
                high_test,
                pred_high
            ) * 100
        ),

        "low_mae_pct": float(
            mean_absolute_error(
                low_test,
                pred_low
            ) * 100
        ),

        "close_mae_pct": float(
            mean_absolute_error(
                close_test,
                pred_close
            ) * 100
        )
    }

    return (
        {
            "classifiers": classifiers,

            "open_model": open_model,
            "high_model": high_model,
            "low_model": low_model,
            "close_model": close_model
        },
        cols,
        metrics
    )


# =========================================================
# PREDICT NEXT
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

    # =====================================================
    # DIRECTION
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

    # All classifiers use 0,1,2
    mapping = {
        int(k): float(v)
        for k, v in zip(
            models["classifiers"][0].classes_,
            avg_probability
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
        )
    }

    best = max(
        probs,
        key=probs.get
    )

    confidence = float(
        probs[best]
    )

    # =====================================================
    # MODEL AGREEMENT
    # =====================================================

    individual_predictions = []

    for model in models["classifiers"]:

        p = model.predict_proba(
            latest
        )[0]

        individual_predictions.append(
            int(np.argmax(p))
        )

    class_map = {
        "bearish": 0,
        "neutral": 1,
        "bullish": 2
    }

    agreement_count = (
        individual_predictions.count(
            class_map[best]
        )
    )

    if (
        confidence >= threshold
        and agreement_count >= 2
    ):

        signal = best.upper()

    else:

        signal = "NO EDGE"

    # =====================================================
    # CURRENT PRICE
    # =====================================================

    current_close = float(
        df["close"].iloc[-1]
    )

    # =====================================================
    # NEXT OHLC PREDICTIONS
    # =====================================================

    predicted_open_return = float(
        models["open_model"].predict(
            latest
        )[0]
    )

    predicted_high_return = float(
        models["high_model"].predict(
            latest
        )[0]
    )

    predicted_low_return = float(
        models["low_model"].predict(
            latest
        )[0]
    )

    predicted_close_return = float(
        models["close_model"].predict(
            latest
        )[0]
    )

    expected_open = (
        current_close
        * (1 + predicted_open_return)
    )

    predicted_high = (
        current_close
        * (1 + predicted_high_return)
    )

    predicted_low = (
        current_close
        * (1 + predicted_low_return)
    )

    predicted_close = (
        current_close
        * (1 + predicted_close_return)
    )

    # =====================================================
    # SANITY ORDER
    # =====================================================

    predicted_high = max(
        predicted_high,
        expected_open,
        predicted_close
    )

    predicted_low = min(
        predicted_low,
        expected_open,
        predicted_close
    )

    expected_move_pct = (
        predicted_close_return
        * 100
    )

    return (
        probs,
        signal,
        expected_open,
        predicted_close,
        expected_move_pct,
        predicted_high,
        predicted_low,
        agreement_count
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

    data = df.copy()

    data = data.dropna(
        subset=cols + [
            "target"
        ]
    ).copy()

    if len(data) <= (
        min_train + step
    ):

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

        classifiers = _make_classifiers()

        probabilities = []

        for model in classifiers:

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

        "signals": len(
            signal_preds
        ),

        "signal_accuracy":
            signal_accuracy,

        "bullish_signals":
            len(bullish),

        "bullish_accuracy":
            (
                float(np.mean(bullish))
                if bullish
                else 0.0
            ),

        "bearish_signals":
            len(bearish),

        "bearish_accuracy":
            (
                float(np.mean(bearish))
                if bearish
                else 0.0
            ),

        "neutral_signals":
            len(neutral),

        "neutral_accuracy":
            (
                float(np.mean(neutral))
                if neutral
                else 0.0
            )
    }
