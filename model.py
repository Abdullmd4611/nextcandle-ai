import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.metrics import accuracy_score, mean_absolute_error


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
    "future_return",
    "future_close",
}


# =========================================================
# FEATURE COLUMNS
# =========================================================

def feature_columns(df):

    return [
        c
        for c in df.columns
        if c not in EXCLUDE
        and pd.api.types.is_numeric_dtype(df[c])
    ]


# =========================================================
# CLASSIFIER
# =========================================================

def _classifier(
    random_state=42,
    max_iter=120,
    learning_rate=0.04,
    max_leaf_nodes=12,
    min_samples_leaf=25,
    l2_regularization=1.5
):

    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state
    )


# =========================================================
# REGRESSOR
# =========================================================

def _regressor():

    return HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=12,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=42
    )


# =========================================================
# EMPTY BACKTEST RESULT
# =========================================================

def _empty_backtest():

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


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(df):

    cols = feature_columns(df)

    if not cols:

        raise ValueError(
            "No usable model features were found."
        )

    data = df.copy()

    # Next-candle return
    data["future_return"] = (
        data["close"].shift(-1)
        / data["close"]
        - 1
    )

    data["future_close"] = (
        data["close"].shift(-1)
    )

    data = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(
            subset=cols + [
                "target",
                "future_return",
                "future_close"
            ]
        )
        .copy()
    )

    if len(data) < 500:

        raise ValueError(
            f"Not enough clean historical candles. "
            f"Only {len(data)} available."
        )

    X = data[cols]

    # -----------------------------------------------------
    # Direction target
    #
    # -1 = bearish = 0
    #  0 = neutral = 1
    #  1 = bullish = 2
    # -----------------------------------------------------

    y_direction = (
        data["target"]
        .map({
            -1: 0,
            0: 1,
            1: 2
        })
        .astype(int)
    )

    y_return = data["future_return"]

    split = int(len(data) * 0.80)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y_direction.iloc[:split]
    y_test = y_direction.iloc[split:]

    r_train = y_return.iloc[:split]
    r_test = y_return.iloc[split:]

    # =====================================================
    # THREE LIVE CLASSIFICATION MODELS
    # =====================================================

    classifiers = [

        _classifier(
            random_state=42,
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=12,
            min_samples_leaf=25,
            l2_regularization=1.5
        ),

        _classifier(
            random_state=123,
            max_iter=140,
            learning_rate=0.03,
            max_leaf_nodes=10,
            min_samples_leaf=30,
            l2_regularization=2.0
        ),

        _classifier(
            random_state=321,
            max_iter=110,
            learning_rate=0.05,
            max_leaf_nodes=10,
            min_samples_leaf=20,
            l2_regularization=1.0
        )
    ]

    # =====================================================
    # TRAIN THREE MODELS
    # =====================================================

    for model in classifiers:

        model.fit(
            X_train,
            y_train
        )

    # =====================================================
    # ENSEMBLE TEST
    # =====================================================

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
    # REGRESSION MODEL
    # =====================================================

    regressor = _regressor()

    regressor.fit(
        X_train,
        r_train
    )

    predicted_return = regressor.predict(
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


# =========================================================
# PREDICT NEXT 15M CANDLE
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
    # ENSEMBLE PROBABILITIES
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

    # =====================================================
    # FIXED CLASS MAPPING
    # =====================================================

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

    # =====================================================
    # BEST DIRECTION
    # =====================================================

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
            int(
                np.argmax(p)
            )
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

    # =====================================================
    # SIGNAL
    # =====================================================

    if (
        confidence >= threshold
        and agreement_count >= 2
    ):

        signal = best.upper()

    else:

        signal = "NO EDGE"

    # =====================================================
    # NEXT CANDLE RETURN
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

    # =====================================================
    # IMPORTANT:
    # app.py expects EXACTLY FIVE VALUES.
    # =====================================================

    return (
        probs,
        signal,
        expected_open,
        predicted_close,
        expected_move_pct
    )


# =========================================================
# FAST WALK-FORWARD BACKTEST
# =========================================================

def walk_forward_backtest(
    df,
    cols,
    min_train=1000,
    step=500,
    signal_threshold=0.65
):

    data = df.copy()

    data["future_return"] = (
        data["close"].shift(-1)
        / data["close"]
        - 1
    )

    data = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(
            subset=cols + [
                "target",
                "future_return"
            ]
        )
        .copy()
    )

    # =====================================================
    # NOT ENOUGH DATA
    # =====================================================

    if len(data) <= min_train + 50:

        return _empty_backtest()

    preds = []
    actuals = []

    signal_preds = []

    bullish = []
    bearish = []
    neutral = []

    # =====================================================
    # FAST VALIDATION
    #
    # step=500 means only a small number of
    # validation training cycles.
    # =====================================================

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

        # =================================================
        # ONE FAST MODEL FOR VALIDATION
        #
        # Live prediction still uses THREE models.
        # =================================================

        model = _classifier(
            random_state=42,
            max_iter=80,
            learning_rate=0.04,
            max_leaf_nodes=10,
            min_samples_leaf=25,
            l2_regularization=1.5
        )

        model.fit(
            train[cols],
            ytrain
        )

        probabilities = model.predict_proba(
            test[cols]
        )

        pred = np.argmax(
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
            pred.tolist()
        )

        actuals.extend(
            actual
        )

        confidence = probabilities.max(
            axis=1
        )

        # =================================================
        # HIGH-CONFIDENCE SIGNALS
        # =================================================

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

    # =====================================================
    # OVERALL ACCURACY
    # =====================================================

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

    # =====================================================
    # SIGNAL ACCURACY
    # =====================================================

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

    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        "predictions": len(preds),

        "accuracy": accuracy,

        "signals": len(signal_preds),

        "signal_accuracy": signal_accuracy,

        "bullish_signals": len(bullish),

        "bullish_accuracy": (
            float(np.mean(bullish))
            if bullish
            else 0.0
        ),

        "bearish_signals": len(bearish),

        "bearish_accuracy": (
            float(np.mean(bearish))
            if bearish
            else 0.0
        ),

        "neutral_signals": len(neutral),

        "neutral_accuracy": (
            float(np.mean(neutral))
            if neutral
            else 0.0
        )
    }
