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

    # Future/target columns
    "target",
    "future_return",
    "future_close",
    "future_open_return",
    "future_high_return",
    "future_low_return",
    "future_close_return",

    # Backtest-only columns
    "next_open",
    "next_close",
    "actual_direction",
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
# EMPTY BACKTEST
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
        "neutral_accuracy": 0.0,

        "correct": 0,
        "wrong": 0,

        "test_results": []
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

    # -----------------------------------------------------
    # Future close return
    # -----------------------------------------------------

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
    # Existing V7 target
    #
    # -1 = bearish
    #  0 = neutral
    #  1 = bullish
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
    # THREE CLASSIFIERS
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
    # TRAIN
    # =====================================================

    for model in classifiers:

        model.fit(
            X_train,
            y_train
        )

    # =====================================================
    # HOLDOUT TEST
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
    # REGRESSION
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
# LIVE NEXT CANDLE PREDICTION
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
    # PROBABILITY MAPPING
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
    # LIVE SIGNAL
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

    return (
        probs,
        signal,
        expected_open,
        predicted_close,
        expected_move_pct
    )


# =========================================================
# 80-CANDLE BLIND NEXT-CANDLE TEST
# =========================================================

def walk_forward_backtest(
    df,
    cols,
    n_tests=80,
    min_train=1000,
    signal_threshold=0.65
):

    data = df.copy()

    # =====================================================
    # ACTUAL NEXT CANDLE
    #
    # IMPORTANT:
    #
    # Prediction is made using candle i.
    #
    # Actual result is candle i+1.
    #
    # next close > next open = BULLISH
    # next close < next open = BEARISH
    # =====================================================

    data["next_open"] = (
        data["open"].shift(-1)
    )

    data["next_close"] = (
        data["close"].shift(-1)
    )

    data["actual_direction"] = np.where(
        data["next_close"]
        > data["next_open"],

        "BULLISH",

        "BEARISH"
    )

    # =====================================================
    # CLEAN
    # =====================================================

    data = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna(
            subset=cols + [
                "target",
                "next_open",
                "next_close"
            ]
        )
        .reset_index(drop=True)
    )

    if len(data) <= min_train + n_tests:

        return _empty_backtest()

    # =====================================================
    # LAST 80 TEST POINTS
    # =====================================================

    start = len(data) - n_tests

    results = []

    # =====================================================
    # TRUE WALK-FORWARD
    # =====================================================

    for i in range(
        start,
        len(data)
    ):

        # -------------------------------------------------
        # ONLY DATA BEFORE TEST POINT
        # -------------------------------------------------

        train = data.iloc[:i]

        if len(train) < min_train:

            continue

        # =================================================
        # TRAINING TARGET
        # =================================================

        y_train = (
            train["target"]
            .map({
                -1: 0,
                0: 1,
                1: 2
            })
            .astype(int)
        )

        if y_train.nunique() < 2:

            continue

        # =================================================
        # THREE MODELS
        # =================================================

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

        # =================================================
        # TRAIN
        # =================================================

        for model in classifiers:

            model.fit(
                train[cols],
                y_train
            )

        # =================================================
        # CURRENT CANDLE
        # =================================================

        current = data.iloc[[i]]

        # =================================================
        # PREDICT
        # =================================================

        probabilities = []

        for model in classifiers:

            probabilities.append(
                model.predict_proba(
                    current[cols]
                )[0]
            )

        avg_probability = np.mean(
            probabilities,
            axis=0
        )

        # =================================================
        # MAP PROBABILITIES
        # =================================================

        mapping = {
            int(k): float(v)
            for k, v in zip(
                classifiers[0].classes_,
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

        # =================================================
        # MODEL PREDICTION
        # =================================================

        best = max(
            probs,
            key=probs.get
        )

        confidence = float(
            probs[best]
        )

        prediction = best.upper()

        # =================================================
        # AGREEMENT
        # =================================================

        individual_predictions = []

        for model in classifiers:

            p = model.predict_proba(
                current[cols]
            )[0]

            individual_predictions.append(
                int(
                    np.argmax(p)
                )
            )

        class_map = {
            "BEARISH": 0,
            "NEUTRAL": 1,
            "BULLISH": 2
        }

        agreement_count = (
            individual_predictions.count(
                class_map[prediction]
            )
        )

        # =================================================
        # SIGNAL
        # =================================================

        if (
            confidence >= signal_threshold
            and agreement_count >= 2
        ):

            signal = prediction

        else:

            signal = "NO EDGE"

        # =================================================
        # ACTUAL NEXT CANDLE
        # =================================================

        actual = str(
            current[
                "actual_direction"
            ].iloc[0]
        )

        # =================================================
        # CORRECT PREDICTION
        # =================================================

        correct = (
            prediction == actual
        )

        # =================================================
        # CORRECT TRADE SIGNAL
        # =================================================

        signal_correct = (
            signal != "NO EDGE"
            and signal == actual
        )

        # =================================================
        # SAVE RESULT
        # =================================================

        results.append({

            "test": len(results) + 1,

            "timestamp": str(
                current[
                    "timestamp"
                ].iloc[0]
            ),

            "prediction": prediction,

            "confidence": confidence,

            "signal": signal,

            "agreement": agreement_count,

            "actual": actual,

            "correct": bool(
                correct
            ),

            "signal_correct": bool(
                signal_correct
            ),

            "next_open": float(
                current[
                    "next_open"
                ].iloc[0]
            ),

            "next_close": float(
                current[
                    "next_close"
                ].iloc[0]
            )
        })

    # =====================================================
    # NO RESULTS
    # =====================================================

    if not results:

        return _empty_backtest()

    result_df = pd.DataFrame(
        results
    )

    # =====================================================
    # CORRECT / WRONG
    # =====================================================

    correct_count = int(
        result_df["correct"].sum()
    )

    wrong_count = int(
        len(result_df)
        - correct_count
    )

    # =====================================================
    # OVERALL ACCURACY
    # =====================================================

    accuracy = float(
        result_df["correct"].mean()
    )

    # =====================================================
    # SIGNAL RESULTS
    # =====================================================

    signal_df = result_df[
        result_df["signal"] != "NO EDGE"
    ]

    if len(signal_df) > 0:

        signal_accuracy = float(
            signal_df[
                "signal_correct"
            ].mean()
        )

    else:

        signal_accuracy = 0.0

    # =====================================================
    # BULLISH
    # =====================================================

    bullish_df = result_df[
        result_df["prediction"]
        == "BULLISH"
    ]

    bullish_accuracy = (

        float(
            bullish_df["correct"].mean()
        )

        if len(bullish_df) > 0

        else 0.0
    )

    # =====================================================
    # BEARISH
    # =====================================================

    bearish_df = result_df[
        result_df["prediction"]
        == "BEARISH"
    ]

    bearish_accuracy = (

        float(
            bearish_df["correct"].mean()
        )

        if len(bearish_df) > 0

        else 0.0
    )

    # =====================================================
    # NEUTRAL
    # =====================================================

    neutral_df = result_df[
        result_df["prediction"]
        == "NEUTRAL"
    ]

    neutral_accuracy = (

        float(
            neutral_df["correct"].mean()
        )

        if len(neutral_df) > 0

        else 0.0
    )

    # =====================================================
    # RETURN
    # =====================================================

    return {

        "predictions": len(
            result_df
        ),

        "accuracy": accuracy,

        "correct": correct_count,

        "wrong": wrong_count,

        "signals": len(
            signal_df
        ),

        "signal_accuracy": (
            signal_accuracy
        ),

        "bullish_signals": len(
            bullish_df
        ),

        "bullish_accuracy": (
            bullish_accuracy
        ),

        "bearish_signals": len(
            bearish_df
        ),

        "bearish_accuracy": (
            bearish_accuracy
        ),

        "neutral_signals": len(
            neutral_df
        ),

        "neutral_accuracy": (
            neutral_accuracy
        ),

        "test_results": results
    }
