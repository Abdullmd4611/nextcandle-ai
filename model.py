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
        "neutral_accuracy": 0.0,

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
    # Next-candle close return
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
    # APP EXPECTS EXACTLY FIVE VALUES
    # =====================================================

    return (
        probs,
        signal,
        expected_open,
        predicted_close,
        expected_move_pct
    )


# =========================================================
# 80-TEST NEXT-CANDLE REALITY BACKTEST
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
    # This is the REAL thing we want to measure:
    #
    # next OPEN -> next CLOSE
    #
    # close > open = BULLISH
    # close < open = BEARISH
    # =====================================================

    data["next_open"] = (
        data["open"].shift(-1)
    )

    data["next_close"] = (
        data["close"].shift(-1)
    )

    data["actual_direction"] = np.select(
        [
            data["next_close"]
            > data["next_open"],

            data["next_close"]
            < data["next_open"]
        ],
        [
            "BULLISH",
            "BEARISH"
        ],
        default="NEUTRAL"
    )

    # =====================================================
    # CLEAN DATA
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
                "next_close",
                "actual_direction"
            ]
        )
        .reset_index(drop=True)
    )

    # =====================================================
    # NOT ENOUGH DATA
    # =====================================================

    if len(data) <= min_train + n_tests:

        return _empty_backtest()

    # =====================================================
    # LAST 80 HISTORICAL PREDICTION POINTS
    # =====================================================

    start = len(data) - n_tests

    results = []

    # =====================================================
    # TRUE WALK-FORWARD TEST
    # =====================================================

    for i in range(
        start,
        len(data)
    ):

        # -------------------------------------------------
        # ONLY USE INFORMATION AVAILABLE BEFORE
        # THE CANDLE BEING PREDICTED
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
        # THREE MODELS — SAME AS LIVE V7
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
        # CURRENT HISTORICAL CANDLE
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
        # PROBABILITY MAPPING
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
        # BEST PREDICTION
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
        # MODEL AGREEMENT
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
        # CORRECT?
        # =================================================

        correct = (
            prediction == actual
        )

        signal_correct = (
            signal != "NO EDGE"
            and signal == actual
        )

        # =================================================
        # SAVE TEST
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
    # OVERALL ACCURACY
    # =====================================================

    accuracy = float(
        result_df[
            "correct"
        ].mean()
    )

    # =====================================================
    # HIGH-CONFIDENCE SIGNALS
    # =====================================================

    signal_df = result_df[
        result_df[
            "signal"
        ] != "NO EDGE"
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
    # BULLISH PREDICTIONS
    # =====================================================

    bullish_df = result_df[
        result_df[
            "prediction"
        ] == "BULLISH"
    ]

    # =====================================================
    # BEARISH PREDICTIONS
    # =====================================================

    bearish_df = result_df[
        result_df[
            "prediction"
        ] == "BEARISH"
    ]

    # =====================================================
    # NEUTRAL PREDICTIONS
    # =====================================================

    neutral_df = result_df[
        result_df[
            "prediction"
        ] == "NEUTRAL"
    ]

    # =====================================================
    # RETURN
    # =====================================================

    return {

        "predictions": len(
            result_df
        ),

        "accuracy": accuracy,

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
            float(
                bullish_df[
                    "correct"
                ].mean()
            )
            if len(bullish_df) > 0
            else 0.0
        ),

        "bearish_signals": len(
            bearish_df
        ),

        "bearish_accuracy": (
            float(
                bearish_df[
                    "correct"
                ].mean()
            )
            if len(bearish_df) > 0
            else 0.0
        ),

        "neutral_signals": len(
            neutral_df
        ),

        "neutral_accuracy": (
            float(
                neutral_df[
                    "correct"
                ].mean()
            )
            if len(neutral_df) > 0
            else 0.0
        ),

        "test_results": results
    }
