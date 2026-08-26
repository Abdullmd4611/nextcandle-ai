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


def _encode_target(series):
    return (
        series
        .map({
            -1: 0,
            0: 1,
            1: 2
        })
        .astype(int)
    )


def _confidence_bucket(confidence):
    if confidence < 0.55:
        return "50-55%"

    if confidence < 0.60:
        return "55-60%"

    if confidence < 0.65:
        return "60-65%"

    if confidence < 0.70:
        return "65-70%"

    if confidence < 0.75:
        return "70-75%"

    if confidence < 0.80:
        return "75-80%"

    return "80%+"


def _build_calibration_table(
    confidence,
    correct
):

    rows = []

    buckets = [
        (0.50, 0.55, "50-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 0.70, "65-70%"),
        (0.70, 0.75, "70-75%"),
        (0.75, 0.80, "75-80%"),
        (0.80, 1.01, "80%+"),
    ]

    for low, high, label in buckets:

        mask = (
            (confidence >= low)
            & (confidence < high)
        )

        count = int(mask.sum())

        if count:

            actual_accuracy = float(
                np.mean(correct[mask])
            )

            average_confidence = float(
                np.mean(confidence[mask])
            )

        else:

            actual_accuracy = 0.0
            average_confidence = 0.0

        rows.append({
            "bucket": label,
            "samples": count,
            "average_confidence":
                average_confidence,
            "actual_accuracy":
                actual_accuracy
        })

    return rows


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
    y = _encode_target(
        data["target"]
    )

    # =====================================================
    # CHRONOLOGICAL SPLIT
    #
    # 70% training
    # 10% calibration
    # 20% final holdout
    #
    # Nothing is randomly shuffled.
    # =====================================================

    train_end = int(
        len(data) * 0.70
    )

    calibration_end = int(
        len(data) * 0.80
    )

    X_train = X.iloc[
        :train_end
    ]

    y_train = y.iloc[
        :train_end
    ]

    X_calibration = X.iloc[
        train_end:calibration_end
    ]

    y_calibration = y.iloc[
        train_end:calibration_end
    ]

    X_test = X.iloc[
        calibration_end:
    ]

    y_test = y.iloc[
        calibration_end:
    ]

    # =====================================================
    # TRAIN INITIAL MODEL
    # =====================================================

    calibration_model = _make_model()

    calibration_model.fit(
        X_train,
        y_train
    )

    # =====================================================
    # CALIBRATION DATA
    #
    # These candles were NOT used for training.
    # =====================================================

    calibration_probs = (
        calibration_model
        .predict_proba(
            X_calibration
        )
    )

    calibration_pred = (
        calibration_model
        .predict(
            X_calibration
        )
    )

    calibration_confidence = (
        calibration_probs.max(
            axis=1
        )
    )

    calibration_correct = (
        calibration_pred
        == y_calibration.to_numpy()
    )

    calibration_table = (
        _build_calibration_table(
            calibration_confidence,
            calibration_correct
        )
    )

    # =====================================================
    # FINAL HOLDOUT
    # =====================================================

    final_model = _make_model()

    # Train on the first 80%.
    X_final_train = X.iloc[
        :calibration_end
    ]

    y_final_train = y.iloc[
        :calibration_end
    ]

    final_model.fit(
        X_final_train,
        y_final_train
    )

    final_pred = final_model.predict(
        X_test
    )

    final_probs = (
        final_model
        .predict_proba(
            X_test
        )
    )

    holdout_accuracy = accuracy_score(
        y_test,
        final_pred
    )

    # =====================================================
    # CALIBRATION SUMMARY
    # =====================================================

    calibration_gap = []

    for row in calibration_table:

        if row["samples"] >= 10:

            calibration_gap.append(
                abs(
                    row["average_confidence"]
                    - row["actual_accuracy"]
                )
            )

    if calibration_gap:

        average_calibration_error = float(
            np.mean(calibration_gap)
        )

    else:

        average_calibration_error = 0.0

    metrics = {

        "holdout_accuracy":
            float(holdout_accuracy),

        "holdout_samples":
            int(len(y_test)),

        "train_samples":
            int(len(y_final_train)),

        "calibration_samples":
            int(len(y_calibration)),

        "calibration_table":
            calibration_table,

        "average_calibration_error":
            average_calibration_error
    }

    # Attach calibration information to model.
    final_model.calibration_table = (
        calibration_table
    )

    return (
        final_model,
        cols,
        metrics
    )


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
        "bearish":
            mapping.get(0, 0.0),

        "neutral":
            mapping.get(1, 0.0),

        "bullish":
            mapping.get(2, 0.0)
    }

    best = max(
        probs,
        key=probs.get
    )

    raw_confidence = probs[best]

    # =====================================================
    # HISTORICAL CALIBRATED CONFIDENCE
    # =====================================================

    calibrated_confidence = (
        raw_confidence
    )

    calibration_bucket = (
        _confidence_bucket(
            raw_confidence
        )
    )

    if hasattr(
        model,
        "calibration_table"
    ):

        for row in (
            model.calibration_table
        ):

            if (
                row["bucket"]
                == calibration_bucket
                and row["samples"] >= 10
            ):

                calibrated_confidence = (
                    row["actual_accuracy"]
                )

                break

    # =====================================================
    # SIGNAL
    #
    # Use raw ML probability for the initial threshold.
    # Calibration is displayed separately so we don't
    # pretend the calibrated number is a guarantee.
    # =====================================================

    signal = (
        best.upper()
        if raw_confidence >= threshold
        else "NO EDGE"
    )

    return (
        probs,
        signal
    )


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
            "neutral_accuracy": 0.0,

            "calibration_table": []
        }

    preds = []
    actuals = []

    signal_preds = []

    bullish = []
    bearish = []
    neutral = []

    confidence_history = []
    correctness_history = []

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

        ytrain = _encode_target(
            train["target"]
        )

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
            _encode_target(
                test["target"]
            )
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

            is_correct = (
                pp == yy
            )

            confidence_history.append(
                cc
            )

            correctness_history.append(
                is_correct
            )

            if cc >= signal_threshold:

                signal_preds.append(
                    (pp, yy)
                )

                if pp == 2:

                    bullish.append(
                        is_correct
                    )

                elif pp == 0:

                    bearish.append(
                        is_correct
                    )

                elif pp == 1:

                    neutral.append(
                        is_correct
                    )

    # =====================================================
    # OVERALL ACCURACY
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
    # HIGH-CONFIDENCE ACCURACY
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

    # =====================================================
    # WALK-FORWARD CALIBRATION
    #
    # This is especially important because these
    # predictions were generated strictly forward in time.
    # =====================================================

    confidence_array = np.array(
        confidence_history
    )

    correctness_array = np.array(
        correctness_history
    )

    calibration_table = (
        _build_calibration_table(
            confidence_array,
            correctness_array
        )
        if len(confidence_array)
        else []
    )

    calibration_errors = []

    for row in calibration_table:

        if row["samples"] >= 10:

            calibration_errors.append(
                abs(
                    row["average_confidence"]
                    - row["actual_accuracy"]
                )
            )

    walk_forward_calibration_error = (
        float(
            np.mean(
                calibration_errors
            )
        )
        if calibration_errors
        else 0.0
    )

    return {

        "predictions":
            int(len(preds)),

        "accuracy":
            accuracy,

        "signals":
            int(len(signal_preds)),

        "signal_accuracy":
            signal_accuracy,

        "bullish_signals":
            int(len(bullish)),

        "bullish_accuracy":
            bullish_accuracy,

        "bearish_signals":
            int(len(bearish)),

        "bearish_accuracy":
            bearish_accuracy,

        "neutral_signals":
            int(len(neutral)),

        "neutral_accuracy":
            neutral_accuracy,

        "calibration_table":
            calibration_table,

        "calibration_error":
            walk_forward_calibration_error
    }
