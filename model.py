import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


# ============================================================
# NextCandle AI — MODEL ENGINE V2
# ============================================================
#
# Classes:
#     0 = BEARISH
#     1 = NEUTRAL
#     2 = BULLISH
#
# Primary prediction:
#     NEXT 15-MINUTE CANDLE
#
# Important:
#     Training and evaluation must remain time ordered.
# ============================================================


MODEL_PATH = "nextcandle_model.joblib"


class NextCandleModel:

    CLASS_NAMES = {
        0: "BEARISH",
        1: "NEUTRAL",
        2: "BULLISH",
    }

    VERSION = "2.1"

    def __init__(
        self,
        model_path=MODEL_PATH,
        random_state=42,
    ):

        self.model_path = model_path
        self.random_state = random_state

        # ----------------------------------------------------
        # Gradient boosting model
        #
        # IMPORTANT:
        #
        # Internal sklearn early stopping is disabled.
        #
        # Market data is time-series data. We don't want the
        # estimator creating its own validation split.
        #
        # The application performs the chronological holdout
        # separately.
        # ----------------------------------------------------

        self.model = HistGradientBoostingClassifier(

            learning_rate=0.035,

            max_iter=500,

            max_leaf_nodes=31,

            max_depth=None,

            min_samples_leaf=30,

            l2_regularization=1.0,

            early_stopping=False,

            validation_fraction=None,

            n_iter_no_change=None,

            random_state=random_state,
        )

        self.feature_columns = None

        self.is_fitted = False


    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    @staticmethod
    def _validate_training_input(X, y):

        if X is None or X.empty:
            raise ValueError(
                "Training features are empty."
            )

        if y is None or len(y) == 0:
            raise ValueError(
                "Training target is empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y length mismatch: "
                f"{len(X)} != {len(y)}"
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = pd.Series(y).reset_index(
            drop=True
        )

        X = X.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Feature names must be unique.
        # ----------------------------------------------------

        if not X.columns.is_unique:

            duplicated = (
                X.columns[
                    X.columns.duplicated()
                ]
                .tolist()
            )

            raise ValueError(
                "Duplicate feature columns detected: "
                f"{duplicated}"
            )

        # ----------------------------------------------------
        # Target must contain only expected classes.
        # ----------------------------------------------------

        y_numeric = pd.to_numeric(
            y,
            errors="coerce",
        )

        if y_numeric.isna().any():

            raise ValueError(
                "Training target contains "
                "non-numeric values."
            )

        y = y_numeric.astype(int)

        unique_classes = sorted(
            y.unique().tolist()
        )

        invalid_classes = [
            value
            for value in unique_classes
            if value not in (0, 1, 2)
        ]

        if invalid_classes:

            raise ValueError(
                "Invalid target classes: "
                f"{invalid_classes}. "
                "Expected only 0, 1, 2."
            )

        if len(unique_classes) < 2:

            raise ValueError(
                "Training data contains fewer "
                "than two classes."
            )

        return X, y


    # ========================================================
    # TRAIN
    # ========================================================

    def fit(self, X, y):

        X, y = self._validate_training_input(
            X,
            y,
        )

        # ----------------------------------------------------
        # Store exact training schema.
        # ----------------------------------------------------

        self.feature_columns = list(
            X.columns
        )

        # ----------------------------------------------------
        # Clean numerical features.
        #
        # HistGradientBoosting supports NaN values.
        # ----------------------------------------------------

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X = X.astype(float)

        # ----------------------------------------------------
        # Train.
        #
        # No random train/test split occurs here.
        # The caller controls the chronological split.
        # ----------------------------------------------------

        self.model.fit(
            X,
            y,
        )

        self.is_fitted = True

        return self


    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    def _prepare_features(self, X):

        if not self.is_fitted:

            raise RuntimeError(
                "Model has not been trained yet."
            )

        if isinstance(X, pd.Series):

            X = X.to_frame().T

        if not isinstance(X, pd.DataFrame):

            X = pd.DataFrame(X)

        X = X.copy()

        # ----------------------------------------------------
        # Check for missing features.
        # ----------------------------------------------------

        missing = [
            column
            for column in self.feature_columns
            if column not in X.columns
        ]

        if missing:

            raise ValueError(
                "Missing model features: "
                f"{missing}"
            )

        # ----------------------------------------------------
        # Extra columns are harmless.
        # Only the exact training schema is used.
        # ----------------------------------------------------

        X = X[
            self.feature_columns
        ]

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X = X.astype(float)

        return X


    # ========================================================
    # PROBABILITY
    # ========================================================

    def predict_proba(self, X):

        X = self._prepare_features(
            X
        )

        probabilities = (
            self.model.predict_proba(X)
        )

        # ----------------------------------------------------
        # Always return:
        #
        # [BEARISH, NEUTRAL, BULLISH]
        #
        # regardless of sklearn's internal class ordering.
        # ----------------------------------------------------

        output = np.zeros(
            (
                len(X),
                3,
            ),
            dtype=float,
        )

        for index, class_id in enumerate(
            self.model.classes_
        ):

            class_id = int(class_id)

            if class_id in (
                0,
                1,
                2,
            ):

                output[
                    :,
                    class_id
                ] = probabilities[
                    :,
                    index
                ]

        # ----------------------------------------------------
        # Numerical safety.
        # ----------------------------------------------------

        output = np.clip(
            output,
            0.0,
            1.0,
        )

        row_sums = output.sum(
            axis=1,
            keepdims=True,
        )

        valid_rows = (
            row_sums[:, 0] > 0
        )

        output[
            valid_rows
        ] = (
            output[
                valid_rows
            ]
            / row_sums[
                valid_rows
            ]
        )

        return output


    # ========================================================
    # CLASS PREDICTION
    # ========================================================

    def predict(self, X):

        probabilities = (
            self.predict_proba(X)
        )

        return np.argmax(
            probabilities,
            axis=1,
        )


    # ========================================================
    # SIGNAL
    # ========================================================

    def signal(
        self,
        X,
        minimum_confidence=0.55,
        minimum_edge=0.10,
    ):

        if not (
            0.0
            <= minimum_confidence
            <= 1.0
        ):

            raise ValueError(
                "minimum_confidence must "
                "be between 0 and 1."
            )

        if not (
            0.0
            <= minimum_edge
            <= 1.0
        ):

            raise ValueError(
                "minimum_edge must "
                "be between 0 and 1."
            )

        probabilities = (
            self.predict_proba(X)
        )

        results = []

        for probs in probabilities:

            bearish = float(
                probs[0]
            )

            neutral = float(
                probs[1]
            )

            bullish = float(
                probs[2]
            )

            ordered = np.argsort(
                probs
            )[::-1]

            best_class = int(
                ordered[0]
            )

            second_class = int(
                ordered[1]
            )

            confidence = float(
                probs[best_class]
            )

            edge = float(
                probs[best_class]
                - probs[second_class]
            )

            # ------------------------------------------------
            # We only issue a directional signal when:
            #
            # 1. The model's strongest class is bullish/bearish
            # 2. Confidence is high enough
            # 3. Probability edge is high enough
            #
            # Otherwise WAIT.
            # ------------------------------------------------

            if (
                best_class == 2
                and confidence
                >= minimum_confidence
                and edge
                >= minimum_edge
            ):

                signal = "BULLISH"

            elif (
                best_class == 0
                and confidence
                >= minimum_confidence
                and edge
                >= minimum_edge
            ):

                signal = "BEARISH"

            else:

                signal = "WAIT"

            results.append(
                {
                    "signal": signal,

                    "prediction":
                        self.CLASS_NAMES[
                            best_class
                        ],

                    "confidence":
                        confidence,

                    "edge":
                        edge,

                    "bearish_probability":
                        bearish,

                    "neutral_probability":
                        neutral,

                    "bullish_probability":
                        bullish,
                }
            )

        return results


    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate(self, X, y):

        if not self.is_fitted:

            raise RuntimeError(
                "Model must be trained "
                "before evaluation."
            )

        X = self._prepare_features(
            X
        )

        y = pd.Series(
            y
        ).astype(int)

        if len(X) != len(y):

            raise ValueError(
                "Evaluation X/y length mismatch: "
                f"{len(X)} != {len(y)}"
            )

        predictions = self.predict(
            X
        )

        probabilities = (
            self.predict_proba(X)
        )

        metrics = {

            "accuracy":
                float(
                    accuracy_score(
                        y,
                        predictions,
                    )
                ),

            "balanced_accuracy":
                float(
                    balanced_accuracy_score(
                        y,
                        predictions,
                    )
                ),

            "log_loss":
                float(
                    log_loss(
                        y,
                        probabilities,
                        labels=[
                            0,
                            1,
                            2,
                        ],
                    )
                ),

            "confusion_matrix":
                confusion_matrix(
                    y,
                    predictions,
                    labels=[
                        0,
                        1,
                        2,
                    ],
                ).tolist(),

            "classification_report":
                classification_report(
                    y,
                    predictions,
                    labels=[
                        0,
                        1,
                        2,
                    ],
                    target_names=[
                        "BEARISH",
                        "NEUTRAL",
                        "BULLISH",
                    ],
                    zero_division=0,
                ),
        }

        return metrics


    # ========================================================
    # SAVE
    # ========================================================

    def save(self, path=None):

        if not self.is_fitted:

            raise RuntimeError(
                "Cannot save an untrained model."
            )

        path = (
            path
            or self.model_path
        )

        directory = os.path.dirname(
            path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

        payload = {

            "model":
                self.model,

            "feature_columns":
                self.feature_columns,

            "random_state":
                self.random_state,

            "version":
                self.VERSION,
        }

        joblib.dump(
            payload,
            path,
        )

        return path


    # ========================================================
    # LOAD
    # ========================================================

    def load(self, path=None):

        path = (
            path
            or self.model_path
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        payload = joblib.load(
            path
        )

        if "model" not in payload:

            raise ValueError(
                "Invalid saved model: "
                "missing model object."
            )

        if "feature_columns" not in payload:

            raise ValueError(
                "Invalid saved model: "
                "missing feature schema."
            )

        self.model = payload[
            "model"
        ]

        self.feature_columns = list(
            payload[
                "feature_columns"
            ]
        )

        self.random_state = payload.get(
            "random_state",
            42,
        )

        self.is_fitted = True

        return self


    # ========================================================
    # HUMAN-READABLE EXPLANATION
    # ========================================================

    def explain_prediction(self, X):

        probabilities = (
            self.predict_proba(X)
        )

        explanations = []

        for probs in probabilities:

            bearish = float(
                probs[0]
            )

            neutral = float(
                probs[1]
            )

            bullish = float(
                probs[2]
            )

            dominant_index = int(
                np.argmax(probs)
            )

            dominant = (
                self.CLASS_NAMES[
                    dominant_index
                ]
            )

            explanations.append(
                {
                    "dominant_class":
                        dominant,

                    "bearish":
                        round(
                            bearish * 100,
                            2,
                        ),

                    "neutral":
                        round(
                            neutral * 100,
                            2,
                        ),

                    "bullish":
                        round(
                            bullish * 100,
                            2,
                        ),
                }
            )

        return explanations
