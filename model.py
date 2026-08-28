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


MODEL_PATH = "nextcandle_model.joblib"


class NextCandleModel:
    """
    NextCandle AI V2

    Classes:
        0 = BEARISH
        1 = NEUTRAL
        2 = BULLISH
    """

    CLASS_NAMES = {
        0: "BEARISH",
        1: "NEUTRAL",
        2: "BULLISH",
    }

    def __init__(
        self,
        model_path=MODEL_PATH,
        random_state=42,
    ):
        self.model_path = model_path
        self.random_state = random_state

        self.model = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=500,
            max_leaf_nodes=31,
            max_depth=None,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=random_state,
        )

        self.feature_columns = None
        self.is_fitted = False

    # ========================================================
    # TRAIN
    # ========================================================

    def fit(self, X, y):
        """
        Train the model.

        X:
            Feature dataframe.

        y:
            0 = bearish
            1 = neutral
            2 = bullish
        """

        if X is None or X.empty:
            raise ValueError("Training features are empty.")

        if y is None or len(y) == 0:
            raise ValueError("Training target is empty.")

        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: "
                f"{len(X)} != {len(y)}"
            )

        X = X.copy()

        y = pd.Series(y).astype(int)

        # Make sure the model sees the exact same columns
        # every time it is trained or used.
        self.feature_columns = list(X.columns)

        # Replace invalid numerical values.
        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # HistGradientBoosting can handle NaN values.
        # We intentionally do NOT blindly forward-fill data.
        X = X.astype(float)

        # Validate target classes.
        unique_classes = sorted(y.unique().tolist())

        invalid_classes = [
            c for c in unique_classes
            if c not in (0, 1, 2)
        ]

        if invalid_classes:
            raise ValueError(
                f"Invalid target classes: {invalid_classes}. "
                "Expected only 0, 1, 2."
            )

        if len(unique_classes) < 2:
            raise ValueError(
                "Training data contains fewer than two "
                "classes. More market history is required."
            )

        self.model.fit(X, y)

        self.is_fitted = True

        return self

    # ========================================================
    # PREDICTION
    # ========================================================

    def _prepare_features(self, X):
        if not self.is_fitted:
            raise RuntimeError(
                "Model has not been trained yet."
            )

        if isinstance(X, pd.Series):
            X = X.to_frame().T

        X = X.copy()

        # Ensure feature order is identical to training.
        missing = [
            col
            for col in self.feature_columns
            if col not in X.columns
        ]

        if missing:
            raise ValueError(
                "Missing model features: "
                f"{missing}"
            )

        X = X[self.feature_columns]

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return X.astype(float)

    def predict_proba(self, X):
        """
        Return probability for every class.
        """

        X = self._prepare_features(X)

        probabilities = self.model.predict_proba(X)

        # sklearn returns probabilities in class order.
        # Rebuild explicitly so our output always has:
        # [bearish, neutral, bullish]
        output = np.zeros(
            (len(X), 3),
            dtype=float,
        )

        for index, class_id in enumerate(
            self.model.classes_
        ):
            output[:, int(class_id)] = probabilities[:, index]

        return output

    def predict(self, X):
        """
        Return class prediction.
        """

        probabilities = self.predict_proba(X)

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
        """
        Convert probabilities into a trading-style signal.

        The model does NOT force a trade.

        If confidence/edge is weak:
            WAIT

        Otherwise:
            BEARISH / BULLISH

        Neutral predictions become WAIT.
        """

        probabilities = self.predict_proba(X)

        results = []

        for probs in probabilities:

            bearish = float(probs[0])
            neutral = float(probs[1])
            bullish = float(probs[2])

            ordered = np.argsort(probs)[::-1]

            best_class = int(ordered[0])
            second_class = int(ordered[1])

            confidence = float(probs[best_class])

            edge = float(
                probs[best_class]
                - probs[second_class]
            )

            if (
                best_class == 2
                and confidence >= minimum_confidence
                and edge >= minimum_edge
            ):
                signal = "BULLISH"

            elif (
                best_class == 0
                and confidence >= minimum_confidence
                and edge >= minimum_edge
            ):
                signal = "BEARISH"

            else:
                signal = "WAIT"

            results.append(
                {
                    "signal": signal,
                    "prediction": self.CLASS_NAMES[
                        best_class
                    ],
                    "confidence": confidence,
                    "edge": edge,
                    "bearish_probability": bearish,
                    "neutral_probability": neutral,
                    "bullish_probability": bullish,
                }
            )

        return results

    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate(self, X, y):
        """
        Evaluate the trained model.

        Returns metrics useful for determining whether
        the model is actually learning or simply guessing.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be trained before evaluation."
            )

        X = self._prepare_features(X)
        y = pd.Series(y).astype(int)

        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = {
            "accuracy": float(
                accuracy_score(
                    y,
                    predictions,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y,
                    predictions,
                )
            ),
            "log_loss": float(
                log_loss(
                    y,
                    probabilities,
                    labels=[0, 1, 2],
                )
            ),
            "confusion_matrix": confusion_matrix(
                y,
                predictions,
                labels=[0, 1, 2],
            ).tolist(),
            "classification_report": classification_report(
                y,
                predictions,
                labels=[0, 1, 2],
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
        """
        Save the trained model and feature schema.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an untrained model."
            )

        path = path or self.model_path

        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "random_state": self.random_state,
            "version": "2.0",
        }

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        joblib.dump(
            payload,
            path,
        )

        return path

    # ========================================================
    # LOAD
    # ========================================================

    def load(self, path=None):
        """
        Load a previously trained model.
        """

        path = path or self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        payload = joblib.load(path)

        self.model = payload["model"]

        self.feature_columns = payload[
            "feature_columns"
        ]

        self.random_state = payload.get(
            "random_state",
            42,
        )

        self.is_fitted = True

        return self

    # ========================================================
    # HUMAN-READABLE PREDICTION
    # ========================================================

    def explain_prediction(self, X):
        """
        Return a compact explanation of the model's
        probability distribution.
        """

        probabilities = self.predict_proba(X)

        explanations = []

        for probs in probabilities:

            bearish = float(probs[0])
            neutral = float(probs[1])
            bullish = float(probs[2])

            if bullish >= bearish and bullish >= neutral:
                dominant = "BULLISH"

            elif bearish >= bullish and bearish >= neutral:
                dominant = "BEARISH"

            else:
                dominant = "NEUTRAL"

            explanations.append(
                {
                    "dominant_class": dominant,
                    "bearish": round(
                        bearish * 100,
                        2,
                    ),
                    "neutral": round(
                        neutral * 100,
                        2,
                    ),
                    "bullish": round(
                        bullish * 100,
                        2,
                    ),
                }
            )

        return explanations
