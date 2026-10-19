
# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


class ModelEvaluator:
    """Train regression models and evaluate their performance."""

    def __init__(self, models):
        self.models = models
        self.results = []

    def evaluate(self, X_train, X_test, y_train, y_test):
        """Train each model and calculate evaluation metrics."""

        self.results = []

        for name, model in self.models.items():
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            self.results.append({
                "Model": name,
                "R2": r2_score(y_test, predictions),
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": mean_squared_error(
                    y_test, predictions
                ) ** 0.5
            })

        return pd.DataFrame(self.results).sort_values(
            by="RMSE"
        ).reset_index(drop=True)
