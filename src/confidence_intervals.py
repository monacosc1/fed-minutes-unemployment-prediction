"""Bootstrap-based prediction intervals."""

import numpy as np
from sklearn.base import clone
from src.config import BOOTSTRAP_N_ITERATIONS, CONFIDENCE_LEVEL


class BootstrapPredictor:
    """Generates prediction intervals via bootstrap aggregation.

    Trains multiple model copies on resampled data, then collects
    predictions to produce percentile-based confidence intervals.
    """

    def __init__(self, base_model, n_iterations=BOOTSTRAP_N_ITERATIONS, confidence=CONFIDENCE_LEVEL):
        self.base_model = base_model
        self.n_iterations = n_iterations
        self.confidence = confidence
        self.models_ = []

    def fit(self, X, y):
        """Fit bootstrap ensemble on training data."""
        n_samples = len(X)
        self.models_ = []

        for i in range(self.n_iterations):
            # Resample with replacement
            idx = np.random.RandomState(i).choice(n_samples, size=n_samples, replace=True)
            X_boot = X.iloc[idx]
            y_boot = y.iloc[idx]

            model = clone(self.base_model)
            model.fit(X_boot, y_boot)
            self.models_.append(model)

        return self

    def predict(self, X):
        """Return median prediction and confidence interval.

        Returns:
            median: median of bootstrap predictions
            lower: lower bound of confidence interval
            upper: upper bound of confidence interval
            all_preds: array of all bootstrap predictions
        """
        all_preds = np.array([m.predict(X) for m in self.models_])
        # all_preds shape: (n_iterations, n_samples)

        alpha = (1 - self.confidence) / 2
        lower = np.percentile(all_preds, alpha * 100, axis=0)
        upper = np.percentile(all_preds, (1 - alpha) * 100, axis=0)
        median = np.median(all_preds, axis=0)

        return median, lower, upper, all_preds
