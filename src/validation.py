"""Time-series validation utilities."""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def time_series_cv_score(model, X, y, n_splits=5):
    """Evaluate model with TimeSeriesSplit cross-validation.

    Returns dict with mean and std of each metric across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, r2s = [], [], []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        maes.append(mean_absolute_error(y_val, preds))
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        r2s.append(r2_score(y_val, preds))

    return {
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "r2_mean": np.mean(r2s),
        "r2_std": np.std(r2s),
    }


def walk_forward_validation(model_class, model_params, X, y, dates, min_train_size=100):
    """Walk-forward expanding window validation.

    Trains on all data up to point t, predicts t+1, then expands.
    Returns DataFrame with date, actual, predicted for each step.
    """
    results = []

    for i in range(min_train_size, len(X)):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[[i]]
        y_test = y.iloc[i]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        results.append({
            "date": dates.iloc[i],
            "actual": y_test,
            "predicted": pred,
            "residual": y_test - pred,
            "train_size": i,
        })

    results_df = pd.DataFrame(results)
    return results_df


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
