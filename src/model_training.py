"""Model training and selection for FOMC unemployment prediction."""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score

from src.config import CV_N_SPLITS


def get_candidate_models():
    """Return dict of model name -> (estimator, param_grid) for grid search."""
    return {
        "LinearRegression": (
            LinearRegression(),
            {},
        ),
        "Ridge": (
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0]},
        ),
        "Lasso": (
            Lasso(max_iter=10000),
            {"alpha": [0.01, 0.1, 1.0]},
        ),
        "ElasticNet": (
            ElasticNet(max_iter=10000),
            {"alpha": [0.1, 1.0], "l1_ratio": [0.3, 0.5, 0.7]},
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=42, n_estimators=100),
            {"max_depth": [3, 5], "min_samples_leaf": [5, 10]},
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42, n_estimators=100),
            {"max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
        ),
        "XGBoost": (
            XGBRegressor(random_state=42, n_estimators=100, verbosity=0),
            {"max_depth": [2, 3], "learning_rate": [0.05, 0.1], "reg_alpha": [1.0, 10.0]},
        ),
    }


def run_model_selection(X_train, y_train):
    """Run GridSearchCV for each candidate model with TimeSeriesSplit.

    Returns:
        results: list of dicts with model name, best params, best score, fitted model
        best_model: the best overall fitted model
        best_name: name of the best model
    """
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)
    candidates = get_candidate_models()
    results = []

    for name, (estimator, param_grid) in candidates.items():
        print(f"  Training {name}...", flush=True)
        if param_grid:
            gs = GridSearchCV(
                estimator, param_grid, cv=tscv,
                scoring="neg_mean_absolute_error", n_jobs=-1, refit=True,
            )
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            best_score = -gs.best_score_
            best_params = gs.best_params_
        else:
            estimator.fit(X_train, y_train)
            best_model = estimator
            scores = cross_val_score(
                estimator, X_train, y_train, cv=tscv,
                scoring="neg_mean_absolute_error",
            )
            best_score = -scores.mean()
            best_params = {}

        results.append({
            "name": name,
            "best_params": best_params,
            "cv_mae": best_score,
            "model": best_model,
        })
        print(f"    CV MAE: {best_score:.4f}", flush=True)

    # Sort by CV MAE (lower is better)
    results.sort(key=lambda r: r["cv_mae"])

    # Try ensemble of top 3
    top3 = results[:3]
    ensemble = VotingRegressor(
        estimators=[(r["name"], r["model"]) for r in top3]
    )
    ensemble_scores = cross_val_score(
        ensemble, X_train, y_train, cv=tscv,
        scoring="neg_mean_absolute_error",
    )
    ensemble_mae = -ensemble_scores.mean()
    print(f"  Ensemble (top 3) CV MAE: {ensemble_mae:.4f}", flush=True)

    results.append({
        "name": "Ensemble_Top3",
        "best_params": {"components": [r["name"] for r in top3]},
        "cv_mae": ensemble_mae,
        "model": ensemble,
    })

    # Re-sort
    results.sort(key=lambda r: r["cv_mae"])
    best_name = results[0]["name"]
    best_model = results[0]["model"]

    return results, best_model, best_name


def retrain_on_full_data(model_template, X, y):
    """Retrain the selected model on all available data."""
    model_template.fit(X, y)
    return model_template
