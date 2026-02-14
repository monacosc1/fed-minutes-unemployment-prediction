"""Full training pipeline orchestrator."""

import json
import os

import joblib
import numpy as np
import pandas as pd

from src.config import TEST_SIZE_FRACTION
from src.data_loader import build_dataset
from src.feature_engineering import FOMCFeatureEngineer
from src.model_training import run_model_selection, retrain_on_full_data
from src.validation import compute_metrics
from src.confidence_intervals import BootstrapPredictor


def run_pipeline(csv_path: str, output_dir: str = "models"):
    """Execute the full training pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    print("Step 1: Loading data...", flush=True)
    df = build_dataset(csv_path)
    print(f"  Loaded {len(df)} meetings from {df['Date'].min()} to {df['Date'].max()}", flush=True)

    # Step 2: Temporal train/test split
    print("Step 2: Splitting data...", flush=True)
    split_idx = int(len(df) * (1 - TEST_SIZE_FRACTION))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}", flush=True)

    # Step 3: Feature engineering
    print("Step 3: Engineering features...", flush=True)
    fe = FOMCFeatureEngineer()
    X_train = fe.fit_transform(train_df)
    X_test = fe.transform(test_df)
    y_train = train_df["unemployment_rate_next"].reset_index(drop=True)
    y_test = test_df["unemployment_rate_next"].reset_index(drop=True)

    # Drop rows with NaN from lag features
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    train_mask = X_train.notna().all(axis=1)
    X_train = X_train[train_mask].reset_index(drop=True)
    y_train = y_train[train_mask].reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    test_dates_all = test_df["Date"].reset_index(drop=True)
    test_mask = X_test.notna().all(axis=1)
    X_test = X_test[test_mask].reset_index(drop=True)
    y_test = y_test[test_mask].reset_index(drop=True)
    test_dates = test_dates_all[test_mask].reset_index(drop=True)

    print(f"  Features: {X_train.shape[1]} | Train: {len(X_train)} | Test: {len(X_test)}", flush=True)

    # Step 4: Model selection
    print("Step 4: Running model selection...", flush=True)
    results, best_model, best_name = run_model_selection(X_train, y_train)
    print(f"  Best model: {best_name} (CV MAE: {results[0]['cv_mae']:.4f})", flush=True)

    # Step 5: Evaluate on test set
    print("Step 5: Evaluating on test set...", flush=True)
    best_model.fit(X_train, y_train)
    test_preds = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, pd.Series(test_preds))
    print(f"  Test MAE: {test_metrics['mae']:.4f} | R2: {test_metrics['r2']:.4f}", flush=True)

    # Build walk-forward-style results from test set for charts
    wf_results = pd.DataFrame({
        "date": test_dates,
        "actual": y_test.values,
        "predicted": test_preds,
        "residual": y_test.values - test_preds,
    })

    # Step 6: Retrain on ALL data
    print("Step 6: Retraining on all data...", flush=True)
    X_all = fe.transform(df)
    y_all = df["unemployment_rate_next"].reset_index(drop=True)
    all_mask = X_all.notna().all(axis=1)
    X_all_clean = X_all[all_mask].reset_index(drop=True)
    y_all_clean = y_all[all_mask].reset_index(drop=True)

    best_result = next(r for r in results if r["name"] == best_name)
    final_model = retrain_on_full_data(best_result["model"], X_all_clean, y_all_clean)

    # Step 7: Bootstrap predictor
    print("Step 7: Fitting bootstrap predictor...", flush=True)
    bootstrap = BootstrapPredictor(final_model)
    bootstrap.fit(X_all_clean, y_all_clean)

    # Step 8: Feature importances
    print("Step 8: Computing feature importances...", flush=True)
    importances = _extract_feature_importances(final_model, fe.feature_names_)

    # Step 9: Save artifacts
    print("Step 9: Saving artifacts...", flush=True)
    joblib.dump(final_model, os.path.join(output_dir, "model.joblib"))
    joblib.dump(fe, os.path.join(output_dir, "feature_pipeline.joblib"))
    joblib.dump(bootstrap, os.path.join(output_dir, "bootstrap_predictor.joblib"))

    report = {
        "best_model": best_name,
        "best_params": best_result["best_params"],
        "n_features": X_all_clean.shape[1],
        "n_training_samples": len(X_all_clean),
        "training_period": {
            "start": str(df["Date"].min().date()),
            "end": str(df["Date"].max().date()),
        },
        "test_metrics": test_metrics,
        "walk_forward_metrics": test_metrics,  # Same as test metrics
        "model_comparison": [
            {
                "name": r["name"],
                "cv_mae": round(r["cv_mae"], 4),
                "best_params": _serialize_params(r["best_params"]),
            }
            for r in results
        ],
    }
    with open(os.path.join(output_dir, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    wf_results.to_csv(os.path.join(output_dir, "walk_forward_results.csv"), index=False)
    importances.to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)

    # Pre-computed features for Data Explorer (no full text - too large)
    fomc_features = X_all.copy()
    fomc_features["Date"] = df["Date"].values
    fomc_features["unemployment_rate"] = df["unemployment_rate"].values
    fomc_features["unemployment_rate_next"] = df["unemployment_rate_next"].values
    fomc_features["text_preview"] = df["Text"].str[:200].values
    fomc_features.to_csv(os.path.join(output_dir, "fomc_features.csv"), index=False)

    # Recent targets for inference
    last_targets = df[["Date", "unemployment_rate"]].tail(12)
    last_targets.to_csv(os.path.join(output_dir, "last_targets.csv"), index=False)

    print("\nDone! All artifacts saved to:", output_dir, flush=True)
    return report


def _extract_feature_importances(model, feature_names):
    """Extract feature importances from the model."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    elif hasattr(model, "estimators_"):
        all_imp = []
        for _, est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                all_imp.append(est.feature_importances_)
            elif hasattr(est, "coef_"):
                all_imp.append(np.abs(est.coef_))
        imp = np.mean(all_imp, axis=0) if all_imp else np.zeros(len(feature_names))
    else:
        imp = np.zeros(len(feature_names))

    df_imp = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df_imp.sort_values("importance", ascending=False).reset_index(drop=True)


def _serialize_params(params):
    """Make params JSON-serializable."""
    serialized = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            serialized[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serialized[k] = float(v)
        elif isinstance(v, np.ndarray):
            serialized[k] = v.tolist()
        else:
            serialized[k] = v
    return serialized
