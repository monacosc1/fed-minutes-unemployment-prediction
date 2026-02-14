"""CLI entry point: run the full training pipeline."""

import sys
from src.pipeline import run_pipeline


def main():
    csv_path = "data/scraped_data_all_years_true.csv"
    output_dir = "models"
    print("=" * 60)
    print("FOMC Unemployment Prediction - Training Pipeline")
    print("=" * 60)
    report = run_pipeline(csv_path, output_dir)
    print("\n" + "=" * 60)
    print("Training Report Summary")
    print("=" * 60)
    print(f"  Best model:       {report['best_model']}")
    print(f"  Test MAE:         {report['test_metrics']['mae']:.4f}")
    print(f"  Test RMSE:        {report['test_metrics']['rmse']:.4f}")
    print(f"  Test R-squared:   {report['test_metrics']['r2']:.4f}")
    print(f"  Walk-forward MAE: {report['walk_forward_metrics']['mae']:.4f}")
    print(f"  # Features:       {report['n_features']}")
    print(f"  # Samples:        {report['n_training_samples']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
