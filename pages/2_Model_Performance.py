"""Model Performance - Metrics, charts, and validation results."""

import json
import streamlit as st
import pandas as pd

from src.visualizations import (
    actual_vs_predicted,
    actual_vs_predicted_scatter,
    residuals_timeline,
    residuals_histogram,
    feature_importance_chart,
)

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("Model Performance")

# Load artifacts
try:
    with open("models/training_report.json") as f:
        report = json.load(f)
    wf = pd.read_csv("models/walk_forward_results.csv", parse_dates=["date"])
    imp = pd.read_csv("models/feature_importances.csv")
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python train.py` first.")
    st.stop()

# Summary metrics
st.markdown("### Summary Metrics")
col1, col2, col3, col4 = st.columns(4)
wf_metrics = report["walk_forward_metrics"]
col1.metric("Walk-Forward MAE", f"{wf_metrics['mae']:.3f}pp")
col2.metric("Walk-Forward RMSE", f"{wf_metrics['rmse']:.3f}pp")
col3.metric("Walk-Forward R²", f"{wf_metrics['r2']:.3f}")
col4.metric("Walk-Forward MAPE", f"{wf_metrics['mape']:.1f}%")

st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Walk-Forward Results", "Residual Analysis", "Feature Importance", "Model Comparison"
])

with tab1:
    st.markdown("#### Actual vs Predicted Unemployment Rate")
    st.markdown(
        "Walk-forward validation: model is trained on all data up to each point, "
        "then predicts the next meeting's unemployment outcome."
    )
    fig1 = actual_vs_predicted(wf)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("#### Scatter Plot")
    fig2 = actual_vs_predicted_scatter(wf)
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown("#### Residuals Over Time")
    fig3 = residuals_timeline(wf)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Residual Distribution")
    col_a, col_b = st.columns(2)
    with col_a:
        fig4 = residuals_histogram(wf)
        st.plotly_chart(fig4, use_container_width=True)
    with col_b:
        st.markdown("**Residual Statistics**")
        st.markdown(f"- Mean: {wf['residual'].mean():.4f}")
        st.markdown(f"- Std Dev: {wf['residual'].std():.4f}")
        st.markdown(f"- Min: {wf['residual'].min():.4f}")
        st.markdown(f"- Max: {wf['residual'].max():.4f}")
        st.markdown(f"- Median: {wf['residual'].median():.4f}")

with tab3:
    st.markdown("#### Top Feature Importances")
    try:
        import joblib
        pipeline = joblib.load("models/feature_pipeline.joblib")
        categories = pipeline.get_feature_categories()
    except Exception:
        categories = {}

    n_features = st.slider("Number of features to show", 5, 30, 15)
    fig5 = feature_importance_chart(imp, categories, top_n=n_features)
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("Full Feature Importance Table"):
        st.dataframe(imp, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("#### Model Comparison (Cross-Validation)")
    st.markdown("All models evaluated with TimeSeriesSplit (5-fold) cross-validation.")

    comparison = report.get("model_comparison", [])
    if comparison:
        comp_df = pd.DataFrame(comparison)
        comp_df = comp_df.rename(columns={
            "name": "Model",
            "cv_mae": "CV MAE",
        })
        # Highlight best
        st.dataframe(
            comp_df[["Model", "CV MAE"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(f"**Selected Model**: {report['best_model']}")
    st.markdown(f"**Best Parameters**: `{report['best_params']}`")

    st.markdown("#### Test Set Performance")
    test_metrics = report["test_metrics"]
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Test MAE", f"{test_metrics['mae']:.3f}pp")
    tc2.metric("Test RMSE", f"{test_metrics['rmse']:.3f}pp")
    tc3.metric("Test R²", f"{test_metrics['r2']:.3f}")
    tc4.metric("Test MAPE", f"{test_metrics['mape']:.1f}%")
