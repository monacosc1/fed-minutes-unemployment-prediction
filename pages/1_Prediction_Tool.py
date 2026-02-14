"""Prediction Tool - Make unemployment predictions from FOMC text."""

import json
from datetime import date

import joblib
import pandas as pd
import numpy as np
import streamlit as st

from src.text_preprocessor import clean_text
from src.visualizations import prediction_context_chart

st.set_page_config(page_title="Prediction Tool", page_icon="🔮", layout="wide")
st.title("Prediction Tool")
st.markdown("Paste FOMC meeting minutes text to predict next month's unemployment rate.")


@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model, feature pipeline, and bootstrap predictor."""
    model = joblib.load("models/model.joblib")
    pipeline = joblib.load("models/feature_pipeline.joblib")
    bootstrap = joblib.load("models/bootstrap_predictor.joblib")
    return model, pipeline, bootstrap


def fetch_recent_unemployment(n_months=12):
    """Fetch recent unemployment rates from FRED CSV endpoint."""
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
        unemp = pd.read_csv(url)
        unemp.columns = ["date", "value"]
        unemp["value"] = pd.to_numeric(unemp["value"], errors="coerce")
        unemp = unemp.dropna(subset=["value"])
        unemp = unemp.tail(n_months).iloc[::-1]  # Most recent first
        return [(row["date"], row["value"]) for _, row in unemp.iterrows()]
    except Exception:
        return None


try:
    model, pipeline, bootstrap = load_model_artifacts()
    report = json.load(open("models/training_report.json"))
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python train.py` first.")
    st.stop()

# Layout
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown("### FOMC Minutes Text")
    fomc_text = st.text_area(
        "Paste the full FOMC meeting minutes here:",
        height=300,
        placeholder="A meeting of the Federal Open Market Committee was held...",
    )
    meeting_date = st.date_input(
        "Meeting date:",
        value=date.today(),
        help="Date of the FOMC meeting. Used for temporal features and lag alignment.",
    )

with right_col:
    st.markdown("### Recent Unemployment Rates")

    # Try live FRED data, fall back to saved
    recent_data = fetch_recent_unemployment()
    if recent_data:
        recent_df = pd.DataFrame(recent_data, columns=["Date", "Rate (%)"])
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
        recent_values = [r[1] for r in recent_data]
    else:
        st.info("Could not fetch live data. Using saved values.")
        try:
            saved = pd.read_csv("models/last_targets.csv")
            saved = saved.sort_values("Date", ascending=False)
            st.dataframe(saved, use_container_width=True, hide_index=True)
            recent_values = saved["unemployment_rate"].tolist()
        except FileNotFoundError:
            st.warning("No unemployment data available.")
            recent_values = []

    st.info(
        f"**Model**: {report['best_model']}  \n"
        f"**Test MAE**: {report['test_metrics']['mae']:.3f}pp  \n"
        f"**Features**: {report['n_features']}"
    )

# Prediction
st.divider()

if st.button("Predict Unemployment Rate", type="primary", use_container_width=True):
    if not fomc_text or len(fomc_text.strip()) < 100:
        st.warning("Please paste FOMC minutes text (at least 100 characters).")
    elif len(recent_values) < 6:
        st.warning("Need at least 6 recent unemployment values for lag features.")
    else:
        with st.spinner("Generating prediction..."):
            cleaned = clean_text(fomc_text)
            month = meeting_date.month

            features = pipeline.transform_single(cleaned, recent_values, month)
            point_pred = model.predict(features)[0]
            median_pred, ci_lower, ci_upper, _ = bootstrap.predict(features)

            pred = float(median_pred[0])
            lower = float(ci_lower[0])
            upper = float(ci_upper[0])
            prev_rate = recent_values[0] if recent_values else None

        # Results
        st.markdown("### Prediction Results")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            delta = f"{pred - prev_rate:+.2f}pp" if prev_rate else None
            st.metric(
                "Predicted Unemployment Rate",
                f"{pred:.2f}%",
                delta=delta,
                delta_color="inverse",
            )

        with res_col2:
            st.metric("90% Confidence Interval", f"[{lower:.2f}%, {upper:.2f}%]")

        with res_col3:
            if prev_rate:
                st.metric("Previous Month", f"{prev_rate:.1f}%")

        # Historical context chart
        if recent_data and len(recent_data) > 2:
            hist_dates = [pd.Timestamp(r[0]) for r in reversed(recent_data)]
            hist_values = [r[1] for r in reversed(recent_data)]
            pred_date = pd.Timestamp(meeting_date) + pd.DateOffset(months=1)

            fig = prediction_context_chart(
                hist_dates, hist_values, pred_date, pred, lower, upper
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feature contributions (top features)
        try:
            imp_df = pd.read_csv("models/feature_importances.csv")
            top_feats = imp_df.head(10)
            feature_cats = pipeline.get_feature_categories()

            st.markdown("### Top Feature Contributions")
            for _, row in top_feats.iterrows():
                feat_name = row["feature"]
                feat_val = features[feat_name].iloc[0] if feat_name in features.columns else 0
                cat = feature_cats.get(feat_name, "Other")
                st.markdown(f"- **{feat_name}** ({cat}): {feat_val:.4f}")
        except FileNotFoundError:
            pass
