"""FOMC Unemployment Prediction - Landing Page."""

import json
import streamlit as st
import pandas as pd

from src.visualizations import unemployment_timeline

st.set_page_config(
    page_title="FOMC Unemployment Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f4e79;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 4px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("Use the pages in the sidebar to explore the app.")
    st.divider()
    st.markdown(
        "<small>Built with Streamlit & scikit-learn</small>",
        unsafe_allow_html=True,
    )

# Hero section
st.markdown('<p class="hero-title">FOMC Minutes Unemployment Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    "Predicting next-month U.S. unemployment rate from Federal Reserve meeting minutes "
    "using NLP features, economic indicators, and machine learning."
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# Load training report
try:
    with open("models/training_report.json") as f:
        report = json.load(f)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{report["test_metrics"]["mae"]:.3f}pp</div>'
            f'<div class="metric-label">Test MAE</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{report["test_metrics"]["r2"]:.3f}</div>'
            f'<div class="metric-label">R-squared</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        period = report["training_period"]
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{period["start"][:4]}-{period["end"][:4]}</div>'
            f'<div class="metric-label">Training Period</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{report["n_training_samples"]}</div>'
            f'<div class="metric-label">FOMC Meetings</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Overview chart
    try:
        wf = pd.read_csv("models/walk_forward_results.csv", parse_dates=["date"])
        fig = unemployment_timeline(
            wf["date"], wf["actual"],
            title="U.S. Unemployment Rate (FOMC Meeting Months)",
        )
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.info("Walk-forward results not found. Run train.py to generate model artifacts.")

    # Model info
    st.markdown("### How It Works")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**1. Text Analysis**")
        st.markdown(
            "FOMC meeting minutes are processed to extract sentiment, "
            "economic keywords, document statistics, and latent topics."
        )
    with col_b:
        st.markdown("**2. Feature Engineering**")
        st.markdown(
            f"{report['n_features']} features including TF-IDF components, "
            "lagged unemployment values, and temporal patterns."
        )
    with col_c:
        st.markdown("**3. Prediction**")
        st.markdown(
            f"Best model ({report['best_model']}) selected via time-series "
            "cross-validation with bootstrap confidence intervals."
        )

except FileNotFoundError:
    st.warning(
        "Model artifacts not found. Run `python train.py` to train the model "
        "and generate all required files in the `models/` directory."
    )
