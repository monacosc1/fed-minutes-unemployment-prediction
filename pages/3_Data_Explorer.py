"""Data Explorer - Browse FOMC data, sentiment trends, correlations."""

import streamlit as st
import pandas as pd
import numpy as np

from src.config import KEYWORD_CATEGORIES
from src.visualizations import sentiment_trends, keyword_trends, correlation_heatmap

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")
st.title("Data Explorer")

# Load pre-computed features
try:
    df = pd.read_csv("models/fomc_features.csv", parse_dates=["Date"])
except FileNotFoundError:
    st.error("Pre-computed features not found. Run `python train.py` first.")
    st.stop()

# Date range filter
st.markdown("### Filter by Date Range")
col1, col2 = st.columns(2)
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
with col1:
    start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End date", max_date, min_value=min_date, max_value=max_date)

mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
filtered = df[mask].copy()

st.markdown(f"Showing **{len(filtered)}** meetings from {start_date} to {end_date}")
st.divider()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "FOMC Minutes", "Sentiment Trends", "Keyword Trends", "Correlations"
])

with tab1:
    st.markdown("#### FOMC Meeting Minutes")
    display_df = filtered[["Date", "unemployment_rate", "unemployment_rate_next"]].copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    display_df = display_df.rename(columns={
        "unemployment_rate": "Unemployment (%)",
        "unemployment_rate_next": "Next Month (%)",
    })

    # Add text preview
    if "text_preview" in filtered.columns:
        display_df["Text Preview"] = filtered["text_preview"].values
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

with tab2:
    st.markdown("#### Sentiment Analysis Trends")
    if "tb_polarity" in filtered.columns and "tb_subjectivity" in filtered.columns:
        fig = sentiment_trends(
            filtered["Date"],
            filtered["tb_polarity"],
            filtered["tb_subjectivity"],
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Sentiment features not available in pre-computed data.")

with tab3:
    st.markdown("#### Economic Keyword Frequency Trends")
    available_keywords = list(KEYWORD_CATEGORIES.keys())
    selected = st.multiselect(
        "Select keyword categories:",
        available_keywords,
        default=["employment_positive", "employment_negative", "inflation", "recession"],
    )
    if selected:
        fig = keyword_trends(filtered["Date"], filtered, selected)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("#### Feature Correlations")
    # Select numeric features (exclude text, date)
    numeric_cols = filtered.select_dtypes(include=[np.number]).columns
    exclude = ["unemployment_rate_next"]  # Don't include target
    feature_cols = [c for c in numeric_cols if c not in exclude]

    n_feats = st.slider("Number of features in heatmap", 5, min(30, len(feature_cols)), 15)
    selected_feats = feature_cols[:n_feats]
    fig = correlation_heatmap(filtered[selected_feats + ["unemployment_rate_next"]])
    st.plotly_chart(fig, use_container_width=True)
