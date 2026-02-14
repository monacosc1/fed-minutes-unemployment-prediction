"""Shared Plotly chart builders for the Streamlit app."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.config import (
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_ACCENT,
    COLOR_SUCCESS,
    COLOR_DANGER,
    FEATURE_CATEGORY_COLORS,
)

LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, Arial, sans-serif", color="#333333"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    hovermode="x unified",
)


def unemployment_timeline(dates, values, title="U.S. Unemployment Rate"):
    """Line chart of unemployment rate over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines",
        line=dict(color=COLOR_PRIMARY, width=2),
        name="Unemployment Rate",
        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=title,
        yaxis_title="Unemployment Rate (%)",
        xaxis_title="",
    )
    fig.update_yaxes(gridcolor="#e0e0e0", zeroline=False)
    fig.update_xaxes(gridcolor="#e0e0e0")
    return fig


def actual_vs_predicted(wf_df):
    """Walk-forward actual vs predicted timeline."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wf_df["date"], y=wf_df["actual"],
        mode="lines",
        line=dict(color=COLOR_PRIMARY, width=2),
        name="Actual",
    ))
    fig.add_trace(go.Scatter(
        x=wf_df["date"], y=wf_df["predicted"],
        mode="lines",
        line=dict(color=COLOR_ACCENT, width=2, dash="dash"),
        name="Predicted",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Walk-Forward Validation: Actual vs Predicted",
        yaxis_title="Unemployment Rate (%)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    fig.update_yaxes(gridcolor="#e0e0e0", zeroline=False)
    fig.update_xaxes(gridcolor="#e0e0e0")
    return fig


def actual_vs_predicted_scatter(wf_df):
    """Scatter plot of actual vs predicted with perfect-prediction line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wf_df["actual"], y=wf_df["predicted"],
        mode="markers",
        marker=dict(color=COLOR_SECONDARY, size=6, opacity=0.7),
        name="Predictions",
    ))
    min_val = min(wf_df["actual"].min(), wf_df["predicted"].min()) - 0.5
    max_val = max(wf_df["actual"].max(), wf_df["predicted"].max()) + 0.5
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color="#999", dash="dot"),
        name="Perfect Prediction",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Predicted vs Actual",
        xaxis_title="Actual (%)",
        yaxis_title="Predicted (%)",
    )
    fig.update_yaxes(gridcolor="#e0e0e0", zeroline=False)
    fig.update_xaxes(gridcolor="#e0e0e0")
    return fig


def residuals_timeline(wf_df):
    """Residuals over time."""
    fig = go.Figure()
    colors = [COLOR_SUCCESS if r >= 0 else COLOR_DANGER for r in wf_df["residual"]]
    fig.add_trace(go.Bar(
        x=wf_df["date"], y=wf_df["residual"],
        marker_color=colors,
        name="Residual",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#999")
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Prediction Residuals Over Time",
        yaxis_title="Residual (Actual - Predicted)",
    )
    fig.update_yaxes(gridcolor="#e0e0e0")
    fig.update_xaxes(gridcolor="#e0e0e0")
    return fig


def residuals_histogram(wf_df):
    """Histogram of residuals."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=wf_df["residual"],
        nbinsx=25,
        marker_color=COLOR_SECONDARY,
        opacity=0.8,
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="#999")
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Residual Distribution",
        xaxis_title="Residual",
        yaxis_title="Count",
    )
    fig.update_yaxes(gridcolor="#e0e0e0", zeroline=False)
    fig.update_xaxes(gridcolor="#e0e0e0")
    return fig


def feature_importance_chart(importances_df, feature_categories, top_n=15):
    """Horizontal bar chart of top feature importances, colored by category."""
    top = importances_df.head(top_n).copy()
    top = top.iloc[::-1]  # Reverse for horizontal bars (top at top)
    top["category"] = top["feature"].map(feature_categories).fillna("Other")
    top["color"] = top["category"].map(FEATURE_CATEGORY_COLORS).fillna("#999999")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker_color=top["color"],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        height=max(400, top_n * 30),
    )
    fig.update_yaxes(gridcolor="#e0e0e0")
    fig.update_xaxes(gridcolor="#e0e0e0")
    return fig


def prediction_context_chart(historical_dates, historical_values, pred_date, pred_value, ci_lower, ci_upper):
    """Historical unemployment with prediction point and CI."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_dates, y=historical_values,
        mode="lines",
        line=dict(color=COLOR_PRIMARY, width=2),
        name="Historical",
    ))
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=[pred_date], y=[ci_upper],
        mode="markers",
        marker=dict(size=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[pred_date, pred_date], y=[ci_lower, ci_upper],
        mode="lines",
        line=dict(color=COLOR_ACCENT, width=3),
        name="90% CI",
    ))
    # Prediction point
    fig.add_trace(go.Scatter(
        x=[pred_date], y=[pred_value],
        mode="markers",
        marker=dict(color=COLOR_ACCENT, size=12, symbol="diamond",
                    line=dict(width=2, color="white")),
        name="Prediction",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Prediction in Historical Context",
        yaxis_title="Unemployment Rate (%)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    fig.update_yaxes(gridcolor="#e0e0e0", zeroline=False)
    fig.update_xaxes(gridcolor="#e0e0e0")
    return fig


def sentiment_trends(dates, polarity, subjectivity):
    """Dual-axis sentiment chart over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=polarity,
        mode="lines",
        line=dict(color=COLOR_PRIMARY, width=2),
        name="Polarity",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=subjectivity,
        mode="lines",
        line=dict(color=COLOR_ACCENT, width=2),
        name="Subjectivity",
        yaxis="y2",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Sentiment Trends in FOMC Minutes",
        yaxis=dict(title="Polarity", gridcolor="#e0e0e0"),
        yaxis2=dict(title="Subjectivity", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )
    fig.update_xaxes(gridcolor="#e0e0e0")
    return fig


def keyword_trends(dates, keyword_data, selected_keywords):
    """Line chart of keyword frequencies over time."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, kw in enumerate(selected_keywords):
        col = f"kw_{kw}_rate"
        if col in keyword_data.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=keyword_data[col],
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=2),
                name=kw.replace("_", " ").title(),
            ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Keyword Frequency Trends (per 1,000 words)",
        yaxis_title="Occurrences per 1,000 words",
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
    )
    fig.update_yaxes(gridcolor="#e0e0e0", zeroline=False)
    fig.update_xaxes(gridcolor="#e0e0e0")
    return fig


def correlation_heatmap(features_df, max_features=20):
    """Feature correlation heatmap."""
    # Select most important numeric features
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns[:max_features]
    corr = features_df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title="Feature Correlations",
        height=600,
    )
    return fig
