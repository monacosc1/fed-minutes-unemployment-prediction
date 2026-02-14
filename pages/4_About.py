"""About - Methodology, data sources, and limitations."""

import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
st.title("About This Project")

st.markdown("""
### Overview

This application predicts the next month's U.S. unemployment rate using text from Federal
Reserve Open Market Committee (FOMC) meeting minutes. The model analyzes the language,
sentiment, and economic themes in these minutes to forecast labor market conditions.

---

### Data Sources

| Source | Description | Coverage |
|--------|------------|----------|
| **FOMC Meeting Minutes** | Full text of FOMC meeting minutes scraped from the Federal Reserve website | 1993 - Present (~240 meetings) |
| **FRED (Federal Reserve Economic Data)** | Monthly U.S. unemployment rate (UNRATE series) | 1993 - Present |

---

### Methodology

#### Feature Engineering (~80 features)

1. **Document Statistics** (9 features): Word count, sentence count, vocabulary richness,
   average word/sentence length, punctuation count, etc.

2. **Sentiment Analysis** (2 features): TextBlob polarity and subjectivity.

3. **Economic Keywords** (20 features): Counts and per-1000-word rates for 10 curated
   keyword categories: employment (positive/negative), inflation, growth, recession,
   monetary policy, housing, financial conditions, uncertainty, and confidence.

4. **TF-IDF + SVD** (20 features): TF-IDF vectorization (max 200 terms) reduced to 20
   latent components via Truncated SVD.

5. **Topic Model** (8 features): Non-negative Matrix Factorization (NMF) extracts 8
   latent topics from the TF-IDF matrix.

6. **Lagged Targets** (~15 features): Previous unemployment values (lags 1-4, 6),
   rolling means/standard deviations (windows 2-6), differences, and percent changes.

7. **Temporal** (2 features): Cyclical month encoding (sin/cos).

#### Model Selection

Multiple models are evaluated using **TimeSeriesSplit cross-validation** (5 folds) to
prevent data leakage:

- Linear Regression (baseline)
- Ridge, Lasso, Elastic Net (regularized linear models)
- Random Forest, Gradient Boosting, XGBoost (tree ensembles)
- Voting Regressor (ensemble of top 3)

The best model is selected based on cross-validation MAE, then retrained on all data.

#### Validation

- **Walk-Forward Validation**: Expanding-window evaluation where the model is retrained at
  each step using only past data, then predicts the next observation.
- **Bootstrap Confidence Intervals**: 200 bootstrap samples produce prediction intervals
  (90% confidence level).

---

### Limitations

- **Small dataset**: ~240 meetings is modest for machine learning. Strong regularization
  and dimensionality reduction mitigate overfitting.
- **Temporal structure**: Unemployment is highly autocorrelated. Lag features dominate
  predictions; the text features provide incremental improvement.
- **Meeting frequency**: FOMC meets ~8 times per year, so predictions are sparse relative
  to monthly unemployment releases.
- **Text quality**: Historical minutes (1990s) have different formatting and length than
  modern ones.
- **Not financial advice**: This is an educational/research project. Do not use predictions
  for investment decisions.

---

### Technical Stack

- **Language**: Python 3.10+
- **ML Framework**: scikit-learn, XGBoost
- **NLP**: TextBlob, VADER Sentiment
- **Web App**: Streamlit
- **Visualization**: Plotly
- **Data**: FRED API, Federal Reserve website
""")
