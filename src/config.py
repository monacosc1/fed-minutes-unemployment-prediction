"""Configuration constants for the FOMC Unemployment Prediction project."""

# Number of rows in the CSV where Date and Text columns are swapped
SWAPPED_ROWS = 24

# Economic keyword dictionaries (terms → category)
KEYWORD_CATEGORIES = {
    "employment_positive": [
        "job gains", "hiring", "payroll growth", "labor demand",
        "employment increased", "job creation", "labor market tightened",
        "low unemployment", "strong labor", "robust employment",
    ],
    "employment_negative": [
        "job losses", "layoffs", "unemployment rose", "unemployment increased",
        "labor market weakened", "slack", "underemployment",
        "jobless", "furlough", "downsizing",
    ],
    "inflation": [
        "inflation", "price pressures", "consumer prices", "cpi",
        "pce", "core inflation", "price stability", "deflation",
        "disinflation", "inflationary",
    ],
    "growth": [
        "gdp", "economic growth", "expansion", "output",
        "productivity", "economic activity", "real gdp",
        "gross domestic", "aggregate demand", "recovery",
    ],
    "recession": [
        "recession", "contraction", "downturn", "slowdown",
        "decline", "deterioration", "weakness", "stagnation",
        "economic weakness", "sharp decline",
    ],
    "monetary_policy": [
        "federal funds rate", "interest rate", "monetary policy",
        "accommodation", "tightening", "easing", "quantitative",
        "balance sheet", "open market", "policy rate",
    ],
    "housing": [
        "housing", "mortgage", "residential", "home sales",
        "home prices", "housing starts", "real estate",
        "foreclosure", "subprime", "housing market",
    ],
    "financial_conditions": [
        "financial conditions", "credit conditions", "lending",
        "bank lending", "credit growth", "financial stress",
        "liquidity", "spreads", "financial markets", "equity prices",
    ],
    "uncertainty": [
        "uncertainty", "risk", "downside risk", "volatile",
        "geopolitical", "trade tensions", "pandemic",
        "disruption", "headwinds", "unpredictable",
    ],
    "confidence": [
        "confidence", "optimism", "sentiment", "consumer confidence",
        "business confidence", "outlook improved", "positive outlook",
        "expectations", "favorable", "upbeat",
    ],
}

# Feature engineering parameters
TFIDF_MAX_FEATURES = 200
TFIDF_MIN_DF = 5
SVD_N_COMPONENTS = 20
NMF_N_TOPICS = 8
LAG_PERIODS = [1, 2, 3, 4, 6]
ROLLING_WINDOWS = [2, 3, 4, 6]

# Model training parameters
TEST_SIZE_FRACTION = 0.20
CV_N_SPLITS = 5
BOOTSTRAP_N_ITERATIONS = 50
CONFIDENCE_LEVEL = 0.90

# Visualization colors
COLOR_PRIMARY = "#1f4e79"
COLOR_SECONDARY = "#2e75b6"
COLOR_ACCENT = "#ed7d31"
COLOR_BG = "#f8f9fa"
COLOR_TEXT = "#333333"
COLOR_SUCCESS = "#27ae60"
COLOR_DANGER = "#e74c3c"

FEATURE_CATEGORY_COLORS = {
    "Document Stats": COLOR_PRIMARY,
    "Sentiment": "#8e44ad",
    "Keywords": COLOR_ACCENT,
    "TF-IDF/SVD": "#2ecc71",
    "Topics": "#e67e22",
    "Lagged Targets": COLOR_SECONDARY,
    "Temporal": "#95a5a6",
}
