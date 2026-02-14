"""Feature engineering for FOMC minutes text."""

import re
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from textblob import TextBlob

from src.config import (
    KEYWORD_CATEGORIES,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    SVD_N_COMPONENTS,
    NMF_N_TOPICS,
    LAG_PERIODS,
    ROLLING_WINDOWS,
)


class FOMCFeatureEngineer:
    """Extracts features from FOMC minutes with fit/transform pattern.

    Stateful components (fitted on training data):
    - TF-IDF vectorizer
    - SVD reducer
    - NMF topic model
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.svd = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=42)
        self.nmf = NMF(n_components=NMF_N_TOPICS, random_state=42, max_iter=500)
        self._is_fitted = False
        self.feature_names_ = []

    def fit(self, df: pd.DataFrame) -> "FOMCFeatureEngineer":
        """Fit stateful components on training data."""
        tfidf_matrix = self.tfidf.fit_transform(df["Text"])
        self.svd.fit(tfidf_matrix)
        # NMF needs non-negative input, TF-IDF is already non-negative
        self.nmf.fit(tfidf_matrix)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform FOMC data into feature matrix."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")

        features = pd.DataFrame(index=df.index)

        # 1. Document statistics
        doc_feats = self._document_stats(df["Text"])
        features = pd.concat([features, doc_feats], axis=1)

        # 2. Sentiment features
        sent_feats = self._sentiment_features(df["Text"])
        features = pd.concat([features, sent_feats], axis=1)

        # 3. Economic keyword features
        kw_feats = self._keyword_features(df["Text"])
        features = pd.concat([features, kw_feats], axis=1)

        # 4. TF-IDF + SVD
        tfidf_matrix = self.tfidf.transform(df["Text"])
        svd_components = self.svd.transform(tfidf_matrix)
        svd_cols = [f"svd_{i}" for i in range(SVD_N_COMPONENTS)]
        features[svd_cols] = pd.DataFrame(svd_components, index=df.index)

        # 5. NMF topics
        nmf_components = self.nmf.transform(tfidf_matrix)
        nmf_cols = [f"topic_{i}" for i in range(NMF_N_TOPICS)]
        features[nmf_cols] = pd.DataFrame(nmf_components, index=df.index)

        # 6. Lagged target features
        if "unemployment_rate" in df.columns:
            lag_feats = self._lagged_features(df["unemployment_rate"])
            features = pd.concat([features, lag_feats], axis=1)

        # 7. Temporal features (cyclical month encoding)
        if "month" in df.columns:
            features["month_sin"] = np.sin(2 * math.pi * df["month"] / 12)
            features["month_cos"] = np.cos(2 * math.pi * df["month"] / 12)

        self.feature_names_ = features.columns.tolist()
        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def transform_single(
        self, text: str, recent_unemployment: list[float], month: int
    ) -> pd.DataFrame:
        """Transform a single FOMC text for prediction.

        Args:
            text: Cleaned FOMC minutes text.
            recent_unemployment: Recent unemployment rates, most recent first.
                                 Needs at least max(LAG_PERIODS) + max(ROLLING_WINDOWS) values.
            month: Month of the meeting (1-12).
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform_single()")

        features = {}

        # Document stats
        doc = self._document_stats(pd.Series([text]))
        for col in doc.columns:
            features[col] = doc[col].iloc[0]

        # Sentiment
        sent = self._sentiment_features(pd.Series([text]))
        for col in sent.columns:
            features[col] = sent[col].iloc[0]

        # Keywords
        kw = self._keyword_features(pd.Series([text]))
        for col in kw.columns:
            features[col] = kw[col].iloc[0]

        # TF-IDF + SVD
        tfidf_vec = self.tfidf.transform([text])
        svd_vec = self.svd.transform(tfidf_vec)[0]
        for i in range(SVD_N_COMPONENTS):
            features[f"svd_{i}"] = svd_vec[i]

        # NMF topics
        nmf_vec = self.nmf.transform(tfidf_vec)[0]
        for i in range(NMF_N_TOPICS):
            features[f"topic_{i}"] = nmf_vec[i]

        # Lagged features from recent_unemployment
        features.update(
            self._compute_lag_features_from_list(recent_unemployment)
        )

        # Temporal
        features["month_sin"] = math.sin(2 * math.pi * month / 12)
        features["month_cos"] = math.cos(2 * math.pi * month / 12)

        result = pd.DataFrame([features])
        # Ensure column order matches training
        for col in self.feature_names_:
            if col not in result.columns:
                result[col] = 0.0
        return result[self.feature_names_]

    def _document_stats(self, texts: pd.Series) -> pd.DataFrame:
        """Extract document-level statistics."""
        stats = pd.DataFrame(index=texts.index)
        words = texts.str.split()
        stats["num_words"] = words.str.len()
        stats["num_chars"] = texts.str.len()
        stats["num_sentences"] = texts.apply(
            lambda t: len(re.split(r"[.!?]+", t)) if isinstance(t, str) else 0
        )
        stats["avg_sentence_length"] = (
            stats["num_words"] / stats["num_sentences"].replace(0, 1)
        )
        stats["num_unique_words"] = words.apply(
            lambda w: len(set(w)) if isinstance(w, list) else 0
        )
        stats["vocabulary_richness"] = (
            stats["num_unique_words"] / stats["num_words"].replace(0, 1)
        )
        stats["avg_word_length"] = words.apply(
            lambda w: np.mean([len(x) for x in w]) if isinstance(w, list) and len(w) > 0 else 0
        )
        stats["num_punctuations"] = texts.str.count(r"[,;:!?\-\(\)]")
        stats["num_uppercase_words"] = words.apply(
            lambda w: sum(1 for x in w if x.isupper() and len(x) > 1)
            if isinstance(w, list) else 0
        )
        return stats

    def _sentiment_features(self, texts: pd.Series) -> pd.DataFrame:
        """Extract sentiment scores using TextBlob (fast, no VADER)."""
        results = []
        for text in texts:
            # Truncate to first 5000 chars for speed (sentiment is stable beyond this)
            snippet = str(text)[:5000]
            blob = TextBlob(snippet)
            results.append({
                "tb_polarity": blob.sentiment.polarity,
                "tb_subjectivity": blob.sentiment.subjectivity,
            })
        return pd.DataFrame(results, index=texts.index)

    def _keyword_features(self, texts: pd.Series) -> pd.DataFrame:
        """Count economic keyword occurrences per category."""
        feats = pd.DataFrame(index=texts.index)
        texts_lower = texts.str.lower()
        word_counts = texts.str.split().str.len().replace(0, 1)

        for category, keywords in KEYWORD_CATEGORIES.items():
            count = texts_lower.apply(
                lambda t: sum(t.count(kw) for kw in keywords)
            )
            feats[f"kw_{category}_count"] = count
            feats[f"kw_{category}_rate"] = count / word_counts * 1000

        return feats

    def _lagged_features(self, unemployment: pd.Series) -> pd.DataFrame:
        """Create lagged target features from the unemployment series."""
        feats = pd.DataFrame(index=unemployment.index)

        for lag in LAG_PERIODS:
            feats[f"unemp_lag_{lag}"] = unemployment.shift(lag)

        for window in ROLLING_WINDOWS:
            feats[f"unemp_roll_mean_{window}"] = unemployment.rolling(window).mean()
            feats[f"unemp_roll_std_{window}"] = unemployment.rolling(window).std()

        feats["unemp_diff_1"] = unemployment.diff(1)
        feats["unemp_diff_2"] = unemployment.diff(2)
        feats["unemp_pct_change"] = unemployment.pct_change()

        return feats

    def _compute_lag_features_from_list(
        self, recent_values: list[float]
    ) -> dict[str, float]:
        """Compute lag features from a list of recent unemployment values.

        recent_values[0] = most recent, recent_values[1] = 1 month ago, etc.
        """
        features = {}
        series = pd.Series(recent_values)

        for lag in LAG_PERIODS:
            features[f"unemp_lag_{lag}"] = (
                recent_values[lag - 1] if lag - 1 < len(recent_values) else np.nan
            )

        for window in ROLLING_WINDOWS:
            if window <= len(recent_values):
                vals = recent_values[:window]
                features[f"unemp_roll_mean_{window}"] = np.mean(vals)
                features[f"unemp_roll_std_{window}"] = np.std(vals, ddof=1) if window > 1 else 0.0
            else:
                features[f"unemp_roll_mean_{window}"] = np.nan
                features[f"unemp_roll_std_{window}"] = np.nan

        features["unemp_diff_1"] = (
            recent_values[0] - recent_values[1] if len(recent_values) > 1 else np.nan
        )
        features["unemp_diff_2"] = (
            recent_values[0] - recent_values[2] if len(recent_values) > 2 else np.nan
        )
        features["unemp_pct_change"] = (
            (recent_values[0] - recent_values[1]) / recent_values[1]
            if len(recent_values) > 1 and recent_values[1] != 0
            else np.nan
        )

        return features

    def get_feature_categories(self) -> dict[str, str]:
        """Return mapping of feature name → category for visualization."""
        categories = {}
        for name in self.feature_names_:
            if name.startswith(("num_", "avg_", "vocabulary_")):
                categories[name] = "Document Stats"
            elif name.startswith(("tb_", "vader_")):
                categories[name] = "Sentiment"
            elif name.startswith("kw_"):
                categories[name] = "Keywords"
            elif name.startswith("svd_"):
                categories[name] = "TF-IDF/SVD"
            elif name.startswith("topic_"):
                categories[name] = "Topics"
            elif name.startswith("unemp_"):
                categories[name] = "Lagged Targets"
            elif name.startswith("month_"):
                categories[name] = "Temporal"
            else:
                categories[name] = "Other"
        return categories
