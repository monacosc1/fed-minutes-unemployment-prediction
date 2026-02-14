"""Data loading and merging for FOMC minutes + unemployment data."""

import pandas as pd

from src.config import SWAPPED_ROWS
from src.text_preprocessor import clean_text


def load_fomc_csv(csv_path: str) -> pd.DataFrame:
    """Load FOMC minutes CSV, fix column swap in early rows."""
    df = pd.read_csv(csv_path)

    # First SWAPPED_ROWS have Date and Text columns swapped
    swapped = df.iloc[:SWAPPED_ROWS].copy()
    swapped["Date"], swapped["Text"] = swapped["Text"].values, swapped["Date"].values
    df.iloc[:SWAPPED_ROWS] = swapped

    # Parse date column (format: YYYYMMDD)
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%Y%m%d")
    df = df.sort_values("Date").reset_index(drop=True)

    # Clean text
    df["Text"] = df["Text"].apply(clean_text)
    df = df[df["Text"].str.len() > 100].reset_index(drop=True)

    # Extract year and month for merging
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    return df


def fetch_unemployment() -> pd.DataFrame:
    """Fetch monthly unemployment rate from FRED CSV endpoint (no API key needed)."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"
    unemp = pd.read_csv(url)
    unemp.columns = ["date", "unemployment_rate"]
    unemp["date"] = pd.to_datetime(unemp["date"])
    unemp["unemployment_rate"] = pd.to_numeric(unemp["unemployment_rate"], errors="coerce")
    unemp = unemp.dropna(subset=["unemployment_rate"])
    unemp["year"] = unemp["date"].dt.year
    unemp["month"] = unemp["date"].dt.month
    return unemp


def build_dataset(csv_path: str) -> pd.DataFrame:
    """Build the full dataset: FOMC minutes merged with unemployment targets.

    Target is next month's unemployment rate (shifted by -1 relative to
    the meeting month).
    """
    fomc = load_fomc_csv(csv_path)
    unemp = fetch_unemployment()

    # Current month unemployment (for lag features later)
    merged = fomc.merge(
        unemp[["year", "month", "unemployment_rate"]],
        on=["year", "month"],
        how="left",
    )

    # Next month unemployment = target
    unemp_next = unemp.copy()
    # Shift: for each row, get next month's value
    unemp_next["unemployment_rate_next"] = unemp_next["unemployment_rate"].shift(-1)
    unemp_next = unemp_next.dropna(subset=["unemployment_rate_next"])

    merged = merged.merge(
        unemp_next[["year", "month", "unemployment_rate_next"]],
        on=["year", "month"],
        how="left",
    )

    merged = merged.dropna(subset=["unemployment_rate", "unemployment_rate_next"])
    merged = merged.sort_values("Date").reset_index(drop=True)

    return merged
