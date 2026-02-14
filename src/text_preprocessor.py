"""Text preprocessing utilities for FOMC minutes."""

import re


def clean_text(text: str) -> str:
    """Clean FOMC minutes text with minimal transformations.

    Preserves meaningful content while normalizing whitespace and
    removing artifacts from web scraping.
    """
    if not isinstance(text, str):
        return ""
    # Normalize unicode quotes and dashes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Normalize whitespace (tabs, newlines, multiple spaces)
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text
