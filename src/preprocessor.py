"""
Text preprocessing and feature engineering for customer complaint classification.
Includes:
- Anonymization cleanup (stripping CFPB XXXX redaction artifacts).
- Noise removal (special characters, URLs, irregular whitespace).
- Urgency / Escalation keyword scoring.
- Train / Test split helper with label encoding.
"""

import re
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


URGENCY_KEYWORDS = [
    r"\blawyer\b", r"\battorney\b", r"\bsue\b", r"\bsuing\b", r"\blawsuit\b",
    r"\blegal action\b", r"\bcourt\b", r"\bfraud\b", r"\bscam\b", r"\bpolice\b",
    r"\bstolen\b", r"\bunauthorized\b", r"\bidentity theft\b", r"\bemergency\b",
    r"\bforeclosure\b", r"\beviction\b", r"\bbankrupt\b", r"\bbankruptcy\b"
]

URGENCY_REGEX = re.compile("|".join(URGENCY_KEYWORDS), re.IGNORECASE)


class TextPreprocessor:
    def __init__(self):
        # Compiled regexes for fast cleaning
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._redaction_pattern = re.compile(r"\bX{2,}\b|\bX{1,2}/\b|\b/X{1,2}\b", re.IGNORECASE)
        self._special_chars = re.compile(r"[^a-zA-Z0-9\s.,!?$'\"]")
        self._multi_spaces = re.compile(r"\s+")

    def clean_text(self, text: str) -> str:
        """
        Cleans raw complaint text:
        - Removes URLs
        - Cleans XXXX masking tokens common in CFPB / redacted datasets
        - Strips extraneous symbols and normalizes whitespace
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = self._url_pattern.sub(" ", text)

        # Replace CFPB-style redaction tokens (e.g. "XXXX", "XX/XX/XXXX")
        text = self._redaction_pattern.sub(" ", text)

        # Remove excessive non-standard characters
        text = self._special_chars.sub(" ", text)

        # Normalize whitespace and convert to lowercase
        text = self._multi_spaces.sub(" ", text).strip().lower()

        return text

    def clean_series(self, texts: pd.Series) -> pd.Series:
        """Vectorized or mapped text cleaning over a pandas Series."""
        return texts.apply(self.clean_text)

    def detect_urgency(self, text: str) -> Dict[str, Any]:
        """
        Analyzes complaint text for urgent / high-risk signals (fraud, legal action, foreclosure).
        Returns risk level and matched trigger keywords.
        """
        matches = URGENCY_REGEX.findall(text)
        unique_matches = list(set(m.lower() for m in matches))
        match_count = len(unique_matches)

        if match_count >= 2:
            priority = "CRITICAL / HIGH"
        elif match_count == 1:
            priority = "MEDIUM"
        else:
            priority = "NORMAL"

        return {
            "priority": priority,
            "triggers": unique_matches,
            "urgent_count": match_count
        }


def prepare_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Cleans complaint texts, encodes categorical labels, and produces
    stratified train/test splits.
    """
    preprocessor = TextPreprocessor()
    print("Preprocessing text narratives...")
    df = df.copy()
    df["cleaned_text"] = preprocessor.clean_series(df["text"])

    # Remove any samples that became empty after stripping redactions
    df = df[df["cleaned_text"].str.len() > 10]

    # Encode categories into numerical integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df["category"])

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    print(f"Dataset split complete: {len(X_train):,} train samples, {len(X_test):,} test samples.")
    print(f"Categories ({len(label_encoder.classes_)}): {label_encoder.classes_.tolist()}")

    return X_train, X_test, y_train, y_test, label_encoder
