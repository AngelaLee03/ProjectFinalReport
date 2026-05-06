"""Shared helpers to load cleaned transactions and build train/test splits."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

_DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = _DATA_DIR / "cleaned" / "transactions_cleaned.csv"

def feature_columns(df):
    return (
        ["Transaction Type", "Day of Week", "Month", "Amount_scaled"]
        + [c for c in df.columns if c.startswith("Desc_")]
        + [c for c in df.columns if c.startswith("Acct_")]
    )


def load_xy(csv_path):
    path = csv_path or CSV_PATH
    df = pd.read_csv(path)
    cols = feature_columns(df)
    X = df[cols]
    y = df["Category_Label"]
    return X, y, cols


def load_train_test_split(
    test_size: float = 0.2,
    random_state: int = 42,
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    X, y, feature_cols = load_xy(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols