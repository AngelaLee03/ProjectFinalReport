"""Shared helpers to load cleaned transactions and build train/test splits."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

_DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = _DATA_DIR / "cleaned" / "transactions_cleaned.csv"

def add_shared_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # weekend indicator
    df["Is_Weekend"] = df["Day of Week"].isin([5, 6]).astype(int)

    # amount x transaction type
    if "Amount_scaled" in df.columns and "Transaction Type" in df.columns:
        df["Amount_x_Transaction_Type"] = (
            df["Amount_scaled"] * df["Transaction Type"]
        )

    # amount x account type
    acct_cols = [c for c in df.columns if c.startswith("Acct_")]
    for col in acct_cols:
        df[f"Amount_x_{col}"] = df["Amount_scaled"] * df[col]

    return df

def feature_columns(df):
    return (
        ["Transaction Type", "Day of Week", "Month", "Amount_scaled", "Is_Weekend"]
        + [c for c in df.columns if c.startswith("Acct_")]
        + [c for c in df.columns if c.startswith("Amount_x_")]
    )


def load_xy(csv_path):
    path = csv_path or CSV_PATH
    df = pd.read_csv(path)
    df = add_shared_features(df)
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