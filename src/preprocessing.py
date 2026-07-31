"""
preprocessing.py
-----------------
All data-cleaning, encoding, and scaling logic for the Telco churn dataset.
Kept separate from training/UI code so it can be unit-tested and reused
identically at both training time and inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COL = "Churn"
ID_COL = "customerID"

# Columns that are naturally binary Yes/No (or already numeric 0/1)
BINARY_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Columns with 3+ categories -> one-hot encoded
MULTI_CATEGORY_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "gender",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_data(path: str) -> pd.DataFrame:
    """Loads the raw CSV from disk."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles the well-known data quality issues in this dataset:
    - `TotalCharges` is read as a string and contains blank entries for
      brand-new customers (tenure == 0); these are coerced to NaN then
      imputed with 0 (no charges have accrued yet).
    - Drops the customerID identifier column (no predictive value).
    - Drops exact duplicate rows, if any.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    df = df.drop_duplicates()

    # Any other stray missing values -> impute numeric with median, else mode
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df.reset_index(drop=True)


def encode_features(df: pd.DataFrame, fit: bool = True, encoders: dict | None = None):
    """
    Encodes categorical features:
    - Binary Yes/No columns -> 0/1 via LabelEncoder
    - Multi-category columns -> one-hot via pd.get_dummies
    - Target column (`Churn`) -> 0/1

    Returns (encoded_df, encoders_dict) so the exact same encoders/dummy
    columns can be reapplied at inference time.
    """
    df = df.copy()
    encoders = encoders or {}

    target = None
    if TARGET_COL in df.columns:
        if fit:
            le = LabelEncoder()
            target = le.fit_transform(df[TARGET_COL])
            encoders["target_encoder"] = le
        else:
            le = encoders["target_encoder"]
            target = le.transform(df[TARGET_COL])
        df = df.drop(columns=[TARGET_COL])

    for col in BINARY_COLS:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[f"le_{col}"] = le
            else:
                le = encoders[f"le_{col}"]
                df[col] = le.transform(df[col])

    present_multi_cols = [c for c in MULTI_CATEGORY_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present_multi_cols)

    if fit:
        encoders["feature_columns"] = df.columns.tolist()
    else:
        # Align inference-time columns with training-time columns exactly
        train_cols = encoders["feature_columns"]
        for col in train_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[train_cols]

    if target is not None:
        df[TARGET_COL] = target

    return df, encoders


def scale_features(df: pd.DataFrame, fit: bool = True, scaler: StandardScaler | None = None):
    """Standard-scales the numeric columns; returns (df, scaler)."""
    df = df.copy()
    cols_present = [c for c in NUMERIC_COLS if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
    else:
        df[cols_present] = scaler.transform(df[cols_present])

    return df, scaler


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Splits into train/test feature matrices and target vectors."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def full_preprocessing_pipeline(raw_df: pd.DataFrame):
    """
    Convenience wrapper used by the training script: runs cleaning,
    encoding, and scaling in one call and returns everything needed
    to reproduce the pipeline at inference time.
    """
    cleaned = clean_data(raw_df)
    encoded, encoders = encode_features(cleaned, fit=True)
    scaled, scaler = scale_features(encoded, fit=True)
    return scaled, encoders, scaler, cleaned
