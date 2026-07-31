"""
predict.py
----------
Loads the trained model + encoders + scaler and runs inference on a
single new customer record (as would come from the Streamlit input form).
"""

from __future__ import annotations

import os
import pickle

import pandas as pd

from src.preprocessing import encode_features, scale_features

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def artifacts_exist() -> bool:
    return all(os.path.exists(p) for p in [MODEL_PATH, ENCODERS_PATH, SCALER_PATH])


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, encoders, scaler


def predict_single(customer: dict, model=None, encoders=None, scaler=None) -> dict:
    """
    Runs the full inference pipeline on a single customer dict (matching
    the raw column schema, minus customerID/Churn) and returns the
    predicted label plus churn probability.
    """
    if model is None or encoders is None or scaler is None:
        model, encoders, scaler = load_artifacts()

    df = pd.DataFrame([customer])

    # TotalCharges must be numeric here (form supplies a number already),
    # but coerce defensively in case of blank/odd input.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    encoded_df, _ = encode_features(df, fit=False, encoders=encoders)
    scaled_df, _ = scale_features(encoded_df, fit=False, scaler=scaler)

    proba = model.predict_proba(scaled_df)[0]
    churn_probability = float(proba[1])
    prediction = int(churn_probability >= 0.5)

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": churn_probability,
        "retain_probability": float(proba[0]),
    }
