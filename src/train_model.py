"""
train_model.py
---------------
Trains a Logistic Regression churn classifier on the Telco dataset,
evaluates it, and persists the model + encoders + scaler + metadata
to `models/` so the Streamlit app can load them without retraining.

Run directly:
    python -m src.train_model
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocessing import (
    TARGET_COL,
    clean_data,
    encode_features,
    load_data,
    scale_features,
    split_data,
)

DATA_PATH = os.path.join("data", "telco_churn.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
CLEANED_DATA_PATH = os.path.join(MODEL_DIR, "cleaned_reference.csv")


def train_and_evaluate(data_path: str = DATA_PATH, random_state: int = 42) -> dict:
    """Full train + evaluate run. Returns a metrics dict and saves artifacts."""
    raw_df = load_data(data_path)
    cleaned_df = clean_data(raw_df)

    encoded_df, encoders = encode_features(cleaned_df, fit=True)
    scaled_df, scaler = scale_features(encoded_df, fit=True)

    X_train, X_test, y_train, y_test = split_data(scaled_df, test_size=0.2, random_state=random_state)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)

    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = round(time.time() - start, 3)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": cm.tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_seconds": train_seconds,
        "feature_names": X_train.columns.tolist(),
        "coefficients": model.coef_[0].tolist(),
        "trained_at": pd.Timestamp.now().isoformat(),
    }

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save a cleaned (but not encoded) reference copy for the Dashboard/Data
    # pages to visualize raw-ish distributions without re-running cleaning.
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)

    return metrics


if __name__ == "__main__":
    m = train_and_evaluate()
    print(f"Accuracy:  {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall:    {m['recall']:.4f}")
    print(f"F1 score:  {m['f1_score']:.4f}")
    print(f"ROC AUC:   {m['roc_auc']:.4f}")
    print(f"Confusion matrix: {m['confusion_matrix']}")
    print(f"Artifacts saved to '{MODEL_DIR}/'")
