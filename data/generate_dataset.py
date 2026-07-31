"""
generate_dataset.py
--------------------
Generates a synthetic dataset that exactly mirrors the schema of the
Kaggle "Telco Customer Churn" dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv).

WHY THIS EXISTS
This environment has no internet access, so the real Kaggle CSV can't be
fetched at build time. This script produces a realistic, schema-identical
stand-in so the whole pipeline (preprocessing -> training -> evaluation ->
Streamlit app) runs end-to-end out of the box.

TO USE THE REAL DATA INSTEAD
Download "WA_Fn-UseC_-Telco-Customer-Churn.csv" from Kaggle and drop it in
this `data/` folder under the same filename: `telco_churn.csv`
(rename the Kaggle file). Nothing else in the project needs to change --
every column name, dtype, and category label below matches the original.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 7043  # same row count as the original Kaggle dataset


def _weighted_choice(rng, options, weights, size):
    return rng.choice(options, size=size, p=np.array(weights) / np.sum(weights))


def generate_telco_dataset(n_customers: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Builds a synthetic customer table with realistic churn correlations."""
    rng = np.random.default_rng(seed)

    gender = _weighted_choice(rng, ["Male", "Female"], [0.5, 0.5], n_customers)
    senior_citizen = _weighted_choice(rng, [0, 1], [0.84, 0.16], n_customers)
    partner = _weighted_choice(rng, ["Yes", "No"], [0.48, 0.52], n_customers)
    dependents = _weighted_choice(rng, ["Yes", "No"], [0.3, 0.7], n_customers)

    tenure = rng.integers(0, 73, n_customers)

    phone_service = _weighted_choice(rng, ["Yes", "No"], [0.9, 0.1], n_customers)
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        _weighted_choice(rng, ["Yes", "No"], [0.42, 0.58], n_customers),
    )

    internet_service = _weighted_choice(
        rng, ["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22], n_customers
    )

    def internet_dependent_feature(p_yes):
        vals = np.empty(n_customers, dtype=object)
        has_internet = internet_service != "No"
        vals[~has_internet] = "No internet service"
        yes_no = _weighted_choice(rng, ["Yes", "No"], [p_yes, 1 - p_yes], has_internet.sum())
        vals[has_internet] = yes_no
        return vals

    online_security = internet_dependent_feature(0.29)
    online_backup = internet_dependent_feature(0.34)
    device_protection = internet_dependent_feature(0.34)
    tech_support = internet_dependent_feature(0.29)
    streaming_tv = internet_dependent_feature(0.38)
    streaming_movies = internet_dependent_feature(0.39)

    contract = _weighted_choice(
        rng, ["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24], n_customers
    )
    paperless_billing = _weighted_choice(rng, ["Yes", "No"], [0.59, 0.41], n_customers)
    payment_method = _weighted_choice(
        rng,
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        [0.34, 0.23, 0.22, 0.21],
        n_customers,
    )

    base_charge = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
    addon_cols = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    addon_count = sum((col == "Yes").astype(int) for col in addon_cols)
    monthly_charges = (
        base_charge
        + addon_count * rng.uniform(4, 7, n_customers)
        + (phone_service == "Yes") * rng.uniform(5, 10, n_customers)
        + rng.normal(0, 3, n_customers)
    ).round(2)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

    total_charges = (monthly_charges * np.maximum(tenure, 1) * rng.uniform(0.92, 1.0, n_customers)).round(2)
    total_charges = np.where(tenure == 0, 0.0, total_charges)

    # --- churn probability model: mirrors well-known real-world drivers ---
    churn_logit = (
        -2.15
        + 1.9 * (contract == "Month-to-month")
        + 0.55 * (internet_service == "Fiber optic")
        + 0.5 * (payment_method == "Electronic check")
        - 0.035 * tenure
        + 0.012 * monthly_charges
        - 0.55 * (tech_support == "Yes")
        - 0.35 * (online_security == "Yes")
        + 0.3 * (paperless_billing == "Yes")
        - 0.25 * (partner == "Yes")
        - 0.2 * (dependents == "Yes")
        + 0.2 * senior_citizen
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = np.where(rng.uniform(0, 1, n_customers) < churn_prob, "Yes", "No")

    customer_id = [f"{rng.integers(1000,9999)}-{''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 5))}" for _ in range(n_customers)]

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    # Introduce a handful of blank TotalCharges strings, exactly like the
    # real Kaggle dataset does for brand-new (tenure == 0) customers.
    blank_mask = df["tenure"] == 0
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[blank_mask, "TotalCharges"] = " "

    return df


if __name__ == "__main__":
    import os

    dataset = generate_telco_dataset()
    # Always write next to this script (data/telco_churn.csv), regardless of
    # the current working directory the script is invoked from.
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telco_churn.csv")
    dataset.to_csv(out_path, index=False)
    print(f"Saved {len(dataset)} rows to {out_path}")
    print(dataset["Churn"].value_counts(normalize=True))
