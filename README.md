# ChurnIQ — Customer Churn Prediction Platform

A production-styled churn prediction web application built with **Python, Scikit-learn, and Streamlit**, modeled after a real SaaS analytics product. Predicts whether a telecom customer is likely to churn, using a Logistic Regression model trained on the Kaggle *Telco Customer Churn* dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)

---

## ✨ Features

| Area | Description |
|---|---|
| **Dashboard** | KPI cards (total customers, churn rate, avg. tenure, avg. charges, model accuracy) plus churn distribution, contract-type churn, tenure/charge trend, and internet-service churn charts. |
| **Data** | Filterable, searchable view of the raw dataset with distribution charts and CSV export. |
| **Model** | Accuracy / precision / recall / ROC-AUC KPI cards, confusion matrix heatmap, ROC curve, feature-importance chart (logistic regression coefficients), and a one-click retrain action. |
| **Prediction** | A styled multi-column input form covering every customer attribute, producing a color-coded result card with churn probability and a recommended action. |

The UI uses a dark, blue-accented theme with card-based layout, hover states, KPI badges, spinners, and fade-in transitions — built to *look* like a deployed product rather than a stock Streamlit demo.

---

## 🗂️ Project Structure

```
churn-app/
├── app.py                     # Streamlit entry point + sidebar navigation/router
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml            # Dark theme configuration
├── assets/
│   └── style.css              # Custom SaaS-dashboard CSS
├── data/
│   ├── generate_dataset.py    # Synthetic Telco-schema dataset generator
│   └── telco_churn.csv        # Dataset used for training (see note below)
├── models/                    # Generated at training time
│   ├── churn_model.pkl        # Trained Logistic Regression model
│   ├── encoders.pkl           # Label encoders + one-hot column schema
│   ├── scaler.pkl             # StandardScaler fitted on numeric features
│   ├── metrics.json           # Evaluation metrics, confusion matrix, ROC curve, coefficients
│   └── cleaned_reference.csv
├── src/
│   ├── preprocessing.py       # clean_data / encode_features / scale_features / split_data
│   ├── train_model.py         # train_and_evaluate() — trains + evaluates + saves artifacts
│   ├── predict.py             # predict_single() — runs the saved pipeline on new input
│   └── ui_components.py       # KPI cards, page headers, result cards (HTML helpers)
└── pages_src/
    ├── dashboard.py
    ├── data_explorer.py
    ├── model_insights.py
    └── prediction.py
```

Each concern lives in its own module — preprocessing, training, and inference are fully decoupled from the UI, so the same pipeline functions are reused identically at training time and prediction time (no train/serve skew).

---

## 📊 Dataset

This project targets the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), which includes:

- Customer demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`)
- Account info (`tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`)
- Services (`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`)
- Charges (`MonthlyCharges`, `TotalCharges`)
- Target (`Churn`)

> **Note on `data/telco_churn.csv`:** this build environment has no internet access, so `data/generate_dataset.py` generates a **schema-identical synthetic stand-in** (same columns, dtypes, category labels, and realistic churn correlations — month-to-month contracts, fiber optic, electronic check, and short tenure all increase churn risk, matching well-documented patterns in the real dataset).
>
> **To use the real Kaggle data:** download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle, rename it to `telco_churn.csv`, and drop it into `data/`, replacing the synthetic file. No other code changes are required — every column name and category value matches exactly. Then re-run training (see below).

---

## 🧠 Machine Learning Pipeline

1. **Cleaning** (`preprocessing.clean_data`)
   - Coerces `TotalCharges` to numeric (the raw dataset stores it as text with blank entries for brand-new customers) and imputes with `0`.
   - Drops the `customerID` identifier and duplicate rows.
   - Median/mode imputation for any other missing values.
2. **Encoding** (`preprocessing.encode_features`)
   - Binary Yes/No columns → label-encoded to 0/1.
   - Multi-category columns → one-hot encoded.
   - The exact fitted encoders and final column schema are persisted so inference-time input is aligned identically, even if a category is missing from a single form submission.
3. **Scaling** (`preprocessing.scale_features`)
   - `StandardScaler` fitted on numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`).
4. **Model** (`train_model.train_and_evaluate`)
   - `LogisticRegression(max_iter=1000, class_weight="balanced")` — class weighting compensates for the ~27% churn base rate.
   - 80/20 stratified train/test split.
5. **Evaluation**
   - Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC curve points — all saved to `models/metrics.json` and rendered on the **Model** page.
6. **Persistence**
   - Model, encoders, and scaler are pickled to `models/` so the app loads instantly without retraining on every run.

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset (or supply the real Kaggle CSV — see note above)
```bash
python data/generate_dataset.py
```

### 3. Train the model
```bash
python -m src.train_model
```
This prints evaluation metrics to the console and saves all model artifacts to `models/`.

### 4. Launch the app
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (typically `http://localhost:8501`).

> You can also retrain from inside the app itself — the **Model** page has a **🔄 Retrain Model** button.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — application framework and UI
- **scikit-learn** — Logistic Regression, preprocessing, metrics
- **pandas / numpy** — data handling
- **Plotly** — interactive charts (pie, bar, line, heatmap, ROC curve)

---

## 📈 Example Metrics

On the bundled synthetic dataset, the model typically achieves:

| Metric | Score |
|---|---|
| Accuracy | ~72% |
| Precision | ~50% |
| Recall | ~75% |
| ROC AUC | ~0.81 |

`class_weight="balanced"` is used deliberately: for a churn-prevention product, **recall on the churn class matters more than raw accuracy** — missing an at-risk customer is costlier than a false alarm. Exact numbers will shift slightly if you swap in the real Kaggle dataset.

---

## 📄 License

This project is provided as a portfolio/demo template. The Telco Customer Churn dataset is distributed by Kaggle under its own license terms — see the dataset page for details.
