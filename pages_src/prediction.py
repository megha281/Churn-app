"""
pages_src/prediction.py
-------------------------
The "Prediction" page: a styled form for entering a customer's profile,
which runs through the saved model pipeline and renders a result card
with the predicted churn probability.
"""

import time

import streamlit as st

from src.predict import artifacts_exist, predict_single
from src.ui_components import content_card_close, content_card_open, page_header, prediction_result_card


def _input_form() -> dict:
    content_card_open("Customer Profile", "🧾")

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    with c2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
    with c3:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    st.markdown("**Add-on Services**")
    a1, a2, a3 = st.columns(3)

    def addon_options():
        return ["No internet service"] if internet_service == "No" else ["Yes", "No"]

    with a1:
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No phone service"] if phone_service == "No" else ["Yes", "No"]
        )
        online_security = st.selectbox("Online Security", addon_options())
    with a2:
        online_backup = st.selectbox("Online Backup", addon_options())
        device_protection = st.selectbox("Device Protection", addon_options())
    with a3:
        tech_support = st.selectbox("Tech Support", addon_options())
        streaming_tv = st.selectbox("Streaming TV", addon_options())

    streaming_movies = st.selectbox("Streaming Movies", addon_options())

    content_card_close()

    return {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
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
    }


def render() -> None:
    page_header(
        "Predict",
        "Churn Risk Prediction",
        "Enter a customer's profile to estimate their likelihood of churning.",
    )

    if not artifacts_exist():
        st.warning("No trained model found. Visit the **Model** page and click *Retrain Model* first.")
        return

    customer = _input_form()

    predict_col, _ = st.columns([1, 3])
    with predict_col:
        run_prediction = st.button("🎯  Predict Churn Risk", use_container_width=True)

    if run_prediction:
        with st.spinner("Scoring customer against the model..."):
            time.sleep(0.4)  # small delay so the spinner is visible — matches a "real" inference call
            result = predict_single(customer)

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        result_col, detail_col = st.columns([1, 1.3])

        with result_col:
            st.markdown(
                prediction_result_card(result["prediction"], result["churn_probability"]),
                unsafe_allow_html=True,
            )

        with detail_col:
            content_card_open("Risk Breakdown", "📊")
            st.progress(result["churn_probability"], text=f"Churn probability: {result['churn_probability']*100:.1f}%")
            st.progress(result["retain_probability"], text=f"Retention probability: {result['retain_probability']*100:.1f}%")

            if result["prediction"] == "Churn":
                st.error("Recommended action: proactive retention outreach — consider a loyalty offer or contract upgrade incentive.")
            else:
                st.success("Recommended action: no immediate intervention needed — continue standard engagement.")
            content_card_close()
