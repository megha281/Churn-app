"""
pages_src/data_explorer.py
---------------------------
The "Data" page: lets the user browse the raw dataset, filter it, and
inspect feature distributions before modeling.
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui_components import content_card_close, content_card_open, page_header, render_kpi_row

DATA_PATH = os.path.join("data", "telco_churn.csv")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#93a1bd", family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=30, b=10),
)


@st.cache_data(show_spinner=False)
def _load_raw_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def render() -> None:
    page_header(
        "Explore",
        "Dataset Explorer",
        "Inspect the raw Telco Customer Churn dataset that powers the model.",
    )

    if not os.path.exists(DATA_PATH):
        st.warning("No dataset found. Run `python data/generate_dataset.py` first.")
        return

    with st.spinner("Loading dataset..."):
        df = _load_raw_data()

    render_kpi_row(
        [
            {"label": "Rows", "value": f"{len(df):,}"},
            {"label": "Columns", "value": f"{df.shape[1]}"},
            {"label": "Missing Values", "value": f"{df.isna().sum().sum()}"},
            {"label": "Duplicate Rows", "value": f"{df.duplicated().sum()}"},
        ]
    )

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    content_card_open("Filters", "🔎")
    f1, f2, f3 = st.columns(3)
    with f1:
        contract_filter = st.multiselect("Contract type", options=sorted(df["Contract"].unique()), default=[])
    with f2:
        internet_filter = st.multiselect("Internet service", options=sorted(df["InternetService"].unique()), default=[])
    with f3:
        churn_filter = st.multiselect("Churn status", options=sorted(df["Churn"].unique()), default=[])
    content_card_close()

    filtered = df.copy()
    if contract_filter:
        filtered = filtered[filtered["Contract"].isin(contract_filter)]
    if internet_filter:
        filtered = filtered[filtered["InternetService"].isin(internet_filter)]
    if churn_filter:
        filtered = filtered[filtered["Churn"].isin(churn_filter)]

    content_card_open(f"Customer Records ({len(filtered):,} shown)", "📋")
    st.dataframe(filtered, use_container_width=True, height=360)
    content_card_close()

    col1, col2 = st.columns(2)
    with col1:
        content_card_open("Monthly Charges Distribution", "💰")
        fig = px.histogram(filtered, x="MonthlyCharges", color="Churn", nbins=30, color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"}, opacity=0.8)
        fig.update_layout(**PLOTLY_LAYOUT, height=300, bargap=0.05)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#22304d")
        st.plotly_chart(fig, use_container_width=True)
        content_card_close()

    with col2:
        content_card_open("Tenure Distribution", "⏱️")
        fig = px.histogram(filtered, x="tenure", color="Churn", nbins=30, color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"}, opacity=0.8)
        fig.update_layout(**PLOTLY_LAYOUT, height=300, bargap=0.05)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#22304d")
        st.plotly_chart(fig, use_container_width=True)
        content_card_close()

    content_card_open("Download Filtered Data", "⬇️")
    st.download_button(
        "Export as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_churn_data.csv",
        mime="text/csv",
    )
    content_card_close()
