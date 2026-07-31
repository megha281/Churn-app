"""
pages_src/dashboard.py
-----------------------
The landing "Dashboard" page: top-line KPI cards plus churn distribution
and usage-trend charts, similar to a startup analytics product's home view.
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui_components import content_card_close, content_card_open, page_header, render_kpi_row

DATA_PATH = os.path.join("data", "telco_churn.csv")
METRICS_PATH = os.path.join("models", "metrics.json")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#93a1bd", family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


@st.cache_data(show_spinner=False)
def _load_raw_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def _load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def render() -> None:
    page_header(
        "Overview",
        "Retention Dashboard",
        "A live snapshot of customer churn risk across your subscriber base.",
    )

    if not os.path.exists(DATA_PATH):
        st.warning("No dataset found. Run `python data/generate_dataset.py` first.")
        return

    with st.spinner("Loading customer data..."):
        df = _load_raw_data()
    metrics = _load_metrics()

    total_customers = len(df)
    churned = (df["Churn"] == "Yes").sum()
    churn_rate = churned / total_customers * 100
    avg_tenure = df["tenure"].mean()
    avg_monthly = df["MonthlyCharges"].mean()

    kpis = [
        {"label": "Total Customers", "value": f"{total_customers:,}", "delta": "Active base", "delta_type": "neutral"},
        {"label": "Churn Rate", "value": f"{churn_rate:.1f}%", "delta": f"{churned:,} churned", "delta_type": "down"},
        {"label": "Avg. Tenure", "value": f"{avg_tenure:.1f} mo", "delta": "customer lifetime", "delta_type": "neutral"},
        {"label": "Avg. Monthly Charge", "value": f"${avg_monthly:.2f}", "delta": "per customer", "delta_type": "up"},
    ]
    if metrics:
        kpis.append(
            {
                "label": "Model Accuracy",
                "value": f"{metrics['accuracy']*100:.1f}%",
                "delta": "on holdout set",
                "delta_type": "up",
            }
        )
    render_kpi_row(kpis[:4])
    if metrics:
        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        render_kpi_row([kpis[4]] if len(kpis) > 4 else [])

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2])

    with col1:
        content_card_open("Churn Distribution", "🥧")
        counts = df["Churn"].value_counts()
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Retained", "Churned"],
                    values=[counts.get("No", 0), counts.get("Yes", 0)],
                    hole=0.62,
                    marker=dict(colors=["#3b82f6", "#ef4444"]),
                    textinfo="percent",
                    textfont=dict(color="white", size=13),
                )
            ]
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        content_card_close()

    with col2:
        content_card_open("Churn Rate by Contract Type", "📈")
        contract_churn = (
            df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reindex(
                ["Month-to-month", "One year", "Two year"]
            )
        )
        fig = go.Figure(
            data=[
                go.Bar(
                    x=contract_churn.index,
                    y=contract_churn.values,
                    marker=dict(color="#3b82f6"),
                    text=[f"{v:.1f}%" for v in contract_churn.values],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=300, yaxis_title="Churn Rate (%)")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#22304d")
        st.plotly_chart(fig, use_container_width=True)
        content_card_close()

    col3, col4 = st.columns(2)

    with col3:
        content_card_open("Usage Trend — Charges vs. Tenure", "💳")
        tenure_bins = pd.cut(df["tenure"], bins=[0, 12, 24, 36, 48, 60, 72], include_lowest=True)
        trend = df.groupby(tenure_bins, observed=True)["MonthlyCharges"].mean()
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[str(i) for i in trend.index],
                    y=trend.values,
                    mode="lines+markers",
                    line=dict(color="#22d3ee", width=3),
                    marker=dict(size=7, color="#3b82f6"),
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.08)",
                )
            ]
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Tenure (months)", yaxis_title="Avg. Monthly Charge ($)")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#22304d")
        st.plotly_chart(fig, use_container_width=True)
        content_card_close()

    with col4:
        content_card_open("Churn Rate by Internet Service", "🌐")
        internet_churn = df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=internet_churn.values,
                    y=internet_churn.index,
                    orientation="h",
                    marker=dict(color=["#3b82f6", "#22d3ee", "#64748b"]),
                    text=[f"{v:.1f}%" for v in internet_churn.values],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Churn Rate (%)")
        fig.update_xaxes(showgrid=True, gridcolor="#22304d")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
        content_card_close()
