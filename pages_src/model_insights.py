"""
pages_src/model_insights.py
-----------------------------
The "Model" page: shows evaluation metrics, confusion matrix, ROC curve,
and feature importance (logistic regression coefficients) for the
trained churn model. Includes a "Retrain Model" action.
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.train_model import train_and_evaluate
from src.ui_components import content_card_close, content_card_open, page_header, pill, render_kpi_row

METRICS_PATH = os.path.join("models", "metrics.json")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#93a1bd", family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=30, b=10),
)


def _load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def render() -> None:
    page_header(
        "Evaluation",
        "Model Performance",
        "Logistic Regression churn classifier — evaluation metrics and feature drivers.",
    )

    top_l, top_r = st.columns([3, 1])
    with top_r:
        if st.button("🔄  Retrain Model", use_container_width=True):
            with st.spinner("Training Logistic Regression model..."):
                train_and_evaluate()
            st.success("Model retrained and saved successfully.")
            st.cache_data.clear()

    metrics = _load_metrics()
    if metrics is None:
        st.warning("No trained model found. Click **Retrain Model** or run `python -m src.train_model` first.")
        return

    render_kpi_row(
        [
            {"label": "Accuracy", "value": f"{metrics['accuracy']*100:.2f}%", "delta": "holdout set", "delta_type": "up"},
            {"label": "Precision", "value": f"{metrics['precision']*100:.2f}%"},
            {"label": "Recall", "value": f"{metrics['recall']*100:.2f}%"},
            {"label": "ROC AUC", "value": f"{metrics['roc_auc']:.3f}", "delta": "discrimination", "delta_type": "up"},
        ]
    )

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        content_card_open("Confusion Matrix", "🧮")
        cm = np.array(metrics["confusion_matrix"])
        labels = ["No Churn", "Churn"]
        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale=[[0, "#131d33"], [1, "#3b82f6"]],
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=18, color="white"),
                showscale=False,
            )
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=320, xaxis_title="Predicted", yaxis_title="Actual")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        content_card_close()

    with col2:
        content_card_open("ROC Curve", "📐")
        roc = metrics["roc_curve"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color="#22d3ee", width=3),
                fill="tozeroy", fillcolor="rgba(34,211,238,0.08)", name=f"AUC = {metrics['roc_auc']:.3f}",
            )
        )
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#64749a", width=1, dash="dash"), name="Random"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
        content_card_close()

    content_card_open("Feature Importance (Logistic Regression Coefficients)", "🧠")
    coef_df = pd.DataFrame({"feature": metrics["feature_names"], "coefficient": metrics["coefficients"]})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).head(12).sort_values("coefficient")
    colors = ["#ef4444" if c > 0 else "#3b82f6" for c in coef_df["coefficient"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=coef_df["coefficient"],
                y=coef_df["feature"],
                orientation="h",
                marker=dict(color=colors),
            )
        ]
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_title="Coefficient (impact on churn log-odds)")
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor="#22304d", zeroline=True, zerolinecolor="#3d4d70")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"{pill('Red = increases churn risk', 'red')} &nbsp; {pill('Blue = reduces churn risk', 'blue')}",
        unsafe_allow_html=True,
    )
    content_card_close()

    content_card_open("Model Details", "⚙️")
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(f"**Algorithm**<br>Logistic Regression", unsafe_allow_html=True)
    d2.markdown(f"**Train samples**<br>{metrics['n_train']:,}", unsafe_allow_html=True)
    d3.markdown(f"**Test samples**<br>{metrics['n_test']:,}", unsafe_allow_html=True)
    d4.markdown(f"**Training time**<br>{metrics['train_seconds']}s", unsafe_allow_html=True)
    st.caption(f"Last trained: {metrics['trained_at']}")
    content_card_close()
