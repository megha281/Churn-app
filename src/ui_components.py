"""
ui_components.py
-----------------
Small reusable HTML-rendering helpers that give the app its "SaaS product"
look. Keeping these separate from app.py keeps the page code readable.
"""

import streamlit as st


def load_css(path: str) -> None:
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="fade-in">
            <div class="app-eyebrow">{eyebrow}</div>
            <div class="app-title">{title}</div>
            <div class="app-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None, delta_type: str = "neutral") -> str:
    delta_html = f'<div class="kpi-delta {delta_type}">{delta}</div>' if delta else ""
    return f"""
        <div class="kpi-card fade-in">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
    """


def render_kpi_row(cards: list[dict]) -> None:
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                kpi_card(card["label"], card["value"], card.get("delta"), card.get("delta_type", "neutral")),
                unsafe_allow_html=True,
            )


def content_card_open(title: str, icon: str = "") -> None:
    st.markdown(
        f"""<div class="content-card fade-in"><div class="card-title">{icon} {title}</div>""",
        unsafe_allow_html=True,
    )


def content_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def pill(text: str, kind: str = "blue") -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


def prediction_result_card(prediction: str, probability: float) -> str:
    is_churn = prediction == "Churn"
    css_class = "churn" if is_churn else "no-churn"
    label_color = "#ef4444" if is_churn else "#22c55e"
    icon = "⚠️" if is_churn else "✅"
    headline = "High Churn Risk" if is_churn else "Likely to Stay"
    sub = (
        "This customer shows strong signals of leaving soon."
        if is_churn
        else "This customer shows healthy retention signals."
    )
    return f"""
        <div class="result-card {css_class} fade-in">
            <div class="result-label" style="color:{label_color}">{icon} {headline}</div>
            <div class="result-prob" style="color:{label_color}">{probability * 100:.1f}%</div>
            <div class="result-sub">predicted churn probability</div>
            <div class="result-sub section-gap">{sub}</div>
        </div>
    """
