"""
app.py
------
ChurnIQ — Customer Churn Prediction Platform
Entry point for the Streamlit application. Handles page routing via a
sidebar navigation menu and delegates each section to its own module
under `pages_src/`.
"""

import os

import streamlit as st

from src.ui_components import load_css
from pages_src import dashboard, data_explorer, model_insights, prediction

st.set_page_config(
    page_title="ChurnIQ | Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = os.path.join("assets", "style.css")
if os.path.exists(CSS_PATH):
    load_css(CSS_PATH)

NAV_ITEMS = {
    "Dashboard": {"icon": "📊", "module": dashboard},
    "Data": {"icon": "🗂️", "module": data_explorer},
    "Model": {"icon": "🧠", "module": model_insights},
    "Prediction": {"icon": "🎯", "module": prediction},
}


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-block">
                <div class="brand-logo">C</div>
                <div>
                    <div class="brand-name">ChurnIQ</div>
                    <div class="brand-tag">Retention Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            label="Navigate",
            options=list(NAV_ITEMS.keys()),
            format_func=lambda p: f"{NAV_ITEMS[p]['icon']}   {p}",
            label_visibility="collapsed",
        )

        st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="color:#64749a; font-size:0.75rem; line-height:1.5;">
                Built with Python · Scikit-learn · Streamlit<br>
                Logistic Regression churn model<br>
                v1.0.0
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page


def main() -> None:
    page = render_sidebar()
    NAV_ITEMS[page]["module"].render()


if __name__ == "__main__":
    main()
