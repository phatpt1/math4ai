import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import streamlit as st

from ui.state import init_session_state
from ui.tabs import comparison, eda, explain, prediction, training

st.set_page_config(
    page_title="Online Shopper ML Lab",
    page_icon="🛒",
    layout="wide",
)

init_session_state()

st.title("🛒 Online Shopper Purchase Prediction — ML Lab")

bundle = st.session_state["bundle"]
if bundle is not None:
    trained_at = bundle.get("trained_at_utc", "")[:19].replace("T", " ")
    st.caption(f"✅ Model sẵn sàng · trained at {trained_at} UTC")
else:
    st.warning("⚠️ Chưa có model. Vào tab **2️⃣ Training** để train.")

tabs = st.tabs(
    [
        "1️⃣ Problem & EDA",
        "2️⃣ Training",
        "3️⃣ Model Comparison",
        "4️⃣ Live Prediction",
        "5️⃣ Explain Prediction",
    ]
)

with tabs[0]:
    eda.render()

with tabs[1]:
    training.render()

with tabs[2]:
    comparison.render()

with tabs[3]:
    prediction.render()

with tabs[4]:
    explain.render()