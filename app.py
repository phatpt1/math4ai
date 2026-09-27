import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import streamlit as st

from ui.sidebar import init_session_state, render_sidebar
from ui.tabs import (
    code_math,
    comparison,
    eda,
    explain,
    github,
    prediction,
    training,
)

st.set_page_config(
    page_title="Online Shopper ML Lab",
    page_icon="🛒",
    layout="wide",
)

init_session_state()

st.title("🛒 Online Shopper Purchase Prediction — ML Lab")

st.caption(
    "Một app.py duy nhất: "
    "Load model đã lưu → Predict ngay; "
    "chỉ Re-train khi muốn cập nhật model."
)

if st.session_state.get("bundle") is not None:
    st.success(
        "✅ Model đã được load tự động từ repo/GitHub. "
        "Bạn có thể Predict ngay mà không cần train lại."
    )

render_sidebar()

tabs = st.tabs(
    [
        "1️⃣ Problem & EDA",
        "2️⃣ Training",
        "3️⃣ Model Comparison",
        "4️⃣ Live Prediction",
        "5️⃣ Explain Prediction",
        "6️⃣ Code ↔ Math",
        "7️⃣ GitHub",
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

with tabs[5]:
    code_math.render()

with tabs[6]:
    github.render()