import plotly.express as px
import streamlit as st

from src.config import TARGET
from src.data import normalize_dataset
import pandas as pd

def _render_dataset_source():
    with st.expander("📂 Dataset", expanded=st.session_state["dataset"] is None):
        uploaded = st.file_uploader("Upload online_shoppers.csv", type=["csv"])

        if uploaded is not None and st.session_state.get("uploaded_file_id") != uploaded.file_id:
            try:
                st.session_state["dataset"] = pd.read_csv(uploaded)
                st.session_state["dataset_name"] = uploaded.name
                st.session_state["uploaded_file_id"] = uploaded.file_id
            except Exception as e:
                st.error(f"Không đọc được CSV: {e}")

        name = st.session_state.get("dataset_name")
        if name:
            st.caption(f"Đang dùng: `{name}`")

def render():
    st.header("1. Bài toán và dữ liệu")
    _render_dataset_source()

    st.markdown(
        r"""
### Bài toán

Dựa trên hành vi của một phiên truy cập website:

$$
Y = \text{Revenue} \in \{0, 1\}
$$

- $Y = 1$: mua hàng.
- $Y = 0$: không mua.

Đây là bài toán **binary classification**.
"""
    )

    df = st.session_state["dataset"]

    if df is None:
        st.info("Upload dataset ở mục Dataset phía trên.")
        return

    try:
        preview, _ = normalize_dataset(df)
    except Exception:
        preview = df.copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(preview):,}")
    c2.metric("Features", max(0, preview.shape[1] - 1))
    c3.metric("Missing cells", int(preview.isna().sum().sum()))

    if TARGET in preview.columns:
        try:
            rate = float(preview[TARGET].astype(int).mean())
            c4.metric("Purchase rate", f"{rate * 100:.2f}%")
        except Exception:
            c4.metric("Purchase rate", "N/A")

    st.dataframe(preview.head(20), width="stretch")

    if TARGET in preview.columns:
        target_counts = (
            preview[TARGET]
            .astype(int)
            .value_counts()
            .sort_index()
            .rename_axis("Revenue")
            .reset_index(name="Count")
        )
        target_counts["Revenue"] = target_counts["Revenue"].map(
            {0: "0 - Not purchased", 1: "1 - Purchased"}
        )

        fig = px.bar(
            target_counts,
            x="Revenue",
            y="Count",
            text="Count",
            title="Phân bố target Revenue",
        )
        fig.update_xaxes(type="category")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

        st.warning(
            "Target mất cân bằng → không chỉ nhìn Accuracy. Ưu tiên AP, Recall, F1."
        )