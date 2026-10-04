import plotly.express as px
import streamlit as st

from src.config import TARGET
from src.data import normalize_dataset


def render():
    st.header("1. Bài toán và dữ liệu")

    st.markdown(
        r"""
### Bài toán

Dựa trên hành vi của một phiên truy cập website:

\[
Y = Revenue \in \{0,1\}
\]

- \(Y=1\): mua hàng.
- \(Y=0\): không mua.

Đây là bài toán **binary classification**.
"""
    )

    df = st.session_state["dataset"]

    if df is None:
        st.info("Upload dataset ở sidebar.")
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
            c4.metric("Purchase rate", f"{rate*100:.2f}%")
        except Exception:
            c4.metric("Purchase rate", "N/A")

    st.dataframe(preview.head(20), use_container_width=True)

    if TARGET in preview.columns:
        target_counts = (
            preview[TARGET]
            .astype(str)
            .value_counts()
            .rename_axis("Revenue")
            .reset_index(name="Count")
        )

        fig = px.bar(
            target_counts,
            x="Revenue",
            y="Count",
            text="Count",
            title="Phân bố target Revenue",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "Target mất cân bằng → không chỉ nhìn Accuracy. Ưu tiên AP, Recall, F1."
        )