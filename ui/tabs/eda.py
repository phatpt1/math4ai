import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.data import normalize_dataset

KEY_NUMERIC = ["PageValues", "ExitRates", "BounceRates", "ProductRelated_Duration"]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LABEL_MAP = {0: "0 - Not purchased", 1: "1 - Purchased"}


def _render_problem():
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


def _render_overview(df):
    st.subheader("Tổng quan dữ liệu")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Missing cells", int(df.isna().sum().sum()))
    c4.metric("Purchase rate", f"{df[TARGET].mean() * 100:.2f}%")

    st.dataframe(df.head(20), width="stretch")

    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    skew = df[num_cols].skew()
    zero_pv = (df["PageValues"] == 0).mean() * 100
    st.caption(
        f"Đặc trưng số lệch phải mạnh (skew từ {skew.min():.1f} đến {skew.max():.1f}); "
        f"**{zero_pv:.0f}%** phiên có PageValues = 0."
    )


def _render_target(df):
    st.subheader("Phân bố nhãn Revenue")

    counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
        .rename_axis("Revenue")
        .reset_index(name="Count")
    )
    counts["Revenue"] = counts["Revenue"].map(LABEL_MAP)

    fig = px.bar(counts, x="Revenue", y="Count", text="Count")
    fig.update_xaxes(type="category")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    ratio = counts["Count"].iloc[0] / counts["Count"].iloc[1]
    st.warning(
        f"Target mất cân bằng (~{ratio:.1f} : 1) → không chỉ nhìn Accuracy. Ưu tiên AP, Recall, F1."
    )


def _render_numeric_by_target(df):
    st.subheader("Phân bố đặc trưng số theo Revenue")

    cols = [c for c in KEY_NUMERIC if c in df.columns]
    feature = st.selectbox("Feature", cols, index=0, key="eda_num_feature")

    plot_df = df[[feature, TARGET]].copy()
    plot_df["Revenue"] = plot_df[TARGET].map(LABEL_MAP)
    y_col = f"log1p({feature})"
    plot_df[y_col] = np.log1p(plot_df[feature])

    fig = px.box(plot_df, x="Revenue", y=y_col, color="Revenue", points=False)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Trục y dùng log1p để nhìn rõ phân bố lệch mạnh.")

    means = df.groupby(TARGET)[cols].mean().T
    means.columns = [LABEL_MAP[c] for c in means.columns]
    means["ratio (1 / 0)"] = means.iloc[:, 1] / means.iloc[:, 0].replace(0, np.nan)
    st.dataframe(means.round(3), width="stretch")


def _render_categorical_rate(df):
    st.subheader("Tỷ lệ mua theo đặc trưng phân loại")

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    default = cat_cols.index("Month") if "Month" in cat_cols else 0
    feature = st.selectbox("Feature", cat_cols, index=default, key="eda_cat_feature")

    rate = (
        df.assign(_key=df[feature].astype(str))
        .groupby("_key")[TARGET]
        .agg(purchase_rate="mean", sessions="count")
        .reset_index()
        .rename(columns={"_key": feature})
    )
    rate["purchase_rate"] = rate["purchase_rate"] * 100

    if feature == "Month":
        order = [m for m in MONTH_ORDER if m in set(rate[feature])]
        rate[feature] = pd.Categorical(rate[feature], categories=order, ordered=True)
        rate = rate.sort_values(feature)
    else:
        rate = rate.sort_values("purchase_rate", ascending=False)

    fig = px.bar(
        rate,
        x=feature,
        y="purchase_rate",
        text=rate["purchase_rate"].round(1),
        hover_data=["sessions"],
        labels={"purchase_rate": "Purchase rate (%)"},
    )
    fig.update_xaxes(type="category")
    fig.add_hline(y=df[TARGET].mean() * 100, line_dash="dash", annotation_text="overall rate")
    st.plotly_chart(fig, width="stretch")
    st.caption("Nhóm có ít phiên (xem `sessions` khi hover) thì tỷ lệ mua kém tin cậy.")


def _render_correlation(df):
    st.subheader("Ma trận tương quan (Spearman)")

    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[num_cols + [TARGET]].corr(method="spearman")

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    st.plotly_chart(fig, width="stretch")


def _render_conclusion():
    st.subheader("Kết luận EDA → hướng xử lý")
    st.markdown(
        """
| Quan sát | Hệ quả |
|---|---|
| Nhãn mất cân bằng | `class_weight='balanced'`, `scale_pos_weight`; đánh giá bằng AP, Recall, F1 |
| Đặc trưng số lệch mạnh, nhiều outlier | Dùng model dạng cây → không cần scale / xử lý outlier |
| Có đặc trưng phân loại (Month, VisitorType, ...) | OneHotEncoder (sklearn) / native categorical (LightGBM) |
| Một số cặp đặc trưng tương quan cao | Model dạng cây chịu được đa cộng tuyến, không cần loại bỏ |
| PageValues tách 2 lớp rõ rệt | Kỳ vọng là đặc trưng quan trọng nhất (kiểm chứng ở Feature Importance) |
"""
    )


def render():
    st.header("1. Bài toán và dữ liệu")

    _render_problem()

    raw = st.session_state["dataset"]
    if raw is None:
        st.error("Không tìm thấy `online_shoppers.csv` trong thư mục project.")
        return

    try:
        df, _ = normalize_dataset(raw)
    except ValueError as e:
        st.error(str(e))
        return

    _render_overview(df)
    _render_target(df)
    _render_numeric_by_target(df)
    _render_categorical_rate(df)
    _render_correlation(df)
    _render_conclusion()