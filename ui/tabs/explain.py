import numpy as np
import plotly.express as px
import streamlit as st

from src.inference import get_what_if_sensitivity

PUSH_UP = "Đẩy score lên"
PULL_DOWN = "Kéo score xuống"


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return f"{round(float(value), 4):g}"
    return str(value)


def render():
    st.header("5. Explain Prediction")

    bundle = st.session_state["bundle"]
    if bundle is None:
        st.info("Train model ở tab 2️⃣ trước.")
        return

    if "last_input" not in st.session_state:
        st.info("Hãy chọn một kịch bản và bấm Predict ở tab 4️⃣ trước.")
        return

    model_name = st.session_state.get("last_model_name", bundle["recommended_model"])
    input_df = st.session_state["last_input"]

    score, threshold, pred, sensitivity = get_what_if_sensitivity(model_name, input_df, bundle)

    st.caption(
        f"Đang giải thích: **{model_name}** · score **{score * 100:.1f}%** · "
        f"threshold {threshold * 100:.1f}% · **{'MUA' if pred else 'KHÔNG MUA'}**"
    )

    st.markdown(
        "Với từng feature: **giữ nguyên mọi thứ khác, chỉ đưa feature đó về mức của khách điển hình** "
        "(median/mode của TRAIN) → score thay đổi bao nhiêu điểm %?"
    )

    top = sensitivity[sensitivity["abs_delta"] > 1e-4].head(10).copy()
    if top.empty:
        st.info("Khách này đang trùng với khách điển hình nên không có feature nào tạo khác biệt.")
        return

    top["delta_pct"] = top["score_delta"] * 100
    top["direction"] = np.where(top["delta_pct"] >= 0, PUSH_UP, PULL_DOWN)
    top = top.sort_values("delta_pct")

    fig = px.bar(
        top,
        x="delta_pct",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={PUSH_UP: "#198754", PULL_DOWN: "#dc3545"},
        text=top["delta_pct"].map(lambda v: f"{v:+.1f}"),
        labels={"delta_pct": "Thay đổi score (điểm %)", "feature": "", "direction": ""},
        title=f"What-if sensitivity — {model_name}",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Các delta không cộng lại chính xác thành score, vì các feature tương tác với nhau "
        "(phân tích độ nhạy, không phải SHAP)."
    )

    with st.expander("Bảng chi tiết"):
        table = sensitivity.head(10)[["feature", "current_value", "baseline_value", "score_delta"]].copy()
        table["current_value"] = table["current_value"].map(_fmt)
        table["baseline_value"] = table["baseline_value"].map(_fmt)
        table["score_delta"] = (table["score_delta"] * 100).round(2)
        table = table.rename(columns={"score_delta": "score_delta (điểm %)"})
        st.dataframe(table, width="stretch", hide_index=True)