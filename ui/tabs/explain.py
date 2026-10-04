import numpy as np
import plotly.express as px
import streamlit as st

from src.inference import get_lgbm_contributions, get_what_if_sensitivity, sigmoid


def render():
    st.header("5. Explain Prediction")

    bundle = st.session_state["bundle"]

    if bundle is None:
        st.info("Train/load model trước.")
        return

    if "last_input" not in st.session_state:
        st.info("Hãy sang Live Prediction và Predict một sample trước.")
        return

    model_name = st.session_state.get("last_model_name", bundle["recommended_model"])
    input_df = st.session_state["last_input"]

    st.markdown(
        f"""
### Đang giải thích prediction của **{model_name}**

Có 2 mức giải thích:

1. **What-if sensitivity** — dùng được cho cả 4 model: thay từng feature về baseline của TRAIN và xem score đổi bao nhiêu.
2. **Native LightGBM contribution** — chỉ khi model là LightGBM: phân rã raw score thành base value + đóng góp feature.
        """
    )

    score, threshold, pred, sensitivity = get_what_if_sensitivity(
        model_name, input_df, bundle
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model score", f"{score*100:.2f}%")
    c2.metric("Threshold", f"{threshold:.3f}")
    c3.metric("Prediction", "MUA" if pred else "KHÔNG MUA")

    st.subheader("A. What-if sensitivity — dễ hiểu, dùng cho mọi model")
    st.markdown(
        """
Với từng feature, app hỏi: **nếu giữ mọi thứ như cũ nhưng thay riêng feature này bằng giá trị baseline của TRAIN thì score thay đổi bao nhiêu?**

- `score_delta > 0`: giá trị hiện tại đang đẩy score Mua **cao hơn baseline**.
- `score_delta < 0`: giá trị hiện tại đang kéo score Mua **thấp hơn baseline**.
- Các delta này **không cộng lại chính xác thành score**; đây là phân tích what-if, không phải SHAP.
        """
    )
    top = sensitivity.head(10).copy().sort_values("score_delta")
    top["direction"] = np.where(
        top["score_delta"] >= 0,
        "Đẩy score cao hơn baseline",
        "Kéo score thấp hơn baseline",
    )
    fig = px.bar(
        top, x="score_delta", y="feature", orientation="h",
        color="direction", title=f"What-if sensitivity — {model_name}"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        sensitivity.head(10)[
            ["feature", "current_value", "baseline_value", "score_delta"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    if model_name != "LightGBM":
        st.info(
            "Native additive contribution trong app hiện chỉ có cho LightGBM. "
            "Với model này, phần A vẫn giải thích prediction bằng phân tích what-if nhất quán."
        )
        return

    st.subheader("B. Native LightGBM contribution — phân rã raw score")
    base, raw_score, contrib = get_lgbm_contributions(input_df, bundle)
    probability_like_score = sigmoid(raw_score)
    st.markdown(
        r"""
LightGBM có thể trả contribution trực tiếp:

\[
F(x)=BaseValue+\sum_j \phi_j
\]

Sau đó:

\[
score=\sigma(F(x))=\frac{1}{1+e^{-F(x)}}
\]

- \(\phi_j>0\): đẩy raw score về phía Mua.
- \(\phi_j<0\): đẩy raw score về phía Không mua.
            """
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("Base raw score", f"{base:.4f}")
    d2.metric("Final raw score", f"{raw_score:.4f}")
    d3.metric("Sigmoid(raw)", f"{probability_like_score*100:.2f}%")

    topc = contrib.head(10).copy().sort_values("contribution")
    topc["direction"] = np.where(
        topc["contribution"] >= 0,
        "Đẩy về Mua",
        "Đẩy về Không mua",
    )
    fig2 = px.bar(
        topc, x="contribution", y="feature", orientation="h",
        color="direction", title="Native LightGBM local contribution"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Đây là contribution lấy trực tiếp từ LightGBM, "
        "không phải rule if/else tự viết."
    )