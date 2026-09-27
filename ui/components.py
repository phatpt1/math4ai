import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.inference import get_lgbm_contributions, sigmoid


def render_purchase_demo(score, threshold, pred, model_name, input_df, bundle):
    """
    Visual demo of the prediction.
    Marketing suggestions are app-level demo rules, not model reasoning.
    Local contributions are shown only for LightGBM (native pred_contrib).
    """
    st.markdown("### 🎯 Demo trực quan khả năng mua hàng")

    score_pct = float(score * 100.0)
    threshold_pct = float(threshold * 100.0)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score_pct,
            number={"suffix": "%", "valueformat": ".1f"},
            delta={
                "reference": threshold_pct,
                "valueformat": ".1f",
                "suffix": " điểm",
            },
            title={
                "text": (
                    f"Điểm xu hướng mua — {model_name}<br>"
                    f"<span style='font-size:0.8em'>"
                    f"Ngưỡng quyết định: {threshold_pct:.1f}%"
                    f"</span>"
                )
            },
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.32},
                "steps": [
                    {"range": [0, threshold_pct], "color": "#f8d7da"},
                    {"range": [threshold_pct, min(100, threshold_pct + 20)], "color": "#fff3cd"},
                    {"range": [min(100, threshold_pct + 20), 100], "color": "#d1e7dd"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.8,
                    "value": threshold_pct,
                },
            },
        )
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=80, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if pred == 1:
        st.success(
            f"### ✅ Dự đoán: CÓ KHẢ NĂNG MUA\n"
            f"Model score = **{score_pct:.2f}%** ≥ threshold = **{threshold_pct:.2f}%**"
        )
    else:
        st.error(
            f"### ❌ Dự đoán: KHÔNG MUA\n"
            f"Model score = **{score_pct:.2f}%** < threshold = **{threshold_pct:.2f}%**"
        )

    st.markdown("#### 💼 Gợi ý hành động demo")
    high_cut = min(0.90, threshold + 0.20)
    medium_cut = threshold

    if score >= high_cut:
        st.success(
            "🟢 **Nhóm ưu tiên cao** — khách đã có tín hiệu mua mạnh. "
            "Có thể ưu tiên trải nghiệm checkout, tránh khuyến mãi quá mức."
        )
    elif score >= medium_cut:
        st.warning(
            "🟠 **Nhóm cân nhắc** — model đã xếp vào lớp Mua nhưng chưa cách xa threshold. "
            "Có thể thử ưu đãi nhẹ hoặc nhắc hoàn tất đơn."
        )
    else:
        st.info(
            "🔵 **Nhóm ưu tiên thấp** — chưa vượt threshold. "
            "Có thể dùng nội dung nuôi dưỡng thay vì chi ngân sách remarketing mạnh."
        )

    st.caption(
        "Các gợi ý marketing ở trên là **rule demo của ứng dụng**, "
        "không phải lời giải thích nội bộ của mô hình."
    )

    if model_name == "LightGBM" and "LightGBM" in bundle["models"]:
        st.markdown("#### 🧠 5 yếu tố tác động mạnh nhất cho chính khách này")
        try:
            base, raw_score, contrib = get_lgbm_contributions(input_df, bundle)
            top5 = contrib.head(5).copy()
            top5["direction"] = np.where(
                top5["contribution"] >= 0,
                "Đẩy về Mua",
                "Đẩy về Không mua",
            )
            top5 = top5.sort_values("contribution")

            fig2 = px.bar(
                top5,
                x="contribution",
                y="feature",
                orientation="h",
                color="direction",
                text="contribution",
                title="Local contribution — lấy trực tiếp từ LightGBM",
            )
            fig2.update_traces(texttemplate="%{text:.3f}")
            st.plotly_chart(fig2, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Base raw score", f"{base:.3f}")
            c2.metric("Final raw score", f"{raw_score:.3f}")
            c3.metric("Sigmoid(raw)", f"{sigmoid(raw_score)*100:.1f}%")
        except Exception as e:
            st.caption(f"Không render được local contribution: {e}")
    else:
        st.caption(
            "Prediction hiện tại không dùng LightGBM. Sang tab 5 để xem "
            "What-if sensitivity cho đúng model đang chọn."
        )

    with st.expander("🔎 Xem dữ liệu khách hàng vừa nhập"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)