import plotly.graph_objects as go
import streamlit as st


def render_purchase_demo(score, threshold, pred, model_name):
    """Gauge + final verdict for a single prediction."""
    score_pct = float(score * 100.0)
    threshold_pct = float(threshold * 100.0)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score_pct,
            number={"suffix": "%", "valueformat": ".1f"},
            delta={
                "reference": threshold_pct,
                "valueformat": "+.1f",
                "suffix": " điểm % so với ngưỡng",
                "font": {"size": 18},
            },
            title={
                "text": (
                    f"Điểm xu hướng mua — {model_name}<br>"
                    f"<span style='font-size:0.8em'>Ngưỡng quyết định: {threshold_pct:.1f}%</span>"
                )
            },
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.32, "color": "#198754" if pred == 1 else "#dc3545"},
                "steps": [
                    {"range": [0, threshold_pct], "color": "#f8d7da"},
                    {"range": [threshold_pct, 100], "color": "#d1e7dd"},
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
    st.plotly_chart(fig, width="stretch")

    if pred == 1:
        st.success(
            f"### ✅ Dự đoán: CÓ KHẢ NĂNG MUA\n"
            f"Score = **{score_pct:.1f}%** ≥ ngưỡng **{threshold_pct:.1f}%**"
        )
    else:
        st.error(
            f"### ❌ Dự đoán: KHÔNG MUA\n"
            f"Score = **{score_pct:.1f}%** < ngưỡng **{threshold_pct:.1f}%**"
        )

    st.caption(
        "Score là điểm xu hướng mua của model, chưa được hiệu chỉnh thành xác suất thực. "
        "Xem vì sao model dự đoán như vậy ở tab 5️⃣."
    )