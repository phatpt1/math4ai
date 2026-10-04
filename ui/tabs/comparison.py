import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

METRIC_COLS = ["ap", "recall", "f1", "precision", "roc_auc", "accuracy", "threshold"]


def _to_table(metrics_dict):
    df = (
        pd.DataFrame(metrics_dict).T
        .reset_index()
        .rename(columns={"index": "Model"})
    )
    table = df[["Model"] + METRIC_COLS].copy()
    for col in METRIC_COLS:
        table[col] = table[col].astype(float).round(4)
    return table.sort_values(["ap", "f1", "recall"], ascending=False)


def _render_selection_logic():
    st.markdown(
        """
**Train 4 model → Validation chọn model + threshold → khóa lựa chọn → Test báo cáo cuối.**

Tiêu chí chọn: **AP trên Validation** (hòa thì xét F1, rồi Recall). Test **không** được dùng để chọn.
"""
    )


def _render_tables(bundle):
    winner = bundle["recommended_model"]

    st.subheader("A. Validation — dùng để CHỌN model")
    st.dataframe(_to_table(bundle["validation_metrics"]), width="stretch", hide_index=True)
    st.success(f"🏆 Model được chọn: **{winner}** — dựa trên Validation AP, không nhìn trước Test.")

    st.subheader("B. Test — chỉ báo cáo cuối")
    st.dataframe(_to_table(bundle["metrics"]), width="stretch", hide_index=True)


def _render_confusion_matrix(bundle):
    winner = bundle["recommended_model"]
    m = bundle["metrics"][winner]
    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]

    st.subheader(f"C. Confusion Matrix — {winner} trên Test")

    fig = px.imshow(
        np.array([[tn, fp], [fn, tp]]),
        x=["Pred: Not purchased", "Pred: Purchased"],
        y=["Actual: Not purchased", "Actual: Purchased"],
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        title=f"threshold = {m['threshold']:.3f}",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"""
- Trong **{tp + fn}** khách thực sự mua: bắt được **{tp}**, bỏ sót **{fn}** → Recall = **{tp / max(tp + fn, 1):.1%}**
- Trong **{tp + fp}** phiên model báo mua: đúng **{tp}**, báo nhầm **{fp}** → Precision = **{tp / max(tp + fp, 1):.1%}**
"""
    )


def _render_pr_curve(bundle):
    preds = bundle.get("predictions")
    if not preds:
        return

    st.subheader("D. Precision–Recall curve trên Test")

    winner = bundle["recommended_model"]
    y_true = preds["y_test"]
    fig = go.Figure()

    for name, prob in preds["test_prob"].items():
        precision, recall, _ = precision_recall_curve(y_true, prob)
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{name} (AP = {bundle['metrics'][name]['ap']:.3f})",
                line=dict(width=3 if name == winner else 1.5),
            )
        )

    m = bundle["metrics"][winner]
    fig.add_trace(
        go.Scatter(
            x=[m["recall"]],
            y=[m["precision"]],
            mode="markers",
            marker=dict(size=12, symbol="star"),
            name=f"{winner} @ threshold {m['threshold']:.3f}",
        )
    )

    base_rate = float(np.mean(y_true))
    fig.add_hline(y=base_rate, line_dash="dash", annotation_text=f"random = {base_rate:.3f}")
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("AP = diện tích dưới đường PR. Ngôi sao = điểm làm việc thực tế của model được chọn.")


def _render_feature_importance(bundle):
    st.subheader("E. Feature Importance (LightGBM, Gain)")

    imp = bundle["lgbm_feature_importance"]
    plot_df = imp.nlargest(10, "gain_pct").sort_values("gain_pct")

    fig = px.bar(
        plot_df,
        x="gain_pct",
        y="feature",
        orientation="h",
        text="gain_pct",
        labels={"gain_pct": "Gain (%)", "feature": ""},
    )
    fig.update_traces(texttemplate="%{text:.1f}%")
    st.plotly_chart(fig, width="stretch")
    st.caption("Gain = tổng mức giảm loss mà feature mang lại. Chỉ cho biết mức độ quan trọng, không cho biết chiều tác động.")


def render():
    st.header("3. Model Comparison")

    bundle = st.session_state["bundle"]
    if bundle is None:
        st.info("Train 4 model ở tab 2️⃣ trước.")
        return

    _render_selection_logic()
    _render_tables(bundle)
    _render_confusion_matrix(bundle)
    _render_pr_curve(bundle)
    _render_feature_importance(bundle)