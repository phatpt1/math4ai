import pandas as pd
import plotly.express as px
import streamlit as st


def render():
    st.header("3. Model Comparison")

    bundle = st.session_state["bundle"]

    if bundle is None:
        st.info("Train 4 model trước để có dữ liệu so sánh.")
        return

    st.markdown(
        """
### Logic chọn model

Không phải **train xong rồi nhìn Test để chọn**. Flow đúng là:

**Train 4 model → Validation chọn model + threshold → khóa lựa chọn → Test báo cáo cuối.**

Vì `Revenue=True` là lớp thiểu số, tiêu chí chính là **AP trên Validation**.
Nếu AP bằng nhau, dùng **F1**, sau đó **Recall** để phá hòa.
        """
    )

    if "validation_metrics" not in bundle:
        st.warning(
            "Bundle đang load là phiên bản cũ, chưa lưu Validation metrics. "
            "Hãy Re-train một lần bằng code FINAL để việc chọn model không dùng Test."
        )

    val_metrics_dict = bundle.get("validation_metrics", bundle["metrics"])
    test_metrics_dict = bundle["metrics"]

    val_metrics = (
        pd.DataFrame(val_metrics_dict).T
        .reset_index()
        .rename(columns={"index": "Model"})
    )
    test_metrics = (
        pd.DataFrame(test_metrics_dict).T
        .reset_index()
        .rename(columns={"index": "Model"})
    )

    cols = [
        "Model", "ap", "recall", "f1",
        "precision", "roc_auc", "accuracy", "threshold"
    ]

    st.subheader("A. Validation — dùng để CHỌN model")
    val_table = val_metrics[cols].copy()
    for col in cols[1:]:
        val_table[col] = val_table[col].astype(float).round(4)
    st.dataframe(
        val_table.sort_values(["ap", "f1", "recall"], ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    winner = bundle["recommended_model"]
    st.success(
        f"🏆 Model được chọn: **{winner}** — "
        f"dựa trên Validation, không nhìn trước Test."
    )
    st.caption(bundle.get("selection_rule", "Chọn theo Validation AP."))

    winner_row = val_metrics[val_metrics["Model"] == winner].iloc[0]
    st.markdown(
        f"**Vì sao {winner} thắng?** Validation AP = "
        f"**{float(winner_row['ap']):.4f}**, "
        f"F1 = **{float(winner_row['f1']):.4f}**, "
        f"Recall = **{float(winner_row['recall']):.4f}**. "
        "Theo rule của project, AP được xét trước."
    )

    st.subheader("B. Test — chỉ báo cáo cuối")
    test_table = test_metrics[cols].copy()
    for col in cols[1:]:
        test_table[col] = test_table[col].astype(float).round(4)
    st.dataframe(
        test_table.sort_values("ap", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    metric_name = st.selectbox(
        "Vẽ metric nào?",
        ["ap", "recall", "f1", "precision", "roc_auc", "accuracy"],
    )
    view_split = st.radio(
        "Dữ liệu để vẽ",
        ["Validation", "Test"],
        horizontal=True,
    )
    plot_metrics = val_metrics if view_split == "Validation" else test_metrics

    fig = px.bar(
        plot_metrics.sort_values(metric_name),
        x=metric_name,
        y="Model",
        orientation="h",
        text=metric_name,
        title=f"{view_split}: so sánh {metric_name.upper()}",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Cách đọc: AP = tiêu chí chọn chính; Recall = bắt được bao nhiêu "
        "khách thực sự mua; F1 = cân bằng Precision và Recall. Accuracy chỉ "
        "là chỉ số bổ sung vì target mất cân bằng."
    )

    st.subheader("LightGBM Feature Importance — chẩn đoán riêng LightGBM")
    st.caption(
        "Phần này giúp hiểu LightGBM, KHÔNG phải quy tắc chọn model. "
        "Model thắng cuộc vẫn do Validation metrics quyết định."
    )

    imp_mode = st.radio("Importance", ["Gain", "Split"], horizontal=True)
    imp = bundle["lgbm_feature_importance"].copy()

    if imp_mode == "Gain":
        plot_df = imp.nlargest(12, "gain_pct").sort_values("gain_pct")
        fig = px.bar(
            plot_df, x="gain_pct", y="feature", orientation="h",
            text="gain_pct", title="LightGBM Gain importance"
        )
        fig.update_traces(texttemplate="%{text:.1f}%")
        st.caption(
            "Gain = tổng mức cải thiện objective khi feature được dùng. "
            "Gain không cho biết hướng tác động."
        )
    else:
        plot_df = imp.nlargest(12, "split").sort_values("split")
        fig = px.bar(
            plot_df, x="split", y="feature", orientation="h",
            text="split", title="LightGBM Split importance"
        )
        st.caption("Split = số lần feature được dùng để chia node.")

    st.plotly_chart(fig, use_container_width=True)