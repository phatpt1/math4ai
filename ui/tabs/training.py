import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import normalize_dataset
from src.models import train_everything
from src.storage import save_bundle_local

MODEL_SETUP = pd.DataFrame(
    [
        {
            "Model": "Decision Tree",
            "Type": "Single tree (baseline)",
            "Categorical handling": "OneHotEncoder",
            "Imbalance handling": "class_weight='balanced'",
        },
        {
            "Model": "Bagging",
            "Type": "Bootstrap + 120 trees, averaging",
            "Categorical handling": "OneHotEncoder",
            "Imbalance handling": "class_weight='balanced'",
        },
        {
            "Model": "Random Forest",
            "Type": "Bagging + random feature subsets (250 trees)",
            "Categorical handling": "OneHotEncoder",
            "Imbalance handling": "class_weight='balanced'",
        },
        {
            "Model": "LightGBM",
            "Type": "Gradient boosting (sequential trees)",
            "Categorical handling": "Native categorical",
            "Imbalance handling": "scale_pos_weight + early stopping",
        },
    ]
)


def _render_theory():
    st.markdown(
        r"""
### Data split

$$
\text{Train} = 70\% \quad | \quad \text{Validation} = 15\% \quad | \quad \text{Test} = 15\%
$$

- **Train:** học model (encoder cũng chỉ fit trên Train).
- **Validation:** early stopping cho LightGBM, chọn threshold, chọn model.
- **Test:** chỉ đánh giá cuối, không dùng để tinh chỉnh → tránh **data leakage**.

Chia theo **stratified** để 3 tập giữ cùng tỷ lệ mua (~15.5%).
"""
    )

    st.subheader("4 model so sánh")
    st.dataframe(MODEL_SETUP, hide_index=True, width="stretch")


def _render_train_action():
    df = st.session_state["dataset"]
    if df is None:
        st.info("Chưa có dataset.")
        return

    c1, c2 = st.columns([3, 1])
    c1.caption(f"Dataset: `{st.session_state.get('dataset_name')}` · {len(df):,} rows")

    if c2.button("🧠 Train lại model", type="primary", width="stretch"):
        try:
            with st.spinner("Đang train Decision Tree, Bagging, Random Forest, LightGBM..."):
                clean_df, category_levels = normalize_dataset(df)
                bundle = train_everything(clean_df, category_levels)
                path = save_bundle_local(bundle)

            st.session_state["dataset"] = clean_df
            st.session_state["bundle"] = bundle
            st.session_state["train_notice"] = f"✅ Train xong, đã lưu model vào `{path}`."
            st.rerun()
        except Exception as e:
            st.exception(e)

    notice = st.session_state.pop("train_notice", None)
    if notice:
        st.success(notice)


def _render_learning_curve(evals, best_iter):
    fig = go.Figure()
    for set_name, metrics in evals.items():
        for values in metrics.values():
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(values) + 1)),
                    y=values,
                    mode="lines",
                    name=set_name.capitalize(),
                    line=dict(dash="dash" if set_name == "validation" else "solid"),
                )
            )
    if best_iter:
        fig.add_vline(x=best_iter, line_dash="dot", annotation_text=f"best iteration = {best_iter}")
    fig.update_layout(
        title="LightGBM learning curve",
        xaxis_title="Boosting iteration",
        yaxis_title="Log loss",
    )
    st.plotly_chart(fig, width="stretch")


def _render_results(bundle):
    s = bundle["dataset_summary"]

    st.subheader("Kết quả chia dữ liệu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train", f"{s['train']:,}")
    c2.metric("Validation", f"{s['validation']:,}")
    c3.metric("Test", f"{s['test']:,}")

    st.subheader("Xử lý mất cân bằng & early stopping")
    spw = bundle["scale_pos_weight"]
    best_iter = bundle.get("lgbm_best_iteration")

    st.markdown(
        rf"""
$$
\text{{scale\_pos\_weight}} = \frac{{\#\text{{không mua (Train)}}}}{{\#\text{{mua (Train)}}}} \approx {spw:.2f}
$$

→ Mỗi lỗi bỏ sót khách **mua** bị phạt nặng gấp ~{spw:.1f} lần. LightGBM dừng sớm ở **{best_iter} cây** (tối đa 500).
"""
    )

    evals = bundle.get("lgbm_evals_result")
    if evals:
        _render_learning_curve(evals, best_iter)


def render():
    st.header("2. Training")

    _render_theory()
    _render_train_action()

    bundle = st.session_state["bundle"]
    if bundle is not None:
        _render_results(bundle)