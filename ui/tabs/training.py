import streamlit as st


def render():
    st.header("2. Training")

    bundle = st.session_state["bundle"]

    st.markdown(
        r"""
### Data split

\[
Train = 70\%
\]

\[
Validation = 15\%
\]

\[
Test = 15\%
\]

- Train: học model.
- Validation: early stopping cho LightGBM + chọn threshold + chọn model.
- Test: chỉ đánh giá cuối.
"""
    )

    if bundle is None:
        st.info("Bấm Train ở sidebar.")
        return

    s = bundle["dataset_summary"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Train", s["train"])
    c2.metric("Validation", s["validation"])
    c3.metric("Test", s["test"])

    c1, c2 = st.columns(2)
    c1.metric("scale_pos_weight", f"{bundle['scale_pos_weight']:.3f}")
    c2.metric("LightGBM best iteration", bundle["lgbm_best_iteration"])