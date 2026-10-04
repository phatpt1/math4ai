import pandas as pd
import streamlit as st

from src.inference import score_one
from ui.components import render_purchase_demo


def render():
    st.header("4. Live Prediction")

    bundle = st.session_state["bundle"]

    if bundle is None:
        st.info("Train model trước.")
        return

    model_names = list(bundle["models"].keys())
    model_name = st.selectbox(
        "Model",
        model_names,
        index=model_names.index(bundle["recommended_model"]),
    )

    mode = st.radio(
        "Input mode",
        ["Simple Demo", "Full 17 Features"],
        horizontal=True,
    )

    row = dict(bundle["defaults"])

    st.info("Simple Demo dùng median/mode của TRAIN cho feature không nhập.")

    with st.form("prediction_form"):
        if mode == "Simple Demo":
            c1, c2 = st.columns(2)

            with c1:
                row["ProductRelated"] = st.number_input(
                    "ProductRelated",
                    min_value=0,
                    value=int(bundle["defaults"]["ProductRelated"]),
                    step=1,
                )
                row["ProductRelated_Duration"] = st.number_input(
                    "ProductRelated_Duration",
                    min_value=0.0,
                    value=float(bundle["defaults"]["ProductRelated_Duration"]),
                )
                row["PageValues"] = st.number_input(
                    "PageValues",
                    min_value=0.0,
                    value=float(bundle["defaults"]["PageValues"]),
                )
                row["BounceRates"] = st.number_input(
                    "BounceRates",
                    min_value=0.0,
                    value=float(bundle["defaults"]["BounceRates"]),
                    format="%.4f",
                )
                row["ExitRates"] = st.number_input(
                    "ExitRates",
                    min_value=0.0,
                    value=float(bundle["defaults"]["ExitRates"]),
                    format="%.4f",
                )

            with c2:
                for col in ["Month", "VisitorType", "Weekend", "Region"]:
                    levels = bundle["category_levels"][col]
                    default = bundle["defaults"][col]
                    idx = levels.index(default) if default in levels else 0

                    row[col] = st.selectbox(
                        col,
                        levels,
                        index=idx,
                        key=f"simple_{col}",
                    )

        else:
            st.markdown("#### Numeric")

            num_cols = st.columns(2)

            for i, col in enumerate(bundle["numeric_features"]):
                default = float(bundle["defaults"][col])

                row[col] = num_cols[i % 2].number_input(
                    col,
                    value=default,
                    key=f"num_{col}",
                )

            st.markdown("#### Categorical")

            cat_cols = st.columns(2)

            for i, col in enumerate(bundle["categorical_features"]):
                levels = bundle["category_levels"][col]
                default = bundle["defaults"][col]
                idx = levels.index(default) if default in levels else 0

                row[col] = cat_cols[i % 2].selectbox(
                    col,
                    levels,
                    index=idx,
                    key=f"cat_{col}",
                )

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if not submitted:
        return

    input_df = pd.DataFrame(
        [[row[col] for col in bundle["feature_order"]]],
        columns=bundle["feature_order"],
    )

    score, threshold, pred = score_one(model_name, input_df, bundle)

    st.session_state["last_input"] = input_df
    st.session_state["last_model_name"] = model_name

    c1, c2, c3 = st.columns(3)
    c1.metric("Model score — Mua", f"{score*100:.2f}%")
    c2.metric("Threshold", f"{threshold:.3f}")
    c3.metric("Prediction", "MUA" if pred == 1 else "KHÔNG MUA")

    st.progress(max(0.0, min(1.0, score)))

    st.warning("Model score chưa mặc nhiên là calibrated probability.")

    render_purchase_demo(
        score=score,
        threshold=threshold,
        pred=pred,
        model_name=model_name,
        input_df=input_df,
        bundle=bundle,
    )