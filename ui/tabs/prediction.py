import pandas as pd
import streamlit as st

from src.inference import score_one
from ui.components import render_purchase_demo

MAIN_FEATURES = [
    "PageValues",
    "ExitRates",
    "BounceRates",
    "ProductRelated",
    "ProductRelated_Duration",
    "Month",
    "VisitorType",
    "Weekend",
]
INT_FEATURES = {"Administrative", "Informational", "ProductRelated"}
RATE_FEATURES = {"BounceRates", "ExitRates"}

TYPICAL = "👤 Khách điển hình"

PRESETS = {
    TYPICAL: {},
    "🛒 Khách tiềm năng": {
        "Administrative": 4,
        "Administrative_Duration": 120.0,
        "ProductRelated": 45,
        "ProductRelated_Duration": 1800.0,
        "PageValues": 40.0,
        "BounceRates": 0.0,
        "ExitRates": 0.01,
        "Month": "Nov",
        "VisitorType": "New_Visitor",
        "Weekend": False,
    },
     "🤔 Khách phân vân": {
        "Administrative": 2,
        "Administrative_Duration": 60.0,
        "ProductRelated": 25,
        "ProductRelated_Duration": 900.0,
        "PageValues": 4.0,
        "BounceRates": 0.005,
        "ExitRates": 0.04,
        "Month": "Mar",
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    },
    "👀 Khách lướt web": {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 0.0,
        "PageValues": 0.0,
        "BounceRates": 0.2,
        "ExitRates": 0.2,
        "Month": "Feb",
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    },
}


def _key(col):
    return f"in_{col}"


def _set_inputs(values, bundle):
    """Fill every input widget from a preset, falling back to TRAIN median/mode."""
    for col in bundle["feature_order"]:
        value = values.get(col, bundle["defaults"][col])

        if col in bundle["numeric_features"]:
            value = int(round(value)) if col in INT_FEATURES else float(value)
        elif value not in bundle["category_levels"][col]:
            value = bundle["defaults"][col]

        st.session_state[_key(col)] = value


def _init_inputs(bundle):
    # Re-initialize when a new bundle is trained
    if st.session_state.get("pred_inputs_for") != bundle.get("trained_at_utc"):
        _set_inputs({}, bundle)
        st.session_state["pred_inputs_for"] = bundle.get("trained_at_utc")
        st.session_state["preset_choice"] = TYPICAL


def _on_preset_change(bundle):
    name = st.session_state.get("preset_choice")
    if name is not None:
        _set_inputs(PRESETS[name], bundle)


def _render_presets(bundle):
    st.segmented_control(
        "Kịch bản mẫu",
        options=list(PRESETS.keys()),
        key="preset_choice",
        on_change=_on_preset_change,
        args=(bundle,),
        help="Khách điển hình = median (số) / mode (phân loại) của tập TRAIN",
    )
    st.caption("Chọn kịch bản để điền sẵn input, chỉnh tay nếu cần rồi bấm Predict.")


def _input_widget(col, bundle, container):
    if col in bundle["numeric_features"]:
        kwargs = {"min_value": 0 if col in INT_FEATURES else 0.0}
        if col in INT_FEATURES:
            kwargs["step"] = 1
        if col in RATE_FEATURES:
            kwargs["format"] = "%.4f"
        return container.number_input(col, key=_key(col), **kwargs)

    return container.selectbox(col, bundle["category_levels"][col], key=_key(col))


def _render_group(cols, bundle, row):
    left, right = st.columns(2)
    numeric = [c for c in cols if c in bundle["numeric_features"]]
    categorical = [c for c in cols if c not in bundle["numeric_features"]]

    for col in numeric:
        row[col] = _input_widget(col, bundle, left)
    for col in categorical:
        row[col] = _input_widget(col, bundle, right)


def render():
    st.header("4. Live Prediction")

    bundle = st.session_state["bundle"]
    if bundle is None:
        st.info("Train model ở tab 2️⃣ trước.")
        return

    _init_inputs(bundle)

    model_names = list(bundle["models"].keys())
    model_name = st.selectbox(
        "Model",
        model_names,
        index=model_names.index(bundle["recommended_model"]),
    )

    _render_presets(bundle)

    main = [c for c in MAIN_FEATURES if c in bundle["feature_order"]]
    others = [c for c in bundle["feature_order"] if c not in main]
    row = {}

    with st.form("prediction_form"):
        st.markdown("##### Feature chính")
        _render_group(main, bundle, row)

        with st.expander(f"Các feature khác ({len(others)})"):
            _render_group(others, bundle, row)

        submitted = st.form_submit_button("🔮 Predict", width="stretch")

    if not submitted:
        return

    input_df = pd.DataFrame(
        [[row[col] for col in bundle["feature_order"]]],
        columns=bundle["feature_order"],
    )

    score, threshold, pred = score_one(model_name, input_df, bundle)

    st.session_state["last_input"] = input_df
    st.session_state["last_model_name"] = model_name

    render_purchase_demo(
        score=score,
        threshold=threshold,
        pred=pred,
        model_name=model_name,
    )