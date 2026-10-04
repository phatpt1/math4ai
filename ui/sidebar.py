import os

import pandas as pd
import streamlit as st

from src.data import normalize_dataset
from src.github_client import load_saved_bundle, push_bundle_to_github
from src.models import normalize_legacy_bundle_metrics, train_everything
from src.storage import serialize_bundle


def init_session_state():
    if "dataset" not in st.session_state:
        st.session_state["dataset"] = None

    if "dataset_name" not in st.session_state:
        st.session_state["dataset_name"] = None

    if "bundle" not in st.session_state:
        # Auto-load the previously trained model on app start
        st.session_state["bundle"] = normalize_legacy_bundle_metrics(load_saved_bundle())


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Pipeline")

        if st.session_state["bundle"] is not None:
            st.success("✅ Đã load model đã train. Không cần train lại để Predict.")
            st.caption(
                "Chỉ bấm Train khi bạn thay dataset, "
                "hyperparameter hoặc muốn cập nhật model."
            )
        else:
            st.warning("⚠️ Chưa có model đã lưu. Lần đầu cần Train + cập nhật GitHub.")

        uploaded = st.file_uploader("Upload online_shoppers.csv", type=["csv"])

        if uploaded is not None:
            try:
                st.session_state["dataset"] = pd.read_csv(uploaded)
                st.session_state["dataset_name"] = uploaded.name
                st.success(f"Đã nạp {uploaded.name}")
            except Exception as e:
                st.error(f"Không đọc được CSV: {e}")

        # Fall back to a CSV committed in the repo
        if st.session_state["dataset"] is None:
            for filename in ["online_shoppers.csv", "online_shoppers(1).csv"]:
                if os.path.exists(filename):
                    st.session_state["dataset"] = pd.read_csv(filename)
                    st.session_state["dataset_name"] = filename
                    st.info(f"Tự động dùng `{filename}`")
                    break

        if st.session_state["dataset"] is None:
            st.warning("Upload dataset hoặc đặt `online_shoppers.csv` trong repo.")

        else:
            st.write(f"**Dataset:** {st.session_state['dataset_name']}")
            st.write(f"**Rows:** {len(st.session_state['dataset']):,}")

            train_only = st.button(
                "🧠 Re-train chỉ trong Streamlit",
                use_container_width=True,
            )

            train_and_push = st.button(
                "🚀 Re-train + cập nhật GitHub",
                type="primary",
                use_container_width=True,
            )

            if train_only or train_and_push:
                try:
                    with st.spinner("Đang chuẩn hóa dữ liệu..."):
                        clean_df, category_levels = normalize_dataset(
                            st.session_state["dataset"]
                        )

                    with st.spinner(
                        "Đang train Decision Tree, Bagging, Random Forest, LightGBM..."
                    ):
                        bundle = normalize_legacy_bundle_metrics(
                            train_everything(clean_df, category_levels)
                        )

                    st.session_state["dataset"] = clean_df
                    st.session_state["bundle"] = bundle

                    st.success("✅ Train hoàn tất.")

                    if train_and_push:
                        with st.spinner("Đang cập nhật GitHub..."):
                            pushed = push_bundle_to_github(bundle)

                        st.success("✅ Model + metrics đã được commit lên GitHub.")

                        st.link_button(
                            "🔗 Xem commit model",
                            pushed["model_commit_url"],
                            use_container_width=True,
                        )

                        st.link_button(
                            "🔗 Xem commit metrics",
                            pushed["metrics_commit_url"],
                            use_container_width=True,
                        )

                except Exception as e:
                    st.exception(e)

        if st.session_state["bundle"] is not None:
            st.divider()

            bundle_bytes = serialize_bundle(st.session_state["bundle"])

            st.download_button(
                "⬇️ Tải model bundle",
                data=bundle_bytes,
                file_name="purchase_model_bundle.joblib",
                mime="application/octet-stream",
                use_container_width=True,
            )