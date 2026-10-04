import os

import pandas as pd
import streamlit as st

from src.models import normalize_legacy_bundle_metrics
from src.storage import load_bundle_local

DEFAULT_DATASET_FILES = ["online_shoppers.csv"]


def _load_default_dataset():
    for filename in DEFAULT_DATASET_FILES:
        if os.path.exists(filename):
            return pd.read_csv(filename), filename
    return None, None


def init_session_state():
    if "dataset" not in st.session_state:
        df, name = _load_default_dataset()
        st.session_state["dataset"] = df
        st.session_state["dataset_name"] = name

    if "bundle" not in st.session_state:
        st.session_state["bundle"] = normalize_legacy_bundle_metrics(
            load_bundle_local()
        )