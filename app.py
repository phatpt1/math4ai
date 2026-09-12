# ============================================================
# ONLINE SHOPPER PURCHASE PREDICTION — ONE FILE ONLY
# Streamlit + Training + Evaluation + Prediction + Explainability
# + Auto update trained model/metrics to GitHub
# ============================================================
#
# Streamlit Secrets cần có:
#
# GITHUB_TOKEN = "YOUR_TOKEN"
#
# Có thể thêm:
# GITHUB_REPO = "phatpt1/math4ai"
# GITHUB_BRANCH = "main"
#
# KHÔNG hard-code token vào source code.
#
# requirements.txt tối thiểu:
# streamlit
# pandas
# numpy
# scikit-learn
# lightgbm
# plotly
# joblib
# requests
# ============================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import base64
import json
import math
from datetime import datetime, timezone
from io import BytesIO

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 0. APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Online Shopper ML Lab",
    page_icon="🛒",
    layout="wide",
)

RANDOM_STATE = 42
TARGET = "Revenue"

DEFAULT_GITHUB_REPO = "phatpt1/math4ai"
DEFAULT_GITHUB_BRANCH = "main"

MODEL_PATH_IN_REPO = "models/purchase_model_bundle.joblib"
METRICS_PATH_IN_REPO = "artifacts/latest_metrics.json"

EXPECTED_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Revenue",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]


# ============================================================
# 1. DATA HELPERS
# ============================================================

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_bagging(base_tree):
    try:
        return BaggingClassifier(
            estimator=base_tree,
            n_estimators=120,
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except TypeError:
        return BaggingClassifier(
            base_estimator=base_tree,
            n_estimators=120,
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )


def validate_dataset(df: pd.DataFrame):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset thiếu các cột bắt buộc: " + ", ".join(missing)
        )


def normalize_dataset(df: pd.DataFrame):
    df = df.copy()
    validate_dataset(df)

    # Chuẩn hóa target về 0/1
    if df[TARGET].dtype == bool:
        df[TARGET] = df[TARGET].astype(int)
    else:
        mapping = {
            True: 1,
            False: 0,
            "True": 1,
            "False": 0,
            "TRUE": 1,
            "FALSE": 0,
            "1": 1,
            "0": 0,
            1: 1,
            0: 0,
        }

        converted = df[TARGET].map(mapping)

        if converted.isna().any():
            try:
                converted = df[TARGET].astype(int)
            except Exception as e:
                raise ValueError(
                    "Cột Revenue phải là True/False hoặc 1/0."
                ) from e

        df[TARGET] = converted.astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Revenue phải chỉ gồm 0/1.")

    # Giữ đúng category dtype. Không stringify category số/bool.
    category_levels = {}

    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = (
                df[col]
                .astype("object")
                .where(df[col].notna(), "Missing")
            )

        df[col] = df[col].astype("category")
        category_levels[col] = list(df[col].cat.categories)

    return df, category_levels


def split_data(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 70 / 15 / 15
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    val_fraction = 0.15 / 0.85

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                make_one_hot_encoder(),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def choose_threshold(y_true, prob):
    """
    Chọn threshold trên VALIDATION bằng F1 tối đa.
    Test không được dùng để tune threshold.
    """
    thresholds = np.linspace(0.05, 0.95, 181)

    scores = [
        f1_score(
            y_true,
            (prob >= t).astype(int),
            zero_division=0,
        )
        for t in thresholds
    ]

    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx])


def evaluate_model(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, prob)
        ),
        "pr_auc": float(
            average_precision_score(y_true, prob)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_defaults(X_train):
    defaults = {}

    for col in NUMERIC_FEATURES:
        defaults[col] = float(
            pd.to_numeric(X_train[col]).median()
        )

    for col in CATEGORICAL_FEATURES:
        mode = X_train[col].mode(dropna=True)
        defaults[col] = mode.iloc[0] if len(mode) else None

    return defaults


# ============================================================
# 2. TRAIN ALL MODELS
# ============================================================

def train_everything(df, category_levels):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    models = {}
    results = {}

    # -------------------------
    # Decision Tree
    # -------------------------
    tree = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "model",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=5,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # -------------------------
    # Bagging
    # -------------------------
    base_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    bagging = Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", make_bagging(base_tree)),
        ]
    )

    # -------------------------
    # Random Forest
    # -------------------------
    forest = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    bootstrap=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    sklearn_models = {
        "Decision Tree": tree,
        "Bagging": bagging,
        "Random Forest": forest,
    }

    for name, model in sklearn_models.items():
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, val_prob)

        test_prob = model.predict_proba(X_test)[:, 1]

        results[name] = evaluate_model(
            y_test,
            test_prob,
            threshold,
        )

        models[name] = model

    # -------------------------
    # LightGBM
    # -------------------------
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive

    lgbm = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=25,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    lgbm.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_FEATURES,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(
                40,
                verbose=False,
            )
        ],
    )

    val_prob = lgbm.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_prob)

    test_prob = lgbm.predict_proba(X_test)[:, 1]

    results["LightGBM"] = evaluate_model(
        y_test,
        test_prob,
        threshold,
    )

    models["LightGBM"] = lgbm

    # -------------------------
    # LightGBM feature importance
    # -------------------------
    gain = lgbm.booster_.feature_importance(
        importance_type="gain"
    )

    split = lgbm.booster_.feature_importance(
        importance_type="split"
    )

    names = lgbm.booster_.feature_name()

    gain_total = float(np.sum(gain)) or 1.0

    importance = pd.DataFrame(
        {
            "feature": names,
            "gain": gain.astype(float),
            "gain_pct": gain / gain_total * 100,
            "split": split.astype(int),
        }
    ).sort_values("gain", ascending=False)

    ranking = sorted(
        results.items(),
        key=lambda kv: kv[1]["pr_auc"],
        reverse=True,
    )

    recommended_model = ranking[0][0]

    bundle = {
        "models": models,
        "metrics": results,
        "recommended_model": recommended_model,
        "feature_order": list(X_train.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_levels": category_levels,
        "defaults": get_defaults(X_train),
        "dataset_summary": {
            "rows": int(len(df)),
            "features": int(df.shape[1] - 1),
            "positive": int(df[TARGET].sum()),
            "negative": int((1 - df[TARGET]).sum()),
            "positive_rate": float(df[TARGET].mean()),
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "scale_pos_weight": float(scale_pos_weight),
        "lgbm_best_iteration": int(
            lgbm.best_iteration_
            or lgbm.n_estimators
        ),
        "lgbm_feature_importance": importance,
        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    return bundle


# ============================================================
# 3. PREDICTION + EXPLAINABILITY
# ============================================================

def restore_lgbm_categories(df, bundle):
    df = df.copy()

    for col in bundle["categorical_features"]:
        df[col] = pd.Categorical(
            df[col],
            categories=bundle[
                "category_levels"
            ][col],
        )

    return df


def score_one(model_name, row_df, bundle):
    model = bundle["models"][model_name]

    x = row_df.copy()

    if model_name == "LightGBM":
        x = restore_lgbm_categories(
            x,
            bundle,
        )

    score = float(
        model.predict_proba(x)[0, 1]
    )

    threshold = float(
        bundle["metrics"][
            model_name
        ]["threshold"]
    )

    pred = int(score >= threshold)

    return score, threshold, pred


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)

    z = math.exp(x)
    return z / (1 + z)


def get_lgbm_contributions(row_df, bundle):
    model = bundle["models"]["LightGBM"]

    x = restore_lgbm_categories(
        row_df,
        bundle,
    )

    contrib = model.booster_.predict(
        x,
        pred_contrib=True,
    )[0]

    feature_values = np.asarray(
        contrib[:-1],
        dtype=float,
    )

    base = float(contrib[-1])
    raw_score = base + float(
        feature_values.sum()
    )

    contrib_df = pd.DataFrame(
        {
            "feature": bundle[
                "feature_order"
            ],
            "contribution": feature_values,
            "abs_contribution": np.abs(
                feature_values
            ),
        }
    ).sort_values(
        "abs_contribution",
        ascending=False,
    )

    return base, raw_score, contrib_df


# ============================================================
# 4. SERIALIZATION
# ============================================================

def serialize_bundle(bundle) -> bytes:
    buffer = BytesIO()
    joblib.dump(
        bundle,
        buffer,
        compress=3,
    )
    return buffer.getvalue()


def make_metrics_json(bundle) -> str:
    payload = {
        "updated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "recommended_model": bundle[
            "recommended_model"
        ],
        "scale_pos_weight": bundle.get(
            "scale_pos_weight"
        ),
        "lgbm_best_iteration": bundle.get(
            "lgbm_best_iteration"
        ),
        "dataset_summary": bundle.get(
            "dataset_summary"
        ),
        "metrics": bundle.get(
            "metrics"
        ),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 5. GITHUB API — SAME FILE
# ============================================================

def get_github_config():
    if "GITHUB_TOKEN" not in st.secrets:
        raise RuntimeError(
            "Thiếu GITHUB_TOKEN trong Streamlit Secrets."
        )

    repo = st.secrets.get(
        "GITHUB_REPO",
        DEFAULT_GITHUB_REPO,
    )

    branch = st.secrets.get(
        "GITHUB_BRANCH",
        DEFAULT_GITHUB_BRANCH,
    )

    return {
        "token": st.secrets["GITHUB_TOKEN"],
        "repo": repo,
        "branch": branch,
    }


def github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "math4ai-streamlit",
    }


def github_contents_url(repo, path):
    clean_path = path.lstrip("/")
    return (
        f"https://api.github.com/"
        f"repos/{repo}/contents/{clean_path}"
    )


def github_get_file_sha(
    token,
    repo,
    branch,
    path,
):
    response = requests.get(
        github_contents_url(
            repo,
            path,
        ),
        headers=github_headers(token),
        params={"ref": branch},
        timeout=30,
    )

    if response.status_code == 404:
        return None

    if not response.ok:
        raise RuntimeError(
            "GitHub GET lỗi "
            f"{response.status_code}: "
            f"{response.text}"
        )

    return response.json().get("sha")


def github_upsert_bytes(
    token,
    repo,
    branch,
    path,
    content,
    commit_message,
):
    sha = github_get_file_sha(
        token,
        repo,
        branch,
        path,
    )

    payload = {
        "message": commit_message,
        "content": base64.b64encode(
            content
        ).decode("ascii"),
        "branch": branch,
    }

    if sha:
        payload["sha"] = sha

    response = requests.put(
        github_contents_url(
            repo,
            path,
        ),
        headers=github_headers(token),
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            "GitHub PUT lỗi "
            f"{response.status_code}: "
            f"{response.text}"
        )

    return response.json()


def github_upsert_text(
    token,
    repo,
    branch,
    path,
    text,
    commit_message,
):
    return github_upsert_bytes(
        token,
        repo,
        branch,
        path,
        text.encode("utf-8"),
        commit_message,
    )


def push_bundle_to_github(bundle):
    config = get_github_config()

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    model_bytes = serialize_bundle(
        bundle
    )

    metrics_text = make_metrics_json(
        bundle
    )

    model_result = github_upsert_bytes(
        token=config["token"],
        repo=config["repo"],
        branch=config["branch"],
        path=MODEL_PATH_IN_REPO,
        content=model_bytes,
        commit_message=(
            f"Update trained model - {stamp}"
        ),
    )

    metrics_result = github_upsert_text(
        token=config["token"],
        repo=config["repo"],
        branch=config["branch"],
        path=METRICS_PATH_IN_REPO,
        text=metrics_text,
        commit_message=(
            f"Update ML metrics - {stamp}"
        ),
    )

    return {
        "repo": config["repo"],
        "branch": config["branch"],
        "model_commit_url": (
            model_result["commit"]["html_url"]
        ),
        "metrics_commit_url": (
            metrics_result["commit"]["html_url"]
        ),
    }



# ============================================================
# 5B. LOAD SAVED MODEL ON APP START
# ============================================================

@st.cache_resource(show_spinner=False)
def load_saved_bundle():
    """
    Ưu tiên:
    1) Load model đã commit trong repo local:
       models/purchase_model_bundle.joblib
    2) Nếu file local chưa có, thử tải trực tiếp từ GitHub Contents API.

    Nhờ vậy:
    - Không cần train lại mỗi lần Streamlit restart.
    - Train chỉ dùng khi muốn cập nhật model.
    """
    # 1. Local repo file
    if os.path.exists(MODEL_PATH_IN_REPO):
        try:
            return joblib.load(MODEL_PATH_IN_REPO)
        except Exception as e:
            st.warning(
                f"Tìm thấy model local nhưng load thất bại: {e}"
            )

    # 2. GitHub fallback
    try:
        if "GITHUB_TOKEN" not in st.secrets:
            return None

        config = get_github_config()

        response = requests.get(
            github_contents_url(
                config["repo"],
                MODEL_PATH_IN_REPO,
            ),
            headers=github_headers(
                config["token"]
            ),
            params={
                "ref": config["branch"]
            },
            timeout=60,
        )

        if response.status_code == 404:
            return None

        if not response.ok:
            st.warning(
                "Không tải được model từ GitHub: "
                f"{response.status_code}"
            )
            return None

        payload = response.json()

        # GitHub Contents API trả binary dưới dạng base64
        encoded = payload.get("content")
        if not encoded:
            return None

        model_bytes = base64.b64decode(
            encoded
        )

        return joblib.load(
            BytesIO(model_bytes)
        )

    except Exception as e:
        st.warning(
            f"Không thể load model đã lưu từ GitHub: {e}"
        )
        return None



# ============================================================
# 5C. VISUAL PURCHASE DEMO
# ============================================================

def render_purchase_demo(score, threshold, pred, model_name, input_df, bundle):
    """
    Demo trực quan nhưng KHÔNG bịa reasoning.
    - Gauge hiển thị model score.
    - Threshold lấy từ validation.
    - Marketing suggestion chỉ là rule demo,
      tách biệt khỏi reasoning của model.
    - Với LightGBM, top local contributions lấy trực tiếp từ model.
    """
    st.markdown("### 🎯 Demo trực quan khả năng mua hàng")

    score_pct = float(score * 100.0)
    threshold_pct = float(threshold * 100.0)

    # Gauge
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

    # Decision card
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

    # Score bands — explicitly business demo, not model explanation
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

    # Immediate real explanation for LightGBM
    if "LightGBM" in bundle["models"]:
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

    # Input summary
    with st.expander("🔎 Xem dữ liệu khách hàng vừa nhập"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)


# ============================================================
# 6. SESSION STATE
# ============================================================

if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None

if "bundle" not in st.session_state:
    # QUAN TRỌNG:
    # Khi app khởi động, tự load model đã train trước đó.
    st.session_state["bundle"] = load_saved_bundle()


# ============================================================
# 7. HEADER
# ============================================================

st.title(
    "🛒 Online Shopper Purchase Prediction — ML Lab"
)

st.caption(
    "Một app.py duy nhất: "
    "Load model đã lưu → Predict ngay; "
    "chỉ Re-train khi muốn cập nhật model."
)

if st.session_state.get("bundle") is not None:
    st.success(
        "✅ Model đã được load tự động từ repo/GitHub. "
        "Bạn có thể Predict ngay mà không cần train lại."
    )


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Pipeline")

    if st.session_state["bundle"] is not None:
        st.success(
            "✅ Đã load model đã train. "
            "Không cần train lại để Predict."
        )
        st.caption(
            "Chỉ bấm Train khi bạn thay dataset, "
            "hyperparameter hoặc muốn cập nhật model."
        )
    else:
        st.warning(
            "⚠️ Chưa có model đã lưu. "
            "Lần đầu cần Train + cập nhật GitHub."
        )

    uploaded = st.file_uploader(
        "Upload online_shoppers.csv",
        type=["csv"],
    )

    if uploaded is not None:
        try:
            st.session_state[
                "dataset"
            ] = pd.read_csv(
                uploaded
            )

            st.session_state[
                "dataset_name"
            ] = uploaded.name

            st.success(
                f"Đã nạp {uploaded.name}"
            )

        except Exception as e:
            st.error(
                f"Không đọc được CSV: {e}"
            )

    # Nếu CSV đã nằm trong repo thì tự đọc.
    if st.session_state["dataset"] is None:
        for filename in [
            "online_shoppers.csv",
            "online_shoppers(1).csv",
        ]:
            if os.path.exists(filename):
                st.session_state[
                    "dataset"
                ] = pd.read_csv(
                    filename
                )

                st.session_state[
                    "dataset_name"
                ] = filename

                st.info(
                    f"Tự động dùng `{filename}`"
                )
                break

    if st.session_state["dataset"] is None:
        st.warning(
            "Upload dataset hoặc đặt "
            "`online_shoppers.csv` trong repo."
        )

    else:
        st.write(
            "**Dataset:** "
            f"{st.session_state['dataset_name']}"
        )

        st.write(
            "**Rows:** "
            f"{len(st.session_state['dataset']):,}"
        )

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
                with st.spinner(
                    "Đang chuẩn hóa dữ liệu..."
                ):
                    clean_df, category_levels = (
                        normalize_dataset(
                            st.session_state[
                                "dataset"
                            ]
                        )
                    )

                with st.spinner(
                    "Đang train Decision Tree, "
                    "Bagging, Random Forest, LightGBM..."
                ):
                    bundle = train_everything(
                        clean_df,
                        category_levels,
                    )

                st.session_state[
                    "dataset"
                ] = clean_df

                st.session_state[
                    "bundle"
                ] = bundle

                st.success(
                    "✅ Train hoàn tất."
                )

                if train_and_push:
                    with st.spinner(
                        "Đang cập nhật GitHub..."
                    ):
                        pushed = (
                            push_bundle_to_github(
                                bundle
                            )
                        )

                    st.success(
                        "✅ Model + metrics đã "
                        "được commit lên GitHub."
                    )

                    st.link_button(
                        "🔗 Xem commit model",
                        pushed[
                            "model_commit_url"
                        ],
                        use_container_width=True,
                    )

                    st.link_button(
                        "🔗 Xem commit metrics",
                        pushed[
                            "metrics_commit_url"
                        ],
                        use_container_width=True,
                    )

            except Exception as e:
                st.exception(e)

    if st.session_state["bundle"] is not None:
        st.divider()

        bundle_bytes = serialize_bundle(
            st.session_state[
                "bundle"
            ]
        )

        st.download_button(
            "⬇️ Tải model bundle",
            data=bundle_bytes,
            file_name=(
                "purchase_model_bundle.joblib"
            ),
            mime=(
                "application/octet-stream"
            ),
            use_container_width=True,
        )


# ============================================================
# 9. TABS
# ============================================================

tabs = st.tabs(
    [
        "1️⃣ Problem & EDA",
        "2️⃣ Training",
        "3️⃣ Model Comparison",
        "4️⃣ Live Prediction",
        "5️⃣ Explain Prediction",
        "6️⃣ Code ↔ Math",
        "7️⃣ GitHub",
    ]
)


# ============================================================
# TAB 1 — EDA
# ============================================================

with tabs[0]:
    st.header("1. Bài toán và dữ liệu")

    st.markdown(
        r"""
### Bài toán

Dựa trên hành vi của một phiên truy cập website:

\[
Y = Revenue \in \{0,1\}
\]

- \(Y=1\): mua hàng.
- \(Y=0\): không mua.

Đây là bài toán **binary classification**.
"""
    )

    df = st.session_state[
        "dataset"
    ]

    if df is None:
        st.info(
            "Upload dataset ở sidebar."
        )
    else:
        try:
            preview, _ = normalize_dataset(
                df
            )
        except Exception:
            preview = df.copy()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Rows",
            f"{len(preview):,}",
        )

        c2.metric(
            "Features",
            max(
                0,
                preview.shape[1] - 1,
            ),
        )

        c3.metric(
            "Missing cells",
            int(
                preview.isna()
                .sum()
                .sum()
            ),
        )

        if TARGET in preview.columns:
            try:
                rate = float(
                    preview[
                        TARGET
                    ].astype(int).mean()
                )

                c4.metric(
                    "Purchase rate",
                    f"{rate*100:.2f}%",
                )
            except Exception:
                c4.metric(
                    "Purchase rate",
                    "N/A",
                )

        st.dataframe(
            preview.head(20),
            use_container_width=True,
        )

        if TARGET in preview.columns:
            target_counts = (
                preview[TARGET]
                .astype(str)
                .value_counts()
                .rename_axis(
                    "Revenue"
                )
                .reset_index(
                    name="Count"
                )
            )

            fig = px.bar(
                target_counts,
                x="Revenue",
                y="Count",
                text="Count",
                title=(
                    "Phân bố target Revenue"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.warning(
                "Target mất cân bằng → "
                "không chỉ nhìn Accuracy. "
                "Ưu tiên PR-AUC, Recall, F1."
            )


# ============================================================
# TAB 2 — TRAINING
# ============================================================

with tabs[1]:
    st.header("2. Training")

    bundle = st.session_state[
        "bundle"
    ]

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
- Validation: early stopping + chọn threshold.
- Test: đánh giá cuối.
"""
    )

    if bundle is None:
        st.info(
            "Bấm Train ở sidebar."
        )
    else:
        s = bundle[
            "dataset_summary"
        ]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Train",
            s["train"],
        )

        c2.metric(
            "Validation",
            s["validation"],
        )

        c3.metric(
            "Test",
            s["test"],
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "scale_pos_weight",
            (
                f"{bundle['scale_pos_weight']:.3f}"
            ),
        )

        c2.metric(
            "LightGBM best iteration",
            bundle[
                "lgbm_best_iteration"
            ],
        )


# ============================================================
# TAB 3 — COMPARISON
# ============================================================

with tabs[2]:
    st.header("3. Model Comparison")

    bundle = st.session_state[
        "bundle"
    ]

    if bundle is None:
        st.info(
            "Train model trước."
        )
    else:
        metrics = (
            pd.DataFrame(
                bundle["metrics"]
            )
            .T
            .reset_index()
            .rename(
                columns={
                    "index": "Model"
                }
            )
        )

        cols = [
            "Model",
            "pr_auc",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "threshold",
        ]

        table = metrics[
            cols
        ].copy()

        for col in cols[1:]:
            table[col] = (
                table[col]
                .astype(float)
                .round(4)
            )

        st.dataframe(
            table.sort_values(
                "pr_auc",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.success(
            "Model tốt nhất theo PR-AUC: "
            f"**{bundle['recommended_model']}**"
        )

        metric_name = st.selectbox(
            "Metric",
            [
                "pr_auc",
                "roc_auc",
                "f1",
                "recall",
                "precision",
                "accuracy",
            ],
        )

        fig = px.bar(
            metrics.sort_values(
                metric_name
            ),
            x=metric_name,
            y="Model",
            orientation="h",
            text=metric_name,
            title=(
                f"So sánh "
                f"{metric_name.upper()}"
            ),
        )

        fig.update_traces(
            texttemplate=(
                "%{text:.3f}"
            ),
            textposition="outside",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader(
            "LightGBM Feature Importance"
        )

        imp_mode = st.radio(
            "Importance",
            ["Gain", "Split"],
            horizontal=True,
        )

        imp = bundle[
            "lgbm_feature_importance"
        ].copy()

        if imp_mode == "Gain":
            plot_df = (
                imp.nlargest(
                    12,
                    "gain_pct",
                )
                .sort_values(
                    "gain_pct"
                )
            )

            fig = px.bar(
                plot_df,
                x="gain_pct",
                y="feature",
                orientation="h",
                text="gain_pct",
                title=(
                    "Gain importance"
                ),
            )

            fig.update_traces(
                texttemplate=(
                    "%{text:.1f}%"
                )
            )

            st.caption(
                "Gain ≠ hướng tác động."
            )

        else:
            plot_df = (
                imp.nlargest(
                    12,
                    "split",
                )
                .sort_values(
                    "split"
                )
            )

            fig = px.bar(
                plot_df,
                x="split",
                y="feature",
                orientation="h",
                text="split",
                title=(
                    "Split importance"
                ),
            )

            st.caption(
                "Split = số lần "
                "feature được dùng."
            )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# TAB 4 — LIVE PREDICTION
# ============================================================

with tabs[3]:
    st.header("4. Live Prediction")

    bundle = st.session_state[
        "bundle"
    ]

    if bundle is None:
        st.info(
            "Train model trước."
        )
    else:
        model_name = st.selectbox(
            "Model",
            list(
                bundle[
                    "models"
                ].keys()
            ),
            index=list(
                bundle[
                    "models"
                ].keys()
            ).index(
                bundle[
                    "recommended_model"
                ]
            ),
        )

        mode = st.radio(
            "Input mode",
            [
                "Simple Demo",
                "Full 17 Features",
            ],
            horizontal=True,
        )

        row = dict(
            bundle[
                "defaults"
            ]
        )

        st.info(
            "Simple Demo dùng median/mode "
            "của TRAIN cho feature không nhập."
        )

        with st.form(
            "prediction_form"
        ):
            if mode == "Simple Demo":
                c1, c2 = st.columns(2)

                with c1:
                    row[
                        "ProductRelated"
                    ] = st.number_input(
                        "ProductRelated",
                        min_value=0,
                        value=int(
                            bundle[
                                "defaults"
                            ][
                                "ProductRelated"
                            ]
                        ),
                        step=1,
                    )

                    row[
                        "ProductRelated_Duration"
                    ] = st.number_input(
                        "ProductRelated_Duration",
                        min_value=0.0,
                        value=float(
                            bundle[
                                "defaults"
                            ][
                                "ProductRelated_Duration"
                            ]
                        ),
                    )

                    row[
                        "PageValues"
                    ] = st.number_input(
                        "PageValues",
                        min_value=0.0,
                        value=float(
                            bundle[
                                "defaults"
                            ][
                                "PageValues"
                            ]
                        ),
                    )

                    row[
                        "BounceRates"
                    ] = st.number_input(
                        "BounceRates",
                        min_value=0.0,
                        value=float(
                            bundle[
                                "defaults"
                            ][
                                "BounceRates"
                            ]
                        ),
                        format="%.4f",
                    )

                    row[
                        "ExitRates"
                    ] = st.number_input(
                        "ExitRates",
                        min_value=0.0,
                        value=float(
                            bundle[
                                "defaults"
                            ][
                                "ExitRates"
                            ]
                        ),
                        format="%.4f",
                    )

                with c2:
                    for col in [
                        "Month",
                        "VisitorType",
                        "Weekend",
                        "Region",
                    ]:
                        levels = bundle[
                            "category_levels"
                        ][col]

                        default = bundle[
                            "defaults"
                        ][col]

                        idx = (
                            levels.index(
                                default
                            )
                            if default
                            in levels
                            else 0
                        )

                        row[col] = (
                            st.selectbox(
                                col,
                                levels,
                                index=idx,
                                key=(
                                    f"simple_{col}"
                                ),
                            )
                        )

            else:
                st.markdown(
                    "#### Numeric"
                )

                num_cols = st.columns(2)

                for i, col in enumerate(
                    bundle[
                        "numeric_features"
                    ]
                ):
                    default = float(
                        bundle[
                            "defaults"
                        ][col]
                    )

                    row[col] = (
                        num_cols[
                            i % 2
                        ].number_input(
                            col,
                            value=default,
                            key=(
                                f"num_{col}"
                            ),
                        )
                    )

                st.markdown(
                    "#### Categorical"
                )

                cat_cols = st.columns(2)

                for i, col in enumerate(
                    bundle[
                        "categorical_features"
                    ]
                ):
                    levels = bundle[
                        "category_levels"
                    ][col]

                    default = bundle[
                        "defaults"
                    ][col]

                    idx = (
                        levels.index(
                            default
                        )
                        if default
                        in levels
                        else 0
                    )

                    row[col] = (
                        cat_cols[
                            i % 2
                        ].selectbox(
                            col,
                            levels,
                            index=idx,
                            key=(
                                f"cat_{col}"
                            ),
                        )
                    )

            submitted = (
                st.form_submit_button(
                    "🔮 Predict",
                    use_container_width=True,
                )
            )

        if submitted:
            input_df = pd.DataFrame(
                [
                    [
                        row[col]
                        for col
                        in bundle[
                            "feature_order"
                        ]
                    ]
                ],
                columns=bundle[
                    "feature_order"
                ],
            )

            score, threshold, pred = (
                score_one(
                    model_name,
                    input_df,
                    bundle,
                )
            )

            st.session_state[
                "last_input"
            ] = input_df

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Model score — Mua",
                f"{score*100:.2f}%",
            )

            c2.metric(
                "Threshold",
                f"{threshold:.3f}",
            )

            c3.metric(
                "Prediction",
                (
                    "MUA"
                    if pred == 1
                    else "KHÔNG MUA"
                ),
            )

            st.progress(
                max(
                    0.0,
                    min(
                        1.0,
                        score,
                    ),
                )
            )

            st.warning(
                "Model score chưa mặc nhiên "
                "là calibrated probability."
            )

            render_purchase_demo(
                score=score,
                threshold=threshold,
                pred=pred,
                model_name=model_name,
                input_df=input_df,
                bundle=bundle,
            )


# ============================================================
# TAB 5 — EXPLAIN
# ============================================================

with tabs[4]:
    st.header(
        "5. Explain Prediction"
    )

    bundle = st.session_state[
        "bundle"
    ]

    if bundle is None:
        st.info(
            "Train model trước."
        )

    elif "last_input" not in st.session_state:
        st.info(
            "Predict một sample trước."
        )

    else:
        base, raw_score, contrib = (
            get_lgbm_contributions(
                st.session_state[
                    "last_input"
                ],
                bundle,
            )
        )

        score = sigmoid(
            raw_score
        )

        st.markdown(
            r"""
\[
F(x)
=
E[F(x)]
+
\sum_j \phi_j
\]

\[
score
=
\sigma(F(x))
\]

- \(\phi_j>0\): đẩy về Mua.
- \(\phi_j<0\): đẩy về Không mua.
"""
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Base raw score",
            f"{base:.4f}",
        )

        c2.metric(
            "Final raw score",
            f"{raw_score:.4f}",
        )

        c3.metric(
            "Sigmoid(raw score)",
            f"{score*100:.2f}%",
        )

        top = (
            contrib.head(10)
            .copy()
        )

        top[
            "direction"
        ] = np.where(
            top[
                "contribution"
            ] >= 0,
            "Đẩy về Mua",
            "Đẩy về Không mua",
        )

        top = top.sort_values(
            "contribution"
        )

        fig = px.bar(
            top,
            x="contribution",
            y="feature",
            orientation="h",
            color="direction",
            title=(
                "Local feature contribution"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "Đây là contribution lấy trực tiếp "
            "từ LightGBM, không phải if/else tự viết."
        )


# ============================================================
# TAB 6 — CODE ↔ MATH
# ============================================================

with tabs[5]:
    st.header(
        "6. Code ↔ Toán ↔ Ý nghĩa"
    )

    st.markdown(
        r"""
### Decision Tree

```python
criterion="gini"
max_depth=5
```

\[
Gini(S)
=
1-\sum_k p_k^2
\]

---

### Bagging

```python
bootstrap=True
n_estimators=120
```

\[
D_b
\sim
Bootstrap(D)
\]

\[
\hat y
=
mode(
h_1(x),...,h_B(x)
)
\]

---

### Random Forest

```python
max_features="sqrt"
```

\[
m
\approx
\sqrt{p}
\]

Random Forest = Bagging + random feature subset.

---

### LightGBM

```python
learning_rate=0.03
reg_lambda=1.0
scale_pos_weight=...
```

Boosting:

\[
F_m(x)
=
F_{m-1}(x)
+
\eta f_m(x)
\]

với:

\[
\eta=0.03
\]

Binary loss:

\[
L_i
=
-w_i
[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
]
\]

`scale_pos_weight` là class weighting.

`reg_lambda` mới là L2 regularization:

\[
\lambda
\]
"""
    )


# ============================================================
# TAB 7 — GITHUB
# ============================================================

with tabs[6]:
    st.header(
        "7. GitHub Auto Update"
    )

    st.markdown(
        f"""
Repo mặc định:

```text
{DEFAULT_GITHUB_REPO}
```

Branch:

```text
{DEFAULT_GITHUB_BRANCH}
```

Sau khi bấm:

```text
🚀 Train + cập nhật GitHub
```

app sẽ tự cập nhật:

```text
{MODEL_PATH_IN_REPO}
```

và:

```text
{METRICS_PATH_IN_REPO}
```
"""
    )

    st.markdown(
        """
### Streamlit Secrets

Trong **Streamlit Cloud → App → Settings → Secrets**:

```toml
GITHUB_TOKEN = "YOUR_NEW_TOKEN"
GITHUB_REPO = "phatpt1/math4ai"
GITHUB_BRANCH = "main"
```

Không lưu token trực tiếp trong `app.py`.
"""
    )

    st.warning(
        "Token GitHub đã từng được dán trực tiếp vào chat/source "
        "nên nên revoke token cũ và tạo token mới. "
        "Token mới chỉ cần Contents: Read and write cho đúng repo."
    )

    if st.session_state["bundle"] is not None:
        st.subheader(
            "Thông tin model hiện tại"
        )

        st.json(
            {
                "trained_at_utc": st.session_state[
                    "bundle"
                ].get(
                    "trained_at_utc"
                ),
                "recommended_model": st.session_state[
                    "bundle"
                ][
                    "recommended_model"
                ],
                "repo_model_path": MODEL_PATH_IN_REPO,
                "repo_metrics_path": METRICS_PATH_IN_REPO,
            }
        )
