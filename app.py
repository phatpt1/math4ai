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
    """
    Train 4 models trên cùng một split.

    Quy tắc khoa học của project:
    - TRAIN: học tham số model.
    - VALIDATION: chọn threshold + chọn model thắng cuộc.
    - TEST: chỉ dùng báo cáo cuối, KHÔNG dùng để chọn model.

    Model được chọn theo VALIDATION PR-AUC. Nếu bằng nhau, dùng F1 rồi
    Recall làm tie-breaker.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    models = {}
    validation_results = {}
    test_results = {}

    # 1) Decision Tree
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

    # 2) Bagging
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

    # 3) Random Forest
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

        # VALIDATION: chọn threshold + dùng để xếp hạng model
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, val_prob)
        validation_results[name] = evaluate_model(y_val, val_prob, threshold)

        # TEST: chỉ báo cáo với threshold đã khóa từ validation
        test_prob = model.predict_proba(X_test)[:, 1]
        test_results[name] = evaluate_model(y_test, test_prob, threshold)
        models[name] = model

    # 4) LightGBM / Gradient Boosting
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
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    val_prob = lgbm.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_prob)
    validation_results["LightGBM"] = evaluate_model(y_val, val_prob, threshold)

    test_prob = lgbm.predict_proba(X_test)[:, 1]
    test_results["LightGBM"] = evaluate_model(y_test, test_prob, threshold)
    models["LightGBM"] = lgbm

    # LightGBM global importance: chỉ để chẩn đoán, không dùng chọn model
    gain = lgbm.booster_.feature_importance(importance_type="gain")
    split = lgbm.booster_.feature_importance(importance_type="split")
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

    # Chọn model bằng VALIDATION, tuyệt đối không dùng TEST để chọn
    ranking = sorted(
        validation_results.items(),
        key=lambda kv: (
            kv[1]["pr_auc"],
            kv[1]["f1"],
            kv[1]["recall"],
        ),
        reverse=True,
    )
    recommended_model = ranking[0][0]

    bundle = {
        "models": models,
        "metrics": test_results,  # TEST metrics
        "validation_metrics": validation_results,
        "recommended_model": recommended_model,
        "selection_rule": (
            "Chọn model theo Validation PR-AUC; "
            "nếu bằng nhau dùng F1 rồi Recall."
        ),
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
        "lgbm_best_iteration": int(lgbm.best_iteration_ or lgbm.n_estimators),
        "lgbm_feature_importance": importance,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
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


def get_what_if_sensitivity(model_name, row_df, bundle):
    """Model-agnostic local what-if explanation for all 4 models."""
    base_score, threshold, pred = score_one(model_name, row_df, bundle)
    rows = []
    for col in bundle["feature_order"]:
        changed = row_df.copy()
        changed.loc[changed.index[0], col] = bundle["defaults"][col]
        changed_score, _, _ = score_one(model_name, changed, bundle)
        delta = float(base_score - changed_score)
        rows.append({
            "feature": col,
            "current_value": row_df.iloc[0][col],
            "baseline_value": bundle["defaults"][col],
            "score_delta": delta,
            "abs_delta": abs(delta),
        })
    return (
        base_score, threshold, pred,
        pd.DataFrame(rows).sort_values("abs_delta", ascending=False),
    )


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
        "validation_metrics": bundle.get(
            "validation_metrics"
        ),
        "selection_rule": bundle.get(
            "selection_rule"
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

    # Native local contribution chỉ đúng khi prediction hiện tại dùng LightGBM
    if model_name == "LightGBM" and "LightGBM" in bundle["models"]:
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
    else:
        st.caption(
            "Prediction hiện tại không dùng LightGBM. Sang tab 5 để xem "
            "What-if sensitivity cho đúng model đang chọn."
        )

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

    bundle = st.session_state["bundle"]

    if bundle is None:
        st.info("Train 4 model trước để có dữ liệu so sánh.")
    else:
        st.markdown(
            """
### Logic chọn model

Không phải **train xong rồi nhìn Test để chọn**. Flow đúng là:

**Train 4 model → Validation chọn model + threshold → khóa lựa chọn → Test báo cáo cuối.**

Vì `Revenue=True` là lớp thiểu số, tiêu chí chính là **PR-AUC trên Validation**.
Nếu PR-AUC bằng nhau, dùng **F1**, sau đó **Recall** để phá hòa.
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
            "Model", "pr_auc", "recall", "f1",
            "precision", "roc_auc", "accuracy", "threshold"
        ]

        st.subheader("A. Validation — dùng để CHỌN model")
        val_table = val_metrics[cols].copy()
        for col in cols[1:]:
            val_table[col] = val_table[col].astype(float).round(4)
        st.dataframe(
            val_table.sort_values(
                ["pr_auc", "f1", "recall"],
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        winner = bundle["recommended_model"]
        st.success(
            f"🏆 Model được chọn: **{winner}** — "
            f"dựa trên Validation, không nhìn trước Test."
        )
        st.caption(
            bundle.get("selection_rule", "Chọn theo Validation PR-AUC.")
        )

        winner_row = val_metrics[val_metrics["Model"] == winner].iloc[0]
        st.markdown(
            f"**Vì sao {winner} thắng?** Validation PR-AUC = "
            f"**{float(winner_row['pr_auc']):.4f}**, "
            f"F1 = **{float(winner_row['f1']):.4f}**, "
            f"Recall = **{float(winner_row['recall']):.4f}**. "
            "Theo rule của project, PR-AUC được xét trước."
        )

        st.subheader("B. Test — chỉ báo cáo cuối")
        test_table = test_metrics[cols].copy()
        for col in cols[1:]:
            test_table[col] = test_table[col].astype(float).round(4)
        st.dataframe(
            test_table.sort_values("pr_auc", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        metric_name = st.selectbox(
            "Vẽ metric nào?",
            ["pr_auc", "recall", "f1", "precision", "roc_auc", "accuracy"],
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
            "Cách đọc: PR-AUC = tiêu chí chọn chính; Recall = bắt được bao nhiêu "
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
            st.session_state[
                "last_model_name"
            ] = model_name

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
    st.header("5. Explain Prediction")

    bundle = st.session_state["bundle"]

    if bundle is None:
        st.info("Train/load model trước.")
    elif "last_input" not in st.session_state:
        st.info("Hãy sang Live Prediction và Predict một sample trước.")
    else:
        model_name = st.session_state.get(
            "last_model_name", bundle["recommended_model"]
        )
        input_df = st.session_state["last_input"]

        st.markdown(
            f"""
### Đang giải thích prediction của **{model_name}**

Có 2 mức giải thích:

1. **What-if sensitivity** — dùng được cho cả 4 model: thay từng feature về baseline của TRAIN và xem score đổi bao nhiêu.
2. **Native LightGBM contribution** — chỉ khi model là LightGBM: phân rã raw score thành base value + đóng góp feature.
            """
        )

        score, threshold, pred, sensitivity = get_what_if_sensitivity(
            model_name, input_df, bundle
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Model score", f"{score*100:.2f}%")
        c2.metric("Threshold", f"{threshold:.3f}")
        c3.metric("Prediction", "MUA" if pred else "KHÔNG MUA")

        st.subheader("A. What-if sensitivity — dễ hiểu, dùng cho mọi model")
        st.markdown(
            """
Với từng feature, app hỏi: **nếu giữ mọi thứ như cũ nhưng thay riêng feature này bằng giá trị baseline của TRAIN thì score thay đổi bao nhiêu?**

- `score_delta > 0`: giá trị hiện tại đang đẩy score Mua **cao hơn baseline**.
- `score_delta < 0`: giá trị hiện tại đang kéo score Mua **thấp hơn baseline**.
- Các delta này **không cộng lại chính xác thành score**; đây là phân tích what-if, không phải SHAP.
            """
        )
        top = sensitivity.head(10).copy().sort_values("score_delta")
        top["direction"] = np.where(
            top["score_delta"] >= 0,
            "Đẩy score cao hơn baseline",
            "Kéo score thấp hơn baseline",
        )
        fig = px.bar(
            top, x="score_delta", y="feature", orientation="h",
            color="direction", title=f"What-if sensitivity — {model_name}"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            sensitivity.head(10)[
                ["feature", "current_value", "baseline_value", "score_delta"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        if model_name == "LightGBM":
            st.subheader("B. Native LightGBM contribution — phân rã raw score")
            base, raw_score, contrib = get_lgbm_contributions(input_df, bundle)
            probability_like_score = sigmoid(raw_score)
            st.markdown(
                r"""
LightGBM có thể trả contribution trực tiếp:

\[
F(x)=BaseValue+\sum_j \phi_j
\]

Sau đó:

\[
score=\sigma(F(x))=\frac{1}{1+e^{-F(x)}}
\]

- \(\phi_j>0\): đẩy raw score về phía Mua.
- \(\phi_j<0\): đẩy raw score về phía Không mua.
                """
            )
            d1, d2, d3 = st.columns(3)
            d1.metric("Base raw score", f"{base:.4f}")
            d2.metric("Final raw score", f"{raw_score:.4f}")
            d3.metric("Sigmoid(raw)", f"{probability_like_score*100:.2f}%")

            topc = contrib.head(10).copy().sort_values("contribution")
            topc["direction"] = np.where(
                topc["contribution"] >= 0,
                "Đẩy về Mua",
                "Đẩy về Không mua",
            )
            fig2 = px.bar(
                topc, x="contribution", y="feature", orientation="h",
                color="direction", title="Native LightGBM local contribution"
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                "Đây là contribution lấy trực tiếp từ LightGBM, "
                "không phải rule if/else tự viết."
            )
        else:
            st.info(
                "Native additive contribution trong app hiện chỉ có cho LightGBM. "
                "Với model này, phần A vẫn giải thích prediction bằng phân tích what-if nhất quán."
            )

# ============================================================
# TAB 6 — CODE ↔ MATH
# ============================================================

with tabs[5]:
    st.header("6. Code ↔ Math — học từ đầu, không học thuộc công thức")

    st.markdown(
        """
Trang này trả lời một câu duy nhất: **mỗi dòng code trong project đang đại diện cho ý tưởng toán học nào?**

Hãy đọc theo flow:

**Dữ liệu → Vector x,y → Decision Tree → Bagging → Random Forest → LightGBM → Score → Threshold → Mua/Không mua**
        """
    )

    math_tabs = st.tabs(
        [
            "0️⃣ Bản đồ tổng",
            "1️⃣ Data → Vector",
            "2️⃣ Decision Tree",
            "3️⃣ Bagging",
            "4️⃣ Random Forest",
            "5️⃣ LightGBM",
            "6️⃣ Prediction & Metrics",
        ]
    )

    with math_tabs[0]:
        st.subheader("Bản đồ toàn bộ project")
        st.code(
            """Dataset (12,330 sessions)
        ↓
Preprocessing
        ↓
Split: Train 70% | Validation 15% | Test 15%
        ↓
Train 4 models
  ├─ Decision Tree
  ├─ Bagging
  ├─ Random Forest
  └─ LightGBM
        ↓
Validation: chọn threshold + chọn model theo PR-AUC
        ↓
Khóa model thắng cuộc
        ↓
Test: báo cáo cuối
        ↓
New customer → score → threshold → Mua / Không mua""",
            language="text",
        )
        st.success(
            "Điểm phải nhớ: Test KHÔNG dùng để chọn model. "
            "Validation mới dùng để chọn; Test chỉ kiểm tra cuối."
        )

    with math_tabs[1]:
        st.subheader("1. Một dòng dữ liệu biến thành vector như thế nào?")
        st.markdown(
            r"""
Một phiên truy cập có 17 feature. Ta viết ngắn gọn:

\[
x=[x_1,x_2,\ldots,x_{17}]
\]

Target:

\[
y=Revenue\in\{0,1\}
\]

- \(y=1\): khách mua.
- \(y=0\): khách không mua.

Ví dụ: `PageValues`, `BounceRates`, `Month`, `VisitorType`... là các thành phần của \(x\).
            """
        )
        st.code(
            'X = df.drop(columns=["Revenue"])\n'
            'y = df["Revenue"]\n\n'
            '# 70 / 15 / 15\n'
            'X_train, X_val, X_test, ... = split_data(df)',
            language="python",
        )
        st.info("Code đang làm đúng phép tách toán học: X = đầu vào, y = nhãn cần học.")

    with math_tabs[2]:
        st.subheader("2. Decision Tree — model đang chọn câu hỏi tốt nhất")
        st.markdown(
            r"""
Giả sử một node có 100 khách: 85 Không mua và 15 Mua. Node này còn lẫn hai lớp nên chưa "thuần".

Gini impurity:

\[
Gini(S)=1-\sum_k p_k^2
\]

Với bài toán 2 lớp:

\[
Gini(S)=1-p_0^2-p_1^2
\]

Tree thử nhiều câu hỏi kiểu `PageValues <= 12.5 ?` và chọn split làm impurity sau chia nhỏ nhất, hay tương đương làm **Gini giảm nhiều nhất**:

\[
\Delta Gini = Gini(parent)-\left(\frac{N_L}{N}Gini(L)+\frac{N_R}{N}Gini(R)\right)
\]
            """
        )
        st.code(
            'DecisionTreeClassifier(\n'
            '    criterion="gini",   # dùng Gini impurity\n'
            '    max_depth=5,        # giới hạn độ sâu\n'
            '    min_samples_leaf=20,\n'
            '    class_weight="balanced"\n'
            ')',
            language="python",
        )
        st.success(
            "Cách nhớ: Decision Tree = liên tục hỏi câu hỏi làm hai nhóm Mua/Không mua tách nhau rõ hơn."
        )

    with math_tabs[3]:
        st.subheader("3. Bagging — nhiều cây độc lập rồi bỏ phiếu")
        st.markdown(
            r"""
Một Decision Tree có thể thay đổi mạnh khi dữ liệu train thay đổi. Bagging giảm vấn đề này bằng cách tạo nhiều bộ dữ liệu bootstrap:

\[
D_b\sim Bootstrap(D),\quad b=1,2,\ldots,B
\]

Mỗi \(D_b\) train một cây \(h_b\). Với classification, prediction cuối được tổng hợp từ các cây:

\[
\hat y = mode\{h_1(x),h_2(x),\ldots,h_B(x)\}
\]

`bootstrap=True` nghĩa là lấy mẫu **có hoàn lại**. Một dòng có thể xuất hiện nhiều lần, dòng khác có thể không xuất hiện trong một bootstrap sample.
            """
        )
        st.code(
            'BaggingClassifier(\n'
            '    estimator=DecisionTreeClassifier(...),\n'
            '    n_estimators=120,   # B = 120 cây\n'
            '    bootstrap=True     # D_b ~ Bootstrap(D)\n'
            ')',
            language="python",
        )
        st.success(
            "Cách nhớ: Tree dễ dao động → train nhiều Tree trên nhiều bootstrap sample → vote → giảm variance."
        )

    with math_tabs[4]:
        st.subheader("4. Random Forest — Bagging + random feature")
        p_features = 17
        approx_features = math.sqrt(p_features)
        st.markdown(
            rf"""
Random Forest vẫn dùng nhiều cây + bootstrap giống Bagging, nhưng tại mỗi split nó chỉ cho cây nhìn **một tập con feature ngẫu nhiên**.

Nếu có \(p={p_features}\) feature và dùng `max_features="sqrt"`:

\[
m\approx\sqrt{{p}}=\sqrt{{{p_features}}}\approx {approx_features:.2f}
\]

Tức là tại một split, model thường chỉ xét khoảng **4 feature**, thay vì luôn xét cả 17.

Mục tiêu: làm các cây **bớt giống nhau**, tức giảm correlation giữa các cây.
            """
        )
        st.code(
            'RandomForestClassifier(\n'
            '    n_estimators=250,\n'
            '    bootstrap=True,       # giống Bagging\n'
            '    max_features="sqrt", # random feature subset\n'
            '    max_depth=8\n'
            ')',
            language="python",
        )
        st.success("Cách nhớ: Random Forest = Bagging + random feature selection.")

    with math_tabs[5]:
        st.subheader("5. LightGBM — các cây học tuần tự để cải thiện lỗi")
        st.markdown(
            r"""
Khác Bagging/Random Forest, Boosting không xây các cây hoàn toàn độc lập. Nó xây tuần tự:

\[
F_m(x)=F_{m-1}(x)+\eta f_m(x)
\]

Trong đó:

- \(F_{m-1}(x)\): ensemble trước khi thêm cây mới.
- \(f_m(x)\): cây mới.
- \(\eta\): learning rate, điều khiển cây mới đóng góp mạnh đến đâu.

Với binary classification, một dạng loss phổ biến là binary log-loss:

\[
L_i=-w_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
\]

LightGBM dùng thông tin đạo hàm của loss:

\[
g_i=\frac{\partial L}{\partial F(x_i)},\qquad
h_i=\frac{\partial^2 L}{\partial F(x_i)^2}
\]

Gradient cho biết hướng loss thay đổi; Hessian cho biết độ cong. LightGBM dùng chúng để tìm các split/cây giúp objective tốt hơn.
            """
        )
        st.code(
            'LGBMClassifier(\n'
            '    n_estimators=500,       # tối đa M vòng boosting\n'
            '    learning_rate=0.03,     # eta = 0.03\n'
            '    max_depth=5,\n'
            '    num_leaves=25,\n'
            '    reg_lambda=1.0,         # lambda: L2 regularization\n'
            '    scale_pos_weight=...    # trọng số lớp Mua\n'
            ')',
            language="python",
        )
        if st.session_state["bundle"] is not None:
            b = st.session_state["bundle"]
            st.metric(
                "scale_pos_weight hiện tại",
                f"{b['scale_pos_weight']:.3f}",
                help="N_negative / N_positive trên TRAIN",
            )
        st.warning(
            "Đừng nhầm: scale_pos_weight là class weighting; "
            "reg_lambda mới liên hệ với λ của L2 regularization."
        )
        st.success(
            "Cách nhớ: LightGBM = cây sau được thêm vào để cải thiện ensemble hiện tại; "
            "learning_rate quyết định bước cải thiện lớn hay nhỏ."
        )

    with math_tabs[6]:
        st.subheader("6. Từ score đến quyết định Mua / Không mua")
        st.markdown(
            r"""
Sau khi model tạo score \(s(x)\), ta so với threshold \(t\):

\[
\hat y=
\begin{cases}
1 & s(x)\ge t\\
0 & s(x)<t
\end{cases}
\]

Trong project, threshold được chọn trên **Validation** bằng F1 tốt nhất, sau đó khóa lại và mang sang Test.

### Ba metric chính

**Recall** — trong khách thực sự Mua, bắt được bao nhiêu:

\[
Recall=\frac{TP}{TP+FN}
\]

**F1** — cân bằng Precision và Recall:

\[
F1=2\frac{Precision\cdot Recall}{Precision+Recall}
\]

**PR-AUC** — tóm tắt quan hệ Precision–Recall qua nhiều threshold; đây là tiêu chí chính để chọn model vì lớp Mua là lớp thiểu số.
            """
        )
        st.code(
            '# threshold chọn trên VALIDATION\n'
            'threshold = choose_threshold(y_val, val_prob)\n\n'
            '# model winner chọn bằng VALIDATION PR-AUC\n'
            'recommended_model = max(\n'
            '    validation_results,\n'
            '    key=lambda name: (\n'
            '        validation_results[name]["pr_auc"],\n'
            '        validation_results[name]["f1"],\n'
            '        validation_results[name]["recall"],\n'
            '    ),\n'
            ')\n\n'
            '# TEST chỉ dùng báo cáo sau khi đã khóa model + threshold',
            language="python",
        )
        st.error(
            "Sai về phương pháp: dùng Test để chọn model hoặc tune threshold. "
            "Đúng: Validation chọn; Test chỉ báo cáo cuối."
        )

    with st.expander("📘 Từ điển 12 từ phải nhớ"):
        st.markdown(
            """
- **Feature**: biến đầu vào.
- **Target/Label**: nhãn cần dự đoán.
- **Train**: dữ liệu để model học.
- **Validation**: dữ liệu để chọn model/threshold/hyperparameter.
- **Test**: dữ liệu kiểm tra cuối.
- **Split**: phép chia node của cây.
- **Leaf**: node cuối của cây.
- **Bootstrap**: lấy mẫu ngẫu nhiên có hoàn lại.
- **Ensemble**: tổ hợp nhiều model.
- **Boosting**: học tuần tự để cải thiện lỗi.
- **Threshold**: ngưỡng biến score thành 0/1.
- **PR-AUC**: chất lượng Precision–Recall qua nhiều threshold.
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
