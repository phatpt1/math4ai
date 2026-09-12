"""
train_models.py
===============
Train và so sánh 4 họ mô hình:
1) Decision Tree
2) Bagging
3) Random Forest
4) LightGBM Gradient Boosting

Thiết kế để phục vụ 3 mục tiêu:
- hiểu bài toán,
- mapping code <-> toán,
- demo tường minh trên Streamlit.

QUAN TRỌNG:
- Validation dùng để chọn decision threshold.
- Test chỉ dùng để báo cáo cuối cùng.
- LightGBM dùng native categorical.
- Các model sklearn dùng OneHotEncoder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

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


RANDOM_STATE = 42
TARGET = "Revenue"

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


def make_one_hot_encoder():
    """Tương thích nhiều phiên bản scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_bagging(base_tree):
    """Tương thích sklearn mới/cũ: estimator vs base_estimator."""
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


def prepare_data(csv_file: str):
    df = pd.read_csv(csv_file)

    if TARGET not in df.columns:
        raise ValueError(f"Không tìm thấy target '{TARGET}'.")

    # bool -> 0/1
    df[TARGET] = df[TARGET].astype(int)

    # Lưu category levels trước khi split để train/val/test cùng schema.
    category_levels = {}
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Thiếu categorical feature: {col}")
        if df[col].isna().any():
            # Với cột số categorical, đổi sang object trước khi fill chuỗi.
            df[col] = df[col].astype("object").where(df[col].notna(), "Missing")
        df[col] = df[col].astype("category")
        category_levels[col] = list(df[col].cat.categories)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 70/15/15: train / validation / test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    val_fraction_of_train_val = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_of_train_val,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    return df, X_train, X_val, X_test, y_train, y_val, y_test, category_levels


def choose_threshold(y_true, prob):
    """
    Chọn threshold trên VALIDATION bằng F1 tối đa.
    Không dùng test để tune threshold.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    scores = [
        f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def evaluate(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def get_defaults(X_train):
    """
    Baseline cho Simple Demo:
    - numeric: median của train
    - categorical: mode của train
    """
    defaults = {}

    for col in NUMERIC_FEATURES:
        defaults[col] = float(pd.to_numeric(X_train[col]).median())

    for col in CATEGORICAL_FEATURES:
        mode = X_train[col].mode(dropna=True)
        defaults[col] = mode.iloc[0] if len(mode) else None

    return defaults


def main():
    csv_file = os.environ.get("SHOPPERS_CSV", "online_shoppers.csv")
    output_file = os.environ.get("MODEL_BUNDLE", "purchase_model_bundle.joblib")

    if not Path(csv_file).exists():
        raise FileNotFoundError(
            f"Không tìm thấy '{csv_file}'. "
            "Đặt CSV cùng thư mục hoặc set biến môi trường SHOPPERS_CSV."
        )

    print("1) Load + chuẩn hóa schema...")
    (
        df,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        category_levels,
    ) = prepare_data(csv_file)

    print(f"   Dataset: {df.shape[0]} rows, {df.shape[1]-1} features")
    print(
        f"   Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    # -----------------------------
    # Preprocessor cho sklearn tree ensembles
    # -----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    # 1) Decision Tree
    tree = Pipeline(
        [
            ("prep", preprocessor),
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
    bag_base_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    bagging = Pipeline(
        [
            ("prep", preprocessor),
            ("model", make_bagging(bag_base_tree)),
        ]
    )

    # 3) Random Forest
    forest = Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=10,
                    max_features="sqrt",
                    class_weight="balanced",
                    bootstrap=True,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("2) Train Decision Tree / Bagging / Random Forest...")
    sklearn_models = {
        "Decision Tree": tree,
        "Bagging": bagging,
        "Random Forest": forest,
    }

    trained_models = {}
    results = {}

    for name, model in sklearn_models.items():
        print(f"   - {name}")
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        threshold, _ = choose_threshold(y_val, val_prob)

        test_prob = model.predict_proba(X_test)[:, 1]
        results[name] = evaluate(y_test, test_prob, threshold)
        trained_models[name] = model

    # -----------------------------
    # 4) LightGBM Gradient Boosting
    # -----------------------------
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive

    print("3) Train LightGBM...")
    print(f"   scale_pos_weight = {scale_pos_weight:.4f}")

    lgbm = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=500,          # upper bound; early stopping chọn số cây thực tế
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
    lgb_threshold, _ = choose_threshold(y_val, val_prob)

    test_prob = lgbm.predict_proba(X_test)[:, 1]
    results["LightGBM"] = evaluate(y_test, test_prob, lgb_threshold)
    trained_models["LightGBM"] = lgbm

    # -----------------------------
    # Importance LightGBM
    # -----------------------------
    gain = lgbm.booster_.feature_importance(importance_type="gain")
    split = lgbm.booster_.feature_importance(importance_type="split")
    names = lgbm.booster_.feature_name()

    gain_total = float(np.sum(gain)) or 1.0
    importance = pd.DataFrame(
        {
            "feature": names,
            "gain": gain.astype(float),
            "gain_pct": (gain / gain_total * 100).astype(float),
            "split": split.astype(int),
        }
    ).sort_values("gain", ascending=False)

    # -----------------------------
    # Model ranking
    # Primary metric = PR-AUC do imbalance
    # -----------------------------
    ranking = sorted(
        results.items(),
        key=lambda kv: kv[1]["pr_auc"],
        reverse=True,
    )
    recommended_model = ranking[0][0]

    defaults = get_defaults(X_train)

    dataset_summary = {
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1] - 1),
        "positive": int(df[TARGET].sum()),
        "negative": int((1 - df[TARGET]).sum()),
        "positive_rate": float(df[TARGET].mean()),
        "split": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
    }

    bundle = {
        "models": trained_models,
        "metrics": results,
        "recommended_model": recommended_model,
        "target": TARGET,
        "feature_order": list(X_train.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_levels": category_levels,
        "defaults": defaults,
        "dataset_summary": dataset_summary,
        "lgbm_feature_importance": importance,
        "lgbm_best_iteration": int(lgbm.best_iteration_ or lgbm.n_estimators),
        "scale_pos_weight": float(scale_pos_weight),
        "notes": {
            "probability_warning": (
                "LightGBM được train với scale_pos_weight nên predict_proba "
                "nên được xem là model score nếu chưa calibration."
            ),
            "pagevalues_warning": (
                "PageValues là feature rất mạnh trong bộ dữ liệu này. "
                "Phải xác nhận feature có sẵn tại đúng thời điểm business muốn prediction."
            ),
        },
    }

    joblib.dump(bundle, output_file, compress=3)

    print("\n4) KẾT QUẢ TEST")
    result_df = pd.DataFrame(results).T[
        ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy", "threshold"]
    ].sort_values("pr_auc", ascending=False)
    print(result_df.round(4))

    print("\nTop LightGBM features theo GAIN")
    print(
        importance[["feature", "gain_pct", "split"]]
        .head(10)
        .round({"gain_pct": 2})
        .to_string(index=False)
    )

    print(f"\nRecommended model theo PR-AUC: {recommended_model}")
    print(f"Saved bundle: {output_file}")


if __name__ == "__main__":
    main()
