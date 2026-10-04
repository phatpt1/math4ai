from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)
from src.data import get_defaults, split_data
from src.evaluation import choose_threshold, evaluate_model


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


def train_everything(df, category_levels):
    """
    Train 4 models on the same split.

    - TRAIN: fit model parameters.
    - VALIDATION: pick threshold and select the winning model.
    - TEST: final report only, never used for selection.

    Winner is ranked by validation AP, then F1, then Recall as tie-breakers.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    models = {}
    validation_results = {}
    test_results = {}

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
        validation_results[name] = evaluate_model(y_val, val_prob, threshold)

        # Test uses the threshold locked on validation
        test_prob = model.predict_proba(X_test)[:, 1]
        test_results[name] = evaluate_model(y_test, test_prob, threshold)
        models[name] = model

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
        subsample=0.9,  # inactive unless subsample_freq/bagging_freq > 0
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

    # Diagnostic only, not used for model selection
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

    ranking = sorted(
        validation_results.items(),
        key=lambda kv: (
            kv[1]["ap"],
            kv[1]["f1"],
            kv[1]["recall"],
        ),
        reverse=True,
    )
    recommended_model = ranking[0][0]

    bundle = {
        "models": models,
        "metrics": test_results,
        "validation_metrics": validation_results,
        "selected_thresholds": {
            name: float(metrics["threshold"])
            for name, metrics in validation_results.items()
        },
        "recommended_model": recommended_model,
        "selection_rule": (
            "Chọn model theo Validation AP; "
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


def normalize_legacy_bundle_metrics(bundle):
    """Backward compatibility for bundles saved with `pr_auc` instead of `ap`."""
    if bundle is None:
        return None

    for bucket_name in ("metrics", "validation_metrics"):
        bucket = bundle.get(bucket_name, {})
        for _, m in bucket.items():
            if "ap" not in m and "pr_auc" in m:
                m["ap"] = m["pr_auc"]

    if "selected_thresholds" not in bundle:
        source = bundle.get("validation_metrics", {})
        bundle["selected_thresholds"] = {
            name: float(metrics["threshold"])
            for name, metrics in source.items()
            if isinstance(metrics, dict) and "threshold" in metrics
        }

    rule = bundle.get("selection_rule")
    if isinstance(rule, str):
        bundle["selection_rule"] = rule.replace("PR-AUC", "AP")

    return bundle