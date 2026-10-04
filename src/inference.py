import math

import numpy as np
import pandas as pd


def restore_lgbm_categories(df, bundle):
    df = df.copy()

    for col in bundle["categorical_features"]:
        df[col] = pd.Categorical(
            df[col],
            categories=bundle["category_levels"][col],
        )

    return df


def score_one(model_name, row_df, bundle):
    model = bundle["models"][model_name]

    x = row_df.copy()

    if model_name == "LightGBM":
        x = restore_lgbm_categories(x, bundle)

    score = float(model.predict_proba(x)[0, 1])

    threshold = float(
        bundle.get("selected_thresholds", {}).get(
            model_name,
            bundle.get("validation_metrics", {}).get(
                model_name,
                bundle["metrics"][model_name],
            )["threshold"],
        )
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

    x = restore_lgbm_categories(row_df, bundle)

    contrib = model.booster_.predict(x, pred_contrib=True)[0]

    feature_values = np.asarray(contrib[:-1], dtype=float)

    base = float(contrib[-1])
    raw_score = base + float(feature_values.sum())

    contrib_df = pd.DataFrame(
        {
            "feature": bundle["feature_order"],
            "contribution": feature_values,
            "abs_contribution": np.abs(feature_values),
        }
    ).sort_values("abs_contribution", ascending=False)

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