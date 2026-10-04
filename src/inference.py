import pandas as pd


def restore_lgbm_categories(df, bundle):
    df = df.copy()
    for col in bundle["categorical_features"]:
        df[col] = pd.Categorical(df[col], categories=bundle["category_levels"][col])
    return df


def score_one(model_name, row_df, bundle):
    model = bundle["models"][model_name]

    x = row_df.copy()
    if model_name == "LightGBM":
        x = restore_lgbm_categories(x, bundle)

    score = float(model.predict_proba(x)[0, 1])
    threshold = float(bundle["selected_thresholds"][model_name])
    pred = int(score >= threshold)

    return score, threshold, pred


def get_what_if_sensitivity(model_name, row_df, bundle):
    """Model-agnostic local what-if explanation for all 4 models."""
    base_score, threshold, pred = score_one(model_name, row_df, bundle)

    rows = []
    for col in bundle["feature_order"]:
        changed = row_df.copy()
        changed.loc[changed.index[0], col] = bundle["defaults"][col]
        changed_score, _, _ = score_one(model_name, changed, bundle)
        delta = float(base_score - changed_score)
        rows.append(
            {
                "feature": col,
                "current_value": row_df.iloc[0][col],
                "baseline_value": bundle["defaults"][col],
                "score_delta": delta,
                "abs_delta": abs(delta),
            }
        )

    return (
        base_score,
        threshold,
        pred,
        pd.DataFrame(rows).sort_values("abs_delta", ascending=False),
    )