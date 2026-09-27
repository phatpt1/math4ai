"""
CLI training entry point.

Usage:
    python train_models.py

Env vars (optional):
    SHOPPERS_CSV  - path to dataset (default: online_shoppers.csv)
    MODEL_BUNDLE  - output bundle path (default: models/purchase_model_bundle.joblib)
"""

import os
from pathlib import Path

import joblib
import pandas as pd

from src.config import METRICS_PATH_IN_REPO, MODEL_PATH_IN_REPO
from src.data import normalize_dataset
from src.models import train_everything
from src.storage import make_metrics_json

METRIC_COLUMNS = ["ap", "roc_auc", "precision", "recall", "f1", "accuracy", "threshold"]


def main():
    csv_file = Path(os.environ.get("SHOPPERS_CSV", "online_shoppers.csv"))
    model_path = Path(os.environ.get("MODEL_BUNDLE", MODEL_PATH_IN_REPO))
    metrics_path = Path(METRICS_PATH_IN_REPO)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Dataset '{csv_file}' not found. "
            "Place the CSV in the project root or set SHOPPERS_CSV."
        )

    print("1) Load + normalize dataset...")
    df, category_levels = normalize_dataset(pd.read_csv(csv_file))
    print(f"   Dataset: {len(df)} rows, {df.shape[1] - 1} features")

    print("2) Train Decision Tree / Bagging / Random Forest / LightGBM...")
    bundle = train_everything(df, category_levels)

    s = bundle["dataset_summary"]
    print(f"   Split: train={s['train']}, val={s['validation']}, test={s['test']}")
    print(f"   scale_pos_weight = {bundle['scale_pos_weight']:.4f}")
    print(f"   LightGBM best iteration = {bundle['lgbm_best_iteration']}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(make_metrics_json(bundle), encoding="utf-8")

    print("\n3) VALIDATION metrics (used for model selection)")
    val_df = pd.DataFrame(bundle["validation_metrics"]).T[METRIC_COLUMNS]
    print(val_df.sort_values(["ap", "f1", "recall"], ascending=False).round(4))

    print("\n4) TEST metrics (final report only)")
    test_df = pd.DataFrame(bundle["metrics"]).T[METRIC_COLUMNS]
    print(test_df.sort_values("ap", ascending=False).round(4))

    print("\nTop LightGBM features by gain")
    print(
        bundle["lgbm_feature_importance"][["feature", "gain_pct", "split"]]
        .head(10)
        .round({"gain_pct": 2})
        .to_string(index=False)
    )

    print(f"\nRecommended model (validation AP): {bundle['recommended_model']}")
    print(f"Saved bundle:  {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()