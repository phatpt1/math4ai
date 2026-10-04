import json
from datetime import datetime, timezone
from io import BytesIO

import joblib


def serialize_bundle(bundle) -> bytes:
    buffer = BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    return buffer.getvalue()


def make_metrics_json(bundle) -> str:
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recommended_model": bundle["recommended_model"],
        "scale_pos_weight": bundle.get("scale_pos_weight"),
        "lgbm_best_iteration": bundle.get("lgbm_best_iteration"),
        "dataset_summary": bundle.get("dataset_summary"),
        "metrics": bundle.get("metrics"),
        "validation_metrics": bundle.get("validation_metrics"),
        "selection_rule": bundle.get("selection_rule"),
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)