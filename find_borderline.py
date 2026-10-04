import numpy as np
import pandas as pd

from src.inference import score_one
from src.storage import load_bundle_local

bundle = load_bundle_local()
model = bundle["recommended_model"]
threshold = bundle["selected_thresholds"][model]

base = dict(bundle["defaults"])
base.update(
    {
        "Administrative": 2,
        "Administrative_Duration": 60.0,
        "ProductRelated": 25,
        "ProductRelated_Duration": 900.0,
        "BounceRates": 0.005,
        "Month": "Mar",
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }
)

rows = []
for page_values in np.arange(0.0, 10.01, 0.5):
    for exit_rates in [0.02, 0.04, 0.06, 0.08, 0.1]:
        row = {**base, "PageValues": float(page_values), "ExitRates": exit_rates}
        df = pd.DataFrame([[row[c] for c in bundle["feature_order"]]], columns=bundle["feature_order"])
        score, _, _ = score_one(model, df, bundle)
        rows.append({"PageValues": page_values, "ExitRates": exit_rates, "score": round(score, 3)})

result = pd.DataFrame(rows)
result["gap"] = (result["score"] - threshold).abs()
print(f"Model: {model} · threshold = {threshold:.3f}")
print(result.sort_values("gap").head(10).to_string(index=False))