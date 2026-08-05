#!/usr/bin/env python3
"""Fail closed unless generated artifacts support the documented résumé claim."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sponsorship_deals.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "trained_model.joblib"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.json"
TARGETS = {"media_value_ratio", "brand_lift_pct", "roi_score"}


def main() -> int:
    missing = [path for path in (DATA_PATH, MODEL_PATH, METRICS_PATH) if not path.is_file()]
    if missing:
        raise SystemExit(f"missing generated artifacts: {', '.join(map(str, missing))}")

    data = pd.read_csv(DATA_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    artifact = joblib.load(MODEL_PATH)

    if len(data) != 5_000:
        raise SystemExit(f"expected 5,000 synthetic deals, found {len(data):,}")
    if set(metrics) != TARGETS or set(artifact.get("models", {})) != TARGETS:
        raise SystemExit("expected one metric and one fitted model for each of three targets")

    roi_r_squared = float(metrics["roi_score"]["r2"])
    if round(roi_r_squared, 2) != 0.78:
        raise SystemExit(f"ROI R-squared no longer rounds to 0.78: {roi_r_squared}")

    held_out = artifact.get("held_out", {})
    for target in TARGETS:
        actual = held_out.get(target, {}).get("actual", [])
        predicted = held_out.get(target, {}).get("predicted", [])
        if len(actual) != 1_000 or len(predicted) != 1_000:
            raise SystemExit(f"{target} does not preserve 1,000 held-out observations")

    print(
        json.dumps(
            {
                "synthetic_deals": len(data),
                "regressors": len(artifact["models"]),
                "held_out_rows_per_target": 1_000,
                "roi_r_squared": roi_r_squared,
                "resume_rounding": 0.78,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
