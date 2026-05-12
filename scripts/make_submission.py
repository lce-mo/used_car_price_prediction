from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import PREDICTION_DIR, SUBMISSION_DIR
from src.utils.io import ensure_dir


def main() -> None:
    ensure_dir(SUBMISSION_DIR)
    prediction_path = PREDICTION_DIR / "test_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing {prediction_path}. Run `make train` or `make predict` before `make submit`."
        )

    submission = pd.read_csv(prediction_path)
    required = {"SaleID", "price"}
    missing = required.difference(submission.columns)
    if missing:
        raise ValueError(f"{prediction_path} missing required columns: {sorted(missing)}")

    submission = submission[["SaleID", "price"]]
    submission.to_csv(SUBMISSION_DIR / "submission_001_baseline.csv", index=False)
    submission.to_csv(SUBMISSION_DIR / "submission_002_improved.csv", index=False)
    print("Saved standard submissions to outputs/submissions/")


if __name__ == "__main__":
    main()
