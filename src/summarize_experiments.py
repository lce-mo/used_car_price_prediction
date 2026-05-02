from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUTS_DIR = Path("outputs")
DEFAULT_SCOREBOARD_PATH = Path("outputs/experiment_scoreboard.csv")


def _get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _safe_round(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _read_metrics(metrics_path: Path, outputs_dir: Path) -> dict[str, Any] | None:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    run_dir = metrics_path.parent
    cv = metrics.get("cv") or {}
    folds = metrics.get("folds") or []
    overall = metrics.get("overall") or {}
    relative_run_dir = run_dir.relative_to(outputs_dir.parent)

    return {
        "run": run_dir.name,
        "run_dir": str(relative_run_dir),
        "metrics_path": str(metrics_path.relative_to(outputs_dir.parent)),
        "mae_mean": _safe_round(overall.get("mae_mean")),
        "mae_std": _safe_round(overall.get("mae_std")),
        "global_mae_mean": _safe_round(overall.get("global_mae_mean")),
        "global_mae_std": _safe_round(overall.get("global_mae_std")),
        "cv_strategy": cv.get("cv_strategy", "legacy"),
        "n_splits": cv.get("n_splits", len(folds) if folds else None),
        "cv_repeats": cv.get("cv_repeats"),
        "total_folds": cv.get("total_folds", len(folds) if folds else None),
        "cv_random_state": cv.get("cv_random_state"),
        "stratify_price_bins": cv.get("stratify_price_bins"),
        "stratify_unique_labels": cv.get("stratify_unique_labels"),
        "stratify_smallest_bucket": cv.get("stratify_smallest_bucket"),
        "model_name": metrics.get("model_name"),
        "learning_rate": metrics.get("learning_rate"),
        "n_estimators": metrics.get("n_estimators"),
        "num_leaves": metrics.get("num_leaves"),
        "model_random_state": metrics.get("model_random_state"),
        "subsample": metrics.get("subsample"),
        "colsample_bytree": metrics.get("colsample_bytree"),
        "lightgbm_objective": metrics.get("lightgbm_objective"),
        "target_mode": metrics.get("target_mode"),
        "use_group_stats": metrics.get("use_group_stats"),
        "use_brand_target_encoding": metrics.get("use_brand_target_encoding"),
        "use_brand_age_target_encoding": metrics.get("use_brand_age_target_encoding"),
        "use_model_target_encoding": metrics.get("use_model_target_encoding"),
        "use_model_age_target_encoding": metrics.get("use_model_age_target_encoding"),
        "use_model_backoff_target_encoding": metrics.get("use_model_backoff_target_encoding"),
        "target_encoding_smoothing": metrics.get("target_encoding_smoothing"),
        "model_backoff_min_count": metrics.get("model_backoff_min_count"),
        "use_model_low_freq_flag": metrics.get("use_model_low_freq_flag"),
        "use_model_age_group_stats": metrics.get("use_model_age_group_stats"),
        "use_sample_weighting": metrics.get("use_sample_weighting"),
        "sample_weight_mode": metrics.get("sample_weight_mode"),
        "high_price_weight": metrics.get("high_price_weight"),
        "new_car_weight": metrics.get("new_car_weight"),
        "price_age_slice_weight": metrics.get("price_age_slice_weight"),
        "price_age_slice_targets": metrics.get("price_age_slice_targets"),
        "normalize_sample_weight": metrics.get("normalize_sample_weight"),
        "use_segmented_modeling": metrics.get("use_segmented_modeling"),
        "segment_scope": metrics.get("segment_scope"),
        "segment_routing_mode": metrics.get("segment_routing_mode"),
        "prediction_mode": _get_nested(metrics, "prediction_details", "mode"),
        "metrics_modified_time": metrics_path.stat().st_mtime,
    }


def build_scoreboard(outputs_dir: Path) -> pd.DataFrame:
    rows = [
        row
        for metrics_path in outputs_dir.rglob("cv_metrics.json")
        if (row := _read_metrics(metrics_path, outputs_dir)) is not None
    ]
    if not rows:
        return pd.DataFrame()

    scoreboard = pd.DataFrame(rows)
    return scoreboard.sort_values(
        by=["mae_mean", "mae_std", "run"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize local CV experiments into a scoreboard CSV.")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Root directory containing experiment output folders.",
    )
    parser.add_argument(
        "--scoreboard-path",
        type=Path,
        default=DEFAULT_SCOREBOARD_PATH,
        help="CSV path for the generated scoreboard.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of best rows to print after writing the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scoreboard = build_scoreboard(args.outputs_dir)
    args.scoreboard_path.parent.mkdir(parents=True, exist_ok=True)
    scoreboard.to_csv(args.scoreboard_path, index=False)

    print(f"Wrote {len(scoreboard)} rows to {args.scoreboard_path}")
    if not scoreboard.empty and args.top > 0:
        print(
            scoreboard[
                [
                    "run",
                    "mae_mean",
                    "mae_std",
                    "cv_strategy",
                    "total_folds",
                    "learning_rate",
                    "n_estimators",
                    "num_leaves",
                    "target_encoding_smoothing",
                    "sample_weight_mode",
                    "price_age_slice_weight",
                    "use_model_target_encoding",
                    "use_model_backoff_target_encoding",
                    "use_segmented_modeling",
                ]
            ]
            .head(args.top)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
