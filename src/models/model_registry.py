from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor

try:
    from src.config import MODEL_DIR, PREDICTION_DIR, REPORT_DIR, SUBMISSION_DIR
except ModuleNotFoundError:
    from config import MODEL_DIR, PREDICTION_DIR, REPORT_DIR, SUBMISSION_DIR


MODEL_REGISTRY = {
    "legacy_lightgbm": "Existing LightGBM pipeline in src/train.py",
    "legacy_gbrt": "Existing GradientBoostingRegressor option in src/train.py",
    "lightgbm": "LightGBM regressor used by the current training pipeline",
    "gbrt": "GradientBoostingRegressor fallback used by the current training pipeline",
    "baseline_placeholder": "Reserved for a future simple baseline model",
}

ModelType = GradientBoostingRegressor | LGBMRegressor


def list_models() -> dict[str, str]:
    """Return the available model entrypoints."""
    return dict(MODEL_REGISTRY)


def get_model(
    model_name: str,
    learning_rate: float,
    n_estimators: int,
    num_leaves: int,
    model_random_state: int,
    subsample: float,
    colsample_bytree: float,
    lightgbm_objective: str,
) -> ModelType:
    """Build a supported regressor from the model registry."""
    if model_name == "gbrt":
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=4,
            min_samples_leaf=20,
            subsample=subsample,
            random_state=model_random_state,
        )

    if model_name == "lightgbm":
        return LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=model_random_state,
            objective=lightgbm_objective,
            verbose=-1,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def save_outputs(
    output_dir: Path,
    prepared: PreparedData,
    fold_metrics: list[dict[str, float | int]],
    oof_predictions: np.ndarray,
    test_predictions: np.ndarray | None,
    use_log_target: bool,
    model_name: str,
    learning_rate: float,
    n_estimators: int,
    num_leaves: int,
    model_random_state: int,
    subsample: float,
    colsample_bytree: float,
    lightgbm_objective: str,
    use_sample_weighting: bool,
    sample_weight_mode: str,
    high_price_quantile: float,
    high_price_weight: float,
    new_car_max_years: float,
    new_car_weight: float,
    price_age_slice_weight: float,
    price_age_slice_targets: str,
    normalize_sample_weight: bool,
    use_brand_target_encoding: bool,
    use_brand_age_target_encoding: bool,
    use_model_target_encoding: bool,
    use_model_age_target_encoding: bool,
    use_model_backoff_target_encoding: bool,
    model_backoff_min_count: int,
    use_model_low_freq_flag: bool,
    model_low_freq_min_count: int,
    target_encoding_smoothing: float,
    use_model_age_group_stats: bool,
    model_age_group_min_count: int,
    use_segmented_modeling: bool,
    segment_routing_mode: str,
    segment_scope: str,
    cv_metadata: dict[str, int | str | None],
    oof_extra_columns: dict[str, np.ndarray] | None = None,
    prediction_details: dict[str, float | int | None] | None = None,
    model_artifact: dict[str, object] | None = None,
) -> None:
    try:
        from src.models.train_model import format_target_mode
    except ModuleNotFoundError:
        from models.train_model import format_target_mode

    output_dir.mkdir(parents=True, exist_ok=True)

    maes = [item["mae"] for item in fold_metrics]
    metrics = {
        "model_name": model_name,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "model_random_state": model_random_state,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lightgbm_objective": lightgbm_objective,
        "use_sample_weighting": use_sample_weighting,
        "sample_weight_mode": sample_weight_mode,
        "high_price_quantile": high_price_quantile,
        "high_price_weight": high_price_weight,
        "new_car_max_years": new_car_max_years,
        "new_car_weight": new_car_weight,
        "price_age_slice_weight": price_age_slice_weight,
        "price_age_slice_targets": price_age_slice_targets,
        "normalize_sample_weight": normalize_sample_weight,
        "use_brand_target_encoding": use_brand_target_encoding,
        "use_brand_age_target_encoding": use_brand_age_target_encoding,
        "use_model_target_encoding": use_model_target_encoding,
        "use_model_age_target_encoding": use_model_age_target_encoding,
        "use_model_backoff_target_encoding": use_model_backoff_target_encoding,
        "model_backoff_min_count": model_backoff_min_count,
        "use_model_low_freq_flag": use_model_low_freq_flag,
        "model_low_freq_min_count": model_low_freq_min_count,
        "target_encoding_smoothing": target_encoding_smoothing,
        "use_model_age_group_stats": use_model_age_group_stats,
        "model_age_group_min_count": model_age_group_min_count,
        "use_segmented_modeling": use_segmented_modeling,
        "segment_routing_mode": segment_routing_mode,
        "segment_scope": segment_scope,
        "cv": cv_metadata,
        "target_mode": format_target_mode(use_log_target),
        "folds": fold_metrics,
        "overall": {
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
        },
    }
    global_maes = [item["global_mae"] for item in fold_metrics if item.get("global_mae") is not None]
    if global_maes:
        metrics["overall"]["global_mae_mean"] = float(np.mean(global_maes))
        metrics["overall"]["global_mae_std"] = float(np.std(global_maes))
    if prediction_details is not None:
        metrics["prediction_details"] = prediction_details

    oof_df = pd.DataFrame(
        {
            "SaleID": prepared.train_ids,
            "price": prepared.target,
            "oof_pred": oof_predictions,
        }
    )
    if oof_extra_columns is not None:
        for col_name, values in oof_extra_columns.items():
            oof_df[col_name] = values
    oof_df["abs_error"] = (oof_df["price"] - oof_df["oof_pred"]).abs()
    metrics["overall"]["oof_mae"] = float(oof_df["abs_error"].mean())
    with (output_dir / "cv_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    if test_predictions is not None:
        submission_df = pd.DataFrame(
            {
                "SaleID": prepared.test_ids,
                "price": test_predictions,
            }
        )
        submission_df.to_csv(output_dir / "submission.csv", index=False)
    else:
        submission_df = None

    write_standard_outputs(
        prepared=prepared,
        metrics=metrics,
        oof_df=oof_df,
        submission_df=submission_df,
        model_artifact=model_artifact,
    )

    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))


def write_standard_outputs(
    prepared: PreparedData,
    metrics: dict[str, object],
    oof_df: pd.DataFrame,
    submission_df: pd.DataFrame | None,
    model_artifact: dict[str, object] | None,
) -> None:
    """Write stable top-level artifacts consumed by docs and downstream scripts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    valid_predictions = oof_df[["SaleID", "price", "oof_pred", "abs_error"]].rename(
        columns={"oof_pred": "prediction"}
    )
    valid_predictions.to_csv(PREDICTION_DIR / "valid_predictions.csv", index=False)

    if submission_df is not None:
        submission_df.to_csv(PREDICTION_DIR / "test_predictions.csv", index=False)
        submission_df.to_csv(SUBMISSION_DIR / "submission_001_baseline.csv", index=False)
        submission_df.to_csv(SUBMISSION_DIR / "submission_002_improved.csv", index=False)

    if model_artifact is not None:
        bundle = {
            "artifact_type": "used_car_price_prediction_model",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "model_artifact": model_artifact,
        }
        for filename in ["baseline_model.pkl", "best_model.pkl"]:
            with (MODEL_DIR / filename).open("wb") as file:
                pickle.dump(bundle, file)

    write_reports(
        prepared=prepared,
        metrics=metrics,
        oof_df=oof_df,
        submission_df=submission_df,
        model_artifact=model_artifact,
    )


def write_reports(
    prepared: PreparedData,
    metrics: dict[str, object],
    oof_df: pd.DataFrame,
    submission_df: pd.DataFrame | None,
    model_artifact: dict[str, object] | None,
) -> None:
    overall = metrics.get("overall", {})
    cv = metrics.get("cv", {})
    target = prepared.target.astype(float)
    abs_error = oof_df["abs_error"].astype(float)
    error_quantiles = abs_error.quantile([0.5, 0.75, 0.9, 0.95, 0.99])

    model_summary = [
        "# Model Summary",
        "",
        f"- model_name: `{metrics.get('model_name')}`",
        f"- target_mode: `{metrics.get('target_mode')}`",
        f"- fold_mae_mean: `{overall.get('mae_mean')}`",
        f"- fold_mae_std: `{overall.get('mae_std')}`",
        f"- aggregated_oof_mae: `{overall.get('oof_mae')}`",
        f"- cv_strategy: `{cv.get('cv_strategy')}`",
        f"- total_folds: `{cv.get('total_folds')}`",
        f"- model_pickle_written: `{model_artifact is not None}`",
        f"- test_predictions_written: `{submission_df is not None}`",
        "",
        "## Standard Artifacts",
        "",
        "- `outputs/models/baseline_model.pkl`",
        "- `outputs/models/best_model.pkl`",
        "- `outputs/predictions/valid_predictions.csv`",
        "- `outputs/predictions/test_predictions.csv`",
        "- `outputs/submissions/submission_001_baseline.csv`",
        "- `outputs/submissions/submission_002_improved.csv`",
    ]
    (REPORT_DIR / "model_summary.md").write_text("\n".join(model_summary) + "\n", encoding="utf-8")

    feature_report = [
        "# Feature Report",
        "",
        f"- train_rows: `{len(prepared.train_features)}`",
        f"- test_rows: `{len(prepared.test_features)}`",
        f"- numeric_feature_count: `{len(prepared.numeric_columns)}`",
        f"- categorical_feature_count: `{len(prepared.categorical_columns)}`",
        f"- total_feature_count: `{prepared.train_features.shape[1]}`",
        f"- categorical_columns: `{', '.join(prepared.categorical_columns)}`",
        "",
        "## Feature Switches",
        "",
        f"- use_model_age_group_stats: `{metrics.get('use_model_age_group_stats')}`",
        f"- use_brand_target_encoding: `{metrics.get('use_brand_target_encoding')}`",
        f"- use_model_target_encoding: `{metrics.get('use_model_target_encoding')}`",
        f"- use_model_age_target_encoding: `{metrics.get('use_model_age_target_encoding')}`",
        f"- use_model_backoff_target_encoding: `{metrics.get('use_model_backoff_target_encoding')}`",
    ]
    (REPORT_DIR / "feature_report.md").write_text("\n".join(feature_report) + "\n", encoding="utf-8")

    eda_report = [
        "# EDA Report",
        "",
        f"- train_rows: `{len(prepared.target)}`",
        f"- test_rows: `{len(prepared.test_ids)}`",
        f"- target_min: `{float(target.min())}`",
        f"- target_median: `{float(target.median())}`",
        f"- target_mean: `{float(target.mean())}`",
        f"- target_max: `{float(target.max())}`",
    ]
    if submission_df is not None:
        pred = submission_df["price"].astype(float)
        eda_report.extend(
            [
                f"- test_prediction_min: `{float(pred.min())}`",
                f"- test_prediction_median: `{float(pred.median())}`",
                f"- test_prediction_max: `{float(pred.max())}`",
            ]
        )
    (REPORT_DIR / "eda_report.md").write_text("\n".join(eda_report) + "\n", encoding="utf-8")

    error_report = [
        "# Error Analysis",
        "",
        f"- aggregated_oof_mae: `{float(abs_error.mean())}`",
        f"- abs_error_median: `{float(error_quantiles.loc[0.5])}`",
        f"- abs_error_p75: `{float(error_quantiles.loc[0.75])}`",
        f"- abs_error_p90: `{float(error_quantiles.loc[0.9])}`",
        f"- abs_error_p95: `{float(error_quantiles.loc[0.95])}`",
        f"- abs_error_p99: `{float(error_quantiles.loc[0.99])}`",
        f"- worst_10pct_abs_error_share: `{float(abs_error.nlargest(max(1, int(len(abs_error) * 0.1))).sum() / abs_error.sum())}`",
    ]
    (REPORT_DIR / "error_analysis.md").write_text("\n".join(error_report) + "\n", encoding="utf-8")
