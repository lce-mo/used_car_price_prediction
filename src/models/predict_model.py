from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.models.train_model import (
        build_age_bin,
        build_sample_weights,
        fit_model_artifact,
        fit_predict_model,
        inverse_target,
        prepare_model_inputs,
        transform_target,
    )
except ModuleNotFoundError:
    from models.train_model import (
        build_age_bin,
        build_sample_weights,
        fit_model_artifact,
        fit_predict_model,
        inverse_target,
        prepare_model_inputs,
        transform_target,
    )


def predict() -> None:
    """Placeholder for standalone prediction.

    The current validated pipeline writes predictions from `src/train.py`.
    A future standalone implementation should load a registered model from
    `outputs/models/`, load processed test features, and write predictions to
    `outputs/submissions/`.
    """
    print(
        "Standalone predict is not configured yet. "
        "The current legacy training pipeline produces predictions during training."
    )
def build_segment_mask(
    price_values: pd.Series | np.ndarray,
    age_bin: pd.Series,
    price_threshold: float,
    segment_scope: str,
) -> pd.Series:
    if segment_scope != "q5_5plus":
        raise ValueError(f"Unsupported segment_scope: {segment_scope}")

    price_series = pd.Series(price_values, index=age_bin.index, dtype=float)
    return (price_series >= price_threshold) & age_bin.isin(["5_8y", "8y_plus"])
def build_routing_mask(
    predicted_prices: np.ndarray,
    age_bin: pd.Series,
    price_threshold: float,
    routing_mode: str,
    segment_scope: str,
) -> pd.Series:
    if routing_mode != "global_pred_plus_age":
        raise ValueError(f"Unsupported segment_routing_mode: {routing_mode}")
    return build_segment_mask(
        price_values=predicted_prices,
        age_bin=age_bin,
        price_threshold=price_threshold,
        segment_scope=segment_scope,
    )
def fit_full_and_predict(
    prepared: PreparedData,
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
    use_model_low_freq_flag: bool,
    model_backoff_min_count: int,
    model_low_freq_min_count: int,
    target_encoding_smoothing: float,
    use_segmented_modeling: bool,
    segment_routing_mode: str,
    segment_scope: str,
) -> tuple[np.ndarray, dict[str, float | int | None], dict[str, object]]:
    X_train = prepared.train_features
    raw_target = prepared.target.astype(float)
    y_train = transform_target(raw_target, use_log_target)
    X_test = prepared.test_features
    global_sample_weight, global_weight_summary = build_sample_weights(
        raw_target=raw_target,
        train_features=X_train,
        sample_weight_mode=sample_weight_mode,
        high_price_quantile=high_price_quantile,
        high_price_weight=high_price_weight,
        new_car_max_years=new_car_max_years,
        new_car_weight=new_car_weight,
        price_age_slice_weight=price_age_slice_weight,
        price_age_slice_targets=price_age_slice_targets,
        normalize_sample_weight=normalize_sample_weight,
    )
    prepared_for_fit, X_train_ready, X_test_ready = prepare_model_inputs(
        prepared=prepared,
        fit_features=X_train,
        fit_target=y_train,
        apply_features=X_test,
        use_brand_target_encoding=use_brand_target_encoding,
        use_brand_age_target_encoding=use_brand_age_target_encoding,
        use_model_target_encoding=use_model_target_encoding,
        use_model_age_target_encoding=use_model_age_target_encoding,
        use_model_backoff_target_encoding=use_model_backoff_target_encoding,
        use_model_low_freq_flag=use_model_low_freq_flag,
        model_backoff_min_count=model_backoff_min_count,
        model_low_freq_min_count=model_low_freq_min_count,
        target_encoding_smoothing=target_encoding_smoothing,
    )
    global_artifact = fit_model_artifact(
        prepared=prepared_for_fit,
        train_features=X_train_ready,
        train_target=y_train,
        model_name=model_name,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        model_random_state=model_random_state,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        lightgbm_objective=lightgbm_objective,
        sample_weight=global_sample_weight,
    )
    global_test_predictions = inverse_target(
        global_artifact["model"].predict(global_artifact["preprocessor"].transform(X_test_ready)),
        use_log_target,
    )

    final_predictions = global_test_predictions.copy()
    details: dict[str, float | int | None] = {
        "segmented_modeling_enabled": int(use_segmented_modeling),
        "segment_threshold": None,
        "segment_train_count": 0,
        "routed_count": 0,
        "route_rate": 0.0,
        "weighted_sample_count": global_weight_summary["weighted_sample_count"],
        "weight_mean": global_weight_summary["weight_mean"],
        "weight_max": global_weight_summary["weight_max"],
    }

    if not use_segmented_modeling:
        return final_predictions, details, {"global_model": global_artifact}

    train_age_bin = build_age_bin(X_train["car_age_years"])
    test_age_bin = build_age_bin(X_test["car_age_years"])
    segment_threshold = float(raw_target.quantile(0.8))
    segment_train_mask = build_segment_mask(
        price_values=raw_target,
        age_bin=train_age_bin,
        price_threshold=segment_threshold,
        segment_scope=segment_scope,
    )
    routed_mask = build_routing_mask(
        predicted_prices=global_test_predictions,
        age_bin=test_age_bin,
        price_threshold=segment_threshold,
        routing_mode=segment_routing_mode,
        segment_scope=segment_scope,
    )
    segment_train_count = int(segment_train_mask.sum())
    routed_count = int(routed_mask.sum())
    details.update(
        {
            "segment_threshold": segment_threshold,
            "segment_train_count": segment_train_count,
            "routed_count": routed_count,
            "route_rate": routed_count / len(X_test) if len(X_test) > 0 else 0.0,
        }
    )

    if segment_train_count < 20 or routed_count == 0:
        return final_predictions, details, {"global_model": global_artifact}

    X_segment_train = X_train.loc[segment_train_mask].copy()
    y_segment_train_raw = raw_target.loc[segment_train_mask]
    y_segment_train = transform_target(y_segment_train_raw, use_log_target)
    X_segment_apply = X_test.loc[routed_mask].copy()
    segment_sample_weight, _ = build_sample_weights(
        raw_target=y_segment_train_raw,
        train_features=X_segment_train,
        sample_weight_mode=sample_weight_mode,
        high_price_quantile=high_price_quantile,
        high_price_weight=high_price_weight,
        new_car_max_years=new_car_max_years,
        new_car_weight=new_car_weight,
        price_age_slice_weight=price_age_slice_weight,
        price_age_slice_targets=price_age_slice_targets,
        normalize_sample_weight=normalize_sample_weight,
    )
    prepared_for_segment, X_segment_train_ready, X_segment_apply_ready = prepare_model_inputs(
        prepared=prepared,
        fit_features=X_segment_train,
        fit_target=y_segment_train,
        apply_features=X_segment_apply,
        use_brand_target_encoding=use_brand_target_encoding,
        use_brand_age_target_encoding=use_brand_age_target_encoding,
        use_model_target_encoding=use_model_target_encoding,
        use_model_age_target_encoding=use_model_age_target_encoding,
        use_model_backoff_target_encoding=use_model_backoff_target_encoding,
        use_model_low_freq_flag=use_model_low_freq_flag,
        model_backoff_min_count=model_backoff_min_count,
        model_low_freq_min_count=model_low_freq_min_count,
        target_encoding_smoothing=target_encoding_smoothing,
    )
    segment_artifact = fit_model_artifact(
        prepared=prepared_for_segment,
        train_features=X_segment_train_ready,
        train_target=y_segment_train,
        model_name=model_name,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        model_random_state=model_random_state,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        lightgbm_objective=lightgbm_objective,
        sample_weight=segment_sample_weight,
    )
    segment_predictions = inverse_target(
        segment_artifact["model"].predict(segment_artifact["preprocessor"].transform(X_segment_apply_ready)),
        use_log_target,
    )
    final_predictions[routed_mask.to_numpy()] = segment_predictions
    return final_predictions, details, {
        "global_model": global_artifact,
        "segment_model": segment_artifact,
    }
