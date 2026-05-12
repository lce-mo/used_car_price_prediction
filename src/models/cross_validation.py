from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold

try:
    from src.models.predict_model import build_routing_mask, build_segment_mask
    from src.models.train_model import (
        build_age_bin,
        build_price_quantile_bin,
        build_sample_weights,
        fit_predict_model,
        inverse_target,
        prepare_model_inputs,
        transform_target,
    )
except ModuleNotFoundError:
    from models.predict_model import build_routing_mask, build_segment_mask
    from models.train_model import (
        build_age_bin,
        build_price_quantile_bin,
        build_sample_weights,
        fit_predict_model,
        inverse_target,
        prepare_model_inputs,
        transform_target,
    )


def build_model_frequency_bin(model_series: pd.Series) -> pd.Series:
    """Bucket model frequency for repeated stratified validation."""
    model_key = model_series.astype("string").fillna("__MODEL_MISSING__")
    model_count = model_key.map(model_key.value_counts(dropna=False)).astype(float)
    freq_bin = pd.cut(
        model_count,
        bins=[-np.inf, 1, 2, 5, 10, 20, 50, np.inf],
        labels=["f1", "f2", "f3_5", "f6_10", "f11_20", "f21_50", "f51_plus"],
    )
    return freq_bin.astype("string").fillna("__FREQ_MISSING__")
def collapse_rare_stratification_labels(
    primary_labels: pd.Series,
    fallback_levels: list[pd.Series],
    n_splits: int,
) -> pd.Series:
    """Collapse rare stratification labels until every bucket can be split."""
    collapsed = primary_labels.astype("string").copy()

    for fallback in fallback_levels:
        label_count = collapsed.value_counts(dropna=False)
        rare_mask = collapsed.map(label_count) < n_splits
        if not bool(rare_mask.any()):
            break
        fallback_label = fallback.astype("string")
        collapse_groups = fallback_label.loc[rare_mask].unique().tolist()
        collapse_mask = fallback_label.isin(collapse_groups)
        collapsed.loc[collapse_mask] = fallback_label.loc[collapse_mask]

    final_count = collapsed.value_counts(dropna=False)
    remaining_rare_mask = collapsed.map(final_count) < n_splits
    if bool(remaining_rare_mask.any()):
        collapsed.loc[remaining_rare_mask] = "__ALL__"
        final_count = collapsed.value_counts(dropna=False)
        if bool((final_count < n_splits).any()):
            collapsed[:] = "__ALL__"

    return collapsed.astype("string")
def build_stratification_labels(
    train_features: pd.DataFrame,
    target: pd.Series,
    n_splits: int,
    price_bins: int,
) -> pd.Series:
    """Build labels for repeated stratified validation."""
    if price_bins < 2:
        raise ValueError("price_bins must be at least 2 for stratified CV.")

    price_bin = build_price_quantile_bin(target.astype(float), bucket_count=price_bins)
    age_bin = build_age_bin(train_features["car_age_years"])
    model_freq_bin = build_model_frequency_bin(train_features["model"])

    full_label = price_bin + "|" + age_bin + "|" + model_freq_bin
    price_age_label = price_bin + "|" + age_bin
    price_label = price_bin
    age_label = age_bin

    return collapse_rare_stratification_labels(
        primary_labels=full_label,
        fallback_levels=[price_age_label, price_label, age_label],
        n_splits=n_splits,
    )
def build_cv_splits(
    train_features: pd.DataFrame,
    target: pd.Series,
    n_splits: int,
    cv_strategy: str,
    cv_repeats: int,
    cv_random_state: int,
    stratify_price_bins: int,
) -> tuple[list[tuple[int, int, np.ndarray, np.ndarray]], dict[str, int | str | None]]:
    """Build the legacy KFold or repeated stratified split plan."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if cv_strategy == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
        split_plan = [
            (1, fold_idx, train_idx, valid_idx)
            for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(train_features), start=1)
        ]
        metadata: dict[str, int | str | None] = {
            "cv_strategy": cv_strategy,
            "n_splits": n_splits,
            "cv_repeats": 1,
            "total_folds": len(split_plan),
            "cv_random_state": cv_random_state,
            "stratify_price_bins": None,
            "stratify_unique_labels": None,
            "stratify_smallest_bucket": None,
        }
        return split_plan, metadata

    if cv_strategy != "repeated_stratified":
        raise ValueError(f"Unsupported cv_strategy: {cv_strategy}")

    if cv_repeats < 1:
        raise ValueError("cv_repeats must be at least 1.")

    stratify_labels = build_stratification_labels(
        train_features=train_features,
        target=target,
        n_splits=n_splits,
        price_bins=stratify_price_bins,
    )
    split_plan: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    for repeat_idx in range(cv_repeats):
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=cv_random_state + repeat_idx,
        )
        split_plan.extend(
            (repeat_idx + 1, fold_idx, train_idx, valid_idx)
            for fold_idx, (train_idx, valid_idx) in enumerate(
                splitter.split(train_features, stratify_labels),
                start=1,
            )
        )

    label_count = stratify_labels.value_counts(dropna=False)
    metadata = {
        "cv_strategy": cv_strategy,
        "n_splits": n_splits,
        "cv_repeats": cv_repeats,
        "total_folds": len(split_plan),
        "cv_random_state": cv_random_state,
        "stratify_price_bins": stratify_price_bins,
        "stratify_unique_labels": int(label_count.shape[0]),
        "stratify_smallest_bucket": int(label_count.min()),
    }
    return split_plan, metadata
def cross_validate_train(
    prepared: PreparedData,
    n_splits: int,
    cv_strategy: str,
    cv_repeats: int,
    cv_random_state: int,
    stratify_price_bins: int,
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
) -> tuple[list[dict[str, float | int | None]], np.ndarray, dict[str, np.ndarray], dict[str, int | str | None]]:
    X = prepared.train_features
    y = prepared.target.astype(float)
    split_plan, cv_metadata = build_cv_splits(
        train_features=X,
        target=y,
        n_splits=n_splits,
        cv_strategy=cv_strategy,
        cv_repeats=cv_repeats,
        cv_random_state=cv_random_state,
        stratify_price_bins=stratify_price_bins,
    )

    oof_prediction_sums = np.zeros(len(X), dtype=float)
    global_oof_prediction_sums = np.zeros(len(X), dtype=float)
    routed_flag_sums = np.zeros(len(X), dtype=float)
    actual_segment_flag_sums = np.zeros(len(X), dtype=float)
    route_hit_flag_sums = np.zeros(len(X), dtype=float)
    prediction_counts = np.zeros(len(X), dtype=int)
    fold_metrics: list[dict[str, float | int | None]] = []

    for fold_number, (repeat_idx, fold_in_repeat, train_idx, valid_idx) in enumerate(split_plan, start=1):
        X_train_fold = X.iloc[train_idx]
        X_valid_fold = X.iloc[valid_idx]
        y_train_raw = y.iloc[train_idx]
        y_train_fold = transform_target(y.iloc[train_idx], use_log_target)
        y_valid_fold = y.iloc[valid_idx]
        global_sample_weight, weight_summary = build_sample_weights(
            raw_target=y_train_raw,
            train_features=X_train_fold,
            sample_weight_mode=sample_weight_mode,
            high_price_quantile=high_price_quantile,
            high_price_weight=high_price_weight,
            new_car_max_years=new_car_max_years,
            new_car_weight=new_car_weight,
            price_age_slice_weight=price_age_slice_weight,
            price_age_slice_targets=price_age_slice_targets,
            normalize_sample_weight=normalize_sample_weight,
        )
        prepared_for_fold, X_train_ready, X_valid_ready = prepare_model_inputs(
            prepared=prepared,
            fit_features=X_train_fold,
            fit_target=y_train_fold,
            apply_features=X_valid_fold,
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
        global_pred = inverse_target(
            fit_predict_model(
                prepared=prepared_for_fold,
                train_features=X_train_ready,
                train_target=y_train_fold,
                apply_features=X_valid_ready,
                model_name=model_name,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                model_random_state=model_random_state,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                lightgbm_objective=lightgbm_objective,
                sample_weight=global_sample_weight,
            ),
            use_log_target,
        )
        final_pred = global_pred.copy()

        train_age_bin = build_age_bin(X_train_fold["car_age_years"])
        valid_age_bin = build_age_bin(X_valid_fold["car_age_years"])
        segment_threshold = float(y_train_raw.quantile(0.8))
        segment_train_mask = build_segment_mask(
            price_values=y_train_raw,
            age_bin=train_age_bin,
            price_threshold=segment_threshold,
            segment_scope=segment_scope,
        )
        actual_segment_mask = build_segment_mask(
            price_values=y_valid_fold,
            age_bin=valid_age_bin,
            price_threshold=segment_threshold,
            segment_scope=segment_scope,
        )
        routed_mask = pd.Series(False, index=X_valid_fold.index)
        route_hit_mask = pd.Series(False, index=X_valid_fold.index)
        segment_train_count = int(segment_train_mask.sum())

        if use_segmented_modeling and segment_train_count >= 20:
            routed_mask = build_routing_mask(
                predicted_prices=global_pred,
                age_bin=valid_age_bin,
                price_threshold=segment_threshold,
                routing_mode=segment_routing_mode,
                segment_scope=segment_scope,
            )
            if bool(routed_mask.any()):
                X_segment_train = X_train_fold.loc[segment_train_mask].copy()
                y_segment_train_raw = y_train_raw.loc[segment_train_mask]
                y_segment_train = transform_target(y_segment_train_raw, use_log_target)
                X_segment_apply = X_valid_fold.loc[routed_mask].copy()
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
                segment_pred = inverse_target(
                    fit_predict_model(
                        prepared=prepared_for_segment,
                        train_features=X_segment_train_ready,
                        train_target=y_segment_train,
                        apply_features=X_segment_apply_ready,
                        model_name=model_name,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        num_leaves=num_leaves,
                        model_random_state=model_random_state,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        lightgbm_objective=lightgbm_objective,
                        sample_weight=segment_sample_weight,
                    ),
                    use_log_target,
                )
                final_pred[routed_mask.to_numpy()] = segment_pred
                route_hit_mask = routed_mask & actual_segment_mask

        global_oof_prediction_sums[valid_idx] += global_pred
        oof_prediction_sums[valid_idx] += final_pred
        routed_flag_sums[valid_idx] += routed_mask.to_numpy().astype(float)
        actual_segment_flag_sums[valid_idx] += actual_segment_mask.to_numpy().astype(float)
        route_hit_flag_sums[valid_idx] += route_hit_mask.to_numpy().astype(float)
        prediction_counts[valid_idx] += 1

        global_mae = mean_absolute_error(y_valid_fold, global_pred)
        final_mae = mean_absolute_error(y_valid_fold, final_pred)
        actual_segment_count = int(actual_segment_mask.sum())
        routed_count = int(routed_mask.sum())
        route_hit_count = int(route_hit_mask.sum())
        segment_mae = (
            float(mean_absolute_error(y_valid_fold.loc[actual_segment_mask], final_pred[actual_segment_mask.to_numpy()]))
            if actual_segment_count > 0
            else None
        )
        global_segment_mae = (
            float(mean_absolute_error(y_valid_fold.loc[actual_segment_mask], global_pred[actual_segment_mask.to_numpy()]))
            if actual_segment_count > 0
            else None
        )
        non_segment_mae = (
            float(mean_absolute_error(y_valid_fold.loc[~actual_segment_mask], final_pred[(~actual_segment_mask).to_numpy()]))
            if int((~actual_segment_mask).sum()) > 0
            else None
        )
        routing_precision = (route_hit_count / routed_count) if routed_count > 0 else None
        routing_recall = (route_hit_count / actual_segment_count) if actual_segment_count > 0 else None

        fold_metrics.append(
            {
                "fold": fold_number,
                "repeat": repeat_idx,
                "fold_in_repeat": fold_in_repeat,
                "mae": float(final_mae),
                "global_mae": float(global_mae),
                **weight_summary,
                "segment_train_count": segment_train_count,
                "actual_segment_count": actual_segment_count,
                "routed_count": routed_count,
                "segment_coverage_rate": routed_count / len(X_valid_fold),
                "routing_precision": float(routing_precision) if routing_precision is not None else None,
                "routing_recall": float(routing_recall) if routing_recall is not None else None,
                "segment_mae": segment_mae,
                "global_segment_mae": global_segment_mae,
                "non_segment_mae": non_segment_mae,
                "segment_threshold": segment_threshold,
            }
        )
        print(
            f"Repeat {repeat_idx} Fold {fold_in_repeat} (global fold {fold_number}) MAE: {final_mae:.4f}"
            + (f" | global={global_mae:.4f}" if use_segmented_modeling else "")
        )

    if bool((prediction_counts == 0).any()):
        raise RuntimeError("Some training samples did not receive any validation predictions.")

    prediction_counts_float = prediction_counts.astype(float)
    oof_predictions = oof_prediction_sums / prediction_counts_float
    global_oof_predictions = global_oof_prediction_sums / prediction_counts_float
    routed_flags = routed_flag_sums / prediction_counts_float
    actual_segment_flags = actual_segment_flag_sums / prediction_counts_float
    route_hit_flags = route_hit_flag_sums / prediction_counts_float

    return fold_metrics, oof_predictions, {
        "global_oof_pred": global_oof_predictions,
        "segmented_replaced_flag": routed_flags,
        "actual_segment_flag": actual_segment_flags,
        "routing_hit_flag": route_hit_flags,
        "cv_prediction_count": prediction_counts,
    }, cv_metadata
