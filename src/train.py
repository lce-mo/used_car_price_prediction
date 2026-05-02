from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from features import PreparedData, prepare_features


ModelType = GradientBoostingRegressor | LGBMRegressor

DEFAULT_TRAIN_PATH = Path("data/raw/used_car_train_20200313.csv")
DEFAULT_TEST_PATH = Path("data/raw/used_car_testB_20200421.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/main_lgbm_m2_q3")
DEFAULT_MODEL_NAME = "lightgbm"
DEFAULT_LEARNING_RATE = 0.08
DEFAULT_N_ESTIMATORS = 400
DEFAULT_NUM_LEAVES = 63
DEFAULT_CV_STRATEGY = "repeated_stratified"
DEFAULT_CV_REPEATS = 3
DEFAULT_CV_RANDOM_STATE = 42
DEFAULT_STRATIFY_PRICE_BINS = 5
DEFAULT_PRICE_AGE_SLICE_TARGETS = "Q5:8y_plus,Q4:8y_plus,Q5:5_8y,Q3:8y_plus,Q5:3_5y"
RAW_DATA_SEPARATOR = " "


def validate_raw_dataframe(df: pd.DataFrame, path: Path) -> None:
    problems: list[str] = []

    if "price" in df.columns:
        price = pd.to_numeric(df["price"], errors="coerce")
        if bool(price.lt(0).any()):
            problems.append(f"price has negative values; min={price.min()}")

    range_checks = {
        "kilometer": (0, 15),
        "bodyType": (0, 7),
        "fuelType": (0, 6),
        "gearbox": (0, 1),
        "regionCode": (0, 100000),
    }
    for col, (lower, upper) in range_checks.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        bad_rate = float(((values < lower) | (values > upper)).mean())
        if bad_rate > 0.001:
            problems.append(f"{col} out-of-range rate={bad_rate:.4%}")

    if problems:
        details = "; ".join(problems)
        raise ValueError(
            f"Raw data in {path} looks column-shifted or malformed: {details}. "
            "The Tianchi used-car raw files contain empty fields, so read them with sep=' ', not sep=r'\\s+'."
        )


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path, sep=RAW_DATA_SEPARATOR)
    test_df = pd.read_csv(test_path, sep=RAW_DATA_SEPARATOR)
    validate_raw_dataframe(train_df, train_path)
    validate_raw_dataframe(test_df, test_path)
    return train_df, test_df


def sample_rows(df: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None:
        return df
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")
    if sample_size >= len(df):
        return df.copy()
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def build_preprocessor(prepared: PreparedData) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                prepared.numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-2,
                            ),
                        ),
                    ]
                ),
                prepared.categorical_columns,
            ),
        ]
    )


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


def build_age_bin(series: pd.Series) -> pd.Series:
    age_years = pd.to_numeric(series, errors="coerce").clip(lower=0)
    bins = pd.cut(
        age_years,
        bins=[-np.inf, 1, 3, 5, 8, np.inf],
        labels=["0_1y", "1_3y", "3_5y", "5_8y", "8y_plus"],
    )
    return bins.astype("string").fillna("__AGE_MISSING__")


def build_price_quantile_bin(price: pd.Series, bucket_count: int) -> pd.Series:
    ranked_price = price.rank(method="first")
    bucket_codes = pd.qcut(ranked_price, q=bucket_count, labels=False)
    bucket_series = pd.Series(bucket_codes, index=price.index, dtype="float")
    return bucket_series.map(lambda value: f"Q{int(value) + 1}" if pd.notna(value) else "__PRICE_MISSING__").astype("string")


def build_model_frequency_bin(model_series: pd.Series) -> pd.Series:
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


def add_smoothed_target_encoding(
    fit_keys: pd.Series,
    fit_target: pd.Series,
    apply_keys: pd.Series,
    smoothing: float,
) -> pd.Series:
    global_mean = float(fit_target.mean())
    grouped = (
        pd.DataFrame({"key": fit_keys, "target": fit_target})
        .groupby("key", dropna=False)["target"]
        .agg(["mean", "count"])
    )
    grouped["encoded"] = (
        grouped["mean"] * grouped["count"] + global_mean * smoothing
    ) / (grouped["count"] + smoothing)
    encoded = apply_keys.map(grouped["encoded"]).fillna(global_mean)
    return encoded.astype(float)


def add_model_backoff_target_encoding(
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    apply_features: pd.DataFrame,
    smoothing: float,
    min_count: int,
) -> pd.Series:
    required_columns = {"model", "brand"}
    if not required_columns.issubset(fit_features.columns) or not required_columns.issubset(apply_features.columns):
        raise ValueError("Model backoff target encoding requires 'model' and 'brand' columns in features.")

    model_encoded = add_smoothed_target_encoding(
        fit_keys=fit_features["model"],
        fit_target=fit_target,
        apply_keys=apply_features["model"],
        smoothing=smoothing,
    )
    brand_encoded = add_smoothed_target_encoding(
        fit_keys=fit_features["brand"],
        fit_target=fit_target,
        apply_keys=apply_features["brand"],
        smoothing=smoothing,
    )

    model_counts = fit_features["model"].value_counts(dropna=False)
    apply_model_counts = apply_features["model"].map(model_counts).fillna(0)
    use_model_mask = apply_model_counts >= min_count

    encoded = brand_encoded.copy()
    encoded.loc[use_model_mask] = model_encoded.loc[use_model_mask]
    return encoded.astype(float)


def add_model_low_freq_flag(
    fit_features: pd.DataFrame,
    apply_features: pd.DataFrame,
    min_count: int,
) -> pd.Series:
    required_columns = {"model"}
    if not required_columns.issubset(fit_features.columns) or not required_columns.issubset(apply_features.columns):
        raise ValueError("Model low-frequency flag requires a 'model' column in features.")

    model_counts = fit_features["model"].value_counts(dropna=False)
    apply_model_counts = apply_features["model"].map(model_counts).fillna(0)
    return (apply_model_counts < min_count).astype(float)


def build_target_encoding_features(
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    apply_features: pd.DataFrame,
    use_brand_target_encoding: bool,
    use_brand_age_target_encoding: bool,
    use_model_target_encoding: bool,
    use_model_age_target_encoding: bool,
    use_model_backoff_target_encoding: bool,
    use_model_low_freq_flag: bool,
    model_backoff_min_count: int,
    model_low_freq_min_count: int,
    target_encoding_smoothing: float,
) -> pd.DataFrame:
    encoded_features: dict[str, pd.Series] = {}

    if use_brand_target_encoding:
        if "brand" not in fit_features.columns or "brand" not in apply_features.columns:
            raise ValueError("Brand target encoding requires a 'brand' column in features.")
        encoded_features["brand_target_mean"] = add_smoothed_target_encoding(
            fit_keys=fit_features["brand"],
            fit_target=fit_target,
            apply_keys=apply_features["brand"],
            smoothing=target_encoding_smoothing,
        )

    if use_brand_age_target_encoding:
        required_columns = {"brand", "car_age_years"}
        if not required_columns.issubset(fit_features.columns) or not required_columns.issubset(apply_features.columns):
            raise ValueError("Brand-age target encoding requires 'brand' and 'car_age_years' columns in features.")
        fit_age_bin = build_age_bin(fit_features["car_age_years"])
        apply_age_bin = build_age_bin(apply_features["car_age_years"])
        fit_key = fit_features["brand"].astype("string") + "|" + fit_age_bin
        apply_key = apply_features["brand"].astype("string") + "|" + apply_age_bin
        encoded_features["brand_age_target_mean"] = add_smoothed_target_encoding(
            fit_keys=fit_key,
            fit_target=fit_target,
            apply_keys=apply_key,
            smoothing=target_encoding_smoothing,
        )

    if use_model_target_encoding:
        if "model" not in fit_features.columns or "model" not in apply_features.columns:
            raise ValueError("Model target encoding requires a 'model' column in features.")
        encoded_features["model_target_mean"] = add_smoothed_target_encoding(
            fit_keys=fit_features["model"],
            fit_target=fit_target,
            apply_keys=apply_features["model"],
            smoothing=target_encoding_smoothing,
        )

    if use_model_backoff_target_encoding:
        encoded_features["model_backoff_target_mean"] = add_model_backoff_target_encoding(
            fit_features=fit_features,
            fit_target=fit_target,
            apply_features=apply_features,
            smoothing=target_encoding_smoothing,
            min_count=model_backoff_min_count,
        )

    if use_model_low_freq_flag:
        encoded_features["model_low_freq_flag"] = add_model_low_freq_flag(
            fit_features=fit_features,
            apply_features=apply_features,
            min_count=model_low_freq_min_count,
        )

    if use_model_age_target_encoding:
        required_columns = {"model", "car_age_years"}
        if not required_columns.issubset(fit_features.columns) or not required_columns.issubset(apply_features.columns):
            raise ValueError("Model-age target encoding requires 'model' and 'car_age_years' columns in features.")
        fit_age_bin = build_age_bin(fit_features["car_age_years"])
        apply_age_bin = build_age_bin(apply_features["car_age_years"])
        fit_key = fit_features["model"].astype("string") + "|" + fit_age_bin
        apply_key = apply_features["model"].astype("string") + "|" + apply_age_bin
        encoded_features["model_age_target_mean"] = add_smoothed_target_encoding(
            fit_keys=fit_key,
            fit_target=fit_target,
            apply_keys=apply_key,
            smoothing=target_encoding_smoothing,
        )

    return pd.DataFrame(encoded_features, index=apply_features.index)


def transform_target(y: pd.Series, target_mode: str | bool) -> pd.Series:
    if target_mode is True:
        target_mode = "log1p"
    elif target_mode is False:
        target_mode = "price"

    clipped = y.clip(lower=0)
    if target_mode == "log1p":
        return np.log1p(clipped)
    if target_mode == "sqrt":
        return np.sqrt(clipped)
    if target_mode == "pow075":
        return np.power(clipped, 0.75)
    if target_mode == "price":
        return y
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def inverse_target(y_pred: np.ndarray, target_mode: str | bool) -> np.ndarray:
    if target_mode is True:
        target_mode = "log1p"
    elif target_mode is False:
        target_mode = "price"

    if target_mode == "log1p":
        return np.expm1(y_pred).clip(min=0)
    if target_mode == "sqrt":
        return np.square(np.clip(y_pred, a_min=0, a_max=None))
    if target_mode == "pow075":
        return np.power(np.clip(y_pred, a_min=0, a_max=None), 1.0 / 0.75)
    if target_mode == "price":
        return y_pred.clip(min=0)
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def format_target_mode(target_mode: str | bool) -> str:
    if target_mode is True:
        target_mode = "log1p"
    elif target_mode is False:
        target_mode = "price"
    labels = {
        "log1p": "log1p(price)",
        "sqrt": "sqrt(price)",
        "pow075": "price^0.75",
        "price": "price",
    }
    if target_mode not in labels:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    return labels[target_mode]


def resolve_sample_weight_mode(sample_weight_mode: str | None, use_sample_weighting: bool) -> str:
    if sample_weight_mode is not None:
        return sample_weight_mode
    return "legacy" if use_sample_weighting else "none"


def normalize_age_label(age_label: str) -> str:
    stripped = age_label.strip()
    if stripped == "age_missing":
        return "__AGE_MISSING__"
    return stripped


def parse_price_age_slice_targets(raw_targets: str) -> set[tuple[str, str]]:
    targets: set[tuple[str, str]] = set()
    for raw_item in raw_targets.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" in item:
            price_bucket, age_bucket = item.split(":", maxsplit=1)
        elif "|" in item:
            price_bucket, age_bucket = item.split("|", maxsplit=1)
        else:
            raise ValueError(
                "price-age slice targets must use 'Q5:8y_plus' or 'Q5|8y_plus' format."
            )
        price_bucket = price_bucket.strip()
        age_bucket = normalize_age_label(age_bucket)
        if not price_bucket or not age_bucket:
            raise ValueError(f"Invalid price-age slice target: {raw_item!r}")
        targets.add((price_bucket, age_bucket))
    if not targets:
        raise ValueError("At least one price-age slice target is required.")
    return targets


def summarize_sample_weights(weights: np.ndarray | None, weighted_sample_count: int, sample_count: int) -> dict[str, float | int]:
    if weights is None:
        return {
            "weighted_sample_count": 0,
            "weight_mean": 1.0,
            "weight_min": 1.0,
            "weight_max": 1.0,
            "weight_non_default_count": 0,
        }
    return {
        "weighted_sample_count": int(weighted_sample_count),
        "weight_mean": float(np.mean(weights)) if sample_count > 0 else 1.0,
        "weight_min": float(np.min(weights)) if sample_count > 0 else 1.0,
        "weight_max": float(np.max(weights)) if sample_count > 0 else 1.0,
        "weight_non_default_count": int(np.sum(np.abs(weights - 1.0) > 1e-12)),
    }


def build_sample_weights(
    raw_target: pd.Series,
    train_features: pd.DataFrame,
    sample_weight_mode: str,
    high_price_quantile: float,
    high_price_weight: float,
    new_car_max_years: float,
    new_car_weight: float,
    price_age_slice_weight: float,
    price_age_slice_targets: str,
    normalize_sample_weight: bool,
) -> tuple[np.ndarray | None, dict[str, float | int]]:
    if sample_weight_mode == "none":
        return None, summarize_sample_weights(None, weighted_sample_count=0, sample_count=len(raw_target))

    weights = np.ones(len(raw_target), dtype=float)
    weighted_sample_count = 0

    if sample_weight_mode == "legacy":
        high_price_threshold = float(raw_target.quantile(high_price_quantile))
        high_price_mask = raw_target >= high_price_threshold
        weights = np.where(high_price_mask, weights * high_price_weight, weights)
        weighted_sample_count = int(high_price_mask.sum())

        if "car_age_years" in train_features.columns:
            car_age_years = pd.to_numeric(train_features["car_age_years"], errors="coerce")
            new_car_mask = car_age_years <= new_car_max_years
            weights = np.where(new_car_mask, weights * new_car_weight, weights)
            weighted_sample_count = int((high_price_mask | new_car_mask).sum())
    elif sample_weight_mode == "price_age_slice":
        if "car_age_years" not in train_features.columns:
            raise ValueError("price_age_slice sample weighting requires car_age_years feature.")
        target_pairs = parse_price_age_slice_targets(price_age_slice_targets)
        price_bin = build_price_quantile_bin(raw_target.astype(float), bucket_count=5)
        age_bin = build_age_bin(train_features["car_age_years"])
        target_mask = pd.Series(
            [(price_value, age_value) in target_pairs for price_value, age_value in zip(price_bin, age_bin)],
            index=raw_target.index,
            dtype=bool,
        )
        weights = np.where(target_mask.to_numpy(), weights * price_age_slice_weight, weights)
        weighted_sample_count = int(target_mask.sum())
    else:
        raise ValueError(f"Unsupported sample_weight_mode: {sample_weight_mode}")

    if normalize_sample_weight and len(weights) > 0:
        weight_mean = float(np.mean(weights))
        if weight_mean > 0:
            weights = weights / weight_mean

    return weights, summarize_sample_weights(
        weights,
        weighted_sample_count=weighted_sample_count,
        sample_count=len(raw_target),
    )


def has_target_encoding_features(
    use_brand_target_encoding: bool,
    use_brand_age_target_encoding: bool,
    use_model_target_encoding: bool,
    use_model_age_target_encoding: bool,
    use_model_backoff_target_encoding: bool,
    use_model_low_freq_flag: bool,
) -> bool:
    return any(
        [
            use_brand_target_encoding,
            use_brand_age_target_encoding,
            use_model_target_encoding,
            use_model_age_target_encoding,
            use_model_backoff_target_encoding,
            use_model_low_freq_flag,
        ]
    )


def prepare_model_inputs(
    prepared: PreparedData,
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    apply_features: pd.DataFrame,
    use_brand_target_encoding: bool,
    use_brand_age_target_encoding: bool,
    use_model_target_encoding: bool,
    use_model_age_target_encoding: bool,
    use_model_backoff_target_encoding: bool,
    use_model_low_freq_flag: bool,
    model_backoff_min_count: int,
    model_low_freq_min_count: int,
    target_encoding_smoothing: float,
) -> tuple[PreparedData, pd.DataFrame, pd.DataFrame]:
    prepared_for_fit = prepared
    fit_ready = fit_features.copy()
    apply_ready = apply_features.copy()

    if not has_target_encoding_features(
        use_brand_target_encoding=use_brand_target_encoding,
        use_brand_age_target_encoding=use_brand_age_target_encoding,
        use_model_target_encoding=use_model_target_encoding,
        use_model_age_target_encoding=use_model_age_target_encoding,
        use_model_backoff_target_encoding=use_model_backoff_target_encoding,
        use_model_low_freq_flag=use_model_low_freq_flag,
    ):
        return prepared_for_fit, fit_ready, apply_ready

    fit_encoded = build_target_encoding_features(
        fit_features=fit_ready,
        fit_target=fit_target,
        apply_features=fit_ready,
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
    apply_encoded = build_target_encoding_features(
        fit_features=fit_ready,
        fit_target=fit_target,
        apply_features=apply_ready,
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
    fit_ready = pd.concat([fit_ready, fit_encoded], axis=1)
    apply_ready = pd.concat([apply_ready, apply_encoded], axis=1)
    prepared_for_fit = replace(
        prepared,
        numeric_columns=prepared.numeric_columns + list(fit_encoded.columns),
    )
    return prepared_for_fit, fit_ready, apply_ready


def fit_predict_model(
    prepared: PreparedData,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    apply_features: pd.DataFrame,
    model_name: str,
    learning_rate: float,
    n_estimators: int,
    num_leaves: int,
    model_random_state: int,
    subsample: float,
    colsample_bytree: float,
    lightgbm_objective: str,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    if len(apply_features) == 0:
        return np.array([], dtype=float)

    preprocessor = build_preprocessor(prepared)
    model = get_model(
        model_name=model_name,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        model_random_state=model_random_state,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        lightgbm_objective=lightgbm_objective,
    )
    X_train_ready = preprocessor.fit_transform(train_features)
    X_apply_ready = preprocessor.transform(apply_features)

    if sample_weight is None:
        model.fit(X_train_ready, train_target)
    else:
        model.fit(X_train_ready, train_target, sample_weight=sample_weight)
    return model.predict(X_apply_ready)


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
) -> tuple[np.ndarray, dict[str, float | int | None]]:
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
    global_test_predictions = inverse_target(
        fit_predict_model(
            prepared=prepared_for_fit,
            train_features=X_train_ready,
            train_target=y_train,
            apply_features=X_test_ready,
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
        return final_predictions, details

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
        return final_predictions, details

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
    segment_predictions = inverse_target(
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
    final_predictions[routed_mask.to_numpy()] = segment_predictions
    return final_predictions, details


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
) -> None:
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

    with (output_dir / "cv_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

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
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    if test_predictions is not None:
        pd.DataFrame(
            {
                "SaleID": prepared.test_ids,
                "price": test_predictions,
            }
        ).to_csv(output_dir / "submission.csv", index=False)

    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the current used car price prediction pipeline.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the training file.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Path to the evaluation test file. Defaults to the current online testB set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics and predictions.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds used by the CV splitter.",
    )
    parser.add_argument(
        "--cv-strategy",
        choices=["kfold", "repeated_stratified"],
        default=DEFAULT_CV_STRATEGY,
        help="Cross-validation strategy used for model evaluation.",
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=DEFAULT_CV_REPEATS,
        help="Number of repeated shuffled runs when using repeated stratified CV.",
    )
    parser.add_argument(
        "--cv-random-state",
        type=int,
        default=DEFAULT_CV_RANDOM_STATE,
        help="Base random state used by the CV splitter.",
    )
    parser.add_argument(
        "--stratify-price-bins",
        type=int,
        default=DEFAULT_STRATIFY_PRICE_BINS,
        help="Number of target quantile bins used inside the repeated stratified CV label.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["price", "log1p", "sqrt", "pow075"],
        default="log1p",
        help="Target transformation used for training.",
    )
    parser.add_argument(
        "--use-group-stats",
        choices=["true", "false"],
        default="true",
        help="Whether to include non-target group statistics.",
    )
    parser.add_argument(
        "--use-power-bin",
        choices=["true", "false"],
        default="false",
        help="Whether to include the power bin feature.",
    )
    parser.add_argument(
        "--use-interactions",
        choices=["true", "false"],
        default="false",
        help="Whether to include interaction features.",
    )
    parser.add_argument(
        "--use-brand-relative",
        choices=["true", "false"],
        default="false",
        help="Whether to include brand-relative position features.",
    )
    parser.add_argument(
        "--use-power-age",
        choices=["true", "false"],
        default="false",
        help="Whether to include power-age interaction features.",
    )
    parser.add_argument(
        "--use-age-detail",
        choices=["true", "false"],
        default="false",
        help="Whether to include finer-grained age detail features.",
    )
    parser.add_argument(
        "--use-model-age-group-stats",
        choices=["true", "false"],
        default="false",
        help="Whether to include non-target model-by-age-bin support and median statistics.",
    )
    parser.add_argument(
        "--model-age-group-min-count",
        type=int,
        default=20,
        help="Minimum model-by-age-bin count required before using direct group medians.",
    )
    parser.add_argument(
        "--use-brand-target-encoding",
        choices=["true", "false"],
        default="false",
        help="Whether to include leak-safe brand OOF target encoding.",
    )
    parser.add_argument(
        "--use-brand-age-target-encoding",
        choices=["true", "false"],
        default="false",
        help="Whether to include leak-safe brand-by-age-bin OOF target encoding.",
    )
    parser.add_argument(
        "--use-model-target-encoding",
        choices=["true", "false"],
        default="false",
        help="Whether to include leak-safe model OOF target encoding.",
    )
    parser.add_argument(
        "--use-model-age-target-encoding",
        choices=["true", "false"],
        default="false",
        help="Whether to include leak-safe model-by-age-bin OOF target encoding.",
    )
    parser.add_argument(
        "--use-model-backoff-target-encoding",
        choices=["true", "false"],
        default="false",
        help="Whether to include leak-safe model target encoding with low-frequency brand/global backoff.",
    )
    parser.add_argument(
        "--model-backoff-min-count",
        type=int,
        default=20,
        help="Minimum training-fold count required before using direct model target encoding.",
    )
    parser.add_argument(
        "--use-model-low-freq-flag",
        choices=["true", "false"],
        default="false",
        help="Whether to include a low-frequency model indicator feature.",
    )
    parser.add_argument(
        "--model-low-freq-min-count",
        type=int,
        default=20,
        help="Minimum training-fold count below which a model is flagged as low frequency.",
    )
    parser.add_argument(
        "--target-encoding-smoothing",
        type=float,
        default=20.0,
        help="Smoothing strength for target encoding features.",
    )
    parser.add_argument(
        "--predict-test",
        choices=["true", "false"],
        default="false",
        help="Whether to generate predictions for the test set.",
    )
    parser.add_argument(
        "--use-segmented-modeling",
        choices=["true", "false"],
        default="false",
        help="Whether to use a hard-routed segment model for the highest-price older-car segment.",
    )
    parser.add_argument(
        "--segment-routing-mode",
        choices=["global_pred_plus_age"],
        default="global_pred_plus_age",
        help="Routing rule used to decide whether a sample should use the segment model.",
    )
    parser.add_argument(
        "--segment-scope",
        choices=["q5_5plus"],
        default="q5_5plus",
        help="Definition of the target high-price older-car segment.",
    )
    parser.add_argument(
        "--model-name",
        choices=["gbrt", "lightgbm"],
        default=DEFAULT_MODEL_NAME,
        help="Model backend to use.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for the selected model.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Number of boosting stages for the selected model.",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=DEFAULT_NUM_LEAVES,
        help="Number of leaves for LightGBM.",
    )
    parser.add_argument(
        "--model-random-state",
        type=int,
        default=42,
        help="Random state used by the underlying model.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row sampling ratio used by the underlying model.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Feature sampling ratio used by LightGBM.",
    )
    parser.add_argument(
        "--lightgbm-objective",
        choices=["regression", "mae"],
        default="regression",
        help="Objective function used by LightGBM.",
    )
    parser.add_argument(
        "--use-sample-weighting",
        choices=["true", "false"],
        default="false",
        help="Whether to upweight hard samples such as high-price and newer cars.",
    )
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "legacy", "price_age_slice"],
        default=None,
        help="Sample weighting mode. Defaults to legacy when --use-sample-weighting=true, otherwise none.",
    )
    parser.add_argument(
        "--high-price-quantile",
        type=float,
        default=0.8,
        help="Quantile threshold used to define high-price samples for weighting.",
    )
    parser.add_argument(
        "--high-price-weight",
        type=float,
        default=1.5,
        help="Weight multiplier applied to high-price samples.",
    )
    parser.add_argument(
        "--new-car-max-years",
        type=float,
        default=3.0,
        help="Maximum car age in years treated as newer-car samples for weighting.",
    )
    parser.add_argument(
        "--new-car-weight",
        type=float,
        default=1.3,
        help="Weight multiplier applied to newer-car samples.",
    )
    parser.add_argument(
        "--price-age-slice-weight",
        type=float,
        default=1.1,
        help="Weight multiplier applied to selected price-by-age slices.",
    )
    parser.add_argument(
        "--price-age-slice-targets",
        type=str,
        default=DEFAULT_PRICE_AGE_SLICE_TARGETS,
        help="Comma-separated slice targets, e.g. Q5:8y_plus,Q4:8y_plus.",
    )
    parser.add_argument(
        "--normalize-sample-weight",
        choices=["true", "false"],
        default="true",
        help="Whether to normalize sample weights back to mean 1.0.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=None,
        help="Optional number of training rows to sample before CV and fitting.",
    )
    parser.add_argument(
        "--sample-random-state",
        type=int,
        default=42,
        help="Random state used when sampling rows for smaller experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.train_path.exists():
        raise FileNotFoundError(f"Training file not found: {args.train_path}")
    if not args.test_path.exists():
        raise FileNotFoundError(f"Test file not found: {args.test_path}")

    use_group_stats = args.use_group_stats == "true"
    use_power_bin = args.use_power_bin == "true"
    use_interactions = args.use_interactions == "true"
    use_brand_relative = args.use_brand_relative == "true"
    use_power_age = args.use_power_age == "true"
    use_age_detail = args.use_age_detail == "true"
    use_model_age_group_stats = args.use_model_age_group_stats == "true"
    use_brand_target_encoding = args.use_brand_target_encoding == "true"
    use_brand_age_target_encoding = args.use_brand_age_target_encoding == "true"
    use_model_target_encoding = args.use_model_target_encoding == "true"
    use_model_age_target_encoding = args.use_model_age_target_encoding == "true"
    use_model_backoff_target_encoding = args.use_model_backoff_target_encoding == "true"
    use_model_low_freq_flag = args.use_model_low_freq_flag == "true"
    predict_test = args.predict_test == "true"
    use_segmented_modeling = args.use_segmented_modeling == "true"
    model_name = args.model_name
    learning_rate = args.learning_rate
    n_estimators = args.n_estimators
    num_leaves = args.num_leaves
    model_random_state = args.model_random_state
    subsample = args.subsample
    colsample_bytree = args.colsample_bytree
    lightgbm_objective = args.lightgbm_objective
    use_sample_weighting = args.use_sample_weighting == "true"
    sample_weight_mode = resolve_sample_weight_mode(args.sample_weight_mode, use_sample_weighting)
    high_price_quantile = args.high_price_quantile
    high_price_weight = args.high_price_weight
    new_car_max_years = args.new_car_max_years
    new_car_weight = args.new_car_weight
    price_age_slice_weight = args.price_age_slice_weight
    price_age_slice_targets = args.price_age_slice_targets
    normalize_sample_weight = args.normalize_sample_weight == "true"
    model_backoff_min_count = args.model_backoff_min_count
    model_low_freq_min_count = args.model_low_freq_min_count
    model_age_group_min_count = args.model_age_group_min_count
    target_encoding_smoothing = args.target_encoding_smoothing
    segment_routing_mode = args.segment_routing_mode
    segment_scope = args.segment_scope

    train_df, test_df = load_data(args.train_path, args.test_path)
    train_df = sample_rows(
        df=train_df,
        sample_size=args.train_sample_size,
        random_state=args.sample_random_state,
    )
    prepared = prepare_features(
        train_df,
        test_df,
        use_group_stats=use_group_stats,
        use_power_bin=use_power_bin,
        use_interactions=use_interactions,
        use_brand_relative=use_brand_relative,
        use_power_age=use_power_age,
        use_age_detail=use_age_detail,
        use_model_age_group_stats=use_model_age_group_stats,
        model_age_group_min_count=model_age_group_min_count,
    )
    target_mode = args.target_mode

    fold_metrics, oof_predictions, oof_extra_columns, cv_metadata = cross_validate_train(
        prepared=prepared,
        n_splits=args.n_splits,
        cv_strategy=args.cv_strategy,
        cv_repeats=args.cv_repeats,
        cv_random_state=args.cv_random_state,
        stratify_price_bins=args.stratify_price_bins,
        use_log_target=target_mode,
        model_name=model_name,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        model_random_state=model_random_state,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        lightgbm_objective=lightgbm_objective,
        use_sample_weighting=use_sample_weighting,
        sample_weight_mode=sample_weight_mode,
        high_price_quantile=high_price_quantile,
        high_price_weight=high_price_weight,
        new_car_max_years=new_car_max_years,
        new_car_weight=new_car_weight,
        price_age_slice_weight=price_age_slice_weight,
        price_age_slice_targets=price_age_slice_targets,
        normalize_sample_weight=normalize_sample_weight,
        use_brand_target_encoding=use_brand_target_encoding,
        use_brand_age_target_encoding=use_brand_age_target_encoding,
        use_model_target_encoding=use_model_target_encoding,
        use_model_age_target_encoding=use_model_age_target_encoding,
        use_model_backoff_target_encoding=use_model_backoff_target_encoding,
        use_model_low_freq_flag=use_model_low_freq_flag,
        model_backoff_min_count=model_backoff_min_count,
        model_low_freq_min_count=model_low_freq_min_count,
        target_encoding_smoothing=target_encoding_smoothing,
        use_segmented_modeling=use_segmented_modeling,
        segment_routing_mode=segment_routing_mode,
        segment_scope=segment_scope,
    )

    test_predictions = None
    prediction_details = None
    if predict_test:
        test_predictions, prediction_details = fit_full_and_predict(
            prepared=prepared,
            use_log_target=target_mode,
            model_name=model_name,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            model_random_state=model_random_state,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            lightgbm_objective=lightgbm_objective,
            use_sample_weighting=use_sample_weighting,
            sample_weight_mode=sample_weight_mode,
            high_price_quantile=high_price_quantile,
            high_price_weight=high_price_weight,
            new_car_max_years=new_car_max_years,
            new_car_weight=new_car_weight,
            price_age_slice_weight=price_age_slice_weight,
            price_age_slice_targets=price_age_slice_targets,
            normalize_sample_weight=normalize_sample_weight,
            use_brand_target_encoding=use_brand_target_encoding,
            use_brand_age_target_encoding=use_brand_age_target_encoding,
            use_model_target_encoding=use_model_target_encoding,
            use_model_age_target_encoding=use_model_age_target_encoding,
            use_model_backoff_target_encoding=use_model_backoff_target_encoding,
            use_model_low_freq_flag=use_model_low_freq_flag,
            model_backoff_min_count=model_backoff_min_count,
            model_low_freq_min_count=model_low_freq_min_count,
            target_encoding_smoothing=target_encoding_smoothing,
            use_segmented_modeling=use_segmented_modeling,
            segment_routing_mode=segment_routing_mode,
            segment_scope=segment_scope,
        )

    save_outputs(
        output_dir=args.output_dir,
        prepared=prepared,
        fold_metrics=fold_metrics,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        use_log_target=target_mode,
        model_name=model_name,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        model_random_state=model_random_state,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        lightgbm_objective=lightgbm_objective,
        use_sample_weighting=use_sample_weighting,
        sample_weight_mode=sample_weight_mode,
        high_price_quantile=high_price_quantile,
        high_price_weight=high_price_weight,
        new_car_max_years=new_car_max_years,
        new_car_weight=new_car_weight,
        price_age_slice_weight=price_age_slice_weight,
        price_age_slice_targets=price_age_slice_targets,
        normalize_sample_weight=normalize_sample_weight,
        use_brand_target_encoding=use_brand_target_encoding,
        use_brand_age_target_encoding=use_brand_age_target_encoding,
        use_model_target_encoding=use_model_target_encoding,
        use_model_age_target_encoding=use_model_age_target_encoding,
        use_model_backoff_target_encoding=use_model_backoff_target_encoding,
        model_backoff_min_count=model_backoff_min_count,
        use_model_low_freq_flag=use_model_low_freq_flag,
        model_low_freq_min_count=model_low_freq_min_count,
        target_encoding_smoothing=target_encoding_smoothing,
        use_model_age_group_stats=use_model_age_group_stats,
        model_age_group_min_count=model_age_group_min_count,
        use_segmented_modeling=use_segmented_modeling,
        segment_routing_mode=segment_routing_mode,
        segment_scope=segment_scope,
        cv_metadata=cv_metadata,
        oof_extra_columns=oof_extra_columns,
        prediction_details=prediction_details,
    )


if __name__ == "__main__":
    main()
