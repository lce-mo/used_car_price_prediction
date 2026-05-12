"""Training entrypoint and shared model-fitting helpers.

Responsibilities:
- Own the CLI/main training workflow formerly implemented in src/train.py.
- Load raw train/test data with the verified single-space separator.
- Build PreparedData through src.features, then call cross-validation, optional
  full-fit prediction, and artifact saving.
- Keep shared helpers used by cross_validation.py and predict_model.py during
  the migration: target transforms, sample weighting, preprocessing, fold-safe
  target encoding, and one-model fit/predict.

Dependencies:
- src.features.prepare_features for feature engineering and PreparedData.
- src.models.cross_validation.cross_validate_train for OOF training.
- src.models.predict_model.fit_full_and_predict for test prediction.
- src.models.model_registry.get_model/save_outputs for model construction and
  training artifacts.

TODO:
- Split target transforms, sample weighting, and preprocessing into dedicated
  modules only if they become real implementations rather than re-export shells.
- Reuse src.features.price_proxy_features for target encoding to remove the
  duplicate encoding implementation currently kept here for compatibility.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    from src.config import CV_CONFIG, FEATURE_CONFIG, MODEL_CONFIG, SAMPLE_WEIGHT_CONFIG, TRAINING_PATH_CONFIG
    from src.features import prepare_features
    from src.models.model_registry import get_model
except ModuleNotFoundError:
    from config import CV_CONFIG, FEATURE_CONFIG, MODEL_CONFIG, SAMPLE_WEIGHT_CONFIG, TRAINING_PATH_CONFIG
    from features import prepare_features
    from models.model_registry import get_model


DEFAULT_TRAIN_PATH = TRAINING_PATH_CONFIG.train_path
DEFAULT_TEST_PATH = TRAINING_PATH_CONFIG.test_path
DEFAULT_OUTPUT_DIR = TRAINING_PATH_CONFIG.output_dir
DEFAULT_MODEL_NAME = MODEL_CONFIG.model_name
DEFAULT_LEARNING_RATE = MODEL_CONFIG.learning_rate
DEFAULT_N_ESTIMATORS = MODEL_CONFIG.n_estimators
DEFAULT_NUM_LEAVES = MODEL_CONFIG.num_leaves
DEFAULT_CV_STRATEGY = CV_CONFIG.strategy
DEFAULT_CV_REPEATS = CV_CONFIG.repeats
DEFAULT_CV_RANDOM_STATE = CV_CONFIG.random_state
DEFAULT_STRATIFY_PRICE_BINS = CV_CONFIG.stratify_price_bins
DEFAULT_PRICE_AGE_SLICE_TARGETS = SAMPLE_WEIGHT_CONFIG.price_age_slice_targets
RAW_DATA_SEPARATOR = " "


def train(argv: Sequence[str] | None = None) -> None:
    """Run the current training pipeline through the standardized entrypoint."""
    main(argv)


def validate_raw_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Validate that raw Tianchi files were read with the required single-space separator."""
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
    """Load raw train/test data with the corrected separator and validation guard."""
    train_df = pd.read_csv(train_path, sep=RAW_DATA_SEPARATOR)
    test_df = pd.read_csv(test_path, sep=RAW_DATA_SEPARATOR)
    validate_raw_dataframe(train_df, train_path)
    validate_raw_dataframe(test_df, test_path)
    return train_df, test_df


def sample_rows(df: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    """Optionally sample training rows for faster local experiments."""
    if sample_size is None:
        return df
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")
    if sample_size >= len(df):
        return df.copy()
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
def normalize_target_mode(target_mode: str | bool) -> str:
    """Normalize legacy boolean target-mode flags to explicit names."""
    if target_mode is True:
        return "log1p"
    if target_mode is False:
        return "price"
    return str(target_mode)
def transform_target(y: pd.Series, target_mode: str | bool) -> pd.Series:
    """Transform the target before model fitting."""
    mode = normalize_target_mode(target_mode)
    clipped = y.clip(lower=0)
    if mode == "log1p":
        return np.log1p(clipped)
    if mode == "sqrt":
        return np.sqrt(clipped)
    if mode == "pow075":
        return np.power(clipped, 0.75)
    if mode == "price":
        return y
    raise ValueError(f"Unsupported target_mode: {target_mode}")
def inverse_target(y_pred: np.ndarray, target_mode: str | bool) -> np.ndarray:
    """Invert target transformation and clip predictions to non-negative prices."""
    mode = normalize_target_mode(target_mode)
    if mode == "log1p":
        return np.expm1(y_pred).clip(min=0)
    if mode == "sqrt":
        return np.square(np.clip(y_pred, a_min=0, a_max=None))
    if mode == "pow075":
        return np.power(np.clip(y_pred, a_min=0, a_max=None), 1.0 / 0.75)
    if mode == "price":
        return y_pred.clip(min=0)
    raise ValueError(f"Unsupported target_mode: {target_mode}")
def format_target_mode(target_mode: str | bool) -> str:
    """Return a human-readable target transformation label."""
    mode = normalize_target_mode(target_mode)
    labels = {
        "log1p": "log1p(price)",
        "sqrt": "sqrt(price)",
        "pow075": "price^0.75",
        "price": "price",
    }
    if mode not in labels:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    return labels[mode]
def build_preprocessor(prepared: Any) -> ColumnTransformer:
    """Build the tabular preprocessing pipeline used by the legacy trainer."""
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
def build_age_bin(series: pd.Series) -> pd.Series:
    """Bucket car age into stable labels for slicing and stratification."""
    age_years = pd.to_numeric(series, errors="coerce").clip(lower=0)
    bins = pd.cut(
        age_years,
        bins=[-np.inf, 1, 3, 5, 8, np.inf],
        labels=["0_1y", "1_3y", "3_5y", "5_8y", "8y_plus"],
    )
    return bins.astype("string").fillna("__AGE_MISSING__")
def build_price_quantile_bin(price: pd.Series, bucket_count: int) -> pd.Series:
    """Build stable price quantile labels with deterministic rank tie handling."""
    ranked_price = price.rank(method="first")
    bucket_codes = pd.qcut(ranked_price, q=bucket_count, labels=False)
    bucket_series = pd.Series(bucket_codes, index=price.index, dtype="float")
    return bucket_series.map(
        lambda value: f"Q{int(value) + 1}" if pd.notna(value) else "__PRICE_MISSING__"
    ).astype("string")
def resolve_sample_weight_mode(sample_weight_mode: str | None, use_sample_weighting: bool) -> str:
    """Resolve legacy boolean weighting flag and explicit mode into one mode string."""
    if sample_weight_mode is not None:
        return sample_weight_mode
    return "legacy" if use_sample_weighting else "none"
def normalize_age_label(age_label: str) -> str:
    """Normalize CLI age bucket labels to internal labels."""
    stripped = age_label.strip()
    if stripped == "age_missing":
        return "__AGE_MISSING__"
    return stripped
def parse_price_age_slice_targets(raw_targets: str) -> set[tuple[str, str]]:
    """Parse price-age sample weighting targets such as Q5:8y_plus."""
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
def summarize_sample_weights(
    weights: np.ndarray | None,
    weighted_sample_count: int,
    sample_count: int,
) -> dict[str, float | int]:
    """Summarize sample weights for experiment metadata."""
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
    """Build sample weights for the legacy training modes."""
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
def add_smoothed_target_encoding_with_backoff(
    fit_keys: pd.Series,
    fit_target: pd.Series,
    apply_keys: pd.Series,
    smoothing: float,
    min_count: int,
    fallback: pd.Series,
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

    counts = apply_keys.map(grouped["count"]).fillna(0)
    encoded = apply_keys.map(grouped["encoded"])
    fallback = fallback.reindex(apply_keys.index).fillna(global_mean)
    return encoded.where(counts >= min_count, fallback).fillna(fallback).astype(float)
def build_power_bin(series: pd.Series) -> pd.Series:
    power = pd.to_numeric(series, errors="coerce").clip(lower=0, upper=600)
    bucket = pd.cut(
        power,
        bins=[-1, 0, 60, 90, 120, 150, 200, 300, 600],
        labels=["0", "1_60", "60_90", "90_120", "120_150", "150_200", "200_300", "300_600"],
    )
    return bucket.astype("string").fillna("__POWER_MISSING__")
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
def add_model_power_age_backoff_target_encoding(
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    apply_features: pd.DataFrame,
    smoothing: float,
    min_count: int,
) -> pd.Series:
    required_columns = {"model", "brand", "power", "car_age_years"}
    if not required_columns.issubset(fit_features.columns) or not required_columns.issubset(apply_features.columns):
        raise ValueError(
            "Model-power-age backoff target encoding requires 'model', 'brand', 'power', and 'car_age_years' columns."
        )

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
    fallback = brand_encoded.copy()
    fallback.loc[apply_model_counts >= min_count] = model_encoded.loc[apply_model_counts >= min_count]

    fit_power_bin = build_power_bin(fit_features["power"])
    apply_power_bin = build_power_bin(apply_features["power"])
    fit_age_bin = build_age_bin(fit_features["car_age_years"])
    apply_age_bin = build_age_bin(apply_features["car_age_years"])
    fit_key = fit_features["model"].astype("string") + "|" + fit_power_bin + "|" + fit_age_bin
    apply_key = apply_features["model"].astype("string") + "|" + apply_power_bin + "|" + apply_age_bin

    return add_smoothed_target_encoding_with_backoff(
        fit_keys=fit_key,
        fit_target=fit_target,
        apply_keys=apply_key,
        smoothing=smoothing,
        min_count=min_count,
        fallback=fallback,
    )
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
        encoded_features["model_power_age_backoff_target_mean"] = add_model_power_age_backoff_target_encoding(
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
        model_fallback = add_smoothed_target_encoding(
            fit_keys=fit_features["model"],
            fit_target=fit_target,
            apply_keys=apply_features["model"],
            smoothing=target_encoding_smoothing,
        )
        if "brand" in fit_features.columns and "brand" in apply_features.columns:
            brand_encoded = add_smoothed_target_encoding(
                fit_keys=fit_features["brand"],
                fit_target=fit_target,
                apply_keys=apply_features["brand"],
                smoothing=target_encoding_smoothing,
            )
            model_counts = fit_features["model"].value_counts(dropna=False)
            apply_model_counts = apply_features["model"].map(model_counts).fillna(0)
            model_fallback = brand_encoded.where(apply_model_counts < model_backoff_min_count, model_fallback)
        encoded_features["model_age_target_mean"] = add_smoothed_target_encoding_with_backoff(
            fit_keys=fit_key,
            fit_target=fit_target,
            apply_keys=apply_key,
            smoothing=target_encoding_smoothing,
            min_count=model_backoff_min_count,
            fallback=model_fallback,
        )

        if "power" in fit_features.columns and "power" in apply_features.columns:
            fit_power_bin = build_power_bin(fit_features["power"])
            apply_power_bin = build_power_bin(apply_features["power"])
            fit_power_age_key = fit_power_bin + "|" + fit_age_bin
            apply_power_age_key = apply_power_bin + "|" + apply_age_bin
            power_bin_fallback = add_smoothed_target_encoding(
                fit_keys=fit_power_bin,
                fit_target=fit_target,
                apply_keys=apply_power_bin,
                smoothing=target_encoding_smoothing,
            )
            encoded_features["power_age_bin_target_mean"] = add_smoothed_target_encoding_with_backoff(
                fit_keys=fit_power_age_key,
                fit_target=fit_target,
                apply_keys=apply_power_age_key,
                smoothing=target_encoding_smoothing,
                min_count=model_backoff_min_count,
                fallback=power_bin_fallback,
            )

    return pd.DataFrame(encoded_features, index=apply_features.index)
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
    prepared,
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
    """Fit one model and predict another feature matrix."""
    if len(apply_features) == 0:
        return np.array([], dtype=float)

    artifact = fit_model_artifact(
        prepared=prepared,
        train_features=train_features,
        train_target=train_target,
        model_name=model_name,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        model_random_state=model_random_state,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        lightgbm_objective=lightgbm_objective,
        sample_weight=sample_weight,
    )
    preprocessor = artifact["preprocessor"]
    model = artifact["model"]
    X_apply_ready = preprocessor.transform(apply_features)
    return model.predict(X_apply_ready)


def fit_model_artifact(
    prepared,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    model_name: str,
    learning_rate: float,
    n_estimators: int,
    num_leaves: int,
    model_random_state: int,
    subsample: float,
    colsample_bytree: float,
    lightgbm_objective: str,
    sample_weight: np.ndarray | None = None,
) -> dict[str, object]:
    """Fit one model and keep the fitted preprocessing/model objects."""
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

    if sample_weight is None:
        model.fit(X_train_ready, train_target)
    else:
        model.fit(X_train_ready, train_target, sample_weight=sample_weight)
    return {
        "preprocessor": preprocessor,
        "model": model,
        "feature_columns": list(train_features.columns),
        "numeric_columns": list(prepared.numeric_columns),
        "categorical_columns": list(prepared.categorical_columns),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the training CLI arguments used by the current validated pipeline."""
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
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds used by the CV splitter.")
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
        default=MODEL_CONFIG.target_mode,
        help="Target transformation used for training.",
    )
    parser.add_argument(
        "--use-group-stats",
        choices=["true", "false"],
        default="true" if FEATURE_CONFIG.use_group_stats else "false",
        help="Whether to include non-target group statistics.",
    )
    parser.add_argument(
        "--use-power-bin",
        choices=["true", "false"],
        default="true" if FEATURE_CONFIG.use_power_bin else "false",
        help="Whether to include the power bin feature.",
    )
    parser.add_argument(
        "--use-interactions",
        choices=["true", "false"],
        default="true" if FEATURE_CONFIG.use_interactions else "false",
        help="Whether to include interaction features.",
    )
    parser.add_argument(
        "--use-brand-relative",
        choices=["true", "false"],
        default="true" if FEATURE_CONFIG.use_brand_relative else "false",
        help="Whether to include brand-relative position features.",
    )
    parser.add_argument(
        "--use-power-age",
        choices=["true", "false"],
        default="true" if FEATURE_CONFIG.use_power_age else "false",
        help="Whether to include power-age interaction features.",
    )
    parser.add_argument(
        "--use-age-detail",
        choices=["true", "false"],
        default="true" if FEATURE_CONFIG.use_age_detail else "false",
        help="Whether to include finer-grained age detail features.",
    )
    parser.add_argument(
        "--use-model-age-group-stats",
        choices=["true", "false"],
        default="true" if FEATURE_CONFIG.use_model_age_group_stats else "false",
        help="Whether to include non-target model-by-age-bin support and median statistics.",
    )
    parser.add_argument(
        "--model-age-group-min-count",
        type=int,
        default=FEATURE_CONFIG.model_age_group_min_count,
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
        default=FEATURE_CONFIG.target_encoding_smoothing,
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
    parser.add_argument("--model-name", choices=["gbrt", "lightgbm"], default=DEFAULT_MODEL_NAME)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--num-leaves", type=int, default=DEFAULT_NUM_LEAVES)
    parser.add_argument("--model-random-state", type=int, default=MODEL_CONFIG.random_state)
    parser.add_argument("--subsample", type=float, default=MODEL_CONFIG.subsample)
    parser.add_argument("--colsample-bytree", type=float, default=MODEL_CONFIG.colsample_bytree)
    parser.add_argument(
        "--lightgbm-objective",
        choices=["regression", "mae"],
        default=MODEL_CONFIG.lightgbm_objective,
        help="Objective function used by LightGBM.",
    )
    parser.add_argument(
        "--use-sample-weighting",
        choices=["true", "false"],
        default="true" if SAMPLE_WEIGHT_CONFIG.use_sample_weighting else "false",
        help="Whether to upweight hard samples such as high-price and newer cars.",
    )
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "legacy", "price_age_slice"],
        default=None,
        help="Sample weighting mode. Defaults to legacy when --use-sample-weighting=true, otherwise none.",
    )
    parser.add_argument("--high-price-quantile", type=float, default=SAMPLE_WEIGHT_CONFIG.high_price_quantile)
    parser.add_argument("--high-price-weight", type=float, default=SAMPLE_WEIGHT_CONFIG.high_price_weight)
    parser.add_argument("--new-car-max-years", type=float, default=SAMPLE_WEIGHT_CONFIG.new_car_max_years)
    parser.add_argument("--new-car-weight", type=float, default=SAMPLE_WEIGHT_CONFIG.new_car_weight)
    parser.add_argument("--price-age-slice-weight", type=float, default=SAMPLE_WEIGHT_CONFIG.price_age_slice_weight)
    parser.add_argument(
        "--price-age-slice-targets",
        type=str,
        default=DEFAULT_PRICE_AGE_SLICE_TARGETS,
        help="Comma-separated slice targets, e.g. Q5:8y_plus,Q4:8y_plus.",
    )
    parser.add_argument(
        "--normalize-sample-weight",
        choices=["true", "false"],
        default="true" if SAMPLE_WEIGHT_CONFIG.normalize_sample_weight else "false",
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
        default=MODEL_CONFIG.random_state,
        help="Random state used when sampling rows for smaller experiments.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Train the current model pipeline, evaluate CV, and optionally write test predictions."""
    try:
        from src.models.cross_validation import cross_validate_train
        from src.models.model_registry import save_outputs
        from src.models.predict_model import fit_full_and_predict
    except ModuleNotFoundError:
        from models.cross_validation import cross_validate_train
        from models.model_registry import save_outputs
        from models.predict_model import fit_full_and_predict

    args = parse_args(argv)

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
    use_sample_weighting = args.use_sample_weighting == "true"
    sample_weight_mode = resolve_sample_weight_mode(args.sample_weight_mode, use_sample_weighting)
    normalize_sample_weight = args.normalize_sample_weight == "true"

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
        model_age_group_min_count=args.model_age_group_min_count,
    )

    fold_metrics, oof_predictions, oof_extra_columns, cv_metadata = cross_validate_train(
        prepared=prepared,
        n_splits=args.n_splits,
        cv_strategy=args.cv_strategy,
        cv_repeats=args.cv_repeats,
        cv_random_state=args.cv_random_state,
        stratify_price_bins=args.stratify_price_bins,
        use_log_target=args.target_mode,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        model_random_state=args.model_random_state,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        lightgbm_objective=args.lightgbm_objective,
        use_sample_weighting=use_sample_weighting,
        sample_weight_mode=sample_weight_mode,
        high_price_quantile=args.high_price_quantile,
        high_price_weight=args.high_price_weight,
        new_car_max_years=args.new_car_max_years,
        new_car_weight=args.new_car_weight,
        price_age_slice_weight=args.price_age_slice_weight,
        price_age_slice_targets=args.price_age_slice_targets,
        normalize_sample_weight=normalize_sample_weight,
        use_brand_target_encoding=use_brand_target_encoding,
        use_brand_age_target_encoding=use_brand_age_target_encoding,
        use_model_target_encoding=use_model_target_encoding,
        use_model_age_target_encoding=use_model_age_target_encoding,
        use_model_backoff_target_encoding=use_model_backoff_target_encoding,
        use_model_low_freq_flag=use_model_low_freq_flag,
        model_backoff_min_count=args.model_backoff_min_count,
        model_low_freq_min_count=args.model_low_freq_min_count,
        target_encoding_smoothing=args.target_encoding_smoothing,
        use_segmented_modeling=use_segmented_modeling,
        segment_routing_mode=args.segment_routing_mode,
        segment_scope=args.segment_scope,
    )

    test_predictions = None
    prediction_details = None
    model_artifact = None
    if predict_test:
        test_predictions, prediction_details, model_artifact = fit_full_and_predict(
            prepared=prepared,
            use_log_target=args.target_mode,
            model_name=args.model_name,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            num_leaves=args.num_leaves,
            model_random_state=args.model_random_state,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            lightgbm_objective=args.lightgbm_objective,
            use_sample_weighting=use_sample_weighting,
            sample_weight_mode=sample_weight_mode,
            high_price_quantile=args.high_price_quantile,
            high_price_weight=args.high_price_weight,
            new_car_max_years=args.new_car_max_years,
            new_car_weight=args.new_car_weight,
            price_age_slice_weight=args.price_age_slice_weight,
            price_age_slice_targets=args.price_age_slice_targets,
            normalize_sample_weight=normalize_sample_weight,
            use_brand_target_encoding=use_brand_target_encoding,
            use_brand_age_target_encoding=use_brand_age_target_encoding,
            use_model_target_encoding=use_model_target_encoding,
            use_model_age_target_encoding=use_model_age_target_encoding,
            use_model_backoff_target_encoding=use_model_backoff_target_encoding,
            use_model_low_freq_flag=use_model_low_freq_flag,
            model_backoff_min_count=args.model_backoff_min_count,
            model_low_freq_min_count=args.model_low_freq_min_count,
            target_encoding_smoothing=args.target_encoding_smoothing,
            use_segmented_modeling=use_segmented_modeling,
            segment_routing_mode=args.segment_routing_mode,
            segment_scope=args.segment_scope,
        )

    save_outputs(
        output_dir=args.output_dir,
        prepared=prepared,
        fold_metrics=fold_metrics,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        use_log_target=args.target_mode,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        model_random_state=args.model_random_state,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        lightgbm_objective=args.lightgbm_objective,
        use_sample_weighting=use_sample_weighting,
        sample_weight_mode=sample_weight_mode,
        high_price_quantile=args.high_price_quantile,
        high_price_weight=args.high_price_weight,
        new_car_max_years=args.new_car_max_years,
        new_car_weight=args.new_car_weight,
        price_age_slice_weight=args.price_age_slice_weight,
        price_age_slice_targets=args.price_age_slice_targets,
        normalize_sample_weight=normalize_sample_weight,
        use_brand_target_encoding=use_brand_target_encoding,
        use_brand_age_target_encoding=use_brand_age_target_encoding,
        use_model_target_encoding=use_model_target_encoding,
        use_model_age_target_encoding=use_model_age_target_encoding,
        use_model_backoff_target_encoding=use_model_backoff_target_encoding,
        model_backoff_min_count=args.model_backoff_min_count,
        use_model_low_freq_flag=use_model_low_freq_flag,
        model_low_freq_min_count=args.model_low_freq_min_count,
        target_encoding_smoothing=args.target_encoding_smoothing,
        use_model_age_group_stats=use_model_age_group_stats,
        model_age_group_min_count=args.model_age_group_min_count,
        use_segmented_modeling=use_segmented_modeling,
        segment_routing_mode=args.segment_routing_mode,
        segment_scope=args.segment_scope,
        cv_metadata=cv_metadata,
        oof_extra_columns=oof_extra_columns,
        prediction_details=prediction_details,
        model_artifact=model_artifact,
    )


if __name__ == "__main__":
    main()
