from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .date_features import build_age_bin
from .depreciation_features import build_power_bin


TARGET_ENCODING_WARNING = """
Price proxy statistics such as brand_price_mean and model_price_mean are target
encoding features. They must be built with KFold or out-of-fold logic. Do not
fit them from the full training price column and then use those values as train
features, because that leaks target information into validation.
"""

DEFAULT_PRICE_PROXY_COLUMNS = ("brand", "model")
MISSING_CATEGORY = "__MISSING__"
SUPPORTED_TARGET_SPACES = ("raw", "log1p", "sqrt")


@dataclass
class PriceProxyEncoder:
    """Full-train target encoder used only for applying features to holdout/test data."""

    columns: tuple[str, ...]
    feature_suffix: str
    smoothing: float
    target_space: str
    global_mean: float
    mappings: dict[str, pd.Series]


def normalize_target_space(target_space: str | bool) -> str:
    """Normalize legacy boolean flags and explicit names to target-space names."""
    if target_space is True:
        return "log1p"
    if target_space is False:
        return "raw"
    normalized = str(target_space)
    if normalized == "price":
        return "raw"
    if normalized not in SUPPORTED_TARGET_SPACES:
        raise ValueError(f"Unsupported target_space: {target_space}")
    return normalized


def transform_target_for_encoding(target: pd.Series, target_space: str | bool = "raw") -> pd.Series:
    """Transform target values before fitting price-proxy statistics."""
    space = normalize_target_space(target_space)
    target_numeric = pd.to_numeric(target, errors="coerce").astype(float)
    clipped = target_numeric.clip(lower=0)
    if space == "log1p":
        return np.log1p(clipped)
    if space == "sqrt":
        return np.sqrt(clipped)
    return target_numeric


def _as_target_series(train_df: pd.DataFrame, target: str | pd.Series | Sequence[float]) -> pd.Series:
    if isinstance(target, str):
        if target not in train_df.columns:
            raise ValueError(f"Target column not found: {target}")
        target_series = train_df[target]
    else:
        target_series = pd.Series(target)
        if len(target_series) != len(train_df):
            raise ValueError("Target length must match train_df length.")
        if not target_series.index.equals(train_df.index):
            target_series.index = train_df.index

    target_numeric = pd.to_numeric(target_series, errors="coerce")
    if target_numeric.notna().sum() == 0:
        raise ValueError("Target must contain at least one numeric value.")
    return target_numeric.astype(float)


def build_price_quantile_bin(price: pd.Series, bucket_count: int = 5) -> pd.Series:
    """Build deterministic price quantile labels such as Q1..Q5."""
    if bucket_count < 2:
        raise ValueError("bucket_count must be at least 2.")
    price_numeric = pd.to_numeric(price, errors="coerce")
    ranked_price = price_numeric.rank(method="first")
    bucket_codes = pd.qcut(ranked_price, q=bucket_count, labels=False, duplicates="drop")
    bucket_series = pd.Series(bucket_codes, index=price.index, dtype="float")
    return bucket_series.map(
        lambda value: f"Q{int(value) + 1}" if pd.notna(value) else "__PRICE_MISSING__"
    ).astype("string")


def _validate_columns(df: pd.DataFrame, columns: Sequence[str]) -> tuple[str, ...]:
    resolved_columns = tuple(columns)
    if not resolved_columns:
        raise ValueError("At least one encoding column is required.")
    missing = [col for col in resolved_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Encoding columns not found: {missing}")
    return resolved_columns


def _normalize_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().fillna(MISSING_CATEGORY)


def _feature_name(column: str, feature_suffix: str) -> str:
    return f"{column}{feature_suffix}"


def add_smoothed_target_encoding(
    fit_keys: pd.Series,
    fit_target: pd.Series,
    apply_keys: pd.Series,
    smoothing: float,
) -> pd.Series:
    """Apply smoothed mean encoding from fit rows to apply rows."""
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")
    global_mean = float(pd.to_numeric(fit_target, errors="coerce").mean())
    grouped = (
        pd.DataFrame({"key": _normalize_key(fit_keys), "target": fit_target})
        .dropna(subset=["target"])
        .groupby("key", dropna=False)["target"]
        .agg(["mean", "count"])
    )
    grouped["encoded"] = (
        grouped["mean"] * grouped["count"] + global_mean * smoothing
    ) / (grouped["count"] + smoothing)
    return _normalize_key(apply_keys).map(grouped["encoded"]).fillna(global_mean).astype(float)


def add_smoothed_target_encoding_with_backoff(
    fit_keys: pd.Series,
    fit_target: pd.Series,
    apply_keys: pd.Series,
    smoothing: float,
    min_count: int,
    fallback: pd.Series,
) -> pd.Series:
    """Apply smoothed encoding, using a fallback where the fit key is too rare."""
    if min_count < 1:
        raise ValueError("min_count must be positive.")
    global_mean = float(pd.to_numeric(fit_target, errors="coerce").mean())
    grouped = (
        pd.DataFrame({"key": _normalize_key(fit_keys), "target": fit_target})
        .dropna(subset=["target"])
        .groupby("key", dropna=False)["target"]
        .agg(["mean", "count"])
    )
    grouped["encoded"] = (
        grouped["mean"] * grouped["count"] + global_mean * smoothing
    ) / (grouped["count"] + smoothing)

    normalized_apply_keys = _normalize_key(apply_keys)
    counts = normalized_apply_keys.map(grouped["count"]).fillna(0)
    encoded = normalized_apply_keys.map(grouped["encoded"])
    fallback = fallback.reindex(apply_keys.index).fillna(global_mean)
    return encoded.where(counts >= min_count, fallback).fillna(fallback).astype(float)


def _smoothed_mapping(
    keys: pd.Series,
    target: pd.Series,
    global_mean: float,
    smoothing: float,
) -> pd.Series:
    fit_frame = pd.DataFrame({"key": _normalize_key(keys), "target": target})
    fit_frame = fit_frame.loc[fit_frame["target"].notna()]
    if fit_frame.empty:
        raise ValueError("Cannot fit target encoding mapping with empty numeric target.")

    stats = fit_frame.groupby("key", dropna=False)["target"].agg(["mean", "count"])
    mapping = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return mapping.astype(float)


def _apply_mapping(keys: pd.Series, mapping: pd.Series, fallback: float) -> pd.Series:
    encoded = _normalize_key(keys).map(mapping)
    return encoded.astype(float).fillna(fallback)


def build_oof_price_proxy_features(
    train_df: pd.DataFrame,
    target: str | pd.Series | Sequence[float],
    columns: Sequence[str] = DEFAULT_PRICE_PROXY_COLUMNS,
    n_splits: int = 5,
    smoothing: float = 20.0,
    random_state: int = 42,
    shuffle: bool = True,
    feature_suffix: str = "_price_mean",
    target_space: str | bool = "raw",
) -> pd.DataFrame:
    """Build strict out-of-fold target-encoding features for training rows.

    Each validation fold is encoded only from the corresponding fit folds. This
    function is safe for training-feature generation; it never maps a row using
    a statistic computed from its own target. `target_space` controls whether
    the encoded mean is fitted on raw price, log1p(price), or sqrt(price).
    """
    resolved_columns = _validate_columns(train_df, columns)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for OOF target encoding.")
    if n_splits > len(train_df):
        raise ValueError("n_splits cannot exceed the number of training rows.")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")

    target_series = transform_target_for_encoding(_as_target_series(train_df, target), target_space)
    encoded = pd.DataFrame(index=train_df.index)
    for col in resolved_columns:
        encoded[_feature_name(col, feature_suffix)] = np.nan

    splitter = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    for fit_idx, valid_idx in splitter.split(train_df):
        fit_index = train_df.index[fit_idx]
        valid_index = train_df.index[valid_idx]
        fit_target = target_series.loc[fit_index]
        fold_global_mean = float(fit_target.mean())
        if np.isnan(fold_global_mean):
            raise ValueError("A fold has no numeric target values.")

        for col in resolved_columns:
            mapping = _smoothed_mapping(
                keys=train_df.loc[fit_index, col],
                target=fit_target,
                global_mean=fold_global_mean,
                smoothing=smoothing,
            )
            encoded.loc[valid_index, _feature_name(col, feature_suffix)] = _apply_mapping(
                keys=train_df.loc[valid_index, col],
                mapping=mapping,
                fallback=fold_global_mean,
            ).to_numpy()

    full_global_mean = float(target_series.mean())
    return encoded.astype(float).fillna(full_global_mean)


def fit_price_proxy_encoder(
    train_df: pd.DataFrame,
    target: str | pd.Series | Sequence[float],
    columns: Sequence[str] = DEFAULT_PRICE_PROXY_COLUMNS,
    smoothing: float = 20.0,
    feature_suffix: str = "_price_mean",
    target_space: str | bool = "raw",
) -> PriceProxyEncoder:
    """Fit full-training mappings for applying price proxy features to holdout/test rows."""
    resolved_columns = _validate_columns(train_df, columns)
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")

    target_series = transform_target_for_encoding(_as_target_series(train_df, target), target_space)
    global_mean = float(target_series.mean())
    mappings = {
        col: _smoothed_mapping(
            keys=train_df[col],
            target=target_series,
            global_mean=global_mean,
            smoothing=smoothing,
        )
        for col in resolved_columns
    }
    return PriceProxyEncoder(
        columns=resolved_columns,
        feature_suffix=feature_suffix,
        smoothing=smoothing,
        target_space=normalize_target_space(target_space),
        global_mean=global_mean,
        mappings=mappings,
    )


def transform_price_proxy_features(df: pd.DataFrame, encoder: PriceProxyEncoder) -> pd.DataFrame:
    """Apply fitted full-training target-encoding mappings to holdout/test rows."""
    _validate_columns(df, encoder.columns)
    encoded = pd.DataFrame(index=df.index)
    for col in encoder.columns:
        encoded[_feature_name(col, encoder.feature_suffix)] = _apply_mapping(
            keys=df[col],
            mapping=encoder.mappings[col],
            fallback=encoder.global_mean,
        )
    return encoded.astype(float)


def build_oof_and_apply_price_proxy_features(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    target: str | pd.Series | Sequence[float],
    columns: Sequence[str] = DEFAULT_PRICE_PROXY_COLUMNS,
    n_splits: int = 5,
    smoothing: float = 20.0,
    random_state: int = 42,
    shuffle: bool = True,
    feature_suffix: str = "_price_mean",
    target_space: str | bool = "raw",
) -> tuple[pd.DataFrame, pd.DataFrame, PriceProxyEncoder]:
    """Build strict OOF train encodings and full-train apply encodings together."""
    train_encoded = build_oof_price_proxy_features(
        train_df=train_df,
        target=target,
        columns=columns,
        n_splits=n_splits,
        smoothing=smoothing,
        random_state=random_state,
        shuffle=shuffle,
        feature_suffix=feature_suffix,
        target_space=target_space,
    )
    encoder = fit_price_proxy_encoder(
        train_df=train_df,
        target=target,
        columns=columns,
        smoothing=smoothing,
        feature_suffix=feature_suffix,
        target_space=target_space,
    )
    apply_encoded = transform_price_proxy_features(apply_df, encoder)
    return train_encoded, apply_encoded, encoder


def add_model_backoff_target_encoding(
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    apply_features: pd.DataFrame,
    smoothing: float,
    min_count: int,
) -> pd.Series:
    """Encode model with brand-level fallback for low-frequency models."""
    required_columns = {"model", "brand"}
    if not required_columns.issubset(fit_features.columns) or not required_columns.issubset(apply_features.columns):
        raise ValueError("Model backoff target encoding requires 'model' and 'brand' columns.")

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
    return model_encoded.where(apply_model_counts >= min_count, brand_encoded).astype(float)


def build_target_encoding_features(
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    apply_features: pd.DataFrame,
    use_brand_target_encoding: bool = False,
    use_brand_age_target_encoding: bool = False,
    use_model_target_encoding: bool = False,
    use_model_age_target_encoding: bool = False,
    use_model_backoff_target_encoding: bool = False,
    use_model_low_freq_flag: bool = False,
    model_backoff_min_count: int = 20,
    model_low_freq_min_count: int = 20,
    target_encoding_smoothing: float = 20.0,
    target_space: str | bool = "raw",
) -> pd.DataFrame:
    """Build fold-safe target encoding features from fit rows to apply rows.

    This function centralizes the encoding formulas used by current experiments.
    Call it inside a CV fold for validation rows, or on full train -> test for
    submission features.
    """
    fit_target_encoded = transform_target_for_encoding(fit_target, target_space)
    encoded_features: dict[str, pd.Series] = {}

    if use_brand_target_encoding:
        encoded_features["brand_target_mean"] = add_smoothed_target_encoding(
            fit_features["brand"],
            fit_target_encoded,
            apply_features["brand"],
            target_encoding_smoothing,
        )

    if use_brand_age_target_encoding:
        fit_key = fit_features["brand"].astype("string") + "|" + build_age_bin(fit_features["car_age_years"])
        apply_key = apply_features["brand"].astype("string") + "|" + build_age_bin(apply_features["car_age_years"])
        encoded_features["brand_age_target_mean"] = add_smoothed_target_encoding(
            fit_key,
            fit_target_encoded,
            apply_key,
            target_encoding_smoothing,
        )

    if use_model_target_encoding:
        encoded_features["model_target_mean"] = add_smoothed_target_encoding(
            fit_features["model"],
            fit_target_encoded,
            apply_features["model"],
            target_encoding_smoothing,
        )

    if use_model_backoff_target_encoding:
        encoded_features["model_backoff_target_mean"] = add_model_backoff_target_encoding(
            fit_features=fit_features,
            fit_target=fit_target_encoded,
            apply_features=apply_features,
            smoothing=target_encoding_smoothing,
            min_count=model_backoff_min_count,
        )

        fit_power_age_key = (
            fit_features["model"].astype("string")
            + "|"
            + build_power_bin(fit_features["power"])
            + "|"
            + build_age_bin(fit_features["car_age_years"])
        )
        apply_power_age_key = (
            apply_features["model"].astype("string")
            + "|"
            + build_power_bin(apply_features["power"])
            + "|"
            + build_age_bin(apply_features["car_age_years"])
        )
        fallback = encoded_features["model_backoff_target_mean"]
        encoded_features["model_power_age_backoff_target_mean"] = add_smoothed_target_encoding_with_backoff(
            fit_keys=fit_power_age_key,
            fit_target=fit_target_encoded,
            apply_keys=apply_power_age_key,
            smoothing=target_encoding_smoothing,
            min_count=model_backoff_min_count,
            fallback=fallback,
        )

    if use_model_low_freq_flag:
        model_counts = fit_features["model"].value_counts(dropna=False)
        apply_model_counts = apply_features["model"].map(model_counts).fillna(0)
        encoded_features["model_low_freq_flag"] = (apply_model_counts < model_low_freq_min_count).astype(float)

    if use_model_age_target_encoding:
        fit_age_bin = build_age_bin(fit_features["car_age_years"])
        apply_age_bin = build_age_bin(apply_features["car_age_years"])
        fit_key = fit_features["model"].astype("string") + "|" + fit_age_bin
        apply_key = apply_features["model"].astype("string") + "|" + apply_age_bin
        model_fallback = add_model_backoff_target_encoding(
            fit_features=fit_features,
            fit_target=fit_target_encoded,
            apply_features=apply_features,
            smoothing=target_encoding_smoothing,
            min_count=model_backoff_min_count,
        )
        encoded_features["model_age_target_mean"] = add_smoothed_target_encoding_with_backoff(
            fit_keys=fit_key,
            fit_target=fit_target_encoded,
            apply_keys=apply_key,
            smoothing=target_encoding_smoothing,
            min_count=model_backoff_min_count,
            fallback=model_fallback,
        )

        fit_power_bin = build_power_bin(fit_features["power"])
        apply_power_bin = build_power_bin(apply_features["power"])
        power_bin_fallback = add_smoothed_target_encoding(
            fit_keys=fit_power_bin,
            fit_target=fit_target_encoded,
            apply_keys=apply_power_bin,
            smoothing=target_encoding_smoothing,
        )
        encoded_features["power_age_bin_target_mean"] = add_smoothed_target_encoding_with_backoff(
            fit_keys=fit_power_bin + "|" + fit_age_bin,
            fit_target=fit_target_encoded,
            apply_keys=apply_power_bin + "|" + apply_age_bin,
            smoothing=target_encoding_smoothing,
            min_count=model_backoff_min_count,
            fallback=power_bin_fallback,
        )

    return pd.DataFrame(encoded_features, index=apply_features.index)


def add_price_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return input unchanged unless explicit OOF target encoding is requested.

    Use `build_oof_price_proxy_features` for train rows and
    `transform_price_proxy_features` for holdout/test rows. Keeping this default
    as a no-op avoids accidentally adding leaking full-train price statistics.
    """
    return df.copy()


__all__ = [
    "PriceProxyEncoder",
    "TARGET_ENCODING_WARNING",
    "SUPPORTED_TARGET_SPACES",
    "normalize_target_space",
    "transform_target_for_encoding",
    "build_price_quantile_bin",
    "add_smoothed_target_encoding",
    "add_smoothed_target_encoding_with_backoff",
    "build_oof_price_proxy_features",
    "fit_price_proxy_encoder",
    "transform_price_proxy_features",
    "build_oof_and_apply_price_proxy_features",
    "add_model_backoff_target_encoding",
    "build_target_encoding_features",
    "add_price_proxy_features",
]
