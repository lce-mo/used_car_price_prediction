from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .categorical_features import (
    CATEGORICAL_COLUMNS,
    add_categorical_features,
    add_count_encoding,
)
from .date_features import add_age_detail_features, add_date_features, build_age_bin
from .depreciation_features import (
    add_brand_relative_features,
    add_depreciation_features,
    add_power_age_features,
    normalize_power_features,
)
from .price_proxy_features import (
    DEFAULT_PRICE_PROXY_COLUMNS,
    add_price_proxy_features,
    build_oof_and_apply_price_proxy_features,
)


NUMERIC_BASE_COLUMNS = [
    "power",
    "kilometer",
    "model",
    "brand",
    "bodyType",
    "fuelType",
    "gearbox",
    "regionCode",
]
DROP_FROM_MODEL_FEATURES = ["SaleID", "offerType", "seller", "regDate", "creatDate", "name", "__is_train__"]


@dataclass
class PreparedData:
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    target: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]


def _to_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert legacy numeric-looking raw columns before feature formulas run."""
    featured = df.copy()
    for col in columns:
        if col in featured.columns:
            featured[col] = pd.to_numeric(featured[col], errors="coerce")
    return featured


def normalize_numeric_columns(df: pd.DataFrame, use_power_bin: bool = False) -> pd.DataFrame:
    """Normalize raw numeric fields and add power outlier indicators."""
    numeric_columns = list(NUMERIC_BASE_COLUMNS)
    numeric_columns.extend([col for col in df.columns if col.startswith("v_")])
    featured = _to_numeric(df, numeric_columns)
    featured = normalize_power_features(featured, use_power_bin=use_power_bin)
    return featured


def add_group_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add non-target peer statistics from brand/model groups.

    These statistics use only observable fields such as `power` and `kilometer`;
    they do not use `price`.
    """
    featured = df.copy()
    stats_plan = {
        "brand": {
            "power": ["mean", "median"],
            "kilometer": ["mean"],
        },
        "model": {
            "power": ["mean"],
        },
    }

    for group_col, value_plan in stats_plan.items():
        if group_col not in featured.columns:
            continue
        for value_col, agg_list in value_plan.items():
            if value_col not in featured.columns:
                continue
            grouped = featured.groupby(group_col, dropna=False)[value_col]
            for agg_name in agg_list:
                featured[f"{group_col}_{value_col}_{agg_name}"] = grouped.transform(agg_name)

    return featured


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight interaction features used by legacy experiments."""
    featured = df.copy()

    if "power" in featured.columns and "brand_power_mean" in featured.columns:
        featured["power_minus_brand_power_mean"] = featured["power"] - featured["brand_power_mean"]
        featured["power_ratio_brand_power_mean"] = featured["power"] / (featured["brand_power_mean"] + 1e-6)

    present_columns = [col for col in ["power_ratio_brand_power_mean"] if col in featured.columns]
    if present_columns:
        featured[present_columns] = featured[present_columns].replace([np.inf, -np.inf], np.nan)

    return featured


def add_model_age_group_stats(df: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
    """Add model-by-age-bin non-target support and peer statistics.

    The direct model-age group value is used only when the group has enough
    support; otherwise it backs off to model, brand, and global statistics.
    """
    featured = df.copy()
    required_columns = {"model", "brand", "car_age_years", "power", "kilometer", "model_count"}
    if not required_columns.issubset(featured.columns):
        return featured

    age_bin = build_age_bin(featured["car_age_years"])
    model_key = featured["model"].astype("string")
    group_key = model_key + "|" + age_bin

    model_age_count = group_key.map(group_key.value_counts(dropna=False)).astype(float)
    model_count = pd.to_numeric(featured["model_count"], errors="coerce")
    safe_model_count = model_count.replace(0, np.nan)

    featured["model_age_count"] = model_age_count
    featured["model_age_count_ratio"] = (model_age_count / safe_model_count).fillna(0.0)

    power = pd.to_numeric(featured["power"], errors="coerce")
    kilometer = pd.to_numeric(featured["kilometer"], errors="coerce")
    grouped_power = power.groupby(group_key, dropna=False)
    grouped_kilometer = kilometer.groupby(group_key, dropna=False)
    model_power = power.groupby(featured["model"], dropna=False)
    model_kilometer = kilometer.groupby(featured["model"], dropna=False)
    brand_power = power.groupby(featured["brand"], dropna=False)
    brand_kilometer = kilometer.groupby(featured["brand"], dropna=False)

    use_group_stats = model_age_count >= float(min_count)
    featured["model_age_power_median"] = (
        grouped_power.transform("median").where(use_group_stats, model_power.transform("median")).fillna(power.median())
    )
    featured["model_age_kilometer_median"] = (
        grouped_kilometer.transform("median")
        .where(use_group_stats, model_kilometer.transform("median"))
        .fillna(kilometer.median())
    )

    featured["model_age_power_mean"] = (
        grouped_power.transform("mean")
        .where(use_group_stats, model_power.transform("mean"))
        .fillna(brand_power.transform("mean"))
        .fillna(power.mean())
    )
    featured["model_age_power_std"] = (
        grouped_power.transform("std")
        .where(use_group_stats, model_power.transform("std"))
        .fillna(brand_power.transform("std"))
        .fillna(power.std())
        .fillna(0.0)
    )
    featured["model_age_kilometer_mean"] = (
        grouped_kilometer.transform("mean")
        .where(use_group_stats, model_kilometer.transform("mean"))
        .fillna(brand_kilometer.transform("mean"))
        .fillna(kilometer.mean())
    )
    featured["model_age_kilometer_std"] = (
        grouped_kilometer.transform("std")
        .where(use_group_stats, model_kilometer.transform("std"))
        .fillna(brand_kilometer.transform("std"))
        .fillna(kilometer.std())
        .fillna(0.0)
    )

    featured["power_minus_model_age_power_mean"] = power - featured["model_age_power_mean"]
    featured["power_ratio_model_age_power_mean"] = power / (featured["model_age_power_mean"] + 1e-6)
    featured["kilometer_minus_model_age_kilometer_mean"] = kilometer - featured["model_age_kilometer_mean"]
    featured["kilometer_ratio_model_age_kilometer_mean"] = kilometer / (
        featured["model_age_kilometer_mean"] + 1e-6
    )

    new_columns = [
        "model_age_power_mean",
        "model_age_power_std",
        "model_age_kilometer_mean",
        "model_age_kilometer_std",
        "power_minus_model_age_power_mean",
        "power_ratio_model_age_power_mean",
        "kilometer_minus_model_age_kilometer_mean",
        "kilometer_ratio_model_age_kilometer_mean",
    ]
    featured[new_columns] = featured[new_columns].replace([np.inf, -np.inf], np.nan)
    return featured


def finalize_model_feature_frame(df: pd.DataFrame, drop_unused_columns: bool = False) -> pd.DataFrame:
    """Normalize categorical columns and optionally drop columns unused by models."""
    featured = add_categorical_features(df, add_counts=False)
    if not drop_unused_columns:
        return featured
    return featured.drop(columns=[col for col in DROP_FROM_MODEL_FEATURES if col in featured.columns])


def _build_features(
    df: pd.DataFrame,
    add_count_encoding: bool,
    use_group_stats: bool = True,
    use_power_bin: bool = False,
    use_interactions: bool = False,
    use_brand_relative: bool = False,
    use_power_age: bool = False,
    use_age_detail: bool = False,
    use_model_age_group_stats: bool = False,
    model_age_group_min_count: int = 20,
    use_depreciation_features: bool = False,
    drop_unused_columns: bool = False,
) -> pd.DataFrame:
    """Build the full non-target feature frame in legacy-compatible order."""
    featured = df.copy()
    featured = normalize_numeric_columns(featured, use_power_bin=use_power_bin)
    featured = add_date_features(featured)

    if add_count_encoding:
        featured = add_count_encoding_features(featured)
    if use_group_stats:
        featured = add_group_statistics(featured)
    if use_brand_relative:
        featured = add_brand_relative_features(featured)
    if use_depreciation_features:
        featured = add_depreciation_features(featured)
    if use_power_age:
        featured = add_power_age_features(featured)
    if use_age_detail:
        featured = add_age_detail_features(featured)
    if use_model_age_group_stats:
        featured = add_model_age_group_stats(featured, min_count=model_age_group_min_count)
    if use_interactions:
        featured = add_interaction_features(featured)

    featured = add_price_proxy_features(featured)
    return finalize_model_feature_frame(featured, drop_unused_columns=drop_unused_columns)


def add_count_encoding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add legacy frequency features for name, brand, regionCode, and model."""
    return add_count_encoding(df)


def build_train_features(df: pd.DataFrame, add_count_encoding: bool = True) -> pd.DataFrame:
    """Build training features without target-leaking price statistics."""
    return _build_features(df, add_count_encoding=add_count_encoding)


def build_test_features(df: pd.DataFrame, add_count_encoding: bool = True) -> pd.DataFrame:
    """Build test features with the same conservative feature pipeline."""
    return _build_features(df, add_count_encoding=add_count_encoding)


def build_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    add_count_encoding: bool = True,
    add_price_proxy_encoding: bool = False,
    target_col: str = "price",
    price_proxy_columns: Sequence[str] = DEFAULT_PRICE_PROXY_COLUMNS,
    price_proxy_smoothing: float = 20.0,
    price_proxy_n_splits: int = 5,
    random_state: int = 42,
    price_proxy_target_space: str | bool = "raw",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test features together when OOF price proxies are requested."""
    train_features = build_train_features(train_df, add_count_encoding=add_count_encoding)
    test_features = build_test_features(test_df, add_count_encoding=add_count_encoding)

    if not add_price_proxy_encoding:
        return train_features, test_features

    if target_col not in train_df.columns:
        raise ValueError(f"Target column not found for price proxy encoding: {target_col}")

    train_encoded, test_encoded, _ = build_oof_and_apply_price_proxy_features(
        train_df=train_features,
        apply_df=test_features,
        target=train_df[target_col],
        columns=price_proxy_columns,
        n_splits=price_proxy_n_splits,
        smoothing=price_proxy_smoothing,
        random_state=random_state,
        target_space=price_proxy_target_space,
    )
    train_features = pd.concat([train_features, train_encoded], axis=1)
    test_features = pd.concat([test_features, test_encoded], axis=1)
    return train_features, test_features


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_group_stats: bool = True,
    use_power_bin: bool = False,
    use_interactions: bool = False,
    use_brand_relative: bool = False,
    use_power_age: bool = False,
    use_age_detail: bool = False,
    use_model_age_group_stats: bool = False,
    model_age_group_min_count: int = 20,
) -> PreparedData:
    """Build legacy-compatible train/test feature matrices for model training.

    This is the modular replacement for the old flat `src/features.py`
    `prepare_features` function. It intentionally keeps the same signature and
    train/test concatenation scope so current training, CV, and prediction code
    can continue to call `from features import prepare_features`.
    """
    if "price" not in train_df.columns:
        raise ValueError("Training data must contain a 'price' column.")

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_ids = train_df["SaleID"].copy()
    test_ids = test_df["SaleID"].copy()
    target = pd.to_numeric(train_df["price"], errors="coerce")
    train_df = train_df.drop(columns=["price"])

    full = pd.concat(
        [train_df.assign(__is_train__=1), test_df.assign(__is_train__=0)],
        ignore_index=True,
        sort=False,
    )
    full = _build_features(
        full,
        add_count_encoding=True,
        use_group_stats=use_group_stats,
        use_power_bin=use_power_bin,
        use_interactions=use_interactions,
        use_brand_relative=use_brand_relative,
        use_power_age=use_power_age,
        use_age_detail=use_age_detail,
        use_model_age_group_stats=use_model_age_group_stats,
        model_age_group_min_count=model_age_group_min_count,
        use_depreciation_features=False,
        drop_unused_columns=False,
    )

    feature_df = full.drop(columns=[col for col in DROP_FROM_MODEL_FEATURES if col in full.columns])
    categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in feature_df.columns]
    numeric_columns = [col for col in feature_df.columns if col not in categorical_columns]

    for col in categorical_columns:
        feature_df[col] = feature_df[col].astype("string").fillna("__MISSING__")

    train_features = feature_df.loc[full["__is_train__"] == 1].reset_index(drop=True)
    test_features = feature_df.loc[full["__is_train__"] == 0].reset_index(drop=True)

    return PreparedData(
        train_features=train_features,
        test_features=test_features,
        target=target.reset_index(drop=True),
        train_ids=train_ids.reset_index(drop=True),
        test_ids=test_ids.reset_index(drop=True),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


__all__ = [
    "NUMERIC_BASE_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "DROP_FROM_MODEL_FEATURES",
    "PreparedData",
    "normalize_numeric_columns",
    "add_count_encoding_features",
    "add_group_statistics",
    "add_interaction_features",
    "add_model_age_group_stats",
    "finalize_model_feature_frame",
    "prepare_features",
    "build_train_features",
    "build_test_features",
    "build_train_test_features",
]
