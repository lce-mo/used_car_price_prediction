from __future__ import annotations

import numpy as np
import pandas as pd


def build_power_bin(series: pd.Series) -> pd.Series:
    """Bucket clipped engine power into stable labels for TE keys and slices."""
    power = pd.to_numeric(series, errors="coerce").clip(lower=0, upper=600)
    bucket = pd.cut(
        power,
        bins=[-1, 0, 60, 90, 120, 150, 200, 300, 600],
        labels=["0", "1_60", "60_90", "90_120", "120_150", "150_200", "200_300", "300_600"],
    )
    return bucket.astype("string").fillna("__POWER_MISSING__")


def normalize_power_features(df: pd.DataFrame, use_power_bin: bool = False) -> pd.DataFrame:
    """Normalize engine power and add legacy power quality indicators."""
    featured = df.copy()
    if "power" not in featured.columns:
        return featured

    power = pd.to_numeric(featured["power"], errors="coerce")
    featured["power_outlier_flag"] = ((power < 0) | (power > 600)).astype("int8")
    featured["power"] = power.clip(lower=0, upper=600)
    featured["power_is_zero"] = (featured["power"] == 0).astype("int8")

    if use_power_bin:
        featured["power_bin"] = pd.cut(
            featured["power"],
            bins=[-1, 0, 60, 90, 120, 150, 200, 300, 600],
            labels=False,
        ).astype("float")

    return featured


def add_depreciation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add conservative age, mileage, and depreciation proxy features.

    These features describe usage intensity without using the target:
    - mileage per vehicle age
    - age-mileage interaction
    - coarse new/old vehicle flags
    """
    featured = df.copy()
    age_years = pd.to_numeric(featured.get("car_age_years", pd.Series(np.nan, index=featured.index)), errors="coerce")
    kilometer = pd.to_numeric(featured.get("kilometer", pd.Series(np.nan, index=featured.index)), errors="coerce")

    safe_age = age_years.clip(lower=0)
    denominator = safe_age.where(safe_age > 0)

    featured["kilometer_per_year"] = kilometer / denominator
    featured["age_kilometer_interaction"] = safe_age * kilometer
    featured["is_nearly_new_1y"] = (safe_age <= 1).where(safe_age.notna()).astype("float")
    featured["is_old_car_10y"] = (safe_age >= 10).where(safe_age.notna()).astype("float")

    new_columns = [
        "kilometer_per_year",
        "age_kilometer_interaction",
        "is_nearly_new_1y",
        "is_old_car_10y",
    ]
    featured[new_columns] = featured[new_columns].replace([np.inf, -np.inf], np.nan)
    return featured


def add_power_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add legacy power-age and mileage-age interaction features.

    The current best experiments showed this feature family is useful in the
    blend pool. It captures whether a car is unusually powerful or heavily used
    for its age while remaining independent of `price`.
    """
    featured = df.copy()

    if "power" in featured.columns and "car_age_years" in featured.columns:
        power = pd.to_numeric(featured["power"], errors="coerce")
        age_years = pd.to_numeric(featured["car_age_years"], errors="coerce").clip(lower=0)
        age_base = age_years.fillna(0.0) + 1.0

        featured["power_per_age_year"] = power / age_base
        featured["power_per_age_year_sqrt"] = power / np.sqrt(age_base)
        featured["power_age_product"] = power * age_years
        featured["power_age_ratio"] = power / age_base
        featured["log_power_per_age"] = np.log1p(power.clip(lower=0)) / age_base

    if "kilometer" in featured.columns and "car_age_years" in featured.columns:
        kilometer = pd.to_numeric(featured["kilometer"], errors="coerce")
        age_years = pd.to_numeric(featured["car_age_years"], errors="coerce").clip(lower=0)
        age_base = age_years.fillna(0.0) + 1.0

        featured["kilometer_per_age_year"] = kilometer / age_base
        featured["log_kilometer_per_age"] = np.log1p(kilometer.clip(lower=0)) / age_base
        featured["age_kilometer_product"] = age_years * kilometer

    new_columns = [
        "power_per_age_year",
        "power_per_age_year_sqrt",
        "power_age_product",
        "power_age_ratio",
        "log_power_per_age",
        "kilometer_per_age_year",
        "log_kilometer_per_age",
        "age_kilometer_product",
    ]
    present_columns = [col for col in new_columns if col in featured.columns]
    if present_columns:
        featured[present_columns] = featured[present_columns].replace([np.inf, -np.inf], np.nan)

    return featured


def add_brand_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compare power, mileage, and age against brand-level non-target peers."""
    featured = df.copy()

    if "brand_power_mean" in featured.columns and "power" in featured.columns:
        featured["power_minus_brand_power_mean"] = featured["power"] - featured["brand_power_mean"]
        featured["power_ratio_brand_power_mean"] = featured["power"] / (featured["brand_power_mean"] + 1e-6)

    if "brand_kilometer_mean" in featured.columns and "kilometer" in featured.columns:
        featured["kilometer_minus_brand_kilometer_mean"] = featured["kilometer"] - featured["brand_kilometer_mean"]
        featured["kilometer_ratio_brand_kilometer_mean"] = featured["kilometer"] / (
            featured["brand_kilometer_mean"] + 1e-6
        )

    if "brand" in featured.columns and "car_age_years" in featured.columns:
        brand_age_mean = featured.groupby("brand", dropna=False)["car_age_years"].transform("mean")
        featured["brand_car_age_years_mean"] = brand_age_mean
        featured["car_age_minus_brand_age_mean"] = featured["car_age_years"] - brand_age_mean

    new_columns = [
        "power_ratio_brand_power_mean",
        "kilometer_ratio_brand_kilometer_mean",
    ]
    present_columns = [col for col in new_columns if col in featured.columns]
    if present_columns:
        featured[present_columns] = featured[present_columns].replace([np.inf, -np.inf], np.nan)

    return featured


__all__ = [
    "build_power_bin",
    "normalize_power_features",
    "add_depreciation_features",
    "add_power_age_features",
    "add_brand_relative_features",
]
