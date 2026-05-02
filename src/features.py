from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PreparedData:
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    target: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]


def _parse_compact_date(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    reg_date = _parse_compact_date(df["regDate"]) if "regDate" in df.columns else pd.Series(pd.NaT, index=df.index)
    create_date = _parse_compact_date(df["creatDate"]) if "creatDate" in df.columns else pd.Series(pd.NaT, index=df.index)

    df["reg_year"] = reg_date.dt.year
    df["reg_month"] = reg_date.dt.month
    df["reg_day"] = reg_date.dt.day
    df["reg_weekday"] = reg_date.dt.weekday

    df["create_year"] = create_date.dt.year
    df["create_month"] = create_date.dt.month
    df["create_day"] = create_date.dt.day
    df["create_weekday"] = create_date.dt.weekday

    car_age_days = (create_date - reg_date).dt.days
    df["car_age_days"] = car_age_days
    df["car_age_years"] = car_age_days / 365.25

    return df


def _add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["name", "brand", "regionCode", "model"]:
        if col in df.columns:
            df[f"{col}_count"] = df[col].map(df[col].value_counts(dropna=False))
    return df


def _add_group_statistics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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
        if group_col not in df.columns:
            continue
        for value_col, agg_list in value_plan.items():
            if value_col not in df.columns:
                continue
            grouped = df.groupby(group_col, dropna=False)[value_col]
            for agg_name in agg_list:
                df[f"{group_col}_{value_col}_{agg_name}"] = grouped.transform(agg_name)

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "power" in df.columns and "brand_power_mean" in df.columns:
        df["power_minus_brand_power_mean"] = df["power"] - df["brand_power_mean"]
        df["power_ratio_brand_power_mean"] = df["power"] / (df["brand_power_mean"] + 1e-6)

    return df


def _add_brand_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "brand_power_mean" in df.columns and "power" in df.columns:
        df["power_minus_brand_power_mean"] = df["power"] - df["brand_power_mean"]
        df["power_ratio_brand_power_mean"] = df["power"] / (df["brand_power_mean"] + 1e-6)

    if "brand_kilometer_mean" in df.columns and "kilometer" in df.columns:
        df["kilometer_minus_brand_kilometer_mean"] = df["kilometer"] - df["brand_kilometer_mean"]
        df["kilometer_ratio_brand_kilometer_mean"] = df["kilometer"] / (df["brand_kilometer_mean"] + 1e-6)

    if "brand" in df.columns and "car_age_years" in df.columns:
        brand_age_mean = df.groupby("brand", dropna=False)["car_age_years"].transform("mean")
        df["brand_car_age_years_mean"] = brand_age_mean
        df["car_age_minus_brand_age_mean"] = df["car_age_years"] - brand_age_mean

    return df


def _add_power_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "power" in df.columns and "car_age_years" in df.columns:
        age_base = df["car_age_years"].clip(lower=0).fillna(0) + 1.0
        df["power_per_age_year"] = df["power"] / age_base
        df["power_per_age_year_sqrt"] = df["power"] / np.sqrt(age_base)

    return df


def _add_age_detail_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "car_age_days" in df.columns:
        age_days = df["car_age_days"].clip(lower=0)
        df["car_age_months"] = age_days / 30.4

    if "car_age_years" in df.columns:
        age_years = df["car_age_years"].clip(lower=0)
        df["is_nearly_new_1y"] = (age_years <= 1).astype("int8")
        df["is_nearly_new_3y"] = (age_years <= 3).astype("int8")

    return df


def _build_age_bin(series: pd.Series) -> pd.Series:
    age_years = pd.to_numeric(series, errors="coerce").clip(lower=0)
    bins = pd.cut(
        age_years,
        bins=[-np.inf, 1, 3, 5, 8, np.inf],
        labels=["0_1y", "1_3y", "3_5y", "5_8y", "8y_plus"],
    )
    return bins.astype("string").fillna("__AGE_MISSING__")


def _add_model_age_group_stats(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    df = df.copy()
    required_columns = {"model", "car_age_years", "power", "kilometer", "model_count"}
    if not required_columns.issubset(df.columns):
        return df

    age_bin = _build_age_bin(df["car_age_years"])
    model_key = df["model"].astype("string")
    group_key = model_key + "|" + age_bin

    model_age_count = group_key.map(group_key.value_counts(dropna=False)).astype(float)
    model_count = pd.to_numeric(df["model_count"], errors="coerce")
    safe_model_count = model_count.replace(0, np.nan)

    df["model_age_count"] = model_age_count
    df["model_age_count_ratio"] = (model_age_count / safe_model_count).fillna(0.0)

    power_group_median = df.groupby(group_key, dropna=False)["power"].transform("median")
    kilometer_group_median = df.groupby(group_key, dropna=False)["kilometer"].transform("median")
    model_power_median = df.groupby("model", dropna=False)["power"].transform("median")
    model_kilometer_median = df.groupby("model", dropna=False)["kilometer"].transform("median")
    global_power_median = pd.to_numeric(df["power"], errors="coerce").median()
    global_kilometer_median = pd.to_numeric(df["kilometer"], errors="coerce").median()

    use_group_stats = model_age_count >= float(min_count)
    df["model_age_power_median"] = (
        power_group_median.where(use_group_stats, model_power_median).fillna(global_power_median)
    )
    df["model_age_kilometer_median"] = (
        kilometer_group_median.where(use_group_stats, model_kilometer_median).fillna(global_kilometer_median)
    )

    return df


def _normalize_columns(df: pd.DataFrame, use_power_bin: bool = False) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        "power",
        "kilometer",
        "model",
        "brand",
        "bodyType",
        "fuelType",
        "gearbox",
        "regionCode",
    ]
    numeric_columns.extend([col for col in df.columns if col.startswith("v_")])
    df = _to_numeric(df, numeric_columns)

    if "notRepairedDamage" in df.columns:
        df["notRepairedDamage"] = (
            df["notRepairedDamage"]
            .replace({"-": np.nan, "0.0": "0", "1.0": "1"})
            .astype("string")
        )

    if "power" in df.columns:
        df["power_outlier_flag"] = ((df["power"] < 0) | (df["power"] > 600)).astype("int8")
        df["power"] = df["power"].clip(lower=0, upper=600)
        df["power_is_zero"] = (df["power"] == 0).astype("int8")
        if use_power_bin:
            df["power_bin"] = pd.cut(
                df["power"],
                bins=[-1, 0, 60, 90, 120, 150, 200, 300, 600],
                labels=False,
            ).astype("float")

    return df


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

    full = _normalize_columns(full, use_power_bin=use_power_bin)
    full = _add_date_features(full)
    full = _add_frequency_features(full)
    if use_group_stats:
        full = _add_group_statistics(full)
    if use_brand_relative:
        full = _add_brand_relative_features(full)
    if use_power_age:
        full = _add_power_age_features(full)
    if use_age_detail:
        full = _add_age_detail_features(full)
    if use_model_age_group_stats:
        full = _add_model_age_group_stats(full, min_count=model_age_group_min_count)
    if use_interactions:
        full = _add_interaction_features(full)

    feature_df = full.drop(
        columns=[
            col
            for col in ["SaleID", "offerType", "seller", "regDate", "creatDate", "name", "__is_train__"]
            if col in full.columns
        ]
    )

    categorical_columns = [
        col
        for col in [
            "model",
            "brand",
            "bodyType",
            "fuelType",
            "gearbox",
            "notRepairedDamage",
            "regionCode",
        ]
        if col in feature_df.columns
    ]

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
