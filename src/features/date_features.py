from __future__ import annotations

import numpy as np
import pandas as pd


def parse_compact_date(series: pd.Series) -> pd.Series:
    """Parse dates stored as compact YYYYMMDD integers or strings."""
    text = series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def build_age_bin(
    series: pd.Series,
    output_missing_label: str = "__AGE_MISSING__",
) -> pd.Series:
    """Bucket car age into stable labels for slicing, TE keys, and CV reuse."""
    age_years = pd.to_numeric(series, errors="coerce").clip(lower=0)
    bins = pd.cut(
        age_years,
        bins=[-np.inf, 1, 3, 5, 8, np.inf],
        labels=["0_1y", "1_3y", "3_5y", "5_8y", "8y_plus"],
    )
    return bins.astype("string").fillna(output_missing_label)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add registration/listing date features and vehicle age.

    These are direct equivalents of the legacy mainline features:
    - registration date parts from `regDate`
    - listing/create date parts from `creatDate`
    - car age in days and years, computed as listing date minus reg date
    """
    featured = df.copy()
    reg_date = (
        parse_compact_date(featured["regDate"])
        if "regDate" in featured
        else pd.Series(pd.NaT, index=featured.index)
    )
    create_date = (
        parse_compact_date(featured["creatDate"])
        if "creatDate" in featured
        else pd.Series(pd.NaT, index=featured.index)
    )

    featured["reg_year"] = reg_date.dt.year
    featured["reg_month"] = reg_date.dt.month
    featured["reg_day"] = reg_date.dt.day
    featured["reg_weekday"] = reg_date.dt.weekday

    featured["create_year"] = create_date.dt.year
    featured["create_month"] = create_date.dt.month
    featured["create_day"] = create_date.dt.day
    featured["create_weekday"] = create_date.dt.weekday

    car_age_days = (create_date - reg_date).dt.days
    featured["car_age_days"] = car_age_days
    featured["car_age_years"] = car_age_days / 365.25
    return featured


def add_age_detail_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add optional finer-grained age flags used in legacy experiments."""
    featured = df.copy()

    if "car_age_days" in featured.columns:
        age_days = pd.to_numeric(featured["car_age_days"], errors="coerce").clip(lower=0)
        featured["car_age_months"] = age_days / 30.4

    if "car_age_years" in featured.columns:
        age_years = pd.to_numeric(featured["car_age_years"], errors="coerce").clip(lower=0)
        featured["is_nearly_new_1y"] = (age_years <= 1).astype("int8")
        featured["is_nearly_new_3y"] = (age_years <= 3).astype("int8")

    return featured


__all__ = [
    "parse_compact_date",
    "build_age_bin",
    "add_date_features",
    "add_age_detail_features",
]
