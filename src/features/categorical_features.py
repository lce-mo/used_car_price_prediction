from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS = [
    "model",
    "brand",
    "bodyType",
    "fuelType",
    "gearbox",
    "notRepairedDamage",
    "regionCode",
]

FREQUENCY_COLUMNS = ["name", "brand", "regionCode", "model"]
MISSING_CATEGORY = "__MISSING__"


def _strip_category(series: pd.Series) -> pd.Series:
    """Normalize category keys without changing the row index."""
    return series.astype("string").str.strip()


def normalize_categorical_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize raw categorical fields to stable string keys.

    The legacy training pipeline treats these columns as categorical after
    feature construction. Keeping a single missing token here makes ordinal
    encoding and frequency/target encodings consume the same category space.
    """
    featured = df.copy()
    selected_columns = columns or CATEGORICAL_COLUMNS

    for col in selected_columns:
        if col not in featured.columns:
            continue

        values = _strip_category(featured[col])
        if col == "notRepairedDamage":
            # The raw file uses "-" for missing and "0.0"/"1.0" for the two
            # observed states. The legacy feature builder canonicalized these
            # to missing, "0", and "1" before model preprocessing.
            values = values.replace({"-": pd.NA, "": pd.NA, "0.0": "0", "1.0": "1"})
        else:
            values = values.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})

        featured[col] = values.fillna(MISSING_CATEGORY)

    return featured


def add_count_encoding(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Add row-frequency features for high-cardinality category-like columns."""
    featured = df.copy()
    selected_columns = columns or FREQUENCY_COLUMNS
    for col in selected_columns:
        if col in featured.columns:
            counts = featured[col].value_counts(dropna=False)
            featured[f"{col}_count"] = featured[col].map(counts).astype("float")
    return featured


def add_model_frequency_bin(
    df: pd.DataFrame,
    model_col: str = "model",
    output_col: str = "model_frequency_bin",
) -> pd.DataFrame:
    """Bucket model frequency for reusable stratification or diagnostics.

    This mirrors the frequency buckets used by the current CV code, but keeps
    the calculation available from the feature layer without making training
    depend on it yet.
    """
    featured = df.copy()
    if model_col not in featured.columns:
        return featured

    model_key = _strip_category(featured[model_col]).fillna(MISSING_CATEGORY)
    model_count = model_key.map(model_key.value_counts(dropna=False)).astype(float)
    freq_bin = pd.cut(
        model_count,
        bins=[-np.inf, 1, 2, 5, 10, 20, 50, np.inf],
        labels=["f1", "f2", "f3_5", "f6_10", "f11_20", "f21_50", "f51_plus"],
    )
    featured[output_col] = freq_bin.astype("string").fillna("__FREQ_MISSING__")
    return featured


def add_categorical_features(df: pd.DataFrame, add_counts: bool = False) -> pd.DataFrame:
    """Build categorical features while preserving the legacy public interface."""
    featured = normalize_categorical_columns(df)

    if add_counts:
        featured = add_count_encoding(featured)
    return featured


__all__ = [
    "CATEGORICAL_COLUMNS",
    "FREQUENCY_COLUMNS",
    "MISSING_CATEGORY",
    "normalize_categorical_columns",
    "add_count_encoding",
    "add_model_frequency_bin",
    "add_categorical_features",
]
