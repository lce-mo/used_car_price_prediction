from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)

COMMON_REQUIRED_COLUMNS = (
    "SaleID",
    "name",
    "regDate",
    "model",
    "brand",
    "bodyType",
    "fuelType",
    "gearbox",
    "power",
    "kilometer",
    "notRepairedDamage",
    "regionCode",
    "seller",
    "offerType",
    "creatDate",
)
TARGET_COLUMN = "price"
VALUE_RANGE_CHECKS = {
    "kilometer": (0, 15),
    "bodyType": (0, 7),
    "fuelType": (0, 6),
    "gearbox": (0, 1),
    "regionCode": (0, 100000),
}
MAX_OUT_OF_RANGE_RATE = 0.001


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str = "data",
) -> pd.DataFrame:
    """Validate that a DataFrame contains all required columns.

    Args:
        df: DataFrame to validate.
        required_columns: Column names that must be present.
        dataset_name: Human-readable dataset label for logs and exceptions.

    Returns:
        The original DataFrame, unchanged, to support pipeline-style calls.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
    LOGGER.info("%s column check passed with %d columns.", dataset_name, len(df.columns))
    return df


def validate_value_ranges(
    df: pd.DataFrame,
    dataset_name: str = "data",
    strict: bool = False,
) -> pd.DataFrame:
    """Validate key raw-field value ranges used to detect malformed reads.

    Args:
        df: DataFrame to validate.
        dataset_name: Human-readable dataset label for logs and exceptions.
        strict: When true, any out-of-range value fails validation. When false,
            validation fails only if the out-of-range rate exceeds the project
            threshold used to catch column-shifted raw data.

    Returns:
        The original DataFrame, unchanged.

    Raises:
        ValueError: If numeric value checks indicate malformed or illegal data.
    """
    problems: list[str] = []

    if TARGET_COLUMN in df.columns:
        price = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        if bool(price.lt(0).any()):
            problems.append(f"{TARGET_COLUMN} has negative values; min={price.min()}")

    for column, (lower, upper) in VALUE_RANGE_CHECKS.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        bad_mask = (values < lower) | (values > upper)
        bad_rate = float(bad_mask.mean())
        if (strict and bool(bad_mask.any())) or bad_rate > MAX_OUT_OF_RANGE_RATE:
            problems.append(f"{column} out-of-range rate={bad_rate:.4%}")

    if problems:
        details = "; ".join(problems)
        raise ValueError(
            f"{dataset_name} looks column-shifted or malformed: {details}. "
            "The Tianchi used-car raw files contain empty fields, so read them "
            "with sep=' ', not sep=r'\\s+'."
        )

    LOGGER.info("%s value range check passed.", dataset_name)
    return df


def validate_raw_dataframe(
    df: pd.DataFrame,
    path: str | Path | None = None,
    dataset_name: str = "raw",
    require_target: bool | None = None,
) -> pd.DataFrame:
    """Run the full raw-data validation suite for the used-car dataset.

    Args:
        df: Raw DataFrame loaded from the Tianchi source file.
        path: Optional source path included in error messages and logs.
        dataset_name: Human-readable dataset label, such as ``train`` or
            ``test``.
        require_target: Whether the ``price`` target column must be present. If
            omitted, training datasets are inferred from the dataset name.

    Returns:
        The original DataFrame, unchanged, after all validations pass.

    Raises:
        ValueError: If the frame is empty, required columns are missing, target
            requirements are not met, or value ranges indicate a bad read.
    """
    source = f" ({Path(path)})" if path is not None else ""
    label = f"{dataset_name}{source}"

    if df.empty:
        raise ValueError(f"{label} is empty.")

    validate_required_columns(df, COMMON_REQUIRED_COLUMNS, dataset_name=label)

    target_required = require_target
    if target_required is None:
        target_required = dataset_name.lower() in {"train", "training"}
    if target_required and TARGET_COLUMN not in df.columns:
        raise ValueError(f"{label} must contain target column: {TARGET_COLUMN}")

    validate_value_ranges(df, dataset_name=label)
    LOGGER.info("%s raw validation passed with %d rows.", label, len(df))
    return df


__all__ = [
    "COMMON_REQUIRED_COLUMNS",
    "TARGET_COLUMN",
    "VALUE_RANGE_CHECKS",
    "validate_required_columns",
    "validate_value_ranges",
    "validate_raw_dataframe",
]
