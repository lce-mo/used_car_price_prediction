from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd

try:
    from .validate_data import validate_raw_dataframe
except ImportError:
    from validate_data import validate_raw_dataframe


LOGGER = logging.getLogger(__name__)

MISSING_TOKENS = ("", " ", "nan", "NaN", "None", "NULL")
DEFAULT_NUMERIC_COLUMNS = (
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
    "regionCode",
    "seller",
    "offerType",
    "creatDate",
    "price",
)


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common raw missing tokens to ``pd.NA``.

    Args:
        df: Input DataFrame.

    Returns:
        A copied DataFrame with known textual missing tokens replaced by
        ``pd.NA``.
    """
    cleaned = df.copy()
    return cleaned.replace(list(MISSING_TOKENS), pd.NA)


def clean_not_repaired_damage(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw ``notRepairedDamage`` field without changing its meaning.

    Args:
        df: Input DataFrame that may contain ``notRepairedDamage``.

    Returns:
        A copied DataFrame where ``-`` and empty values in
        ``notRepairedDamage`` are treated as missing.
    """
    cleaned = df.copy()
    if "notRepairedDamage" not in cleaned.columns:
        return cleaned

    cleaned["notRepairedDamage"] = (
        cleaned["notRepairedDamage"]
        .astype("string")
        .str.strip()
        .replace({"-": pd.NA, "": pd.NA})
    )
    return cleaned


def convert_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convert known numeric-looking raw columns with coercion.

    Args:
        df: Input DataFrame.
        columns: Optional numeric column names. Defaults to stable raw numeric
            columns plus all ``v_*`` anonymous feature columns.

    Returns:
        A copied DataFrame with selected columns converted via
        ``pd.to_numeric(errors="coerce")``.
    """
    cleaned = df.copy()
    selected = list(columns or DEFAULT_NUMERIC_COLUMNS)
    selected.extend(col for col in cleaned.columns if col.startswith("v_") and col not in selected)

    for column in selected:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned


def clean_dataframe(
    df: pd.DataFrame,
    dataset_name: str = "data",
    validate: bool = True,
) -> pd.DataFrame:
    """Apply conservative raw-data cleaning for downstream feature engineering.

    This function standardizes missing tokens, cleans ``notRepairedDamage``,
    and converts numeric-looking fields. It intentionally avoids distributional
    edits such as filling missing values or clipping outliers; those remain in
    the existing feature engineering layer.

    Args:
        df: Raw input DataFrame.
        dataset_name: Dataset label used in logs and validation messages.
        validate: Whether to validate the cleaned DataFrame before returning.

    Returns:
        Cleaned DataFrame compatible with ``src.features``.

    Raises:
        ValueError: If validation is enabled and the cleaned frame is malformed.
    """
    LOGGER.info("Cleaning %s data: rows=%d, columns=%d.", dataset_name, len(df), len(df.columns))
    cleaned = normalize_missing_values(df)
    cleaned = clean_not_repaired_damage(cleaned)
    cleaned = convert_numeric_columns(cleaned)

    if validate:
        validate_raw_dataframe(cleaned, dataset_name=dataset_name)

    LOGGER.info("Cleaned %s data: rows=%d, columns=%d.", dataset_name, len(cleaned), len(cleaned.columns))
    return cleaned


def clean_train_data(
    df: pd.DataFrame,
    validate: bool = True,
) -> pd.DataFrame:
    """Clean a raw training DataFrame.

    Args:
        df: Raw training DataFrame containing ``price``.
        validate: Whether to validate required columns and target presence.

    Returns:
        Cleaned training DataFrame.
    """
    cleaned = clean_dataframe(df, dataset_name="train", validate=False)
    if validate:
        validate_raw_dataframe(cleaned, dataset_name="train", require_target=True)
    return cleaned


def clean_test_data(
    df: pd.DataFrame,
    validate: bool = True,
) -> pd.DataFrame:
    """Clean a raw test DataFrame.

    Args:
        df: Raw test DataFrame.
        validate: Whether to validate required columns and key value ranges.

    Returns:
        Cleaned test DataFrame.
    """
    return clean_dataframe(df, dataset_name="test", validate=validate)


__all__ = [
    "MISSING_TOKENS",
    "DEFAULT_NUMERIC_COLUMNS",
    "normalize_missing_values",
    "clean_not_repaired_damage",
    "convert_numeric_columns",
    "clean_dataframe",
    "clean_train_data",
    "clean_test_data",
]
