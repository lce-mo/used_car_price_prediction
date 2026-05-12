from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Sequence

import pandas as pd

from src.config import RAW_DATA_DIR, RAW_DATA_SEPARATOR

try:
    from .validate_data import validate_raw_dataframe
except ImportError:
    from validate_data import validate_raw_dataframe


LOGGER = logging.getLogger(__name__)

TRAIN_FILE_CANDIDATES = (
    "used_car_train_20200313.csv",
    "used_car_train_first50000_correct.csv",
    "used_car_train_first50000.csv",
)
TEST_FILE_CANDIDATES = (
    "used_car_testB_20200421.csv",
    "used_car_testA_20200313.csv",
)


def resolve_raw_file(
    path: str | Path | None,
    candidates: Sequence[str],
    label: str,
) -> Path:
    """Resolve an explicit raw-file path or the first existing default file.

    Args:
        path: Explicit file path. When provided, the file must exist.
        candidates: Ordered filenames searched under ``RAW_DATA_DIR`` when
            ``path`` is omitted.
        label: Human-readable file label used in exceptions.

    Returns:
        Resolved raw-data file path.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    if path is not None:
        resolved = Path(path)
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"{label} file not found: {resolved}")

    for filename in candidates:
        candidate = RAW_DATA_DIR / filename
        if candidate.exists():
            return candidate

    searched = ", ".join(str(RAW_DATA_DIR / filename) for filename in candidates)
    raise FileNotFoundError(f"No {label} file found. Searched: {searched}")


def read_raw_dataframe(
    path: str | Path,
    dataset_name: str = "raw",
    validate: bool = True,
) -> pd.DataFrame:
    """Read one raw Tianchi used-car file with the project-approved separator.

    Args:
        path: Raw data file path.
        dataset_name: Dataset label used for logs and validation messages.
        validate: Whether to run raw-field validation after loading.

    Returns:
        Loaded raw DataFrame.

    Raises:
        ValueError: If validation is enabled and the frame is malformed.
    """
    resolved = Path(path)
    LOGGER.info("Reading %s data from %s with sep=%r.", dataset_name, resolved, RAW_DATA_SEPARATOR)
    df = pd.read_csv(resolved, sep=RAW_DATA_SEPARATOR)
    if validate:
        validate_raw_dataframe(df, path=resolved, dataset_name=dataset_name)
    LOGGER.info("Loaded %s data: rows=%d, columns=%d.", dataset_name, len(df), len(df.columns))
    return df


def load_train_data(
    path: str | Path | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the raw training DataFrame.

    Args:
        path: Optional explicit train file path. Defaults to known raw training
            candidates under ``RAW_DATA_DIR``.
        validate: Whether to validate required fields, target presence, and key
            value ranges.

    Returns:
        Raw training DataFrame containing the ``price`` target.
    """
    train_path = resolve_raw_file(path, TRAIN_FILE_CANDIDATES, "train")
    return read_raw_dataframe(train_path, dataset_name="train", validate=validate)


def load_test_data(
    path: str | Path | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the raw test DataFrame.

    Args:
        path: Optional explicit test file path. Defaults to known raw test
            candidates under ``RAW_DATA_DIR``.
        validate: Whether to validate required fields and key value ranges.

    Returns:
        Raw test DataFrame.
    """
    test_path = resolve_raw_file(path, TEST_FILE_CANDIDATES, "test")
    return read_raw_dataframe(test_path, dataset_name="test", validate=validate)


def load_train_test_data(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test DataFrames using the same validated reader.

    Args:
        train_path: Optional explicit training file path.
        test_path: Optional explicit test file path.
        validate: Whether to run validation for both DataFrames.

    Returns:
        A tuple of ``(train_df, test_df)``.
    """
    train_df = load_train_data(train_path, validate=validate)
    test_df = load_test_data(test_path, validate=validate)
    return train_df, test_df


__all__ = [
    "TRAIN_FILE_CANDIDATES",
    "TEST_FILE_CANDIDATES",
    "resolve_raw_file",
    "read_raw_dataframe",
    "load_train_data",
    "load_test_data",
    "load_train_test_data",
]
