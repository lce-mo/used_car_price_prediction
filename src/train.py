"""Compatibility wrapper for the model training entrypoint.

The training implementation now lives in src.models.train_model. This module
keeps old imports such as `from train import load_data` working for experiment
scripts while avoiding a second copy of the training workflow.
"""

from __future__ import annotations

from src.models.cross_validation import cross_validate_train
from src.models.train_model import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    RAW_DATA_SEPARATOR,
    load_data,
    main,
    parse_args,
    sample_rows,
    train,
    validate_raw_dataframe,
)

__all__ = [
    "DEFAULT_TEST_PATH",
    "DEFAULT_TRAIN_PATH",
    "RAW_DATA_SEPARATOR",
    "cross_validate_train",
    "load_data",
    "main",
    "parse_args",
    "sample_rows",
    "train",
    "validate_raw_dataframe",
]


if __name__ == "__main__":
    main()
