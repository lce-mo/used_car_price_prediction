from __future__ import annotations

import logging
from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src.config import RANDOM_STATE


LOGGER = logging.getLogger(__name__)


def sample_rows(
    df: pd.DataFrame,
    sample_size: int | None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Return a reproducible row sample for smaller experiments.

    Args:
        df: Input DataFrame.
        sample_size: Number of rows to sample. ``None`` returns a copy of the
            full DataFrame.
        random_state: Seed used by pandas sampling.

    Returns:
        Sampled DataFrame with a reset integer index.

    Raises:
        ValueError: If ``sample_size`` is not positive.
    """
    if sample_size is None:
        LOGGER.info("No sampling requested; returning %d rows.", len(df))
        return df.copy().reset_index(drop=True)
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer.")
    if sample_size >= len(df):
        LOGGER.info("Requested sample_size=%d >= rows=%d; returning all rows.", sample_size, len(df))
        return df.copy().reset_index(drop=True)

    sampled = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    LOGGER.info("Sampled %d rows from %d rows.", len(sampled), len(df))
    return sampled


def train_valid_split(
    df: pd.DataFrame,
    valid_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and validation DataFrames.

    Args:
        df: Input DataFrame.
        valid_size: Validation fraction or absolute row count accepted by
            ``sklearn.model_selection.train_test_split``.
        random_state: Seed used when shuffling.
        shuffle: Whether to shuffle before splitting.

    Returns:
        A tuple of ``(train_df, valid_df)`` with reset integer indices.

    Raises:
        ValueError: If the input frame is too small to split.
    """
    if len(df) < 2:
        raise ValueError("At least two rows are required for a train/valid split.")

    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state if shuffle else None,
        shuffle=shuffle,
    )
    LOGGER.info("Created train/valid split: train=%d, valid=%d.", len(train_df), len(valid_df))
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def build_cv_split_indices(
    df: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield KFold train/validation index lists.

    Args:
        df: Input DataFrame.
        n_splits: Number of KFold splits.
        shuffle: Whether to shuffle rows before splitting.
        random_state: Seed used when shuffling.

    Returns:
        Iterator of ``(train_indices, valid_indices)`` lists.

    Raises:
        ValueError: If ``n_splits`` is invalid for the input size.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > len(df):
        raise ValueError("n_splits cannot exceed the number of rows.")

    splitter = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(df), start=1):
        LOGGER.info("Prepared fold %d/%d: train=%d, valid=%d.", fold_idx, n_splits, len(train_idx), len(valid_idx))
        yield train_idx.tolist(), valid_idx.tolist()


def build_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield KFold train/validation DataFrame pairs.

    Args:
        df: Input DataFrame.
        n_splits: Number of KFold splits.
        shuffle: Whether to shuffle rows before splitting.
        random_state: Seed used when shuffling.

    Returns:
        Iterator of ``(train_df, valid_df)`` pairs with reset integer indices.
    """
    for train_idx, valid_idx in build_cv_split_indices(
        df=df,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    ):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
        yield train_df, valid_df


__all__ = [
    "sample_rows",
    "train_valid_split",
    "build_cv_split_indices",
    "build_kfold_splits",
]
