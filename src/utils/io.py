from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file with pandas."""
    return pd.read_csv(Path(path), **kwargs)


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Save a dataframe as CSV after creating the parent directory."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    if "index" not in kwargs:
        kwargs["index"] = False
    df.to_csv(output_path, **kwargs)
    return output_path


def read_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a parquet file with pandas."""
    return pd.read_parquet(Path(path), **kwargs)


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Save a dataframe as parquet after creating the parent directory."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    df.to_parquet(output_path, **kwargs)
    return output_path

