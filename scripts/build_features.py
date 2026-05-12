from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.data.clean_data import clean_test_data, clean_train_data
from src.data.load_data import load_test_data, load_train_data
from src.features.build_features import build_test_features, build_train_features
from src.utils.io import read_csv, save_csv


def _load_clean_data():
    train_path = INTERIM_DATA_DIR / "train_clean.csv"
    test_path = INTERIM_DATA_DIR / "test_clean.csv"
    if train_path.exists() and test_path.exists():
        return read_csv(train_path), read_csv(test_path)
    return clean_train_data(load_train_data()), clean_test_data(load_test_data())


def main() -> None:
    train_df, test_df = _load_clean_data()
    train_features = build_train_features(train_df)
    test_features = build_test_features(test_df)
    save_csv(train_features, PROCESSED_DATA_DIR / "train_features.csv")
    save_csv(test_features, PROCESSED_DATA_DIR / "test_features.csv")
    print("Saved processed features to data/processed/")


if __name__ == "__main__":
    main()
