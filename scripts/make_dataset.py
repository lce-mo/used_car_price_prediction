from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import INTERIM_DATA_DIR
from src.data.clean_data import clean_test_data, clean_train_data
from src.data.load_data import load_test_data, load_train_data
from src.utils.io import save_csv


def main() -> None:
    train_df = clean_train_data(load_train_data())
    test_df = clean_test_data(load_test_data())
    save_csv(train_df, INTERIM_DATA_DIR / "train_clean.csv")
    save_csv(test_df, INTERIM_DATA_DIR / "test_clean.csv")
    print("Saved cleaned data to data/interim/")


if __name__ == "__main__":
    main()
