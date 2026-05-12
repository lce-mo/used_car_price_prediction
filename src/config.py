from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"

EXPERIMENT_DIR = BASE_DIR / "experiments"

TARGET_COL = "price"
ID_COL = "SaleID"
RANDOM_STATE = 42

RAW_DATA_SEPARATOR = " "


@dataclass(frozen=True)
class TrainingPathConfig:
    """Default file locations for the current training pipeline."""

    train_path: Path = Path("data/raw/used_car_train_20200313.csv")
    test_path: Path = Path("data/raw/used_car_testB_20200421.csv")
    output_dir: Path = Path("outputs/main_lgbm_m2_q3")


@dataclass(frozen=True)
class ModelConfig:
    """Default model hyperparameters."""

    model_name: str = "lightgbm"
    learning_rate: float = 0.08
    n_estimators: int = 400
    num_leaves: int = 63
    random_state: int = RANDOM_STATE
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    lightgbm_objective: str = "regression"
    target_mode: str = "log1p"


@dataclass(frozen=True)
class CVConfig:
    """Default cross-validation settings."""

    strategy: str = "repeated_stratified"
    repeats: int = 3
    random_state: int = RANDOM_STATE
    stratify_price_bins: int = 5


@dataclass(frozen=True)
class FeatureConfig:
    """Default feature switch settings for the legacy training pipeline."""

    use_group_stats: bool = True
    use_power_bin: bool = False
    use_interactions: bool = False
    use_brand_relative: bool = False
    use_power_age: bool = False
    use_age_detail: bool = False
    use_model_age_group_stats: bool = False
    model_age_group_min_count: int = 20
    target_encoding_smoothing: float = 20.0


@dataclass(frozen=True)
class SampleWeightConfig:
    """Default sample-weight settings."""

    use_sample_weighting: bool = False
    high_price_quantile: float = 0.8
    high_price_weight: float = 1.5
    new_car_max_years: float = 3.0
    new_car_weight: float = 1.3
    price_age_slice_weight: float = 1.1
    price_age_slice_targets: str = "Q5:8y_plus,Q4:8y_plus,Q5:5_8y,Q3:8y_plus,Q5:3_5y"
    normalize_sample_weight: bool = True


TRAINING_PATH_CONFIG = TrainingPathConfig()
MODEL_CONFIG = ModelConfig()
CV_CONFIG = CVConfig()
FEATURE_CONFIG = FeatureConfig()
SAMPLE_WEIGHT_CONFIG = SampleWeightConfig()


def output_directories() -> list[Path]:
    """Return directories that are safe to create during normal runs."""
    return [
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        PREDICTION_DIR,
        SUBMISSION_DIR,
        REPORT_DIR,
        FIGURE_DIR,
        EXPERIMENT_DIR,
    ]
