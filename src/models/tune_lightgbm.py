from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from src.config import CV_CONFIG, FEATURE_CONFIG, MODEL_CONFIG, SAMPLE_WEIGHT_CONFIG
from src.features import prepare_features
from src.models.cross_validation import cross_validate_train
from src.models.train_model import DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, load_data, resolve_sample_weight_mode


def _parse_float_grid(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one learning rate is required.")
    return [float(item) for item in values]


def _parse_int_grid(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return [int(item) for item in values]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LightGBM parameter search on the current feature pipeline.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH, help="Path to the training file.")
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Path to the current online evaluation file. Defaults to testB.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lightgbm_local_search"),
        help="Directory used to save search results.",
    )
    parser.add_argument("--n-splits", type=int, default=3, help="Number of CV folds for each parameter set.")
    parser.add_argument(
        "--cv-strategy",
        choices=["kfold", "repeated_stratified"],
        default="kfold",
        help="CV strategy for local parameter search.",
    )
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--cv-random-state", type=int, default=CV_CONFIG.random_state)
    parser.add_argument("--stratify-price-bins", type=int, default=CV_CONFIG.stratify_price_bins)
    parser.add_argument("--learning-rates", type=str, default="0.06,0.08,0.10")
    parser.add_argument("--n-estimators-grid", type=str, default="300,400,500")
    parser.add_argument("--num-leaves-grid", type=str, default="31,63,95")
    parser.add_argument("--target-mode", choices=["price", "log1p", "sqrt", "pow075"], default=MODEL_CONFIG.target_mode)
    parser.add_argument("--use-power-age", choices=["true", "false"], default="true" if FEATURE_CONFIG.use_power_age else "false")
    parser.add_argument(
        "--use-model-age-group-stats",
        choices=["true", "false"],
        default="true" if FEATURE_CONFIG.use_model_age_group_stats else "false",
    )
    parser.add_argument("--model-age-group-min-count", type=int, default=FEATURE_CONFIG.model_age_group_min_count)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.train_path.exists():
        raise FileNotFoundError(f"Training file not found: {args.train_path}")
    if not args.test_path.exists():
        raise FileNotFoundError(f"Test file not found: {args.test_path}")

    learning_rates = _parse_float_grid(args.learning_rates)
    n_estimators_grid = _parse_int_grid(args.n_estimators_grid)
    num_leaves_grid = _parse_int_grid(args.num_leaves_grid)

    train_df, test_df = load_data(args.train_path, args.test_path)
    prepared = prepare_features(
        train_df,
        test_df,
        use_group_stats=FEATURE_CONFIG.use_group_stats,
        use_power_bin=FEATURE_CONFIG.use_power_bin,
        use_interactions=FEATURE_CONFIG.use_interactions,
        use_brand_relative=FEATURE_CONFIG.use_brand_relative,
        use_power_age=args.use_power_age == "true",
        use_age_detail=FEATURE_CONFIG.use_age_detail,
        use_model_age_group_stats=args.use_model_age_group_stats == "true",
        model_age_group_min_count=args.model_age_group_min_count,
    )

    use_sample_weighting = SAMPLE_WEIGHT_CONFIG.use_sample_weighting
    sample_weight_mode = resolve_sample_weight_mode(None, use_sample_weighting)
    results: list[dict[str, float | int]] = []
    total = len(learning_rates) * len(n_estimators_grid) * len(num_leaves_grid)

    for run_idx, (learning_rate, n_estimators, num_leaves) in enumerate(
        itertools.product(learning_rates, n_estimators_grid, num_leaves_grid),
        start=1,
    ):
        print(f"[{run_idx}/{total}] lr={learning_rate}, n_estimators={n_estimators}, num_leaves={num_leaves}")
        fold_metrics, _, _, _ = cross_validate_train(
            prepared=prepared,
            n_splits=args.n_splits,
            cv_strategy=args.cv_strategy,
            cv_repeats=args.cv_repeats,
            cv_random_state=args.cv_random_state,
            stratify_price_bins=args.stratify_price_bins,
            use_log_target=args.target_mode,
            model_name="lightgbm",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            model_random_state=MODEL_CONFIG.random_state,
            subsample=MODEL_CONFIG.subsample,
            colsample_bytree=MODEL_CONFIG.colsample_bytree,
            lightgbm_objective=MODEL_CONFIG.lightgbm_objective,
            use_sample_weighting=use_sample_weighting,
            sample_weight_mode=sample_weight_mode,
            high_price_quantile=SAMPLE_WEIGHT_CONFIG.high_price_quantile,
            high_price_weight=SAMPLE_WEIGHT_CONFIG.high_price_weight,
            new_car_max_years=SAMPLE_WEIGHT_CONFIG.new_car_max_years,
            new_car_weight=SAMPLE_WEIGHT_CONFIG.new_car_weight,
            price_age_slice_weight=SAMPLE_WEIGHT_CONFIG.price_age_slice_weight,
            price_age_slice_targets=SAMPLE_WEIGHT_CONFIG.price_age_slice_targets,
            normalize_sample_weight=SAMPLE_WEIGHT_CONFIG.normalize_sample_weight,
            use_brand_target_encoding=False,
            use_brand_age_target_encoding=False,
            use_model_target_encoding=False,
            use_model_age_target_encoding=False,
            use_model_backoff_target_encoding=False,
            use_model_low_freq_flag=False,
            model_backoff_min_count=20,
            model_low_freq_min_count=20,
            target_encoding_smoothing=FEATURE_CONFIG.target_encoding_smoothing,
            use_segmented_modeling=False,
            segment_routing_mode="global_pred_plus_age",
            segment_scope="q5_5plus",
        )
        maes = [item["mae"] for item in fold_metrics]
        results.append(
            {
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "num_leaves": num_leaves,
                "mae_mean": float(sum(maes) / len(maes)),
                "mae_std": float(pd.Series(maes, dtype=float).std(ddof=0)),
            }
        )

    result_df = pd.DataFrame(results).sort_values(
        by=["mae_mean", "mae_std", "learning_rate", "n_estimators", "num_leaves"],
        ascending=[True, True, True, True, True],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_dir / "search_results.csv", index=False)

    best = result_df.iloc[0].to_dict()
    (args.output_dir / "best_result.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Best configuration:")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
