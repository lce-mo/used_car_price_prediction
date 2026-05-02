from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from features import prepare_features
from train import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    cross_validate_train,
    load_data,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small LightGBM parameter search around the current Q3 main version."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the training file.",
    )
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
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of CV folds for each parameter set.",
    )
    parser.add_argument(
        "--learning-rates",
        type=str,
        default="0.06,0.08,0.10",
        help="Comma-separated learning rates to evaluate.",
    )
    parser.add_argument(
        "--n-estimators-grid",
        type=str,
        default="300,400,500",
        help="Comma-separated n_estimators values to evaluate.",
    )
    parser.add_argument(
        "--num-leaves-grid",
        type=str,
        default="31,63,95",
        help="Comma-separated num_leaves values to evaluate.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["price", "log1p"],
        default="log1p",
        help="Whether to train on raw price or log1p(price).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
        use_group_stats=True,
        use_power_bin=False,
        use_interactions=False,
    )
    use_log_target = args.target_mode == "log1p"

    results: list[dict[str, float | int]] = []
    total = len(learning_rates) * len(n_estimators_grid) * len(num_leaves_grid)

    for run_idx, (learning_rate, n_estimators, num_leaves) in enumerate(
        itertools.product(learning_rates, n_estimators_grid, num_leaves_grid),
        start=1,
    ):
        print(
            f"[{run_idx}/{total}] "
            f"lr={learning_rate}, n_estimators={n_estimators}, num_leaves={num_leaves}"
        )
        fold_metrics, _ = cross_validate_train(
            prepared=prepared,
            n_splits=args.n_splits,
            use_log_target=use_log_target,
            model_name="lightgbm",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            num_leaves=num_leaves,
        )
        maes = [item["mae"] for item in fold_metrics]
        result = {
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "mae_mean": float(sum(maes) / len(maes)),
            "mae_std": float(pd.Series(maes, dtype=float).std(ddof=0)),
        }
        results.append(result)

    result_df = pd.DataFrame(results).sort_values(
        by=["mae_mean", "mae_std", "learning_rate", "n_estimators", "num_leaves"],
        ascending=[True, True, True, True, True],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_dir / "search_results.csv", index=False)

    best = result_df.iloc[0].to_dict()
    with (args.output_dir / "best_result.json").open("w", encoding="utf-8") as file:
        json.dump(best, file, ensure_ascii=False, indent=2)

    print("Best configuration:")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
