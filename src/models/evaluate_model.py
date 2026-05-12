from __future__ import annotations

import argparse
import json
from itertools import product
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from src.utils.metrics import mae as metric_mae
from src.utils.metrics import r2, rmse


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Return common regression metrics for used-car price prediction."""
    return {
        "mae": metric_mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search grid weights for multi-model OOF blends.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model OOF in name=path format. Can be repeated.",
    )
    parser.add_argument(
        "--submission",
        action="append",
        default=[],
        help="Optional test prediction in name=path format. Names must match --model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-step", type=float, default=0.025)
    parser.add_argument(
        "--search-mode",
        choices=["grid", "optimize"],
        default="optimize",
        help="Use exhaustive simplex grid or continuous optimization plus rounded-grid candidates.",
    )
    parser.add_argument("--meta-cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=200)
    return parser.parse_args(argv)


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("Named path must use name=path format.")
    name, path = raw.split("=", maxsplit=1)
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty.")
    return name, Path(path.strip())


def load_oof(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"SaleID", "price", "oof_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["SaleID", "price", "oof_pred"]].rename(columns={"oof_pred": name})


def load_submission(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"SaleID", "price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["SaleID", "price"]].rename(columns={"price": name})


def load_oof_matrix(named_paths: list[tuple[str, Path]]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    merged: pd.DataFrame | None = None
    for name, path in named_paths:
        df = load_oof(name, path)
        if merged is None:
            merged = df
            continue
        merged = merged.merge(df[["SaleID", "price", name]], on=["SaleID", "price"], how="inner")

    if merged is None:
        raise ValueError("At least one model is required.")
    if len(merged) == 0:
        raise ValueError("OOF merge produced no rows.")

    pred_matrix = merged[[name for name, _ in named_paths]].to_numpy(dtype=float)
    target = merged["price"].to_numpy(dtype=float)
    return merged, pred_matrix, target


def load_submission_matrix(
    named_paths: list[tuple[str, Path]],
    submission_paths: dict[str, Path],
) -> tuple[pd.DataFrame, np.ndarray] | None:
    if not submission_paths:
        return None

    missing = [name for name, _ in named_paths if name not in submission_paths]
    extra = sorted(set(submission_paths).difference(name for name, _ in named_paths))
    if missing or extra:
        raise ValueError(f"Submission names must match model names. missing={missing}, extra={extra}")

    merged: pd.DataFrame | None = None
    for name, _ in named_paths:
        df = load_submission(name, submission_paths[name])
        if merged is None:
            merged = df
            continue
        merged = merged.merge(df[["SaleID", name]], on="SaleID", how="inner")

    if merged is None:
        return None
    pred_matrix = merged[[name for name, _ in named_paths]].to_numpy(dtype=float)
    return merged, pred_matrix


def build_weight_grid(model_count: int, grid_step: float) -> np.ndarray:
    if model_count < 2:
        raise ValueError("Need at least two models for blending.")
    if grid_step <= 0 or grid_step > 1:
        raise ValueError("grid-step must be in (0, 1].")

    denominator = round(1.0 / grid_step)
    if not np.isclose(denominator * grid_step, 1.0):
        raise ValueError("grid-step must divide 1.0 exactly, e.g. 0.025 or 0.05.")

    rows: list[tuple[int, ...]] = []

    def fill(prefix: tuple[int, ...], remaining: int, slots_left: int) -> None:
        if slots_left == 1:
            rows.append(prefix + (remaining,))
            return
        for value in range(remaining + 1):
            fill(prefix + (value,), remaining - value, slots_left - 1)

    fill((), denominator, model_count)
    return np.asarray(rows, dtype=float) / float(denominator)


def mae_for_weights(pred_matrix: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    pred = pred_matrix @ weights
    return float(np.mean(np.abs(target - pred.clip(min=0))))


def search_best_weights(
    pred_matrix: np.ndarray,
    target: np.ndarray,
    weight_grid: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    scores = np.empty(len(weight_grid), dtype=float)
    for idx, weights in enumerate(weight_grid):
        scores[idx] = mae_for_weights(pred_matrix, target, weights)

    order = np.argsort(scores, kind="mergesort")
    top_order = order[: min(top_k, len(order))]
    top_rows = pd.DataFrame(weight_grid[top_order])
    top_rows["mae"] = scores[top_order]
    return weight_grid[order[0]], float(scores[order[0]]), top_rows


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    total = float(weights.sum())
    if total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def round_to_grid(weights: np.ndarray, grid_step: float) -> np.ndarray:
    denominator = int(round(1.0 / grid_step))
    raw = normalize_weights(weights) * denominator
    floors = np.floor(raw).astype(int)
    remainder = denominator - int(floors.sum())
    if remainder > 0:
        order = np.argsort(raw - floors)[::-1]
        floors[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(raw - floors)
        for idx in order:
            if remainder == 0:
                break
            removable = min(floors[idx], -remainder)
            floors[idx] -= removable
            remainder += removable
    return floors.astype(float) / denominator


def optimize_weights(pred_matrix: np.ndarray, target: np.ndarray, starts: list[np.ndarray]) -> tuple[np.ndarray, float]:
    model_count = pred_matrix.shape[1]

    def objective(weights: np.ndarray) -> float:
        return mae_for_weights(pred_matrix, target, normalize_weights(weights))

    constraints = [{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}]
    bounds = [(0.0, 1.0)] * model_count

    best_weights: np.ndarray | None = None
    best_mae = float("inf")
    for start in starts:
        result = minimize(
            objective,
            normalize_weights(start),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-10, "disp": False},
        )
        candidate_weights = normalize_weights(result.x if result.success else start)
        candidate_mae = mae_for_weights(pred_matrix, target, candidate_weights)
        if candidate_mae < best_mae:
            best_weights = candidate_weights
            best_mae = candidate_mae

    if best_weights is None:
        raise RuntimeError("Optimization did not produce weights.")
    return best_weights, best_mae


def build_rounded_neighbor_grid(center: np.ndarray, grid_step: float, radius_steps: int = 4) -> np.ndarray:
    denominator = int(round(1.0 / grid_step))
    center_units = np.rint(round_to_grid(center, grid_step) * denominator).astype(int)
    model_count = len(center_units)
    rows: set[tuple[int, ...]] = set()
    offsets = range(-radius_steps, radius_steps + 1)
    for delta in product(offsets, repeat=model_count - 1):
        candidate = center_units.copy()
        candidate[:-1] += np.asarray(delta, dtype=int)
        candidate[-1] = denominator - int(candidate[:-1].sum())
        if np.all(candidate >= 0) and np.all(candidate <= denominator):
            rows.add(tuple(int(value) for value in candidate))
    if not rows:
        rows.add(tuple(int(value) for value in center_units))
    return np.asarray(sorted(rows), dtype=float) / float(denominator)


def build_pairwise_neighbor_grid(center: np.ndarray, grid_step: float, max_transfer_steps: int = 2) -> np.ndarray:
    denominator = int(round(1.0 / grid_step))
    center_units = np.rint(round_to_grid(center, grid_step) * denominator).astype(int)
    model_count = len(center_units)
    rows: set[tuple[int, ...]] = {tuple(int(value) for value in center_units)}

    for from_idx in range(model_count):
        for to_idx in range(model_count):
            if from_idx == to_idx:
                continue
            for transfer in range(1, max_transfer_steps + 1):
                if center_units[from_idx] < transfer:
                    break
                candidate = center_units.copy()
                candidate[from_idx] -= transfer
                candidate[to_idx] += transfer
                rows.add(tuple(int(value) for value in candidate))

    return np.asarray(sorted(rows), dtype=float) / float(denominator)


def search_best_weights_optimized(
    pred_matrix: np.ndarray,
    target: np.ndarray,
    model_names: list[str],
    grid_step: float,
    top_k: int,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    model_count = pred_matrix.shape[1]
    starts = [np.full(model_count, 1.0 / model_count)]
    starts.extend(np.eye(model_count))
    if model_names[:4] == ["log_s50", "sqrt_s50", "log_s10", "log_s20"] and model_count == 5:
        starts.append(np.asarray([0.175, 0.425, 0.2, 0.2, 0.0], dtype=float))
        starts.append(np.asarray([0.1575, 0.3825, 0.18, 0.18, 0.1], dtype=float))

    continuous_weights, continuous_mae = optimize_weights(pred_matrix, target, starts)
    rounded_weights = round_to_grid(continuous_weights, grid_step)
    if model_count <= 5:
        radius_steps = 5
    elif model_count <= 7:
        radius_steps = 3
    elif model_count <= 10:
        radius_steps = 2
    else:
        radius_steps = 0

    if radius_steps > 0:
        neighbor_grid = build_rounded_neighbor_grid(continuous_weights, grid_step, radius_steps=radius_steps)
    else:
        neighbor_grid = build_pairwise_neighbor_grid(continuous_weights, grid_step, max_transfer_steps=2)

    extras = [rounded_weights, continuous_weights]
    if model_count == 5 and model_names[:4] == ["log_s50", "sqrt_s50", "log_s10", "log_s20"]:
        extras.extend(
            [
                np.asarray([0.175, 0.425, 0.2, 0.2, 0.0], dtype=float),
                np.asarray([0.1575, 0.3825, 0.18, 0.18, 0.1], dtype=float),
            ]
        )
    extras.extend(np.eye(model_count))

    candidate_weights = np.vstack([neighbor_grid, np.asarray([normalize_weights(item) for item in extras])])
    candidate_weights = np.unique(np.round(candidate_weights, decimals=12), axis=0)
    scores = np.asarray([mae_for_weights(pred_matrix, target, weights) for weights in candidate_weights])
    order = np.argsort(scores, kind="mergesort")

    top_order = order[: min(top_k, len(order))]
    top_rows = pd.DataFrame(candidate_weights[top_order])
    top_rows["mae"] = scores[top_order]
    top_rows["search_note"] = "optimized_local"

    best_idx = int(order[0])
    best_weights = candidate_weights[best_idx]
    best_mae = float(scores[best_idx])
    if continuous_mae < best_mae:
        best_weights = continuous_weights
        best_mae = continuous_mae
        continuous_row = pd.DataFrame([list(continuous_weights) + [continuous_mae, "continuous_slsqp"]])
        continuous_row.columns = list(top_rows.columns)
        top_rows = pd.concat([continuous_row, top_rows], ignore_index=True).sort_values("mae").reset_index(drop=True)

    return best_weights, best_mae, top_rows.head(top_k).reset_index(drop=True)


def run_meta_cv(
    pred_matrix: np.ndarray,
    target: np.ndarray,
    weight_grid: np.ndarray | None,
    n_splits: int,
    random_state: int,
    model_names: list[str],
    grid_step: float,
    search_mode: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_pred = np.zeros(len(target), dtype=float)
    rows: list[dict[str, float | int]] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(pred_matrix), start=1):
        if search_mode == "grid":
            if weight_grid is None:
                raise ValueError("weight_grid is required for grid search.")
            best_weights, train_mae, _ = search_best_weights(
                pred_matrix=pred_matrix[train_idx],
                target=target[train_idx],
                weight_grid=weight_grid,
                top_k=1,
            )
        else:
            best_weights, train_mae, _ = search_best_weights_optimized(
                pred_matrix=pred_matrix[train_idx],
                target=target[train_idx],
                model_names=model_names,
                grid_step=grid_step,
                top_k=1,
            )
        valid_pred = (pred_matrix[valid_idx] @ best_weights).clip(min=0)
        cv_pred[valid_idx] = valid_pred
        valid_mae = float(np.mean(np.abs(target[valid_idx] - valid_pred)))
        row: dict[str, float | int] = {
            "meta_fold": fold_idx,
            "train_mae": train_mae,
            "valid_mae": valid_mae,
        }
        for name, weight in zip(model_names, best_weights, strict=True):
            row[f"w_{name}"] = float(weight)
        rows.append(row)

    return pd.DataFrame(rows), cv_pred


def run_multi_oof_blend_search(args: argparse.Namespace) -> dict[str, object]:
    named_paths = [parse_named_path(raw) for raw in args.model]
    model_names = [name for name, _ in named_paths]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Model names must be unique.")

    submission_paths = dict(parse_named_path(raw) for raw in args.submission)
    merged_oof, pred_matrix, target = load_oof_matrix(named_paths)
    denominator = int(round(1.0 / args.grid_step))
    grid_size = comb(denominator + len(named_paths) - 1, len(named_paths) - 1)
    weight_grid = build_weight_grid(len(named_paths), args.grid_step) if args.search_mode == "grid" else None

    if args.search_mode == "grid":
        if weight_grid is None:
            raise ValueError("weight_grid is required for grid search.")
        best_weights, best_mae, top_results = search_best_weights(
            pred_matrix=pred_matrix,
            target=target,
            weight_grid=weight_grid,
            top_k=args.top_k,
        )
        top_results.columns = [f"w_{name}" for name in model_names] + ["mae"]
    else:
        best_weights, best_mae, top_results = search_best_weights_optimized(
            pred_matrix=pred_matrix,
            target=target,
            model_names=model_names,
            grid_step=args.grid_step,
            top_k=args.top_k,
        )
        top_results.columns = [f"w_{name}" for name in model_names] + ["mae", "search_note"]

    best_pred = (pred_matrix @ best_weights).clip(min=0)
    meta_cv_choices, meta_cv_pred = run_meta_cv(
        pred_matrix=pred_matrix,
        target=target,
        weight_grid=weight_grid,
        n_splits=args.meta_cv_splits,
        random_state=args.random_state,
        model_names=model_names,
        grid_step=args.grid_step,
        search_mode=args.search_mode,
    )
    meta_cv_mae = float(np.mean(np.abs(target - meta_cv_pred)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_results.to_csv(args.output_dir / "blend_results.csv", index=False)
    meta_cv_choices.to_csv(args.output_dir / "blend_cv_choices.csv", index=False)

    oof_output = merged_oof[["SaleID", "price"]].copy()
    oof_output["oof_pred"] = best_pred
    for idx, name in enumerate(model_names):
        oof_output[f"{name}_pred"] = pred_matrix[:, idx]
    oof_output.to_csv(args.output_dir / "best_blend_oof_predictions.csv", index=False)

    cv_oof = merged_oof[["SaleID", "price"]].copy()
    cv_oof["oof_pred"] = meta_cv_pred
    cv_oof.to_csv(args.output_dir / "blend_cv_oof_predictions.csv", index=False)

    summary: dict[str, object] = {
        "n_rows": int(len(target)),
        "grid_step": float(args.grid_step),
        "grid_size": int(grid_size),
        "search_mode": args.search_mode,
        "base_oof_mae": {
            name: mae_for_weights(pred_matrix, target, np.eye(len(model_names))[idx])
            for idx, name in enumerate(model_names)
        },
        "best_full_mae": best_mae,
        "best_weights": {name: float(weight) for name, weight in zip(model_names, best_weights, strict=True)},
        "meta_cv_mae": meta_cv_mae,
        "meta_cv_valid_mean": float(meta_cv_choices["valid_mae"].mean()),
        "meta_cv_valid_std": float(meta_cv_choices["valid_mae"].std(ddof=1)),
        "source_files": {name: str(path) for name, path in named_paths},
    }

    submission_data = load_submission_matrix(named_paths, submission_paths)
    if submission_data is not None:
        submission_df, submission_matrix = submission_data
        blend_submission = (submission_matrix @ best_weights).clip(min=0)
        submission_out = submission_df[["SaleID"]].copy()
        submission_out["price"] = blend_submission
        submission_out.to_csv(args.output_dir / "submission.csv", index=False)

        component_out = submission_df.copy()
        component_out["price"] = blend_submission
        component_out.to_csv(args.output_dir / "submission_with_components.csv", index=False)

        summary["submission"] = {
            "rows": int(len(submission_out)),
            "price_min": float(submission_out["price"].min()),
            "price_median": float(submission_out["price"].median()),
            "price_max": float(submission_out["price"].max()),
            "source_files": {name: str(submission_paths[name]) for name in model_names},
        }

    (args.output_dir / "blend_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTop blend results:")
    print(top_results.head(20).to_string(index=False))
    print("\nMeta-CV choices:")
    print(meta_cv_choices.to_string(index=False))
    return summary


def main(argv: list[str] | None = None) -> None:
    run_multi_oof_blend_search(parse_args(argv))


if __name__ == "__main__":
    main()
