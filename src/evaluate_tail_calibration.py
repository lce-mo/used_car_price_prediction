from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


AGE_LABELS = ["0_1y", "1_3y", "3_5y", "5_8y", "8y_plus"]
PRED_LABELS = ["P1", "P2", "P3", "P4", "P5"]
POWER_LABELS = ["0", "1_60", "60_90", "90_120", "120_150", "150_200", "200_300", "300_600"]
TRUE_PRICE_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
TARGET5 = {
    ("Q5", "8y_plus"),
    ("Q4", "8y_plus"),
    ("Q5", "5_8y"),
    ("Q3", "8y_plus"),
    ("Q5", "3_5y"),
}


@dataclass(frozen=True)
class Candidate:
    experiment: str
    name: str
    params: dict[str, float | int | str]


def parse_compact_date(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def add_eval_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    reg_date = parse_compact_date(df["regDate"])
    create_date = parse_compact_date(df["creatDate"])
    age_years = (create_date - reg_date).dt.days / 365.25
    df["car_age_years"] = age_years
    df["age_bucket"] = pd.cut(
        age_years.clip(lower=0),
        bins=[-np.inf, 1, 3, 5, 8, np.inf],
        labels=AGE_LABELS,
    ).astype("string").fillna("age_missing")

    ranked_price = df["price"].rank(method="first")
    df["true_price_bucket"] = pd.qcut(ranked_price, q=5, labels=TRUE_PRICE_LABELS).astype("string")

    power = pd.to_numeric(df["power"], errors="coerce").clip(lower=0, upper=600)
    df["power_num"] = power
    df["power_bucket"] = pd.cut(
        power,
        bins=[-1, 0, 60, 90, 120, 150, 200, 300, 600],
        labels=POWER_LABELS,
    ).astype("string").fillna("power_missing")

    kilometer = pd.to_numeric(df["kilometer"], errors="coerce")
    df["kilometer_num"] = kilometer
    df["residual"] = df["price"] - df["oof_pred"]
    df["ratio"] = df["price"] / df["oof_pred"].clip(lower=1.0)
    return df


def pred_bucket_from_train(train_pred: pd.Series, apply_pred: pd.Series, bucket_count: int = 5) -> pd.Series:
    quantiles = [i / bucket_count for i in range(1, bucket_count)]
    thresholds = np.quantile(train_pred.to_numpy(dtype=float), quantiles)
    thresholds = np.maximum.accumulate(thresholds)
    bins = np.concatenate(([-np.inf], thresholds, [np.inf]))
    # If quantiles are duplicated, nudge them upward by a tiny amount to keep pd.cut valid.
    for idx in range(1, len(bins) - 1):
        if bins[idx] <= bins[idx - 1]:
            bins[idx] = bins[idx - 1] + 1e-9
    return pd.cut(apply_pred, bins=bins, labels=PRED_LABELS, include_lowest=True).astype("string")


def metrics_for_predictions(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    err = df[pred_col] - df["price"]
    abs_err = err.abs()
    result = {
        "mae": float(abs_err.mean()),
        "signed_err_mean": float(err.mean()),
    }
    for bucket in TRUE_PRICE_LABELS:
        mask = df["true_price_bucket"].eq(bucket)
        result[f"{bucket.lower()}_mae"] = float(abs_err[mask].mean())
    target_mask = pd.Series(
        list(zip(df["true_price_bucket"].astype(str), df["age_bucket"].astype(str))),
        index=df.index,
    ).isin(TARGET5)
    result["target5_mae"] = float(abs_err[target_mask].mean())
    result["target5_abs_err_share"] = float(abs_err[target_mask].sum() / abs_err.sum())
    result["target5_signed_err_mean"] = float(err[target_mask].mean())
    return result


def group_stat_map(
    train_df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    parent_cols: list[str],
    smoothing: float,
    statistic: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if statistic == "median":
        global_value = float(train_df[value_col].median())
        cell = train_df.groupby(group_cols, dropna=False)[value_col].agg(["median", "count"]).reset_index()
        cell = cell.rename(columns={"median": "cell_value", "count": "cell_count"})
        parent = train_df.groupby(parent_cols, dropna=False)[value_col].median().reset_index()
        parent = parent.rename(columns={value_col: "parent_value"})
    elif statistic == "ratio_median":
        global_value = float(train_df[value_col].median())
        cell = train_df.groupby(group_cols, dropna=False)[value_col].agg(["median", "count"]).reset_index()
        cell = cell.rename(columns={"median": "cell_value", "count": "cell_count"})
        parent = train_df.groupby(parent_cols, dropna=False)[value_col].median().reset_index()
        parent = parent.rename(columns={value_col: "parent_value"})
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")

    cell = cell.merge(parent, on=parent_cols, how="left")
    cell["parent_value"] = cell["parent_value"].fillna(global_value)
    if smoothing > 0:
        cell["smoothed_value"] = (
            cell["cell_count"] * cell["cell_value"] + smoothing * cell["parent_value"]
        ) / (cell["cell_count"] + smoothing)
    else:
        cell["smoothed_value"] = cell["cell_value"]
    cell = cell[group_cols + ["smoothed_value", "cell_count"]].drop_duplicates(group_cols)
    parent = parent.drop_duplicates(parent_cols)
    return cell, parent, global_value


def lookup_group_value(
    apply_df: pd.DataFrame,
    group_cols: list[str],
    parent_cols: list[str],
    cell_map: pd.DataFrame,
    parent_map: pd.DataFrame,
    global_value: float,
) -> pd.Series:
    left = apply_df[group_cols].reset_index(drop=True)
    merged = left.merge(cell_map, on=group_cols, how="left", sort=False)
    if len(merged) != len(apply_df):
        raise ValueError(
            f"Group lookup changed row count from {len(apply_df)} to {len(merged)} for {group_cols}."
        )
    values = merged["smoothed_value"]
    if values.isna().any():
        parent_left = apply_df[parent_cols].reset_index(drop=True)
        parent_merged = parent_left.merge(parent_map, on=parent_cols, how="left", sort=False)
        if len(parent_merged) != len(apply_df):
            raise ValueError(
                f"Parent lookup changed row count from {len(apply_df)} to {len(parent_merged)} for {parent_cols}."
            )
        parent_values = parent_merged["parent_value"]
        values = values.fillna(parent_values)
    return values.fillna(global_value).astype(float)


def apply_additive(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    group_cols: list[str],
    parent_cols: list[str],
    alpha: float,
    smoothing: float,
    correction_clip: float,
) -> pd.Series:
    cell_map, parent_map, global_value = group_stat_map(
        train_df=train_df,
        value_col="residual",
        group_cols=group_cols,
        parent_cols=parent_cols,
        smoothing=smoothing,
        statistic="median",
    )
    correction = lookup_group_value(apply_df, group_cols, parent_cols, cell_map, parent_map, global_value)
    correction = correction.clip(lower=-correction_clip, upper=correction_clip)
    calibrated = apply_df["oof_pred"].to_numpy(dtype=float) + alpha * correction.to_numpy(dtype=float)
    return pd.Series(np.clip(calibrated, a_min=0, a_max=None), index=apply_df.index)


def apply_ratio(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    group_cols: list[str],
    parent_cols: list[str],
    alpha: float,
    smoothing: float,
    ratio_min: float,
    ratio_max: float,
) -> pd.Series:
    cell_map, parent_map, global_value = group_stat_map(
        train_df=train_df,
        value_col="ratio",
        group_cols=group_cols,
        parent_cols=parent_cols,
        smoothing=smoothing,
        statistic="ratio_median",
    )
    ratio = lookup_group_value(apply_df, group_cols, parent_cols, cell_map, parent_map, global_value)
    ratio = ratio.clip(lower=ratio_min, upper=ratio_max)
    effective_ratio = 1.0 + alpha * (ratio - 1.0)
    calibrated = apply_df["oof_pred"].to_numpy(dtype=float) * effective_ratio.to_numpy(dtype=float)
    return pd.Series(np.clip(calibrated, a_min=0, a_max=None), index=apply_df.index)


def apply_selective_uplift(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    alpha: float,
    pred_quantile: float,
    age_min: float,
    power_min: float,
    kilometer_max: float | None,
    residual_floor: float,
    correction_clip: float,
) -> pd.Series:
    pred_threshold = float(train_df["oof_pred"].quantile(pred_quantile))
    train_mask = (
        train_df["oof_pred"].ge(pred_threshold)
        & train_df["car_age_years"].ge(age_min)
        & train_df["power_num"].ge(power_min)
    )
    apply_mask = (
        apply_df["oof_pred"].ge(pred_threshold)
        & apply_df["car_age_years"].ge(age_min)
        & apply_df["power_num"].ge(power_min)
    )
    if kilometer_max is not None:
        train_mask &= train_df["kilometer_num"].le(kilometer_max)
        apply_mask &= apply_df["kilometer_num"].le(kilometer_max)

    correction = float(train_df.loc[train_mask, "residual"].median()) if train_mask.any() else 0.0
    correction = max(residual_floor, correction)
    correction = min(correction, correction_clip)
    pred = apply_df["oof_pred"].copy()
    pred.loc[apply_mask] = (pred.loc[apply_mask] + alpha * correction).clip(lower=0)
    return pred


def make_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        for smoothing in [0, 100, 300, 500]:
            candidates.append(
                Candidate(
                    experiment="E007_pred_age_add",
                    name=f"pred_age_add_a{alpha:g}_s{smoothing}",
                    params={"alpha": alpha, "smoothing": smoothing, "correction_clip": 3000},
                )
            )
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        for smoothing in [100, 300, 500, 1000]:
            candidates.append(
                Candidate(
                    experiment="E008_pred_age_power_add",
                    name=f"pred_age_power_add_a{alpha:g}_s{smoothing}",
                    params={"alpha": alpha, "smoothing": smoothing, "correction_clip": 3000},
                )
            )
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        for smoothing in [100, 300, 500, 1000]:
            for ratio_max in [1.05, 1.08, 1.12]:
                candidates.append(
                    Candidate(
                        experiment="E009_ratio_age_power",
                        name=f"ratio_age_power_a{alpha:g}_s{smoothing}_max{ratio_max:g}",
                        params={
                            "alpha": alpha,
                            "smoothing": smoothing,
                            "ratio_min": 0.95,
                            "ratio_max": ratio_max,
                        },
                    )
                )
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        for pred_quantile in [0.75, 0.80, 0.85, 0.90]:
            for age_min in [5.0, 8.0]:
                for power_min in [120.0, 150.0, 200.0]:
                    candidates.append(
                        Candidate(
                            experiment="E010_selective_uplift",
                            name=f"uplift_a{alpha:g}_p{pred_quantile:g}_age{age_min:g}_pow{power_min:g}",
                            params={
                                "alpha": alpha,
                                "pred_quantile": pred_quantile,
                                "age_min": age_min,
                                "power_min": power_min,
                                "kilometer_max": "none",
                                "residual_floor": 0,
                                "correction_clip": 3000,
                            },
                        )
                    )
    return candidates


def apply_candidate(train_df: pd.DataFrame, apply_df: pd.DataFrame, candidate: Candidate) -> pd.Series:
    params = candidate.params
    if candidate.experiment == "E007_pred_age_add":
        return apply_additive(
            train_df=train_df,
            apply_df=apply_df,
            group_cols=["pred_bucket", "age_bucket"],
            parent_cols=["pred_bucket"],
            alpha=float(params["alpha"]),
            smoothing=float(params["smoothing"]),
            correction_clip=float(params["correction_clip"]),
        )
    if candidate.experiment == "E008_pred_age_power_add":
        return apply_additive(
            train_df=train_df,
            apply_df=apply_df,
            group_cols=["pred_bucket", "age_bucket", "power_bucket"],
            parent_cols=["pred_bucket", "age_bucket"],
            alpha=float(params["alpha"]),
            smoothing=float(params["smoothing"]),
            correction_clip=float(params["correction_clip"]),
        )
    if candidate.experiment == "E009_ratio_age_power":
        return apply_ratio(
            train_df=train_df,
            apply_df=apply_df,
            group_cols=["pred_bucket", "age_bucket", "power_bucket"],
            parent_cols=["pred_bucket", "age_bucket"],
            alpha=float(params["alpha"]),
            smoothing=float(params["smoothing"]),
            ratio_min=float(params["ratio_min"]),
            ratio_max=float(params["ratio_max"]),
        )
    if candidate.experiment == "E010_selective_uplift":
        kilometer_raw = params["kilometer_max"]
        kilometer_max = None if kilometer_raw == "none" else float(kilometer_raw)
        return apply_selective_uplift(
            train_df=train_df,
            apply_df=apply_df,
            alpha=float(params["alpha"]),
            pred_quantile=float(params["pred_quantile"]),
            age_min=float(params["age_min"]),
            power_min=float(params["power_min"]),
            kilometer_max=kilometer_max,
            residual_floor=float(params["residual_floor"]),
            correction_clip=float(params["correction_clip"]),
        )
    raise ValueError(f"Unsupported experiment: {candidate.experiment}")


def evaluate_candidate_cv(df: pd.DataFrame, candidate: Candidate, n_splits: int, random_state: int) -> tuple[pd.Series, list[dict[str, float | int]]]:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred = pd.Series(np.nan, index=df.index, dtype=float)
    fold_rows: list[dict[str, float | int]] = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(df), start=1):
        train_fold = df.iloc[train_idx].copy()
        valid_fold = df.iloc[valid_idx].copy()
        train_fold["pred_bucket"] = pred_bucket_from_train(train_fold["oof_pred"], train_fold["oof_pred"])
        valid_fold["pred_bucket"] = pred_bucket_from_train(train_fold["oof_pred"], valid_fold["oof_pred"])
        valid_pred = apply_candidate(train_fold, valid_fold, candidate)
        if len(valid_pred) != len(valid_idx):
            raise ValueError(
                f"Candidate {candidate.name} returned {len(valid_pred)} predictions for {len(valid_idx)} rows."
            )
        pred.iloc[valid_idx] = valid_pred.to_numpy()
        fold_metric = metrics_for_predictions(valid_fold.assign(calibrated_pred=valid_pred), "calibrated_pred")
        fold_rows.append({"fold": fold, **fold_metric})
    return pred, fold_rows


def evaluate_all(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_path, sep=" ")
    oof = pd.read_csv(args.oof_path)
    df = train.merge(oof[["SaleID", "price", "oof_pred"]], on="SaleID", how="inner", suffixes=("_raw", ""))
    if len(df) != len(train):
        raise ValueError(f"Expected {len(train)} merged rows, got {len(df)}")
    if "price_raw" in df.columns:
        mismatch = (pd.to_numeric(df["price_raw"], errors="coerce") != pd.to_numeric(df["price"], errors="coerce")).sum()
        if mismatch:
            raise ValueError(f"OOF price mismatches raw price for {mismatch} rows.")
        df = df.drop(columns=["price_raw"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["oof_pred"] = pd.to_numeric(df["oof_pred"], errors="coerce")
    df = add_eval_features(df)

    base_metrics = metrics_for_predictions(df, "oof_pred")
    candidates = make_candidates()
    rows = []
    best_by_experiment: dict[str, tuple[Candidate, pd.Series, dict[str, float]]] = {}
    for candidate in candidates:
        cv_pred, fold_rows = evaluate_candidate_cv(df, candidate, args.meta_folds, args.random_state)
        eval_df = df.assign(calibrated_pred=cv_pred)
        metrics = metrics_for_predictions(eval_df, "calibrated_pred")
        row = {
            "experiment": candidate.experiment,
            "candidate": candidate.name,
            **candidate.params,
            **{f"cv_{key}": value for key, value in metrics.items()},
            "cv_delta_mae": metrics["mae"] - base_metrics["mae"],
            "cv_delta_q5_mae": metrics["q5_mae"] - base_metrics["q5_mae"],
            "cv_delta_target5_mae": metrics["target5_mae"] - base_metrics["target5_mae"],
            "fold_mae_std": float(np.std([item["mae"] for item in fold_rows])),
        }
        rows.append(row)
        current_best = best_by_experiment.get(candidate.experiment)
        if current_best is None or metrics["mae"] < current_best[2]["mae"]:
            best_by_experiment[candidate.experiment] = (candidate, cv_pred, metrics)

    results = pd.DataFrame(rows).sort_values(["cv_mae", "cv_target5_mae"]).reset_index(drop=True)
    results.to_csv(output_dir / "calibration_grid_results.csv", index=False)

    best_rows = []
    for experiment, (candidate, cv_pred, metrics) in best_by_experiment.items():
        eval_df = df.assign(calibrated_pred=cv_pred)
        eval_df[["SaleID", "price", "oof_pred", "calibrated_pred"]].to_csv(
            output_dir / f"{experiment}_best_cv_oof.csv",
            index=False,
        )
        best_rows.append(
            {
                "experiment": experiment,
                "candidate": candidate.name,
                **candidate.params,
                **{f"cv_{key}": value for key, value in metrics.items()},
                "cv_delta_mae": metrics["mae"] - base_metrics["mae"],
                "cv_delta_q5_mae": metrics["q5_mae"] - base_metrics["q5_mae"],
                "cv_delta_target5_mae": metrics["target5_mae"] - base_metrics["target5_mae"],
            }
        )
    best_df = pd.DataFrame(best_rows).sort_values("cv_mae")
    best_df.to_csv(output_dir / "best_by_experiment.csv", index=False)

    best_candidate_name = str(best_df.iloc[0]["candidate"])
    best_experiment = str(best_df.iloc[0]["experiment"])
    best_candidate, best_pred, best_metrics = best_by_experiment[best_experiment]
    df.assign(calibrated_pred=best_pred)[["SaleID", "price", "oof_pred", "calibrated_pred"]].to_csv(
        output_dir / "best_overall_cv_oof.csv",
        index=False,
    )

    summary = {
        "rows": int(len(df)),
        "base_metrics": base_metrics,
        "best_overall_experiment": best_experiment,
        "best_overall_candidate": best_candidate_name,
        "best_overall_params": best_candidate.params,
        "best_overall_metrics": best_metrics,
        "best_overall_delta_mae": best_metrics["mae"] - base_metrics["mae"],
        "best_by_experiment": best_df.to_dict(orient="records"),
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nBest by experiment:")
    print(best_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OOF tail calibration rules.")
    parser.add_argument("--train-path", type=Path, default=Path("data/raw/used_car_train_20200313.csv"))
    parser.add_argument(
        "--oof-path",
        type=Path,
        default=Path("outputs/correct_full150000_four_model_blend_search/best_blend_oof_predictions.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/correct_tail_calibration_stage1"))
    parser.add_argument("--meta-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=2024)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_all(parse_args())
