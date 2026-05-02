from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


DEFAULT_TRAIN_PATH = Path("data/raw/used_car_train_20200313.csv")
DEFAULT_OOF_PATH = Path("outputs/test/full150000_modelte_s50_repeated/oof_predictions.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/full150000_modelte_s50_pred_age_calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate simple OOF residual calibration rules.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--oof-path", type=Path, default=DEFAULT_OOF_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pred-buckets", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--smoothing-grid",
        type=str,
        default="50,100,200,500,1000",
        help="Comma-separated shrinkage smoothing values for group residual medians.",
    )
    parser.add_argument("--cap-low", type=float, default=-800.0)
    parser.add_argument("--cap-high", type=float, default=1200.0)
    parser.add_argument(
        "--calibrate-pred-bucket-min",
        type=int,
        default=1,
        help="Only apply corrections to prediction buckets with index >= this value.",
    )
    parser.add_argument(
        "--calibrate-age-buckets",
        type=str,
        default="",
        help="Comma-separated age buckets to calibrate. Empty means all age buckets.",
    )
    return parser.parse_args()


def parse_float_grid(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("smoothing grid cannot be empty.")
    return [float(item) for item in values]


def parse_string_set(raw: str) -> set[str] | None:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def parse_compact_date(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def build_age_bucket(age_years: pd.Series) -> pd.Series:
    age_numeric = pd.to_numeric(age_years, errors="coerce")
    bucket = pd.cut(
        age_numeric,
        bins=[-np.inf, 1, 3, 5, 8, np.inf],
        labels=["0_1y", "1_3y", "3_5y", "5_8y", "8y_plus"],
    )
    return bucket.astype("string").fillna("age_missing")


def build_base_dataframe(train_path: Path, oof_path: Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_path, sep=" ")
    oof_df = pd.read_csv(oof_path)
    required = {"SaleID", "price", "oof_pred"}
    missing = required.difference(oof_df.columns)
    if missing:
        raise ValueError(f"OOF file missing columns: {sorted(missing)}")

    df = train_df.merge(oof_df[["SaleID", "oof_pred"]], on="SaleID", how="inner")
    if len(df) != len(train_df):
        raise ValueError(f"OOF row count mismatch: train={len(train_df)}, merged={len(df)}")

    reg_date = parse_compact_date(df["regDate"])
    create_date = parse_compact_date(df["creatDate"])
    car_age_years = (create_date - reg_date).dt.days / 365.25
    df["age_bucket"] = build_age_bucket(car_age_years)
    df["residual"] = df["price"] - df["oof_pred"]
    df["base_abs_err"] = df["residual"].abs()
    return df


def fit_pred_bucket_edges(pred: pd.Series, bucket_count: int) -> np.ndarray:
    _, edges = pd.qcut(pred, q=bucket_count, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float)
    if len(edges) < 2:
        raise ValueError("Cannot build prediction buckets from constant predictions.")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_pred_bucket(pred: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = [f"P{i}" for i in range(1, len(edges))]
    bucket = pd.cut(pred, bins=edges, labels=labels, include_lowest=True)
    return bucket.astype("string").fillna("pred_missing")


def pred_bucket_index(bucket: pd.Series) -> pd.Series:
    return bucket.astype("string").str.replace("P", "", regex=False).astype("Int64")


def compute_group_corrections(
    fit_df: pd.DataFrame,
    group_cols: list[str],
    smoothing: float,
    cap_low: float,
    cap_high: float,
) -> pd.Series:
    grouped = fit_df.groupby(group_cols, dropna=False)["residual"].agg(["median", "count"])
    shrink_weight = grouped["count"] / (grouped["count"] + smoothing)
    correction = (grouped["median"] * shrink_weight).clip(lower=cap_low, upper=cap_high)
    return correction


def apply_group_corrections(
    apply_df: pd.DataFrame,
    corrections: pd.Series,
    group_cols: list[str],
) -> np.ndarray:
    key_index = pd.MultiIndex.from_frame(apply_df[group_cols])
    return corrections.reindex(key_index).fillna(0.0).to_numpy(dtype=float)


def build_application_mask(
    apply_df: pd.DataFrame,
    calibrate_pred_bucket_min: int,
    calibrate_age_buckets: set[str] | None,
) -> np.ndarray:
    mask = pred_bucket_index(apply_df["pred_bucket"]) >= calibrate_pred_bucket_min
    if calibrate_age_buckets is not None:
        mask = mask & apply_df["age_bucket"].isin(calibrate_age_buckets)
    return mask.fillna(False).to_numpy(dtype=bool)


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(true_arr - pred_arr)))


def run_calibration_cv(
    df: pd.DataFrame,
    pred_buckets: int,
    smoothing: float,
    cap_low: float,
    cap_high: float,
    n_splits: int,
    random_state: int,
    calibrate_pred_bucket_min: int,
    calibrate_age_buckets: set[str] | None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    calibrated_pred = np.empty(len(df), dtype=float)
    applied_correction = np.empty(len(df), dtype=float)
    fold_id = np.empty(len(df), dtype=int)

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    group_cols = ["pred_bucket", "age_bucket"]

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(df), start=1):
        fit_df = df.iloc[fit_idx].copy()
        val_df = df.iloc[val_idx].copy()

        edges = fit_pred_bucket_edges(fit_df["oof_pred"], pred_buckets)
        fit_df["pred_bucket"] = apply_pred_bucket(fit_df["oof_pred"], edges)
        val_df["pred_bucket"] = apply_pred_bucket(val_df["oof_pred"], edges)

        corrections = compute_group_corrections(
            fit_df=fit_df,
            group_cols=group_cols,
            smoothing=smoothing,
            cap_low=cap_low,
            cap_high=cap_high,
        )
        correction_values = apply_group_corrections(val_df, corrections, group_cols)
        application_mask = build_application_mask(
            apply_df=val_df,
            calibrate_pred_bucket_min=calibrate_pred_bucket_min,
            calibrate_age_buckets=calibrate_age_buckets,
        )
        correction_values = np.where(application_mask, correction_values, 0.0)
        pred_values = (val_df["oof_pred"].to_numpy(dtype=float) + correction_values).clip(min=0)

        calibrated_pred[val_idx] = pred_values
        applied_correction[val_idx] = correction_values
        fold_id[val_idx] = fold

    result_df = df[["SaleID", "price", "oof_pred", "age_bucket", "residual", "base_abs_err"]].copy()
    result_df["calibrated_pred"] = calibrated_pred
    result_df["calibration_correction"] = applied_correction
    result_df["calibrated_abs_err"] = (result_df["price"] - result_df["calibrated_pred"]).abs()
    result_df["calibration_fold"] = fold_id

    summary = {
        "smoothing": float(smoothing),
        "cap_low": float(cap_low),
        "cap_high": float(cap_high),
        "n_splits": int(n_splits),
        "pred_buckets": int(pred_buckets),
        "calibrate_pred_bucket_min": int(calibrate_pred_bucket_min),
        "calibrate_age_buckets": ",".join(sorted(calibrate_age_buckets)) if calibrate_age_buckets else "ALL",
        "base_mae": float(result_df["base_abs_err"].mean()),
        "calibrated_mae": float(result_df["calibrated_abs_err"].mean()),
        "mae_delta": float(result_df["calibrated_abs_err"].mean() - result_df["base_abs_err"].mean()),
        "mean_abs_correction": float(np.mean(np.abs(applied_correction))),
        "max_abs_correction": float(np.max(np.abs(applied_correction))),
    }
    return result_df, summary


def summarize_by_pred_age(result_df: pd.DataFrame, pred_buckets: int) -> pd.DataFrame:
    # This table is diagnostic only; it uses full OOF prediction buckets for display.
    edges = fit_pred_bucket_edges(result_df["oof_pred"], pred_buckets)
    display_df = result_df.copy()
    display_df["pred_bucket"] = apply_pred_bucket(display_df["oof_pred"], edges)
    grouped = (
        display_df.groupby(["pred_bucket", "age_bucket"], dropna=False)
        .agg(
            n=("SaleID", "size"),
            price_mean=("price", "mean"),
            pred_mean=("oof_pred", "mean"),
            base_mae=("base_abs_err", "mean"),
            calibrated_mae=("calibrated_abs_err", "mean"),
            correction_median=("calibration_correction", "median"),
            correction_mean=("calibration_correction", "mean"),
        )
        .reset_index()
    )
    grouped["mae_delta"] = grouped["calibrated_mae"] - grouped["base_mae"]
    return grouped.sort_values(["mae_delta", "base_mae"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not args.train_path.exists():
        raise FileNotFoundError(args.train_path)
    if not args.oof_path.exists():
        raise FileNotFoundError(args.oof_path)

    smoothing_grid = parse_float_grid(args.smoothing_grid)
    calibrate_age_buckets = parse_string_set(args.calibrate_age_buckets)
    df = build_base_dataframe(args.train_path, args.oof_path)

    summaries: list[dict[str, float | int]] = []
    result_by_smoothing: dict[float, pd.DataFrame] = {}
    for smoothing in smoothing_grid:
        result_df, summary = run_calibration_cv(
            df=df,
            pred_buckets=args.pred_buckets,
            smoothing=smoothing,
            cap_low=args.cap_low,
            cap_high=args.cap_high,
            n_splits=args.n_splits,
            random_state=args.random_state,
            calibrate_pred_bucket_min=args.calibrate_pred_bucket_min,
            calibrate_age_buckets=calibrate_age_buckets,
        )
        summaries.append(summary)
        result_by_smoothing[smoothing] = result_df

    summary_df = pd.DataFrame(summaries).sort_values(["calibrated_mae", "smoothing"]).reset_index(drop=True)
    best_smoothing = float(summary_df.iloc[0]["smoothing"])
    best_result = result_by_smoothing[best_smoothing]
    slice_summary = summarize_by_pred_age(best_result, args.pred_buckets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    best_result.to_csv(args.output_dir / "calibrated_oof_predictions.csv", index=False)
    slice_summary.to_csv(args.output_dir / "pred_age_slice_summary.csv", index=False)
    (args.output_dir / "best_summary.json").write_text(
        json.dumps(summary_df.iloc[0].to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Calibration summary:")
    print(summary_df.to_string(index=False))
    print("\nBest pred_bucket x age_bucket slices by improvement:")
    print(slice_summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
