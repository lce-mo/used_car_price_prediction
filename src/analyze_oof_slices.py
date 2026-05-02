from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TRAIN_PATH = Path("data/raw/used_car_train_20200313.csv")
DEFAULT_OOF_PATH = Path("outputs/test/full150000_modelte_s50_repeated/oof_predictions.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/full150000_modelte_s50_repeated_error_slices")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OOF MAE slice diagnostics.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--oof-path", type=Path, default=DEFAULT_OOF_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--price-buckets", type=int, default=5)
    parser.add_argument("--top", type=int, default=15)
    return parser.parse_args()


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


def build_price_bucket(price: pd.Series, bucket_count: int) -> pd.Series:
    labels = [f"Q{i}" for i in range(1, bucket_count + 1)]
    bucket = pd.qcut(price, q=bucket_count, labels=labels, duplicates="drop")
    return bucket.astype("string")


def build_model_freq_bucket(model_count: pd.Series) -> pd.Series:
    count = pd.to_numeric(model_count, errors="coerce")
    bucket = pd.cut(
        count,
        bins=[-np.inf, 9, 19, 49, 99, np.inf],
        labels=["lt10", "10_19", "20_49", "50_99", "100_plus"],
    )
    return bucket.astype("string").fillna("model_missing")


def load_base_dataframe(train_path: Path, oof_path: Path, price_buckets: int) -> pd.DataFrame:
    train_df = pd.read_csv(train_path, sep=" ")
    oof_df = pd.read_csv(oof_path)
    required_oof = {"SaleID", "price", "oof_pred"}
    missing_oof = required_oof.difference(oof_df.columns)
    if missing_oof:
        raise ValueError(f"OOF file missing columns: {sorted(missing_oof)}")

    df = train_df.merge(oof_df[["SaleID", "oof_pred"]], on="SaleID", how="inner")
    if len(df) != len(train_df):
        raise ValueError(f"OOF row count mismatch: train={len(train_df)}, merged={len(df)}")

    reg_date = parse_compact_date(df["regDate"])
    create_date = parse_compact_date(df["creatDate"])
    car_age_years = (create_date - reg_date).dt.days / 365.25

    model_count = df["model"].map(df["model"].value_counts(dropna=False))
    df["car_age_years"] = car_age_years
    df["price_bucket"] = build_price_bucket(df["price"], price_buckets)
    df["age_bucket"] = build_age_bucket(car_age_years)
    df["model_count"] = model_count
    df["model_freq_bucket"] = build_model_freq_bucket(model_count)
    df["abs_err"] = (df["price"] - df["oof_pred"]).abs()
    df["signed_err"] = df["oof_pred"] - df["price"]
    df["under_pred"] = (df["oof_pred"] < df["price"]).astype("int8")
    return df


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    total_abs_err = float(df["abs_err"].sum())
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("SaleID", "size"),
            price_mean=("price", "mean"),
            price_median=("price", "median"),
            pred_mean=("oof_pred", "mean"),
            mae=("abs_err", "mean"),
            abs_err_sum=("abs_err", "sum"),
            signed_err_mean=("signed_err", "mean"),
            under_pred_rate=("under_pred", "mean"),
            model_count_median=("model_count", "median"),
        )
        .reset_index()
    )
    grouped["sample_share"] = grouped["n"] / len(df)
    grouped["abs_err_share"] = grouped["abs_err_sum"] / total_abs_err
    return grouped.sort_values(
        by=["abs_err_share", "mae", "n"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def write_report(
    output_dir: Path,
    overall: dict[str, float | int],
    price_summary: pd.DataFrame,
    age_summary: pd.DataFrame,
    model_freq_summary: pd.DataFrame,
    price_age_grid: pd.DataFrame,
    price_age_model_freq_grid: pd.DataFrame,
    top: int,
) -> None:
    report = {
        "overall": overall,
        "top_price_buckets": price_summary.head(top).to_dict(orient="records"),
        "top_age_buckets": age_summary.head(top).to_dict(orient="records"),
        "top_model_freq_buckets": model_freq_summary.head(top).to_dict(orient="records"),
        "top_price_age_cells": price_age_grid.head(top).to_dict(orient="records"),
        "top_price_age_model_freq_cells": price_age_model_freq_grid.head(top).to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# OOF Error Slice Report",
        "",
        f"- rows: {overall['rows']}",
        f"- oof_mae: {overall['oof_mae']:.6f}",
        f"- price_buckets: {overall['price_buckets']}",
        "",
        "## Top Price x Age Cells",
    ]
    for row in price_age_grid.head(top).itertuples(index=False):
        lines.append(
            f"- {row.price_bucket} x {row.age_bucket}: "
            f"n={row.n}, mae={row.mae:.2f}, abs_err_share={row.abs_err_share:.2%}, "
            f"under_pred_rate={row.under_pred_rate:.2%}"
        )
    lines.extend(["", "## Top Price x Age x Model Frequency Cells"])
    for row in price_age_model_freq_grid.head(top).itertuples(index=False):
        lines.append(
            f"- {row.price_bucket} x {row.age_bucket} x {row.model_freq_bucket}: "
            f"n={row.n}, mae={row.mae:.2f}, abs_err_share={row.abs_err_share:.2%}, "
            f"under_pred_rate={row.under_pred_rate:.2%}"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.train_path.exists():
        raise FileNotFoundError(args.train_path)
    if not args.oof_path.exists():
        raise FileNotFoundError(args.oof_path)

    df = load_base_dataframe(args.train_path, args.oof_path, args.price_buckets)
    overall = {
        "rows": int(len(df)),
        "oof_mae": float(df["abs_err"].mean()),
        "abs_err_sum": float(df["abs_err"].sum()),
        "price_buckets": int(args.price_buckets),
    }

    price_summary = summarize(df, ["price_bucket"])
    age_summary = summarize(df, ["age_bucket"])
    model_freq_summary = summarize(df, ["model_freq_bucket"])
    price_age_grid = summarize(df, ["price_bucket", "age_bucket"])
    price_model_freq_grid = summarize(df, ["price_bucket", "model_freq_bucket"])
    age_model_freq_grid = summarize(df, ["age_bucket", "model_freq_bucket"])
    price_age_model_freq_grid = summarize(df, ["price_bucket", "age_bucket", "model_freq_bucket"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    price_summary.to_csv(args.output_dir / "price_bucket_summary.csv", index=False)
    age_summary.to_csv(args.output_dir / "age_bucket_summary.csv", index=False)
    model_freq_summary.to_csv(args.output_dir / "model_freq_bucket_summary.csv", index=False)
    price_age_grid.to_csv(args.output_dir / "price_age_grid.csv", index=False)
    price_model_freq_grid.to_csv(args.output_dir / "price_model_freq_grid.csv", index=False)
    age_model_freq_grid.to_csv(args.output_dir / "age_model_freq_grid.csv", index=False)
    price_age_model_freq_grid.to_csv(args.output_dir / "price_age_model_freq_grid.csv", index=False)
    write_report(
        output_dir=args.output_dir,
        overall=overall,
        price_summary=price_summary,
        age_summary=age_summary,
        model_freq_summary=model_freq_summary,
        price_age_grid=price_age_grid,
        price_age_model_freq_grid=price_age_model_freq_grid,
        top=args.top,
    )

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("\nTop price x age cells:")
    print(price_age_grid.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
