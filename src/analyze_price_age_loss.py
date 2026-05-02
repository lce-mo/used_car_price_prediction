from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TRAIN_PATH = Path("data/raw/used_car_train_20200313.csv")
DEFAULT_CURRENT_OOF_PATH = Path("outputs/final_model_te20_est2150_l287/oof_predictions.csv")
DEFAULT_PREVIOUS_OOF_PATH = Path("outputs/m2_lgbm_q3_testB_3fold/oof_predictions.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/price_age_error_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze price-by-age MAE contribution for two OOF runs.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--current-oof-path", type=Path, default=DEFAULT_CURRENT_OOF_PATH)
    parser.add_argument("--previous-oof-path", type=Path, default=DEFAULT_PREVIOUS_OOF_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--price-buckets",
        type=int,
        default=5,
        help="Number of quantile buckets for price.",
    )
    parser.add_argument(
        "--top-cells",
        type=int,
        default=3,
        help="Number of highest-contribution grid cells to highlight in the report.",
    )
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


def load_oof(path: Path, pred_col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"SaleID", "oof_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"OOF file {path} missing columns: {sorted(missing)}")
    return df[["SaleID", "oof_pred"]].rename(columns={"oof_pred": pred_col_name})


def build_base_dataframe(train_path: Path, current_oof_path: Path, previous_oof_path: Path, price_buckets: int) -> pd.DataFrame:
    train_df = pd.read_csv(train_path, sep=" ")
    current_oof = load_oof(current_oof_path, "current_oof_pred")
    previous_oof = load_oof(previous_oof_path, "previous_oof_pred")

    df = train_df.merge(current_oof, on="SaleID", how="inner").merge(previous_oof, on="SaleID", how="inner")

    reg_date = parse_compact_date(df["regDate"])
    create_date = parse_compact_date(df["creatDate"])
    car_age_days = (create_date - reg_date).dt.days
    car_age_years = car_age_days / 365.25

    df["price_bucket"] = build_price_bucket(df["price"], price_buckets)
    df["age_bucket"] = build_age_bucket(car_age_years)
    df["car_age_years"] = car_age_years
    df["current_abs_err"] = (df["price"] - df["current_oof_pred"]).abs()
    df["previous_abs_err"] = (df["price"] - df["previous_oof_pred"]).abs()
    return df


def build_grid_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_current_abs_err = float(df["current_abs_err"].sum())
    grouped = (
        df.groupby(["price_bucket", "age_bucket"], dropna=False)
        .agg(
            n=("SaleID", "size"),
            price_mean=("price", "mean"),
            price_median=("price", "median"),
            current_mae=("current_abs_err", "mean"),
            previous_mae=("previous_abs_err", "mean"),
            current_abs_err_sum=("current_abs_err", "sum"),
        )
        .reset_index()
    )
    grouped["share"] = grouped["n"] / len(df)
    grouped["mae_improvement"] = grouped["previous_mae"] - grouped["current_mae"]
    grouped["current_abs_err_share"] = grouped["current_abs_err_sum"] / total_current_abs_err
    grouped = grouped.drop(columns=["current_abs_err_sum"])
    return grouped.sort_values("current_abs_err_share", ascending=False).reset_index(drop=True)


def build_bucket_summary(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    total_current_abs_err = float(df["current_abs_err"].sum())
    grouped = (
        df.groupby(bucket_col, dropna=False)
        .agg(
            n=("SaleID", "size"),
            price_mean=("price", "mean"),
            price_median=("price", "median"),
            current_mae=("current_abs_err", "mean"),
            previous_mae=("previous_abs_err", "mean"),
            current_abs_err_sum=("current_abs_err", "sum"),
        )
        .reset_index()
        .rename(columns={bucket_col: "bucket"})
    )
    grouped["share"] = grouped["n"] / len(df)
    grouped["mae_improvement"] = grouped["previous_mae"] - grouped["current_mae"]
    grouped["current_abs_err_share"] = grouped["current_abs_err_sum"] / total_current_abs_err
    grouped = grouped.drop(columns=["current_abs_err_sum"])
    return grouped.sort_values("current_abs_err_share", ascending=False).reset_index(drop=True)


def render_report(
    grid_summary: pd.DataFrame,
    price_summary: pd.DataFrame,
    age_summary: pd.DataFrame,
    top_cells: int,
) -> str:
    top_grid = grid_summary.head(top_cells)
    top_lines = [
        f"- `{row.price_bucket} × {row.age_bucket}`：当前 MAE `{row.current_mae:.2f}`，贡献占比 `{row.current_abs_err_share:.2%}`，相对上一版本改善 `{row.mae_improvement:.2f}`"
        for row in top_grid.itertuples()
    ]

    first_cell = top_grid.iloc[0]
    still_high_price_new_car = bool(
        (first_cell["price_bucket"] == "Q5") and (first_cell["age_bucket"] in {"0_1y", "1_3y"})
    )

    if still_high_price_new_car:
        main_conclusion = "- 当前主损失来源仍然首先指向高价新车/准新车组合。"
        direction = "- 下一步特征应继续优先增强高价新车区分能力，而不是回到通用小特征。"
    else:
        main_conclusion = "- 当前主损失来源已不再单纯是高价新车，需要按具体格子重新定义主战场。"
        direction = "- 下一步特征应围绕贡献最高的具体价格×车龄格子来设计，而不是笼统按‘高价新车’推进。"

    lines = [
        "# 当前主版本价格×车龄误差分析",
        "",
        "## 核心结论",
        main_conclusion,
        "- 当前分析以主损失来源为主，上一正式版本 `W5` 只作为改善对照。",
        direction,
        "",
        "## 贡献最高的价格×车龄格子",
        *top_lines,
        "",
        "## 一维观察",
        f"- 价格分桶中贡献最高的是 `{price_summary.iloc[0]['bucket']}`，当前 MAE `{price_summary.iloc[0]['current_mae']:.2f}`。",
        f"- 车龄分桶中贡献最高的是 `{age_summary.iloc[0]['bucket']}`，当前 MAE `{age_summary.iloc[0]['current_mae']:.2f}`。",
        "",
        "## 文件说明",
        "- `price_age_grid.csv`: 价格×车龄二维误差表",
        "- `price_bucket_summary.csv`: 价格分桶一维汇总",
        "- `age_bucket_summary.csv`: 车龄分桶一维汇总",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    for path in [args.train_path, args.current_oof_path, args.previous_oof_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    df = build_base_dataframe(
        train_path=args.train_path,
        current_oof_path=args.current_oof_path,
        previous_oof_path=args.previous_oof_path,
        price_buckets=args.price_buckets,
    )

    grid_summary = build_grid_summary(df)
    price_summary = build_bucket_summary(df, "price_bucket")
    age_summary = build_bucket_summary(df, "age_bucket")
    report = render_report(grid_summary, price_summary, age_summary, top_cells=args.top_cells)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_summary.to_csv(args.output_dir / "price_age_grid.csv", index=False)
    price_summary.to_csv(args.output_dir / "price_bucket_summary.csv", index=False)
    age_summary.to_csv(args.output_dir / "age_bucket_summary.csv", index=False)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")

    sys.stdout.buffer.write(report.encode("utf-8"))


if __name__ == "__main__":
    main()
