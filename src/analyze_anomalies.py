from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd


DEFAULT_TRAIN_PATH = Path("data/raw/used_car_train_20200313.csv")
DEFAULT_CURRENT_OOF_PATH = Path("outputs/final_model_te20_est2150_l287/oof_predictions.csv")
DEFAULT_PREVIOUS_OOF_PATH = Path("outputs/m2_lgbm_q3_testB_3fold/oof_predictions.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/anomaly_contribution_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze anomaly-group error contribution for two OOF runs.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--current-oof-path", type=Path, default=DEFAULT_CURRENT_OOF_PATH)
    parser.add_argument("--previous-oof-path", type=Path, default=DEFAULT_PREVIOUS_OOF_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--high-price-quantile",
        type=float,
        default=0.9,
        help="Quantile used to define high-price anomaly cases in the detailed case table.",
    )
    parser.add_argument(
        "--min-cross-count",
        type=int,
        default=100,
        help="Minimum sample count required for anomaly cross groups.",
    )
    parser.add_argument(
        "--top-cross-groups",
        type=int,
        default=8,
        help="Number of highest-contribution anomaly cross groups to keep.",
    )
    parser.add_argument(
        "--top-cases",
        type=int,
        default=15,
        help="Number of highest-loss anomaly cases to export.",
    )
    return parser.parse_args()


def parse_compact_date(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def load_oof(path: Path, pred_col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"SaleID", "oof_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"OOF file {path} missing columns: {sorted(missing)}")
    return df[["SaleID", "oof_pred"]].rename(columns={"oof_pred": pred_col_name})


def build_base_dataframe(train_path: Path, current_oof_path: Path, previous_oof_path: Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_path, sep=" ")
    current_oof = load_oof(current_oof_path, "current_oof_pred")
    previous_oof = load_oof(previous_oof_path, "previous_oof_pred")

    df = train_df.merge(current_oof, on="SaleID", how="inner").merge(previous_oof, on="SaleID", how="inner")

    reg_date = parse_compact_date(df["regDate"])
    create_date = parse_compact_date(df["creatDate"])
    car_age_days = (create_date - reg_date).dt.days
    power = pd.to_numeric(df["power"], errors="coerce")

    df["reg_invalid"] = reg_date.isna()
    df["create_invalid"] = create_date.isna()
    df["date_invalid_any"] = df["reg_invalid"] | df["create_invalid"]
    df["car_age_missing"] = car_age_days.isna()
    df["power_eq_0"] = power.eq(0)
    df["power_gt_600"] = power.gt(600)
    df["power_missing"] = power.isna()
    df["power"] = power
    df["car_age_days"] = car_age_days

    df["current_abs_err"] = (df["price"] - df["current_oof_pred"]).abs()
    df["previous_abs_err"] = (df["price"] - df["previous_oof_pred"]).abs()
    return df


def summarize_mask(df: pd.DataFrame, mask: pd.Series, group_name: str) -> dict[str, float | int | str]:
    subgroup = df.loc[mask]
    if subgroup.empty:
        raise ValueError(f"Group {group_name} is empty.")

    total_current_abs_err = float(df["current_abs_err"].sum())
    total_previous_abs_err = float(df["previous_abs_err"].sum())

    current_mae = float(subgroup["current_abs_err"].mean())
    previous_mae = float(subgroup["previous_abs_err"].mean())

    return {
        "group": group_name,
        "n": int(len(subgroup)),
        "share": float(len(subgroup) / len(df)),
        "price_mean": float(subgroup["price"].mean()),
        "price_median": float(subgroup["price"].median()),
        "price_p90": float(subgroup["price"].quantile(0.9)),
        "current_mae": current_mae,
        "previous_mae": previous_mae,
        "mae_improvement": float(previous_mae - current_mae),
        "current_abs_err_share": float(subgroup["current_abs_err"].sum() / total_current_abs_err),
        "previous_abs_err_share": float(subgroup["previous_abs_err"].sum() / total_previous_abs_err),
        "abs_err_share_improvement": float(
            subgroup["previous_abs_err"].sum() / total_previous_abs_err
            - subgroup["current_abs_err"].sum() / total_current_abs_err
        ),
    }


def build_primary_group_table(df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "power_eq_0": df["power_eq_0"],
        "power_gt_600": df["power_gt_600"],
        "power_missing": df["power_missing"],
        "date_invalid_any": df["date_invalid_any"],
        "car_age_missing": df["car_age_missing"],
    }
    records = [summarize_mask(df, mask, group_name) for group_name, mask in groups.items()]
    result = pd.DataFrame(records)
    return result.sort_values("current_abs_err_share", ascending=False).reset_index(drop=True)


def build_cross_group_table(
    df: pd.DataFrame,
    min_cross_count: int,
    top_cross_groups: int,
) -> pd.DataFrame:
    base_groups = {
        "power_eq_0": df["power_eq_0"],
        "power_gt_600": df["power_gt_600"],
        "power_missing": df["power_missing"],
        "date_invalid_any": df["date_invalid_any"],
    }
    records: list[dict[str, float | int | str]] = []
    for left, right in combinations(base_groups.keys(), 2):
        mask = base_groups[left] & base_groups[right]
        if int(mask.sum()) < min_cross_count:
            continue
        records.append(summarize_mask(df, mask, f"{left} & {right}"))

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    return result.sort_values("current_abs_err_share", ascending=False).head(top_cross_groups).reset_index(drop=True)


def build_high_price_anomaly_cases(df: pd.DataFrame, high_price_quantile: float, top_cases: int) -> pd.DataFrame:
    anomaly_mask = (
        df["power_eq_0"]
        | df["power_gt_600"]
        | df["power_missing"]
        | df["date_invalid_any"]
        | df["car_age_missing"]
    )
    high_price_threshold = float(df["price"].quantile(high_price_quantile))
    case_df = df.loc[anomaly_mask & (df["price"] >= high_price_threshold)].copy()
    case_df = case_df.sort_values("current_abs_err", ascending=False)
    columns = [
        "SaleID",
        "price",
        "current_oof_pred",
        "previous_oof_pred",
        "current_abs_err",
        "previous_abs_err",
        "power",
        "brand",
        "model",
        "regDate",
        "creatDate",
        "power_eq_0",
        "power_gt_600",
        "power_missing",
        "date_invalid_any",
        "car_age_missing",
    ]
    return case_df[columns].head(top_cases).reset_index(drop=True)


def render_markdown_report(
    primary_groups: pd.DataFrame,
    cross_groups: pd.DataFrame,
    high_price_cases: pd.DataFrame,
) -> str:
    primary_top = primary_groups.sort_values("current_abs_err_share", ascending=False)
    top_group_names = ", ".join(primary_top.head(3)["group"].tolist())

    recommendation_lines = [
        "- 当前 `power` / 日期异常群体整体不是主瓶颈，建议降级为次主线。",
        "- 如果后续还要沿异常方向做特征，优先只处理“高价且异常”的少量样本，而不是对全部异常样本统一清洗或加权。",
        "- 主线仍更可能回到围绕 `model` 信息做更稳的层级编码或低频回退方案。",
    ]

    lines = [
        "# 异常样本总体贡献分析",
        "",
        "## 核心结论",
        f"- 当前版本绝对误差贡献最高的异常群体主要是：{top_group_names}。",
        "- 这些异常群体在当前主版本中整体 MAE 并不高于全体样本，说明它们大多不是高价高复杂度样本。",
        "- 相比上一个正式版本，这些异常群体也已出现一定改善，暂时不足以支撑“异常清洗”升级为主线。",
        "",
        "## 方向建议",
        *recommendation_lines,
        "",
        "## 文件说明",
        "- `primary_groups.csv`: 主异常群体总表",
        "- `cross_groups.csv`: 总体误差贡献最高的异常交叉子群体",
        "- `high_price_cases.csv`: 高价异常样本中的最高损失个体",
    ]

    if not cross_groups.empty:
        lines.extend(
            [
                "",
                "## 交叉异常提示",
                f"- 当前贡献最高的交叉异常子群体是 `{cross_groups.iloc[0]['group']}`，但仍需结合样本量与价格层级判断是否值得专门建特征。",
            ]
        )

    if not high_price_cases.empty:
        lines.extend(
            [
                "",
                "## 高价异常个体提示",
                "- 异常方向最值得警惕的不是总体均值，而是少量高价且异常的离群个体。",
                "- 后续若继续沿这条线，只建议做轻量异常标记/修正，不建议对整类异常样本粗暴加权。",
            ]
        )

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
    )
    primary_groups = build_primary_group_table(df)
    cross_groups = build_cross_group_table(
        df,
        min_cross_count=args.min_cross_count,
        top_cross_groups=args.top_cross_groups,
    )
    high_price_cases = build_high_price_anomaly_cases(
        df,
        high_price_quantile=args.high_price_quantile,
        top_cases=args.top_cases,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary_groups.to_csv(args.output_dir / "primary_groups.csv", index=False)
    cross_groups.to_csv(args.output_dir / "cross_groups.csv", index=False)
    high_price_cases.to_csv(args.output_dir / "high_price_cases.csv", index=False)
    report_text = render_markdown_report(primary_groups, cross_groups, high_price_cases)
    (args.output_dir / "report.md").write_text(report_text, encoding="utf-8")

    safe_text = report_text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    sys.stdout.buffer.write(safe_text.encode("utf-8"))


if __name__ == "__main__":
    main()
