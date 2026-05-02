from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search simple two-model OOF blends.")
    parser.add_argument("--base-oof-path", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in name=path format. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated base prediction weights.",
    )
    return parser.parse_args()


def parse_alpha_grid(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("alpha grid cannot be empty.")
    return values


def parse_candidate(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("candidate must use name=path format.")
    name, path = raw.split("=", maxsplit=1)
    name = name.strip()
    if not name:
        raise ValueError("candidate name cannot be empty.")
    return name, Path(path.strip())


def load_oof(path: Path, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"SaleID", "price", "oof_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["SaleID", "price", "oof_pred"]].rename(columns={"oof_pred": pred_col})


def mae(price: pd.Series, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(price.to_numpy(dtype=float) - pred)))


def main() -> None:
    args = parse_args()
    alpha_grid = parse_alpha_grid(args.alpha_grid)
    base = load_oof(args.base_oof_path, "base_pred")

    rows: list[dict[str, float | str]] = []
    best: dict[str, float | str] | None = None
    best_oof: pd.DataFrame | None = None

    for raw_candidate in args.candidate:
        candidate_name, candidate_path = parse_candidate(raw_candidate)
        candidate = load_oof(candidate_path, "candidate_pred")
        merged = base.merge(candidate[["SaleID", "candidate_pred"]], on="SaleID", how="inner")
        if len(merged) != len(base):
            raise ValueError(f"OOF row mismatch for {candidate_name}: {len(merged)} vs {len(base)}")

        base_pred = merged["base_pred"].to_numpy(dtype=float)
        candidate_pred = merged["candidate_pred"].to_numpy(dtype=float)
        for alpha in alpha_grid:
            blend_pred = (alpha * base_pred + (1.0 - alpha) * candidate_pred).clip(min=0)
            row = {
                "candidate": candidate_name,
                "alpha_base": float(alpha),
                "alpha_candidate": float(1.0 - alpha),
                "mae": mae(merged["price"], blend_pred),
            }
            rows.append(row)
            if best is None or float(row["mae"]) < float(best["mae"]):
                best = row
                best_oof = pd.DataFrame(
                    {
                        "SaleID": merged["SaleID"],
                        "price": merged["price"],
                        "oof_pred": blend_pred,
                        "base_pred": base_pred,
                        "candidate_pred": candidate_pred,
                    }
                )

    result_df = pd.DataFrame(rows).sort_values(["mae", "candidate", "alpha_base"]).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_dir / "blend_results.csv", index=False)
    if best_oof is not None:
        best_oof.to_csv(args.output_dir / "best_blend_oof_predictions.csv", index=False)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
