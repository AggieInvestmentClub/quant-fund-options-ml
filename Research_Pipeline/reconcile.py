"""Compare Ben's wide columns with the calculator's original per-feature CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PAPER_OPTIONS


def _keys(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"trade_date", "underlying"}
    if not required.issubset(frame):
        raise ValueError(f"{name}: missing {sorted(required - set(frame))}")
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame.trade_date).dt.normalize()
    if frame[["trade_date", "underlying"]].isna().any().any():
        raise ValueError(f"{name}: null key")
    if frame.duplicated(["trade_date", "underlying"]).any():
        raise ValueError(f"{name}: duplicate stock/date")
    return frame


def reconcile_features(wide_file: Path, features_dir: Path, *, month: str,
                       tolerance: float = 1e-8) -> dict:
    period = pd.Period(month, freq="M")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    wide = _keys(pd.read_parquet(wide_file, columns=["trade_date", "underlying", *PAPER_OPTIONS]), "wide")
    wide = wide[wide.trade_date.dt.to_period("M").eq(period)]
    if wide.empty:
        raise ValueError(f"wide file has no rows for {period}")
    report = {"month": str(period), "wide_rows": len(wide), "tolerance": tolerance,
              "features": {}, "all_match": True}
    for feature in PAPER_OPTIONS:
        source = Path(features_dir) / f"{feature}.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        raw = pd.read_csv(source, usecols=["trade_date", "underlying", feature])
        raw = _keys(raw, source.name)
        raw = raw[raw.trade_date.dt.to_period("M").eq(period)]
        joined = wide[["trade_date", "underlying", feature]].merge(
            raw, on=["trade_date", "underlying"], how="outer", validate="one_to_one",
            suffixes=("_wide", "_calculator"), indicator=True,
        )
        both = joined[joined._merge.eq("both")]
        a = pd.to_numeric(both[f"{feature}_wide"], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(both[f"{feature}_calculator"], errors="coerce").to_numpy(dtype=float)
        equal = np.isclose(a, b, rtol=0, atol=tolerance, equal_nan=True)
        finite = np.isfinite(a) & np.isfinite(b)
        gap = np.abs(a[finite] - b[finite])
        row = {
            "calculator_rows": len(raw), "matched_keys": len(both),
            "only_wide": int(joined._merge.eq("left_only").sum()),
            "only_calculator": int(joined._merge.eq("right_only").sum()),
            "value_mismatches": int((~equal).sum()),
            "max_finite_absolute_difference": float(gap.max()) if len(gap) else None,
        }
        row["matches"] = row["only_wide"] == row["only_calculator"] == row["value_mismatches"] == 0
        report["features"][feature] = row
        report["all_match"] = report["all_match"] and row["matches"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide-file", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile_features(args.wide_file, args.features_dir, month=args.month)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_match": result["all_match"], "month": result["month"]}))
    return 0 if result["all_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
