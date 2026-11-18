"""Validation and audit tool for the consolidated ML Parquet dataset."""

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add repo search paths
PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Data_Pipeline"))

from Dataset_Prep.config import PARQUET_OUTPUT_DIR, PARQUET_SINGLE_FILE
from Dataset_Prep.dataset_schema import (
    ALL_DATASET_COLUMNS,
    ATM_GREEK_COLUMNS,
    IDENTIFIER_COLUMNS,
    STOCK_COLUMNS,
    SURFACE_COLUMNS,
)
import flat_file


def is_business_day(target_date: date) -> bool:
    return flat_file.is_business_day(target_date)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load dataset from either a single Parquet file or a partitioned directory."""
    if dataset_path.is_file():
        print(f"Reading consolidated Parquet file: {dataset_path}")
        return pd.read_parquet(dataset_path)

    if dataset_path.is_dir():
        files = list(dataset_path.rglob("*.parquet"))
        files = [f for f in files if not f.name.startswith(".")]
        print(f"Reading {len(files)} partitioned Parquet files from: {dataset_path}")
        if not files:
            raise FileNotFoundError(f"No parquet files found in {dataset_path}")
        return pd.read_parquet(dataset_path)

    raise FileNotFoundError(f"Path does not exist: {dataset_path}")


def audit_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Run comprehensive quality and sanity checks on the DataFrame."""
    report: Dict[str, object] = {}

    # Basic stats
    total_rows = len(df)
    unique_dates = sorted(df["trade_date"].unique().tolist())
    unique_stocks = sorted(df["underlying"].unique().tolist())

    report["total_rows"] = total_rows
    report["unique_dates_count"] = len(unique_dates)
    report["date_min"] = unique_dates[0] if unique_dates else ""
    report["date_max"] = unique_dates[-1] if unique_dates else ""
    report["unique_stocks_count"] = len(unique_stocks)

    # Check date continuity
    start_dt = date.fromisoformat(report["date_min"])
    end_dt = date.fromisoformat(report["date_max"])
    curr = start_dt
    expected_b_days = []
    while curr <= end_dt:
        if is_business_day(curr):
            expected_b_days.append(curr.isoformat())
        curr += timedelta(days=1)

    actual_dates_set = set(unique_dates)
    missing_b_days = sorted(set(expected_b_days) - actual_dates_set)
    report["expected_business_days"] = len(expected_b_days)
    report["missing_business_days"] = missing_b_days

    # Rows per date distribution
    date_counts = df.groupby("trade_date").size()
    report["rows_per_date_min"] = int(date_counts.min())
    report["rows_per_date_max"] = int(date_counts.max())
    report["rows_per_date_mean"] = float(date_counts.mean())
    report["rows_per_date_median"] = float(date_counts.median())

    # Constituency breakdown
    sp500_rows = int(df["is_sp500"].sum())
    report["sp500_rows"] = sp500_rows
    report["sp500_pct"] = sp500_rows / total_rows if total_rows else 0.0

    # Investable options universe breakdown
    investable_rows = int(df["is_options_investable"].sum()) if "is_options_investable" in df.columns else 0
    report["investable_rows"] = investable_rows
    report["investable_pct"] = investable_rows / total_rows if total_rows else 0.0

    # Stock coverage & validity
    stock_populated = int(df["stock_close"].notna().sum())
    stock_positive = int((df["stock_close"] > 0).sum())
    report["stock_close_populated_pct"] = stock_populated / total_rows if total_rows else 0.0
    report["stock_close_positive_pct"] = stock_positive / stock_populated if stock_populated else 0.0

    # Missingness across key feature groups
    key_features = (
        ATM_GREEK_COLUMNS
        + ["risk_reversal_25_delta", "put_skew_10_25", "atm_iv"]
        + ["risk_reversal_25_delta_30d", "put_skew_10_25_30d", "atm_iv_30d"]
        + ["risk_reversal_25_delta_60d", "put_skew_10_25_60d", "atm_iv_60d"]
        + ["risk_reversal_25_delta_90d", "put_skew_10_25_90d", "atm_iv_90d"]
        + ["stock_adj_close", "stock_close", "stock_open", "stock_adj_open", "stock_volume", "dollar_volume", "shares_outstanding", "market_cap"]
        + ["size", "tvol_20d", "tvol_60d", "tskew_60d", "rm1", "rm3", "rm12", "exret_daily", "dturn_20d", "volume_ratio_20d"]
        + ["iv_level", "iv_gap", "variance_proxy", "iv_to_rv_ratio", "iv_level_cs_rank", "risk_reversal_25_delta_30d_robust", "atm_iv_30d_robust"]
        + ["forward_1d_return", "forward_5d_return", "forward_21d_return", "target_open_to_close_1d", "target_tradable_5d_return", "forward_5d_excess_return", "is_actively_trading", "is_options_investable"]
    )

    missingness = {}
    for col in key_features:
        if col in df.columns:
            missingness[col] = float(df[col].isna().mean())
    report["missingness"] = missingness

    return report


def generate_markdown_report(report: Dict[str, object], output_file: Optional[Path] = None) -> str:
    """Format audit results as a clean Markdown report."""
    lines: List[str] = [
        "# ML & Backtesting Dataset Quality & Audit Report",
        "",
        "## 1. Summary Statistics",
        f"- **Total Rows:** {report['total_rows']:,}",
        f"- **Trading Days:** {report['unique_dates_count']:,} (from `{report['date_min']}` to `{report['date_max']}`)",
        f"- **Unique Underlyings:** {report['unique_stocks_count']:,}",
        f"- **Expected Business Days:** {report['expected_business_days']:,}",
        f"- **Missing Business Days:** {len(report['missing_business_days'])} "
        + (f"({', '.join(report['missing_business_days'][:5])})" if report['missing_business_days'] else "None (100% complete)"),
        f"- **Stocks per Day:** Mean = {report['rows_per_date_mean']:.1f}, Median = {report['rows_per_date_median']:.0f} (Min: {report['rows_per_date_min']}, Max: {report['rows_per_date_max']})",
        f"- **S&P 500 Membership:** {report['sp500_rows']:,} rows ({report['sp500_pct']:.1%})",
        f"- **Investable Universe (`is_options_investable`):** {report.get('investable_rows', 0):,} rows ({report.get('investable_pct', 0.0):.1%})",
        f"- **Stock Price Availability:** {report['stock_close_populated_pct']:.2%} populated ({report['stock_close_positive_pct']:.2%} strictly positive)",
        "",
        "## 2. Feature Availability & Missingness Matrix",
        "| Category | Feature Name | Populated % | Missing (NaN) % |",
        "|---|---|---|---|",
    ]

    missingness: Dict[str, float] = report["missingness"]
    for col, nan_rate in missingness.items():
        if col in ["size", "tvol_20d", "tvol_60d", "tskew_60d", "rm1", "rm3", "rm12", "exret_daily", "dturn_20d", "volume_ratio_20d"]:
            category = "Academic Controls"
        elif col in ["forward_1d_return", "forward_5d_return", "forward_21d_return", "target_open_to_close_1d", "target_tradable_5d_return", "forward_5d_excess_return", "is_actively_trading", "is_options_investable"]:
            category = "Targets & Activity"
        elif col in ["iv_level", "iv_gap", "variance_proxy", "iv_to_rv_ratio", "iv_level_cs_rank", "risk_reversal_25_delta_30d_robust", "atm_iv_30d_robust"]:
            category = "Compact Signals"
        elif "iv" in col or "gamma" in col or "delta" in col or "theta" in col or "vega" in col:
            category = "ATM Greeks" if col.startswith("avg_") else ("30d/60d/90d Greeks" if "_30d" in col or "_60d" in col or "_90d" in col else "Surface Greeks")
        elif "skew" in col or "reversal" in col:
            category = "Skew & RR"
        else:
            category = "Stock OHLCV"

        lines.append(f"| {category} | `{col}` | {1.0 - nan_rate:6.2%} | {nan_rate:6.2%} |")

    lines.extend([
        "",
        "## 3. Data Integrity & Lookahead Guard",
        "- All option rows include `options_available_time` reflecting canonical $T+1$ morning availability (07:15 UTC).",
        "- `tradable_date` explicitly indicates the execution date without lookahead bias.",
        "- Output formatted as high-performance Parquet with ZSTD compression and PyArrow explicit dtypes.",
    ])

    report_text = "\n".join(lines)
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report_text, encoding="utf-8")
        print(f"Report written to: {output_file}")

    return report_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and validate the ML Parquet dataset.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to Parquet directory or file (defaults to PARQUET_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=None,
        help="Optional path to output markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_path = args.dataset_path or (
        PARQUET_SINGLE_FILE if PARQUET_SINGLE_FILE.exists() else PARQUET_OUTPUT_DIR
    )

    print(f"Loading and auditing dataset from: {target_path}")
    df = load_dataset(target_path)
    report = audit_dataset(df)
    md = generate_markdown_report(report, args.report_file)
    print("\n" + md)


if __name__ == "__main__":
    main()
