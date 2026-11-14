"""Export consolidated daily Parquet dataset for ML modeling and backtesting.

Merges:
1. Stage 1 ATM Greek averages (10 features)
2. Stage 2 Default Strike delta buckets, skew, and risk-reversals (30d, 60d, 90d + surface)
3. Underlying stock history (tradable OHLCV, adjusted close, volume, dividend yield)
4. S&P 500 point-in-time constituency flag (is_sp500)
5. Availability metadata (options_available_time, tradable_date)
"""

import argparse
import concurrent.futures
from datetime import date, datetime, timedelta
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add repo search paths
PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Default_Strike_Pipeline"))
sys.path.insert(0, str(PROJECT_ROOT / "Data_Pipeline"))

from Dataset_Prep.config import (
    ATM_FEATURES_ROOT,
    CLEAN_STOCK_ROOT,
    CONSTITUENCY_ROOT,
    DEFAULT_STRIKE_FEATURES_ROOT,
    FLATFILES_ROOT,
    PARQUET_OUTPUT_DIR,
    PARQUET_SINGLE_FILE,
    RISK_FREE_RATE_CSV,
)
from constituency import active_symbols_on_date, load_constituency_snapshots, Snapshot
from Dataset_Prep.dataset_schema import (
    ALL_DATASET_COLUMNS,
    cast_and_align_frame,
    get_pyarrow_schema,
)
from Dataset_Prep.enrich_features import (
    enrich_stock_history_time_series,
    enrich_single_day_option_features,
    enrich_full_table_option_lags,
)
import flat_file


def is_business_day(target_date: date) -> bool:
    return flat_file.is_business_day(target_date)


def next_business_day(target_date: date) -> date:
    candidate = target_date + timedelta(days=1)
    while not is_business_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def build_atm_file_index(atm_root: Path) -> Dict[str, Path]:
    """Index all ATM feature CSVs by trade_date string (YYYY-MM-DD)."""
    index: Dict[str, Path] = {}
    for p in sorted(atm_root.rglob("*.csv")):
        if p.name != "underlying_close_cache.csv" and p.name.count("-") == 2:
            d = p.stem
            if d not in index or p.stat().st_mtime >= index[d].stat().st_mtime:
                index[d] = p
    return index


def build_ds_file_index(ds_root: Path) -> Dict[str, Path]:
    """Index all Default Strike feature CSVs by trade_date string (YYYY-MM-DD)."""
    index: Dict[str, Path] = {}
    for p in sorted(ds_root.rglob("*.csv")):
        if p.name.count("-") == 2:
            index[p.stem] = p
    return index


def load_all_stocks(stock_root: Path) -> pd.DataFrame:
    """Pre-load all stock daily histories into a single indexed DataFrame for fast joins."""
    frames: List[pd.DataFrame] = []
    stock_files = list(stock_root.glob("*_stock_data.csv"))
    if not stock_files:
        print(f"Warning: No stock files found under {stock_root}")
        return pd.DataFrame()

    for path in stock_files:
        ticker = path.name.replace("_stock_data.csv", "")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: Failed to read {path.name}: {e}")
            continue

        if "Date" not in df.columns or df.empty:
            print(f"Warning: {path.name} is empty or missing 'Date' column")
            continue

        rename_map = {
            "Date": "trade_date",
            "Open": "stock_open",
            "High": "stock_high",
            "Low": "stock_low",
            "Close": "stock_close",
            "Adj Close": "stock_adj_close",
            "Volume": "stock_volume",
            "dividend_yield": "dividend_yield",
            "stock_beta": "stock_beta",
            "historical_beta": "historical_beta",
            "shares_outstanding": "shares_outstanding",
            "market_cap": "market_cap",
        }
        # Exclude any partially written shares/market_cap from CSVs to ensure clean join
        keep_cols = [c for c in rename_map.keys() if c in df.columns and c not in ["shares_outstanding", "market_cap"]]
        sub = df[keep_cols].rename(columns=rename_map).copy()
        sub["underlying"] = ticker
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["trade_date"] = combined["trade_date"].astype(str)
    combined["underlying"] = combined["underlying"].astype(str)

    # Two-way coalesce stock_beta and historical_beta across stock files
    if "stock_beta" in combined.columns and "historical_beta" in combined.columns:
        combined["stock_beta"] = combined["stock_beta"].fillna(combined["historical_beta"])
        combined["historical_beta"] = combined["historical_beta"].fillna(combined["stock_beta"])
    elif "historical_beta" in combined.columns:
        combined["stock_beta"] = combined["historical_beta"]
    elif "stock_beta" in combined.columns:
        combined["historical_beta"] = combined["stock_beta"]

    # Always merge canonical shares_outstanding.parquet lookup
    shares_parquet = stock_root / "shares_outstanding.parquet"
    if shares_parquet.exists():
        try:
            shares_df = pd.read_parquet(shares_parquet)
            shares_df["trade_date"] = shares_df["trade_date"].astype(str)
            shares_df["underlying"] = shares_df["underlying"].astype(str)
            join_cols = [c for c in ["shares_outstanding", "market_cap"] if c in shares_df.columns]
            combined = pd.merge(combined, shares_df[["trade_date", "underlying"] + join_cols], on=["trade_date", "underlying"], how="left")
        except Exception as e:
            print(f"Warning: could not merge shares_outstanding.parquet: {e}")

    # Ensure shares_outstanding is forward-filled per ticker for newer dates
    if "shares_outstanding" in combined.columns:
        combined = combined.sort_values(["underlying", "trade_date"]).reset_index(drop=True)
        combined["shares_outstanding"] = combined.groupby("underlying")["shares_outstanding"].ffill()

    # Ensure market_cap is populated from stock_close * shares_outstanding
    if "shares_outstanding" in combined.columns:
        combined["market_cap"] = combined["stock_close"] * combined["shares_outstanding"]

    # Enrich stock history with academic controls (SIZE, TVOL, TSKEW, RM, EXRET, DTURN, forward returns, labels)
    print("Enriching stock history with academic controls and forward returns...")
    combined = enrich_stock_history_time_series(combined, RISK_FREE_RATE_CSV)

    return combined


def resolve_options_availability(
    trade_date: date,
    flatfiles_root: Path,
) -> Tuple[str, str]:
    """Calculate when options data for trade_date became available without lookahead."""
    nxt_bday = next_business_day(trade_date)
    tradable_date_str = nxt_bday.isoformat()

    # Default canonical availability time: 07:15 UTC on T+1
    canonical_iso = f"{nxt_bday.isoformat()}T07:15:00Z"

    # Check if raw flatfile exists on disk to get actual mtime if sensible
    flatfile_path = flatfiles_root / f"{trade_date:%Y}" / f"{trade_date:%m}" / f"{trade_date:%Y-%m-%d}.csv.gz"
    if flatfile_path.exists():
        try:
            mtime = datetime.fromtimestamp(flatfile_path.stat().st_mtime)
            # If mtime date matches nxt_bday (daily cron), use exact timestamp
            if mtime.date() == nxt_bday:
                return mtime.strftime("%Y-%m-%dT%H:%M:%SZ"), tradable_date_str
        except OSError:
            pass

    return canonical_iso, tradable_date_str


def build_single_day_dataset(
    trade_date_str: str,
    ds_path: Path,
    atm_path: Optional[Path],
    all_stocks: pd.DataFrame,
    snapshots: List[Snapshot],
    flatfiles_root: Path,
) -> pd.DataFrame:
    """Build the unified stock/date feature frame for a single trade date."""
    trade_date = date.fromisoformat(trade_date_str)

    # 1. Load Default Strike features (Stage 2)
    ds_df = pd.read_csv(ds_path)
    ds_df["trade_date"] = trade_date_str
    ds_df["underlying"] = ds_df["underlying"].astype(str).str.strip().str.upper()

    # 2. Load ATM features (Stage 1) if available
    if atm_path is not None and atm_path.exists():
        atm_df = pd.read_csv(atm_path)
        atm_df["trade_date"] = trade_date_str
        atm_df["underlying"] = atm_df["underlying"].astype(str).str.strip().str.upper()
        # Merge options features
        options_merged = pd.merge(ds_df, atm_df, on=["trade_date", "underlying"], how="outer")
    else:
        options_merged = ds_df

    # 3. Add point-in-time S&P 500 constituency flag
    active_sp = active_symbols_on_date(snapshots, trade_date)
    options_merged["is_sp500"] = options_merged["underlying"].isin(active_sp)

    # 4. Attach stock history for this date
    if not all_stocks.empty:
        day_stocks = all_stocks[all_stocks["trade_date"] == trade_date_str]
        merged = pd.merge(options_merged, day_stocks, on=["trade_date", "underlying"], how="left")
    else:
        merged = options_merged

    # 5. Enrich single-day option features (IV Level, IV Gap, Variance Proxy, robust coalescing)
    merged = enrich_single_day_option_features(merged)

    # 6. Attach options availability metadata
    avail_time, tradable_date = resolve_options_availability(trade_date, flatfiles_root)
    merged["options_available_time"] = avail_time
    merged["tradable_date"] = tradable_date

    # 7. Cast and align with strict schema
    aligned = cast_and_align_frame(merged)
    return aligned


def export_partitioned_parquet_date(
    aligned_df: pd.DataFrame,
    trade_date_str: str,
    output_dir: Path,
    schema: pa.Schema,
) -> Optional[Path]:
    """Atomically write a single trade date Parquet partition."""
    trade_date = date.fromisoformat(trade_date_str)
    partition_dir = output_dir / f"year={trade_date.year}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    dest_path = partition_dir / f"{trade_date_str}.parquet"
    if aligned_df.empty:
        print(f"  WARNING: {trade_date_str} has 0 rows. Skipping empty Parquet export.")
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        return None

    tmp_path = partition_dir / f".{trade_date_str}.parquet.tmp"

    table = pa.Table.from_pandas(aligned_df, schema=schema, preserve_index=False)
    pq.write_table(table, tmp_path, compression="zstd")
    tmp_path.replace(dest_path)
    return dest_path


def export_consolidated_parquet(
    output_dir: Path,
    single_file_path: Path,
    schema: pa.Schema,
    update_partitions: bool = True,
) -> Path:
    """Read all partitioned parquet files, enrich multi-day rolling lags, and write consolidated Parquet."""
    parquet_files = sorted(output_dir.rglob("*.parquet"))
    parquet_files = [f for f in parquet_files if not f.name.startswith(".")]

    # Discard any empty 0-row partition files
    valid_files = []
    for f in parquet_files:
        try:
            meta = pq.read_metadata(f)
            if meta.num_rows > 0:
                valid_files.append(f)
            else:
                print(f"Removing empty 0-row Parquet partition: {f.name}")
                f.unlink(missing_ok=True)
        except Exception:
            valid_files.append(f)
    parquet_files = valid_files

    if not parquet_files:
        raise ValueError(f"No valid parquet partition files found in {output_dir}")

    tmp_path = single_file_path.with_name(f".{single_file_path.name}.tmp")
    single_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Read and concatenate tables
    tables = [pq.read_table(f, schema=schema) for f in parquet_files]
    combined_table = pa.concat_tables(tables)

    # Enrich full-table option lags (delta_5d_iv_gap, delta_20d_iv_gap, delta_5d_rr_60d)
    combined_df = combined_table.to_pandas()
    combined_df = enrich_full_table_option_lags(combined_df)
    combined_df = cast_and_align_frame(combined_df)

    final_table = pa.Table.from_pandas(combined_df, schema=schema, preserve_index=False)
    pq.write_table(final_table, tmp_path, compression="zstd")
    tmp_path.replace(single_file_path)

    # Update individual partition files so partitioned queries also include enriched lag features
    if update_partitions:
        print(f"Updating {len(parquet_files)} partitioned Parquet files with enriched lags...")
        for trade_date_str, group_df in combined_df.groupby("trade_date"):
            trade_date = date.fromisoformat(str(trade_date_str))
            partition_dir = output_dir / f"year={trade_date.year}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            dest_path = partition_dir / f"{trade_date_str}.parquet"
            tmp_part = partition_dir / f".{trade_date_str}.parquet.tmp"
            part_table = pa.Table.from_pandas(group_df, schema=schema, preserve_index=False)
            pq.write_table(part_table, tmp_part, compression="zstd")
            tmp_part.replace(dest_path)

    return single_file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build consolidated daily Parquet dataset for ML and backtesting."
    )
    parser.add_argument(
        "--target-date",
        type=date.fromisoformat,
        default=None,
        help="Optional single trade date to process (YYYY-MM-DD), ideal for daily cron.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=None,
        help="Optional inclusive start date filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="Optional inclusive end date filter (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PARQUET_OUTPUT_DIR,
        help=f"Root directory for partitioned Parquet files (default: {PARQUET_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--single-file",
        type=Path,
        default=PARQUET_SINGLE_FILE,
        help=f"Destination for consolidated Parquet table (default: {PARQUET_SINGLE_FILE}).",
    )
    parser.add_argument(
        "--build-single-file",
        action="store_true",
        help="Also build/update the consolidated single-file Parquet table after partitioning.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force rebuild of existing Parquet partitions.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Worker processes for date-level parallelism.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.monotonic()

    print("=" * 60)
    print("Stage 3: Export Consolidated Daily Parquet Dataset")
    print(f"Output Directory:    {args.output_dir}")
    print(f"ATM Features:        {ATM_FEATURES_ROOT}")
    print(f"Default Strike:      {DEFAULT_STRIKE_FEATURES_ROOT}")
    print(f"Clean Stocks:        {CLEAN_STOCK_ROOT}")
    print(f"Constituency:        {CONSTITUENCY_ROOT}")
    print(f"Workers:             {args.workers}")
    print("=" * 60)

    # 1. Index source files
    print("\n[1/4] Indexing source files...")
    ds_index = build_ds_file_index(DEFAULT_STRIKE_FEATURES_ROOT)
    atm_index = build_atm_file_index(ATM_FEATURES_ROOT)
    print(f"Found {len(ds_index)} Default Strike date files.")
    print(f"Found {len(atm_index)} ATM date files.")

    # 2. Filter dates
    all_dates = sorted(ds_index.keys())
    if args.target_date is not None:
        target_str = args.target_date.isoformat()
        if target_str not in ds_index:
            print(f"Target date {target_str} not found in Default Strike features. Exiting.")
            sys.exit(0)
        selected_dates = [target_str]
    else:
        selected_dates = [
            d
            for d in all_dates
            if (args.start_date is None or date.fromisoformat(d) >= args.start_date)
            and (args.end_date is None or date.fromisoformat(d) <= args.end_date)
        ]

    if not args.overwrite:
        unprocessed = []
        for d in selected_dates:
            dt = date.fromisoformat(d)
            dest = args.output_dir / f"year={dt.year}" / f"{d}.parquet"
            if not dest.exists():
                unprocessed.append(d)
        print(f"Selected {len(selected_dates)} dates ({len(unprocessed)} need processing).")
        selected_dates = unprocessed
    else:
        print(f"Selected {len(selected_dates)} dates (overwrite=True).")

    if not selected_dates:
        print("All selected dates are up to date. Nothing to export.")
        if args.build_single_file and not args.single_file.exists():
            print("Building consolidated Parquet file...")
            schema = get_pyarrow_schema()
            export_consolidated_parquet(args.output_dir, args.single_file, schema)
            print(f"Saved: {args.single_file}")
        return

    # 3. Load constituency snapshots and stock histories
    print("\n[2/4] Loading constituency snapshots and stock histories...")
    snapshots = load_constituency_snapshots(CONSTITUENCY_ROOT)
    print(f"Loaded {len(snapshots)} constituency snapshots.")

    stock_t0 = time.monotonic()
    all_stocks = load_all_stocks(CLEAN_STOCK_ROOT)
    print(f"Loaded {len(all_stocks):,} stock rows in {time.monotonic() - stock_t0:.2f}s.")

    # 4. Process and export Parquet partitions
    print(f"\n[3/4] Exporting Parquet partitions for {len(selected_dates)} dates...")
    schema = get_pyarrow_schema()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed_count = 0
    total_rows = 0

    for idx, d_str in enumerate(selected_dates, start=1):
        ds_file = ds_index[d_str]
        atm_file = atm_index.get(d_str)

        day_df = build_single_day_dataset(
            trade_date_str=d_str,
            ds_path=ds_file,
            atm_path=atm_file,
            all_stocks=all_stocks,
            snapshots=snapshots,
            flatfiles_root=FLATFILES_ROOT,
        )

        out_path = export_partitioned_parquet_date(
            aligned_df=day_df,
            trade_date_str=d_str,
            output_dir=args.output_dir,
            schema=schema,
        )

        completed_count += 1
        total_rows += len(day_df)

        out_name = out_path.name if out_path is not None else "(skipped 0 rows)"
        if idx % 50 == 0 or idx == len(selected_dates):
            pct = int(100 * idx / len(selected_dates))
            print(f"  [{idx}/{len(selected_dates)}] {pct:3d}% - {d_str}: {len(day_df)} rows -> {out_name}")

    elapsed = time.monotonic() - start_time
    print(f"\nSuccessfully exported {completed_count} dates ({total_rows:,} rows) in {elapsed:.1f}s.")

    # 5. Build consolidated single file if requested
    if args.build_single_file:
        print("\n[4/4] Building consolidated single-file Parquet...")
        single_path = export_consolidated_parquet(args.output_dir, args.single_file, schema)
        size_mb = single_path.stat().st_size / (1024 * 1024)
        print(f"Saved consolidated Parquet to {single_path} ({size_mb:.1f} MB)")
    else:
        print("\n[4/4] Skipping single-file consolidation (pass --build-single-file to generate).")

    print("\nDataset Export Completed Successfully.")


if __name__ == "__main__":
    main()
