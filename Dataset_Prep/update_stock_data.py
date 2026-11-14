"""Tool to fetch missing daily stock OHLCV from Yahoo Finance and append to /srv/data/stocks."""

import argparse
from datetime import date, timedelta
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT))

from Dataset_Prep.config import CLEAN_STOCK_ROOT


def update_stocks_for_date(
    target_date: date,
    stock_root: Path = CLEAN_STOCK_ROOT,
    batch_size: int = 50,
) -> Dict[str, int]:
    """Download OHLCV for target_date and append to stock CSVs."""
    target_date_str = target_date.isoformat()
    stock_files = sorted(stock_root.glob("*_stock_data.csv"))
    if not stock_files:
        print(f"No stock files found in {stock_root}")
        return {"updated": 0, "skipped": 0, "failed": 0}

    symbols = [f.name.replace("_stock_data.csv", "") for f in stock_files]
    symbols_to_update: List[str] = []

    # Check which symbols need target_date
    for s in symbols:
        p = stock_root / f"{s}_stock_data.csv"
        try:
            tail = pd.read_csv(p, nrows=5)  # quick check
            # Read last few lines
            df_tail = pd.read_csv(p).tail(5)
            if target_date_str not in df_tail["Date"].values:
                symbols_to_update.append(s)
        except Exception:
            symbols_to_update.append(s)

    print(f"Found {len(symbols_to_update)} / {len(symbols)} stocks needing {target_date_str}")
    if not symbols_to_update:
        return {"updated": 0, "skipped": len(symbols), "failed": 0}

    updated = 0
    failed = 0
    end_date = target_date + timedelta(days=1)

    # Download in batches
    for i in range(0, len(symbols_to_update), batch_size):
        chunk = symbols_to_update[i : i + batch_size]
        yf_map = {s.replace(".", "-"): s for s in chunk}
        print(f"  Downloading batch [{i+1}-{min(i+batch_size, len(symbols_to_update))}] of {len(symbols_to_update)}...")

        try:
            downloaded = yf.download(
                tickers=list(yf_map.keys()),
                start=target_date.isoformat(),
                end=end_date.isoformat(),
                group_by="ticker",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            print(f"  Batch download failed: {exc}")
            failed += len(chunk)
            continue

        for yf_sym, sym in yf_map.items():
            csv_path = stock_root / f"{sym}_stock_data.csv"
            try:
                if len(chunk) == 1:
                    raw = downloaded
                else:
                    if yf_sym not in downloaded.columns.levels[0]:
                        failed += 1
                        continue
                    raw = downloaded[yf_sym]

                sub = raw.dropna(how="all")
                if sub.empty:
                    failed += 1
                    continue

                # Filter row for target date
                sub = sub[sub.index.strftime("%Y-%m-%d") == target_date_str]
                if sub.empty:
                    failed += 1
                    continue

                row = sub.iloc[0]
                existing_df = pd.read_csv(csv_path)
                if target_date_str in existing_df["Date"].astype(str).values:
                    continue

                last_row = existing_df.iloc[-1] if not existing_df.empty else {}

                new_record = {}
                for col in existing_df.columns:
                    if col == "Date":
                        new_record[col] = target_date_str
                    elif col == "Open":
                        new_record[col] = float(row.get("Open", row.get("Close", 0.0)))
                    elif col == "High":
                        new_record[col] = float(row.get("High", row.get("Close", 0.0)))
                    elif col == "Low":
                        new_record[col] = float(row.get("Low", row.get("Close", 0.0)))
                    elif col == "Close":
                        new_record[col] = float(row.get("Close", 0.0))
                    elif col == "Adj Close":
                        new_record[col] = float(row.get("Adj Close", row.get("Close", 0.0)))
                    elif col == "Volume":
                        new_record[col] = float(row.get("Volume", 0.0))
                    elif col == "Dividends":
                        new_record[col] = float(row.get("Dividends", 0.0))
                    elif col == "Stock Splits":
                        new_record[col] = float(row.get("Stock Splits", 0.0))
                    elif col == "ticker":
                        new_record[col] = sym
                    elif col == "lseg_universe":
                        new_record[col] = "yfinance-raw-split-safe"
                    elif col in last_row:
                        new_record[col] = last_row[col]
                    else:
                        new_record[col] = np.nan

                # If dividend yield exists
                if "dividend_yield" in new_record and "Close" in new_record and new_record["Close"] > 0:
                    trailing_divs = existing_df["Dividends"].tail(252).sum() + float(row.get("Dividends", 0.0))
                    new_record["dividend_yield"] = trailing_divs / new_record["Close"]

                combined = pd.concat([existing_df, pd.DataFrame([new_record])], ignore_index=True)
                tmp_path = csv_path.with_name(f".{sym}_stock_data.csv.tmp")
                combined.to_csv(tmp_path, index=False)
                os.chmod(tmp_path, 0o666)
                tmp_path.replace(csv_path)
                updated += 1
            except Exception as e:
                failed += 1

    print(f"\nUpdate completed: updated={updated}, failed={failed}, skipped={len(symbols) - len(symbols_to_update)}")
    return {"updated": updated, "failed": failed, "skipped": len(symbols) - len(symbols_to_update)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch missing stock OHLCV from yfinance.")
    parser.add_argument("--target-date", type=date.fromisoformat, default=date(2026, 11, 12))
    args = parser.parse_args()
    update_stocks_for_date(args.target_date)


if __name__ == "__main__":
    main()

