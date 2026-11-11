"""Fetch historical shares outstanding and calculate daily market cap using yfinance.

Builds a point-in-time daily series of:
- shares_outstanding
- market_cap = shares_outstanding * stock_close

Saves output to:
- /srv/data/stocks/shares_outstanding.parquet
"""

import argparse
import concurrent.futures
from datetime import date
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT))

from Dataset_Prep.config import CLEAN_STOCK_ROOT

SHARES_OUTPUT_PARQUET = CLEAN_STOCK_ROOT / "shares_outstanding.parquet"
SHARES_OUTPUT_CSV = CLEAN_STOCK_ROOT / "shares_outstanding.csv"


def yf_symbol_for_ticker(symbol: str) -> str:
    """Map ticker symbols like BRK.B or BRKB to yfinance form BRK-B."""
    sym = symbol.strip().upper().replace(".", "-")
    if sym == "BRKB":
        return "BRK-B"
    if sym == "BFB":
        return "BF-B"
    return sym


def fetch_ticker_shares(ticker: str, start_date: str = "2023-01-01") -> Tuple[str, Optional[pd.DataFrame], Optional[float]]:
    """Fetch shares timeseries or current shares from yfinance for a single ticker."""
    yf_symbol = yf_symbol_for_ticker(ticker)
    t = yf.Ticker(yf_symbol)

    # 1. Try historical shares outstanding series
    try:
        shares = t.get_shares_full(start=start_date)
        if shares is not None and len(shares) > 0:
            df = shares.reset_index()
            date_col = df.columns[0]
            val_col = df.columns[-1]
            df = df.rename(columns={date_col: "Date", val_col: "shares_outstanding"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None).dt.strftime("%Y-%m-%d")
            df["shares_outstanding"] = pd.to_numeric(df["shares_outstanding"], errors="coerce")
            df = df.dropna(subset=["Date", "shares_outstanding"])
            df = df[df["shares_outstanding"] > 0]
            df = df.drop_duplicates(subset=["Date"], keep="last")
            if not df.empty:
                return ticker, df[["Date", "shares_outstanding"]], None
    except Exception:
        pass

    # 2. Fallback to fast_info shares
    try:
        fast_shares = t.fast_info.get("shares")
        if fast_shares and float(fast_shares) > 0:
            return ticker, None, float(fast_shares)
    except Exception:
        pass

    # 3. Fallback to info shares
    try:
        info_shares = t.info.get("sharesOutstanding")
        if info_shares and float(info_shares) > 0:
            return ticker, None, float(info_shares)
    except Exception:
        pass

    return ticker, None, None


def enrich_stock_history_with_shares(
    stock_path: Path,
    shares_df: Optional[pd.DataFrame],
    fallback_shares: Optional[float],
) -> Optional[pd.DataFrame]:
    """Merge shares outstanding onto a stock's CSV history and compute market_cap."""
    if not stock_path.exists():
        return None

    try:
        stk_df = pd.read_csv(stock_path)
    except Exception:
        return None

    if stk_df.empty or "Date" not in stk_df.columns or "Close" not in stk_df.columns:
        return None

    ticker = stock_path.name.replace("_stock_data.csv", "")
    stk_df["Date"] = stk_df["Date"].astype(str)

    # Clean existing shares/market_cap columns to prevent collisions
    keep_cols = [c for c in ["Date", "Close"] if c in stk_df.columns]
    sub = stk_df[keep_cols].copy()

    if shares_df is not None and not shares_df.empty:
        merged = pd.merge(sub, shares_df[["Date", "shares_outstanding"]], on="Date", how="left")
        merged["shares_outstanding"] = merged["shares_outstanding"].ffill().bfill()
    elif fallback_shares is not None:
        merged = sub.copy()
        merged["shares_outstanding"] = float(fallback_shares)
    else:
        merged = sub.copy()
        merged["shares_outstanding"] = np.nan

    merged["stock_close"] = pd.to_numeric(merged["Close"], errors="coerce")
    merged["shares_outstanding"] = pd.to_numeric(merged["shares_outstanding"], errors="coerce")
    merged["market_cap"] = merged["stock_close"] * merged["shares_outstanding"]

    out = pd.DataFrame({
        "trade_date": merged["Date"],
        "underlying": ticker,
        "stock_close": merged["stock_close"],
        "shares_outstanding": merged["shares_outstanding"],
        "market_cap": merged["market_cap"],
    })
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical shares and compute market cap via yfinance.")
    parser.add_argument(
        "--stock-root",
        type=Path,
        default=CLEAN_STOCK_ROOT,
        help="Path to stock CSV directory (default: /srv/data/stocks).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download threads (default: 8).",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=SHARES_OUTPUT_PARQUET,
        help="Path for combined shares/market-cap parquet lookup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.monotonic()

    stock_files = sorted(args.stock_root.glob("*_stock_data.csv"))
    tickers = [f.name.replace("_stock_data.csv", "") for f in stock_files]
    print("=" * 60)
    print("Fetching Historical Shares Outstanding & Market Cap via yfinance")
    print(f"Stock Directory: {args.stock_root}")
    print(f"Total Tickers:   {len(tickers)}")
    print(f"Threads:         {args.workers}")
    print("=" * 60)

    # 1. Fetch shares in parallel
    print(f"\n[1/3] Downloading shares data from yfinance for {len(tickers)} tickers...")
    results: Dict[str, Tuple[Optional[pd.DataFrame], Optional[float]]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_ticker = {executor.submit(fetch_ticker_shares, ticker): ticker for ticker in tickers}
        for completed, future in enumerate(concurrent.futures.as_completed(future_to_ticker), start=1):
            ticker, shares_df, fallback_val = future.result()
            results[ticker] = (shares_df, fallback_val)
            if completed % 50 == 0 or completed == len(tickers):
                pct = int(100 * completed / len(tickers))
                print(f"  [{completed}/{len(tickers)}] {pct:3d}% complete")

    # 2. Enrich each stock file and build lookup table
    print("\n[2/3] Merging shares with stock prices and calculating market cap...")
    all_rows: List[pd.DataFrame] = []
    success_historical = 0
    success_fallback = 0
    missing_count = 0

    for ticker in tickers:
        stock_path = args.stock_root / f"{ticker}_stock_data.csv"
        shares_df, fallback_val = results.get(ticker, (None, None))

        if shares_df is not None:
            success_historical += 1
        elif fallback_val is not None:
            success_fallback += 1
        else:
            missing_count += 1

        enriched = enrich_stock_history_with_shares(stock_path, shares_df, fallback_val)
        if enriched is not None:
            all_rows.append(enriched)

    print(f"  Tickers with full historical shares: {success_historical}")
    print(f"  Tickers with fallback current shares: {success_fallback}")
    print(f"  Tickers with missing shares:           {missing_count}")

    # 3. Save combined lookup Parquet and CSV
    print(f"\n[3/3] Saving combined shares & market cap lookup to {args.output_parquet}...")
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        tmp_parquet = args.output_parquet.with_name(f".{args.output_parquet.name}.tmp")
        combined.to_parquet(tmp_parquet, compression="zstd", index=False)
        tmp_parquet.replace(args.output_parquet)
        print(f"  Saved {len(combined):,} rows to {args.output_parquet} ({args.output_parquet.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print("  Warning: No data to save.")

    elapsed = time.monotonic() - t0
    print(f"\nCompleted in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
