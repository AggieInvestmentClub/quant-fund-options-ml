"""Convert Ben's wide daily feature file into the dated research handoff."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PAPER_OPTIONS, SKEW_FEATURES, market_close_utc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stock_histories(stocks_dir: Path, raw: pd.DataFrame,
                     adjusted_column: str) -> tuple[pd.DataFrame, dict]:
    files = sorted(Path(stocks_dir).glob("*.csv"))
    if not files:
        raise ValueError(f"no stock history CSVs in {stocks_dir}")
    tickers = set(raw.underlying.astype(str))
    pieces = []
    manifest_digest = hashlib.sha256()
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        required = {"Date", "Close", "Adj Close", "Volume"}
        if not required.issubset(frame):
            continue  # Other CSVs can coexist in the stocks directory.
        if "ticker" not in frame:
            frame["ticker"] = path.stem
        frame = frame[frame.ticker.astype(str).isin(tickers)].copy()
        if frame.empty:
            continue
        frame = frame.rename(columns={"Date": "trade_date", "Close": "close",
                                      "Adj Close": "adjusted_close", "Volume": "volume"})
        frame = frame[["trade_date", "ticker", "close", "adjusted_close", "volume"]]
        frame["trade_date"] = pd.to_datetime(frame.trade_date, errors="raise").dt.normalize()
        for col in ("close", "adjusted_close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["trade_date", "close", "adjusted_close", "volume"])
        pieces.append(frame)
        manifest_digest.update(path.name.encode())
        manifest_digest.update(_sha256(path).encode())
    if not pieces:
        raise ValueError("stock history files contain no matching tickers")
    prices = pd.concat(pieces, ignore_index=True)
    if prices.duplicated(["trade_date", "ticker"]).any():
        raise ValueError("duplicate stock/date across stock history CSVs")
    wide_prices = raw[["trade_date", "underlying", "stock_close", adjusted_column, "market_cap"]].rename(
        columns={"underlying": "ticker", adjusted_column: "wide_adjusted_close"}
    )
    overlap = prices.merge(wide_prices, on=["trade_date", "ticker"], how="inner", validate="one_to_one")
    close_mismatch = (~np.isclose(overlap.close, overlap.stock_close, rtol=1e-5, atol=1e-4,
                                equal_nan=True)).sum()
    adjusted_bad = ~np.isclose(overlap.adjusted_close, overlap.wide_adjusted_close,
                               rtol=1e-5, atol=1e-4, equal_nan=True)
    adjusted_mismatch = int(adjusted_bad.sum())
    mismatch_rows = overlap.loc[adjusted_bad]
    relative_error = ((mismatch_rows.adjusted_close - mismatch_rows.wide_adjusted_close).abs()
                      / mismatch_rows.wide_adjusted_close.abs())
    prices = prices.merge(wide_prices[["trade_date", "ticker", "market_cap"]],
                          on=["trade_date", "ticker"], how="left", validate="one_to_one")
    report = {
        "stock_history_files_used": len(pieces), "stock_history_rows": len(prices),
        "stock_history_tickers": int(prices.ticker.nunique()),
        "overlap_stock_price_rows": len(overlap),
        "overlap_close_mismatches": int(close_mismatch),
        "overlap_adjusted_close_mismatches": int(adjusted_mismatch),
        "overlap_adjusted_close_mismatch_first_date": (
            str(mismatch_rows.trade_date.min().date()) if adjusted_mismatch else None
        ),
        "overlap_adjusted_close_mismatch_last_date": (
            str(mismatch_rows.trade_date.max().date()) if adjusted_mismatch else None
        ),
        "overlap_adjusted_close_mismatch_tickers": sorted(mismatch_rows.ticker.unique().tolist()),
        "overlap_adjusted_close_max_relative_error": (
            float(relative_error.max()) if adjusted_mismatch else 0.0
        ),
        "stock_history_manifest_sha256": manifest_digest.hexdigest(),
        "market_cap_missing_on_supplemental_days": int(prices.market_cap.isna().sum()),
    }
    return prices, report


def convert_ben_file(
    source: Path, output_dir: Path, *, ticker_history: Path | None = None,
    allow_ticker_proxy: bool = False, stocks_dir: Path | None = None,
) -> dict:
    """Write a pilot handoff. Never promote ticker or per-row flags to verified history."""
    source, output_dir = Path(source), Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if ticker_history is None and not allow_ticker_proxy:
        raise ValueError("supply ticker_history or opt in to pilot-only ticker proxy")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("use a fresh handoff directory")
    digest = _sha256(source)
    raw = pd.read_parquet(source)
    if raw.empty:
        raise ValueError("wide feature file is empty")
    adjusted_column = "stock_adj_close" if "stock_adj_close" in raw else "adj_close"
    needed = {"trade_date", "underlying", "tradable_date", *PAPER_OPTIONS,
              "stock_close", adjusted_column, "market_cap"}
    absent = needed - set(raw)
    if absent:
        raise ValueError(f"wide feature file missing {sorted(absent)}")
    volume_col = next((name for name in ("stock_volume", "volume") if name in raw), None)
    if volume_col is None and stocks_dir is None:
        raise ValueError("wide feature file missing stock volume")
    raw["trade_date"] = pd.to_datetime(raw.trade_date).dt.normalize()
    raw["tradable_date"] = pd.to_datetime(raw.tradable_date).dt.normalize()
    if raw[["trade_date", "tradable_date", "underlying"]].isna().any().any():
        raise ValueError("wide feature file has null stock/date keys")
    if (raw.tradable_date <= raw.trade_date).any():
        raise ValueError("tradable_date must follow trade_date")
    if raw.duplicated(["trade_date", "underlying"]).any():
        raise ValueError("duplicate stock/day in wide feature file")
    original_times = (pd.to_datetime(raw.options_available_time, utc=True, errors="coerce")
                      if "options_available_time" in raw
                      else pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]"))
    # The supplied 07:15 UTC timestamps precede documented provider readiness.
    # Use a conservative assumed 11:30 ET on the next tradable date for pilots.
    ready_lookup = {
        day: (day.tz_localize("America/New_York") + pd.Timedelta(hours=11, minutes=30)).tz_convert("UTC")
        for day in raw.tradable_date.unique()
    }
    ready = raw.tradable_date.map(ready_lookup)
    availability = pd.Series(pd.to_datetime(ready, utc=True), index=raw.index)
    availability = pd.concat([availability, original_times], axis=1).max(axis=1)
    options = raw[["trade_date", "underlying", *PAPER_OPTIONS,
                   *[col for col in SKEW_FEATURES if col in raw]]].copy()
    options = options.rename(columns={"underlying": "ticker"})
    options["available_at"] = availability
    options["source_version"] = f"ben-wide-sha256:{digest}"
    options["is_us_common_stock"] = pd.Series(pd.NA, index=options.index, dtype="boolean")
    # We keep is_sp500 only as an observation-level audit field; it is not a
    # complete membership snapshot for every stock on every historical date.
    if "is_sp500" in raw:
        options["observed_is_sp500"] = raw.is_sp500.astype("boolean")
    if stocks_dir is None:
        prices = raw[["trade_date", "underlying", "stock_close", adjusted_column,
                      volume_col, "market_cap"]].copy()
        prices = prices.rename(columns={"underlying": "ticker", "stock_close": "close",
                                        adjusted_column: "adjusted_close", volume_col: "volume"})
        for col in ("close", "adjusted_close", "volume", "market_cap"):
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
        price_missing = int(prices[["close", "adjusted_close", "volume", "market_cap"]].isna().any(axis=1).sum())
        prices = prices.dropna(subset=["close", "adjusted_close", "volume", "market_cap"]).copy()
        stock_report = {"stock_price_source": "wide_file_only"}
    else:
        prices, stock_report = _stock_histories(stocks_dir, raw, adjusted_column)
        stock_report["stock_price_source"] = "dated_stock_history_csvs"
        price_missing = int(prices[["close", "adjusted_close", "volume"]].isna().any(axis=1).sum())
    close_lookup = {day: market_close_utc(day) for day in prices.trade_date.unique()}
    prices["available_at"] = prices.trade_date.map(close_lookup)
    if stocks_dir is None:
        prices["source_version"] = f"ben-wide-sha256:{digest}"
    else:
        prices["source_version"] = f"stock-csv-manifest-sha256:{stock_report['stock_history_manifest_sha256']}"
    if ticker_history is None:
        options["security_id"] = "ticker_proxy:" + options.ticker.astype(str)
        prices["security_id"] = "ticker_proxy:" + prices.ticker.astype(str)
    output_dir.mkdir(parents=True, exist_ok=True)
    if ticker_history is not None:
        shutil.copy2(ticker_history, output_dir / "ticker_history.parquet")
    options.to_parquet(output_dir / "options_features.parquet", index=False)
    prices.to_parquet(output_dir / "stock_prices.parquet", index=False)
    report = {
        "source": str(source), "source_sha256": digest, "source_bytes": source.stat().st_size,
        "rows": len(raw), "first_date": str(raw.trade_date.min().date()),
        "last_date": str(raw.trade_date.max().date()),
        "month_count": int(raw.trade_date.dt.to_period("M").nunique()),
        "stock_count_by_ticker": int(raw.underlying.nunique()),
        "price_rows_dropped_for_missing": price_missing,
        "original_availability_before_assumed_ready": int((original_times < ready).fillna(False).sum()),
        "identity": "ticker_proxy_pilot_only" if ticker_history is None else "provided_ticker_history",
        "sp500_membership": "observation_flag_only_not_complete_snapshots",
        "us_common_stock_status": "unknown",
        "option_price_basis": "last_trade_not_bid_ask_midpoint",
        "availability_policy": "max(original timestamp, 11:30 America/New_York on tradable_date); assumed for pilot",
        **stock_report,
    }
    (output_dir / "adapter_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
