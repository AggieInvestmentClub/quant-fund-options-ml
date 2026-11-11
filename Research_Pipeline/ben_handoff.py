"""Convert Ben's wide daily feature file into the dated research handoff."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from .data import PAPER_OPTIONS, SKEW_FEATURES, market_close_utc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_ben_file(
    source: Path, output_dir: Path, *, ticker_history: Path | None = None,
    allow_ticker_proxy: bool = False,
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
    needed = {"trade_date", "underlying", "tradable_date", *PAPER_OPTIONS,
              "stock_close", "adj_close", "market_cap"}
    absent = needed - set(raw)
    if absent:
        raise ValueError(f"wide feature file missing {sorted(absent)}")
    volume_col = next((name for name in ("stock_volume", "volume") if name in raw), None)
    if volume_col is None:
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
    prices = raw[["trade_date", "underlying", "stock_close", "adj_close",
                  volume_col, "market_cap"]].copy()
    prices = prices.rename(columns={"underlying": "ticker", "stock_close": "close",
                                    "adj_close": "adjusted_close", volume_col: "volume"})
    for col in ("close", "adjusted_close", "volume", "market_cap"):
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    price_missing = int(prices[["close", "adjusted_close", "volume", "market_cap"]].isna().any(axis=1).sum())
    prices = prices.dropna(subset=["close", "adjusted_close", "volume", "market_cap"]).copy()
    close_lookup = {day: market_close_utc(day) for day in prices.trade_date.unique()}
    prices["available_at"] = prices.trade_date.map(close_lookup)
    prices["source_version"] = f"ben-wide-sha256:{digest}"
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
    }
    (output_dir / "adapter_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
