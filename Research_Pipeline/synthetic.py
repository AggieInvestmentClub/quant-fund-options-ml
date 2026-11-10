"""Synthetic engineering fixture; generated values have no financial meaning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import PAPER_CONTROLS


def _at(day: pd.Timestamp, hour: int) -> pd.Timestamp:
    return (pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=hour)).tz_convert("UTC")


def write_synthetic_handoff(root: Path, *, n_stocks: int = 60, n_months: int = 108, seed: int = 17) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2015-01", periods=n_months, freq="M")
    options, prices, controls, membership = [], [], [], []
    log_prices = np.log(rng.uniform(20, 100, size=n_stocks))
    states = np.resize(np.array([-1.0, 0.0, 1.0]), n_stocks)
    shares = rng.uniform(2e7, 1e9, size=n_stocks)
    for month_index, period in enumerate(periods):
        month_end = pd.offsets.BMonthEnd().rollback(period.to_timestamp(how="end").normalize())
        rebalance = month_end + pd.offsets.BDay(1)
        if month_index % 18 == 0 and month_index:
            states = rng.permutation(states)
        log_prices += 0.06 * states + rng.normal(0, 0.045, size=n_stocks)
        close = np.exp(log_prices)
        for stock_index in range(n_stocks):
            security_id = f"S{stock_index:04d}"
            ticker = f"T{stock_index:04d}" if stock_index or month_index < 54 else "RENAMED"
            market_cap = close[stock_index] * shares[stock_index]
            for day, price in ((month_end, close[stock_index]), (rebalance, close[stock_index] * np.exp(rng.normal(0, 0.003)))):
                prices.append({
                    "trade_date": day, "available_at": _at(day, 17), "security_id": security_id,
                    "ticker": ticker, "source_version": "synthetic-v1", "close": price,
                    "adjusted_close": price, "volume": int(rng.integers(10_000, 2_000_000)),
                    "market_cap": market_cap,
                })
            iv = 0.25 + 0.04 * abs(states[stock_index]) + rng.normal(0, 0.01)
            gap = 0.025 * states[stock_index] + rng.normal(0, 0.005)
            feature = {
                "trade_date": month_end, "available_at": _at(rebalance, 11),
                "security_id": security_id, "ticker": ticker, "source_version": "synthetic-v1",
                "is_us_common_stock": True,
                "avg_call_iv": iv + gap / 2, "avg_put_iv": iv - gap / 2,
                "avg_call_delta": 0.5 + rng.normal(0, 0.03),
                "avg_put_delta": -0.5 + rng.normal(0, 0.03),
                "avg_call_gamma": 0.02, "avg_put_gamma": 0.02,
                "avg_call_theta": -0.015, "avg_put_theta": -0.015,
                "avg_call_vega": 0.15, "avg_put_vega": 0.15,
                "atm_iv_30d": iv, "atm_iv_60d": iv + 0.01,
                "atm_iv_90d": iv + 0.015,
                "risk_reversal_25_delta_30d": gap,
                "risk_reversal_25_delta_60d": gap * 0.8,
                "risk_reversal_25_delta_90d": gap * 0.6,
                "put_skew_10_25_30d": -gap,
                "atm_term_slope": 0.015, "rr_term_slope": -0.4 * gap,
            }
            options.append(feature)
            control = {
                "as_of_date": month_end, "available_at": _at(month_end, 17),
                "security_id": security_id, "ticker": ticker, "source_version": "synthetic-v1",
            }
            for name in PAPER_CONTROLS:
                control[name] = float(rng.normal() + 0.2 * states[stock_index])
            controls.append(control)
            if month_index % 3 == 0 and stock_index < n_stocks // 2:
                membership.append({
                    "snapshot_date": month_end, "available_at": _at(month_end, 17),
                    "security_id": security_id, "source_version": "synthetic-v1",
                })
    for name, rows in (
        ("options_features", options), ("stock_prices", prices),
        ("stock_controls", controls), ("sp500_membership", membership),
    ):
        pd.DataFrame(rows).to_parquet(root / f"{name}.parquet", index=False)
