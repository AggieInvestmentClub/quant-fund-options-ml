"""Unit tests for Dataset_Prep pipeline modules."""

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT))

from Dataset_Prep.config import PARQUET_OUTPUT_DIR, PARQUET_SINGLE_FILE
from Dataset_Prep.dataset_schema import (
    ALL_DATASET_COLUMNS,
    ATM_GREEK_COLUMNS,
    IDENTIFIER_COLUMNS,
    STOCK_COLUMNS,
    cast_and_align_frame,
    get_pyarrow_schema,
)
from Dataset_Prep.export_daily_parquet import (
    build_single_day_dataset,
    next_business_day,
    resolve_options_availability,
)


def test_schema_definition():
    schema = get_pyarrow_schema()
    assert isinstance(schema, pa.Schema)
    assert len(schema.names) == len(ALL_DATASET_COLUMNS)
    assert "trade_date" in schema.names
    assert "underlying" in schema.names
    assert "is_sp500" in schema.names
    assert "avg_call_iv" in schema.names
    assert "risk_reversal_25_delta_30d" in schema.names
    assert "stock_adj_close" in schema.names
    assert "shares_outstanding" in schema.names
    assert "market_cap" in schema.names
    assert "size" in schema.names
    assert "tvol_20d" in schema.names
    assert "rm1" in schema.names
    assert "iv_gap" in schema.names
    assert "forward_5d_return" in schema.names
    assert "jump_label_5d" in schema.names
    assert "is_actively_trading" in schema.names
    assert "is_options_investable" in schema.names


def test_cast_and_align_frame():
    raw_df = pd.DataFrame([
        {
            "trade_date": "2026-11-09",
            "underlying": "AAPL",
            "is_sp500": True,
            "stock_close": 340.5,
            "stock_adj_close": 340.5,
            "avg_call_iv": 0.285,
        }
    ])
    aligned = cast_and_align_frame(raw_df)
    assert list(aligned.columns) == ALL_DATASET_COLUMNS
    assert aligned["is_sp500"].dtype == bool
    assert aligned["is_options_investable"].dtype == bool
    assert aligned["avg_call_iv"].dtype == np.float32
    assert aligned["stock_close"].dtype == np.float64
    assert pd.isna(aligned["avg_put_iv"].iloc[0])


def test_filter_investable_universe():
    from Dataset_Prep.enrich_features import filter_investable_universe

    test_df = pd.DataFrame([
        {
            "underlying": "AAPL",
            "is_sp500": True,
            "is_actively_trading": True,
            "stock_close": 150.0,
            "market_cap": 2.5e12,
            "stock_volume": 1_000_000,
            "n_contracts_total": 500,
            "iv_level": 0.25,
            "is_options_investable": True,
        },
        {
            "underlying": "DEAD_TICKER",
            "is_sp500": False,
            "is_actively_trading": False,
            "stock_close": 2.0,
            "market_cap": 1.0e7,
            "stock_volume": 100,
            "n_contracts_total": 2,
            "iv_level": np.nan,
            "is_options_investable": False,
        },
    ])
    filtered = filter_investable_universe(test_df)
    assert len(filtered) == 1
    assert filtered.iloc[0]["underlying"] == "AAPL"


def test_next_business_day_skip_weekend():
    friday = date(2026, 11, 6)
    monday = next_business_day(friday)
    assert monday == date(2026, 11, 9)


def test_options_availability():
    thursday = date(2026, 11, 5)
    avail_time, tradable_date = resolve_options_availability(thursday, Path("/tmp/fake_flatfiles"))
    assert tradable_date == "2026-11-06"
    assert "2026-11-06T07:15:00Z" in avail_time


def test_consolidated_parquet_readable():
    if PARQUET_SINGLE_FILE.exists():
        df = pd.read_parquet(PARQUET_SINGLE_FILE)
        assert len(df) > 1000
        assert "avg_call_iv" in df.columns
        assert "risk_reversal_25_delta_30d" in df.columns
        assert "stock_adj_close" in df.columns


def test_enrich_single_day_option_features_coalescing():
    from Dataset_Prep.enrich_features import enrich_single_day_option_features

    # Case 1: Only call IV is present (put IV is NaN)
    df1 = pd.DataFrame([{
        "avg_call_iv": 0.30,
        "avg_put_iv": np.nan,
        "call_25_delta_iv": 0.28,
        "put_25_delta_iv": 0.32,
        "tvol_20d": 0.20,
    }])
    res1 = enrich_single_day_option_features(df1)
    assert not pd.isna(res1["iv_level"].iloc[0])
    assert np.isclose(res1["iv_level"].iloc[0], 0.30, atol=1e-4)
    assert pd.isna(res1["iv_gap"].iloc[0])
    assert not pd.isna(res1["variance_proxy"].iloc[0])
    assert np.isclose(res1["variance_proxy"].iloc[0], 0.30**2 - 0.20**2, atol=1e-4)

    # Case 2: Exact ATM IV missing, but 25-delta wing IVs present
    df2 = pd.DataFrame([{
        "avg_call_iv": np.nan,
        "avg_put_iv": np.nan,
        "call_25_delta_iv": 0.26,
        "put_25_delta_iv": 0.34,
        "tvol_20d": 0.25,
    }])
    res2 = enrich_single_day_option_features(df2)
    assert not pd.isna(res2["atm_iv"].iloc[0])
    assert np.isclose(res2["atm_iv"].iloc[0], 0.30, atol=1e-4)
    assert np.isclose(res2["iv_level"].iloc[0], 0.30, atol=1e-4)


def test_robust_risk_reversal_fallbacks():
    from Dataset_Prep.enrich_features import enrich_single_day_option_features

    # 30d RR missing, but 90d RR and wing differences present
    df = pd.DataFrame([{
        "risk_reversal_25_delta_30d": np.nan,
        "risk_reversal_25_delta_60d": np.nan,
        "risk_reversal_25_delta_90d": -0.04,
        "call_25_delta_iv_30d": np.nan,
        "put_25_delta_iv_30d": np.nan,
    }])
    res = enrich_single_day_option_features(df)
    assert not pd.isna(res["risk_reversal_25_delta_30d_robust"].iloc[0])
    assert np.isclose(res["risk_reversal_25_delta_30d_robust"].iloc[0], -0.04, atol=1e-4)


def test_enrich_stock_history_time_series_fallbacks():
    from Dataset_Prep.enrich_features import enrich_stock_history_time_series

    # Stock with 30 days of data (enough for 20d vol, 1m momentum, but < 60d/252d)
    dates = pd.date_range("2026-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
    prices = [100.0 * (1.002 ** i) for i in range(30)]
    stocks_df = pd.DataFrame({
        "trade_date": dates,
        "underlying": "TEST",
        "stock_close": prices,
        "stock_adj_close": prices,
        "stock_volume": [1_000_000.0] * 30,
        "shares_outstanding": [10_000_000.0] * 30,
        "market_cap": [p * 10_000_000.0 for p in prices],
    })
    enriched = enrich_stock_history_time_series(stocks_df)
    # Check that tvol_60d fell back to tvol_20d
    assert not pd.isna(enriched["tvol_60d"].iloc[-1])
    assert np.isclose(enriched["tvol_60d"].iloc[-1], enriched["tvol_20d"].iloc[-1], atol=1e-4)
    # Check that rm3 compounded rm1
    assert not pd.isna(enriched["rm3"].iloc[-1])
    assert np.isclose(enriched["rm3"].iloc[-1], ((1.0 + enriched["rm1"].iloc[-1]) ** 3.0) - 1.0, atol=1e-4)
    # Check that rm12 compounded rm1
    assert not pd.isna(enriched["rm12"].iloc[-1])
    # Check dturn_252d fell back to 20d turnover
    assert not pd.isna(enriched["dturn_252d"].iloc[-1])


def test_split_adjusted_ohlc_and_tradable_targets():
    from Dataset_Prep.enrich_features import enrich_stock_history_time_series

    # Simulate stock with a 2-for-1 split halfway through (price halves, volume doubles)
    dates = pd.date_range("2026-01-01", periods=10, freq="B").strftime("%Y-%m-%d")
    # First 5 days unadjusted close is $200, adj close is $100. Next 5 days close is $100, adj close is $100.
    unadj_close = [200.0] * 5 + [100.0] * 5
    unadj_open = [200.0] * 5 + [100.0] * 5
    adj_close = [100.0] * 10
    volume = [500_000.0] * 5 + [1_000_000.0] * 5

    stocks_df = pd.DataFrame({
        "trade_date": dates,
        "underlying": "SPLIT_CO",
        "stock_open": unadj_open,
        "stock_high": unadj_open,
        "stock_low": unadj_close,
        "stock_close": unadj_close,
        "stock_adj_close": adj_close,
        "stock_volume": volume,
        "shares_outstanding": [10_000_000.0] * 10,
        "market_cap": [1_000_000_000.0] * 10,
    })
    enriched = enrich_stock_history_time_series(stocks_df)

    # 1. Verify split-adjusted open matches adjusted close ($100 across all 10 days)
    assert np.allclose(enriched["stock_adj_open"], 100.0)
    # 2. Verify dollar volume is properly computed
    assert np.isclose(enriched["dollar_volume"].iloc[0], 200.0 * 500_000.0)
    # 3. Verify volume ratio
    assert not pd.isna(enriched["volume_ratio_20d"].iloc[-1])
    # 4. Tradable returns: since split-adjusted price is constant ($100), tradable return should be 0.0
    assert np.isclose(enriched["target_open_to_close_1d"].iloc[0], 0.0, atol=1e-4)
    assert np.isclose(enriched["target_tradable_5d_return"].iloc[0], 0.0, atol=1e-4)
    # 5. Target validity: row 0 has 5d forward return, row 9 does not
    assert enriched["is_target_valid_5d"].iloc[0] == True
    assert enriched["is_target_valid_5d"].iloc[9] == False


def test_cross_sectional_ranks_and_dimensionless_ratios():
    from Dataset_Prep.enrich_features import enrich_single_day_option_features

    df = pd.DataFrame([
        {
            "trade_date": "2026-11-09",
            "underlying": "LOW_IV",
            "avg_call_iv": 0.15,
            "avg_put_iv": 0.15,
            "tvol_20d": 0.10,
            "is_sp500": True,
            "stock_close": 100.0,
            "market_cap": 10e9,
            "stock_volume": 500_000,
            "n_contracts_total": 50,
        },
        {
            "trade_date": "2026-11-09",
            "underlying": "HIGH_IV",
            "avg_call_iv": 0.60,
            "avg_put_iv": 0.60,
            "tvol_20d": 0.30,
            "is_sp500": True,
            "stock_close": 50.0,
            "market_cap": 5e9,
            "stock_volume": 300_000,
            "n_contracts_total": 100,
        },
    ])
    res = enrich_single_day_option_features(df)

    # 1. Dimensionless IV/RV ratio
    assert np.isclose(res["iv_to_rv_ratio"].iloc[0], 0.15 / 0.10, atol=1e-4)
    assert np.isclose(res["iv_to_rv_ratio"].iloc[1], 0.60 / 0.30, atol=1e-4)

    # 2. Cross-sectional ranks: LOW_IV should have rank 0.5, HIGH_IV should have rank 1.0
    assert np.isclose(res["iv_level_cs_rank"].iloc[0], 0.5, atol=1e-4)
    assert np.isclose(res["iv_level_cs_rank"].iloc[1], 1.0, atol=1e-4)


