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
    assert aligned["avg_call_iv"].dtype == np.float32
    assert aligned["stock_close"].dtype == np.float64
    assert pd.isna(aligned["avg_put_iv"].iloc[0])


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
