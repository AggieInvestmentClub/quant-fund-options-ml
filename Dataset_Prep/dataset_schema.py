"""Schema definitions and type casting for the consolidated ML Parquet dataset."""

from typing import Dict, List
import pyarrow as pa
import pandas as pd
import numpy as np

# Primary Identifiers
IDENTIFIER_COLUMNS: List[str] = [
    "trade_date",
    "underlying",
    "is_sp500",
]

# Stock Price & Volume Features
STOCK_COLUMNS: List[str] = [
    "stock_open",
    "stock_high",
    "stock_low",
    "stock_close",
    "stock_adj_close",
    "stock_volume",
    "dividend_yield",
    "stock_beta",
    "historical_beta",
    "shares_outstanding",
    "market_cap",
]

# Ten ATM IV and Greek Averages (from Phase 1 ATM stage)
ATM_GREEK_COLUMNS: List[str] = [
    "avg_call_iv",
    "avg_put_iv",
    "avg_call_delta",
    "avg_put_delta",
    "avg_call_gamma",
    "avg_put_gamma",
    "avg_call_theta",
    "avg_put_theta",
    "avg_call_vega",
    "avg_put_vega",
]

# Surface & Delta-Bucket Skew / Risk-Reversal Features (from Phase 2 Default Strike stage)
SURFACE_COLUMNS: List[str] = [
    "put_10_delta_iv",
    "put_25_delta_iv",
    "atm_call_iv",
    "atm_put_iv",
    "atm_iv",
    "call_25_delta_iv",
    "call_10_delta_iv",
    "risk_reversal_25_delta",
    "risk_reversal_10_delta",
    "put_skew_10_25",
    "call_skew_10_25",
    "curvature_25_delta",
    "curvature_10_delta",
    "skew_slope",
    "put_25_delta_gamma",
    "call_25_delta_gamma",
    "put_10_delta_gamma",
    "call_10_delta_gamma",
    "gamma_rr_25_delta",
    "gamma_wing_ratio",
    "vega_rr_25_delta",
    "theta_rr_25_delta",
    "near_atm_gamma_sum_raw",
    "put_gamma_sum_raw",
    "call_gamma_sum_raw",
    "gamma_imbalance_sum_raw",
    "near_atm_gamma_mean",
    "put_gamma_mean",
    "call_gamma_mean",
    "gamma_imbalance_mean",
]

DTE_BUCKET_COLUMNS: List[str] = []
for dte in ["30d", "60d", "90d"]:
    DTE_BUCKET_COLUMNS.extend([
        f"put_10_delta_iv_{dte}",
        f"put_25_delta_iv_{dte}",
        f"atm_call_iv_{dte}",
        f"atm_put_iv_{dte}",
        f"atm_iv_{dte}",
        f"call_25_delta_iv_{dte}",
        f"call_10_delta_iv_{dte}",
        f"risk_reversal_25_delta_{dte}",
        f"risk_reversal_10_delta_{dte}",
        f"put_skew_10_25_{dte}",
        f"call_skew_10_25_{dte}",
        f"curvature_25_delta_{dte}",
        f"curvature_10_delta_{dte}",
        f"skew_slope_{dte}",
        f"put_25_delta_gamma_{dte}",
        f"call_25_delta_gamma_{dte}",
        f"put_10_delta_gamma_{dte}",
        f"call_10_delta_gamma_{dte}",
        f"gamma_rr_25_delta_{dte}",
        f"gamma_wing_ratio_{dte}",
        f"vega_rr_25_delta_{dte}",
        f"theta_rr_25_delta_{dte}",
    ])

TERM_AND_COUNT_COLUMNS: List[str] = [
    "atm_term_slope",
    "put_skew_term_slope",
    "rr_term_slope",
    "n_contracts_total",
    "n_contracts_30d",
    "n_contracts_60d",
    "n_contracts_90d",
    "put_10_delta_distance",
    "put_25_delta_distance",
    "atm_call_delta_distance",
    "atm_put_delta_distance",
    "call_25_delta_distance",
    "call_10_delta_distance",
]

METADATA_COLUMNS: List[str] = [
    "options_available_time",
    "tradable_date",
]

ALL_DATASET_COLUMNS: List[str] = (
    IDENTIFIER_COLUMNS
    + METADATA_COLUMNS
    + STOCK_COLUMNS
    + ATM_GREEK_COLUMNS
    + SURFACE_COLUMNS
    + DTE_BUCKET_COLUMNS
    + TERM_AND_COUNT_COLUMNS
)


def get_pyarrow_schema() -> pa.Schema:
    fields = [
        # Identifiers
        pa.field("trade_date", pa.string(), nullable=False),
        pa.field("underlying", pa.string(), nullable=False),
        pa.field("is_sp500", pa.bool_(), nullable=False),
        # Metadata
        pa.field("options_available_time", pa.string(), nullable=True),
        pa.field("tradable_date", pa.string(), nullable=True),
        # Stock prices (float64 for precision) & volume (int64/float64)
        pa.field("stock_open", pa.float64(), nullable=True),
        pa.field("stock_high", pa.float64(), nullable=True),
        pa.field("stock_low", pa.float64(), nullable=True),
        pa.field("stock_close", pa.float64(), nullable=True),
        pa.field("stock_adj_close", pa.float64(), nullable=True),
        pa.field("stock_volume", pa.float64(), nullable=True),
        pa.field("dividend_yield", pa.float64(), nullable=True),
        pa.field("stock_beta", pa.float64(), nullable=True),
        pa.field("historical_beta", pa.float64(), nullable=True),
        pa.field("shares_outstanding", pa.float64(), nullable=True),
        pa.field("market_cap", pa.float64(), nullable=True),
    ]

    # ATM Greek Averages (float32 is sufficient for Greeks and saves memory)
    for col in ATM_GREEK_COLUMNS:
        fields.append(pa.field(col, pa.float32(), nullable=True))

    # Surface columns
    for col in SURFACE_COLUMNS:
        fields.append(pa.field(col, pa.float32(), nullable=True))

    # Bucketed columns
    for col in DTE_BUCKET_COLUMNS:
        fields.append(pa.field(col, pa.float32(), nullable=True))

    # Term & count columns
    for col in TERM_AND_COUNT_COLUMNS:
        if col.startswith("n_contracts"):
            fields.append(pa.field(col, pa.float32(), nullable=True))
        else:
            fields.append(pa.field(col, pa.float32(), nullable=True))

    return pa.schema(fields)


def cast_and_align_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected columns exist and cast to target dtypes."""
    missing = [col for col in ALL_DATASET_COLUMNS if col not in df.columns]
    if missing:
        missing_dict = {col: False if col == "is_sp500" else np.nan for col in missing}
        missing_df = pd.DataFrame(missing_dict, index=df.index)
        df = pd.concat([df, missing_df], axis=1)

    df = df[ALL_DATASET_COLUMNS].copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df["underlying"] = df["underlying"].astype(str)
    df["is_sp500"] = df["is_sp500"].astype(bool)
    df["options_available_time"] = df["options_available_time"].astype(str)
    df["tradable_date"] = df["tradable_date"].astype(str)

    # Cast stock columns to float64
    for col in STOCK_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Cast option features to float32
    float32_cols = ATM_GREEK_COLUMNS + SURFACE_COLUMNS + DTE_BUCKET_COLUMNS + TERM_AND_COUNT_COLUMNS
    for col in float32_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df
