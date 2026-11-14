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
    "stock_adj_open",
    "stock_adj_high",
    "stock_adj_low",
    "stock_volume",
    "dollar_volume",
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

# Academic Company Controls (from Andreou, Han, and Li 2025)
ACADEMIC_CONTROL_COLUMNS: List[str] = [
    "size",
    "tvol_20d",
    "tvol_60d",
    "tskew_60d",
    "rm1",
    "rm3",
    "rm12",
    "exret_daily",
    "dturn_20d",
    "dturn_252d",
    "volume_ratio_20d",
]

# Compact Option and Variance Features (from Paper Part II)
COMPACT_OPTION_COLUMNS: List[str] = [
    "iv_level",
    "iv_gap",
    "variance_proxy",
    "iv_to_rv_ratio",
    "delta_5d_iv_gap",
    "delta_20d_iv_gap",
    "delta_5d_rr_60d",
    "risk_reversal_25_delta_30d_robust",
    "risk_reversal_25_delta_60d_robust",
    "atm_iv_30d_robust",
    "iv_level_cs_rank",
    "risk_reversal_30d_cs_rank",
    "variance_proxy_cs_rank",
]

# Multi-Horizon Return Targets, Labels, and Trading Activity Flags
LABEL_AND_TARGET_COLUMNS: List[str] = [
    "forward_1d_return",
    "forward_5d_return",
    "forward_10d_return",
    "forward_21d_return",
    "forward_63d_return",
    "target_open_to_close_1d",
    "target_tradable_5d_return",
    "target_tradable_21d_return",
    "forward_5d_excess_return",
    "forward_21d_excess_return",
    "outperformance_label_5d",
    "outperformance_label_21d",
    "is_target_valid_5d",
    "is_target_valid_21d",
    "jump_label_5d",
    "crash_label_5d",
    "jump_label_21d",
    "crash_label_21d",
    "is_actively_trading",
    "is_options_investable",
]

ALL_DATASET_COLUMNS: List[str] = (
    IDENTIFIER_COLUMNS
    + METADATA_COLUMNS
    + STOCK_COLUMNS
    + ACADEMIC_CONTROL_COLUMNS
    + ATM_GREEK_COLUMNS
    + SURFACE_COLUMNS
    + DTE_BUCKET_COLUMNS
    + TERM_AND_COUNT_COLUMNS
    + COMPACT_OPTION_COLUMNS
    + LABEL_AND_TARGET_COLUMNS
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
        pa.field("stock_adj_open", pa.float64(), nullable=True),
        pa.field("stock_adj_high", pa.float64(), nullable=True),
        pa.field("stock_adj_low", pa.float64(), nullable=True),
        pa.field("stock_volume", pa.float64(), nullable=True),
        pa.field("dollar_volume", pa.float64(), nullable=True),
        pa.field("dividend_yield", pa.float64(), nullable=True),
        pa.field("stock_beta", pa.float64(), nullable=True),
        pa.field("historical_beta", pa.float64(), nullable=True),
        pa.field("shares_outstanding", pa.float64(), nullable=True),
        pa.field("market_cap", pa.float64(), nullable=True),
    ]

    # Academic controls (float32)
    for col in ACADEMIC_CONTROL_COLUMNS:
        fields.append(pa.field(col, pa.float32(), nullable=True))

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
        fields.append(pa.field(col, pa.float32(), nullable=True))

    # Compact Option Columns (float32)
    for col in COMPACT_OPTION_COLUMNS:
        fields.append(pa.field(col, pa.float32(), nullable=True))

    # Return targets, labels, and activity/investability flags
    bool_set = {
        "jump_label_5d",
        "crash_label_5d",
        "jump_label_21d",
        "crash_label_21d",
        "outperformance_label_5d",
        "outperformance_label_21d",
        "is_target_valid_5d",
        "is_target_valid_21d",
        "is_actively_trading",
        "is_options_investable",
    }
    for col in LABEL_AND_TARGET_COLUMNS:
        if col in bool_set:
            fields.append(pa.field(col, pa.bool_(), nullable=True))
        else:
            fields.append(pa.field(col, pa.float32(), nullable=True))

    return pa.schema(fields)


def cast_and_align_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected columns exist and cast to target dtypes."""
    missing = [col for col in ALL_DATASET_COLUMNS if col not in df.columns]
    bool_cols = {
        "is_sp500",
        "jump_label_5d",
        "crash_label_5d",
        "jump_label_21d",
        "crash_label_21d",
        "outperformance_label_5d",
        "outperformance_label_21d",
        "is_target_valid_5d",
        "is_target_valid_21d",
        "is_actively_trading",
        "is_options_investable",
    }
    if missing:
        missing_dict = {col: False if col in bool_cols else np.nan for col in missing}
        missing_df = pd.DataFrame(missing_dict, index=df.index)
        df = pd.concat([df, missing_df], axis=1)

    df = df[ALL_DATASET_COLUMNS].copy()
    df["trade_date"] = df["trade_date"].astype(str)
    df["underlying"] = df["underlying"].astype(str)
    df["is_sp500"] = df["is_sp500"].fillna(False).astype(bool)
    df["options_available_time"] = df["options_available_time"].astype(str)
    df["tradable_date"] = df["tradable_date"].astype(str)

    # Cast stock columns to float64
    for col in STOCK_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Cast float32 columns
    float32_cols = (
        ACADEMIC_CONTROL_COLUMNS
        + ATM_GREEK_COLUMNS
        + SURFACE_COLUMNS
        + DTE_BUCKET_COLUMNS
        + TERM_AND_COUNT_COLUMNS
        + COMPACT_OPTION_COLUMNS
        + [c for c in LABEL_AND_TARGET_COLUMNS if c not in bool_cols]
    )
    for col in float32_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Cast boolean label columns
    bool_label_cols = [c for c in LABEL_AND_TARGET_COLUMNS if c in bool_cols]
    for col in bool_label_cols:
        df[col] = df[col].fillna(False).astype(bool)

    return df
    for col in bool_label_cols:
        df[col] = df[col].fillna(False).astype(bool)

    return df
