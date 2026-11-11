"""Configuration paths and defaults for the ML/backtesting dataset prep stage."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DATA_ROOT = Path(
    os.environ.get(
        "OPTIONS_DATASET_ROOT",
        os.environ.get("OPTIONS_MODEL_FEATURES_ROOT", "/srv/data/options_model_features"),
    )
).expanduser()

ATM_FEATURES_ROOT = Path(
    os.environ.get(
        "OPTIONS_ATM_FEATURES_ROOT",
        str(FEATURE_DATA_ROOT / "ATM_Normalized_Options" / "features"),
    )
).expanduser()

DEFAULT_STRIKE_FEATURES_ROOT = Path(
    os.environ.get(
        "OPTIONS_DEFAULT_STRIKE_FEATURES_ROOT",
        str(FEATURE_DATA_ROOT / "Default_Strike" / "Features"),
    )
).expanduser()

CLEAN_STOCK_ROOT = Path(
    os.environ.get(
        "OPTIONS_CLEAN_STOCK_ROOT",
        os.environ.get("OPTIONS_MODEL_CLEAN_STOCK_ROOT", "/srv/data/stocks"),
    )
).expanduser()

CONSTITUENCY_ROOT = Path(
    os.environ.get(
        "OPTIONS_CONSTITUENCY_ROOT",
        os.environ.get("OPTIONS_MODEL_CONSTITUENCY_ROOT", str(FEATURE_DATA_ROOT / "Constituency")),
    )
).expanduser()

FLATFILES_ROOT = Path(
    os.environ.get(
        "OPTIONS_FLATFILES_ROOT",
        str(FEATURE_DATA_ROOT / "FlatFiles" / "us_options_opra" / "day_aggs_v1"),
    )
).expanduser()

PARQUET_OUTPUT_DIR = Path(
    os.environ.get(
        "OPTIONS_PARQUET_OUTPUT_DIR",
        str(FEATURE_DATA_ROOT / "ml_dataset_daily"),
    )
).expanduser()

PARQUET_SINGLE_FILE = Path(
    os.environ.get(
        "OPTIONS_PARQUET_FILE",
        str(FEATURE_DATA_ROOT / "ml_dataset_daily.parquet"),
    )
).expanduser()
