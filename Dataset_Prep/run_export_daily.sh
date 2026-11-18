#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${OPTIONS_DATASET_ROOT:-/srv/data/options_model_features}"
LOG_DIR="${OPTIONS_CRON_LOG_DIR:-$REPO_ROOT/logs}"
LOCK_FILE="${EXPORT_PARQUET_LOCK_FILE:-$DATA_ROOT/export_parquet_pipeline.lock}"

export OPTIONS_DATASET_ROOT="$DATA_ROOT"
export OPTIONS_PARQUET_OUTPUT_DIR="${OPTIONS_PARQUET_OUTPUT_DIR:-$DATA_ROOT/ml_dataset_daily}"
export OPTIONS_PARQUET_FILE="${OPTIONS_PARQUET_FILE:-$DATA_ROOT/ml_dataset_daily.parquet}"

mkdir -p "$LOG_DIR" "$OPTIONS_PARQUET_OUTPUT_DIR"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/.venv/bin/python"
else
  echo "Missing project virtualenv: $REPO_ROOT/.venv" >&2
  exit 127
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="$LOG_DIR/export_parquet_$timestamp.log"
latest_log="$LOG_DIR/export_parquet_latest.log"
ln -sfn "$log_file" "$latest_log"

{
  echo "Started Daily ML Parquet Export at $(date -u --iso-8601=seconds)"
  echo "Output Directory: $OPTIONS_PARQUET_OUTPUT_DIR"
  echo "Consolidated File: $OPTIONS_PARQUET_FILE"
  cd "$REPO_ROOT"
  flock -n "$LOCK_FILE" "$PYTHON" Dataset_Prep/export_daily_parquet.py --build-single-file "$@"
  echo "Finished Daily ML Parquet Export at $(date -u --iso-8601=seconds)"
} >>"$log_file" 2>&1
