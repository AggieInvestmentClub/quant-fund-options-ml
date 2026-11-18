#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

echo "=== [$(date -u --iso-8601=seconds)] Daily Pipeline Started ==="

echo "--- Stage 1/3: ATM Normalized Options Pipeline ---"
if ! "$REPO_ROOT/data_copy/run_main_to_srv.sh"; then
  echo "[ERROR] Stage 1 failed! See details in: $REPO_ROOT/logs/main_pipeline_latest.log" >&2
  exit 1
fi
echo "[SUCCESS] Stage 1 completed."

echo "--- Stage 2/3: Default Strike Pipeline ---"
if ! "$SCRIPT_DIR/run_default_strike_daily.sh"; then
  echo "[ERROR] Stage 2 failed! See details in: $REPO_ROOT/logs/default_strike_latest.log" >&2
  exit 1
fi
echo "[SUCCESS] Stage 2 completed."

echo "--- Stage 3/3: Daily ML Parquet Export ---"
if ! "$REPO_ROOT/Dataset_Prep/run_export_daily.sh"; then
  echo "[ERROR] Stage 3 failed! See details in: $REPO_ROOT/logs/export_parquet_latest.log" >&2
  exit 1
fi
echo "[SUCCESS] Stage 3 completed."

echo "=== [$(date -u --iso-8601=seconds)] Daily Pipeline Completed Successfully ==="
