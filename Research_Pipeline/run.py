"""Run the dated research experiment from Parquet handoff to reports."""

from __future__ import annotations

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import pandas as pd

from .backtest import attach_forward_trade_returns, backtest_all
from .data import PAPER_CONTROLS, build_monthly_panel, load_handoff, write_coverage
from .models import prediction_metrics, train_and_score
from .monte_carlo import monte_carlo_all
from .synthetic import write_synthetic_handoff


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    if "period" in output:
        output["period"] = output.period.astype(str)
    if "target_period" in output:
        output["target_period"] = output.target_period.astype(str)
    output.to_parquet(path, index=False)


def _manifest(input_dir: Path, *, seed: int, iterations: int, pilot: bool, synthetic: bool) -> dict:
    files = sorted(input_dir.glob("*.parquet"))
    digests = {}
    for file in files:
        sha = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        digests[file.name] = {"sha256": sha.hexdigest(), "bytes": file.stat().st_size}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "inputs": digests, "git_commit": commit, "seed": seed,
        "monte_carlo_iterations": iterations, "engineering_pilot": pilot,
        "synthetic_input": synthetic,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm")},
    }


def run_pipeline(
    input_dir: Path, output_dir: Path, *, pilot: bool = False,
    synthetic: bool = False, iterations: int = 2_000, seed: int = 17,
) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        "manifest.json", "coverage.json", "monthly_panel.parquet", "model_selection.json",
        "predictions.parquet", "prediction_metrics.json", "trades.parquet",
        "monthly_returns.parquet", "portfolio_metrics.json", "monte_carlo.json", "status.json",
    )
    existing = [name for name in expected_outputs if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"use a fresh output directory; existing results: {existing}")
    tables = load_handoff(input_dir)
    synthetic = synthetic or any(
        table.source_version.astype(str).str.contains("synthetic", case=False).any()
        for table in tables.values()
    )
    _json(output_dir / "manifest.json", _manifest(input_dir, seed=seed, iterations=iterations, pilot=pilot, synthetic=synthetic))
    panel, coverage = build_monthly_panel(tables, min_snapshot_members=0 if pilot else 450)
    write_coverage(output_dir / "coverage.json", coverage)
    _parquet(output_dir / "monthly_panel.parquet", panel)
    if not pilot:
        reasons = []
        if synthetic:
            reasons.append("synthetic source data cannot produce a research backtest")
        if coverage["unknown_us_common_stock_rows"]:
            reasons.append("US common-stock status is missing for some option rows")
        if not panel[PAPER_CONTROLS].notna().any().all():
            reasons.append("one or more paper stock/company controls have no observations")
        if coverage["missing_labels_with_target_month_present"]:
            reasons.append("some stocks lack 12-month outcomes even though the target month exists")
        if reasons:
            status = {"status": "blocked_data_gate", "reason": "; ".join(reasons), "coverage": coverage}
            _json(output_dir / "status.json", status)
            return status
    try:
        predictions, model_report = train_and_score(
            panel, test_months=6 if pilot else 24,
            min_train_months=12 if pilot else 120,
            validation_months=3 if pilot else 24, seed=seed,
        )
    except ValueError as exc:
        status = {"status": "blocked_data_gate", "reason": str(exc), "coverage": coverage}
        _json(output_dir / "status.json", status)
        return status
    _parquet(output_dir / "predictions.parquet", predictions)
    if not pilot and predictions.label.isna().any():
        status = {"status": "blocked_data_gate", "reason": "test predictions have missing 12-month outcomes"}
        _json(output_dir / "status.json", status)
        return status
    try:
        trades, monthly, portfolio_report = backtest_all(predictions, tables["stock_prices"])
    except ValueError as exc:
        status = {"status": "blocked_data_gate", "reason": str(exc), "coverage": coverage}
        _json(output_dir / "status.json", status)
        return status
    _parquet(output_dir / "trades.parquet", trades)
    _parquet(output_dir / "monthly_returns.parquet", monthly)
    attached = attach_forward_trade_returns(predictions, tables["stock_prices"])
    mc = monte_carlo_all(attached, iterations=iterations, seed=seed)
    _json(output_dir / "model_selection.json", model_report)
    _json(output_dir / "prediction_metrics.json", prediction_metrics(predictions))
    _json(output_dir / "portfolio_metrics.json", portfolio_report)
    _json(output_dir / "monte_carlo.json", mc)
    status = {
        "status": "engineering_pilot_inconclusive" if pilot or synthetic else "research_backtest_complete",
        "synthetic_input": synthetic, "months_scored": model_report["test_months"],
        "feature_model_combinations": len(model_report["candidate_models"]),
        "monte_carlo_iterations": iterations,
    }
    _json(output_dir / "status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engineering-pilot", action="store_true", help="short-history software demonstration, never a research result")
    parser.add_argument("--synthetic-demo", action="store_true", help="generate synthetic handoff and run an engineering pilot")
    parser.add_argument("--iterations", type=int, help="Monte Carlo iterations (default 2000, synthetic demo 20)")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.synthetic_demo:
        input_dir = args.output_dir / "synthetic_input"
        write_synthetic_handoff(input_dir, n_stocks=40, n_months=60, seed=args.seed)
    elif args.input_dir is None:
        parser.error("--input-dir is required unless --synthetic-demo is set")
    else:
        input_dir = args.input_dir
    iterations = args.iterations if args.iterations is not None else (20 if args.synthetic_demo else 2_000)
    result = run_pipeline(
        input_dir, args.output_dir, pilot=args.engineering_pilot or args.synthetic_demo,
        synthetic=args.synthetic_demo, iterations=iterations, seed=args.seed,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] != "blocked_data_gate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
