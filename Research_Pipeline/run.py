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
import pyarrow as pa

from .backtest import attach_forward_trade_returns, backtest_all
from .ben_handoff import convert_ben_file
from .data import PAPER_CONTROLS, build_monthly_panel, load_handoff, write_coverage
from .factors import load_monthly_factors, newey_west_alpha, newey_west_paired_mean
from .models import _training_rows, prediction_metrics, train_and_score
from .monte_carlo import monte_carlo_all
from .paper import train_paper
from .paper_descriptives import describe_panel
from .paper_reference import compare_table_six
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


def _limit_cores(max_cores: int) -> list[int] | None:
    if not 1 <= max_cores <= 4:
        raise ValueError("max_cores must be between 1 and 4")
    pa.set_cpu_count(max_cores)
    pa.set_io_thread_count(max_cores)
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        chosen = allowed[:max_cores]
        os.sched_setaffinity(0, chosen)
        return chosen
    return None


def _manifest(input_dir: Path, *, seed: int, iterations: int, pilot: bool,
              synthetic: bool, factor_file: Path | None = None,
              max_cores: int = 4, cpu_affinity: list[int] | None = None) -> dict:
    files = sorted(input_dir.glob("*.parquet"))
    digests = {}
    for file in files:
        sha = hashlib.sha256()
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        digests[file.name] = {"sha256": sha.hexdigest(), "bytes": file.stat().st_size}
    if factor_file is not None:
        sha = hashlib.sha256()
        with Path(factor_file).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        digests[str(factor_file)] = {"sha256": sha.hexdigest(), "bytes": Path(factor_file).stat().st_size}
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
        "max_cpu_cores": max_cores, "cpu_affinity": cpu_affinity,
        "synthetic_input": synthetic,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm")},
    }


def _research_readiness(panel: pd.DataFrame, *, experiment: str, label_policy: str,
                        paper_start: str, paper_end: str) -> tuple[dict, list[str]]:
    """Show the date gate even when other required inputs are also missing."""
    labeled = sorted(panel.loc[panel.label.notna(), "period"].unique())
    report = {"labeled_months": len(labeled)}
    reasons = []
    if experiment == "paper":
        expected = pd.period_range(paper_start, paper_end, freq="M")
        if not len(expected):
            raise ValueError("last paper test month precedes first")
        present = set(labeled)
        test_count = sum(period in present for period in expected)
        report.update({"required_test_months": len(expected), "labeled_paper_test_months": test_count,
                       "first_required_test_month": str(expected[0]),
                       "last_required_test_month": str(expected[-1])})
        if test_count != len(expected):
            reasons.append(f"paper window has {test_count} of {len(expected)} labeled test months")
        first_test = expected[0]
        training = panel.loc[panel.period.lt(first_test) & panel.label.notna()]
        if label_policy == "point_in_time" and first_test in set(panel.period):
            cutoff = panel.loc[panel.period.eq(first_test), "rebalance_close_at"].max()
            training = _training_rows(panel, cutoff)
        train_count = training.period.nunique()
        report["initial_training_months"] = int(train_count)
        report["required_training_months"] = 120
        if train_count < 120:
            reasons.append(f"only {train_count} training months before {first_test}; need 120")
    else:
        report["required_test_months"] = 24
        if len(labeled) < 24:
            reasons.append(f"only {len(labeled)} labeled months; need 24 test months")
            report["initial_matured_training_months"] = None
            if len(labeled) < 120:
                reasons.append(f"only {len(labeled)} labeled months total; need 120 matured training months")
        else:
            first_test = labeled[-24]
            cutoff = panel.loc[panel.period.eq(first_test), "rebalance_close_at"].max()
            train_count = _training_rows(panel, cutoff).period.nunique()
            report.update({"first_test_month": str(first_test),
                           "initial_matured_training_months": int(train_count)})
            if train_count < 120:
                reasons.append(f"only {train_count} matured training months before {first_test}; need 120")
        report["required_matured_training_months"] = 120
    report["ready"] = not reasons
    return report, reasons


def run_pipeline(
    input_dir: Path, output_dir: Path, *, pilot: bool = False,
    synthetic: bool = False, iterations: int = 2_000, seed: int = 17,
    experiment: str = "extended", label_policy: str = "point_in_time",
    paper_start: str = "2006-01", paper_end: str = "2022-12",
    factor_file: Path | None = None,
    pilot_train_months: int = 12, pilot_test_months: int = 6,
    max_cores: int = 4,
) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    cpu_affinity = _limit_cores(max_cores)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        "manifest.json", "coverage.json", "monthly_panel.parquet", "model_selection.json",
        "predictions.parquet", "prediction_metrics.json", "trades.parquet",
        "monthly_returns.parquet", "portfolio_metrics.json", "monte_carlo.json", "status.json",
        "factor_metrics.json", "paper_comparison.json",
        "paper_descriptives.json",
        "paper_baseline_tests.json",
    )
    existing = [name for name in expected_outputs if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"use a fresh output directory; existing results: {existing}")
    tables = load_handoff(input_dir)
    if experiment not in {"extended", "paper"}:
        raise ValueError("experiment must be extended or paper")
    if pilot_train_months < 1 or pilot_test_months < 3:
        raise ValueError("pilot needs at least one training month and three test months")
    synthetic = synthetic or any(
        table.source_version.astype(str).str.contains("synthetic", case=False).any()
        for table in tables.values()
    )
    _json(output_dir / "manifest.json", _manifest(input_dir, seed=seed, iterations=iterations,
                                                  pilot=pilot, synthetic=synthetic,
                                                  factor_file=factor_file,
                                                  max_cores=max_cores, cpu_affinity=cpu_affinity))
    panel, coverage = build_monthly_panel(
        tables, min_snapshot_members=0 if pilot else 450,
        paper_control_lags=experiment == "paper",
        paper_sample_filters=experiment == "paper",
    )
    readiness, readiness_reasons = _research_readiness(
        panel, experiment=experiment, label_policy=label_policy,
        paper_start=paper_start, paper_end=paper_end,
    )
    coverage["research_readiness"] = readiness
    write_coverage(output_dir / "coverage.json", coverage)
    _parquet(output_dir / "monthly_panel.parquet", panel)
    if experiment == "paper":
        _json(output_dir / "paper_descriptives.json", describe_panel(panel))
    if not pilot:
        reasons = list(readiness_reasons)
        if synthetic:
            reasons.append("synthetic source data cannot produce a research backtest")
        if experiment == "paper" and factor_file is None:
            reasons.append("paper research backtest requires FF5 plus momentum factors for alpha")
        if coverage["unknown_us_common_stock_rows"]:
            reasons.append("US common-stock status is missing for some option rows")
        adapter_path = input_dir / "adapter_report.json"
        if adapter_path.exists():
            adapter = json.loads(adapter_path.read_text())
            if adapter.get("identity") == "ticker_proxy_pilot_only":
                reasons.append("ticker proxy is not a stable security identifier")
            if experiment == "paper" and adapter.get("option_price_basis") != "bid_ask_midpoint":
                reasons.append("paper reproduction requires bid/ask midpoint option prices")
        if not panel[PAPER_CONTROLS].notna().any().all():
            reasons.append("one or more paper stock/company controls have no observations")
        if coverage["missing_labels_with_target_month_present"]:
            reasons.append("some stocks lack 12-month outcomes even though the target month exists")
        if reasons:
            status = {"status": "blocked_data_gate", "reason": "; ".join(reasons), "coverage": coverage}
            _json(output_dir / "status.json", status)
            return status
    try:
        if experiment == "paper":
            predictions, model_report = train_paper(
                panel, first_test=paper_start, last_test=paper_end,
                label_policy=label_policy, min_train_months=pilot_train_months if pilot else 120,
                seed=seed,
            )
            if not pilot and model_report["test_months"] < 24:
                raise ValueError("paper research backtest needs at least 24 labeled test months")
            expected_months = len(pd.period_range(paper_start, paper_end, freq="M"))
            if not pilot and model_report["test_months"] != expected_months:
                raise ValueError(f"paper window has {model_report['test_months']} labeled months; "
                                 f"need all {expected_months} months from {paper_start} through {paper_end}")
        else:
            predictions, model_report = train_and_score(
                panel, test_months=pilot_test_months if pilot else 24,
                min_train_months=pilot_train_months if pilot else 120,
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
    if monthly.empty:
        status = {"status": "strategy_insolvent", "synthetic_input": synthetic,
                  "months_scored": model_report["test_months"], "experiment": experiment,
                  "feature_model_combinations": len(model_report["candidate_models"])}
        _json(output_dir / "status.json", status)
        return status
    if experiment == "paper":
        paired_report = {}
        for (kind, universe), group in monthly.groupby(["model", "universe"]):
            baseline = group[group.feature_set.eq("B")]
            if baseline.empty:
                continue
            for specification, candidate in group.groupby("feature_set"):
                if specification == "B":
                    continue
                paired_report[f"{specification}/{kind}/{universe}"] = newey_west_paired_mean(
                    candidate, baseline,
                )
        _json(output_dir / "paper_baseline_tests.json", paired_report)
    if factor_file is not None:
        factors = load_monthly_factors(factor_file)
        factor_report = {}
        for (features, kind, universe), group in monthly.groupby(["feature_set", "model", "universe"]):
            try:
                factor_report[f"{features}/{kind}/{universe}"] = {
                    field: newey_west_alpha(group, factors, return_column=field)
                    for field in ("gross_return", "net_return_0pct_borrow", "net_return_3pct_borrow")
                }
            except ValueError as exc:
                if not pilot:
                    raise
                factor_report[f"{features}/{kind}/{universe}"] = {"status": f"unavailable: {exc}"}
        _json(output_dir / "factor_metrics.json", factor_report)
    if experiment == "paper":
        comparison = compare_table_six(portfolio_report)
        comparison["data_comparable_to_paper"] = False if pilot or synthetic else "not_assessed"
        comparison["paper_best_gross_alpha_monthly_ff5_mom"] = 0.021
        comparison["paper_best_gross_alpha_newey_west_t"] = 4.59
        _json(output_dir / "paper_comparison.json", comparison)
    status = {
        "status": "engineering_pilot_inconclusive" if pilot or synthetic else
                  "paper_replay_hindsight_diagnostic" if experiment == "paper" and label_policy == "paper_stated"
                  else "research_backtest_complete",
        "synthetic_input": synthetic, "months_scored": model_report["test_months"],
        "experiment": experiment, "label_policy": label_policy,
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
    parser.add_argument("--experiment", choices=("extended", "paper"), default="extended")
    parser.add_argument("--label-policy", choices=("point_in_time", "paper_stated"), default="point_in_time")
    parser.add_argument("--paper-start", default="2006-01")
    parser.add_argument("--paper-end", default="2022-12")
    parser.add_argument("--factor-file", type=Path, help="monthly decimal FF5 plus momentum file")
    parser.add_argument("--wide-input", type=Path, help="Ben's ml_dataset_daily.parquet")
    parser.add_argument("--ticker-history", type=Path)
    parser.add_argument("--stocks-dir", type=Path, help="dated stock history CSVs for labels and trade prices")
    parser.add_argument("--allow-ticker-proxy", action="store_true", help="pilot only; not stable identity")
    parser.add_argument("--pilot-train-months", type=int, default=12)
    parser.add_argument("--pilot-test-months", type=int, default=6)
    parser.add_argument("--max-cores", type=int, default=4, help="maximum 1-4 CPU cores")
    args = parser.parse_args()
    _limit_cores(args.max_cores)
    if args.synthetic_demo:
        input_dir = args.output_dir / "synthetic_input"
        write_synthetic_handoff(input_dir, n_stocks=40, n_months=60, seed=args.seed)
    elif args.wide_input is not None:
        input_dir = args.output_dir / "adapted_handoff"
        convert_ben_file(args.wide_input, input_dir, ticker_history=args.ticker_history,
                         allow_ticker_proxy=args.allow_ticker_proxy, stocks_dir=args.stocks_dir)
    elif args.input_dir is None:
        parser.error("--input-dir is required unless --synthetic-demo is set")
    else:
        input_dir = args.input_dir
    iterations = args.iterations if args.iterations is not None else (20 if args.synthetic_demo else 2_000)
    result = run_pipeline(
        input_dir, args.output_dir, pilot=args.engineering_pilot or args.synthetic_demo,
        synthetic=args.synthetic_demo, iterations=iterations, seed=args.seed,
        experiment=args.experiment, label_policy=args.label_policy,
        paper_start=args.paper_start, paper_end=args.paper_end,
        factor_file=args.factor_file,
        pilot_train_months=args.pilot_train_months, pilot_test_months=args.pilot_test_months,
        max_cores=args.max_cores,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] != "blocked_data_gate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
