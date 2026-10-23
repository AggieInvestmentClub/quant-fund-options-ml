"""Run a fixed, synthetic pricing workload through the production row workers.

The fixture records outputs from the imported calculator at commit a621095.
No market data or server access is required. This measures calculation time,
not OPRA parsing, network downloads, or feature CSV writes.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Data_Pipeline"))
sys.path.insert(0, str(ROOT / "Default_Strike_Pipeline"))

from flatfile_iv import (  # noqa: E402
    compute_row_atm_greeks_task,
    compute_row_iv_task,
    run_batched_tasks,
)
from flatfile_regular_greeks import compute_row_regular_greeks_task  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_160_reference.json"
GREEK_KEYS = ("delta", "gamma", "theta", "vega")


def cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def run_stage(tasks: list[tuple], workers: int, label: str, worker_fn):
    start_cpu = cpu_seconds()
    start_wall = time.perf_counter()
    with redirect_stdout(StringIO()):
        results = run_batched_tasks(tasks, workers, 5000, label, date(2026, 10, 20), worker_fn)
    return results, time.perf_counter() - start_wall, cpu_seconds() - start_cpu


def check_iv(actual: list[tuple], expected: list[list]) -> None:
    for index, ((value, reason), (reference, reference_reason)) in enumerate(zip(actual, expected, strict=True)):
        if reason != reference_reason or not math.isclose(value, reference, rel_tol=0.0, abs_tol=1e-8):
            raise AssertionError(f"IV mismatch at row {index}: {(value, reason)} != {(reference, reference_reason)}")


def check_greeks(actual: list[dict], expected: list[dict], label: str) -> None:
    for index, (result, reference) in enumerate(zip(actual, expected, strict=True)):
        for key in GREEK_KEYS:
            if not math.isclose(result[key], reference[key], rel_tol=0.0, abs_tol=1e-8):
                raise AssertionError(f"{label} mismatch at row {index}, {key}: {result[key]} != {reference[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or any(workers < 1 for workers in args.workers):
        parser.error("workers and repeats must be positive")

    fixture = json.loads(FIXTURE.read_text())
    iv_tasks = [tuple(row) for row in fixture["iv_tasks"]]
    print(json.dumps({
        "type": "setup", "rows": len(iv_tasks), "workers": args.workers,
        "repeats": args.repeats, "python": platform.python_version(),
        "platform": platform.platform(), "fixture": str(FIXTURE),
    }), flush=True)

    for workers in args.workers:
        for repeat in range(1, args.repeats + 1):
            iv, iv_wall, iv_cpu = run_stage(iv_tasks, workers, "IV", compute_row_iv_task)
            check_iv(iv, fixture["iv_results"])
            atm_tasks = [
                (result[0], spot, tenor, rate, q, side)
                for (spot, strike, close, tenor, rate, q, side), result in zip(iv_tasks, iv, strict=True)
            ]
            regular_tasks = [
                (result[0], spot, strike, tenor, rate, q, side)
                for (spot, strike, close, tenor, rate, q, side), result in zip(iv_tasks, iv, strict=True)
            ]
            atm, atm_wall, atm_cpu = run_stage(atm_tasks, workers, "ATM Greeks", compute_row_atm_greeks_task)
            check_greeks(atm, fixture["atm_greeks"], "ATM Greek")
            regular, regular_wall, regular_cpu = run_stage(
                regular_tasks, workers, "Regular Greeks", compute_row_regular_greeks_task
            )
            check_greeks(regular, fixture["regular_greeks"], "Regular Greek")
            print(json.dumps({
                "type": "run", "workers": workers, "repeat": repeat, "rows": len(iv_tasks),
                "iv_wall_s": round(iv_wall, 3), "iv_cpu_s": round(iv_cpu, 3),
                "atm_wall_s": round(atm_wall, 3), "atm_cpu_s": round(atm_cpu, 3),
                "regular_wall_s": round(regular_wall, 3), "regular_cpu_s": round(regular_cpu, 3),
                "total_wall_s": round(iv_wall + atm_wall + regular_wall, 3),
                "total_cpu_s": round(iv_cpu + atm_cpu + regular_cpu, 3),
                "reference_match": True,
            }), flush=True)


if __name__ == "__main__":
    main()
