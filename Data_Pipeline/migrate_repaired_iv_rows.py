#!/usr/bin/env python3
"""Migrate repaired IV rows from regular-Greek backups without re-solving IV."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

import flatfile_iv


BACKUP_ROOT = Path(
    "/srv/data/options_model_features/Default_Strike_repair_backup/contracts"
)
CONTRACT_ROOT = Path(
    "/srv/data/options_model_features/ATM_Normalized_Options/contracts"
)
FEATURE_ROOT = Path(
    "/srv/data/options_model_features/ATM_Normalized_Options/features"
)
ATM_GREEK_COLUMNS = {
    "atm_delta": "delta",
    "atm_gamma": "gamma",
    "atm_theta": "theta",
    "atm_vega": "vega",
}


@dataclass(frozen=True)
class MigrationResult:
    trade_date: str
    old_rows: int
    replacement_rows: int
    projected_rows: int
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, default=BACKUP_ROOT)
    parser.add_argument("--contracts-root", type=Path, default=CONTRACT_ROOT)
    parser.add_argument("--features-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def trade_date_from_path(path: Path) -> date:
    return date.fromisoformat(path.stem)


def selected_files(
    root: Path,
    start_date: date | None,
    end_date: date | None,
) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(root.glob("*/*/*.csv")):
        trade_date = trade_date_from_path(path)
        if start_date is not None and trade_date < start_date:
            continue
        if end_date is not None and trade_date > end_date:
            continue
        selected.append(path)
    return selected


def normalized_underlying(frame: pd.DataFrame) -> pd.Series:
    return frame["underlying"].astype("string").str.strip().str.upper()


def add_atm_greeks(
    replacement: pd.DataFrame,
    workers: int,
    trade_date: date,
) -> pd.DataFrame:
    """Add ATM Greeks while preserving the output schema for an empty day."""
    replacement = replacement.copy()
    if replacement.empty:
        for output_column in ATM_GREEK_COLUMNS:
            replacement[output_column] = pd.Series(
                index=replacement.index,
                dtype="float64",
            )
        return replacement

    greek_tasks: list[flatfile_iv.GreekTask] = list(
        zip(
            replacement["implied_volatility"].tolist(),
            replacement["underlying_close"].tolist(),
            replacement["time_to_expiry_years"].tolist(),
            replacement["risk_free_rate"].tolist(),
            replacement[flatfile_iv.DIVIDEND_YIELD_FIELD].tolist(),
            replacement["option_type"].tolist(),
        )
    )
    greek_values = flatfile_iv.run_batched_tasks(
        greek_tasks,
        workers,
        5000,
        "migrated ATM Greeks",
        trade_date,
        flatfile_iv.compute_row_atm_greeks_task,
    )
    greek_frame = pd.DataFrame(greek_values, index=replacement.index)
    for output_column, result_column in ATM_GREEK_COLUMNS.items():
        replacement[output_column] = greek_frame[result_column]
    return replacement


def write_report_atomic(results: list[MigrationResult], report_path: Path) -> None:
    """Checkpoint migration results without leaving a partial report file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_name(f".{report_path.name}.tmp")
    pd.DataFrame([asdict(result) for result in results]).to_csv(
        temporary_path,
        index=False,
    )
    temporary_path.replace(report_path)


def migrate_file(
    current_path: Path,
    contracts_root: Path,
    features_root: Path,
    backup_root: Path,
    symbol: str,
    workers: int,
    apply: bool,
) -> MigrationResult | None:
    trade_date = trade_date_from_path(current_path)
    current = pd.read_csv(current_path)
    current_symbol = normalized_underlying(current).eq(symbol)
    old_rows = int(current_symbol.sum())
    if old_rows == 0:
        return None

    relative = current_path.relative_to(contracts_root)
    backup_path = backup_root / relative
    if not backup_path.exists():
        return MigrationResult(
            trade_date=trade_date.isoformat(),
            old_rows=old_rows,
            replacement_rows=0,
            projected_rows=len(current),
            status="missing_backup",
        )

    backup = pd.read_csv(backup_path)
    backup_dte = pd.to_numeric(backup["dte_days"], errors="coerce")
    replacement = backup.loc[
        normalized_underlying(backup).eq(symbol) & backup_dte.between(30, 90)
    ].copy()
    required_core = flatfile_iv.REQUIRED_IV_OUTPUT_COLS - {
        "atm_delta",
        "atm_gamma",
        "atm_theta",
        "atm_vega",
    }
    missing_core = sorted(required_core - set(replacement.columns))
    if missing_core:
        raise ValueError(f"{backup_path} is missing IV columns: {missing_core}")
    valid_core = replacement[
        [
            "underlying_close",
            "strike",
            "time_to_expiry_years",
            "risk_free_rate",
            "implied_volatility",
        ]
    ].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    if not valid_core.all():
        raise ValueError(
            f"{backup_path} has {int((~valid_core).sum())} invalid replacement rows"
        )

    if apply:
        replacement = add_atm_greeks(replacement, workers, trade_date)
        valid_greeks = replacement[
            ["atm_delta", "atm_gamma", "atm_theta", "atm_vega"]
        ].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        if not valid_greeks.all():
            raise ValueError(
                f"{backup_path} produced {int((~valid_greeks).sum())} invalid ATM Greek rows"
            )

        missing_output = sorted(set(current.columns) - set(replacement.columns))
        if missing_output:
            raise ValueError(
                f"{backup_path} cannot populate current columns: {missing_output}"
            )
        replacement = replacement[list(current.columns)].copy()

    remaining = current.loc[~current_symbol].copy()
    projected_rows = len(remaining) + len(replacement)
    if apply:
        output = pd.concat([remaining, replacement], ignore_index=True, sort=False)
        if output.duplicated(subset=["trade_date", "ticker"]).any():
            raise ValueError(f"Duplicate contracts after migrating {current_path}")
        sort_columns = [
            column
            for column in [
                "trade_date",
                "underlying",
                "expiration_date",
                "option_type",
                "strike",
            ]
            if column in output.columns
        ]
        if sort_columns:
            output = output.sort_values(sort_columns).reset_index(drop=True)
        flatfile_iv.write_output_csv_atomic(output, current_path)
        flatfile_iv.export_daily_feature_file(
            current_path,
            features_root,
            contracts_root,
            reuse_existing=False,
        )

    return MigrationResult(
        trade_date=trade_date.isoformat(),
        old_rows=old_rows,
        replacement_rows=len(replacement),
        projected_rows=projected_rows,
        status="migrated" if apply else "ready",
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.start_date and args.end_date and args.start_date > args.end_date:
        raise ValueError("--start-date cannot be after --end-date")
    symbol = flatfile_iv.normalize_underlying_symbol(args.symbol)
    if not symbol:
        raise ValueError("--symbol must be a supported underlying")

    results: list[MigrationResult] = []
    for path in selected_files(args.contracts_root, args.start_date, args.end_date):
        result = migrate_file(
            path,
            args.contracts_root,
            args.features_root,
            args.backup_root,
            symbol,
            args.workers,
            args.apply,
        )
        if result is None:
            continue
        results.append(result)
        if args.report is not None:
            write_report_atomic(results, args.report)
    print(
        f"mode={'APPLY' if args.apply else 'DRY RUN'} "
        f"files={len(results)} "
        f"ready={sum(result.status in {'ready', 'migrated'} for result in results)} "
        f"missing_backup={sum(result.status == 'missing_backup' for result in results)} "
        f"old_rows={sum(result.old_rows for result in results)} "
        f"replacement_rows={sum(result.replacement_rows for result in results)}"
    )


if __name__ == "__main__":
    main()
