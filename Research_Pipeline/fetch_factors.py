"""Freeze official French monthly FF5 and momentum inputs outside Git."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


SOURCES = {
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "momentum": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
}


def _monthly_table(archive: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        if len(package.namelist()) != 1:
            raise ValueError("expected one CSV in French factor archive")
        lines = package.read(package.namelist()[0]).decode("latin1").splitlines()
    start = next(i for i, line in enumerate(lines) if line.lstrip().startswith(","))
    data = [lines[start]]
    for line in lines[start + 1:]:
        if not re.match(r"^\s*\d{6}\s*,", line):
            break
        data.append(line)
    frame = pd.read_csv(io.StringIO("\n".join(data))).rename(columns={"Unnamed: 0": "period"})
    frame.columns = [str(col).strip() for col in frame.columns]
    frame["period"] = pd.PeriodIndex(frame.period.astype(str), freq="M")
    for col in frame.columns.difference(["period"]):
        values = pd.to_numeric(frame[col], errors="raise")
        if values.isin((-99.99, -999)).any():
            raise ValueError(f"missing coded factor value in {col}")
        frame[col] = values / 100
    return frame


def fetch_factors(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("use a fresh factor directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, manifest = {}, {}
    for name, url in SOURCES.items():
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        (output_dir / f"{name}.zip").write_bytes(data)
        tables[name] = _monthly_table(data)
        manifest[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(),
                          "bytes": len(data)}
    momentum = tables["momentum"].rename(columns={"Mom": "MOM"})
    merged = tables["ff5"].merge(momentum, on="period", validate="one_to_one")
    merged["source_version"] = ";".join(f"{name}:{info['sha256']}" for name, info in manifest.items())
    merged["period"] = merged.period.astype(str)
    merged.to_parquet(output_dir / "monthly_factors.parquet", index=False)
    manifest["first_month"] = str(merged.period.min())
    manifest["last_month"] = str(merged.period.max())
    manifest["rows"] = len(merged)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(fetch_factors(parser.parse_args().output_dir), indent=2))


if __name__ == "__main__":
    main()
