# Options calculator: parallel computing work

This repository is the working copy of the options data calculator from
[`AggieInvestmentClub/options_updater`](https://github.com/AggieInvestmentClub/options_updater/tree/834e57563b0d10eb82edc491fe31a2c1e7a92b0a)
at source commit `834e57563b0d10eb82edc491fe31a2c1e7a92b0a`. The immediate goal is to
make its CPU-heavy implied-volatility (IV) and Greek calculations faster while
preserving the results. Modeling the relationship between option data and future
underlying-stock returns is a later project phase. There is no neural network or
training pipeline here yet.

## What was imported

| Path | Role |
| --- | --- |
| `Data_Pipeline/utils.py` | American-option Leisen-Reimer tree pricing, IV inversion, and hypothetical at-the-money (ATM) Greeks. |
| `Data_Pipeline/flatfile_iv.py` | Parse OPRA daily aggregates, filter contracts, join underlying prices and rates, run IV/ATM Greek calculations, and export features. |
| `Data_Pipeline/main.py` | Orchestrate constituency snapshots, flatfile downloads, and ATM calculations. |
| `Data_Pipeline/flat_file.py` | Download missing Massive OPRA flatfiles. |
| `Data_Pipeline/interest_rate_getter.py`, `config.py` | Load DGS3MO rates and basic configuration. |
| `Data_Pipeline/test_iv_quality_regressions.py`, `migrate_repaired_iv_rows.py` | Existing pricing regression tests and their helper. |
| `Default_Strike_Pipeline/` | Reuse ATM-stage IV to compute Greeks at actual contract strikes and aggregate strike-specific features. |
| `data_copy/run_main_to_srv.sh` | Existing server-oriented main pipeline wrapper. |

The imported calculator logic is unchanged. The two credential defaults in
`config.py` and `flat_file.py` were replaced with environment lookups so this
repository does not copy provider credentials. `flat_file.py` reads them at
download time, after `main.py` has loaded `.env` if present. No market datasets
were copied.

## Current data flow

1. LSEG provides dated S&P 500 constituency snapshots when configured; existing
   snapshot CSVs are used otherwise.
2. Massive provides daily OPRA aggregate `.csv.gz` files. The current main
   pipeline downloads missing files one date at a time.
3. Externally maintained, split-safe stock histories provide underlying closes
   and dividend yields. The stock download code in `main.py` is currently
   disabled. DGS3MO risk-free rates come from a local CSV, with a FRED API
   fallback when that file has no observations.
4. The ATM stage keeps eligible 30–90 day contracts, solves an IV for each
   valid row, computes hypothetical ATM Greeks, and produces stock/date features.
5. The optional Default Strike stage reuses those IVs, computes Greeks at each
   actual strike, and produces delta-bucket, skew, term-structure, and gamma
   features.
6. The Dataset Prep stage (`Dataset_Prep/`) merges the ATM Greek averages (10 call/put averages),
   Default Strike features (30/60/90-day skew, risk-reversal, and gamma metrics),
   underlying stock OHLCV / adjusted prices / volume, and point-in-time S&P 500
   constituency flags into a high-performance daily Parquet table (`ml_dataset_daily/`
   and `ml_dataset_daily.parquet`).

The server wrappers default to `/srv/data/options_model_features` for pipeline
data and `/srv/data/stocks` for stock histories. Generated data, `.env`, and
logs are ignored by Git. These paths and provider access are prerequisites for
the full server run; cloning this repository alone does not provide them.

## Local setup and a small baseline run

Use Python 3.10 or newer with the imported dependencies, then add
`pytest` for the existing regression tests:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r Data_Pipeline/requirements.txt pytest
.venv/bin/python -m pytest -q Data_Pipeline/test_iv_quality_regressions.py
```

For a calculator benchmark, supply a **local copy** of a representative OPRA
flatfile, matching stock histories, and a DGS3MO CSV. Write every output to a
scratch directory, not the production `/srv` dataset. Replace the example
paths below with real paths on the machine doing the measurement:

```bash
export OPTIONS_CLEAN_STOCK_ROOT=/path/to/stock-histories
export OPTIONS_RISK_FREE_RATE_CSV=/path/to/DGS3MO_risk_free_rate.csv

.venv/bin/python Data_Pipeline/flatfile_iv.py \
  --input-root /path/to/opra/day_aggs_v1 \
  --output-root /path/to/scratch/atm-contracts \
  --daily-features-root /path/to/scratch/atm-features \
  --cache-output /path/to/scratch/underlying-close-cache.csv \
  --final-features-dir /path/to/scratch/final-features \
  --start-date YYYY-MM-DD --end-date YYYY-MM-DD \
  --workers 1
```

### ML Dataset Preparation & Validation (Stage 3)

The Stage 3 pipeline consolidates daily underlying stock OHLCV, market cap, shares outstanding, academic company controls (*Andreou, Han, and Li 2025*), ATM option averages, full volatility surface smile/skew metrics, compact option signals, and multi-horizon forward targets into high-performance Parquet tables.

Export the consolidated daily Parquet dataset across all historical trade dates:

```bash
.venv/bin/python Dataset_Prep/export_daily_parquet.py --build-single-file
```

Validate the exported Parquet dataset and print the feature availability & missingness matrix:

```bash
.venv/bin/python Dataset_Prep/audit_and_validate.py
```

#### Pulling Options-Investable Data for ML & Backtesting
The dataset provides a precomputed boolean flag, `is_options_investable`, that identifies liquid, institutional S&P 500 stocks with actively traded option surfaces (`stock_close >= $5.00`, `market_cap >= $2.0B`, `stock_volume >= 100k`, `n_contracts_total >= 10`, and valid IV). 

Filtering on `is_options_investable == True` eliminates micro-cap noise, stale odd-lot trades, and empty contract chains, boosting surface skew and volatility feature completeness to **>93%–100%** across **231,152 observations (~353 stocks/day)** without sacrificing statistical sample size:

```python
import pandas as pd

# Load options-investable universe directly with Parquet pushdown filtering
df = pd.read_parquet(
    "/srv/data/options_model_features/ml_dataset_daily.parquet",
    filters=[("is_options_investable", "==", True)],
)
```

Or using the built-in filtering helper:

```python
from Dataset_Prep.enrich_features import filter_investable_universe

df_investable = filter_investable_universe(df)
```

For detailed instructions on using the data for ML modeling (target return calculations, label construction, missingness handling), algorithmic backtesting (zero-lookahead execution rules, unadjusted fill prices), and managing the daily cron operations, see the [ML & Backtesting User Guide](Dataset_Prep/USER_GUIDE.md).

The daily automated runner `Default_Strike_Pipeline/run_daily_pipeline.sh` automatically runs
Stages 1, 2, and 3 sequentially during the morning cron job.

The full `Data_Pipeline/main.py` workflow additionally needs Massive access
(`MASSIVE_S3_ACCESS_KEY` and `MASSIVE_API_KEY`) and either LSEG access or
existing constituency snapshots. The FRED fallback needs `FRED_API_KEY`.
Keep all credentials outside Git. The shell wrappers expect Linux utilities
including `flock` and `date --iso-8601`.

## Measured CPU change

The IV row worker now passes its already computed volatility-bracket residuals to
the IV solver, avoiding two repeated price evaluations per eligible contract.
The [fixed-input benchmark](benchmarks/README.md) records the local before and
after CPU and wall times and checks IV and both kinds of Greeks against the
imported calculator. This synthetic result does not establish the runtime of a
full OPRA history on the server.

An [optional C++ Leisen-Reimer kernel](Data_Pipeline/native/README.md) accelerates
the American-option tree calculation while preserving the Python pipeline and
NumPy fallback. The LR parametrisation (Peizer-Pratt Method 2) centres the
lattice around the strike, eliminating the CRR sawtooth oscillation and
allowing a single 101-step tree to replace the former 201+202-step pair.
The [benchmark comparison](benchmarks/README.md) records the synthetic speedup.

## First work package: make the calculator faster

### 1. Establish a reproducible baseline

- Select one representative single-day file and a multi-day catch-up set.
  Record input sizes, contract counts after each filter, hardware, CPU quota,
  Python and package versions, and whether the filesystem cache is warm.
- Run the same scratch workload with 1, 2, 4, and 8 workers, only where the
  machine has that many available cores. Measure wall time, CPU time, peak
  memory, IV rows per second, and output file sizes. Repeat runs to separate
  normal variance from actual gains.
- Profile the CPU path and the CSV read/write path separately. Keep the raw
  logs and a compact results table so later changes can be compared against
  the same baseline. Do not benchmark a live data download as if it were
  calculator time.

### 2. Optimize the measured hot path

- The IV row worker now reuses its bracket checks inside `utils.compute_iv()`.
  Evaluate any further solver changes against the saved failure reasons and
  numerical reference before adopting them.
- Profile the American-option Leisen-Reimer tree and finite-difference Greek calls.
  Preserve the pricing model, tree-depth rules, tolerances, and distinction
  between hypothetical ATM and actual-strike Greeks. A faster calculation
  that changes the financial definition is a separate experiment.
- Inspect process startup, per-row task serialization, chunk sizes, and memory
  pressure. The current code uses file-level workers for multiple dates and
  row-level workers for one date; avoid spawning nested pools. Try bounded
  batches and shared read-only inputs where profiling shows a benefit.
- Avoid rebuilding historical final feature CSVs or repeatedly parsing stock
  histories when only one date changed, if doing so can be made restart-safe.
  Benchmark I/O changes separately from CPU changes.
- Profile and optimize the Default Strike stage after the ATM stage is stable;
  it reuses IV but still performs multiple contract-level price evaluations.

### 3. Prove the output is still usable

- Keep the existing regression tests passing, then add a small fixed-input
  comparison against the imported baseline for IV, failure reasons, all four
  Greeks, contract counts, output schemas, and daily aggregates.
- Compare floating-point outputs with documented tolerances and compare
  membership, dates, and row keys exactly. Include calls, puts, dividends,
  near-intrinsic prices, split dates, missing stock prices, and missing rates.
- Demonstrate a measured speedup on the same hardware and inputs without an
  unacceptable rise in peak memory. Report the full worker-count curve;
  parallel overhead may make small files slower.

## Later research phase

After the calculator is fast and validated, build a dated options-feature
dataset and study whether option signals **predict** subsequent underlying
returns or volatility. Define the target horizon, stock universe, timestamp
rules, and train/validation/test periods before choosing neural networks or
other models. Use time-based evaluation and leakage checks. Predictive results
alone would not establish that option prices *cause* underlying prices to move.
