# Options calculator and research pipeline

Use **`integration/consolidated`** for the complete project. It combines the
calculator and data-preparation work from `feat/data-prep` with the models,
backtests, Monte Carlo checks, diagnostics, sensitivity analysis and reports
from the research development line. [BRANCH_AUDIT.md](BRANCH_AUDIT.md) records
all original branch tips and their recovery tags. `main` and Ben's branches
are retained; they are not the consolidated working branch.

```sh
git clone --branch integration/consolidated https://github.com/AggieInvestmentClub/quant-fund-options-ml.git
cd quant-fund-options-ml
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python Data_Pipeline/native/build.py --model all --jobs 4
.venv/bin/python -m pytest -q Data_Pipeline Dataset_Prep Research_Pipeline
```

Python 3.10+ is required; research dependencies may require a newer supported
Python release. Native builds require a C++17 compiler, Python headers, CMake
and zlib development headers on macOS/Linux. The vendored libdeflate license
is retained under `Data_Pipeline/native/libdeflate/COPYING`. Rebuild extensions
on each machine; binaries and data are not committed.

## Pipeline and starting points

| Stage | Purpose | Entry point |
|---|---|---|
| ATM calculator | Filter local OPRA files; calculate IV and hypothetical ATM Greeks | `Data_Pipeline/flatfile_iv.py` |
| Default Strike | Reuse compatible IV; calculate Greeks at actual strikes | `Default_Strike_Pipeline/flatfile_regular_greeks.py` |
| Surface features | Aggregate delta buckets, skew and term structure | `Default_Strike_Pipeline/regular_features.py` |
| Data preparation | Join dated stock histories and options; export daily Parquet | `Dataset_Prep/export_daily_parquet.py` |
| Research | Paper-method diagnostic, weekly return models and backtests | [Research experiments](Research_Pipeline/EXPERIMENTS.md) |
| Feature sensitivity | Pruning, correlations, importance and sensitivity comparisons | [Data and sensitivity guide](Research_Pipeline/DATA_SENSITIVITY.md) |
| Holding periods and trading rules | 1/2/4-week absolute/relative targets, momentum confirmations and retention buffers | [Horizon strategy guide](Research_Pipeline/HORIZON_STRATEGY.md) |
| Calculator verification | Fixed synthetic pricing workload; no downloads | `benchmark.py` |

The calculator originated in `AggieInvestmentClub/options_updater` at
`834e57563b0d10eb82edc491fe31a2c1e7a92b0a`. OPRA daily aggregates come from
Massive; stock CSVs and rates are externally maintained. Dataset preparation
adds features and forward labels, while the research adapter controls which
columns may enter predictors. Predictive results do not establish that options
cause underlying stock prices to move.

## Select the financial model and implementation

**CRR is the default.** Leisen–Reimer changes the financial approximation and
solver; it is an explicit alternative, not an interchangeable speed toggle.

| Environment setting | Values | Default |
|---|---|---|
| `OPTIONS_PRICING_MODEL` | `crr`, `lr` | `crr` |
| `OPTIONS_CRR_BACKEND` | `auto`, `python`, `native` | `auto` |
| `OPTIONS_LR_BACKEND` | `auto`, `python`, `native` | `auto` |
| `OPTIONS_OPRA_BACKEND` | `auto`, `python`, `native` | `auto` |

A CRR-named function always prices a CRR tree. Generic pipeline functions use
the selected model consistently for IV, ATM and actual-strike Greeks.
`auto` may fall back from native to Python within an implementation; it never
switches the pricing model. Explicit native selection fails if unavailable.
Ingestion is independent: native OPRA decoding can accompany Python CRR.

```sh
OPTIONS_PRICING_MODEL=crr OPTIONS_CRR_BACKEND=native .venv/bin/python benchmark.py --workers 1 4 --repeats 2
OPTIONS_PRICING_MODEL=crr OPTIONS_CRR_BACKEND=python .venv/bin/python benchmark.py --workers 1 --repeats 2
OPTIONS_PRICING_MODEL=lr OPTIONS_LR_BACKEND=native .venv/bin/python benchmark.py --workers 1 4 --repeats 2
```

CRR checks the original fixed reference to `1e-8`. LR checks its own untimed
Python reference on the same workload; it is not required to equal CRR. The
legacy LR fixture is retained separately with provenance clarified. These
small synthetic timings do not establish a full-history production speedup.

## Offline calculation and dataset preparation

Use a **fresh scratch output root per pricing model**. New contract CSVs have
`.pricing.json` sidecars recording model definition and file identity.
Unversioned, modified or different-model contract checkpoints are not reused.
Historical source files are not relabeled; choose fresh roots or explicitly
recalculate them. Existing frozen research inputs remain unchanged.

Set `OPTIONS_CLEAN_STOCK_ROOT`, `OPTIONS_RISK_FREE_RATE_CSV` and the input roots
to existing local data. Supply `--workers 1` through `--workers 4`; defaults
are four for calculator/export workers. Do not spawn nested file and row pools.

```sh
.venv/bin/python Data_Pipeline/flatfile_iv.py --help
.venv/bin/python Default_Strike_Pipeline/flatfile_regular_greeks.py --help
.venv/bin/python Default_Strike_Pipeline/regular_features.py --help
.venv/bin/python Dataset_Prep/export_daily_parquet.py --help
```

Server defaults are `/srv/data/options_model_features` and `/srv/data/stocks`;
a clone does not provide data or credentials. See the
[data export guide](Dataset_Prep/USER_GUIDE.md) and
[full benchmark handoff](FULL_PIPELINE_BENCHMARK_GUIDE.md). Historical coverage
figures in branch-era documents describe their recorded snapshots, not a
current completeness guarantee. The full-history repeated disk/RAM matrix in
the handoff remains separate from the synthetic consolidation checks.

Downloads and scheduled wrappers are optional operational tools. Install their
provider packages with `pip install -r Data_Pipeline/requirements.txt` and
configure credentials outside Git. Consolidation does not install cron jobs,
refresh market data or change server production outputs.

## Modeling and reproducibility

The monthly mode retains 12-month jump/crash definitions as a limited
paper-method diagnostic. The weekly mode predicts signed entry-to-exit returns.
Research code preserves strict chronological label maturity, training-only
preprocessing, missing-price failures and the original accounting rules.
Forward targets and target-validity flags are outcome metadata, never a
historical portfolio-selection filter. Prepared fallback fields are not silently
substituted for exact-maturity research features.

See [research setup](Research_Pipeline/README.md),
[portfolio accounting](Research_Pipeline/PORTFOLIO_REVISION.md), and
[feature experiment](Research_Pipeline/FEATURE_EXPERIMENT.md). Private artifacts
record source hashes, code commits, feature schemas and split dates. Preserve
those versions when reproducing an experiment; a branch name alone is not a
frozen specification. Data, fitted models and derived research reports remain
outside the public checkout. Short history, unresolved identities/corporate
actions and missing selected prices continue to limit investment conclusions.
