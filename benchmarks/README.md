# Fixed-input calculator benchmark

The [October 28 server measurements](server_2026-10-28.md) compare the original
calculator, IV bracket reuse, the C++ CRR kernel, hard-disk versus RAM storage,
and vectorized OPRA parsing on real flatfiles.

`benchmark_pricing.py` runs the production IV, hypothetical ATM Greek, and
actual-strike Greek row workers on 160 **synthetic** contracts. Its fixture
contains fixed option prices and reference outputs from the imported calculator
at commit `a621095`. The mix is balanced across calls and puts, with and
without a dividend yield. Every run checks IV and all four Greeks against the
reference with an absolute tolerance of `1e-8`.

```bash
.venv/bin/python benchmarks/benchmark_pricing.py --workers 1 4 --repeats 2
```

The script prints JSON lines with wall time and total process CPU time for each
stage. CPU time includes completed child processes. Run it while the machine is
otherwise idle, and compare runs on the same machine and Python environment.

## Local result, October 22, 2026

macOS ARM, Python 3.14.7, 160 rows, two runs per worker count. Values below are
the mean seconds of the two runs. The only pricing change reuses the endpoint
price residuals that `compute_row_iv_task()` already calculated before calling
`compute_iv()`. The pricing model, tree depth, solver, and Greek formulas remain
the same.

| Workers | Version | IV wall | Total wall | Total CPU |
| --- | --- | ---: | ---: | ---: |
| 1 | Imported baseline | 8.030 | 11.377 | 11.331 |
| 1 | Reused IV bracket | 7.244 | 10.479 | 10.441 |
| 4 | Imported baseline | 2.609 | 4.382 | 16.402 |
| 4 | Reused IV bracket | 2.428 | 4.204 | 15.876 |

This was about **7.9% less total CPU time with one worker** and **3.2% less
with four workers** for the complete synthetic calculation. Single-worker IV
wall time fell about 9.8%. All 160 IV values and both sets of four Greeks
matched the saved reference within `1e-8`. A separate eight-row, file-based
smoke run completed the ATM and Default Strike pipelines and wrote both daily
feature files.

This benchmark excludes OPRA file parsing, stock-history loading, downloads,
and full historical feature aggregation. The owner's hours-long runtime still
needs the real input files, server hardware, and original date range for a
faithful comparison. Measure those separately before estimating production
savings or changing worker allocation.

## Optional C++ CRR kernel on `perf/native-crr-pricing`

The native extension compiles the CRR tree's backward loop while leaving the
pricing definitions and the rest of the pipeline in Python. Build and select
it as described in [the native README](../Data_Pipeline/native/README.md). Use
`OPTIONS_CRR_BACKEND=python` for the same branch's Python comparison.

On the same macOS ARM machine and fixed 160-row fixture, two runs per backend
and worker count gave these means in seconds. Both backends matched the saved
IV and Greek reference within `1e-8`.

| Workers | Backend | Total wall | Total process CPU |
| --- | --- | ---: | ---: |
| 1 | Python | 10.701 | 10.518 |
| 1 | C++ | 0.318 | 0.272 |
| 4 | Python | 4.396 | 16.148 |
| 4 | C++ | 1.894 | 6.410 |

At one worker this is a **33.7x wall-time speedup** and **38.7x CPU-time
reduction** for the synthetic row calculations. Four native workers are slower
than one for this small fixture because each IV/Greek stage creates a fresh
process pool. Repeating the fixture ten times (1,600 rows) gave native wall
times of about 2.66 seconds with one worker and 2.26 seconds with four; the
four-worker run consumed about 8.4 seconds of process CPU versus 2.6 seconds
with one worker. These repeated synthetic rows are useful for testing process
overhead, but they are not a full OPRA pipeline benchmark.

An eight-contract file-based smoke run produced identical ATM contract,
ATM feature, Default Strike contract, and Default Strike feature CSVs between
the Python and native backends within `1e-8`. Server results may differ with
real input sizes, I/O, CPU limits, and process startup costs.
