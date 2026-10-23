# Optional C++ CRR pricing kernel

The C++ extension replaces the inner loop of one American-option CRR tree.
Python still selects tree depth, checks parameters, averages adjacent tree
depths, solves IV, computes Greeks, filters contracts, and writes features.
Non-dividend calls retain their existing closed-form fast path.

Build the extension **on each machine where it will run**. It is not a portable
binary and is ignored by Git. No Python build package is required, but a C++17
compiler and Python development headers are required on macOS or Linux:

```bash
.venv/bin/python Data_Pipeline/native/build.py
```

`OPTIONS_CRR_BACKEND=auto` is the default: it uses C++ when the extension is
available and otherwise uses the original NumPy loop. Select a backend
explicitly for verification or benchmarking:

```bash
OPTIONS_CRR_BACKEND=python .venv/bin/python benchmarks/benchmark_pricing.py --workers 1 4 --repeats 2
OPTIONS_CRR_BACKEND=native .venv/bin/python benchmarks/benchmark_pricing.py --workers 1 4 --repeats 2
OPTIONS_CRR_BACKEND=native .venv/bin/python -m pytest -q Data_Pipeline/test_iv_quality_regressions.py Data_Pipeline/test_crr_native.py
```

Explicit `native` selection fails if the extension has not been built. The
numerical comparison test skips only its native cases when no extension is
present. The backend selection is inherited by worker processes. See the
[benchmark notes](../../benchmarks/README.md) for measured results and scope.
