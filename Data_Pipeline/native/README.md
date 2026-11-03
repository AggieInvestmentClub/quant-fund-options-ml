# Optional C++ Leisen-Reimer pricing kernel

The C++ extension replaces the inner loop of one American-option Leisen-Reimer
tree.  Python retains the Peizer-Pratt parameter computation, odd-step
enforcement, IV solver, and Greek logic.  Non-dividend calls retain their
existing closed-form Black-Scholes fast path.

Build the extension **on each machine where it will run**. It is not a portable
binary and is ignored by Git. No Python build package is required, but a C++17
compiler and Python development headers are required on macOS or Linux:

```bash
.venv/bin/python Data_Pipeline/native/build.py
```

`OPTIONS_LR_BACKEND=auto` is the default: it uses C++ when the extension is
available and otherwise uses the original NumPy loop. Select a backend
explicitly for verification or benchmarking:

```bash
OPTIONS_LR_BACKEND=python  .venv/bin/python benchmarks/benchmark_pricing.py --workers 1 4 --repeats 2
OPTIONS_LR_BACKEND=native  .venv/bin/python benchmarks/benchmark_pricing.py --workers 1 4 --repeats 2
OPTIONS_LR_BACKEND=native  .venv/bin/python -m pytest -q Data_Pipeline/test_iv_quality_regressions.py Data_Pipeline/test_lr_native.py
```

Explicit `native` selection fails if the extension has not been built. The
numerical comparison test skips only its native cases when no extension is
present. The backend selection is inherited by worker processes. See the
[benchmark notes](../../benchmarks/README.md) for measured results and scope.

## Why Leisen-Reimer?

The Leisen-Reimer (LR) parametrisation uses the Peizer-Pratt Method 2 inversion
to centre the lattice terminal nodes around the strike price when the step count
is **odd**.  This eliminates the odd/even sawtooth oscillation that required CRR
to average two adjacent tree depths.  LR converges at O(1/N²) vs O(1/N) for
CRR, so a single 101-step tree is more accurate than a CRR average of 201+202
steps, and approximately 4–6× faster end-to-end on the IV inversion workload.
