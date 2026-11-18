# Optional native pricing and ingestion

Build both implementations on the machine where they will run:

```sh
.venv/bin/python Data_Pipeline/native/build.py --model all --jobs 4
```

Use `--model crr` for only the CRR extension, or `--model lr` for LR plus
native OPRA ingestion. CMake, a C++17 compiler, Python headers and zlib are
required. The included libdeflate static library is built with at most four
jobs; its license remains in `libdeflate/COPYING`. Native files are not portable
between Python versions/architectures and are ignored by Git.

`OPTIONS_PRICING_MODEL=crr` is the default. CRR retains the historical 201-step
adjacent-depth treatment and bisection solver. `OPTIONS_PRICING_MODEL=lr`
selects the separate odd-depth Leisen–Reimer engine and solver. LR convergence
and empirical pricing differences must be assessed separately from CRR backend
parity; a change of model is not a bitwise-preserving optimization.

`OPTIONS_CRR_BACKEND` and `OPTIONS_LR_BACKEND` independently accept `auto`,
`python`, or `native`. `OPTIONS_OPRA_BACKEND` chooses ingestion using the same
values and does not change pricing. Explicit native selection fails if the
required implementation is unavailable. Auto can fall back to Python, not to
another financial model.

```sh
.venv/bin/python -m pytest -q Data_Pipeline/test_crr_native.py Data_Pipeline/test_lr_native.py Data_Pipeline/test_consolidated_pipeline.py
```

Tests compare both native engines with their own Python implementations,
verify IV/Greek routing, and check gzip-member, symbol-boundary, final-line,
empty-filter and corrupt-input behavior. See `../../benchmarks/README.md`
for historical measurements and the portable current benchmark.
