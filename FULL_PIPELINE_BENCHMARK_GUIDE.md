# Options pipeline benchmark: server handoff guide

This guide is for the person running the **offline calculator benchmark** on the
shared `darkpool` server. It covers every available OPRA trading day twice:
once for all supported underlyings and once for date-specific S&P 500 members.
The goal is to find the fastest *correct* combination of pricing, parsing,
parallelism, and storage choices. No model training is part of this job.

**Current state:** the four code versions below exist on GitHub. The
full-history, date-filtered benchmark driver described in section 4 is still
to be written. The single-day smoke commands in this guide are runnable, but
they are not a substitute for that driver.

## 1. Get access and put the code on the server

1. Sign in to Tailscale with the account that received the server share. Accept
   the share, then check that `darkpool` appears in `tailscale status` or the
   Tailscale **Shared with me** machine list. If it does not, ask the server
   owner for the share and the machine's Tailscale name or IP. Do not guess an
   address or run the job on a different machine.
2. Connect with `ssh aarya_options@darkpool` (or use the verified Tailscale IP).
   Keep the SSH password and any GitHub/provider tokens out of commands, logs,
   Markdown, and Git. A Tailscale share and an SSH account are separate forms
   of access; both must work.
3. Clone the existing repository on the server. This does **not** migrate to a
   new GitHub repository:

   ```bash
   mkdir -p ~/quant-benchmark
   cd ~/quant-benchmark
   git clone https://github.com/AggieInvestmentClub/quant-fund-options-ml.git source
   cd source
   git fetch --all --prune
   ```

   If GitHub denies the clone, request repository access. For a one-time code
   transfer from a machine that already has the repo, a tracked-file snapshot
   is also possible:

   ```bash
   git archive <commit> | ssh aarya_options@darkpool \
     'mkdir -p ~/quant-benchmark/snapshot && tar -xf - -C ~/quant-benchmark/snapshot'
   ```

   Use a separate snapshot directory per commit. Do not transfer `.env`,
   data, caches, or built native binaries.

Create separate, **pinned** checkouts so a later branch push cannot change a
run halfway through:

```bash
cd ~/quant-benchmark/source
git worktree add --detach ../original a62109563a0defe15efe5b5a9f5f6959eb1a6f7d
git worktree add --detach ../tree a9c6de9e3e8d74662f466ae9a2fc785e45c66575
git worktree add --detach ../native 5f56883fb44f10a5099a8d43a0e4c16c95d35b18
git worktree add --detach ../vectorized fa7a007426e79728158a139d8c9b7088220c1fca
```

The branches behind those commits are respectively `main`,
`perf/reduce-options-cpu-cost`, `perf/native-crr-pricing`, and
`perf/vectorized-opra-parsing`. The performance branches are **cumulative**:
native includes the tree change; vectorized includes both. The tree change
reuses two IV bracket price evaluations; it does not shorten the tree.

For a new optimization, make and test the change in its own performance
branch, push that branch to the **same** GitHub repo, then fetch it on the
server. Add a new detached worktree pinned to the pushed commit; do not update
a checkout while a benchmark is using it:

```bash
# On the development machine, after testing and committing the change:
git push -u origin perf/your-change

# On darkpool:
cd ~/quant-benchmark/source
git fetch origin perf/your-change
git worktree add --detach ../your-change FETCH_HEAD
git -C ../your-change rev-parse HEAD
```

Build its native extension on `darkpool` if the new checkout uses C++.

Set up one Python environment on the server and build the native extension in
each checkout that will use it:

```bash
cd ~/quant-benchmark
python3 -m venv venv
venv/bin/python -m pip install -r source/Data_Pipeline/requirements.txt pytest psutil
cd native && ../venv/bin/python Data_Pipeline/native/build.py
cd ../vectorized && ../venv/bin/python Data_Pipeline/native/build.py
cd ..
```

The C++ extension must be built **on the server**. On the native and
vectorized checkouts, set `OPTIONS_CRR_BACKEND=native` to require C++, or
`OPTIONS_CRR_BACKEND=python` to force the Python kernel. Avoid `auto` in a
benchmark: it can silently change the tested implementation. Run the pricing
regressions in each native checkout before timing:

```bash
cd ~/quant-benchmark/vectorized
OPTIONS_CRR_BACKEND=native ../venv/bin/python -m pytest -q \
  Data_Pipeline/test_iv_quality_regressions.py Data_Pipeline/test_crr_native.py
```

## 2. Freeze the data and check the machine

The previous server profile found these paths; verify them before a new run:

| Input or device | Expected location |
| --- | --- |
| OPRA daily `.csv.gz` files | `/srv/data/options_model_features/FlatFiles/us_options_opra/day_aggs_v1` |
| Split-safe stock histories | `/srv/data/stocks` |
| Local DGS3MO rates | `/srv/data/risk_free_rate/DGS3MO_risk_free_rate.csv` |
| Dated S&P 500 snapshots | `/srv/data/options_model_features/Constituency/constituency_YYYY-MM-DD.csv` |
| Rotational data volume | `/srv/data` |
| SSD | `/home` |
| RAM-backed temporary space | `/dev/shm` |

Before timing, record `lscpu`, `free -h`, `df -h`,
`lsblk -o NAME,ROTA,SIZE,TYPE,MOUNTPOINT`, Python/package versions, the four
commit SHAs, and the CPU affinity/quota. Inventory every OPRA date, file size,
stock history, rate observation, and constituency snapshot. For example:

```bash
RUN_ROOT=/srv/data/quant-bench-scratch/full-history-$(date -u +%Y%m%d)
OPRA=/srv/data/options_model_features/FlatFiles/us_options_opra/day_aggs_v1
STOCKS=/srv/data/stocks
SP500=/srv/data/options_model_features/Constituency
RATES=/srv/data/risk_free_rate/DGS3MO_risk_free_rate.csv
mkdir -p "$RUN_ROOT"
find "$OPRA" -name '*.csv.gz' -type f -printf '%P\t%s\n' | sort > "$RUN_ROOT/opra_files.tsv"
find "$OPRA" "$STOCKS" "$SP500" -type f -print0 | sort -z | \
  xargs -0 -r sha256sum > "$RUN_ROOT/input_files.sha256"
sha256sum "$RATES" >> "$RUN_ROOT/input_files.sha256"
sort -o "$RUN_ROOT/input_files.sha256" "$RUN_ROOT/input_files.sha256"
```

Save this manifest outside the repository and recheck it afterward; a changed
input invalidates the comparison. The checksums themselves read the data, so
finish this step **before** the cold-cache timing sequence.

For S&P 500 runs, use the latest snapshot **on or before each trading day**.
Require a snapshot on or before the first OPRA day and no snapshot gap over
120 days. Stop the S&P run and obtain missing historical snapshots if coverage
fails; never use today's members as a substitute. Check that local rates cover
all OPRA dates before timing so the calculator cannot fall back to an API call.
Also estimate scratch output size from the pilot and confirm enough free space
for a fresh output directory per run.

## 3. What to compare

Run each row below on both universes. Each configuration gets three complete
offline runs with fresh scratch outputs. Use the same input manifest, numerical
library versions, and worker count for every configuration within a universe.

| ID | Checkout / branch | Backend | What it isolates |
| --- | --- | --- | --- |
| A | `original` / `main` at `a621095` | Original Python | Baseline |
| B | `tree` / `perf/reduce-options-cpu-cost` | Python | Avoid repeated IV bracket pricing: B vs A |
| C | `native` / `perf/native-crr-pricing` | `native` | C++ after tree change: C vs B |
| D | `vectorized` / `perf/vectorized-opra-parsing` | `python` | Parser change without C++: D vs B |
| E | `vectorized` / `perf/vectorized-opra-parsing` | `native` | Tree + C++ + parser; parser effect: E vs C; C++ effect with parser: E vs D |

Test storage separately with A and E on the **same full pipeline**: rotational
disk input/output versus bounded RAM staging. This shows whether faster CPU
code makes storage a bigger share of total time. An SSD-output-only condition
can help split read cost from write cost. Include time to copy inputs into RAM
and flush outputs back to disk in the end-to-end RAM result; also report the
calculation-only portion. Stage OPRA files **and the stock histories actually
used by the batch**, not just OPRA. Work in 1–4 day batches within at most half
of available `/dev/shm` and half of available system RAM; never swap. If one
day will not fit, mark the full-history RAM comparison infeasible rather than
silently sampling fewer days.

Pilot the smallest, median, and largest OPRA days at **1, 2, and 4 workers**,
twice per setting, for A and E in both universes. Pick the fastest valid count
for E in each universe; break a close tie in favor of fewer workers. Use that
same count for A–E and the storage pairs. Pin the entire process tree to **at
most four CPU cores** (`taskset -c <four-verified-core-IDs>`) and set
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1`. Run only one benchmark at a time. Rotate A–E order
across the three rounds and pair disk/RAM conditions nearby in time.

## 4. Run the *whole* offline pipeline

Use a benchmark-only driver with explicit paths and no production checkpoints.
The three stages, in order, are:

1. `Data_Pipeline/flatfile_iv.py`: read OPRA, join stock prices/rates, calculate
   IV and hypothetical ATM Greeks, write contract and daily/final features.
2. `Default_Strike_Pipeline/flatfile_regular_greeks.py`: use the ATM output as
   the IV source, calculate actual-strike Greeks, write daily/final features.
3. `Default_Strike_Pipeline/regular_features.py`: build delta-bucket feature
   shards from the actual-strike contract output.

The driver must enumerate **every** file in the manifest, pass the dated
membership set as `allowed_underlyings` to the ATM file task for S&P runs
(`None` for the full universe), and apply the **same** set to Default Strike.
It must use a fresh scratch output root per run, pass `--workers` explicitly,
and time each stage separately. Do not use `--testing`, `--underlyings AAPL`,
`--limit-files`, or date limits in a full run. Do not use
`Data_Pipeline/main.py` or the server `.sh` wrappers for this historical
benchmark: they can download data and skip older files through checkpoints.
The standalone ATM CLI also has no date-specific S&P filter, so running it
unchanged is **not** an S&P benchmark.

**Important existing behavior:** the Default Strike CLI's
`source_iv_output_covers_target()` checks whether a day's saved IV rows reach
both the 30- and 90-day DTE endpoints. A valid sparse day can fail that check
and trigger another IV calculation. Keep this behavior in the primary
as-shipped benchmark, use `--recalculate-missing-iv` where required, and log
the number of days and seconds spent recalculating. A separate diagnostic
using direct IV reuse can quantify that avoidable cost; do not mix its timing
with the primary result.

For a **single-day, full-universe smoke test**, the existing CLIs can be run
directly. Replace `YYYY-MM-DD`, choose a fresh scratch directory, and set the
backend to match the chosen checkout:

```bash
cd ~/quant-benchmark/vectorized
export OPTIONS_CLEAN_STOCK_ROOT=/srv/data/stocks
export OPTIONS_RISK_FREE_RATE_CSV=/srv/data/risk_free_rate/DGS3MO_risk_free_rate.csv
export OPTIONS_CRR_BACKEND=native
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SCRATCH=/srv/data/quant-bench-scratch/smoke-YYYY-MM-DD
OPRA=/srv/data/options_model_features/FlatFiles/us_options_opra/day_aggs_v1
PY=../venv/bin/python

taskset -c 0-3 "$PY" Data_Pipeline/flatfile_iv.py \
  --input-root "$OPRA" --output-root "$SCRATCH/atm/contracts" \
  --daily-features-root "$SCRATCH/atm/daily" \
  --cache-output "$SCRATCH/atm/close-cache.csv" \
  --final-features-dir "$SCRATCH/atm/final" \
  --start-date YYYY-MM-DD --end-date YYYY-MM-DD --workers 4

taskset -c 0-3 "$PY" Default_Strike_Pipeline/flatfile_regular_greeks.py \
  --input-root "$OPRA" --source-iv-root "$SCRATCH/atm/contracts" \
  --output-root "$SCRATCH/regular/contracts" \
  --daily-features-root "$SCRATCH/regular/daily" \
  --cache-output "$SCRATCH/regular/close-cache.csv" \
  --final-features-dir "$SCRATCH/regular/final" \
  --all-underlyings --recalculate-missing-iv \
  --start-date YYYY-MM-DD --end-date YYYY-MM-DD --workers 4

taskset -c 0-3 "$PY" Default_Strike_Pipeline/regular_features.py \
  --input-root "$SCRATCH/regular/contracts" \
  --output-root "$SCRATCH/regular/delta-buckets" \
  --start-date YYYY-MM-DD --end-date YYYY-MM-DD --workers 4
```

The smoke test is a setup check, **not** the full benchmark or the S&P run.
For a full run, automate these stages with the dated membership filter and
fresh paths in the benchmark driver. Keep the driver outside the four code
checkouts so all versions use identical orchestration.

## 5. Measurements, validation, and decision

For each run, keep a log and machine-readable timing record with: input file
and contract counts; eligible and successful IV rows; missing underlying
closes and failure reasons; wall and summed process-tree CPU time; peak
combined memory; physical disk bytes read/written; per-stage time; and daily
throughput. Sample `iostat -dx 1` and process-tree CPU/I/O while the job runs.
Use the Linux per-process I/O delay counter only if available; otherwise mark
I/O wait **unavailable**. `wall time - CPU time` is not an I/O-wait measure.
Separate cold-read and warm-cache results, and verify actual physical reads
for cold trials. Do not drop the server-wide file cache.

Compare A–E outputs before claiming a win: exact file lists, dates, contract
keys, row counts, failure reasons, and feature schemas; finite IV and Greek
values within `1e-8`. Compare the storage conditions' outputs the same way.
The previous 160-contract fixture and small server tests passed this tolerance,
but they do not validate a full historical run. See the
[previous server results](https://github.com/AggieInvestmentClub/quant-fund-options-ml/blob/perf/vectorized-opra-parsing/benchmarks/server_2026-10-28.md)
for context, not for a projected full-dataset speedup.

Report each of the three full-run times and the **median** for every condition,
plus CPU-time reduction, memory cost, worker pilot curve, stage breakdown,
disk/RAM copy and flush time, and S&P coverage. Keep the frozen input
manifest, raw logs, and one complete output set per code version and universe.
If inputs change, outputs differ, server contention is high, or paired cold
reads are materially unequal, rerun the pair or label it inconclusive.

Earlier tests were small: one AAPL day averaged **20.77 s** original, **19.74 s**
with bracket reuse, and **2.60 s** with bracket reuse plus C++. Five AAPL days
averaged **12.44 s** with disk input/SSD output and **12.83 s** with RAM input
and output. A one-day full-universe profile spent roughly **33 s on IV**, **7 s
on ATM Greeks**, and **3 s on cold stock lookups**, with about **92 MB** of
physical reads. These observations motivate the full-history comparison; they
do not establish its outcome.

The final decision should name the fastest **validated** combination for each
universe and whether disk waiting actually dominates any stage. A faster
single-day or AAPL result is useful for diagnosis, not proof of a full-history
gain.
