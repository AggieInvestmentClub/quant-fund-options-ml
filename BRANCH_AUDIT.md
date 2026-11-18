# Branch consolidation audit

Audited November 17, 2026. The single working branch is `integration/consolidated`. Original remote tips were frozen before integration. Branch ownership is inferred from commit authors and the approved cleanup list; Git does not record branch creators.

## Inventory

| Original branch | Pinned commit | Tip author | Retained through | Disposition |
|---|---|---|---|---|
| `feat/data-prep` | `3f4d0366de5c027b1c3b728828ec9428eda49671` | Ben Hoang | feat/data-prep development line | leave branch unchanged |
| `feat/options-ml-pipeline` | `be823350a3c9fcd1b9a0051c273a36b6714f2c5e` | Aarya Bookseller | research development line | archive tag then delete superseded branch |
| `main` | `4ceae00a3661d6447f4340eda989f1cb5e5e0b78` | Ben Hoang | research development line | leave branch unchanged |
| `perf/fast-gzip` | `e6937ec44c512ad29754354cb07eecaf95c81b09` | Ben Hoang | feat/data-prep development line | leave branch unchanged |
| `perf/iv-compute-refactor` | `0928f6505c2fd7fa7e7b9bdb02f79b9c3ddaf1d1` | Ben Hoang | feat/data-prep development line | leave branch unchanged |
| `perf/leisen-reimer-pricing` | `eab440adab74cd5882aa6fa95667607dd0c693c2` | Ben Hoang | feat/data-prep development line | leave branch unchanged |
| `perf/native-crr-pricing` | `5f56883fb44f10a5099a8d43a0e4c16c95d35b18` | Aarya Bookseller | feat/data-prep development line | archive tag then delete superseded branch |
| `perf/native-greeks` | `9e6738dc701a3c5c51edfa0d547cbdda29843a57` | Ben Hoang | feat/data-prep development line | leave branch unchanged |
| `perf/opra-ingestion` | `c82c4dd311ac7e11347d1dbaf848edd7dbb4fa79` | Ben Hoang | feat/data-prep development line | leave branch unchanged |
| `perf/reduce-options-cpu-cost` | `a9c6de9e3e8d74662f466ae9a2fc785e45c66575` | Aarya Bookseller | feat/data-prep development line | archive tag then delete superseded branch |
| `perf/stage-2-optimizations` | `1cea3e0718abf1e54dc9577be45d916603f151df` | Ben Hoang | feat/data-prep development line | leave branch unchanged |
| `perf/stage-2-optimizations2` | `0bf45c38a0d72f2b7848ea78af97f6add9cd7f06` | Ben Hoang | feat/data-prep development line | leave branch unchanged |
| `perf/stream-ingestion` | `19334626c884837ef00073a8284c13ccfd621c99` | Ben Hoang | feat/data-prep development line | leave branch unchanged |
| `perf/vectorized-opra-parsing` | `fa7a007426e79728158a139d8c9b7088220c1fca` | Aarya Bookseller | feat/data-prep development line | archive tag then delete superseded branch |
| `research/data-feature-sensitivity` | `4ce68091a5d43abd042ca24c3b9069b7e716b907` | Aarya Bookseller | research development line | archive tag then delete superseded branch |
| `research/feature-ablation` | `a9eb39c849ac5d7610f96ddfaef7dc89a1a68233` | Aarya Bookseller | research development line | archive tag then delete superseded branch |
| `research/paper-replication` | `29353af6c4a3a3aa74115dd36b304ec63390f04a` | Aarya Bookseller | research development line | archive tag then delete superseded branch |
| `research/portfolio-cost-revision` | `cd1c806cdc9d4c096aedccd0a7bb647d457465c1` | Aarya Bookseller | research development line | archive tag then delete superseded branch |
| `research/weekly-options-strategy` | `caf2d851512cd71a16be8f0911071b82e8cf64ed` | Aarya Bookseller | research development line | archive tag then delete superseded branch |

## Retained functionality and resolutions

- The research tip contains the monthly paper diagnostic, weekly strategy, dollar ledger/cost revision, missing-price stress tests, feature ablation, updated-data sensitivity and report exports. Current main documentation is already an ancestor.
- The data-prep tip contains all eleven performance tips: bracket reuse, native CRR, vectorized parsing, Stage 2 improvements, LR/solver changes, streaming ingestion, native Greek batching and libdeflate, plus the Parquet exporter.
- Preserve CRR (default) and LR as distinct financial models. The former CRR-to-LR alias is removed; pricing/Greek routing and OPRA backend selection are explicit. Both extensions build independently or together.
- Restore the original CRR fixture. The LR-modified fixture is retained separately as a historical artifact, with its mislabeled description corrected. LR comparisons use their own untimed Python reference.
- Replace the broken absolute `/home/ben_options/Documents/benchmark.py` symlink with a portable synthetic benchmark command. The full-history rotated disk/RAM matrix remains an operational follow-up, not a completed result.
- Python/native ingestion tests cover exact ticker boundaries, concatenated gzip members, final lines without newlines and corruption. Auto fallback stays within ingestion; explicit native failures propagate.
- New contract checkpoints carry pricing signatures. Unversioned, changed or incompatible checkpoints require fresh/recalculated outputs; frozen research inputs remain untouched.
- Qualify older dataset coverage claims and speculative research rules. Future-dependent validity flags do not become portfolio filters; exact and fallback features remain distinct.
- Retain libdeflate sources and COPYING. Ignore compiled extensions, build artifacts, credentials, market data, fitted models and private research outputs. No server cron or production data changes are part of consolidation.

## Validation and acceptance

Verified locally on November 17, 2026 (macOS arm64, Python 3.14.7), with both native extensions built:

- **123 tests passed**, with no skipped tests. This includes model-specific Python/native parity, ingestion edge cases, checkpoint compatibility and an offline ATM-to-Default-Strike-to-Parquet-to-research handoff. One pandas fragmentation warning remains in the exporter; it does not affect correctness.
- CRR Python/native synthetic runs match the original 160-row reference within `1e-8`. LR native runs match an independent untimed Python LR reference. One- and four-worker runs pass; these are compatibility checks, not production speedup estimates.
- Retrained all four saved stock-only/full-combined Ridge/LightGBM controls across the original 26 historical weeks (11,918 predictions each). Largest absolute prediction difference: `2.79e-15`, below `1e-8`.
- Replayed the original ledger: five weekly and four monthly valid portfolios reproduce saved metrics, holdings, daily equity and charges. The same two weekly portfolios remain invalid because selected daily prices are missing; consolidation does not repair that research limitation.
- Frozen panel, prices and saved-prediction SHA-256 hashes are unchanged. Model artifacts and raw validation logs remain in private storage outside this repository.
- Calculator, Default Strike, surface, Parquet export and all research command help checks pass. Checks use at most four workers and one numerical thread per worker.

The full real-data offline pipeline, historical disk/RAM matrix and server-native rebuild were not rerun as part of branch consolidation.

## Archive and recovery

For each approved superseded branch, create and verify the annotated tag `archive/2026-11-17/<original-branch>` at its pinned tip before deleting the local and remote branch. Verify every original tip is an ancestor of the consolidated branch. Abort deletion if any remote tip changed. Do not delete main or Ben's branches.

To recover an archived branch:

```sh
git fetch origin --tags
git switch -c recovered-branch archive/2026-11-17/research/data-feature-sensitivity
```

Cleanup completion is recorded in the final audit commit after remote tags, surviving branches and the consolidated commit have been verified.
