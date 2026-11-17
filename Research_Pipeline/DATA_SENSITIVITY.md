# Updated data and feature sensitivity experiment

This command preserves the original monthly diagnostic and compares updated data, training-only feature pruning and verified additional signals. All historical comparisons are exploratory: the evaluation periods were inspected before this experiment.

## Start on darkpool

Use the existing research Python environment and a new private directory outside the public checkout. Freeze the source while no exporter is writing, and verify every copy against before/after source hashes:

```sh
python - <<'PY'
from Research_Pipeline.strategy_experiments import freeze
freeze('/srv/data/options_model_features/ml_dataset_daily.parquet',
       '/srv/data/stocks', '/private/new-snapshot')
PY
python -m Research_Pipeline.data_feature_sensitivity \
  --old-snapshot /private/original-snapshot \
  --new-snapshot /private/new-snapshot \
  --reference-results /private/original-results \
  --output-dir /private/new-sensitivity-results \
  --max-cores 4 --iterations 2000
```

`--stage audit`, `baseline`, `features`, and `sensitivity` stop after the named stage. `all` also computes diagnostics and charts. Resume by using exactly the same arguments and output directory: configuration, software versions, code commit and input hashes must match. This project's own model cache is trusted private input; never load joblib files from unknown sources. Partial files must not be moved between runs. A smaller `--iterations` is an engineering pilot only.

The main process pins itself and its child workers to at most four available CPUs on Linux. Each numerical library and LightGBM worker uses one thread. Model fits use at most four processes; simulations remain sequential. No downloads occur.

## Data controls

1. Reproduce legacy predictions within `1e-8` and original daily equity, positions and charges.
2. Replay saved predictions against updated prices, retaining every original candidate and selection rule.
3. Refit unchanged stock-only/full models on updated inputs; report comparisons on common keys and full eligible universes.
4. Fit the twelve main variants on one shared new-data panel and identical outcome masks.

Historical slopes are reconstructed as 90-day minus 30-day endpoints. Filled exporter slopes, robust fallback columns, target/forward columns and undocumented investability flags are excluded. Reconcile the wide file with dated stock CSVs; use CSV prices consistently for features, labels and valuation. Record ticker identity limitations, stale/zero-volume prices, adjusted-price disagreements and unsupported corporate actions. No missing future prices are filled or removed from selection. Provider timestamps retain the conservative next-tradable-session readiness assumption.

The data audit saves source hashes, added columns/keys, revised historical values, coverage and adapter reconciliation. Every generated model saves its ordered schema, retained/removed features, training cutoff, source manifest and preprocessing.

## Features and models

Both Ridge (`alpha=10`) and LightGBM (100 trees, 15 leaves, learning rate .05, minimum 50 samples per leaf, seed 17) run on stock-only, full combined, compact combined, compact plus alternatives, pruned full and pruned extended inputs.

The first four reproduce the previous feature definitions. Extended adds 63/252-session adjusted returns, annualized 60/90-session realized volatility, 60-session return skew, daily volume/trailing 20-session mean, and IV/annualized realized-volatility ratio. Scheduled sessions must be complete; missing lookbacks stay missing. These are new verified definitions, not assumed equivalents of the exporter's similarly named columns. Ninety-day realized volatility is distinct from ninety-day implied volatility.

Pruning is refitted on each matured training window. Exclude >50% missing and constant features; rank remaining features by better coverage then feature name; greedily retain representatives with absolute Pearson correlation below .90 against already-retained features. At least 200 paired observations are required before a correlation can remove a feature. Eligibility never depends on this pruning.

Every fit saves before-imputation Pearson and Spearman matrices, pair counts, missingness and heatmaps. At six fixed reporting endpoints (the four earlier blocks and first/last historical weeks), separate diagnostic fits use only earlier matured labels and validate on the latest 13 matured weeks. They require 39 earlier fitting weeks and eight scorable validation weeks. Twenty permutation repetitions are performed within validation weeks, including joint permutations of correlated groups. Coefficients and gain are supplementary. Local effect curves use ten training-quantile bins, minimum support 20, and a deterministic sample of at most 3,000 rows per bin. Missing/sparse bins break curves; responses are descriptive rather than causal. CSVs retain every feature; plots show the leading features for readability.

## Windows and inference

Retain the 52 assessment signal weeks May 2, 2025–April 24, 2026 and 26 historical holds May 11–November 9, 2026. Require 52 matured training weeks, strict label availability before each noon decision, and 200 known outcomes for weekly rank statistics. Keep undercovered weeks in saved predictions and disclose unavailable statistics. The target remains signed entry-to-exit adjusted return.

Each main variant receives 2,000 shared within-week, market-cap-decile shuffles. Ranking and portfolio families each contain twelve variants. Missing-price portfolio draws yield invalid counts and conservative bounds, never valid-draw-only p-values. Gross/net portfolios share permutations. Four-week bootstrap indices are paired across variants. The challenger is `pruned_extended/ridge` versus `full/ridge`; original Combined Ridge remains the original primary. No new pass/fail profitability claim is inferred from these retrospective comparisons.

## Sensitivities

One setting changes at a time. For full and pruned-extended groups, Ridge uses alpha 1/10/100; LightGBM uses leaves 7/15/31 or trees 50/100/200 with other parameters fixed. The baseline value is supplied by the main variants, giving twelve extra model variants. Assess both date windows; do not select settings from historical performance.

For full/pruned-extended Ridge, compare selection 2.5/5/10%, cap 2.5/5%, side exposure .5/1, execution 0/1/5 bps, borrow 0/3/6%, and capital $10k/$100k/$1m. Include long-only and short-only portfolios at the main side exposure. These are thirteen distinct portfolio cases including the common baseline. Same predictions, dated prices, commission approximation and self-financing ledger apply throughout. The frozen November 12, 2026 fee schedule is a current-cost scenario, not historical commission evidence.

Main portfolios equal-weight the highest/lowest 5%, target +100%/-100%, cap individual stocks at 5% of post-cost equity and reduce exposure for small selected sides. Apply IBKR Fixed estimates, 1 bp execution and 3% borrow. Target weights rebalance weekly; report price-driven drift. Trades are netted per security. Attribute reversal costs proportionally to long/short traded notionals. Total leg P&L, all cost components, cash and equity must reconcile. Long-only/short-only runs have their own cash, costs and equity; they are not the leg contribution divided by capital.

## Artifacts and verification

Private artifacts include source/configuration manifests, all fits, predictions, schemas, coverage/correlations, importance/effect CSVs and plots, quintiles/IC/MSE, holdings/trades/equity, cost and leg attribution, sensitivity comparisons, simulation draws and paired bootstrap intervals. Alpha is an internal-universe proxy only where complete benchmark coverage permits it. Missing prices, paper controls, historical identity and availability assumptions remain explicit.

```sh
python -m pytest -q Research_Pipeline
```

Tests cover source/target exclusion, exact exchange-session lookbacks, gaps, ticker separation, strict chronology, training-only pruning, stable schemas/cache checks, all twelve synthetic variants through ledger/simulation, shared randomizations, cost splits/reversals, caps and standalone modes. Publish code, tests and instructions only. Keep data, fits and research results outside Git.

## Presentation exports

`Research_Pipeline.sensitivity_report` creates charts, a plain-language report, reconciled leg attribution and paired incremental-options summaries. Its optional export takes existing private Word and TeX reference reports so original tables and the monthly paper diagnostic remain available:

```sh
python -m Research_Pipeline.sensitivity_report --results /private/new-sensitivity-results
python -m Research_Pipeline.sensitivity_report \
  --results /private/new-sensitivity-results \
  --base-docx /private/reference-report.docx \
  --base-tex /private/reference-paper/main.tex \
  --output-dir /private/presentation
```

The document export additionally needs `python-docx`; use the configured document runtime. The TeX project includes its figures and needs `longtable` and `graphicx`. Compile the TeX or render the Word file to PDF and inspect every page before delivery. Data results are loaded at export time rather than copied into public source. Original report files are read-only inputs. `report_sources.json` records the result manifest and reference paths. Exported findings use Experiment → Model → Feature set → Results, keeping high-level model explanations and application details.
