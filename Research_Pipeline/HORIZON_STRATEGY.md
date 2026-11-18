# Longer holding periods and selective trading

Run only on frozen, private inputs. No downloads, server cron changes or live orders occur. All comparisons are exploratory; the historical dates were inspected previously. The monthly diagnostic remains unchanged.

```sh
python -m Research_Pipeline.horizon_strategy \
  --panel /home/aarya_options/quant_data_sensitivity_20261116/results/prepared/weekly_panel.parquet \
  --prices /home/aarya_options/quant_data_sensitivity_20261116/results/prepared/stock_prices.parquet \
  --reference-results /home/aarya_options/quant_data_sensitivity_20261116/results \
  --output-dir /home/aarya_options/quant_horizon_20261118/results \
  --max-cores 4 --iterations 2000
```

Use `--stage controls`, `horizons`, `rules`, `report`, or `all`. Resume using identical inputs, code commit, environment and arguments. The output manifest rejects incompatible resumes. Fitted caches are trusted private artifacts; do not load unknown joblib files. Counts below 2,000 are engineering pilots.

## Fixed comparisons

Compare 1/2/4-week holds, absolute/relative targets, stock-only/full inputs, and Ridge/LightGBM: 24 configurations. All training examples are weekly; a horizon outcome enters fitting only after its exit close. Training requires 52 matured signal weeks. Actual holdings do not overlap. Relative targets subtract the equal-weight mean finite stock outcome in that decision-time universe, with at least 200 known outcomes. Relative labels do not make that incomplete universe an executable benchmark.

Development entries/exits span June 2 2025 to March 9 2026, or 40 calendar weeks. Historical comparisons span May 11 to October 26 2026, or 24 weeks. Signal timing uses exchange sessions, noon decisions and closing executions. Holiday Mondays enter at the next exchange session. All eligible rows remain; missing future prices never change selection. April 2026 undercoverage is disclosed separately.

Select only from development configurations with positive IC, positive net return and a complete ledger. Choose highest net weekly Sharpe, then lower turnover, then name. If none qualifies, h4/relative/full/ridge is a diagnostic fallback, not a successful model. Historical results cannot change selection.

## Trading policies

Baseline enters/retains the top/bottom 5%. Strict entry uses 2.5%; retention remains 5%. Buffer entry uses 5%; retention is 10%. Both uses 2.5%/10%. Percentiles always use the entire eligible universe. Keep qualifying incumbents first, then fill vacancies. Positions must continue to satisfy confirmations. No unscheduled trades occur.

Momentum ranks verified trailing 63-session return, with positive long and negative short momentum. Gap confirmation uses above-median call-minus-put IV for longs, below for shorts. Skew confirmation adds below-median put skew for longs, above for shorts. Missing confirmations fail. These are experimental economic rules, not matched-contract parity measures.

Run four ML policies, plain/gap/gap+skew momentum under baseline, and gap+skew under both. Evaluate each long-short, long-only and short-only. Add full/stock-only controls under identical both-policy settings for incremental options value. Declare all 50 portfolio cases before historical rule evaluation, including duplicate controls conservatively; duplicate simulations may reuse identical draws. Rank inference has 24 ML configurations.

Use $100k, equal weights, 5% caps, maximum ceil(5% of full universe) stocks per side, frozen IBKR estimates, 1bp execution and 3% borrow. If entries are sparse, exposure remains in cash. No password, source data, fitted outputs or research results belong in Git.

## Inference and delivery

Run 2,000 paired size-decile shuffles and 2,000 paired eight-calendar-week bootstrap resamples. Shuffle score/confirmation bundles together, replay retention and costs, retain invalid-draw counts and conservative bounds. Independent period shuffling is not turnover matched. Weekly valuation supplies comparable Sharpe/return statistics across horizons; only six four-week historical holds exist.

The private report includes model explanations, all configurations, IC/quintiles/MSE, holdings, daily and weekly equity, costs, concentration/NVDA exposure, individual-stock and long/short contributions, available internal-universe alpha, diagnostics and paired contrasts. No profitability or paper-replication claim follows from software completion.

```sh
python -m pytest -q Research_Pipeline
```

## Presentation report

Use a Python runtime with the document dependencies to append this experiment to the retained Word report:

```sh
python -m Research_Pipeline.horizon_report --results /private/horizon-results \
  --base-docx /private/reference-report.docx --output-docx /private/horizon-report.docx
```

Render the resulting Word file and export its PDF, inspect the pages, and retain both privately. Existing reports are inputs, not overwritten. The Markdown report and CSVs remain the complete machine-readable comparisons, including unavailable results.
