# Matched pruning test for the four-week long-only strategy

This experiment refits the exact four-week relative-return Ridge strategy with
full and training-pruned inputs. It fixes alpha at 10, expanding matured training,
the original eligible rows, May 11–October 26 2026 historical dates, top-5% entry,
top-10% retention, equal weights, 5% rebalance caps, $100,000 capital and the frozen
IBKR-plus-1-bp cost scenario. It also reports the unchanged earlier development
window, without selecting a new model or policy from either result.

```sh
python -m Research_Pipeline.horizon_pruning \
  --panel /private/original/weekly_panel.parquet \
  --prices /private/original/stock_prices.parquet \
  --reference-results /private/horizon/results \
  --output-dir /private/horizon-pruning/results \
  --max-cores 2 --iterations 2000
```

Inputs must match the prior horizon manifest hashes. Before accepting the new
comparison, refitted full-model scores must reproduce saved scores within 1e-8;
full portfolio return, Sharpe, drawdown and costs must reproduce saved results.
Outputs use a separate private directory and cannot overwrite the reference run.

Pruning uses only that fit's matured training rows: remove columns with more than
50% missingness or constant values, then greedily retain better-covered columns
with absolute Pearson correlation below .90, requiring 200 paired observations.
The exact retained and removed columns are saved for each refit. Preprocessing is
fitted separately on each training window. Candidate rows and unknown outcomes
remain unchanged. Correlations before and after pruning and chronological
importance/local response diagnostics are retained for both model comparisons.

Each historical variant receives 2,000 identical within-period market-cap-decile
donor permutations and paired gross/net ledger replays. Recompute retention,
turnover and all costs rather than shuffle portfolio returns. Price failures remain
invalid and contribute conservative p-value bounds. Eight-calendar-week bootstrap
blocks are shared across both variants. Paired net-return and rank-IC differences
are reported as well as standalone results. Conservative family sizes include the
50 prior horizon portfolio cases plus these two comparisons and the 24 prior
ranking variants plus these two; they do not encompass every historical research
trial. Independent period shuffling is not a turnover-matched null.

Only six nonoverlapping four-week holds exist in the historical comparison.
These previously inspected dates provide exploratory evidence, even if a result
improves. Missing selected prices invalidate the affected portfolio. Invalid
development runs do not justify retrospective strategy selection. No data is
downloaded, no prices are fabricated and the monthly paper diagnostic is unchanged.

Artifacts include configuration, source/code hashes, ordered model schemas,
training cutoffs, fitted models, predictions, ranking tables, both ledgers,
holdings, equity, attribution, costs, random draws, bootstrap samples and paired
contrasts. Keep all artifacts private. At most four worker cores are allowed;
numerical libraries remain single-threaded.
