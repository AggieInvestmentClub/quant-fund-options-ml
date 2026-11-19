# Fixed universe return forecasting and long or cash trading

This offline experiment forecasts **actual four-week adjusted percentage returns** for the same configured 25 stocks. It does not rank stocks or select a top percentage. The broader frozen universe supplies training examples. Each stock has its own 4% post-cost equity slot: forecast strictly above zero means long; otherwise cash. Unused slots are never redistributed. Targets reset every four weeks and drift between rebalances. Zero cash interest, no leverage and no short borrow apply.

The predeclared primary is full-feature Ridge (`alpha=10`). Ridge and LightGBM each run stock-only, full combined and training-pruned combined inputs. LightGBM uses 100 trees, 15 leaves, learning rate .05, minimum leaf 50 and seed 17. Models refit on expanding history with at least 52 matured weekly training examples, with four-week labels strictly available before each noon decision. Missing-value imputation and scaling fit on training data only. Cached models are trusted private files, never load untrusted joblib artifacts.

## Run on existing frozen inputs

Create a fresh private output directory outside Git. Existing monthly and trading artifacts remain immutable. No network downloads, cron edits or live orders occur.

```sh
python -m Research_Pipeline.fixed_universe_forecast \
  --panel /private/frozen/weekly_panel.parquet \
  --prices /private/frozen/stock_prices.parquet \
  --reference-results /private/horizon-results/results \
  --config Data_Pipeline/config.py \
  --spy /private/frozen/spy_prices.parquet \
  --output-dir /private/fixed-universe-results \
  --max-cores 2 --iterations 2000
```

The command supports `--stage audit`, `controls`, `fit`, `diagnostics`, `backtest`, `inference`, `sensitivity`, `report`, `latest` or `all`. Execution phases stop at the requested stage; `report` and `latest` run independently. Resume with identical inputs, source hashes, environment, commit and arguments. Recomputed controls must reproduce existing four-week absolute stock/full Ridge/LightGBM scores within `1e-8` on the fixed 25 names. Input/code/software manifests reject incompatible resumes. Do not move caches or partial outputs between experiments.

Default two workers, at most four; numerical libraries and LightGBM use one thread each. Linux affinity also pins the process tree to the requested cores. `--iterations` below 2,000 labels output as an engineering pilot. The independent `latest` mode uses the most recent **frozen** signal snapshot and its matured labels; it is not a live market service. Its output includes the scheduled four-week entry/exit, decimal-return forecast, long/cash action and target weight.

## Chronology and common observations

Preserve development June 2, 2025–March 9, 2026 (40 weeks), and historical May 11–October 26, 2026 (24 weeks, six four-week decisions). All dates were inspected in prior work and all new findings remain exploratory. Refits train on broader covered names; score rows must include every configured name at every decision. No silent universe shrinkage. Stock tickers are identity proxies, not independently verified historical S&P membership.

Forecasts and outcomes use the existing exchange calendar, Friday/last-session signals, next-session noon New York decisions and closing execution. A label ending at today's close cannot enter today's noon training. Future missing returns do not remove forecast rows or trading candidates. Forecast metrics use a common finite-outcome mask, with at least 20 known outcomes for a cross-sectional statistic. Missing held daily prices or required raw execution prices invalidate the affected ledger; never replace raw with adjusted prices or retrospectively remove a holding.

## Forecast quality, pruning and importance

Report MAE, RMSE, direction accuracy, balanced direction accuracy, Spearman IC and calibration. Primary forecast quality is MSE improvement over a stock-specific expanding historical mean, with date means equally weighted. A stock without matured examples uses the broader training mean and records the fallback. Zero forecasts and training majority-direction accuracy are additional references. Positive means `>0`; exactly zero is nonpositive/cash. Balanced accuracy is unavailable when only one realized class exists.

Before each pruned fit remove >50%-missing and constant/all-missing features, then greedily retain better-covered/name-ordered representatives with absolute Pearson correlation **below .90**, requiring at least 200 paired observations before removal. Always retain full-feature controls. Save before/after broad-training Pearson/Spearman matrices, overlap counts and coverage, plus fixed-25 training-subset matrices. The latter do not change the training universe or pruning decisions.

First/last historical decisions get separate diagnostic fits with the latest 13 matured training signal weeks as validation. Fit rows must have labels matured before the earliest validation decision; require 39 fitting weeks and eight validation weeks with 20 known fixed-stock outcomes. Twenty individual/correlated-group permutations measure validation MSE increase. Supported ten-bin local effects require 20 observations per bin and cap samples at 3,000. These are model responses, not causal effects.

## Trading controls and costs

Reuse the dollar ledger with `PortfolioSpec(selection_mode="fixed_slots", weighting="equal", mode="long_only", max_position=.04, entry_threshold=0)`. The mode rejects excessive total slot exposure and nonfinite forecasts. `exposure_matched` is a comparator mode: spread the same aggregate positive-slot exposure over all 25 stocks equally. Legacy ranked/all/fixed-count defaults remain unchanged.

Use $100,000 and the existing frozen November 12, 2026 IBKR Fixed approximation plus 1 bp execution allowance and separate regulatory charges. Commission share equivalents use raw execution closes; adjusted prices account for total returns without extra dividend cash credits. Opening, drift-netted rebalance trades and final liquidation incur charges. Report gross/net returns, profit, weekly Sharpe/volatility, daily drawdown, cash/exposure, turnover, separate costs, stock contributions and NVDA exposure. SPY regression alpha uses zero cash rate and four-lag Newey–West uncertainty; it is not paper factor alpha.

Compare each main strategy with passive 25-stock four-week rebalancing, its exposure-matched all-stock control, stock-historical-mean long/cash and zero-forecast cash. SPY is the separately frozen buy-and-hold market benchmark. Differences in cash exposure must accompany return comparisons.

## Randomization and block resampling

Each of the six main variants receives 2,000 identical donor permutations within decision date and market-cap decile. Recompute MSE improvement, direction accuracy and rank IC, then replay paired gross/net portfolios and retain turnover, costs and failures. Raw one-sided p-values use `(1 + at-least-observed draws) / 2001`. Bonferroni families contain six main variants per statistic; sensitivity cells are descriptive, and these families do not cover all earlier project trials.

The groups have two or three stocks and condition on each date's forecast distribution and active-slot count. They test stock-specific assignments, not aggregate market timing independently. Missing observed ledgers have unavailable portfolio inference; invalid random draws stay in conservative lower/upper p-value bounds. Never use valid-draw-only p-values.

Run 2,000 shared two-hold/eight-week moving-block samples. Resample each decision's forecast metrics with its four weekly portfolio returns. Save intervals for forecast error improvement, direction, net mean/compounded/annualized return, Sharpe and paired model/control differences. Resampling observations does not rebuild execution for a newly concatenated market path. Six decisions provide limited precision.

## Sensitivities and artifacts

For full/pruned inputs vary Ridge alpha 1/10/100, LightGBM leaves 7/15/31 or trees 50/100/200. Pruned models also vary Pearson .80/.90/.95. Full/pruned Ridge portfolios vary entry threshold 0/.5/1%, execution 0/1/5 bp and capital $10k/$100k/$1m. Each changes one setting; main 4% slots remain fixed. No post-result winner selection.

Private artifacts include configuration and hashes, controls, fitted schemas/models, all dated predictions, per-stock/date coverage and errors, correlation/pruning diagnostics, importance/effect outputs, daily ledgers, costs/contributions, every simulation draw, paired block indices, intervals, sensitivities, report tables and charts. `REPORT.md` and `comparison.csv` summarize Experiment → Model → Features → Results. Word/PDF presentation exports use the document runtime and require rendered page inspection.

```sh
python -m pytest -q Research_Pipeline
```

Completion means reproducible forecasts, reconciled ledgers and explicit uncertainty, regardless of profitability. The monthly 12-month jump/crash paper diagnostic is unchanged. This four-week regression adaptation cannot establish paper replication or validated live trading.
