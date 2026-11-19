# Passive portfolios and frozen Ridge stock selection

This experiment changes portfolio construction only. It does not train models,
change option features, tune parameters, or replace the original primary strategy.
Every comparison is exploratory because these historical periods were inspected.

## Inputs and commands

The explicit download command accepts only SPY, covering the existing development
and historical windows. It uses unmodified Yahoo Finance adjusted close for the
total-return proxy and raw close for commission quantities. It saves provider
version, retrieval time, actions, row counts and hashes, or a failure record.
No other symbol, option history, membership data or model input is downloaded.

```sh
python -m Research_Pipeline.passive_benchmarks fetch-spy \
  --output-dir /private/new-spy-freeze

python -m Research_Pipeline.passive_benchmarks run \
  --panel /private/original/weekly_panel.parquet \
  --prices /private/original/stock_prices.parquet \
  --reference-results /private/horizon/results \
  --spy-prices /private/new-spy-freeze/spy_prices.parquet \
  --spy-metadata /private/new-spy-freeze/metadata.json \
  --output-dir /private/passive-comparisons \
  --max-cores 2 --iterations 2000
```

If acquisition fails, omit `--spy-prices` and retain the failed `--spy-metadata`.
All other comparisons still run. No synthetic or alternate index is substituted.
The replay command itself never downloads data. New output directories are
required; previous artifacts and source files remain unchanged. Counts below
2,000 are labeled engineering pilots in the manifest.

## Fixed comparisons

| Name | Holdings | Rebalancing |
|---|---|---|
| fixed25_monthly | All 25 literal configured names, 4% targets | First exchange close of each calendar month |
| fixed25_four_week | Same 25 names, 4% targets | Existing four-week dates |
| ml25 | Highest 20 saved Ridge scores among those 25, 5% targets | Four weeks, no retention buffer |
| broad_monthly | All names in the latest available shared weekly eligibility snapshot | Calendar month |
| broad_four_week | All original decision-time eligible names | Four weeks |
| ml_broad | Original saved full-feature Ridge and 5% entry/10% retention buffer | Four weeks |
| spy | One 100% SPY position, held to liquidation | Opening and closing trades only |

The main new contrast is `ml25 - fixed25_four_week`. It matches dates, universe,
capital and frequency, while selecting 20 names instead of all 25 necessarily
changes concentration. Randomized top-20 portfolios preserve that holding count.
The monthly passive series are additional controls, not cadence-matched ML tests.

Each window starts independently with $100,000: June 2, 2025–March 9, 2026 for
development context; May 11–October 26, 2026 for the main historical comparison.
All are long-only, with no leverage and no short borrow charge. Targets are
defined on post-cost equity; weights drift until scheduled rebalancing.

The 25 names come from `Data_Pipeline/config.py` and are frozen in the manifest.
The model uses the existing broader-universe fits; scores are merely restricted
to those 25 names. Every decision requires all 25 finite scores. No target return
or future coverage determines eligibility or stock selection. Monthly snapshot
availability is checked at noon New York time; delayed execution uses the actual
session close, including early closes. The fixed baseline does not rank signals.
The configured list and broader covered universe are not historical index
membership records and can retain selection/survivorship biases.

## Ledger and measurements

The shared `PortfolioSpec` adds `selection_mode='all'` and
`selection_mode='fixed_count', selection_count=20`. Both require equal long-only
allocation without a retention policy; legacy ranked defaults are unchanged.
The 5% stock cap remains; SPY explicitly has a 100% ETF allowance.

Use the frozen IBKR approximation plus 1 bp execution allowance, raw execution
closes, adjusted total-return marks, netted orders, opening/rebalance/liquidation
charges, zero cash interest and no additional dividend cash credit. Returns
already reflect the adjusted-price dividend treatment. Gross controls have zero
trading costs. Missing required prices invalidate a case, with candidate coverage
reported separately and no retroactive deletion or filling.

Artifacts include daily equity/drawdown inputs, holdings, stock contributions,
trade/fee records, weekly returns, Sharpe, annualized volatility, daily drawdown,
concentration, turnover, NVDA exposure and invested exposure. SPY-relative
regressions use weekly returns, four-lag Newey–West uncertainty and zero cash rate.
Label the intercept SPY benchmark alpha, not internal-universe or FF5/momentum
alpha. Cumulative return differences are percentage-point outperformance, not
regression alpha. SPY is an ETF total-return proxy, not a directly observed index.

## Statistics and verification

Each ML universe receives 2,000 fixed-seed within-period size-decile score
shuffles, replaying gross and net portfolios together and retaining costs,
turnover, invalid reasons and conservative p-value bounds. Deterministic passive
portfolios are fixed comparators, not score-shuffle candidates. The 25-name
deciles contain only 2–3 names; this restricted null and changes in turnover and
persistence must be disclosed. Portfolio and ranking family sizes are 53 and 27,
including prior counted horizon/pruning variants and the new 25-name comparison;
they do not capture all historical research trials.

The 2,000 eight-calendar-week block samples are shared by all completed main-window
portfolios. Save return/Sharpe intervals and paired mean/cumulative outperformance
intervals. Six four-week decisions provide little precision. ML rank IC requires
200 known outcomes in the broad universe and 20 in the fixed universe. Passive
portfolios have no prediction IC. Missing observed/comparator paths make the
associated paired inference unavailable; never report valid-draw-only p-values.

The original +7.53% portfolio must reproduce gross/net return, Sharpe, drawdown
and costs within 1e-8. Captured and capture-free paths must agree. Freeze and
recheck source and prediction hashes. Run the research tests, including synthetic
full command, score-invariant passive weights, top-20 ties, calendars, missing
prices, raw commission units and no double-counted dividends.

Default to two workers, at most four, and one numerical thread per worker. Keep
prices, predictions, output manifests, results and reports private. Publish only
code, tests and these instructions on the consolidated branch. Negative returns
and explicit unavailable comparisons are valid completed research outcomes.
