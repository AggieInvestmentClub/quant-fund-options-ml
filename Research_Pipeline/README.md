# Options research pipeline

This is the **downstream research layer**. It takes dated IV/Greek features from
the calculator, joins stock data, constructs the paper's 12-month jump/crash
labels, trains models in chronological order, backtests monthly portfolios, and
runs Monte Carlo checks. The included synthetic fixture tests software behavior;
its returns are not research findings.

## Teammate handoff

Supply Parquet files in one input directory. Dates are calendar dates; every
`available_at` is a timezone-aware timestamp indicating when that row could
first have been used. All rows need `source_version` and a stable `security_id`.
If only tickers are available, also supply `ticker_history.parquet` with
`ticker`, `security_id`, `start_date`, and `end_date`; every ticker/date must map
to exactly one security. Ticker changes alone must not create a new stock.

| File | Required columns | Grain |
| --- | --- | --- |
| `options_features.parquet` | `trade_date`, `available_at`, `security_id`, `source_version`, `is_us_common_stock`, `avg_call_iv`, `avg_put_iv`, and `avg_{call,put}_{delta,gamma,theta,vega}` | One stock and option observation date. `is_us_common_stock` must describe that date. |
| `stock_prices.parquet` | `trade_date`, `available_at`, `security_id`, `source_version`, `close`, `adjusted_close`, `volume`, `market_cap` | One stock and trading date. Supply every month-end and subsequent rebalance close, including delisting total returns where applicable. A zero terminal adjusted value is allowed for a total loss; entry prices must be positive. |
| `stock_controls.parquet` | `as_of_date`, `available_at`, `security_id`, `source_version`, `AGE`, `DTURN`, `EXRET`, `TANG`, `TSKEW`, `TVOL`, `RM12`, `SG`, `SIZE` | One stock and control observation date. Values must reflect when information became public. |
| `sp500_membership.parquet` | `snapshot_date`, `available_at`, `security_id`, `source_version` | One row per member in each **complete historical snapshot**. Missing IDs in that snapshot are nonmembers. Snapshots older than 93 days are not used. |

The last two files may be omitted for an engineering pilot. A paper-style full
run requires the controls; the S&P result is unavailable without current-enough
historical snapshots. A full run also requires known US common-stock status for
every options row. The handoff rejects duplicate stock/date keys and ambiguous
ticker mappings. It records missing features, stock histories, controls, and
membership coverage instead of silently filling the universe.

The calculator's daily ATM feature files provide the ten `avg_*` columns.
Join the optional Default Strike features into `options_features.parquet` using
`trade_date` and `security_id`. This code recognizes `atm_iv_{30d,60d,90d}`,
`risk_reversal_25_delta_{30d,60d,90d}`, `put_skew_10_25_30d`,
`atm_term_slope`, and `rr_term_slope` as an **additional** feature group. These
are not part of the paper's original ten option variables. Keep raw contract
outputs separately for reconciliation.

The current Massive daily file uses last-trade close, while the paper used
OptionMetrics bid/ask midpoints and a broader 1996–2022 history. Record this
source difference in any research presentation. If 12-month labels, market-cap
history, or delisting returns are incomplete, do not treat the backtest as a
paper replication.

## Run

Install Python 3.10+ dependencies in an isolated environment. On macOS,
LightGBM also needs `libomp` (`brew install libomp`). Model execution uses one
thread per numerical library and LightGBM fit.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r Research_Pipeline/requirements.txt

# Fully synthetic, fast engineering demonstration (20 Monte Carlo draws).
.venv/bin/python -m Research_Pipeline.run \
  --synthetic-demo --output-dir /path/to/scratch/options-ml-demo

# Real-data run. The default is 2,000 Monte Carlo draws per model/universe.
.venv/bin/python -m Research_Pipeline.run \
  --input-dir /path/to/handoff --output-dir /path/to/scratch/options-ml-run

# Short-history real-data software pilot; explicitly inconclusive.
.venv/bin/python -m Research_Pipeline.run \
  --input-dir /path/to/handoff --output-dir /path/to/scratch/options-ml-pilot \
  --engineering-pilot --iterations 20
```

The full run needs at least 120 **matured** training months before the first
test month and 24 labeled test months. Otherwise `status.json` says
`blocked_data_gate`, and `coverage.json` explains what was present. The pilot
uses 12 training months and six test months only to verify plumbing; it is
always marked `engineering_pilot_inconclusive`.

Outputs include `manifest.json` (input hashes, versions and seed),
`coverage.json`, `monthly_panel.parquet`, `model_selection.json`,
`predictions.parquet`, `prediction_metrics.json`, `trades.parquet`,
`monthly_returns.parquet`, `portfolio_metrics.json`, `monte_carlo.json`, and
`status.json`. Save the complete output directory for reproducibility. Do not
copy market data or credentials into Git. Use a **fresh output directory** for
each run; the command refuses to mix new results with old files.

## Research definitions

- A stock/date enters the panel only when its option features exist on the
  stock market's last trading day of the month and are available by the first
  following trading close. Company controls must have been released by the
  prediction time and have `as_of_date` no later than that month-end.
- A 12-month adjusted-price **log return** above +0.70 is `jump`, below -0.70
  is `crash`; all other finite outcomes are `normal`. A label becomes trainable
  only after the future month-end price was available.
- Logistic regression and LightGBM are the paper-centered models. Feature
  groups are stock-only, ten option features only, combined, and combined plus
  skew. A small MLP is tested only on the extended group. Hyperparameters are
  selected on earlier matured months, then frozen for monthly refits. The final
  24 eligible labeled months are never used for tuning.
- Each prediction month buys the top 5% by jump probability and shorts the top
  5% by crash probability. Overlap goes to the higher-probability side. Each
  side is normalized to 100% market-cap weight. The position enters at the
  first close after the option data became available and exits at the next
  rebalance close. Costs are 30 basis points per traded notional, plus reported
  0% and 3% annual short-borrow cases. Missing entry/exit prices fail the run.
- On the untouched test months, 2,000 probability-pair shuffles within month
  and market-cap decile form a random-signal null. Another 2,000 three-month
  moving-block resamples form 95% intervals for annualized net return and
  Sharpe. The report also gives a conservative family-wise p-value adjustment
  for the multiple model/feature combinations in each universe. All random
  draws use a saved seed.

The software cannot infer borrow availability, exact spreads, historical
provider revisions, or the paper's missing OptionMetrics filters from these
tables. Report those limitations with any real-data result.
