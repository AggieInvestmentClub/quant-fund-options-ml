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
thread per numerical library and LightGBM fit. The runner caps Arrow's CPU and
I/O pools at four threads and pins its process to at most four available cores
on Linux. Use `--max-cores 1`, `2`, or `4` to lower that cap; the manifest
records the choice and any Linux CPU affinity.

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
  following trading close. The trailing price month is treated as incomplete
  until a date from the following month exists. Company controls must have
  been released by the prediction time and have `as_of_date` no later than
  that month-end.
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

## Paper replication mode

`--experiment paper` runs the 19 Table 3 feature specifications (stock-only B,
and SDGTV/S/D/G/T/V for call-plus-put, call-only, and put-only) with both
class-balanced logit and LightGBM. Each stock variable is lagged one month,
except TANG and SG, which are lagged six months. Missing features are filled
by the same month's cross-sectional median, then all 19 inputs are z-scored
within that month. LightGBM uses its default tree parameters, a chronological
80/20 fitting/validation split, and five rounds of early stopping. Logit uses
all eligible training observations. The saved model report records each
training window and the LightGBM best iteration.
Paper mode also requires nonmissing call and put IV and a prior-month stock
closing price of at least $1. The coverage report counts rows removed by
these two filters. The upstream calculator must still establish the paper's
contract-level 31–90-day maturity, positive volume/bid, valid bid/ask,
$0.10 midpoint, no-arbitrage, and 0.01–5 IV filters. Aggregated rows cannot
prove those rules after the fact.

The paper says the January 2006 prediction was trained through December 2005,
but the late-2005 **12-month labels were unknowable in January 2006**. Two
explicit schedules make this discrepancy visible:

```bash
# Tradable schedule: only fully matured labels, at least 120 training months.
# With options beginning in 1996, the earliest possible start is in 2007.
.venv/bin/python -m Research_Pipeline.run \
  --input-dir /path/to/complete_handoff --output-dir /path/to/scratch/safe \
  --experiment paper --label-policy point_in_time \
  --paper-start 2007-01 --paper-end 2022-12 \
  --factor-file /path/to/monthly_factors.parquet

# Literal paper-stated 2006-2022 schedule. This uses future-known labels
# for the early training months and is explicitly a hindsight diagnostic.
.venv/bin/python -m Research_Pipeline.run \
  --input-dir /path/to/complete_handoff --output-dir /path/to/scratch/paper_stated \
  --experiment paper --label-policy paper_stated \
  --paper-start 2006-01 --paper-end 2022-12 \
  --factor-file /path/to/monthly_factors.parquet
```

Fetch and freeze the official French monthly FF5 and momentum archives into a
fresh scratch directory with:

```bash
.venv/bin/python -m Research_Pipeline.fetch_factors \
  --output-dir /path/to/scratch/french_factors
```

The factor file contains decimal returns, source hashes, and a release-vintage
manifest. `factor_metrics.json` reports monthly alpha and Newey-West t
statistics for gross and both net cost cases. The monthly factors are aligned
to the calendar month after the signal; our first-close-to-first-close holding
period differs slightly from calendar-month factors, so this alpha is an
approximation until matching daily factors are used. Current French factor
history may also differ from the paper's original data vintage.

`paper_comparison.json` compares all 38 model/specification portfolios with
the paper's Table 6 (including 30% gross annual arithmetic mean and 1.37
Sharpe for LGBM-SDGTV cp, versus 10% and 0.51 for stock-only LGBM-B).
`portfolio_metrics.json` includes both
compound annualized return and annualized arithmetic mean; use the latter
when comparing with the paper's reported mean. Reproducing those numbers
requires comparable option prices, universe, company controls, time window,
and delisting treatment. A difference is a research finding to investigate,
not automatically a software bug.
The portfolio report also separates the jump-stock long leg and crash-stock
short selection. `paper_baseline_tests.json` gives paired Newey-West tests of
each gross long-short strategy against the corresponding stock-only B model,
matching the role of the parenthesized comparisons in Table 6.
`paper_descriptives.json` records jump/crash/normal frequencies and raw IV/Greek
means for a Table 2/4 sanity check, even when the model run is blocked by
history. `coverage.json` includes month-by-month counts of usable, labeled,
and fully controlled stock observations.

`prediction_metrics.json` reports one-vs-rest accuracy ratio (`2 × AUC − 1`)
and precision/FDR among each month's top 5% picks. The paper does not fully
specify the implementation behind its Table 5 PPV/FDR numbers, so those two
columns are diagnostic here rather than asserted to reproduce Table 5.

## Ben's current wide file

The server file `/srv/data/options_model_features/ml_dataset_daily.parquet`
can be adapted without modifying it:

```bash
.venv/bin/python -m Research_Pipeline.run \
  --wide-input /srv/data/options_model_features/ml_dataset_daily.parquet \
  --stocks-dir /srv/data/stocks \
  --allow-ticker-proxy --engineering-pilot --iterations 20 \
  --pilot-train-months 3 --pilot-test-months 3 \
  --output-dir /path/to/fresh/scratch/ben_pilot
```

The adapter writes `adapted_handoff/` and `adapter_report.json` under that
scratch directory. `--allow-ticker-proxy` explicitly makes ticker a temporary
identifier for a software pilot; a full study requires `--ticker-history`
with stable IDs. It treats the supplied `is_sp500` value as an observation
flag, not as complete date-specific constituent snapshots, and marks US common
stock status unknown. It records how many supplied option availability times
precede an assumed 11:30 a.m. New York provider-readiness time on the next
tradable date; that assumption needs upstream publication logs before a live
trading claim. The wide file uses last-trade option prices, not the paper's
bid/ask midpoints. `--stocks-dir` uses the dated CSV stock histories for
entry/exit and 12-month outcome prices, checks overlapping closes against the
wide file, and preserves each source's hash. Market cap may be null on
supplemental price dates because it is used only when forming a stock/month
portfolio; a valid cap is still required on every selected option month-end.
Without `--stocks-dir`, missing wide-file prices are counted and excluded,
then coverage and backtest gates expose the impact.

If the upstream calculator's ten `avg_*.csv` files are available, reconcile
one complete month by stock/date and value before using the wide handoff:

```bash
.venv/bin/python -m Research_Pipeline.reconcile \
  --wide-file /srv/data/options_model_features/ml_dataset_daily.parquet \
  --features-dir /path/to/calculator/final_features \
  --month 2026-10 --output /path/to/scratch/reconciliation.json
```

The command fails on any missing/extra stock-date key or finite difference
above `1e-8`. A missing source feature CSV is a hard error. Preserve this
report with the adapter report and source hashes.

As audited on 2026-11-11, this file covers April 2024 through November 2026,
with 318,202 stock-days, 545 tickers, and 32 calendar months, the last of
which is incomplete. It is almost entirely S&P names. The file lacks the
paper's nine company controls and enough history for 120 matured training
months plus 24 labeled test months. Expect a coverage report and
possibly `blocked_data_gate` if labels, classes, or executable stock prices
are insufficient. A short-history run that passes is still an engineering
pilot, not paper or portfolio evidence. Obtain a broad
optionable-US-common-stock history back to roughly 1996 for literal
replication, dated stock controls, stable security IDs, quote-based option
inputs, and complete historical S&P snapshots before claiming a reproduction.
