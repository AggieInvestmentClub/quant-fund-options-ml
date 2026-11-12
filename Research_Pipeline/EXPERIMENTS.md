# Existing-data paper diagnostic and weekly strategy

Run both experiments using a frozen copy of the existing wide options file and stock CSVs. No market-data download is performed. Run from the repository root. Python dependencies are in `Research_Pipeline/requirements.txt`. Maximum CPU affinity is four cores; numerical libraries and each model use one thread.

```bash
python -m Research_Pipeline.strategy_experiments freeze \
  --wide-input /srv/data/options_model_features/ml_dataset_daily.parquet \
  --stocks-dir /srv/data/stocks --output-dir /private/path/snapshot
python -m Research_Pipeline.strategy_experiments run \
  --snapshot /private/path/snapshot --output-dir /private/path/results \
  --mode both --iterations 2000 --max-cores 4
```

The snapshot records SHA-256 hashes before and after copying every file. Runs verify the frozen files again before and after execution. Both snapshot and results must use fresh directories. Run `--mode weekly` or `--mode paper_diagnostic` separately for troubleshooting. Keep the snapshot, models, results, and Parquet files outside the public repository.

## Paper-method diagnostic

Month-end observations predict 12-month adjusted log returns: jump above +0.70, crash below -0.70, normal otherwise. Signal months are May–October 2025, traded after publication at the following monthly close schedule. Only labels available strictly before the decision enter training. Initial history is roughly April–May 2024, expanding monthly. Four adaptations compare IV pair versus ten IV/Greek features, with balanced logistic regression and balanced LightGBM. Inputs are median-filled and standardized within their own monthly cross section. LightGBM uses chronological 80/20 fitting/validation months and five-round early stopping; missing classes or validation produce an explicit not-estimable result.

This is a limited paper-method diagnostic. The paper used 1996–2022, ten initial training years, optionable US common stocks, bid/ask midpoints, and nine company controls. The 19 controlled Table 3 specifications are unavailable here. The report records unverified 31–90-day maturity, volume, bid/ask, midpoint, no-arbitrage and IV contract filters. Published performance is not a pass/fail target for six diagnostic months.

## Weekly strategy

Daily inputs are sampled on each week's last NYSE session. Decisions occur at noon ET on the first following session and positions enter at its actual close. Readiness is conservatively at least 11:30 ET on the supplied next tradable date. The target is adjusted return to the next weekly entry close. A label becoming known at today's close cannot train today's noon model.

The last 26 completed periods form the frozen test. Before each test decision, at least 52 matured training weeks are required; each test week requires 200 stocks. Eligibility requires both IVs, a $5 stock close, positive cap, unambiguous keys, and 20 sessions of history. All models use identical rows. Stock features are trailing 5/20-session returns, 20-session volatility and dollar volume, and log cap. Option features are ten averages, their IV gap, and available ATM IV/skew/term columns. Training-only median imputation, missing indicators, and scaling are used.

Ridge(alpha=10) and LightGBM(100 trees, 15 leaves, learning rate .05, minimum leaf samples 50) each run stock-only, option-only and combined features. An IV-gap ranking is the seventh comparison. Combined Ridge is the predeclared primary strategy. Every test fit uses expanding matured history; no test tuning or winner replacement occurs. Models and exact training windows are retained.

Each period longs the top 5% and shorts the bottom 5%, value-weighting each leg to +1/-1. A daily cash/position ledger solves for post-fee equity, accrues 3% annual short borrow by actual calendar days, and charges 30bp per traded notional at opening, rebalancing after drift, and final liquidation. Gross, zero-borrow net, and 10/50bp sensitivities are also reported. A missing selected daily price invalidates that strategy; insolvency is recorded. Cash earns zero and stock financing costs beyond borrow are unmodeled.

Metrics include cumulative/annualized return, periodic Sharpe and volatility (sqrt(52) weekly, sqrt(12) monthly), daily maximum drawdown, turnover, fees, hit rate, and both legs. Weekly rank IC and paired net return differences from stock-only Ridge measure incremental option information. An equal-weight available-universe benchmark supplies a four-lag Newey–West internal-universe alpha proxy, not published FF5+momentum alpha.

## Monte Carlo and decision

For each variant, 2,000 score shuffles within period and contemporaneous cap decile rebuild the same daily ledger under 30bp and 3% borrow. Monthly class probabilities move together. The one-sided statistic is mean net period return; p=(1+null>=observed)/(2001). Bonferroni families are seven weekly and four monthly variants. Invalid random portfolios make inference inconclusive rather than being silently discarded.

Another 2,000 moving-block resamples use four-week blocks for weekly results and three-month blocks for the six monthly observations. Report 95% intervals for mean return, annualized compounded return, and Sharpe; six-month diagnostic intervals are highly uncertain. Seed is 17.

Combined Ridge is a promising pilot only if all gates pass: positive primary net cumulative return; Sharpe >=1; daily drawdown >=-20%; higher mean net return than stock-only Ridge; adjusted randomization p<.05; and bootstrap lower bound for mean net return >0. Otherwise the strategy is not supported. Partial/invalid variants yield inconclusive status. Implementation completion does not require profitability.

## Artifacts and scoring

Each track writes predictions, holdings, daily equity, periodic returns, training windows, metrics, Monte Carlo results, model files, and an equity/drawdown chart. The paper track includes descriptives, classification metrics, nonestimable cases, and replication deviations. The weekly track includes success gates, incremental option value, a feature schema and latest primary model/scores.

```bash
python -m Research_Pipeline.strategy_experiments score \
  --model /private/path/results/weekly/latest_model.joblib \
  --panel /private/path/results/weekly/weekly_panel.parquet \
  --output /private/path/latest_scores.parquet
```

Only load model artifacts created by this project in trusted private storage. Scores are research outputs; no orders are submitted. Ticker proxies, unavailable delisting outcomes, last-trade IV, incomplete historical membership and unknown borrow availability limit claims to the covered-stock universe.

## Final audit when valuation gaps remain

After a completed run, reconcile the saved models and ledger and create `FINAL_REPORT.md`:

```bash
python -m Research_Pipeline.final_report \
  --results /private/path/results \
  --prices /private/path/results/handoff/stock_prices.parquet
```

This verifies unchanged equity/performance, fee/cash reconciliation and model-score round trips. It lists exact missing prices, retains the invalid primary strategy, and gives conservative p-value bounds for incomplete randomizations. Original model-run files remain intact; corrected daily fee-event logs and verification hashes are in `final_audit/`. Run this once per result directory. Unresolved outcomes remain unavailable; the audit never drops affected holdings or fabricates a terminal value.

## Conditional missing-price stress test

Use saved predictions and the exact original adapted-price hash:

```bash
python -m Research_Pipeline.price_stress \
  --results /private/path/results \
  --prices /private/path/results/handoff/stock_prices.parquet \
  --output-dir /private/path/results/price_stress
```

The five scenarios apply -25%, -10%, 0%, +10%, or +25% to each missing run relative to its last observed adjusted close, jumping on the first missing session and staying flat until an observed price resumes or the assumed exit. Recorded prices, predictions, rankings, weights, and the original report remain unchanged. A selected missing entry close is prohibited; liquidation at an assumed exit is explicitly hypothetical. Costs and borrow use the original ledger. The output includes all strategies, assumed price cells, affected positions, scenario equity curves and verification checks. These are conditional outcomes, not historical corrections, probability estimates, recovered alpha, or validated backtests. No model fitting or new market data is involved.
