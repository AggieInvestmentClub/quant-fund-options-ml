# Frozen prediction portfolio and cost comparison

This command replays both saved experiments without model fitting or downloading data. Results are exploratory because the historical test period was already inspected. Monthly equal weighting is a further adaptation, not reproduction of the paper's weighting.

```sh
python -m Research_Pipeline.reevaluate \
  --results /private/original-results \
  --prices /private/original-results/private_inputs/stock_prices.parquet \
  --output-dir /private/portfolio-cost-revision \
  --max-cores 4
```

Use a fresh output directory outside the public checkout. Inputs must match the original `final_audit/artifact_hashes.json`. The command first reproduces legacy metrics, holdings, daily equity and fees; it refuses changed inputs. It retains original stock selection and missing-price failures. It does not load models. Thread libraries are limited to one thread, with a four-core maximum.

## Predeclared configuration

Defaults are $100,000 equity, equal weights on each selected side, 5% of post-cost equity maximum per stock at rebalancing, 1 bp execution allowance and 3% annual borrow. The intended exposure remains +100%/-100%; with fewer than 20 stocks per side it is reduced equally. Price drift can breach the target cap between rebalances: this is reported, not traded daily. There are no sector controls.

Four main conditions isolate the effects: market-cap/30 bp control; market-cap/IBKR cost only; equal-weight/30 bp weight only; and equal-weight/IBKR both changes. Gross results are provided for both weightings. Both-changes sensitivities use $10,000/$1 million at 1 bp; 0/5 bp at $100,000; and zero borrow at $100,000/1 bp. Combined Ridge stays the weekly primary regardless of results.

An optional `--config /private/config.json` overrides these fields only:

```json
{
  "initial_capital": 100000,
  "max_position": 0.05,
  "execution_bps": 1,
  "borrow_rate": 0.03,
  "iterations": 2000,
  "seed": 17
}
```

Use the defaults for the declared experiment; changes are separate exploratory research. A small iteration count is only an engineering pilot, never the final inference.

## Commission and execution approximation

Source: https://www.interactivebrokers.com/en/pricing/commissions-stocks.php, inspected November 12, 2026. This is a **current-cost scenario** applied constantly across historical dates, not reconstructed historical charges.

Each security has one net order per rebalance, including reversals, with a separate final liquidation order. Its estimated share quantity is absolute dollar trade notional divided by the actual **unadjusted** close. Commission is `min(max(0.005 * shares, 1), 0.01 * notional)`. This applies the whole-share schedule to idealized share-equivalent quantities: fractional order pricing, actual shares, cent rounding, execution fills and broker statements are not reproduced. Adjusted prices and synthetic total-return units remain the P&L basis. A missing required unadjusted execution price fails the broker-cost run; it is never replaced by an adjusted price.

Separately accrue the published current regulatory schedule: SEC 0.0000206 times sell notional; CAT $0.000003 per share-equivalent; FINRA TAF $0 under the current October–December 2026 holiday. These are constant scenario rates, even for earlier dates. Fixed commission already incorporates normal exchange/clearing costs; do not add tiered pass-through charges. The 1 bp allowance combines spread and slippage and is assumed, not measured. Borrow accrues on prior-close short notional by actual calendar days. Cash earns zero; additional financing and borrow availability are not established.

## Validation and outputs

The dollar ledger reconciles cash/positions, trades/fees, long/short attribution and compounded returns. `verification.json` is saved for each complete condition. Market-cap control returns must reproduce the legacy values. Predictions and price hashes must be unchanged after the run. Rank IC is independent of construction and must match saved values where available.

Each condition saves holdings, trades, daily equity/exposure, periodic returns, component charges and stock attribution privately. `metrics.json` includes effective holdings, daily breaches, NVDA exposure and contributions, internal-universe alpha where available, and the cost/account-size sensitivities. Borrow's effective bps denominator is traded notional for display only, not its accrual base. `comparison.csv`, `REPORT.md`, source verification and the manifest provide the run summary. Incremental options value uses paired differences against the same model's stock-only variant within each condition.

For both-changes only, every model gets 2,000 paired gross/net within-size-decile score shuffles with fixed seeds. Monthly probability vectors move together. Save invalid draws and conservative p-value bounds; do not discard missing-price portfolios. Missing observed results have no p-value. Multiplicity families are seven weekly/four monthly variants; sensitivities are descriptive. The paired gross comparison exposes the role of turnover/costs, but independent period-wise shuffling is not a turnover-matched null. Bootstrap uses 2,000 four-week/three-month block samples. Six monthly returns are especially weak evidence. Neither simulations nor lower costs repair the supplied price anomalies or establish paper replication.
