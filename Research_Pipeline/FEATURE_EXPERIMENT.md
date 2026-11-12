# Weekly feature ablation

This experiment retrains fixed Ridge and LightGBM models on existing frozen inputs. It does not change the IV calculator, acquire data, tune hyperparameters, or modify the monthly paper diagnostic. Results are exploratory: the original holdout and pooled correlation chart have already been inspected.

```sh
python -m Research_Pipeline.feature_experiments \
  --panel /private/original/weekly/weekly_panel.parquet \
  --prices /private/original/private_inputs/stock_prices.parquet \
  --reference-results /private/original \
  --output-dir /private/new-feature-results \
  --max-cores 4
```

Use a fresh private output directory outside the public checkout. Input hashes must match the original `final_audit/artifact_hashes.json`. The default is 2,000 simulations and seed 17. `--iterations` can reduce simulations for an engineering pilot only. No new model is fitted on future-known labels. Numerical libraries use one thread; CPU affinity is restricted on supported servers and the process never requests more than four cores. Fits are sequential to keep memory and CPU usage bounded.

## Fixed feature sets

Each group is run with both models, for eight variants:

| Group | Numeric feature count | Definition |
|---|---:|---|
| stock_only | 5 | Existing 5/20-session return, 20-session volatility/dollar-volume, log cap |
| full | 25 | The exact existing stock plus options inputs, in their original order |
| compact | 13 | Five stock inputs plus the eight options inputs below |
| compact_plus | 17 | Compact plus the four alternative signals below |

Compact options: `(avg_call_iv+avg_put_iv)/2`, `iv_gap`, `avg_call_delta`, `avg_call_gamma`, `avg_call_vega`, `risk_reversal_25_delta_60d`, `put_skew_10_25_30d`, `atm_term_slope`. The level and gap preserve the information in the two average IVs while avoiding the redundant third representation. Dropping other inputs is an experiment, not proof that they are uninformative.

Alternative signals: IV-gap change over exactly one and four scheduled weeks, one-week change in the 60-day risk reversal, and `iv_level**2 - 252*vol_20**2`. The last is an unmatched-horizon variance proxy, not a measured variance risk premium. Lags join by security and exact W-FRI period, including holiday-shortened weeks. Missing weeks are not bridged or interpolated. No target enters feature generation. New features do not alter eligibility or row order. The numerical counts exclude missingness indicators added by the existing preprocessing pipeline.

The existing training-only median imputer/missingness indicators and standard scaler are retained. All-missing training features remain in the schema. Sparse skew and term signals do not remove observations. Before-imputation Pearson matrices, pair counts, nonmissing counts and scoring coverage are saved for each training window. They do not automatically select features.

## Chronology and controls

Assessment: 52 signals May 2, 2025–April 24, 2026, reported in four consecutive 13-week blocks. Each fit expands only through outcomes known strictly before its decision; approximately 55 matured training weeks are available at the first fit.

Historical comparison: the original 26 holds May 11–November 9, 2026. The command first retrains stock-only/full Ridge and LightGBM and checks every saved score against the original reference, maximum absolute difference `1e-8`, with exact stock-week keys and targets. A failure stops the experiment before candidate fitting. Compact variants are then fitted, and all eight are fitted for the earlier assessment.

Models stay fixed: Ridge `alpha=10`; LightGBM 100 trees, 15 leaves, learning rate .05, minimum 50 samples per leaf, seed 17, one thread. Model artifacts include the pipeline, explicit feature order/definitions, source hashes, target and training cutoff. The original Combined Ridge primary remains identified; `compact_plus/ridge` is the predeclared exploratory challenger versus `full/ridge`.

## Prediction and trading measurements

Every variant retains identical stock-week rows and known-outcome masks. Weekly metrics are Spearman rank IC, MSE, five equally populated predicted-return buckets and highest-minus-lowest realized return spread. Ties use stable security ordering. At least 200 finite outcomes are required. Missing outcomes are disclosed and excluded from prediction metrics only, never used to filter portfolio selection.

Both assessment and historical windows are backtested with the shared audited dollar ledger: $100,000, equal weights, 5% rebalance cap, +100%/-100% target exposure, current IBKR Fixed approximation plus 1 bp execution and 3% annual borrow; gross is separate. See PORTFOLIO_REVISION.md for exact fee approximations and limitations. Missing selected daily or execution prices invalidate the affected portfolio while ranking evaluation continues. Outputs include component charges, turnover, drawdown, concentration/drift, effective names, NVDA and stock/leg contributions. Internal-universe alpha is reported only with a complete benchmark; no factor-alpha claim is made.

## Inference and research screens

For the 26-week historical window, run 2,000 size-decile score shuffles per variant. One donor permutation is shared across all variants for each week/draw and across gross/net portfolios. Rank IC uses the fixed known-outcome masks. Keep randomized gross/net returns, costs, turnover and failure reasons. Bonferroni family size is eight, separately for ranking and portfolio p-values. Missing portfolio draws yield conservative bounds and no complete point p-value. Independent weekly shuffles change persistence and are not a turnover-matched null.

Run 2,000 paired four-week moving-block bootstrap samples, using common indices across models, for mean IC, the primary IC difference and available gross/net return and Sharpe series. Paired weekly IC contrasts cover reduction, alternatives, combined change and incremental options information versus stock-only. The earlier 52 weeks provide block-level descriptive stability evidence, not a second independent holdout or a parameter search.

The challenger is called promising for further prediction research only if its mean IC improvement is positive in the assessment and at least three of four blocks; its historical mean IC is positive; the 95% paired IC-difference interval is above zero; and adjusted ranking p is below .05. These retrospective screens do not validate trading profitability or reproduce the paper. All variants and failed screens are reported.

## Artifacts and tests

Private outputs contain source/configuration/split manifests, control reproduction errors, per-fit models and training diagnostics, predictions, weekly ranking/quintile metrics, block summaries, paired IC differences, holdings/trades/equity, cost and attribution metrics, paired random draws, shared bootstrap indices, confidence intervals and success gates. Original source hashes are checked again at completion. Keep all data and fitted/derived artifacts out of Git; publish code, tests and instructions only.

```sh
python -m pytest -q Research_Pipeline
```

Tests cover exact lags, holidays/missing weeks/security separation, future-input/label exclusion, strict maturity, fixed feature counts, all-missing features and training-only preprocessing, control-score reproduction, shared keys/outcome masks, deterministic paired simulations and a synthetic fit-to-ledger-to-inference run. Existing portfolio and source-mutation checks remain in force.

### Coverage gaps and safe fit reuse

Every scheduled nonempty stock-week is predicted even when its universe has fewer than 200 names. Ranking/MSE/quintile statistics remain unavailable below 200 known outcomes, and a one-stock universe cannot form two disjoint portfolio sides. In the frozen earlier assessment, April 2 and April 10, 2026 each have one eligible stock. Retain both weeks: report 50 estimable ranking weeks out of 52, with 11 rather than 13 in the last block. Do not invent rows or bridge the gap. The complete 52-week improvement screen cannot pass with unavailable scoring weeks; partial point estimates remain descriptive.

A restart can use `--fit-cache /private/trusted-prior-attempt` in addition to a **fresh** output directory. Only use this project's own trusted artifacts. The prior manifest must match input hashes, feature definitions, software and seed; each model must match feature order, training metadata and estimator/preprocessor parameters. Reused models are re-scored, copied to the new result directory with fit provenance, and controls still pass the full original-prediction reproduction gate. Cache reuse does not skip test weeks, alter training data or change model settings.
