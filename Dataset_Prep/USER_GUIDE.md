# Options & Stock Daily Feature Dataset: User & Operational Guide

This guide explains how to query, engineer, and apply the consolidated daily options and equity dataset for **machine learning modeling**, **algorithmic backtesting**, and **daily automated cron operations**.

---

## 1. Dataset Overview & Storage Architecture

The dataset unites underlying stock price action, historical shares outstanding, ATM options averages, default-strike volatility smile metrics, and risk-reversals into a single daily table.

### File Locations
- **Consolidated Monolithic Parquet:**  
  `/srv/data/options_model_features/ml_dataset_daily.parquet` (~138 MB, 318,202 rows)
- **Year-Partitioned Directory (Hive-style):**  
  `/srv/data/options_model_features/ml_dataset_daily/year=YYYY/`
- **Shares Outstanding Master File:**  
  `/srv/data/stocks/shares_outstanding.parquet` (5.38 MB, 474,103 rows)

### Key Specifications
- **Grain:** Exactly one row per `(underlying, trade_date)`.
- **Date Range:** `2024-04-01` to `2026-11-09` (655 trading days; only 1 missing business day on `2026-08-03`).
- **Universe:** 545 unique US optionable tickers, covering point-in-time S&P 500 members (`is_sp500 == True` on 99.7% of rows).
- **Compression:** Parquet with Zstandard (`zstd`) compression and strict PyArrow type casting.

---

## 2. Reading & Querying the Dataset

Because the dataset is stored in standard Apache Parquet with column projection and dictionary encoding, you can load individual columns or dates without reading the entire dataset into memory.

### Option A: Python / Pandas & PyArrow (Standard In-Memory)
```python
import pandas as pd

# 1. Read specific columns with date filtering
df = pd.read_parquet(
    "/srv/data/options_model_features/ml_dataset_daily.parquet",
    columns=[
        "trade_date", "underlying", "tradable_date", "is_sp500",
        "stock_adj_close", "stock_close", "stock_volume", "market_cap",
        "avg_call_iv", "avg_put_iv", "risk_reversal_25_delta", "put_skew_10_25"
    ],
    filters=[("trade_date", ">=", "2025-01-01")]
)

# 2. Filter for S&P 500 constituents at trade time
sp500_df = df[df["is_sp500"] == True].copy()
```

### Option B: DuckDB (Zero-Memory Overhead & Lightning Fast SQL)
DuckDB allows SQL queries directly on the Parquet file without loading into RAM:
```python
import duckdb

con = duckdb.connect()

# Query high-volatility regime stocks in S&P 500
query = """
SELECT 
    trade_date,
    underlying,
    tradable_date,
    stock_close,
    market_cap,
    avg_call_iv,
    risk_reversal_25_delta
FROM '/srv/data/options_model_features/ml_dataset_daily.parquet'
WHERE is_sp500 = true
  AND trade_date BETWEEN '2026-01-01' AND '2026-11-09'
  AND avg_call_iv > 0.40
ORDER BY trade_date, market_cap DESC
"""
df_high_iv = con.execute(query).df()
```

### Option C: Polars (High-Throughput Vectorized Processing)
```python
import polars as pl

lazy_df = (
    pl.scan_parquet("/srv/data/options_model_features/ml_dataset_daily.parquet")
    .filter(pl.col("is_sp500") == True)
    .select([
        "trade_date", "underlying", "stock_adj_close",
        "avg_call_iv", "risk_reversal_25_delta"
    ])
    .collect()
)
```

---

## 3. Machine Learning Workflows

### 3.1 Label Construction: Jump / Crash Targets
To train models that predict market crashes, volatility surges, or sharp upside jumps, you must construct target labels from forward stock returns.

> [!IMPORTANT]
> **Always use `stock_adj_close` for forward return calculation.**  
> Never compute forward returns from unadjusted `stock_close`. Unadjusted prices do not account for stock splits or dividend distributions; using them will produce severe false "crash" labels (e.g. an artificial -50% return on a 2-for-1 split).

```python
import numpy as np
import pandas as pd

# Load required price series sorted chronologically
df = pd.read_parquet(
    "/srv/data/options_model_features/ml_dataset_daily.parquet",
    columns=["trade_date", "underlying", "stock_adj_close", "is_sp500"]
).sort_values(["underlying", "trade_date"]).reset_index(drop=True)

# 1. Compute Forward k-day Returns (e.g., k = 5 business days)
k_days = 5
df["forward_5d_adj_close"] = df.groupby("underlying")["stock_adj_close"].shift(-k_days)
df["forward_5d_return"] = (df["forward_5d_adj_close"] - df["stock_adj_close"]) / df["stock_adj_close"]

# 2. Define Crash and Jump Labels
# Example A: Fixed Thresholds (e.g., <= -7% crash, >= +7% jump)
df["label_crash"] = (df["forward_5d_return"] <= -0.07).astype(int)
df["label_jump"] = (df["forward_5d_return"] >= 0.07).astype(int)

# Example B: Rolling Volatility-Standardized Jumps (Z-Score)
# Prevents high-beta tech stocks from dominating labels vs. low-vol utility stocks
df["daily_log_ret"] = df.groupby("underlying")["stock_adj_close"].pct_change()
df["hist_vol_20d"] = df.groupby("underlying")["daily_log_ret"].transform(lambda s: s.rolling(20).std() * np.sqrt(5))
df["z_forward_5d"] = df["forward_5d_return"] / df["hist_vol_20d"]

df["label_crash_z"] = (df["z_forward_5d"] <= -2.0).astype(int)
df["label_jump_z"] = (df["z_forward_5d"] >= 2.0).astype(int)
```

### 3.2 Feature Selection & Handling Missingness
The dataset provides two tiers of volatility and Greek signals:

| Feature Group | Key Columns | Availability | Recommended Usage |
|---|---|---|---|
| **ATM Greeks (10 Averages)** | `avg_call_iv`, `avg_put_iv`, `avg_call_delta`, `avg_put_delta`, `avg_call_gamma`, `avg_put_gamma`, `avg_call_theta`, `avg_put_theta`, `avg_call_vega`, `avg_put_vega` | 95%–97.5% | Core baseline features. Clean, dense coverage across almost all trading days. |
| **Surface Skew & RR (All Exp.)** | `risk_reversal_25_delta`, `put_skew_10_25`, `call_skew_10_25`, `atm_iv`, `skew_slope`, `gamma_rr_25_delta` | 70%–92% | Measures directional options market positioning (calls vs. puts across all 30–90 DTE contracts). |
| **Bucketed DTE Features** | `*_30d`, `*_60d`, `*_90d` (e.g., `risk_reversal_25_delta_30d`, `atm_iv_30d`) | 25%–70% | High-fidelity term structure slices, subject to monthly cycle gaps. |

#### How to Handle Missing Values in 30d/60d/90d Buckets
Stocks without weekly options series (only standard 3rd-Friday monthly expirations) cycle in and out of fixed DTE windows (e.g., 20–40 DTE). When contracts fall outside the slice window, bucket features are `NaN`.

**Recommended Handling Strategies:**
1. **Fallback Coalescing:** Fall back to the broad 30–90 DTE aggregate if the bucket is `NaN`:
   ```python
   df["feature_rr_30d_robust"] = df["risk_reversal_25_delta_30d"].fillna(df["risk_reversal_25_delta"])
   df["feature_atm_iv_30d_robust"] = df["atm_iv_30d"].fillna(df["atm_iv"])
   ```
2. **Missingness Indicator:** Add a binary flag `is_bucket_missing = df["risk_reversal_25_delta_30d"].isna().astype(int)`. Tree models (LightGBM/XGBoost) naturally handle NaNs if paired with indicator flags.
3. **Cross-Sectional Standard Scaling:** For linear or deep models, standardize features per `trade_date` across all active stocks:
   ```python
   def cross_sectional_zscore(group):
       return (group - group.mean()) / (group.std() + 1e-8)
   
   df["cs_risk_reversal"] = df.groupby("trade_date")["risk_reversal_25_delta"].transform(cross_sectional_zscore)
   ```

### 3.3 Splitting Rules: Purged Time-Series Cross Validation
> [!CAUTION]
> **NEVER use random `train_test_split` or standard $K$-fold cross-validation!**  
> Because forward $k$-day returns overlap across adjacent calendar days, a random split leaks future returns from the training set into the test set.

**Proper Splitting Practice:**
1. **Time-Based Boundaries:** Split by date (e.g., Train: `2024-04-01` to `2025-12-31`; Val: `2026-01-01` to `2026-06-30`; Test: `2026-07-01` to `2026-11-09`).
2. **Purge Window:** When computing a $k$-day forward return, purge $k$ days between Train and Val/Test sets so that no label touches future evaluation bars.

---

## 4. Algorithmic Backtesting Workflows

### 4.1 Strict Lookahead Guard & Execution Timing
To reflect realistic live execution, backtesters must strictly respect when options data is delivered:

- **OPRA Flatfile Delivery Time:** Options market aggs for trade date $T$ are published overnight and ingested around **07:15 UTC on $T+1$** (approx. 2:15 AM EST).
- **Metadata Columns:**
  - `options_available_time`: Timestamp when day $T$'s data was verified available (e.g. `2026-11-10 07:15:00`).
  - `tradable_date`: The earliest trading day on which signals can be executed (points to $T+1$).

```
Timeline:
[Day T: Market Open to Close 09:30 - 16:00 EST] --> Trading occurs
[Day T Night / Day T+1 Morning: ~02:15 EST / 07:15 UTC] --> Flatfiles arrive & features compute
[Day T+1 Market Open: 09:30 EST] --> First possible trade execution (tradable_date)
```

> [!WARNING]
> **Zero-Lookahead Rule:**  
> A signal computed from features on `trade_date = T` CANNOT trade at `stock_close` on date $T$.  
> Execution MUST occur on `tradable_date` (day $T+1$) using `stock_open`, an intraday VWAP, or day $T+1$ `stock_close`.

### 4.2 Fill Pricing vs. Return Accounting
- **Use Unadjusted Prices for Trade Execution & Sizing:**  
  When simulating order fills, slippage, and position sizing, use `stock_open` or `stock_close` (unadjusted). This represents the actual dollar quote at the exchange.
  $$\text{Shares to Buy} = \left\lfloor \frac{\text{Target Allocation (\$)}}{\text{stock\_open}_{T+1}} \right\rfloor$$
- **Use Adjusted Prices for Long-Horizon Return Tracking:**  
  Use `stock_adj_close` when auditing synthetic portfolio mark-to-market performance over multi-week holding periods to incorporate dividends and stock splits.

### 4.3 Liquidity & Market Cap Filtering
To prevent testing strategies on illiquid names that cannot be traded at scale:
```python
# Filter candidate trade signals on tradable_date
liquid_universe = df[
    (df["is_sp500"] == True) &
    (df["market_cap"] >= 2.0e9) &             # Minimum $2B market cap
    (df["stock_volume"] >= 500_000) &          # Minimum 500k shares daily volume
    (df["stock_close"] >= 5.0)                 # Exclude penny stocks
]
```

---

## 5. Daily Cron Operations & Maintenance

### 5.1 The Three-Stage Architecture
The daily pipeline runs in three ordered stages:
1. **Stage 1 (`data_copy/run_main_to_srv.sh`):** Ingests OPRA flatfiles, synchronizes stock CSVs, and computes ATM normalized options & Greek averages.
2. **Stage 2 (`Default_Strike_Pipeline/run_default_strike_daily.sh`):** Computes default strike volatility surfaces, delta buckets (10/25/50 delta), and 30d/60d/90d skew and risk-reversals.
3. **Stage 3 (`Dataset_Prep/run_export_daily.sh`):** Joins Stages 1 & 2 with stock OHLCV, shares outstanding, and S&P 500 snapshots into the daily Parquet table.

### 5.2 Unified Daily Runner
The entire pipeline is run via the master orchestrator:
```bash
/home/ben_options/Data_Prep/Default_Strike_Pipeline/run_daily_pipeline.sh
```

### 5.3 Crontab Setup
To schedule the daily run automatically on weekdays after OPRA flatfiles arrive (e.g. 04:15 EST / 09:15 UTC Tuesday through Saturday):
```cron
# Run Daily Options & ML Prep Pipeline at 04:15 AM EST (Tue-Sat)
15 4 * * 2-6 /bin/bash -lc "cd /home/ben_options/Data_Prep && ./Default_Strike_Pipeline/run_daily_pipeline.sh" >> /home/ben_options/Data_Prep/logs/daily_cron_master.log 2>&1
```

### 5.4 Manual Runs & Catch-Up Operations
If the cron run fails, or if you need to reprocess or backfill a specific date:

#### Running Stage 3 for a Single Target Date
```bash
cd /home/ben_options/Data_Prep
./Dataset_Prep/run_export_daily.sh --target-date 2026-11-10
```

#### Rebuilding the Entire Historical Monolithic Parquet File
To re-export all dates from scratch and build both partitioned and single files:
```bash
cd /home/ben_options/Data_Prep
.venv/bin/python Dataset_Prep/export_daily_parquet.py --build-single-file
```

#### Periodically Refreshing Shares Outstanding & Market Cap
Historical shares outstanding change with quarterly corporate filings (10-Q / 10-K). Run this refresh script once a week or month:
```bash
cd /home/ben_options/Data_Prep
.venv/bin/python Dataset_Prep/fetch_shares_outstanding.py
```

### 5.5 Monitoring & Troubleshooting

#### Check Recent Execution Logs
```bash
# View the latest Stage 3 export log
tail -n 60 /home/ben_options/Data_Prep/logs/export_parquet_latest.log

# View the latest master cron run log
tail -n 60 /home/ben_options/Data_Prep/logs/daily_cron_master.log
```

#### Resolving Pipeline Lock Files
All pipeline stages use Linux `flock` to prevent concurrent colliding runs. If an execution is abruptly killed or terminated by the OS, a stale lock file might prevent the next run:
- Master Lock: `/srv/data/options_model_features/main_pipeline.lock`
- Stage 2 Lock: `/srv/data/options_model_features/default_strike_pipeline.lock`
- Stage 3 Lock: `/srv/data/options_model_features/export_parquet_pipeline.lock`

If you verify no pipeline processes are running (`ps aux | grep python`), you can safely remove the lock file:
```bash
rm -f /srv/data/options_model_features/export_parquet_pipeline.lock
```

### 5.6 Data Health Check & Audit
To verify that all partitions and files are consistent, without missing prices or corrupted schemas:
```bash
cd /home/ben_options/Data_Prep
.venv/bin/python Dataset_Prep/audit_and_validate.py
```
This script checks:
- Trading calendar continuity against NYSE/NASDAQ schedule.
- Row counts and stock coverage per day.
- Price positivity (`stock_close > 0`).
- Feature missingness matrix (saved to `Dataset_Prep/DATASET_AUDIT_REPORT.md`).

---

## 6. Critical Caveats & Quirks

1. **One Missing Historical Day (`2026-08-03`):**
   - On `2026-08-03`, upstream OPRA flatfiles were empty. 655 of 656 expected business days are fully present.
   - *Impact on ML/Backtests:* Always use calendar date diffs or reindex with a continuous trading day index rather than assuming `row[i-1]` is exactly 1 calendar day prior.
2. **Survivorship Bias Mitigation (`is_sp500`):**
   - The column `is_sp500` is a **point-in-time snapshot** based on official index composition as of `trade_date`.
   - Never backtest using only stocks that are currently in the S&P 500 today. Always filter by `df["is_sp500"] == True` for each historical date.
3. **Monthly vs. Weekly Options Missingness:**
   - Approximately 50% of S&P 500 stocks do not have weekly options expirations.
   - For these tickers, contracts only fall into the 30-day bucket (20–40 DTE) every four weeks. When analyzing skew trends, prefer the full-surface aggregate `risk_reversal_25_delta` or use fallback coalescing.
4. **Market Cap Calculation & Shares Outstanding Forward Filling:**
   - SEC filings occur quarterly. Shares outstanding values are forward-filled between filing dates.
   - `market_cap` is dynamically calculated per row as $\text{stock\_close} \times \text{shares\_outstanding}$.
5. **Precision & Memory Optimization:**
   - Option Greeks and IV metrics are cast to `float32` (saving 50% RAM compared to standard `float64`).
   - Prices (`stock_close`, `stock_open`) and `market_cap` are retained in `float64` to prevent floating-point rounding errors in dollar amounts.

