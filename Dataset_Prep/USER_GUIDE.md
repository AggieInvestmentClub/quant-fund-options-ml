# Options & Stock Daily Feature Dataset: User & Operational Guide

This guide is the complete operational and technical reference for the **165-column consolidated daily options and equity dataset** designed for **machine learning modeling**, **algorithmic backtesting**, and **daily automated cron operations**.

---

## 1. Dataset Overview & Storage Architecture

The dataset unifies daily stock price action, historical shares outstanding, market cap, academic company controls, ATM option averages, volatility smile metrics, risk-reversals, and forward multi-horizon targets into a single daily table.

### File Locations
- **Consolidated Monolithic Parquet:**  
  `/srv/data/options_model_features/ml_dataset_daily.parquet` (**179.2 MB, 318,202 rows, 165 columns**)
- **Year-Partitioned Directory (Hive-style):**  
  `/srv/data/options_model_features/ml_dataset_daily/year=YYYY/` (656 daily partitions, fully synchronized with multi-day rolling lags)
- **Shares Outstanding Master File:**  
  `/srv/data/stocks/shares_outstanding.parquet` (5.42 MB, 477,384 rows)

### Key Specifications
- **Grain:** Exactly one row per `(underlying, trade_date)`.
- **Primary Key Uniqueness:** Verified **0 duplicate rows** across all 318,202 observations.
- **Date Range:** `2024-04-01` to `2026-11-09` (655 trading days; only 1 missing business day on `2026-08-03` due to upstream OPRA delivery).
- **Universe:** 545 unique US optionable tickers, covering point-in-time S&P 500 members (`is_sp500 == True` on 99.70% of rows).
- **Investable Universe:** 235,216 observations (`is_options_investable == True`, 73.92% of universe, ~359 stocks/day) with dense surface skew and zero micro-cap noise.
- **Stock Price Coverage:** **100.00% fully populated** across all 318,202 observations (including 14 acquired/delisted S&P 500 tickers recovered from OPRA archives).
- **Compression:** Parquet with Zstandard (`zstd`) compression and strict PyArrow explicit schema types.

---

## 2. Reading & Querying the Dataset

Because the dataset is stored in standard Apache Parquet with column projection and dictionary encoding, you can load individual columns or dates without reading the entire dataset into memory.

### 2.1 Pulling Options-Investable Data (Recommended for ML & Production Models)

> [!TIP]
> **Always filter for `is_options_investable == True` when training machine learning models or backtesting quantitative options strategies.**
> 
> Illiquid options data contains wide bid-ask bounce, zero-volume noise, and missing expiration cycles that degrade model gradients and create false "phantom alpha". Filtering on `is_options_investable` ensures that:
> 1. The stock is an actively trading S&P 500 constituent (`is_sp500 == True`, `is_actively_trading == True`).
> 2. The stock has institutional liquidity (`stock_close >= $5.00`, `market_cap >= $2.0B`, `stock_volume >= 100,000`).
> 3. The option chain is active and priced (`n_contracts_total >= 10`, `iv_level` is valid).
> 
> This filter boosts skew and surface feature completeness to **>93%–100%** while preserving **235,216 rows (~359 liquid stocks per day)**.

#### Option A: Python / Pandas & PyArrow (Standard In-Memory)
```python
import pandas as pd

# Load options-investable universe directly with Parquet pushdown filtering
df = pd.read_parquet(
    "/srv/data/options_model_features/ml_dataset_daily.parquet",
    columns=[
        "trade_date", "underlying", "tradable_date", "is_sp500", "is_options_investable",
        "stock_adj_close", "stock_close", "market_cap",
        # Academic controls (Andreou, Han, and Li 2025)
        "size", "tvol_20d", "tskew_60d", "rm1", "rm12", "exret_daily", "dturn_20d",
        # Compact option signals
        "iv_level", "iv_gap", "variance_proxy", "risk_reversal_25_delta_30d_robust", "atm_iv_30d_robust",
        # Multi-horizon forward targets
        "forward_5d_return", "jump_label_5d", "crash_label_5d"
    ],
    filters=[
        ("trade_date", ">=", "2025-01-01"),
        ("is_options_investable", "==", True)
    ]
)
```

#### Option B: DuckDB (Zero-Memory Overhead & High-Speed SQL)
DuckDB queries directly on the Parquet file on disk without loading unneeded columns into RAM:
```python
import duckdb

con = duckdb.connect()

query = """
SELECT 
    trade_date,
    underlying,
    tradable_date,
    stock_close,
    market_cap,
    iv_level,
    iv_gap,
    variance_proxy,
    risk_reversal_25_delta_30d_robust,
    forward_5d_return,
    crash_label_5d
FROM '/srv/data/options_model_features/ml_dataset_daily.parquet'
WHERE is_options_investable = true
  AND trade_date BETWEEN '2026-01-01' AND '2026-11-09'
  AND iv_level > 0.35
ORDER BY trade_date, market_cap DESC
"""
df_high_iv = con.execute(query).df()
```

#### Option C: Polars (High-Throughput Vectorized Processing)
```python
import polars as pl

lazy_df = (
    pl.scan_parquet("/srv/data/options_model_features/ml_dataset_daily.parquet")
    .filter(pl.col("is_options_investable") == True)
    .select([
        "trade_date", "underlying", "stock_adj_close", "size", "tvol_20d",
        "iv_level", "iv_gap", "variance_proxy", "forward_5d_return", "crash_label_5d"
    ])
    .collect()
)
```

#### Option D: Built-in Python Helper Function
```python
from Dataset_Prep.enrich_features import filter_investable_universe

raw_df = pd.read_parquet("/srv/data/options_model_features/ml_dataset_daily.parquet")
investable_df = filter_investable_universe(raw_df)
```

---

## 3. Comprehensive Feature Availability & Accuracy Master Matrix

All 165 columns have been audited across the full 318,202 observations. The tables below detail availability (% populated) and audited value domains (Min, Median, Max) for each functional group:

### 3.1 Primary Identifiers & Lookahead Metadata (5 columns)
| Column | Type | Populated % | Value Domain / Description |
|---|---|---|---|
| `trade_date` | `string` | 100.00% | Date of market session (`YYYY-MM-DD`). |
| `underlying` | `string` | 100.00% | Stock ticker symbol (545 unique symbols). |
| `is_sp500` | `bool` | 100.00% | Point-in-time S&P 500 constituent flag (**99.70% True**). |
| `options_available_time` | `string` | 100.00% | Canonical timestamp when options aggs arrived (07:15 UTC on $T+1$). |
| `tradable_date` | `string` | 100.00% | First business day on which trade can execute ($T+1$). |

### 3.2 Underlying Stock OHLCV & Market Cap (16 columns)
| Column | Type | Populated % | Min | Median | Max | Notes |
|---|---|---|---|---|---|---|
| `stock_open` | `float64` | 100.00% | 0.0000 | 129.47 | 5815.00 | Exchange open quote (**100% complete**). |
| `stock_high` | `float64` | 100.00% | 6.7100 | 131.05 | 5839.41 | Session high quote. |
| `stock_low` | `float64` | 100.00% | 6.6400 | 127.85 | 5707.93 | Session low quote. |
| `stock_close` | `float64` | 100.00% | 6.7100 | 129.46 | 5815.92 | Unadjusted closing price (**100.0% strictly positive**). |
| `stock_adj_close` | `float64` | 100.00% | 6.7100 | 122.90 | 4354.54 | Split & dividend backward-adjusted price. |
| `stock_adj_open` | `float64` | 100.00% | 0.0000 | 122.88 | 4354.54 | Split-adjusted open quote (eliminates split cliff distortion). |
| `stock_adj_high` | `float64` | 100.00% | 6.7100 | 124.38 | 4372.82 | Split-adjusted high quote. |
| `stock_adj_low` | `float64` | 100.00% | 6.6400 | 121.36 | 4274.07 | Split-adjusted low quote. |
| `stock_volume` | `float64` | 100.00% | 0.00 | 2.56M | 875.20M | Daily traded share volume. |
| `dollar_volume` | `float64` | 100.00% | $0.00 | $331.2M | $82.4B | Dollar traded volume: $\text{stock\_close} \times \text{stock\_volume}$. |
| `dividend_yield` | `float64` | 100.00% | 0.00% | 1.48% | 14.17% | Annualized dividend yield. |
| `historical_beta` | `float64` | 100.00% | -1.19 | 0.78 | 4.57 | Trailing beta to market benchmark. |
| `stock_beta` | `float64` | 100.00% | -1.19 | 0.78 | 4.57 | Coalesced equity beta. |
| `shares_outstanding` | `float64` | 100.00% | 16.09M | 332.14M | 26.34B | Point-in-time reported shares outstanding. |
| `market_cap` | `float64` | 100.00% | $3.80B | $39.26B | $5.78T | $\text{stock\_close} \times \text{shares\_outstanding}$. |

### 3.3 Academic Company Controls (*Andreou, Han, and Li 2025*) (10 columns)
| Column | Type | Populated % | Min | Median | Max | Economic Meaning / Formula |
|---|---|---|---|---|---|---|
| `size` | `float32` | 100.00% | 22.06 | 24.39 | 29.38 | Firm size: $\ln(\text{market\_cap})$. |
| `tvol_20d` | `float32` | 99.98% | 1.50% | 26.24% | 642.79% | Trailing 20-day annualized realized volatility ($\sigma \times \sqrt{252}$). |
| `tvol_60d` | `float32` | 99.98% | 1.50% | 27.97% | 375.86% | Trailing 60-day annualized realized volatility with 20d fallback. |
| `tskew_60d` | `float32` | 99.94% | -7.49 | -0.03 | 7.25 | Trailing 60-day return skewness. |
| `rm1` | `float32` | 99.91% | -63.06% | +0.78% | +192.29% | 1-month trailing return momentum ($P_t / P_{t-21} - 1$). |
| `rm3` | `float32` | 99.91% | -71.36% | +2.37% | +379.73% | 3-month trailing return momentum with 1-month compounded fallback. |
| `rm12` | `float32` | 99.91% | -97.20% | +12.49% | +15936.51% | 12-month trailing return momentum with quarterly/monthly compounded fallback. |
| `exret_daily` | `float32` | 100.00% | -53.16% | +0.04% | +176.95% | Daily stock return minus daily risk-free rate ($R_f / 252$). |
| `dturn_20d` | `float32` | 99.98% | -1.10 | -0.0006 | 4.64 | Detrended turnover vs. 20-day trailing average. |
| `dturn_252d` | `float32` | 99.98% | -1.19 | -0.0005 | 5.15 | Detrended turnover vs. 252-day trailing average with 20d fallback. |

### 3.4 10 ATM Call/Put IV and Greek Averages (10 columns)
| Column | Type | Populated % | Min | Median | Max | Notes |
|---|---|---|---|---|---|---|
| `avg_call_iv` | `float32` | 97.47% | 0.0038 | 0.3065 | 2.6369 | Near-the-money call implied volatility. |
| `avg_put_iv` | `float32` | 94.82% | 0.0241 | 0.3285 | 2.8435 | Near-the-money put implied volatility. |
| `avg_call_delta` | `float32` | 97.47% | 0.4693 | 0.5368 | 1.0000 | Average ATM call delta. |
| `avg_put_delta` | `float32` | 94.82% | -0.5761 | -0.4655 | -0.3203 | Average ATM put delta. |
| `avg_call_gamma` | `float32` | 97.47% | 0.0003 | 0.0307 | 3.6794 | Average ATM call gamma. |
| `avg_put_gamma` | `float32` | 94.82% | 0.0005 | 0.0302 | 2.0946 | Average ATM put gamma. |
| `avg_call_theta` | `float32` | 97.47% | -3409.52 | -23.45 | -0.0955 | Average ATM call theta. |
| `avg_put_theta` | `float32` | 94.82% | -8601.14 | -22.03 | -0.1056 | Average ATM put theta. |
| `avg_call_vega` | `float32` | 97.47% | 0.0013 | 18.73 | 942.55 | Average ATM call vega. |
| `avg_put_vega` | `float32` | 94.82% | 0.7762 | 18.71 | 1026.21 | Average ATM put vega. |

### 3.5 Surface Skew & Smile Features (30 columns)
| Column | Type | Populated % | Min | Median | Max | Notes |
|---|---|---|---|---|---|---|
| `atm_iv` | `float32` | 99.97% | 0.0038 | 0.2935 | 3.3346 | Composite ATM IV across all surface contracts (**99.97% dense**). |
| `put_25_delta_iv` | `float32` | 83.70% | 0.0321 | 0.3164 | 2.1271 | 25-delta put implied volatility. |
| `call_25_delta_iv` | `float32` | 86.41% | 0.0038 | 0.2882 | 2.0862 | 25-delta call implied volatility. |
| `risk_reversal_25_delta` | `float32` | 75.68% | -1.8229 | -0.0249 | 1.0640 | Full surface 25-delta RR ($\text{Call}_{25\Delta} - \text{Put}_{25\Delta}$). |
| `put_skew_10_25` | `float32` | 69.26% | -0.4079 | 0.0366 | 1.0543 | 10-delta to 25-delta put wing slope. |
| `call_skew_10_25` | `float32` | 67.57% | -0.4137 | 0.0270 | 0.9991 | 10-delta to 25-delta call wing slope. |
| `curvature_25_delta` | `float32` | 75.68% | -0.5891 | 0.0076 | 0.7247 | Smile curvature ($\text{Call}_{25} + \text{Put}_{25} - 2 \times \text{ATM}$). |
| `gamma_rr_25_delta` | `float32` | 75.68% | -0.3297 | 0.0008 | 0.9634 | Gamma risk reversal. |
| `gamma_imbalance_sum_raw`| `float32` | 99.69% | -108.97 | -0.0084 | 148.88 | Total gamma imbalance (call gamma minus put gamma). |

### 3.6 Compact Option Signals & Generalized Slopes (16 columns)
*Specially designed to eliminate multicollinearity in Ridge/LightGBM regressions:*
| Column | Type | Populated % | Min | Median | Max | Notes / Formula |
|---|---|---|---|---|---|---|
| `iv_level` | `float32` | 99.97% | 0.0038 | 0.3161 | 3.3346 | Near-the-money IV level with multi-source fallback (**99.97% dense**). |
| `iv_gap` | `float32` | 93.37% | -2.6206 | -0.0209 | 2.2900 | $\text{avg\_call\_iv} - \text{avg\_put\_iv}$ (put skew proxy). |
| `variance_proxy` | `float32` | 99.95% | -40.7101 | 0.0272 | 11.0362 | $(\text{iv\_level})^2 - (\text{tvol\_20d})^2$ (**99.95% dense**). |
| `iv_to_rv_ratio` | `float32` | 99.95% | 0.0125 | 1.1560 | 18.4210 | Dimensionless ratio: $\text{iv\_level} / \text{tvol\_20d}$. |
| `delta_5d_iv_gap` | `float32` | 88.38% | -2.2950 | -0.0005 | 2.3146 | 1-week change in IV gap. |
| `delta_20d_iv_gap` | `float32` | 85.62% | -2.3020 | -0.0012 | 2.7686 | 4-week change in IV gap. |
| `delta_5d_rr_60d` | `float32` | 89.85% | -1.7879 | -0.0001 | 1.8091 | 1-week change in 60-day risk reversal. |
| `atm_term_slope` | `float32` | 63.11% | -2.2216 | -0.0065 | 1.6427 | Generalized multi-point ATM term slope (**81.51% in investable**). |
| `put_skew_term_slope` | `float32` | 52.83% | -2.1628 | -0.0043 | 1.4533 | Generalized 25-delta put skew term slope (**70.48% in investable**). |
| `rr_term_slope` | `float32` | 46.41% | -1.9545 | -0.0029 | 2.0200 | Generalized risk reversal term slope (**62.63% in investable**). |
| `risk_reversal_25_delta_30d_robust`| `float32`| 94.26% | -1.8229 | -0.0255 | 1.0640 | Multi-tier robust 30d RR (**99.93% in investable**). |
| `risk_reversal_25_delta_60d_robust`| `float32`| 94.26% | -1.8229 | -0.0260 | 1.0204 | Multi-tier robust 60d RR (**99.93% in investable**). |
| `atm_iv_30d_robust` | `float32` | 99.97% | 0.0038 | 0.2932 | 3.3346 | Multi-tier robust 30d ATM IV (**100.00% in investable**). |
| `iv_level_cs_rank` | `float32` | 99.97% | 0.0020 | 0.5000 | 1.0000 | Same-day cross-sectional percentile rank of IV level (stationary for ML). |
| `risk_reversal_30d_cs_rank` | `float32` | 94.26% | 0.0020 | 0.5000 | 1.0000 | Same-day cross-sectional percentile rank of 30d robust risk reversal. |
| `variance_proxy_cs_rank` | `float32` | 99.95% | 0.0020 | 0.5000 | 1.0000 | Same-day cross-sectional percentile rank of variance proxy. |

### 3.7 Forward Return Targets, Labels & Activity Flags (20 columns)
| Column | Type | Populated % | Value Domain | Notes / Formula |
|---|---|---|---|---|
| `forward_1d_return` | `float32` | 100.00% | -53.14% to +36.74% | Legacy next-day close-to-close return ($P^{\text{adj}}_{t+1} / P^{\text{adj}}_t - 1$). |
| `forward_5d_return` | `float32` | 99.51% | -55.34% to +173.88% | Legacy 1-week close-to-close forward return. |
| `forward_10d_return` | `float32` | 98.74% | -55.82% to +209.98% | Legacy 2-week close-to-close forward return. |
| `forward_21d_return` | `float32` | 96.96% | -63.06% to +192.29% | Legacy 1-month close-to-close forward return. |
| `forward_63d_return` | `float32` | 90.96% | -71.36% to +379.73% | Legacy 3-month close-to-close forward return. |
| `target_open_to_close_1d` | `float32` | 99.84% | -52.40% to +35.10% | **Tradable 1-day intraday return** ($P^{\text{adj}}_{C, T+1} / P^{\text{adj}}_{O, T+1} - 1$, zero overnight bias). |
| `target_tradable_5d_return` | `float32` | 99.51% | -54.90% to +170.20% | **Tradable 5-day holding return** from entry at $T+1$ Open to $T+5$ Close. |
| `target_tradable_21d_return` | `float32` | 96.96% | -62.80% to +188.50% | **Tradable 21-day holding return** from entry at $T+1$ Open to $T+21$ Close. |
| `forward_5d_excess_return` | `float32` | 99.51% | -57.20% to +171.40% | 5-day stock return minus daily S&P 500 cross-sectional mean return (alpha). |
| `forward_21d_excess_return` | `float32` | 96.96% | -65.10% to +185.30% | 21-day stock return minus daily S&P 500 cross-sectional mean return (alpha). |
| `outperformance_label_5d` | `bool` | 100.00% | **49.80% True** | Binary indicator: `forward_5d_excess_return > 0.0`. |
| `outperformance_label_21d` | `bool` | 100.00% | **49.70% True** | Binary indicator: `forward_21d_excess_return > 0.0`. |
| `is_target_valid_5d` | `bool` | 100.00% | **99.51% True** | Guard flag: True when 5d forward return has realized; False near end of history. |
| `is_target_valid_21d` | `bool` | 100.00% | **96.96% True** | Guard flag: True when 21d forward return has realized; False near end of history. |
| `jump_label_5d` | `bool` | 100.00% | **11.26% True** | Binary indicator: `forward_5d_return >= +5%`. |
| `crash_label_5d` | `bool` | 100.00% | **9.26% True** | Binary indicator: `forward_5d_return <= -5%`. |
| `jump_label_21d` | `bool` | 100.00% | **13.38% True** | Binary indicator: `forward_21d_return >= +10%`. |
| `crash_label_21d` | `bool` | 100.00% | **8.93% True** | Binary indicator: `forward_21d_return <= -10%`. |
| `is_actively_trading` | `bool` | 100.00% | **100.00% True** | `True` if `volume > 0` and stock price is valid. |
| `is_options_investable` | `bool` | 100.00% | **73.92% True** | Precomputed liquid, institutional options universe flag (**235,203 rows**). |

---

### 3.8 Deep-Dive: Why Options-Investable Filtering Dramatically Improves ML Performance

When training machine learning models on options surfaces (e.g. predicting forward returns, jump/crash probabilities, or directional excess returns), **filtering out illiquid options data significantly improves out-of-sample accuracy and prevents backtest degradation**.

#### The Three Structural Failure Modes of Unfiltered Options Data
1. **Bid-Ask Bounce and Quote Noise:**
   - On illiquid tickers (e.g., small REITs or low-volume utilities averaging $<5$ contracts per day), trades occur sporadically at the wide bid or ask.
   - This creates artificial 500–1,000 bps swings in implied volatility and extreme skew fluctuations that reflect dealer spread rather than informational order flow.
2. **Gradient Distortion in Tree-Based & Regression Models:**
   - In LightGBM, XGBoost, or Ridge regression, loss functions (e.g. MSE or Log-loss) are quadratic in prediction errors.
   - Outlier IV readings from 1-contract odd-lots distort tree splits, causing the model to overfit on spurious signals that do not generalize to liquid market leaders.
3. **Elimination of "Phantom Alpha":**
   - A model trained on the raw universe may detect statistical alpha in thinly traded tickers.
   - In live trading, these signals cannot be monetized because the market lacks depth, resulting in 100% slippage.
   - Filtering on `is_options_investable == True` ensures that the model learns patterns on names that can actually be executed in institutional size.

#### Empirical Feature Availability: All Rows vs. Investable Universe
Filtering by `is_options_investable == True` retains **235,216 observations (~359 liquid stocks per day)** while driving feature completeness to near-perfection:

| Feature Name | All Data Populated % | Investable Populated % | Absolute Gain |
|---|---|---|---|
| `stock_close` | **100.00%** | **100.00%** | 0.00% (100% complete) |
| `market_cap` | **100.00%** | **100.00%** | 0.00% (100% complete) |
| `stock_beta` | **100.00%** | **100.00%** | 0.00% (100% complete) |
| `atm_iv` | **99.97%** | **100.00%** | +0.03% |
| `iv_level` | **99.97%** | **100.00%** | +0.03% |
| `atm_iv_30d_robust` | **99.97%** | **100.00%** | +0.03% |
| `variance_proxy` | **99.95%** | **99.99%** | +0.04% |
| `risk_reversal_25_delta_30d_robust` | **94.26%** | **99.93%** | **+5.67%** |
| `risk_reversal_25_delta_60d_robust` | **94.26%** | **99.93%** | **+5.67%** |
| `atm_term_slope` | **63.11%** | **81.51%** | **+18.40%** |
| `put_skew_term_slope` | **52.83%** | **70.48%** | **+17.65%** |
| `rr_term_slope` | **46.41%** | **62.63%** | **+16.22%** |
| `delta_5d_iv_gap` | **88.38%** | **96.50%** | +8.12% |
| `risk_reversal_25_delta` (raw) | 75.68% | **93.01%** | +17.33% |
| `put_skew_10_25` (raw) | 69.26% | **87.81%** | +18.55% |

---

## 4. Machine Learning Workflows

### 4.1 Label Construction: Jump / Crash Targets
To train models that predict market crashes, volatility surges, or sharp upside jumps, you can use the precomputed binary targets:
- `jump_label_5d`: `forward_5d_return >= +5%` (11.26% base rate)
- `crash_label_5d`: `forward_5d_return <= -5%` (9.26% base rate)
- `jump_label_21d`: `forward_21d_return >= +10%` (13.38% base rate)
- `crash_label_21d`: `forward_21d_return <= -10%` (8.93% base rate)

> [!IMPORTANT]
> **Forward returns are always computed from `stock_adj_close`.**  
> Never compute forward returns from unadjusted `stock_close`. Unadjusted prices do not account for stock splits or dividend distributions; using them will produce severe false "crash" labels (e.g. an artificial -50% return on a 2-for-1 split).

### 4.2 Cross-Sectional Ranking & Z-Score Normalization
To prevent macroeconomic drift and regime bias from dominating cross-sectional stock picking, rank or standardize features across stocks on each individual trading date:

```python
def cross_sectional_standardize(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Normalize features per day across the cross-section (mean=0, std=1)."""
    df_norm = df.copy()
    for col in feature_cols:
        df_norm[col + "_zscore"] = df_norm.groupby("trade_date")[col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-8)
        )
    return df_norm
```

### 4.3 Handling Missing Data in Quant Models
1. **Primary recommendation:** Filter on `is_options_investable == True`. This eliminates over 70% of missing values across skew and smile features.
2. **Dense fallback columns:** Use `risk_reversal_25_delta_30d_robust` and `atm_iv_30d_robust`. These columns automatically coalesce term-structure bucket values with the full-surface aggregate.
3. **Cross-sectional median imputation:** If residual missing values remain in smaller sub-buckets (e.g. 90-day wing skews), impute using the daily cross-sectional median:
   ```python
   df[col] = df.groupby("trade_date")[col].transform(lambda s: s.fillna(s.median()))
   ```

---

## 5. Algorithmic Backtesting & Zero-Lookahead Execution Rules

To ensure backtest results match live execution reality, follow these three non-negotiable rules:

### 5.1 The $T+1$ Morning Availability Rule
OPRA end-of-day options data aggregates arrive early in the morning following the trade session ($T+1$ at approximately 07:15 UTC).
- Features calculated from date $T$ options data are **only available to trade on date $T+1$**.
- The column `tradable_date` explicitly indicates the earliest market open when orders can be executed.
- Never use date $T$ options features to execute at date $T$ Market Close.

### 5.2 Execution Prices: Unadjusted Open (`stock_open`) for Fills
In live market operations, trade fills occur at actual quoted exchange prices, not split-adjusted numbers:
- **Order Execution:** Use `stock_open` on `tradable_date` as the execution entry price.
- **Slippage Modeling:** Add 5–10 bps slippage to `stock_open` for conservative fill simulation.

### 5.3 Tracking P&L: Adjusted Close (`stock_adj_close`) for Returns
While executions occur at raw dollar prices, position mark-to-market and cumulative portfolio returns across holding periods spanning corporate actions must use `stock_adj_close` to prevent artificial split-cliff drawdowns.

---

## 6. Daily Pipeline Operations & Cron Automation

### 6.1 Three-Stage Pipeline Architecture
1. **Stage 1 (`Data_Pipeline/main.py`):** Downloads OPRA flatfiles, runs Leisen-Reimer American-option lattice pricing, and computes ATM Greek averages (10 call/put features).
2. **Stage 2 (`Default_Strike_Pipeline/run_default_strike_daily.sh`):** Computes default strike volatility surfaces, delta buckets (10/25/50 delta), and 30d/60d/90d skew and risk-reversals.
3. **Stage 3 (`Dataset_Prep/run_export_daily.sh`):** Joins Stages 1 & 2 with stock OHLCV, shares outstanding, academic controls, and S&P 500 snapshots into the daily Parquet table.

### 6.2 Unified Daily Runner
The entire pipeline is run via the master orchestrator:
```bash
/home/ben_options/Data_Prep/Default_Strike_Pipeline/run_daily_pipeline.sh
```

### 6.3 Crontab Setup
To schedule the daily run automatically on weekdays after OPRA flatfiles arrive (e.g. 04:15 EST / 09:15 UTC Tuesday through Saturday):
```cron
# Run Daily Options & ML Prep Pipeline at 04:15 AM EST (Tue-Sat)
15 4 * * 2-6 /bin/bash -lc "cd /home/ben_options/Data_Prep && ./Default_Strike_Pipeline/run_daily_pipeline.sh" >> /home/ben_options/Data_Prep/logs/daily_cron_master.log 2>&1
```

### 6.4 Manual Runs & Catch-Up Operations
If the cron run fails or if you need to reprocess or backfill a specific date:

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

### 6.5 Monitoring & Troubleshooting

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

### 6.6 Data Health Check & Audit
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

## 7. Critical Caveats & Quirks

1. **One Missing Historical Day (`2026-08-03`):**
   - On `2026-08-03`, upstream OPRA flatfiles were empty. 655 of 656 expected business days are fully present.
   - *Impact on ML/Backtests:* Always use calendar date diffs or reindex with a continuous trading day index rather than assuming `row[i-1]` is exactly 1 calendar day prior.
2. **Survivorship Bias Mitigation (`is_sp500`):**
   - The column `is_sp500` is a **point-in-time snapshot** based on official index composition as of `trade_date`.
   - Never backtest using only stocks that are currently in the S&P 500 today. Always filter by `df["is_sp500"] == True` for each historical date.
3. **Monthly vs. Weekly Options Missingness:**
   - Approximately 50% of S&P 500 stocks do not have weekly options expirations.
   - For these tickers, contracts only fall into the 30-day bucket (20–40 DTE) every four weeks. When analyzing skew trends, prefer the full-surface aggregate `risk_reversal_25_delta` or use fallback coalescing (`risk_reversal_25_delta_30d_robust`).
4. **Market Cap Calculation & Shares Outstanding Forward Filling:**
   - SEC filings occur quarterly. Shares outstanding values are forward-filled between filing dates.
   - `market_cap` is dynamically calculated per row as $\text{stock\_close} \times \text{shares\_outstanding}$.
5. **Precision & Memory Optimization:**
   - Option Greeks and IV metrics are cast to `float32` (saving 50% RAM compared to standard `float64`).
   - Prices (`stock_close`, `stock_open`) and `market_cap` are retained in `float64` to prevent floating-point rounding errors in dollar amounts.
