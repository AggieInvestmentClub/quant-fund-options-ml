# ML & Backtesting Dataset Quality & Audit Report

## 1. Summary Statistics
- **Total Rows:** 319,181
- **Trading Days:** 657 (from `2024-04-01` to `2026-11-11`)
- **Unique Underlyings:** 545
- **Expected Business Days:** 658
- **Missing Business Days:** 1 (2026-08-03)
- **Stocks per Day:** Mean = 485.8, Median = 491 (Min: 1, Max: 502)
- **S&P 500 Membership:** 318,218 rows (99.7%)
- **Investable Universe (`is_options_investable`):** 235,984 rows (73.9%)
- **Stock Price Availability:** 100.00% populated (100.00% strictly positive)

## 2. Feature Availability & Missingness Matrix
| Category | Feature Name | Populated % | Missing (NaN) % |
|---|---|---|---|
| ATM Greeks | `avg_call_iv` | 97.47% |  2.53% |
| ATM Greeks | `avg_put_iv` | 94.83% |  5.17% |
| ATM Greeks | `avg_call_delta` | 97.47% |  2.53% |
| ATM Greeks | `avg_put_delta` | 94.83% |  5.17% |
| ATM Greeks | `avg_call_gamma` | 97.47% |  2.53% |
| ATM Greeks | `avg_put_gamma` | 94.83% |  5.17% |
| ATM Greeks | `avg_call_theta` | 97.47% |  2.53% |
| ATM Greeks | `avg_put_theta` | 94.83% |  5.17% |
| ATM Greeks | `avg_call_vega` | 97.47% |  2.53% |
| ATM Greeks | `avg_put_vega` | 94.83% |  5.17% |
| Surface Greeks | `risk_reversal_25_delta` | 75.69% | 24.31% |
| Skew & RR | `put_skew_10_25` | 69.28% | 30.72% |
| Surface Greeks | `atm_iv` | 99.97% |  0.03% |
| 30d/60d/90d Greeks | `risk_reversal_25_delta_30d` | 57.37% | 42.63% |
| Skew & RR | `put_skew_10_25_30d` | 52.69% | 47.31% |
| 30d/60d/90d Greeks | `atm_iv_30d` | 70.01% | 29.99% |
| 30d/60d/90d Greeks | `risk_reversal_25_delta_60d` | 48.41% | 51.59% |
| Skew & RR | `put_skew_10_25_60d` | 42.84% | 57.16% |
| 30d/60d/90d Greeks | `atm_iv_60d` | 64.46% | 35.54% |
| 30d/60d/90d Greeks | `risk_reversal_25_delta_90d` | 29.24% | 70.76% |
| Skew & RR | `put_skew_10_25_90d` | 25.32% | 74.68% |
| 30d/60d/90d Greeks | `atm_iv_90d` | 38.36% | 61.64% |
| Stock OHLCV | `stock_adj_close` | 100.00% |  0.00% |
| Stock OHLCV | `stock_close` | 100.00% |  0.00% |
| Stock OHLCV | `stock_open` | 100.00% |  0.00% |
| Stock OHLCV | `stock_adj_open` | 100.00% |  0.00% |
| Stock OHLCV | `stock_volume` | 100.00% |  0.00% |
| Stock OHLCV | `dollar_volume` | 100.00% |  0.00% |
| Stock OHLCV | `shares_outstanding` | 100.00% |  0.00% |
| Stock OHLCV | `market_cap` | 100.00% |  0.00% |
| Academic Controls | `size` | 100.00% |  0.00% |
| Academic Controls | `tvol_20d` | 99.98% |  0.02% |
| Academic Controls | `tvol_60d` | 99.98% |  0.02% |
| Academic Controls | `tskew_60d` | 99.94% |  0.06% |
| Academic Controls | `rm1` | 99.91% |  0.09% |
| Academic Controls | `rm3` | 99.91% |  0.09% |
| Academic Controls | `rm12` | 99.91% |  0.09% |
| Academic Controls | `exret_daily` | 100.00% |  0.00% |
| Academic Controls | `dturn_20d` | 99.98% |  0.02% |
| Academic Controls | `volume_ratio_20d` | 99.98% |  0.02% |
| Compact Signals | `iv_level` | 99.97% |  0.03% |
| Compact Signals | `iv_gap` | 93.37% |  6.63% |
| Compact Signals | `variance_proxy` | 99.95% |  0.05% |
| Compact Signals | `iv_to_rv_ratio` | 99.95% |  0.05% |
| Compact Signals | `iv_level_cs_rank` | 99.97% |  0.03% |
| Compact Signals | `risk_reversal_25_delta_30d_robust` | 94.26% |  5.74% |
| Compact Signals | `atm_iv_30d_robust` | 99.97% |  0.03% |
| Targets & Activity | `forward_1d_return` | 99.84% |  0.16% |
| Targets & Activity | `forward_5d_return` | 99.21% |  0.79% |
| Targets & Activity | `forward_21d_return` | 96.66% |  3.34% |
| Targets & Activity | `target_open_to_close_1d` | 99.84% |  0.16% |
| Targets & Activity | `target_tradable_5d_return` | 99.21% |  0.79% |
| Targets & Activity | `forward_5d_excess_return` | 99.21% |  0.79% |
| Targets & Activity | `is_actively_trading` | 100.00% |  0.00% |
| Targets & Activity | `is_options_investable` | 100.00% |  0.00% |

## 3. Data Integrity & Lookahead Guard
- All option rows include `options_available_time` reflecting canonical $T+1$ morning availability (07:15 UTC).
- `tradable_date` explicitly indicates the execution date without lookahead bias.
- Output formatted as high-performance Parquet with ZSTD compression and PyArrow explicit dtypes.