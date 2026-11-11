# ML & Backtesting Dataset Quality & Audit Report

## 1. Summary Statistics
- **Total Rows:** 318,202
- **Trading Days:** 655 (from `2024-04-01` to `2026-11-09`)
- **Unique Underlyings:** 545
- **Expected Business Days:** 656
- **Missing Business Days:** 1 (2026-08-03)
- **Stocks per Day:** Mean = 485.8, Median = 491 (Min: 1, Max: 502)
- **S&P 500 Membership:** 317,239 rows (99.7%)
- **Stock Price Availability:** 98.97% populated (100.00% strictly positive)

## 2. Feature Availability & Missingness Matrix
| Category | Feature Name | Populated % | Missing (NaN) % |
|---|---|---|---|
| ATM Greeks | `avg_call_iv` | 97.47% |  2.53% |
| ATM Greeks | `avg_put_iv` | 94.82% |  5.18% |
| ATM Greeks | `avg_call_delta` | 97.47% |  2.53% |
| ATM Greeks | `avg_put_delta` | 94.82% |  5.18% |
| ATM Greeks | `avg_call_gamma` | 97.47% |  2.53% |
| ATM Greeks | `avg_put_gamma` | 94.82% |  5.18% |
| ATM Greeks | `avg_call_theta` | 97.47% |  2.53% |
| ATM Greeks | `avg_put_theta` | 94.82% |  5.18% |
| ATM Greeks | `avg_call_vega` | 97.47% |  2.53% |
| ATM Greeks | `avg_put_vega` | 94.82% |  5.18% |
| Surface Greeks | `risk_reversal_25_delta` | 75.68% | 24.32% |
| Skew & RR | `put_skew_10_25` | 69.26% | 30.74% |
| Surface Greeks | `atm_iv` | 91.90% |  8.10% |
| 30d/60d/90d Greeks | `risk_reversal_25_delta_30d` | 57.33% | 42.67% |
| Skew & RR | `put_skew_10_25_30d` | 52.64% | 47.36% |
| 30d/60d/90d Greeks | `atm_iv_30d` | 69.96% | 30.04% |
| 30d/60d/90d Greeks | `risk_reversal_25_delta_60d` | 48.37% | 51.63% |
| Skew & RR | `put_skew_10_25_60d` | 42.79% | 57.21% |
| 30d/60d/90d Greeks | `atm_iv_60d` | 64.41% | 35.59% |
| 30d/60d/90d Greeks | `risk_reversal_25_delta_90d` | 29.33% | 70.67% |
| Skew & RR | `put_skew_10_25_90d` | 25.40% | 74.60% |
| 30d/60d/90d Greeks | `atm_iv_90d` | 38.48% | 61.52% |
| Stock OHLCV | `stock_adj_close` | 98.97% |  1.03% |
| Stock OHLCV | `stock_close` | 98.97% |  1.03% |
| Stock OHLCV | `stock_volume` | 98.97% |  1.03% |
| Stock OHLCV | `shares_outstanding` | 98.97% |  1.03% |
| Stock OHLCV | `market_cap` | 98.97% |  1.03% |

## 3. Data Integrity & Lookahead Guard
- All option rows include `options_available_time` reflecting canonical $T+1$ morning availability (07:15 UTC).
- `tradable_date` explicitly indicates the execution date without lookahead bias.
- Output formatted as high-performance Parquet with ZSTD compression and PyArrow explicit dtypes.