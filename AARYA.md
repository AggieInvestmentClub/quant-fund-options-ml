# 5 Critical Rules for Real ML Profit & Backtesting

To avoid false alpha, backtest leakage, and live execution losses, follow these 5 non-negotiable quantitative modeling rules:

---

### 1. Train on Tradable Targets, NOT Legacy Targets
- **The Trap:** Training on `forward_1d_return` or `forward_5d_return` assumes order entry at date $T$ Close (16:00 EST). However, OPRA options files arrive at 07:15 UTC on morning $T+1$. This creates artificial "phantom alpha" from the overnight gap that cannot be captured in live markets.
- **The Rule:** Train strictly on:
  - `target_tradable_5d_return`: Enters at $T+1$ Open and exits at $T+5$ Close (realistic execution).
  - `forward_5d_excess_return`: Stock return minus the daily S&P 500 cross-sectional mean (pure relative stock alpha, immune to macro market beta).

---

### 2. Prevent Time-Series Data Leakage (Purging & Embargoing)
- **The Trap:** Standard `train_test_split(shuffle=True)` or scikit-learn standard `KFold` causes fatal lookahead bias. A 5-day or 21-day holding period creates overlapping future price paths; shuffling leaks future labels into training folds, producing an artificial 95%+ win rate that collapses out-of-sample.
- **The Rule:** Use strict **chronological time-series splits** with an **embargo buffer of at least 5 to 21 trading days** between train and validation/test folds. Also filter `is_target_valid_5d == True` to drop trailing historical boundary days where future returns have not yet realized.

---

### 3. Account for Transaction Costs & Slippage (Turnover Drag)
- **The Trap:** Daily rebalancing strategies generate massive portfolio turnover. In real markets, equity commissions, exchange fees, bid-ask spreads, and market impact cost **10 to 15 bps round-trip per trade**. A strategy trading daily across 50 stocks bleeds **25% to 35% annually in friction alone**.
- **The Rule:** Target **5-day to 21-day holding horizons** rather than daily churning. Deduct a mandatory **10 bps friction penalty per trade** in the backtester. Edge that survives realistic transaction costs is true institutional alpha.

---

### 4. Avoid Multicollinearity in Linear Models (Use Compact Features)
- **The Trap:** The dataset has 183 columns, including dozens of raw Greeks and maturities (`atm_iv_30d`, `atm_iv_60d`, `avg_call_iv`, `avg_put_iv`) that correlate at $>0.95$. Feeding all columns directly into Ridge, Lasso, OLS, or Neural Networks destabilizes weights and blows up out-of-sample variance.
- **The Rule:** Train linear models and neural nets on the **orthogonalized compact feature set**:
  - `iv_level_cs_rank` (dimensionless cross-sectional IV percentile, stationary across regimes)
  - `risk_reversal_30d_cs_rank` (normalized directional skew)
  - `variance_proxy` and `iv_to_rv_ratio` (dimensionless variance risk premium)
  - `size`, `tvol_20d`, `rm12`, `dturn_20d`, `volume_ratio_20d` (academic controls)

---

### 5. Always Filter `is_options_investable == True`
- **The Trap:** Training on the unfiltered universe includes penny stocks ($< \$5$), micro-caps ($< \$2\text{B}$), and illiquid option chains ($< 10$ contracts) where wide bid-ask bounces and stale prints distort loss gradients.
- **The Rule:** Always apply `df[df["is_options_investable"] == True]`. This retains **235,984 rows (~359 liquid stocks/day)**, boosts skew completeness to $>99.9\%$, eliminates quote noise, and guarantees all signals are tradeable in institutional size.

