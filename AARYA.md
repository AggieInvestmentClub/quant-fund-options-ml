# 5 Critical Rules for Real ML Profit & Backtesting
*(Updated with empirical findings from the Options Signals Research Paper)*

To turn the exploratory results from the research paper into profitable, validated out-of-sample models, follow these 5 non-negotiable quantitative rules:

---

### 1. Isolate Cross-Sectional Alpha from Market Beta (`forward_5d_excess_return`)
- **The Research Paper Trap:** All 8 weekly models produced negative rank IC (-0.026 to -0.001) and negative returns (-10% to -27%) because raw stock return targets ($y_{i,w} = P_{t+1}/P_t - 1$) forced models to forecast macro market direction rather than relative stock mispricing.
- **The Rule:** Train strictly on `forward_5d_excess_return` (stock return minus the daily cross-sectional market mean) and `target_tradable_5d_return` ($T+1$ Open fill). Options order flow reveals firm-specific information, not macro index direction.

---

### 2. Mandate Academic Controls to Avoid Trading Junk (Fixing Section 1)
- **The Research Paper Trap:** Section 1 explicitly noted that paper replication was impossible because company controls (*AGE, DTURN, EXRET, TSKEW, TVOL, RM12, SIZE*) were missing. Uncontrolled option signals merely proxied for distressed, high-beta, or small-cap stocks.
- **The Rule:** Always include the newly added academic controls in every feature matrix:
  - `size` (log market cap) & `volume_ratio_20d` (liquidity shock)
  - `rm12`, `rm3`, `rm1` (multi-horizon stock momentum)
  - `tvol_20d`, `tvol_60d`, `tskew_60d` (underlying stock volatility and realized skew)
  - `dturn_20d` (share turnover)
  Options alpha must be measured *after* neutralizing for market cap and volatility.

---

### 3. Use the Compact Orthogonal Feature Set (Validated in Sections 11–12)
- **The Research Paper Trap:** Regressing on 25 raw, highly collinear Greeks blew up variance and degraded performance (Full Combined: -13.91% net).
- **The Research Paper Insight & Rule:** The paper's feature ablation proved that condensing inputs into orthogonal signals reduced noise (Compact-plus LightGBM achieved +0.30% gross). Train models on the 183-column pipeline's engineered signals:
  - `iv_level_cs_rank` & `risk_reversal_30d_cs_rank` (daily cross-sectional percentiles)
  - `variance_proxy` & `iv_to_rv_ratio` (dimensionless implied vs. realized volatility)
  - Discard redundant raw call/put Greek levels that correlate at $>0.95$.

---

### 4. Eliminate the Short-Side & Turnover Bleed (Tables 15–18)
- **The Research Paper Trap:** Naive shorting produced catastrophic losses (-10.16% in Combined LightGBM, -14.71% in Stock LightGBM). Combined with 3% annual borrow and 2.5x–3.5x weekly turnover ($\$1,800$–$\$2,000$ in IBKR commissions per $\$100k$), fees consumed >4.2% of equity. On $\$10k$ accounts, $\$1$ minimum ticket commissions blew net losses to -39%.
- **The Rule:**
  - Avoid naive unhedged shorting of high-IV names (vulnerable to short squeezes). Prefer long-only top-decile or market-hedged spreads.
  - Lengthen holding periods to 5–21 trading days to reduce turnover by >70%.
  - Factor in mandatory broker costs (IBKR Pro Fixed: $\$0.005$/share, $\$1$ min, 3% borrow, 1 bp slippage) directly in the objective function.

---

### 5. Enforce `is_options_investable` & Price Split Consistency (Fixing WBD/CTVA)
- **The Research Paper Trap:** Combined Ridge and the IV-gap baseline were **completely invalidated ("Unavailable")** because missing valuation prices (*WBD, EQR, KVUE*) and split anomalies (*CTVA* dropping from $\$80$ to $\$11$ unadjusted) ruined portfolio evaluation and invalidated 500 to 900 Monte Carlo draws.
- **The Rule:** Filter `is_options_investable == True` and `is_target_valid_5d == True`. This guarantees:
  - 100% price and quote completeness (zero missing price invalidations).
  - Elimination of penny stocks ($< \$5$), micro-caps, and illiquid option chains.
  - Consistent split adjustments across Open, High, Low, and Close (`stock_adj_open`, `stock_adj_close`).
