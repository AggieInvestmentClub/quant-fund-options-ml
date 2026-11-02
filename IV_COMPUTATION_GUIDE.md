# How to Compute Implied Volatility (IV)

This guide shows how to compute American option implied volatility using this repository's solver.

---

## 1. Direct Python Usage

Import `compute_iv` from `Data_Pipeline.utils`.

### Function Signature

```python
from Data_Pipeline.utils import compute_iv

iv = compute_iv(
    market_price: float,     # Option market price / close ($)
    S: float,                # Underlying stock price ($)
    K: float,                # Strike price ($)
    T: float,                # Time to expiration in years (calendar DTE / 365.25)
    r: float,                # Annualized risk-free rate as decimal (e.g. 0.045 for 4.5%)
    option_type: str,        # "call" or "put" (case-insensitive)
    q: float = 0.0,          # Continuous dividend yield as decimal (e.g. 0.015 for 1.5%, default 0.0)
)
```

**Returns:**
* `float`: Annualized implied volatility as a decimal (e.g., `0.25` = 25% IV).
* `None`: The solver cannot find a valid solution.

---

### Single Option Example

```python
from Data_Pipeline.utils import compute_iv

# Example: 45-day American call option
iv = compute_iv(
    market_price=12.50,
    S=150.00,
    K=145.00,
    T=45.0 / 365.25,
    r=0.045,
    option_type="call",
    q=0.005,
)

if iv is not None:
    print(f"Implied Volatility: {iv:.4f} ({iv * 100:.2f}%)")
else:
    print("Could not compute IV")
```

---

### Pandas DataFrame Example

```python
import pandas as pd
from Data_Pipeline.utils import compute_iv

# Sample option data
df = pd.DataFrame([
    {"price": 12.50, "spot": 150.0, "strike": 145.0, "dte": 45, "rate": 0.045, "dividend_yield": 0.005, "type": "call"},
    {"price": 4.20,  "spot": 150.0, "strike": 145.0, "dte": 45, "rate": 0.045, "dividend_yield": 0.005, "type": "put"},
])

# Compute annualized time to expiration (T)
df["T"] = df["dte"] / 365.25

# Compute IV row-by-row
df["implied_volatility"] = [
    compute_iv(p, s, k, t, r, o_type, q)
    for p, s, k, t, r, o_type, q in zip(
        df["price"], df["spot"], df["strike"], df["T"],
        df["rate"], df["type"], df["dividend_yield"]
    )
]

print(df[["type", "strike", "price", "implied_volatility"]])
```

---

## 2. Input Requirements

| Parameter | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `market_price` | `float` | Yes | Option market price / premium. Must be $> 0$. | `12.50` |
| `S` | `float` | Yes | Underlying asset price (spot or close). Must be $> 0$. | `150.00` |
| `K` | `float` | Yes | Option strike price. Must be $> 0$. | `145.00` |
| `T` | `float` | Yes | Time to expiration in years ($\text{DTE} / 365.25$). Must be $> 0$. | `0.1232` |
| `r` | `float` | Yes | Annualized risk-free rate as a decimal. | `0.045` (for 4.5%) |
| `option_type` | `str` | Yes | `"call"` or `"put"` (case-insensitive). | `"call"` |
| `q` | `float` | No | Annual continuous dividend yield as a decimal (default: `0.0`). | `0.012` (for 1.2%) |

---

## 3. When `compute_iv` Returns None

If `compute_iv()` returns `None`, the row failed one of the following checks:

1. **Missing or Non-Finite Inputs**: Any input is `NaN`, `None`, or $\le 0$.
2. **Arbitrage Violations**: The price is below intrinsic value or above the maximum allowable price ($S$ for calls, $K$ for puts).
3. **No Time Value**: The market price is at or below immediate exercise value ($\text{marketprice} \le \text{intrinsic} + 10^{-6}$). For American options, options at intrinsic value have no identifiable volatility.
4. **Out of Range**: The required volatility is outside the solver's range of $[0.1\%, 500\%]$ ($\sigma \in [0.001, 5.0]$).

### Debugging with Error Reasons

If you want the exact failure reason instead of `None`, use `compute_row_iv_task` from `Data_Pipeline.flatfile_iv`:

```python
from Data_Pipeline.flatfile_iv import compute_row_iv_task

# task format: (underlying_close, strike, market_price, T_years, rate, dividend_yield, option_type)
task = (150.0, 145.0, 12.50, 45.0 / 365.25, 0.045, 0.005, "call")
iv, status = compute_row_iv_task(task)

print(f"IV: {iv}, Reason: {status}")
# status will be "ok", or one of:
# "missing_input", "invalid_expiry", "missing_risk_free_rate",
# "arbitrage_bounds", "nonpositive_time_value", "no_root_in_range", "solver_error"
```

---

## 4. Running via Command Line (OPRA Flatfiles)

If processing raw OPRA `.csv.gz` day aggregate files directly:

```bash
# Set required data paths
export OPTIONS_CLEAN_STOCK_ROOT=/srv/data/stocks
export OPTIONS_RISK_FREE_RATE_CSV=/srv/data/risk_free_rate/DGS3MO_risk_free_rate.csv

# Run the IV calculator
python Data_Pipeline/flatfile_iv.py \
  --input-root /path/to/opra/day_aggs_v1 \
  --output-root /path/to/output/contracts \
  --start-date 2024-04-01 \
  --end-date 2024-04-05 \
  --underlyings AAPL MSFT NVDA \
  --workers 4
```
*(Note: The CLI script filters for options with 30 to 90 days to expiration by default).*
