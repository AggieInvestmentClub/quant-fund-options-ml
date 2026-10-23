"""
Option pricing utilities and implied-volatility computation.

This module now uses a Cox-Ross-Rubinstein binomial tree for American-style
equity options when computing implied volatilities and hypothetical ATM Greeks.
"""

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# American-option CRR pricing
# ---------------------------------------------------------------------------

DEFAULT_CRR_STEPS = 201
MIN_CRR_STEPS = 101
MAX_CRR_STEPS = 1001
IV_MIN_SIGMA = 0.001
IV_MAX_SIGMA = 5.0
IV_PRICE_TOLERANCE = 1e-6


def option_intrinsic_value(
    S: float,
    K: float,
    option_type: str,
) -> float:
    """Return the immediate-exercise value for a call or put."""
    if option_type.lower() == "call":
        return max(float(S) - float(K), 0.0)
    return max(float(K) - float(S), 0.0)

# AGENTS: Mission-critical pricing helper; chooses CRR tree depth for IV and ATM Greek calculations.
def _resolve_crr_steps(T: float, steps: Optional[int] = None) -> int:
    if steps is not None:
        return max(MIN_CRR_STEPS, int(steps))
    scaled = int(np.ceil(max(T, 1.0 / 365.25) * 365.25 * 4))
    return int(min(MAX_CRR_STEPS, max(MIN_CRR_STEPS, scaled)))

# AGENTS: Mission-critical math helper for Black-Scholes call pricing and Greeks.
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# AGENTS: Mission-critical math helper for Black-Scholes Greeks.
def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

# AGENTS: Mission-critical fast path for non-dividend-paying calls in American option pricing.
def _black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    if S <= 0 or K <= 0 or sigma <= 0:
        return 0.0

    option_type = option_type.lower()
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sqrt_t = float(np.sqrt(T))
    sigma_sqrt_t = sigma * sqrt_t
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)
    if option_type == "call":
        return float(discounted_spot * _norm_cdf(d1) - discounted_strike * _norm_cdf(d2))
    return float(discounted_strike * _norm_cdf(-d2) - discounted_spot * _norm_cdf(-d1))

# AGENTS: Mission-critical fast path for non-dividend-paying call ATM Greeks.
def _black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> dict:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    option_type = option_type.lower()
    sqrt_t = math.sqrt(T)
    sigma_sqrt_t = sigma * sqrt_t
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    pdf_d1 = _norm_pdf(d1)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    gamma = disc_q * pdf_d1 / (S * sigma_sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
            - r * K * disc_r * _norm_cdf(d2)
            + q * S * disc_q * _norm_cdf(d1)
        )
    else:
        delta = disc_q * (_norm_cdf(d1) - 1.0)
        theta = (
            -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
            + r * K * disc_r * _norm_cdf(-d2)
            - q * S * disc_q * _norm_cdf(-d1)
        )

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
    }

# AGENTS: Mission-critical CRR helper; validates tree parameters before pricing American options.
def _crr_parameters(
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    steps: Optional[int] = None,
):
    """
    Compute CRR tree parameters using the standard formulas:
      u = exp(sigma * sqrt(dt))
      d = exp(-sigma * sqrt(dt)) = 1 / u
      p = (exp((r - q) * dt) - d) / (u - d)

    If the requested or automatically selected step count produces an invalid
    probability, increase the step count until the tree satisfies the CRR
    no-arbitrage condition or we hit the configured cap.
    """
    n_steps = _resolve_crr_steps(T, steps)

    while True:
        dt = T / n_steps
        if dt <= 0:
            return None

        u = float(np.exp(sigma * np.sqrt(dt)))
        d = float(np.exp(-sigma * np.sqrt(dt)))
        denom = u - d
        if denom == 0:
            return None

        growth = float(np.exp((r - q) * dt))
        p = (growth - d) / denom
        if 0.0 <= p <= 1.0:
            disc = float(np.exp(-r * dt))
            return n_steps, dt, u, d, p, disc

        if n_steps >= MAX_CRR_STEPS:
            return None

        n_steps = min(MAX_CRR_STEPS, max(n_steps + 1, int(np.ceil(n_steps * 1.5))))

# AGENTS: Mission-critical CRR core; called by american_option_price_crr() for American-style pricing.
def _american_option_price_crr_single(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    steps: Optional[int] = None,
) -> float:
    """
    Price an American option with a single CRR binomial tree.

    Parameters match the standard option-pricing inputs, with `steps`
    controlling tree depth.
    """
    if S <= 0 or K <= 0 or sigma <= 0:
        return 0.0

    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    params = _crr_parameters(T=T, r=r, sigma=sigma, q=q, steps=steps)
    if params is None:
        return math.nan

    n_steps, _, u, d, p, disc = params
    option_type = option_type.lower()

    terminal_idx = np.arange(n_steps + 1)
    stock_prices = S * (u ** (n_steps - terminal_idx)) * (d ** terminal_idx)
    if option_type == "call":
        option_values = np.maximum(stock_prices - K, 0.0)
    else:
        option_values = np.maximum(K - stock_prices, 0.0)

    for step in range(n_steps - 1, -1, -1):
        stock_prices = stock_prices[:-1] / u
        continuation = disc * (p * option_values[:-1] + (1.0 - p) * option_values[1:])
        if option_type == "call":
            exercise = np.maximum(stock_prices - K, 0.0)
        else:
            exercise = np.maximum(K - stock_prices, 0.0)
        option_values = np.maximum(continuation, exercise)

    return float(option_values[0])

# AGENTS: Mission-critical public pricing function; compute_iv() and hypothetical_atm_greeks() depend on it.
def american_option_price_crr(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    steps: Optional[int] = None,
) -> float:
    """
    Price an American option with a CRR binomial tree.

    To reduce the well-known odd/even oscillation of binomial trees,
    return the average of two adjacent tree depths.
    """
    option_type = option_type.lower()

    # For non-dividend-paying calls, early exercise is not optimal.
    # Use the closed-form European price for a more accurate result.
    if option_type == "call" and q <= 1e-12:
        return _black_scholes_price(S, K, T, r, sigma, option_type, q)

    valid_params = _crr_parameters(T=T, r=r, sigma=sigma, q=q, steps=steps)
    if valid_params is None:
        return math.nan
    n_steps = valid_params[0]
    price_n = _american_option_price_crr_single(S, K, T, r, sigma, option_type, q, n_steps)
    price_np1 = _american_option_price_crr_single(S, K, T, r, sigma, option_type, q, n_steps + 1)
    if not math.isfinite(price_n) or not math.isfinite(price_np1):
        return math.nan
    return 0.5 * (price_n + price_np1)


# ---------------------------------------------------------------------------
# Hypothetical ATM Greeks (paper: Appendix B with K = S)
# ---------------------------------------------------------------------------
# AGENTS: Mission-critical public Greek function; flatfile_iv.py uses it for per-option and final feature Greeks.
def hypothetical_atm_greeks(sigma: float, S: float, T: float, r: float,
                            option_type: str = "call",
                            q: float = 0.0,
                            steps: Optional[int] = None) -> dict:
    """
    Given an option's implied volatility, reprice as a hypothetical ATM
    option (K = S) and return its Greeks.

    This implements Section 2.3.2 / Appendix B of Andreou, Han & Li (2025):
    take each option's IV, set moneyness = 1, compute Greeks at that strike
    so they are comparable across options with different strikes.

    Returns dict with keys: delta, gamma, theta, vega.
    """
    if sigma <= 0 or S <= 0 or T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    K = S
    if option_type.lower() == "call" and q <= 1e-12:
        return _black_scholes_greeks(S, K, T, r, sigma, option_type, q)

    n_steps = _resolve_crr_steps(T, steps)
    dS = max(0.01, 0.01 * S)
    d_sigma = max(1e-4, 0.01 * sigma)
    dT = min(1.0 / 365.25, max(T / 10.0, 1.0 / 365.25))

    base = american_option_price_crr(S, K, T, r, sigma, option_type, q, n_steps)
    up = american_option_price_crr(S + dS, K, T, r, sigma, option_type, q, n_steps)
    down = american_option_price_crr(max(1e-8, S - dS), K, T, r, sigma, option_type, q, n_steps)
    delta = (up - down) / (2.0 * dS)
    gamma = (up - 2.0 * base + down) / (dS ** 2)

    vega_up = american_option_price_crr(S, K, T, r, sigma + d_sigma, option_type, q, n_steps)
    vega_down = american_option_price_crr(
        S, K, T, r, max(1e-6, sigma - d_sigma), option_type, q, n_steps
    )
    vega = (vega_up - vega_down) / (2.0 * d_sigma)

    shorter_T = max(1e-6, T - dT)
    shorter = american_option_price_crr(S, K, shorter_T, r, sigma, option_type, q, n_steps)
    theta = (shorter - base) / dT

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
    }


# ---------------------------------------------------------------------------
# Implied Volatility
# ---------------------------------------------------------------------------
# AGENTS: Mission-critical public IV solver; flatfile_iv.py calls this for every eligible option row.
def compute_iv(market_price: float, S: float, K: float, T: float,
               r: float, option_type: str = "call",
               q: float = 0.0,
               steps: Optional[int] = None,
               bracket_residuals: Optional[tuple[float, float]] = None) -> Optional[float]:
    """
    Compute American-option implied volatility using CRR-tree inversion.

    Returns None when IV cannot be computed (arbitrage violation, illiquid,
    or numerical failure).

    Bounds: sigma in [0.001, 5.0]  (0.1 % – 500 % annualised volatility)
    bracket_residuals, when supplied, must be the already computed pricing
    errors at those two bounds for these same option inputs.
    """
    numeric_inputs = (market_price, S, K, T, r, q)
    if not all(math.isfinite(float(value)) for value in numeric_inputs):
        return None
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = option_intrinsic_value(S, K, option_type)
    upper = S if option_type == "call" else K
    if market_price < intrinsic - 1e-6 or market_price > upper + 1e-6:
        return None
    # At (or within pricing tolerance of) immediate-exercise value, an
    # American option can have the same price across a range of volatilities.
    # Returning the first bisection point in that flat region fabricates an IV.
    if market_price <= intrinsic + IV_PRICE_TOLERANCE:
        return None

    # AGENTS: Mission-critical nested solver target; used only inside compute_iv().
    def objective(sigma: float) -> float:
        return american_option_price_crr(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            q=q,
            steps=steps,
        ) - market_price

    # The pipeline row worker already prices both endpoints to classify a
    # missing root. Reuse those prices instead of building four extra trees.
    if bracket_residuals is None:
        try:
            lo_val = objective(IV_MIN_SIGMA)
            hi_val = objective(IV_MAX_SIGMA)
        except Exception:
            return None
    else:
        lo_val, hi_val = bracket_residuals

    if not math.isfinite(lo_val) or not math.isfinite(hi_val):
        return None
    if abs(lo_val) <= IV_PRICE_TOLERANCE:
        return IV_MIN_SIGMA
    if abs(hi_val) <= IV_PRICE_TOLERANCE:
        return IV_MAX_SIGMA
    if lo_val * hi_val > 0:
        # Both same sign — no root in interval
        return None

    lo = IV_MIN_SIGMA
    hi = IV_MAX_SIGMA
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        mid_val = objective(mid)
        if not math.isfinite(mid_val):
            return None
        if abs(mid_val) < IV_PRICE_TOLERANCE:
            return float(mid)
        if lo_val * mid_val <= 0:
            hi = mid
            hi_val = mid_val
        else:
            lo = mid
            lo_val = mid_val

    iv = 0.5 * (lo + hi)
    if not math.isfinite(iv) or not IV_MIN_SIGMA <= iv <= IV_MAX_SIGMA:
        return None
    residual = objective(iv)
    if not math.isfinite(residual) or abs(residual) > IV_PRICE_TOLERANCE:
        return None
    return float(iv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# AGENTS: Currently probably unused by main.py path; simple helper kept for ad hoc analysis or future filters.
def moneyness(S: float, K: float) -> float:
    """Strike / Spot ratio (K/S).  < 1 → put OTM territory, > 1 → call OTM."""
    return K / S

# AGENTS: Currently probably unused by main.py path; flatfile_iv.py computes DTE years inline instead.
def dte_to_years(days: int) -> float:
    """Convert calendar days to year fraction (252 trading days / year)."""
    return days / 365.25
