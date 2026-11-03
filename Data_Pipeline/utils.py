"""
Option pricing utilities and implied-volatility computation.

This module uses a Leisen-Reimer (LR) binomial tree for American-style equity
options when computing implied volatilities and hypothetical ATM Greeks.  The
LR parametrisation (Peizer-Pratt Method 2 inversion) centres the lattice
around the strike price so that a single odd-depth tree converges at
O(1/N^2), eliminating the sawtooth oscillation that required CRR to average
two adjacent tree depths.
"""

import math
import os
from typing import Optional

import numpy as np

_LR_BACKEND = os.environ.get("OPTIONS_LR_BACKEND", os.environ.get("OPTIONS_CRR_BACKEND", "auto")).strip().lower()
if _LR_BACKEND not in {"auto", "python", "native"}:
    raise ValueError("OPTIONS_LR_BACKEND (or OPTIONS_CRR_BACKEND) must be auto, python, or native")
try:
    from _lr_native import price_single_lr as _native_price_single_lr
except ImportError as exc:
    if _LR_BACKEND == "native":
        raise RuntimeError(
            "Native LR extension is unavailable; run Data_Pipeline/native/build.py"
        ) from exc
    _native_price_single_lr = None
if _LR_BACKEND == "python":
    _native_price_single_lr = None


# ---------------------------------------------------------------------------
# American-option Leisen-Reimer pricing
# ---------------------------------------------------------------------------

DEFAULT_LR_STEPS = 65
MIN_LR_STEPS = 31
MAX_LR_STEPS = 1001
IV_MIN_SIGMA = 0.001
IV_MAX_SIGMA = 5.0
IV_PRICE_TOLERANCE = 1e-6
IV_SIGMA_TOLERANCE = 1e-4
IV_MAX_ITERATIONS = 6


def option_intrinsic_value(
    S: float,
    K: float,
    option_type: str,
) -> float:
    """Return the immediate-exercise value for a call or put."""
    if option_type.lower() == "call":
        return max(float(S) - float(K), 0.0)
    return max(float(K) - float(S), 0.0)

# AGENTS: Mission-critical pricing helper; chooses LR tree depth for IV and ATM Greek calculations.
# Step count is always forced to an odd integer (LR requirement for strike-centring).
def _resolve_lr_steps(T: float, steps: Optional[int] = None) -> int:
    if steps is not None:
        n = max(3, int(steps))
    else:
        n = DEFAULT_LR_STEPS
    # Leisen-Reimer requires an odd step count for correct strike centring.
    if n % 2 == 0:
        n += 1
    return n

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

# AGENTS: Mission-critical math helper; Peizer-Pratt Method 2 inversion used by _lr_parameters().
# Returns (h, 1-h) as a pair to avoid catastrophic cancellation when |z| is large
# (e.g. deep ITM options or near-zero sigma), where 1 - h would lose precision.
def _peizer_pratt(z: float, n: int):
    if z == 0.0:
        return 0.5, 0.5
    # PP2 term: (z / (n + 1/3 + 0.1/(n+1)))^2 * (n + 1/6)
    w = (z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2 * (n + 1.0 / 6.0)
    if w > 700.0:
        # exp(-w) underflows to 0; return clipped extremes
        return (1.0 - 1e-15, 1e-15) if z > 0 else (1e-15, 1.0 - 1e-15)
    exp_neg_w = math.exp(-w)
    sqrt_term = math.sqrt(max(0.0, 1.0 - exp_neg_w))
    if z > 0:
        # h is close to 1; compute 1-h via the numerically stable form
        h = 0.5 + 0.5 * sqrt_term
        one_minus_h = 0.5 * exp_neg_w / (1.0 + sqrt_term)
    else:
        # h is close to 0; compute h via the numerically stable form
        one_minus_h = 0.5 + 0.5 * sqrt_term
        h = 0.5 * exp_neg_w / (1.0 + sqrt_term)
    return h, one_minus_h

# AGENTS: Mission-critical LR helper; validates tree parameters before pricing American options.
def _lr_parameters(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    steps: Optional[int] = None,
):
    """
    Compute Leisen-Reimer tree parameters using the Peizer-Pratt Method 2 inversion.

    The step count is forced to an odd integer (LR requirement).  Unlike CRR,
    LR does not need a retry loop because the risk-neutral probability p is
    derived directly from the Black-Scholes d2 value, which always lies in
    (0, 1) for finite sigma.

    Returns (n_steps, dt, u, d, p, disc) or None on degenerate input.
    """
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return None

    n_steps = _resolve_lr_steps(T, steps)
    dt = T / n_steps

    sqrt_t = math.sqrt(T)
    sigma_sqrt_t = sigma * sqrt_t
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    p, one_minus_p = _peizer_pratt(d2, n_steps)
    p_prime, one_minus_p_prime = _peizer_pratt(d1, n_steps)

    if p <= 0.0 or one_minus_p <= 0.0:
        return None

    growth = math.exp((r - q) * dt)
    u = growth * (p_prime / p)
    d = growth * (one_minus_p_prime / one_minus_p)

    if not (math.isfinite(u) and math.isfinite(d) and u > 0 and d > 0):
        return None

    disc = math.exp(-r * dt)
    return n_steps, dt, u, d, p, disc

# AGENTS: Mission-critical LR core; called by american_option_price_lr() for American-style pricing.
def _american_option_price_lr_single(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    steps: Optional[int] = None,
    scratch: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> float:
    """
    Price an American option with a single Leisen-Reimer binomial tree.

    Parameters match the standard option-pricing inputs, with `steps`
    controlling tree depth (forced to an odd integer internally).
    """
    if S <= 0 or K <= 0 or sigma <= 0:
        return 0.0

    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    params = _lr_parameters(S=S, K=K, T=T, r=r, sigma=sigma, q=q, steps=steps)
    if params is None:
        return math.nan

    n_steps, _, u, d, p, disc = params
    option_type = option_type.lower()

    # Dispatch to the C++ kernel for normal operating ranges; fall back to the
    # NumPy path for unusual inputs where overflow / NaN propagation is safer
    # to handle through NumPy's array operations.
    if (_native_price_single_lr is not None and n_steps <= 1002 and T <= 5.0
            and sigma <= 6.0 and abs(r) <= 1.0 and abs(q) <= 1.0
            and S <= 1e100 and K <= 1e100):
        return _native_price_single_lr(S, K, u, d, p, disc, n_steps, int(option_type == "call"))

    if scratch is None or len(scratch[0]) < n_steps + 1:
        stock_prices = np.empty(n_steps + 1, dtype=np.float64)
        option_values = np.empty(n_steps + 1, dtype=np.float64)
        cont_buffer = np.empty(n_steps + 1, dtype=np.float64)
    else:
        stock_prices, option_values, cont_buffer = scratch

    terminal_idx = np.arange(n_steps + 1, dtype=np.float64)
    stock_prices[:n_steps + 1] = S * (u ** (n_steps - terminal_idx)) * (d ** terminal_idx)
    is_call = (option_type == "call")
    if is_call:
        np.maximum(stock_prices[:n_steps + 1] - K, 0.0, out=option_values[:n_steps + 1])
    else:
        np.maximum(K - stock_prices[:n_steps + 1], 0.0, out=option_values[:n_steps + 1])

    down_p = 1.0 - p
    for step in range(n_steps - 1, -1, -1):
        stock_prices[:step + 1] /= u
        np.multiply(option_values[:step + 1], p, out=cont_buffer[:step + 1])
        cont_buffer[:step + 1] += down_p * option_values[1:step + 2]
        cont_buffer[:step + 1] *= disc
        if is_call:
            np.maximum(stock_prices[:step + 1] - K, 0.0, out=option_values[:step + 1])
        else:
            np.maximum(K - stock_prices[:step + 1], 0.0, out=option_values[:step + 1])
        np.maximum(cont_buffer[:step + 1], option_values[:step + 1], out=option_values[:step + 1])

    return float(option_values[0])

# AGENTS: Mission-critical public pricing function; compute_iv() and hypothetical_atm_greeks() depend on it.
def american_option_price_lr(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    steps: Optional[int] = None,
    scratch: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> float:
    """
    Price an American option with a Leisen-Reimer binomial tree.

    The Leisen-Reimer parametrisation (Peizer-Pratt Method 2) centres the
    terminal lattice nodes around the strike price when the step count is odd,
    eliminating the CRR odd/even sawtooth oscillation.  A single tree
    therefore suffices — no adjacent-depth averaging is needed.
    """
    option_type = option_type.lower()

    # For non-dividend-paying calls, early exercise is not optimal.
    # Use the closed-form European price for a more accurate result.
    if option_type == "call" and q <= 1e-12:
        return _black_scholes_price(S, K, T, r, sigma, option_type, q)

    params = _lr_parameters(S=S, K=K, T=T, r=r, sigma=sigma, q=q, steps=steps)
    if params is None:
        return math.nan
    n_steps = params[0]
    price = _american_option_price_lr_single(S, K, T, r, sigma, option_type, q, n_steps, scratch=scratch)
    if not math.isfinite(price):
        return math.nan
    return price


# Backward-compatible alias so existing callers (flatfile_iv, flatfile_regular_greeks,
# tests, benchmarks) do not require immediate renaming.
american_option_price_crr = american_option_price_lr


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

    n_steps = _resolve_lr_steps(T, steps)
    dS = max(0.01, 0.01 * S)
    d_sigma = max(1e-4, 0.01 * sigma)
    dT = min(1.0 / 365.25, max(T / 10.0, 1.0 / 365.25))

    base = american_option_price_lr(S, K, T, r, sigma, option_type, q, n_steps)
    up = american_option_price_lr(S + dS, K, T, r, sigma, option_type, q, n_steps)
    down = american_option_price_lr(max(1e-8, S - dS), K, T, r, sigma, option_type, q, n_steps)
    delta = (up - down) / (2.0 * dS)
    gamma = (up - 2.0 * base + down) / (dS ** 2)

    vega_up = american_option_price_lr(S, K, T, r, sigma + d_sigma, option_type, q, n_steps)
    vega_down = american_option_price_lr(
        S, K, T, r, max(1e-6, sigma - d_sigma), option_type, q, n_steps
    )
    vega = (vega_up - vega_down) / (2.0 * d_sigma)

    shorter_T = max(1e-6, T - dT)
    shorter = american_option_price_lr(S, K, shorter_T, r, sigma, option_type, q, n_steps)
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

def _bsm_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """
    Closed-form European Black-Scholes Vega: V_BSM = S * exp(-q * T) * sqrt(T) * phi(d1).
    Used as the analytical denominator for Quasi-Newton American IV solves.
    """
    if S <= 0.0 or K <= 0.0 or sigma <= 0.0 or T <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(T)
    sigma_sqrt_t = sigma * sqrt_t
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_t
    pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return float(S * math.exp(-q * T) * sqrt_t * pdf_d1)


def _european_iv_seed(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """
    Compute closed-form European inversion seed (Corrado-Miller 1996 with Halley fallback)
    and adjust upward for in-the-money American puts.
    """
    is_call = (option_type.lower() == "call")
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    Sd = S * disc_q
    Kd = K * disc_r
    M = Sd - Kd
    A = market_price - 0.5 * M if is_call else market_price + 0.5 * M
    D = A * A - (M * M) / math.pi
    sqrt_t = math.sqrt(max(T, 1e-8))

    if D >= 0.0 and (Sd + Kd) > 0.0:
        sigma_seed = (math.sqrt(2.0 * math.pi) / ((Sd + Kd) * sqrt_t)) * (A + math.sqrt(D))
    else:
        # Microsecond European inversion fallback (Halley's method on Black-Scholes)
        sigma = 0.3
        for _ in range(4):
            sigma_sqrt_t = max(1e-8, sigma * sqrt_t)
            d1 = (math.log(max(1e-12, Sd / Kd)) + 0.5 * sigma * sigma * T) / sigma_sqrt_t
            d2 = d1 - sigma_sqrt_t
            pdf = math.exp(-0.5 * min(700.0, d1 * d1)) / math.sqrt(2.0 * math.pi)
            vega = Sd * sqrt_t * pdf
            c = Sd * _norm_cdf(d1) - Kd * _norm_cdf(d2) if is_call else Kd * _norm_cdf(-d2) - Sd * _norm_cdf(-d1)
            diff = c - market_price
            if abs(diff) < 1e-8 or vega < 1e-15:
                break
            step = diff / vega
            volga = d1 * d2 / max(1e-6, sigma)
            denom = 1.0 - 0.5 * step * volga
            denom = denom if abs(denom) > 1e-3 else 1.0
            sigma = max(IV_MIN_SIGMA, min(IV_MAX_SIGMA, sigma - step / denom))
        sigma_seed = sigma

    # For ITM American puts, adjust the seed upward:
    # sigma_0 ~= sigma_seed * (1 + max(0, (K - S)/K))
    if not is_call and K > S:
        sigma_seed *= (1.0 + (K - S) / K)

    return float(min(IV_MAX_SIGMA, max(IV_MIN_SIGMA, sigma_seed)))


# AGENTS: Mission-critical public IV solver; flatfile_iv.py calls this for every eligible option row.
def compute_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    steps: Optional[int] = None,
    bracket_residuals: Optional[tuple[float, float]] = None,
    initial_guess: Optional[float] = None,
    return_nan: bool = False,
) -> Optional[float]:
    """
    Compute American-option implied volatility using an optimized Quasi-Newton pipeline
    with closed-form European Vega and Leisen-Reimer binomial tree pricing.

    Requirements:
    1. Arbitrage & Boundary Fast-Fail:
       - If P <= intrinsic (max(0, S - K) for calls, max(0, K - S) for puts), return 0.0 or NaN/None.
       - If P >= S (call) or P >= K (put), return NaN/None.
    2. Seeding Logic:
       - Accept optional initial_guess for historical warm-starting (sigma_{t-1}).
       - Otherwise use closed-form European inversion (Corrado-Miller with Halley fallback).
       - For ITM American puts, adjust upward: sigma_0 ~= sigma_seed * (1 + max(0, (K-S)/K)).
    3. Lattice Engine (Leisen-Reimer):
       - Enforce odd step count (default: N = 65).
       - Flat 1D scratch buffer backward induction without dynamic reallocations.
    4. Root Finder (Quasi-Newton with European Vega):
       - sigma_{k+1} = sigma_k - (V_LR(sigma_k) - P_market) / V_BSM(sigma_k)
       - Closed-form analytical BSM Vega denominator (zero tree bumping).
       - Safeguards: Clamp to [0.001, 5.0], bounded bisection fallback on divergence or failure to improve.
    5. Convergence Criteria & Exit:
       - Terminate when |sigma_{k+1} - sigma_k| < 1e-4 or |V_LR(sigma_k) - P_market| < max(0.005, 0.001 * P_market).
       - Cap maximum iterations at 6.
    """
    numeric_inputs = (market_price, S, K, T, r, q)
    if not all(math.isfinite(float(value)) for value in numeric_inputs):
        return math.nan if return_nan else None
    if T <= 0.0 or market_price <= 0.0 or S <= 0.0 or K <= 0.0:
        return math.nan if return_nan else None

    option_type = option_type.lower()
    intrinsic = option_intrinsic_value(S, K, option_type)
    upper = S if option_type == "call" else K

    # 1. Arbitrage & Boundary Fast-Fail
    if market_price <= intrinsic + 1e-12:
        if abs(market_price - intrinsic) <= 1e-12:
            return 0.0 if return_nan else None
        return math.nan if return_nan else None
    if market_price >= upper - 1e-12:
        return math.nan if return_nan else None

    # Check optional legacy bracket residuals if provided
    if bracket_residuals is not None:
        lo_val, hi_val = bracket_residuals
        if math.isfinite(lo_val) and math.isfinite(hi_val):
            if abs(lo_val) <= IV_PRICE_TOLERANCE:
                return IV_MIN_SIGMA
            if abs(hi_val) <= IV_PRICE_TOLERANCE:
                return IV_MAX_SIGMA
            if lo_val * hi_val > 0.0:
                return math.nan if return_nan else None

    # 3. Lattice Engine: enforce odd step count (default: N = 65)
    n_steps = _resolve_lr_steps(T, steps)
    scratch = (
        np.empty(n_steps + 1, dtype=np.float64),
        np.empty(n_steps + 1, dtype=np.float64),
        np.empty(n_steps + 1, dtype=np.float64),
    )

    # 2. Seeding Logic
    a = IV_MIN_SIGMA
    b = IV_MAX_SIGMA
    if initial_guess is not None and math.isfinite(initial_guess):
        sigma = float(min(b, max(a, float(initial_guess))))
    else:
        sigma = _european_iv_seed(market_price, S, K, T, r, option_type, q)

    prev_res: Optional[float] = None
    tol_price_rel = max(0.005, 0.001 * market_price)

    # 4 & 5. Quasi-Newton Loop with safeguards and max 6 iterations
    for _ in range(IV_MAX_ITERATIONS):
        v = american_option_price_lr(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            q=q,
            steps=n_steps,
            scratch=scratch,
        )
        if not math.isfinite(v):
            return math.nan if return_nan else None

        res = v - market_price
        abs_res = abs(res)

        # Update dynamic bracket
        if res > 0.0:
            b = min(b, sigma)
        else:
            a = max(a, sigma)

        # Convergence on price residual
        if abs_res < tol_price_rel:
            return float(sigma)

        # Closed-form European Vega denominator
        vega = _bsm_vega(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

        # Quasi-Newton proposal and safeguard checks
        take_bisection = False
        if vega < 1e-12:
            take_bisection = True
        else:
            newton_step = res / vega
            sigma_newton = sigma - newton_step
            if not math.isfinite(sigma_newton) or sigma_newton <= a or sigma_newton >= b:
                take_bisection = True
            elif prev_res is not None and abs_res >= prev_res:
                take_bisection = True

        if take_bisection:
            sigma_next = 0.5 * (a + b)
        else:
            sigma_next = sigma_newton

        # Domain clamping
        sigma_next = min(IV_MAX_SIGMA, max(IV_MIN_SIGMA, sigma_next))

        # Convergence on parameter step change
        if abs(sigma_next - sigma) < IV_SIGMA_TOLERANCE:
            return float(sigma_next)

        prev_res = abs_res
        sigma = sigma_next

    return math.nan if return_nan else None


def compute_iv_vectorized(
    market_prices: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    option_types: np.ndarray | list[str],
    q: Optional[np.ndarray] = None,
    steps: Optional[int] = 65,
    initial_guess: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Vectorized implied volatility solver across contract slices.
    """
    market_prices = np.asarray(market_prices, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if q is None:
        q = np.zeros_like(market_prices)
    else:
        q = np.asarray(q, dtype=np.float64)

    is_call = np.asarray([str(opt).lower() == "call" for opt in option_types], dtype=bool)
    intrinsic = np.where(is_call, np.maximum(0.0, S - K), np.maximum(0.0, K - S))
    upper = np.where(is_call, S, K)

    results = np.full(market_prices.shape, np.nan, dtype=np.float64)

    zero_mask = (market_prices <= intrinsic + 1e-12) & (market_prices >= intrinsic - 1e-12)
    results[zero_mask] = 0.0

    valid_mask = (
        (market_prices > intrinsic + 1e-12)
        & (market_prices < upper - 1e-12)
        & (T > 0.0)
        & (S > 0.0)
        & (K > 0.0)
        & np.isfinite(market_prices)
        & np.isfinite(S)
        & np.isfinite(K)
        & np.isfinite(T)
        & np.isfinite(r)
        & np.isfinite(q)
    )

    active_indices = np.where(valid_mask)[0]
    if len(active_indices) == 0:
        return results

    n_steps = _resolve_lr_steps(1.0, steps)
    scratch = (
        np.empty(n_steps + 1, dtype=np.float64),
        np.empty(n_steps + 1, dtype=np.float64),
        np.empty(n_steps + 1, dtype=np.float64),
    )

    sigmas = np.empty(len(market_prices), dtype=np.float64)
    for idx in active_indices:
        if initial_guess is not None and math.isfinite(initial_guess[idx]):
            sigmas[idx] = min(IV_MAX_SIGMA, max(IV_MIN_SIGMA, float(initial_guess[idx])))
        else:
            opt_str = "call" if is_call[idx] else "put"
            sigmas[idx] = _european_iv_seed(
                market_prices[idx], S[idx], K[idx], T[idx], r[idx], opt_str, q[idx]
            )

    a = np.full(len(market_prices), IV_MIN_SIGMA, dtype=np.float64)
    b = np.full(len(market_prices), IV_MAX_SIGMA, dtype=np.float64)
    prev_res = np.full(len(market_prices), np.nan, dtype=np.float64)

    active = active_indices
    for _ in range(IV_MAX_ITERATIONS):
        if len(active) == 0:
            break
        converged_this_round = []
        for idx in active:
            opt_str = "call" if is_call[idx] else "put"
            v = american_option_price_lr(
                S=S[idx],
                K=K[idx],
                T=T[idx],
                r=r[idx],
                sigma=sigmas[idx],
                option_type=opt_str,
                q=q[idx],
                steps=n_steps,
                scratch=scratch,
            )
            if not math.isfinite(v):
                converged_this_round.append(idx)
                continue

            res = v - market_prices[idx]
            abs_res = abs(res)
            if res > 0.0:
                b[idx] = min(b[idx], sigmas[idx])
            else:
                a[idx] = max(a[idx], sigmas[idx])

            tol_price = max(0.005, 0.001 * market_prices[idx])
            if abs_res < tol_price:
                results[idx] = sigmas[idx]
                converged_this_round.append(idx)
                continue

            vega = _bsm_vega(S[idx], K[idx], T[idx], r[idx], sigmas[idx], q[idx])
            take_bisection = False
            if vega < 1e-12:
                take_bisection = True
            else:
                step_val = res / vega
                sigma_newton = sigmas[idx] - step_val
                if not math.isfinite(sigma_newton) or sigma_newton <= a[idx] or sigma_newton >= b[idx]:
                    take_bisection = True
                elif math.isfinite(prev_res[idx]) and abs_res >= prev_res[idx]:
                    take_bisection = True

            if take_bisection:
                sigma_next = 0.5 * (a[idx] + b[idx])
            else:
                sigma_next = sigma_newton

            sigma_next = min(IV_MAX_SIGMA, max(IV_MIN_SIGMA, sigma_next))

            if abs(sigma_next - sigmas[idx]) < IV_SIGMA_TOLERANCE:
                results[idx] = sigma_next
                converged_this_round.append(idx)
                continue

            prev_res[idx] = abs_res
            sigmas[idx] = sigma_next

        if converged_this_round:
            active = np.setdiff1d(active, converged_this_round)

    return results


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
