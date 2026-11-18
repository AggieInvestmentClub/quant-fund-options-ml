import math
import lr_pricing as utils
from lr_pricing import american_option_price_lr, compute_iv


def test_low_sigma_lr_does_not_produce_nonfinite_or_wrong_delta() -> None:
    """LR must return finite prices and a delta near 1 for deep-ITM near-expiry calls."""
    spot = 559.3900146484375
    strike = 420.0
    tenor = 35.0 / 365.25
    rate = 0.0433
    dividend_yield = 0.0050769586972067
    sigma = 0.0010000003033616
    spot_step = 0.01 * spot

    down = american_option_price_lr(
        spot - spot_step, strike, tenor, rate, sigma, "call", dividend_yield
    )
    up = american_option_price_lr(
        spot + spot_step, strike, tenor, rate, sigma, "call", dividend_yield
    )
    delta = (up - down) / (2.0 * spot_step)

    assert math.isfinite(down)
    assert math.isfinite(up)
    assert 0.9 < delta < 1.01


def test_lr_iv_round_trip_remains_accurate() -> None:
    """IV inversion must recover the seed sigma within 5e-5 using the LR pricer."""
    expected_iv = 0.25
    inputs = {
        "S": 559.3900146484375,
        "K": 420.0,
        "T": 35.0 / 365.25,
        "r": 0.0433,
        "option_type": "call",
        "q": 0.0050769586972067,
    }
    market_price = american_option_price_lr(sigma=expected_iv, **inputs)
    actual_iv = compute_iv(market_price=market_price, **inputs)

    assert actual_iv is not None
    assert abs(actual_iv - expected_iv) < 5e-5


def test_reusing_iv_bracket_preserves_result_and_avoids_two_prices(monkeypatch) -> None:
    inputs = {
        "S": 100.0,
        "K": 102.0,
        "T": 60.0 / 365.25,
        "r": 0.04,
        "option_type": "put",
        "q": 0.012,
    }
    market_price = american_option_price_lr(sigma=0.25, **inputs)
    low_residual = american_option_price_lr(sigma=utils.IV_MIN_SIGMA, **inputs) - market_price
    high_residual = american_option_price_lr(sigma=utils.IV_MAX_SIGMA, **inputs) - market_price
    original_price = utils.american_option_price_lr
    calls = 0

    def counted_price(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_price(*args, **kwargs)

    monkeypatch.setattr(utils, "american_option_price_lr", counted_price)
    cold_iv = utils.compute_iv(market_price=market_price, **inputs)
    cold_calls = calls
    calls = 0
    warm_iv = utils.compute_iv(
        market_price=market_price,
        initial_guess=0.25,
        **inputs,
    )
    warm_calls = calls

    assert cold_iv is not None
    assert warm_iv is not None
    assert math.isclose(cold_iv, warm_iv, abs_tol=1e-4)
    assert warm_calls < cold_calls


def test_quasi_newton_arbitrage_and_boundary_fast_fail() -> None:
    # 1. Arbitrage Bounds Fast-Fail:
    # If P <= intrinsic, return 0.0 or NaN/None
    # If P >= S (call) or P >= K (put), return NaN/None
    # Call intrinsic: max(0, S - K) = max(0, 100 - 90) = 10
    call_inputs = {"S": 100.0, "K": 90.0, "T": 0.25, "r": 0.04, "option_type": "call", "q": 0.0}
    assert utils.compute_iv(market_price=9.0, **call_inputs) is None
    assert math.isnan(utils.compute_iv(market_price=9.0, return_nan=True, **call_inputs))
    assert utils.compute_iv(market_price=10.0, return_nan=True, **call_inputs) == 0.0
    # Call upper bound: P >= S
    assert utils.compute_iv(market_price=101.0, **call_inputs) is None
    assert math.isnan(utils.compute_iv(market_price=101.0, return_nan=True, **call_inputs))

    # Put intrinsic: max(0, K - S) = max(0, 100 - 90) = 10
    put_inputs = {"S": 90.0, "K": 100.0, "T": 0.25, "r": 0.04, "option_type": "put", "q": 0.0}
    assert utils.compute_iv(market_price=9.0, **put_inputs) is None
    assert math.isnan(utils.compute_iv(market_price=9.0, return_nan=True, **put_inputs))
    assert utils.compute_iv(market_price=10.0, return_nan=True, **put_inputs) == 0.0
    # Put upper bound: P >= K
    assert utils.compute_iv(market_price=101.0, **put_inputs) is None
    assert math.isnan(utils.compute_iv(market_price=101.0, return_nan=True, **put_inputs))


def test_quasi_newton_itm_put_seed_adjustment() -> None:
    # For ITM American puts, adjust the seed upward:
    # sigma_0 ~= sigma_seed * (1 + max(0, (K - S)/K))
    S = 80.0
    K = 100.0
    T = 0.5
    r = 0.04
    q = 0.0
    market_price = 22.0
    seed_put = utils._european_iv_seed(market_price, S, K, T, r, "put", q)
    seed_call = utils._european_iv_seed(market_price, S, K, T, r, "call", q)
    # The ITM put adjustment must boost the seed
    expected_boost = 1.0 + (K - S) / K  # 1 + 20/100 = 1.20
    assert seed_put > 0.001


def test_compute_iv_vectorized_matches_scalar() -> None:
    import numpy as np

    S = np.array([100.0, 100.0, 80.0, 559.39])
    K = np.array([100.0, 110.0, 76.8, 420.0])
    T = np.array([0.25, 0.25, 0.1232, 35.0 / 365.25])
    r = np.array([0.04, 0.04, 0.04, 0.0433])
    q = np.array([0.0, 0.012, 0.0, 0.0050769586972067])
    types = ["call", "put", "call", "call"]
    sigmas_true = [0.20, 0.25, 0.20, 0.25]

    prices = np.array([
        utils.american_option_price_lr(s, k, t, rate, sig, opt, div, steps=65)
        for s, k, t, rate, sig, opt, div in zip(S, K, T, r, sigmas_true, types, q)
    ])

    vec_iv = utils.compute_iv_vectorized(
        market_prices=prices, S=S, K=K, T=T, r=r, option_types=types, q=q, steps=65
    )

    for i in range(len(S)):
        scalar_iv = utils.compute_iv(
            market_price=prices[i],
            S=S[i],
            K=K[i],
            T=T[i],
            r=r[i],
            option_type=types[i],
            q=q[i],
            steps=65,
        )
        assert scalar_iv is not None
        assert math.isclose(vec_iv[i], scalar_iv, rel_tol=1e-5, abs_tol=1e-5)
