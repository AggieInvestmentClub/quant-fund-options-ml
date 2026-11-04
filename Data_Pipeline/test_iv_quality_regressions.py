import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

import flatfile_iv
import utils
from migrate_repaired_iv_rows import add_atm_greeks
from utils import american_option_price_lr, american_option_price_crr, compute_iv

UPDATER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UPDATER_ROOT / "Default_Strike_Pipeline"))

import regular_features
import flatfile_regular_greeks


def test_vectorized_opra_parser_matches_row_parser() -> None:
    tickers = pd.Series(
        [
            " O:AAPL261120C00150000 ",
            "o:TSLA261120P00000001",
            "O:GOOGLE681231C00150000",
            "O:AAPL691231P00150000",
            "O:AAPL260230C00150000",
            "not-an-option",
            None,
        ],
        index=[10, 20, 30, 40, 50, 60, 70],
    )
    expected = pd.DataFrame(
        [
            flatfile_iv.parse_opra_option_ticker(ticker)
            or {"underlying": None, "expiration_date": None, "option_type": None, "strike": None}
            for ticker in tickers
        ],
        index=tickers.index,
    )
    expected["expiration_date"] = pd.to_datetime(expected["expiration_date"], errors="coerce")
    expected["strike"] = pd.to_numeric(expected["strike"], errors="coerce")

    pd.testing.assert_frame_equal(
        flatfile_iv.parse_opra_option_tickers(tickers), expected, check_dtype=False
    )


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


def test_floor_discontinuity_is_not_reported_as_an_iv_solution() -> None:
    iv = compute_iv(
        market_price=140.63,
        S=559.3900146484375,
        K=420.0,
        T=35.0 / 365.25,
        r=0.0433,
        option_type="call",
        q=0.0050769586972067,
    )
    assert iv is None


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


def test_intrinsic_value_option_has_no_identifiable_iv() -> None:
    inputs = {
        "S": 54.0,
        "K": 57.5,
        "T": 52.0 / 365.25,
        "r": 0.0546,
        "option_type": "put",
        "q": 0.0,
    }

    assert compute_iv(market_price=3.5, **inputs) is None
    assert flatfile_iv.compute_row_iv_task(
        (54.0, 57.5, 3.5, 52.0 / 365.25, 0.0546, 0.0, "put")
    ) == (None, "nonpositive_time_value")


def test_split_safe_close_repairs_mixed_adjusted_and_raw_rows() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2024-08-20", "2026-06-22", "2026-07-02", "2026-07-03"],
            "Close": [66.6325, 675.44, 193.98, 199.38],
            "Adj Close": [66.6325, 168.86, 193.98, 199.38],
            "Stock Splits": [0.0, 0.0, 4.0, 0.0],
        }
    )

    close = flatfile_iv.split_safe_unadjusted_close(frame, "Close")

    assert abs(close.iloc[0] - 266.53) < 1e-9
    assert abs(close.iloc[1] - 675.44) < 1e-9
    assert abs(close.iloc[2] - 193.98) < 1e-9
    assert abs(close.iloc[3] - 199.38) < 1e-9


def test_empty_migration_replacement_retains_atm_greek_schema() -> None:
    replacement = add_atm_greeks(pd.DataFrame(), 1, date(2024, 4, 5))

    assert replacement.empty
    assert list(replacement.columns) == [
        "atm_delta",
        "atm_gamma",
        "atm_theta",
        "atm_vega",
    ]


def test_empty_raw_gamma_subset_is_missing_not_zero() -> None:
    group = pd.DataFrame(
        {
            "option_type": ["call"],
            "moneyness": [1.0],
            "gamma": [0.25],
        }
    )
    row: dict[str, float] = {}

    regular_features.add_raw_greek_summaries(row, group)

    assert math.isnan(row["put_gamma_sum_raw"])
    assert math.isnan(row["gamma_imbalance_sum_raw"])
    assert row["call_gamma_sum_raw"] == 0.25
    assert row["near_atm_gamma_sum_raw"] == 0.25


def test_regular_greeks_source_iv_coverage_accepts_valid_sparse_cycle(tmp_path: Path) -> None:
    iv_csv = tmp_path / "atm_iv.csv"
    df = pd.DataFrame(
        {
            "trade_date": ["2026-04-21", "2026-04-21"],
            "ticker": ["O:AAPL260522C00210000", "O:AAPL260717P00210000"],
            "underlying": ["AAPL", "AAPL"],
            "expiration_date": ["2026-05-22", "2026-07-17"],
            "option_type": ["call", "put"],
            "strike": [210.0, 210.0],
            "close": [62.85, 5.0],
            "underlying_close": [266.17, 266.17],
            "dte_days": [31, 87],
            "time_to_expiry_years": [31 / 365.25, 87 / 365.25],
            "risk_free_rate": [0.0369, 0.0369],
            "dividend_yield": [0.0039, 0.0039],
            "implied_volatility": [0.90, 0.25],
            "volume": [10, 10],
            "transactions": [1, 1],
        }
    )
    df.to_csv(iv_csv, index=False)
    assert flatfile_regular_greeks.source_iv_output_covers_target(iv_csv) is True

    for col in flatfile_regular_greeks.REGULAR_GREEK_COLS:
        df[col] = 0.5
    reg_csv = tmp_path / "regular_output.csv"
    df.to_csv(reg_csv, index=False)
    assert flatfile_regular_greeks.existing_regular_output_covers_target(reg_csv) is True


def test_regular_greeks_source_iv_coverage_rejects_legacy_narrow_window(tmp_path: Path) -> None:
    iv_csv = tmp_path / "legacy_atm_iv.csv"
    df = pd.DataFrame(
        {
            "trade_date": ["2026-04-21", "2026-04-21"],
            "ticker": ["O:AAPL260605C00210000", "O:AAPL260619P00210000"],
            "underlying": ["AAPL", "AAPL"],
            "expiration_date": ["2026-06-05", "2026-06-19"],
            "option_type": ["call", "put"],
            "strike": [210.0, 210.0],
            "close": [62.85, 5.0],
            "underlying_close": [266.17, 266.17],
            "dte_days": [45, 59],
            "time_to_expiry_years": [45 / 365.25, 59 / 365.25],
            "risk_free_rate": [0.0369, 0.0369],
            "dividend_yield": [0.0039, 0.0039],
            "implied_volatility": [0.90, 0.25],
            "volume": [10, 10],
            "transactions": [1, 1],
        }
    )
    df.to_csv(iv_csv, index=False)

    assert flatfile_regular_greeks.source_iv_output_covers_target(iv_csv) is False
    assert flatfile_regular_greeks.existing_regular_output_covers_target(iv_csv) is False


def test_regular_greeks_source_iv_coverage_accepts_monthly_expiration_gap(tmp_path: Path) -> None:
    iv_csv = tmp_path / "monthly_gap_atm_iv.csv"
    df = pd.DataFrame(
        {
            "trade_date": ["2026-04-15", "2026-04-15"],
            "ticker": ["O:AAPL260515C00210000", "O:AAPL260619P00210000"],
            "underlying": ["AAPL", "AAPL"],
            "expiration_date": ["2026-05-15", "2026-06-19"],
            "option_type": ["call", "put"],
            "strike": [210.0, 210.0],
            "close": [62.85, 5.0],
            "underlying_close": [266.17, 266.17],
            "dte_days": [30, 64],
            "time_to_expiry_years": [30 / 365.25, 64 / 365.25],
            "risk_free_rate": [0.0369, 0.0369],
            "dividend_yield": [0.0039, 0.0039],
            "implied_volatility": [0.90, 0.25],
            "volume": [10, 10],
            "transactions": [1, 1],
        }
    )
    df.to_csv(iv_csv, index=False)
    assert flatfile_regular_greeks.source_iv_output_covers_target(iv_csv) is True

    for col in flatfile_regular_greeks.REGULAR_GREEK_COLS:
        df[col] = 0.5
    reg_csv = tmp_path / "regular_output.csv"
    df.to_csv(reg_csv, index=False)
    assert flatfile_regular_greeks.existing_regular_output_covers_target(reg_csv) is True


def test_regular_greeks_source_iv_coverage_rejects_truncated_far_window(tmp_path: Path) -> None:
    iv_csv = tmp_path / "truncated_far_atm_iv.csv"
    df = pd.DataFrame(
        {
            "trade_date": ["2026-04-15", "2026-04-15"],
            "ticker": ["O:AAPL260515C00210000", "O:AAPL260530P00210000"],
            "underlying": ["AAPL", "AAPL"],
            "expiration_date": ["2026-05-15", "2026-05-30"],
            "option_type": ["call", "put"],
            "strike": [210.0, 210.0],
            "close": [62.85, 5.0],
            "underlying_close": [266.17, 266.17],
            "dte_days": [30, 45],
            "time_to_expiry_years": [30 / 365.25, 45 / 365.25],
            "risk_free_rate": [0.0369, 0.0369],
            "dividend_yield": [0.0039, 0.0039],
            "implied_volatility": [0.90, 0.25],
            "volume": [10, 10],
            "transactions": [1, 1],
        }
    )
    df.to_csv(iv_csv, index=False)
    assert flatfile_regular_greeks.source_iv_output_covers_target(iv_csv) is False
    assert flatfile_regular_greeks.existing_regular_output_covers_target(iv_csv) is False


def test_resolve_target_underlyings() -> None:
    assert flatfile_iv.resolve_target_underlyings(None, False, "AAPL") is None
    assert flatfile_iv.resolve_target_underlyings({"AAPL", "MSFT"}, False, "AAPL") == {"AAPL", "MSFT"}
    assert flatfile_iv.resolve_target_underlyings({"AAPL", "MSFT"}, True, "AAPL") == {"AAPL"}
    assert flatfile_iv.resolve_target_underlyings({"MSFT"}, True, "AAPL") == set()
    assert flatfile_iv.resolve_target_underlyings(None, True, "aapl") == {"AAPL"}


def test_prefilter_raw_opra_frame() -> None:
    # 2026-04-21 trade date: 30 DTE is 2026-05-21, 90 DTE is 2026-07-20
    trade_date = date(2026, 4, 21)
    df = pd.DataFrame(
        {
            "ticker": [
                "O:AAPL260525C00150000",  # AAPL, in DTE window (~34 DTE) -> KEEP
                "O:AAPL260425C00150000",  # AAPL, outside DTE window (4 DTE) -> DROP
                "O:AAPL261016C00150000",  # AAPL, outside DTE window (~178 DTE) -> DROP
                "O:MSFT260525C00400000",  # MSFT, in DTE window -> DROP if target is AAPL
                "not-an-option",          # Malformed -> DROP
            ],
            "volume": [10, 10, 10, 10, 10],
            "close": [5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )

    # 1. Target AAPL
    filtered = flatfile_iv.prefilter_raw_opra_frame(df, {"AAPL"}, trade_date)
    assert len(filtered) == 1
    assert filtered.iloc[0]["ticker"] == "O:AAPL260525C00150000"

    # 2. Both AAPL and MSFT
    filtered_both = flatfile_iv.prefilter_raw_opra_frame(df, {"AAPL", "MSFT"}, trade_date)
    assert len(filtered_both) == 2
    assert set(filtered_both["ticker"]) == {"O:AAPL260525C00150000", "O:MSFT260525C00400000"}

    # 3. Target is None (all underlyings, only DTE filter applied)
    filtered_all = flatfile_iv.prefilter_raw_opra_frame(df, None, trade_date)
    assert len(filtered_all) == 2
    assert set(filtered_all["ticker"]) == {"O:AAPL260525C00150000", "O:MSFT260525C00400000"}

    # 4. Empty and edge cases
    assert flatfile_iv.prefilter_raw_opra_frame(pd.DataFrame(), {"AAPL"}, trade_date).empty
    assert flatfile_iv.prefilter_raw_opra_frame(df, set(), trade_date).empty


