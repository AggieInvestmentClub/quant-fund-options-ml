import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd

import flatfile_iv
import utils
from migrate_repaired_iv_rows import add_atm_greeks
from utils import american_option_price_crr, compute_iv

UPDATER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UPDATER_ROOT / "Default_Strike_Pipeline"))

import regular_features


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


def test_low_sigma_crr_does_not_average_a_valid_tree_with_zero() -> None:
    spot = 559.3900146484375
    strike = 420.0
    tenor = 35.0 / 365.25
    rate = 0.0433
    dividend_yield = 0.0050769586972067
    sigma = 0.0010000003033616
    spot_step = 0.01 * spot

    down = american_option_price_crr(
        spot - spot_step, strike, tenor, rate, sigma, "call", dividend_yield
    )
    up = american_option_price_crr(
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


def test_crr_iv_round_trip_remains_accurate() -> None:
    expected_iv = 0.25
    inputs = {
        "S": 559.3900146484375,
        "K": 420.0,
        "T": 35.0 / 365.25,
        "r": 0.0433,
        "option_type": "call",
        "q": 0.0050769586972067,
    }
    market_price = american_option_price_crr(sigma=expected_iv, **inputs)
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
    market_price = american_option_price_crr(sigma=0.25, **inputs)
    low_residual = american_option_price_crr(sigma=utils.IV_MIN_SIGMA, **inputs) - market_price
    high_residual = american_option_price_crr(sigma=utils.IV_MAX_SIGMA, **inputs) - market_price
    original_price = utils.american_option_price_crr
    calls = 0

    def counted_price(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_price(*args, **kwargs)

    monkeypatch.setattr(utils, "american_option_price_crr", counted_price)
    uncached = utils.compute_iv(market_price=market_price, **inputs)
    uncached_calls = calls
    calls = 0
    cached = utils.compute_iv(
        market_price=market_price,
        bracket_residuals=(low_residual, high_residual),
        **inputs,
    )

    assert cached == uncached
    assert uncached_calls - calls == 2


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
