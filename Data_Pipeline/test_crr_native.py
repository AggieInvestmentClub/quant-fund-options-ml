"""Numerical comparison for the optional compiled CRR kernel."""

import math

import pytest

import utils

native = pytest.importorskip("_crr_native")


@pytest.mark.parametrize(
    "inputs",
    [
        (100.0, 105.0, 60.0 / 365.25, 0.04, 0.25, "put", 0.0),
        (100.0, 95.0, 45.0 / 365.25, 0.04, 0.20, "call", 0.012),
        (559.3900146484375, 420.0, 35.0 / 365.25, 0.0433, 0.0010000003033616, "call", 0.0050769586972067),
        (50.0, 70.0, 90.0 / 365.25, 0.05, 0.65, "put", 0.03),
        (120.0, 125.0, 1.5, 0.03, 0.30, "put", 0.01),
    ],
)
def test_native_crr_matches_python_on_diverse_trees(monkeypatch, inputs) -> None:
    spot, strike, tenor, rate, sigma, option_type, dividend = inputs
    kwargs = dict(S=spot, K=strike, T=tenor, r=rate, sigma=sigma, option_type=option_type, q=dividend)
    monkeypatch.setattr(utils, "_native_price_single_crr", None)
    python_price = utils.american_option_price_crr(**kwargs)
    monkeypatch.setattr(utils, "_native_price_single_crr", native.price_single_crr)
    compiled_price = utils.american_option_price_crr(**kwargs)

    assert math.isfinite(python_price)
    assert math.isclose(compiled_price, python_price, rel_tol=1e-12, abs_tol=1e-10)
