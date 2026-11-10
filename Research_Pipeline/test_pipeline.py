"""Behavior tests for the research handoff, chronology, portfolio and simulations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Research_Pipeline.backtest import _positions, attach_forward_trade_returns, backtest_one
from Research_Pipeline.data import build_monthly_panel, load_handoff, market_close_utc
from Research_Pipeline.models import _training_rows, prediction_metrics
from Research_Pipeline.monte_carlo import (
    monte_carlo_all, monte_carlo_one, randomized_mean_monthly_return,
    shuffle_scores_within_month_and_size,
)
from Research_Pipeline.run import run_pipeline
from Research_Pipeline.synthetic import write_synthetic_handoff


@pytest.fixture
def handoff(tmp_path):
    path = tmp_path / "input"
    write_synthetic_handoff(path, n_stocks=20, n_months=30, seed=3)
    return path


def test_month_ends_labels_and_maturity(handoff):
    tables = load_handoff(handoff)
    panel, coverage = build_monthly_panel(tables)
    assert str(panel.period.min()) == "2015-01"
    assert str(panel.period.max()) == "2017-06"
    assert coverage["month_end_option_rows"] == 600
    assert not panel.duplicated(["trade_date", "security_id"]).any()
    labeled = panel[panel.label.notna()]
    assert (labeled.target_period == labeled.period + 12).all()
    assert np.allclose(labeled.forward_log_return, np.log(labeled.target_adjusted_close / labeled.adjusted_close))
    assert (labeled.label_available_at > labeled.rebalance_close_at).all()
    assert (labeled.sp500_member.notna()).all()
    assert panel.loc[panel.period.eq(pd.Period("2016-07")), "label"].isna().all()


def test_duplicate_contract_day_key_is_rejected(handoff):
    path = handoff / "options_features.parquet"
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_handoff(handoff)


def test_ambiguous_ticker_history_is_rejected(handoff):
    path = handoff / "options_features.parquet"
    frame = pd.read_parquet(path).drop(columns="security_id")
    frame.to_parquet(path, index=False)
    mapping = pd.DataFrame([
        {"ticker": "T0000", "security_id": "A", "start_date": "2010-01-01", "end_date": "2020-01-01"},
        {"ticker": "T0000", "security_id": "B", "start_date": "2010-01-01", "end_date": "2020-01-01"},
    ])
    mapping.to_parquet(handoff / "ticker_history.parquet", index=False)
    with pytest.raises(ValueError, match="ambiguous"):
        load_handoff(handoff)


def test_ticker_change_resolves_to_same_security_id(tmp_path):
    root = tmp_path / "input"
    write_synthetic_handoff(root, n_stocks=5, n_months=60)
    path = root / "options_features.parquet"
    frame = pd.read_parquet(path)
    history = frame.groupby(["ticker", "security_id"], as_index=False).agg(
        start_date=("trade_date", "min"), end_date=("trade_date", "max"),
    )
    history.to_parquet(root / "ticker_history.parquet", index=False)
    frame.drop(columns="security_id").to_parquet(path, index=False)
    resolved = load_handoff(root)["options_features"]
    assert set(resolved.loc[resolved.ticker.isin(["T0000", "RENAMED"]), "security_id"]) == {"S0000"}


def test_late_features_and_future_controls_do_not_enter_panel(handoff):
    tables = load_handoff(handoff)
    first_day = tables["options_features"].trade_date.min()
    first_stock = "S0000"
    feature_mask = tables["options_features"].trade_date.eq(first_day) & tables["options_features"].security_id.eq(first_stock)
    next_close = market_close_utc(first_day + pd.offsets.BDay(1))
    tables["options_features"].loc[feature_mask, "available_at"] = next_close + pd.Timedelta(minutes=1)
    second_day = sorted(tables["options_features"].trade_date.unique())[1]
    control_mask = tables["stock_controls"].as_of_date.eq(second_day) & tables["stock_controls"].security_id.eq(first_stock)
    tables["stock_controls"].loc[control_mask, "available_at"] = next_close + pd.Timedelta(days=40)
    panel, coverage = build_monthly_panel(tables)
    assert not ((panel.trade_date.eq(first_day)) & panel.security_id.eq(first_stock)).any()
    second = panel[panel.trade_date.eq(second_day) & panel.security_id.eq(first_stock)].iloc[0]
    assert second.as_of_date == first_day
    assert coverage["timely_month_end_rows"] == coverage["month_end_option_rows"] - 1


def test_future_dated_control_falls_back_to_prior_release(handoff):
    tables = load_handoff(handoff)
    second_day = sorted(tables["options_features"].trade_date.unique())[1]
    first_day = sorted(tables["options_features"].trade_date.unique())[0]
    mask = tables["stock_controls"].as_of_date.eq(second_day) & tables["stock_controls"].security_id.eq("S0000")
    tables["stock_controls"].loc[mask, "as_of_date"] = second_day + pd.Timedelta(days=1)
    panel, _ = build_monthly_panel(tables)
    row = panel[panel.trade_date.eq(second_day) & panel.security_id.eq("S0000")].iloc[0]
    assert row.as_of_date == first_day


def test_missing_optional_tables_and_common_stock_filter(handoff):
    feature_path = handoff / "options_features.parquet"
    features = pd.read_parquet(feature_path)
    features.loc[features.security_id.eq("S0001"), "is_us_common_stock"] = False
    features.to_parquet(feature_path, index=False)
    (handoff / "stock_controls.parquet").unlink()
    (handoff / "sp500_membership.parquet").unlink()
    panel, coverage = build_monthly_panel(load_handoff(handoff))
    assert not panel.security_id.eq("S0001").any()
    assert coverage["rows_with_all_paper_controls"] == 0
    assert coverage["rows_with_known_sp500_membership"] == 0


def _predictions_for_backtest() -> pd.DataFrame:
    rows = []
    for month, returns in (("2020-01", [0.1, -0.1, 0.02, 0.0]), ("2020-02", [0.0, 0.05, -0.02, 0.01]), ("2020-03", [0.03, 0.01, -0.04, 0.02])):
        period = pd.Period(month)
        for index, stock_return in enumerate(returns):
            rows.append({
                "period": period, "security_id": f"S{index}", "market_cap": 100.0,
                "rebalance_date": period.to_timestamp(), "exit_date": (period + 1).to_timestamp(),
                "p_jump": [0.9, 0.1, 0.2, 0.3][index],
                "p_crash": [0.1, 0.9, 0.2, 0.3][index],
                "p_normal": 0.0, "stock_return": stock_return,
            })
    return pd.DataFrame(rows)


def test_portfolio_weights_returns_and_transaction_costs():
    predictions = _predictions_for_backtest()
    chosen = _positions(predictions[predictions.period.eq(pd.Period("2020-01"))])
    assert chosen.loc[chosen.side.eq("long"), "weight"].sum() == pytest.approx(1)
    assert chosen.loc[chosen.side.eq("short"), "weight"].sum() == pytest.approx(-1)
    trades, monthly = backtest_one(predictions)
    assert len(trades) == 6
    assert monthly.gross_return.iloc[0] == pytest.approx(0.2)
    assert monthly.transaction_cost.iloc[0] == pytest.approx(0.006)
    assert monthly.transaction_cost.iloc[1] == pytest.approx(0)
    assert monthly.transaction_cost.iloc[-1] == pytest.approx(0.006)
    assert monthly.net_return_0pct_borrow.iloc[0] == pytest.approx(0.194)
    assert monthly.net_return_3pct_borrow.iloc[0] == pytest.approx(0.1915)


def test_only_matured_labels_can_train(handoff):
    panel, _ = build_monthly_panel(load_handoff(handoff))
    cutoff = panel.loc[panel.period.eq(pd.Period("2016-06")), "rebalance_close_at"].max()
    training = _training_rows(panel, cutoff)
    assert not training.empty
    assert training.label_available_at.le(cutoff).all()
    assert training.rebalance_close_at.lt(cutoff).all()
    assert not training.period.eq(pd.Period("2015-07")).any()


def test_missing_trade_price_fails_instead_of_dropping_stock(handoff):
    tables = load_handoff(handoff)
    panel, _ = build_monthly_panel(tables)
    row = panel[panel.period.eq(pd.Period("2015-01")) & panel.security_id.eq("S0000")]
    predictions = row[["period", "security_id", "rebalance_date"]].copy()
    entry = predictions.rebalance_date.iloc[0]
    prices = tables["stock_prices"]
    prices = prices[~(prices.trade_date.eq(entry) & prices.security_id.eq("S0000"))]
    with pytest.raises(ValueError, match="missing entry/exit"):
        attach_forward_trade_returns(predictions, prices)


def test_zero_terminal_value_is_a_crash_not_a_missing_label(handoff):
    tables = load_handoff(handoff)
    prices = tables["stock_prices"]
    target_month = prices.trade_date.dt.to_period("M").eq(pd.Period("2016-01"))
    target_day = prices.loc[target_month, "trade_date"].max()
    terminal = prices.trade_date.eq(target_day) & prices.security_id.eq("S0000")
    prices.loc[terminal, ["close", "adjusted_close", "market_cap"]] = 0.0
    panel, _ = build_monthly_panel(tables)
    first = panel[panel.period.eq(pd.Period("2015-01")) & panel.security_id.eq("S0000")].iloc[0]
    assert first.label == "crash"
    assert np.isneginf(first.forward_log_return)
    assert not (panel.period.eq(pd.Period("2016-01")) & panel.security_id.eq("S0000")).any()


def test_monte_carlo_reproducible_and_preserves_monthly_scores():
    predictions = _predictions_for_backtest()
    baseline = predictions.loc[predictions.period.eq(pd.Period("2020-01")), ["p_jump", "p_crash"]].to_numpy()
    shuffled = shuffle_scores_within_month_and_size(predictions, np.random.default_rng(7))
    actual = shuffled.loc[shuffled.period.eq(pd.Period("2020-01")), ["p_jump", "p_crash"]].to_numpy()
    assert sorted(map(tuple, baseline)) == sorted(map(tuple, actual))
    _, monthly = backtest_one(predictions)
    fast_mean = randomized_mean_monthly_return(predictions, np.random.default_rng(7), randomize=False)
    assert fast_mean == pytest.approx(monthly.net_return_0pct_borrow.mean())
    first = monte_carlo_one(predictions, iterations=5, seed=7)
    second = monte_carlo_one(predictions, iterations=5, seed=7)
    assert first == second
    assert 0 < first["random_signal_one_sided_p"] <= 1


def test_monte_carlo_reports_multiple_comparison_adjustment():
    base = _predictions_for_backtest()
    base["model"] = "logistic"
    base["sp500_member"] = True
    variants = []
    for name in ("options_only", "combined"):
        copy = base.copy()
        copy["feature_set"] = name
        variants.append(copy)
    result = monte_carlo_all(pd.concat(variants, ignore_index=True), iterations=2)
    for value in result.values():
        assert value["random_signal_familywise_p_bonferroni"] == min(
            1.0, 2 * value["random_signal_one_sided_p"],
        )


def test_prediction_top_five_precision_is_selected_each_month():
    rows = []
    for month, top_label, top_probability in (
        ("2020-01", "jump", 0.9), ("2020-02", "normal", 0.8),
    ):
        for index in range(4):
            rows.append({
                "period": pd.Period(month), "security_id": f"S{index}",
                "feature_set": "options_only", "model": "logistic",
                "sp500_member": True,
                "label": top_label if index == 0 else "normal" if index < 3 else "crash",
                "p_jump": top_probability if index == 0 else 0.2 - 0.01 * index,
                "p_crash": 0.2 if index == 3 else 0.1,
            })
    metrics = prediction_metrics(pd.DataFrame(rows))
    assert metrics["options_only/logistic/full"]["jump_ppv_top5"] == pytest.approx(0.5)


def test_short_history_produces_coverage_and_blocked_status(handoff, tmp_path):
    output = tmp_path / "out"
    result = run_pipeline(handoff, output, iterations=1)
    assert result["status"] == "blocked_data_gate"
    assert (output / "coverage.json").exists()
    assert not (output / "portfolio_metrics.json").exists()
    with pytest.raises(FileExistsError, match="fresh output directory"):
        run_pipeline(handoff, output, iterations=1)


def test_full_run_requires_ten_years_of_matured_training(tmp_path):
    source = tmp_path / "input"
    write_synthetic_handoff(source, n_stocks=10, n_months=48)
    for path in source.glob("*.parquet"):
        frame = pd.read_parquet(path)
        frame["source_version"] = "fixture-v1"
        frame.to_parquet(path, index=False)
    result = run_pipeline(source, tmp_path / "out", iterations=1)
    assert result["status"] == "blocked_data_gate"
    assert "matured training months" in result["reason"]


def test_no_timely_options_produces_blocked_status(handoff, tmp_path):
    path = handoff / "options_features.parquet"
    features = pd.read_parquet(path)
    features["available_at"] = features.available_at + pd.Timedelta(days=10)
    features.to_parquet(path, index=False)
    result = run_pipeline(handoff, tmp_path / "out", pilot=True, iterations=1)
    assert result["status"] == "blocked_data_gate"
    assert result["coverage"]["timely_month_end_rows"] == 0
