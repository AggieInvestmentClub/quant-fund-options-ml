"""Checks for the paper protocol and the real-file adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Research_Pipeline.ben_handoff import convert_ben_file
from Research_Pipeline.data import PAPER_OPTIONS, build_monthly_panel, load_handoff
from Research_Pipeline.factors import FACTOR_COLUMNS, newey_west_alpha, newey_west_paired_mean
from Research_Pipeline.paper import normalize_by_month, paper_feature_sets, train_paper
from Research_Pipeline.paper_reference import compare_table_six
from Research_Pipeline.paper_descriptives import describe_panel
from Research_Pipeline.run import run_pipeline
from Research_Pipeline.reconcile import reconcile_features
from Research_Pipeline.synthetic import write_synthetic_handoff


def test_table_three_specifications_and_monthly_normalization(tmp_path):
    assert len(paper_feature_sets()) == 19
    comparisons = compare_table_six({})["models"]
    assert len(comparisons) == 38
    assert {key.split("/")[0] for key in comparisons} == set(paper_feature_sets())
    assert comparisons["SDGTV_cp/lightgbm/full"]["paper"]["sharpe"] == 1.37
    assert paper_feature_sets()["B"] == ["AGE", "DTURN", "EXRET", "TANG", "TSKEW", "TVOL", "RM12", "SG", "SIZE"]
    assert paper_feature_sets()["S_cp"][-2:] == ["avg_call_iv", "avg_put_iv"]
    root = tmp_path / "input"
    write_synthetic_handoff(root, n_stocks=10, n_months=18, seed=11)
    panel, _ = build_monthly_panel(load_handoff(root))
    description = describe_panel(panel)
    assert sum(description["class_counts"].values()) == description["labeled_rows"]
    assert set(description["raw_option_feature_means_by_future_class"]) <= {"jump", "crash", "normal"}
    normalized = normalize_by_month(panel)
    for _, month in normalized.groupby("period"):
        assert abs(month.avg_call_iv.mean()) < 1e-10
        assert abs(month.AGE.mean()) < 1e-10


def test_paper_controls_use_one_and_six_month_lags(tmp_path):
    root = tmp_path / "input"
    write_synthetic_handoff(root, n_stocks=5, n_months=18, seed=12)
    panel, _ = build_monthly_panel(load_handoff(root), paper_control_lags=True)
    sample = panel[panel.period.eq(pd.Period("2016-01"))]
    assert not sample.empty
    assert (sample.monthly_controls_as_of_date.dt.to_period("M") <= pd.Period("2015-12")).all()
    assert (sample.quarterly_controls_as_of_date.dt.to_period("M") <= pd.Period("2015-07")).all()


def test_paper_sample_requires_both_option_sides_and_prior_dollar_price(tmp_path):
    root = tmp_path / "input"
    write_synthetic_handoff(root, n_stocks=5, n_months=18, seed=13)
    tables = load_handoff(root)
    prices = tables["stock_prices"]
    jan = prices.trade_date.dt.to_period("M").eq(pd.Period("2015-01"))
    jan_end = prices.loc[jan, "trade_date"].max()
    prices.loc[prices.trade_date.eq(jan_end) & prices.security_id.eq("S0000"), "close"] = 0.5
    options = tables["options_features"]
    feb = options.trade_date.dt.to_period("M").eq(pd.Period("2015-02"))
    options.loc[feb & options.security_id.eq("S0001"), "avg_put_iv"] = np.nan
    panel, coverage = build_monthly_panel(tables, paper_sample_filters=True)
    feb_rows = panel[panel.period.eq(pd.Period("2015-02"))]
    assert not feb_rows.security_id.isin(["S0000", "S0001"]).any()
    assert coverage["paper_both_sides_rows"] < coverage["month_end_option_rows"]
    assert coverage["paper_price_eligible_rows"] < coverage["paper_both_sides_rows"]


def test_point_in_time_paper_schedule_excludes_unmatured_labels(tmp_path):
    root = tmp_path / "input"
    write_synthetic_handoff(root, n_stocks=24, n_months=48, seed=8)
    panel, _ = build_monthly_panel(load_handoff(root))
    kwargs = dict(first_test="2017-01", last_test="2017-02", min_train_months=4,
                  specifications=["B", "S_cp"], seed=3)
    predictions, safe = train_paper(panel, label_policy="point_in_time", **kwargs)
    assert len(predictions) == 2 * 24 * 2 * 2
    assert set(predictions.feature_set) == {"B", "S_cp"}
    assert set(predictions.model) == {"logistic", "lightgbm"}
    assert safe["training_windows"][0]["latest_training_period"] <= "2016-01"
    _, hindsight = train_paper(panel, label_policy="paper_stated", **kwargs)
    assert hindsight["hindsight_training_labels"]
    assert hindsight["training_windows"][0]["latest_training_period"] == "2016-12"


def test_factor_alpha_and_missing_month_gate():
    rng = np.random.default_rng(9)
    periods = pd.period_range("2018-01", periods=48, freq="M")
    factors = pd.DataFrame({"period": periods + 1, "source_version": "fixture"})
    for col in FACTOR_COLUMNS:
        factors[col] = rng.normal(0, 0.02, size=len(periods)) if col != "RF" else 0.001
    gross = 0.02 + 0.3 * factors["Mkt-RF"].to_numpy()
    monthly = pd.DataFrame({"period": periods, "gross_return": gross})
    result = newey_west_alpha(monthly, factors)
    assert result["alpha_monthly"] == pytest.approx(0.02, abs=1e-10)
    assert result["factor_betas"]["Mkt-RF"] == pytest.approx(0.3, abs=1e-10)
    with pytest.raises(ValueError, match="missing factor months"):
        newey_west_alpha(monthly, factors.iloc[1:])


def test_paired_newey_west_requires_same_months():
    months = pd.period_range("2020-01", periods=12, freq="M")
    baseline = pd.DataFrame({"period": months, "gross_return": np.linspace(-0.04, 0.05, 12)})
    candidate = baseline.copy()
    candidate["gross_return"] += np.linspace(0.01, 0.04, 12)
    result = newey_west_paired_mean(candidate, baseline)
    assert result["mean_monthly_difference"] == pytest.approx(0.025)
    assert result["newey_west_t"] > 0
    with pytest.raises(ValueError, match="months differ"):
        newey_west_paired_mean(candidate.iloc[1:], baseline)


def test_wide_adapter_flags_proxies_and_conservative_availability(tmp_path):
    rows = []
    for day, next_day in (("2024-04-30", "2024-05-01"), ("2024-05-01", "2024-05-02")):
        for ticker in ("AAA", "BBB"):
            item = {"trade_date": day, "tradable_date": next_day, "underlying": ticker,
                    "stock_close": 10.0, "adj_close": 10.0, "stock_volume": 1000,
                    "market_cap": 1e6, "is_sp500": True,
                    "options_available_time": f"{next_day}T07:15:00Z"}
            item.update({name: 0.2 for name in PAPER_OPTIONS})
            rows.append(item)
    source = tmp_path / "ml_dataset_daily.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)
    with pytest.raises(ValueError, match="ticker_history"):
        convert_ben_file(source, tmp_path / "refused")
    report = convert_ben_file(source, tmp_path / "handoff", allow_ticker_proxy=True)
    assert report["original_availability_before_assumed_ready"] == 4
    assert report["identity"] == "ticker_proxy_pilot_only"
    tables = load_handoff(tmp_path / "handoff")
    assert tables["options_features"].security_id.str.startswith("ticker_proxy:").all()
    assert tables["options_features"].available_at.dt.hour.min() >= 15
    assert tables["sp500_membership"].empty
    assert tables["options_features"].is_us_common_stock.isna().all()
    pilot = run_pipeline(tmp_path / "handoff", tmp_path / "pilot", pilot=True,
                         pilot_train_months=1, pilot_test_months=3, iterations=1)
    assert pilot["status"] == "blocked_data_gate"
    assert (tmp_path / "pilot" / "coverage.json").exists()


def test_calculator_reconciliation_catches_one_changed_value(tmp_path):
    rows = [{"trade_date": "2024-04-30", "underlying": ticker,
             **{name: 0.2 for name in PAPER_OPTIONS}} for ticker in ("AAA", "BBB")]
    wide = tmp_path / "wide.parquet"
    pd.DataFrame(rows).to_parquet(wide, index=False)
    features = tmp_path / "features"
    features.mkdir()
    for name in PAPER_OPTIONS:
        pd.DataFrame(rows)[["trade_date", "underlying", name]].to_csv(features / f"{name}.csv", index=False)
    assert reconcile_features(wide, features, month="2024-04")["all_match"]
    path = features / "avg_put_iv.csv"
    changed = pd.read_csv(path)
    changed.loc[0, "avg_put_iv"] = 0.25
    changed.to_csv(path, index=False)
    report = reconcile_features(wide, features, month="2024-04")
    assert not report["all_match"]
    assert report["features"]["avg_put_iv"]["value_mismatches"] == 1
