"""Validate dated handoff tables and construct point-in-time monthly examples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_OPTIONS = [
    f"avg_{side}_{metric}"
    for metric in ("iv", "delta", "gamma", "theta", "vega")
    for side in ("call", "put")
]
PAPER_CONTROLS = ["AGE", "DTURN", "EXRET", "TANG", "TSKEW", "TVOL", "RM12", "SG", "SIZE"]
SKEW_FEATURES = [
    "atm_iv_30d", "atm_iv_60d", "atm_iv_90d",
    "risk_reversal_25_delta_30d", "risk_reversal_25_delta_60d",
    "risk_reversal_25_delta_90d", "put_skew_10_25_30d",
    "atm_term_slope", "rr_term_slope",
]
TABLES = ("options_features", "stock_prices", "stock_controls", "sp500_membership")
REQUIRED_TABLES = ("options_features", "stock_prices")


def _dates(frame: pd.DataFrame, column: str, table: str) -> None:
    if column not in frame:
        raise ValueError(f"{table}: missing {column}")
    frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize().astype("datetime64[ns]")
    if frame[column].isna().any():
        raise ValueError(f"{table}: null {column}")


def _timestamps(frame: pd.DataFrame, column: str, table: str) -> None:
    if column not in frame:
        raise ValueError(f"{table}: missing {column}")
    frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise").astype("datetime64[ns, UTC]")
    if frame[column].isna().any():
        raise ValueError(f"{table}: null {column}")


def _unique(frame: pd.DataFrame, keys: list[str], table: str) -> None:
    if frame.duplicated(keys).any():
        sample = frame.loc[frame.duplicated(keys, keep=False), keys].head(3).to_dict("records")
        raise ValueError(f"{table}: duplicate {keys}: {sample}")


def _source(frame: pd.DataFrame, table: str) -> None:
    if "source_version" not in frame or frame["source_version"].isna().any():
        raise ValueError(f"{table}: source_version is required on every row")
    if (frame["source_version"].astype(str).str.strip() == "").any():
        raise ValueError(f"{table}: empty source_version")


def _resolve_ids(frame: pd.DataFrame, date_col: str, table: str, mapping: pd.DataFrame | None) -> None:
    if "security_id" in frame:
        if frame["security_id"].isna().any():
            raise ValueError(f"{table}: null security_id")
        frame["security_id"] = frame["security_id"].astype(str)
        return
    if mapping is None or "ticker" not in frame:
        raise ValueError(f"{table}: supply security_id or ticker_history.parquet")
    candidate = frame.reset_index(names="_row").merge(mapping, on="ticker", how="left")
    valid = candidate[
        candidate[date_col].between(candidate["start_date"], candidate["end_date"])
    ]
    counts = valid.groupby("_row").size().reindex(range(len(frame)), fill_value=0)
    if not counts.eq(1).all():
        raise ValueError(f"{table}: ticker history has missing or ambiguous date mappings")
    frame["security_id"] = valid.sort_values("_row")["security_id"].astype(str).to_numpy()


def load_handoff(root: Path) -> dict[str, pd.DataFrame]:
    root = Path(root)
    mapping_path = root / "ticker_history.parquet"
    mapping = pd.read_parquet(mapping_path) if mapping_path.exists() else None
    if mapping is not None:
        required = {"ticker", "security_id", "start_date", "end_date"}
        if not required.issubset(mapping):
            raise ValueError(f"ticker_history: missing {sorted(required - set(mapping))}")
        _dates(mapping, "start_date", "ticker_history")
        _dates(mapping, "end_date", "ticker_history")
        if (mapping.start_date > mapping.end_date).any():
            raise ValueError("ticker_history: reversed date range")
    frames = {}
    date_cols = {
        "options_features": "trade_date", "stock_prices": "trade_date",
        "stock_controls": "as_of_date", "sp500_membership": "snapshot_date",
    }
    for name in TABLES:
        path = root / f"{name}.parquet"
        if not path.exists():
            if name in REQUIRED_TABLES:
                raise FileNotFoundError(path)
            date_col = date_cols[name]
            frame = pd.DataFrame({
                date_col: pd.Series(dtype="datetime64[ns]"),
                "available_at": pd.Series(dtype="datetime64[ns, UTC]"),
                "security_id": pd.Series(dtype="str"),
                "source_version": pd.Series(dtype="str"),
            })
            if name == "stock_controls":
                for col in PAPER_CONTROLS:
                    frame[col] = pd.Series(dtype=float)
            frames[name] = frame
            continue
        frame = pd.read_parquet(path)
        _dates(frame, date_cols[name], name)
        _timestamps(frame, "available_at", name)
        _source(frame, name)
        _resolve_ids(frame, date_cols[name], name, mapping)
        if (frame["available_at"].dt.tz_convert(None).dt.normalize() < frame[date_cols[name]]).any():
            raise ValueError(f"{name}: available_at precedes observation date")
        frames[name] = frame
    options, prices, controls, members = (frames[name] for name in TABLES)
    if "is_us_common_stock" not in options:
        options["is_us_common_stock"] = pd.Series(pd.NA, index=options.index, dtype="boolean")
    else:
        options["is_us_common_stock"] = options["is_us_common_stock"].astype("boolean")
    for name, frame, cols in (
        ("options_features", options, PAPER_OPTIONS),
        ("stock_prices", prices, ["close", "adjusted_close", "volume", "market_cap"]),
        ("stock_controls", controls, PAPER_CONTROLS),
    ):
        missing = set(cols) - set(frame)
        if missing:
            raise ValueError(f"{name}: missing {sorted(missing)}")
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            if np.isinf(frame[col].to_numpy(dtype=float)).any():
                raise ValueError(f"{name}: infinite {col}")
    for col in SKEW_FEATURES:
        if col in options:
            options[col] = pd.to_numeric(options[col], errors="coerce")
    _unique(options, ["trade_date", "security_id"], "options_features")
    _unique(prices, ["trade_date", "security_id"], "stock_prices")
    _unique(controls, ["as_of_date", "security_id"], "stock_controls")
    _unique(members, ["snapshot_date", "security_id"], "sp500_membership")
    if not members.empty and members.groupby("snapshot_date").available_at.nunique().max() != 1:
        raise ValueError("sp500_membership: every row in a snapshot must share available_at")
    if prices[["close", "adjusted_close", "volume"]].isna().any().any():
        raise ValueError("stock_prices: prices and volume cannot be missing")
    if (prices[["close", "adjusted_close", "market_cap"]] < 0).any().any():
        raise ValueError("stock_prices: close, adjusted_close and market_cap cannot be negative")
    if (prices["volume"] < 0).any():
        raise ValueError("stock_prices: negative volume")
    return frames


def market_close_utc(day: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=16)).tz_convert("UTC")


def build_monthly_panel(
    tables: dict[str, pd.DataFrame], *, min_snapshot_members: int = 0,
    paper_control_lags: bool = False,
    paper_sample_filters: bool = False,
) -> tuple[pd.DataFrame, dict]:
    options = tables["options_features"].copy()
    verified_common = int(options.is_us_common_stock.eq(True).fillna(False).sum())
    unknown_common = int(options.is_us_common_stock.isna().sum())
    options = options[options.is_us_common_stock.ne(False).fillna(True)].copy()
    prices = tables["stock_prices"].copy()
    controls = tables["stock_controls"].copy()
    members = tables["sp500_membership"].copy()
    prices["period"] = prices.trade_date.dt.to_period("M")
    all_periods = set(prices.period.unique())
    completed_periods = {period for period in all_periods if period + 1 in all_periods}
    ends = prices[prices.period.isin(completed_periods)].groupby("period").trade_date.max().sort_index()
    rebalance_dates = {
        period: prices.loc[prices.trade_date > day, "trade_date"].min()
        for period, day in ends.items()
    }
    options["period"] = options.trade_date.dt.to_period("M")
    options = options[options.trade_date.eq(options.period.map(ends))].copy()
    input_month_end_rows = len(options)
    paper_both_sides_rows = input_month_end_rows
    paper_price_eligible_rows = input_month_end_rows
    if paper_sample_filters:
        options = options[options.avg_call_iv.notna() & options.avg_put_iv.notna()].copy()
        paper_both_sides_rows = len(options)
        prior = prices[prices.trade_date.eq(prices.period.map(ends))][
            ["period", "security_id", "close"]
        ].copy()
        prior["period"] = prior.period + 1
        prior = prior.rename(columns={"close": "prior_month_close"})
        options = options.merge(prior, on=["period", "security_id"], how="left", validate="many_to_one")
        options = options[options.prior_month_close.ge(1)].copy()
        paper_price_eligible_rows = len(options)
    options["rebalance_date"] = options.period.map(rebalance_dates)
    options = options[options.rebalance_date.notna()].copy()
    options["rebalance_close_at"] = options.rebalance_date.map(market_close_utc)
    options = options[options.available_at.le(options.rebalance_close_at)].copy()
    timely_rows = len(options)
    month_price = prices.rename(columns={
        "available_at": "price_available_at", "source_version": "price_source_version",
    })
    keep_price = ["trade_date", "security_id", "price_available_at", "adjusted_close", "close", "volume", "market_cap", "price_source_version"]
    panel = options.merge(month_price[keep_price], on=["trade_date", "security_id"], how="inner", validate="one_to_one")
    panel = panel[panel.price_available_at.le(panel.available_at)].copy()
    priced_rows = len(panel)
    panel = panel[panel.adjusted_close.gt(0) & panel.market_cap.gt(0)].copy()
    eligible_price_rows = len(panel)
    panel = panel.rename(columns={"available_at": "score_at", "source_version": "options_source_version"})
    if panel.empty:
        for col in PAPER_CONTROLS:
            panel[col] = pd.Series(dtype=float)
        panel["label"] = pd.Series(dtype=str)
        panel["target_period"] = pd.PeriodIndex([], freq="M")
        panel["sp500_member"] = pd.Series(dtype="boolean")
        return panel, {
            "option_rows": len(tables["options_features"]),
            "verified_us_common_stock_rows": verified_common,
            "unknown_us_common_stock_rows": unknown_common,
            "month_end_option_rows": input_month_end_rows,
            "paper_both_sides_rows": paper_both_sides_rows,
            "paper_price_eligible_rows": paper_price_eligible_rows,
            "timely_month_end_rows": timely_rows,
            "rows_with_month_end_stock_price": priced_rows,
            "rows_with_positive_base_price_and_cap": eligible_price_rows,
            "monthly_rows": 0, "labeled_rows": 0,
            "missing_labels_with_target_month_present": 0,
            "rows_with_all_paper_controls": 0,
            "rows_with_known_sp500_membership": 0,
            "sp500_snapshot_member_counts": {
                str(day.date()): int(count)
                for day, count in members.groupby("snapshot_date").size().items()
            },
            "feature_nonnull_counts": {col: 0 for col in PAPER_OPTIONS + [c for c in SKEW_FEATURES if c in panel]},
            "first_month": None, "last_month": None,
            "month_end_rows_by_month": {}, "labeled_rows_by_month": {},
            "rows_with_all_paper_controls_by_month": {},
            "missing_stock_ids": sorted(set(tables["options_features"].security_id) - set(prices.security_id)),
        }
    controls = controls.rename(columns={"available_at": "controls_available_at", "source_version": "controls_source_version"})
    if controls.empty:
        panel["as_of_date"] = pd.NaT
        panel["controls_available_at"] = pd.NaT
        panel["controls_source_version"] = pd.NA
        for col in PAPER_CONTROLS:
            panel[col] = np.nan
    else:
        joined = []
        for day, month_rows in panel.groupby("trade_date", sort=True):
            if paper_control_lags:
                lagged = month_rows.copy()
                for lag, columns, prefix in (
                    (1, [c for c in PAPER_CONTROLS if c not in ("TANG", "SG")], "monthly"),
                    (6, ["TANG", "SG"], "quarterly"),
                ):
                    cutoff = (pd.Period(day, freq="M") - lag).to_timestamp(how="end").normalize()
                    eligible_lagged = controls[controls.as_of_date.le(cutoff)]
                    if eligible_lagged.empty:
                        lagged[f"{prefix}_controls_as_of_date"] = pd.NaT
                        lagged[f"{prefix}_controls_available_at"] = pd.NaT
                        for column in columns:
                            lagged[column] = np.nan
                        continue
                    names = {"as_of_date": f"{prefix}_controls_as_of_date",
                             "controls_available_at": f"{prefix}_controls_available_at",
                             "controls_source_version": f"{prefix}_controls_source_version"}
                    lookup = eligible_lagged[["security_id", "as_of_date", "controls_available_at",
                                              "controls_source_version", *columns]].rename(columns=names)
                    lagged = pd.merge_asof(
                        lagged.sort_values("score_at"),
                        lookup.sort_values(names["controls_available_at"]),
                        left_on="score_at", right_on=names["controls_available_at"],
                        by="security_id", direction="backward",
                    )
                lagged["as_of_date"] = lagged["monthly_controls_as_of_date"]
                lagged["controls_available_at"] = lagged["monthly_controls_available_at"]
                joined.append(lagged)
                continue
            eligible_controls = controls[controls.as_of_date.le(day)]
            if eligible_controls.empty:
                blank = month_rows.copy()
                blank["as_of_date"] = pd.NaT
                blank["controls_available_at"] = pd.NaT
                blank["controls_source_version"] = pd.NA
                for col in PAPER_CONTROLS:
                    blank[col] = np.nan
                joined.append(blank)
            else:
                joined.append(pd.merge_asof(
                    month_rows.sort_values("score_at"),
                    eligible_controls.sort_values("controls_available_at")[[
                        "security_id", "as_of_date", "controls_available_at", "controls_source_version", *PAPER_CONTROLS,
                    ]],
                    left_on="score_at", right_on="controls_available_at",
                    by="security_id", direction="backward",
                ))
        panel = pd.concat(joined, ignore_index=True)
    price_by_period = prices[["period", "security_id", "trade_date", "available_at", "adjusted_close"]].copy()
    price_by_period = price_by_period[price_by_period.trade_date.eq(price_by_period.period.map(ends))]
    price_by_period = price_by_period.rename(columns={
        "period": "target_period", "trade_date": "label_date",
        "available_at": "label_available_at", "adjusted_close": "target_adjusted_close",
    })
    panel["target_period"] = panel.period + 12
    panel = panel.merge(price_by_period, on=["target_period", "security_id"], how="left", validate="many_to_one")
    with np.errstate(divide="ignore", invalid="ignore"):
        panel["forward_log_return"] = np.log(panel.target_adjusted_close / panel.adjusted_close)
    panel["label"] = np.select(
        [panel.forward_log_return < -0.70, panel.forward_log_return > 0.70],
        ["crash", "jump"], default="normal",
    )
    panel.loc[panel.forward_log_return.isna(), "label"] = None
    panel["sp500_member"] = pd.Series(pd.NA, index=panel.index, dtype="boolean")
    panel["membership_snapshot_date"] = pd.NaT
    for day in panel.trade_date.drop_duplicates():
        decision_close = panel.loc[panel.trade_date.eq(day), "rebalance_close_at"].max()
        eligible = members[(members.snapshot_date <= day) & (members.available_at <= decision_close)]
        if eligible.empty:
            continue
        snapshot = eligible.snapshot_date.max()
        if (day - snapshot).days > 93:
            continue
        member_ids = set(eligible.loc[eligible.snapshot_date.eq(snapshot), "security_id"])
        if not (min_snapshot_members <= len(member_ids) <= 550):
            continue
        mask = panel.trade_date.eq(day)
        panel.loc[mask, "sp500_member"] = panel.loc[mask, "security_id"].isin(member_ids).to_numpy()
        panel.loc[mask, "membership_snapshot_date"] = snapshot
    panel = panel.sort_values(["trade_date", "security_id"]).reset_index(drop=True)
    observed_periods = set(ends.index)
    missing_matured_labels = int((panel.target_period.isin(observed_periods) & panel.target_adjusted_close.isna()).sum())
    coverage = {
        "option_rows": len(tables["options_features"]),
        "verified_us_common_stock_rows": verified_common,
        "unknown_us_common_stock_rows": unknown_common,
        "month_end_option_rows": input_month_end_rows,
        "paper_both_sides_rows": paper_both_sides_rows,
        "paper_price_eligible_rows": paper_price_eligible_rows,
        "timely_month_end_rows": timely_rows,
        "rows_with_month_end_stock_price": priced_rows,
        "rows_with_positive_base_price_and_cap": eligible_price_rows,
        "monthly_rows": len(panel),
        "labeled_rows": int(panel.label.notna().sum()),
        "missing_labels_with_target_month_present": missing_matured_labels,
        "rows_with_all_paper_controls": int(panel[PAPER_CONTROLS].notna().all(axis=1).sum()),
        "rows_with_known_sp500_membership": int(panel.sp500_member.notna().sum()),
        "sp500_snapshot_member_counts": {
            str(day.date()): int(count)
            for day, count in members.groupby("snapshot_date").size().items()
        },
        "feature_nonnull_counts": {col: int(panel[col].notna().sum()) for col in PAPER_OPTIONS + [c for c in SKEW_FEATURES if c in panel]},
        "first_month": str(panel.period.min()) if len(panel) else None,
        "last_month": str(panel.period.max()) if len(panel) else None,
        "month_end_rows_by_month": {str(month): int(count)
                                    for month, count in panel.groupby("period").size().items()},
        "labeled_rows_by_month": {str(month): int(count)
                                  for month, count in panel[panel.label.notna()].groupby("period").size().items()},
        "rows_with_all_paper_controls_by_month": {
            str(month): int(count) for month, count in
            panel[panel[PAPER_CONTROLS].notna().all(axis=1)].groupby("period").size().items()
        },
        "missing_stock_ids": sorted(set(tables["options_features"].security_id) - set(prices.security_id)),
    }
    return panel, coverage


def write_coverage(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
