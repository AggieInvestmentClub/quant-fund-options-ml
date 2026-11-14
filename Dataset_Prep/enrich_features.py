"""Feature enrichment module for ML modeling, backtesting, and academic paper replication.

Computes:
1. Academic company controls from Andreou, Han, and Li (2025):
   - SIZE (log market cap)
   - TVOL (trailing 20d & 60d realized volatility)
   - TSKEW (trailing 60d return skewness)
   - RM1, RM3, RM12 (1-month, 3-month, 12-month momentum)
   - EXRET (daily excess return over risk-free rate)
   - DTURN (detrended turnover over 20d and 252d)
2. Compact option and variance features (from research study Part II):
   - IV Level (mean of call & put IV)
   - IV Gap (call IV minus put IV)
   - Variance Proxy (IV Level squared minus realized variance)
   - 1-week and 4-week changes in IV Gap (delta_5d_iv_gap, delta_20d_iv_gap)
   - 1-week change in 60d Risk Reversal (delta_5d_rr_60d)
   - Robust coalesced skew/RR metrics (falling back to surface aggregates)
3. Multi-horizon targets and jump/crash labels for ML:
   - forward_1d_return, forward_5d_return, forward_10d_return, forward_21d_return, forward_63d_return
   - jump_label_5d, crash_label_5d (5% threshold)
   - jump_label_21d, crash_label_21d (10% threshold)
   - is_actively_trading flag
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def load_risk_free_rate_table(risk_free_csv: Path) -> pd.DataFrame:
    """Load daily 3-month Treasury risk-free rate from FRED CSV."""
    if not risk_free_csv.exists():
        return pd.DataFrame(columns=["trade_date", "daily_risk_free_rate"])

    try:
        rf_df = pd.read_csv(risk_free_csv)
        if "Date" not in rf_df.columns or "risk_free_rate" not in rf_df.columns:
            return pd.DataFrame(columns=["trade_date", "daily_risk_free_rate"])

        rf_df = rf_df.dropna(subset=["Date", "risk_free_rate"]).copy()
        rf_df["trade_date"] = rf_df["Date"].astype(str)
        # Annualized rate in decimal divided by 252 trading days
        rf_df["daily_risk_free_rate"] = rf_df["risk_free_rate"].astype(float) / 252.0
        return rf_df[["trade_date", "daily_risk_free_rate"]].drop_duplicates("trade_date")
    except Exception as exc:
        print(f"Warning: Failed to load risk-free rate table: {exc}")
        return pd.DataFrame(columns=["trade_date", "daily_risk_free_rate"])


def enrich_stock_history_time_series(
    stocks_df: pd.DataFrame,
    risk_free_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Calculate rolling volatility, momentum, detrended turnover, and forward returns across full stock history."""
    if stocks_df.empty:
        return stocks_df

    df = stocks_df.copy()
    if "Date" in df.columns:
        df["trade_date"] = df["Date"].astype(str)
    elif "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str)

    df["underlying"] = df["underlying"].astype(str)
    df = df.sort_values(["underlying", "trade_date"]).reset_index(drop=True)

    # 1. Split-Adjusted Open, High, and Low
    if "stock_close" in df.columns and "stock_adj_close" in df.columns:
        close_valid = df["stock_close"].fillna(0.0) > 0.0
        adj_factor = np.where(close_valid, df["stock_adj_close"] / df["stock_close"], 1.0)
        df["stock_adj_open"] = (df["stock_open"] * adj_factor).astype("float64") if "stock_open" in df.columns else df["stock_adj_close"].astype("float64")
        df["stock_adj_high"] = (df["stock_high"] * adj_factor).astype("float64") if "stock_high" in df.columns else df["stock_adj_close"].astype("float64")
        df["stock_adj_low"] = (df["stock_low"] * adj_factor).astype("float64") if "stock_low" in df.columns else df["stock_adj_close"].astype("float64")
    else:
        df["stock_adj_open"] = df["stock_adj_close"].astype("float64") if "stock_adj_close" in df.columns else np.nan
        df["stock_adj_high"] = df["stock_adj_close"].astype("float64") if "stock_adj_close" in df.columns else np.nan
        df["stock_adj_low"] = df["stock_adj_close"].astype("float64") if "stock_adj_close" in df.columns else np.nan

    # 2. Market Cap & Size
    if "market_cap" in df.columns and df["market_cap"].notna().any():
        df["size"] = np.log(np.maximum(df["market_cap"].fillna(1.0), 1.0)).astype("float32")
    else:
        df["size"] = np.nan

    # 3. Daily Returns & Volatility
    adj = df["stock_adj_close"]
    prev_adj = df.groupby("underlying")["stock_adj_close"].shift(1)
    daily_pct = (adj / prev_adj) - 1.0
    log_ret = np.log(np.maximum(adj / prev_adj, 1e-8))
    # Replace infinite or NaN returns
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan)

    # Rolling Volatility (20-day and 60-day annualized)
    tvol_20d = df.groupby("underlying")["stock_adj_close"].transform(
        lambda s: s.pct_change(fill_method=None).rolling(20, min_periods=5).std() * np.sqrt(252.0)
    )
    tvol_60d = df.groupby("underlying")["stock_adj_close"].transform(
        lambda s: s.pct_change(fill_method=None).rolling(60, min_periods=15).std() * np.sqrt(252.0)
    )
    tskew_60d = df.groupby("underlying")["stock_adj_close"].transform(
        lambda s: s.pct_change(fill_method=None).rolling(60, min_periods=15).skew()
    )

    df["tvol_20d"] = tvol_20d.astype("float32")
    # Fallback to 20d annualized volatility if 60d window not yet filled
    df["tvol_60d"] = tvol_60d.fillna(tvol_20d).astype("float32")
    df["tskew_60d"] = tskew_60d.astype("float32")

    # 4. Momentum (1-Month, 3-Month, 12-Month)
    df["rm1"] = df.groupby("underlying")["stock_adj_close"].transform(
        lambda s: (s / s.shift(21)) - 1.0
    ).astype("float32")

    def calc_rm3(s: pd.Series) -> pd.Series:
        m3 = (s / s.shift(63)) - 1.0
        # If 63-day history not yet available, compound available 1-month momentum
        if m3.isna().any():
            rm1_s = (s / s.shift(21)) - 1.0
            compounded_1m = ((1.0 + rm1_s) ** 3.0) - 1.0
            m3 = m3.fillna(compounded_1m)
        return m3

    df["rm3"] = df.groupby("underlying")["stock_adj_close"].transform(calc_rm3).astype("float32")

    def calc_rm12(s: pd.Series) -> pd.Series:
        m12 = (s / s.shift(252)) - 1.0
        # If full 252-day history not yet available (early 2024), compound available 3-month momentum
        if m12.isna().any():
            rm3_s = (s / s.shift(63)) - 1.0
            compounded = ((1.0 + rm3_s) ** 4.0) - 1.0
            m12 = m12.fillna(compounded)
        # If 3-month also unavailable, compound 1-month momentum
        if m12.isna().any():
            rm1_s = (s / s.shift(21)) - 1.0
            compounded_1m = ((1.0 + rm1_s) ** 12.0) - 1.0
            m12 = m12.fillna(compounded_1m)
        return m12

    df["rm12"] = df.groupby("underlying")["stock_adj_close"].transform(calc_rm12).astype("float32")

    # 5. Excess Return (EXRET)
    daily_pct_clean = daily_pct.fillna(0.0)
    if risk_free_csv and risk_free_csv.exists():
        rf_table = load_risk_free_rate_table(risk_free_csv)
        if not rf_table.empty:
            df = df.merge(rf_table, on="trade_date", how="left")
            df["daily_risk_free_rate"] = df["daily_risk_free_rate"].ffill().bfill().fillna(0.05 / 252.0)
            df["exret_daily"] = (daily_pct_clean - df["daily_risk_free_rate"]).astype("float32")
            df = df.drop(columns=["daily_risk_free_rate"])
        else:
            df["exret_daily"] = (daily_pct_clean - (0.05 / 252.0)).astype("float32")
    else:
        df["exret_daily"] = (daily_pct_clean - (0.05 / 252.0)).astype("float32")

    # 6. Detrended Turnover (DTURN), Dollar Volume & Volume Ratio
    if "stock_volume" in df.columns:
        if "stock_close" in df.columns:
            df["dollar_volume"] = (df["stock_close"] * df["stock_volume"]).astype("float64")
        else:
            df["dollar_volume"] = np.nan

        vol_ma20 = df.groupby("underlying")["stock_volume"].transform(
            lambda s: s.rolling(20, min_periods=5).mean()
        )
        df["volume_ratio_20d"] = (df["stock_volume"] / np.maximum(vol_ma20, 1.0)).astype("float32")

        if "shares_outstanding" in df.columns:
            turnover = df["stock_volume"] / np.maximum(df["shares_outstanding"].fillna(1.0), 1.0)
            df["_turnover_temp"] = turnover
            mean_20d = df.groupby("underlying")["_turnover_temp"].transform(
                lambda s: s.rolling(20, min_periods=5).mean()
            )
            mean_252d = df.groupby("underlying")["_turnover_temp"].transform(
                lambda s: s.rolling(252, min_periods=30).mean()
            )
            df.drop(columns=["_turnover_temp"], inplace=True)
            df["dturn_20d"] = (turnover - mean_20d).astype("float32")
            # Fallback to 20-day mean turnover if 252-day history not yet accumulated
            df["dturn_252d"] = (turnover - mean_252d.fillna(mean_20d)).astype("float32")
        else:
            df["dturn_20d"] = np.nan
            df["dturn_252d"] = np.nan
    else:
        df["dollar_volume"] = np.nan
        df["volume_ratio_20d"] = np.nan
        df["dturn_20d"] = np.nan
        df["dturn_252d"] = np.nan

    # 7. Multi-Horizon Forward Return Targets
    # Legacy close-to-close returns
    df["forward_1d_return"] = df.groupby("underlying")["stock_adj_close"].transform(
        lambda s: (s.shift(-1) / s) - 1.0
    ).astype("float32")
    df["forward_5d_return"] = df.groupby("underlying")["stock_adj_close"].transform(
        lambda s: (s.shift(-5) / s) - 1.0
    ).astype("float32")
    df["forward_10d_return"] = df.groupby("underlying")["stock_adj_close"].transform(
        lambda s: (s.shift(-10) / s) - 1.0
    ).astype("float32")
    df["forward_21d_return"] = df.groupby("underlying")["stock_adj_close"].transform(
        lambda s: (s.shift(-21) / s) - 1.0
    ).astype("float32")
    df["forward_63d_return"] = df.groupby("underlying")["stock_adj_close"].transform(
        lambda s: (s.shift(-63) / s) - 1.0
    ).astype("float32")

    # Tradable execution targets (entry at T+1 Open when morning OPRA signals are available)
    adj_open_next = df.groupby("underlying")["stock_adj_open"].shift(-1)
    adj_close_next = df.groupby("underlying")["stock_adj_close"].shift(-1)
    adj_close_5d = df.groupby("underlying")["stock_adj_close"].shift(-5)
    adj_close_21d = df.groupby("underlying")["stock_adj_close"].shift(-21)

    # 1-day intraday return on tradable date (Open to Close on T+1)
    df["target_open_to_close_1d"] = (
        (adj_close_next / np.maximum(adj_open_next, 1e-4)) - 1.0
    ).astype("float32")

    # 5-day tradable holding return (Open on T+1 to Close on T+5)
    df["target_tradable_5d_return"] = (
        (adj_close_5d / np.maximum(adj_open_next, 1e-4)) - 1.0
    ).astype("float32")

    # 21-day tradable holding return (Open on T+1 to Close on T+21)
    df["target_tradable_21d_return"] = (
        (adj_close_21d / np.maximum(adj_open_next, 1e-4)) - 1.0
    ).astype("float32")

    # Target validity masks (True if future return exists, False near dataset boundary)
    df["is_target_valid_5d"] = df["target_tradable_5d_return"].notna().astype(bool)
    df["is_target_valid_21d"] = df["target_tradable_21d_return"].notna().astype(bool)

    # Market-relative excess returns (alpha over daily S&P 500 cross-sectional mean)
    mkt_mean_5d = df.groupby("trade_date")["forward_5d_return"].transform("mean")
    mkt_mean_21d = df.groupby("trade_date")["forward_21d_return"].transform("mean")
    df["forward_5d_excess_return"] = (df["forward_5d_return"] - mkt_mean_5d).astype("float32")
    df["forward_21d_excess_return"] = (df["forward_21d_return"] - mkt_mean_21d).astype("float32")
    df["outperformance_label_5d"] = (df["forward_5d_excess_return"] > 0.0).fillna(False).astype(bool)
    df["outperformance_label_21d"] = (df["forward_21d_excess_return"] > 0.0).fillna(False).astype(bool)

    # 8. Classification Jump & Crash Labels
    df["jump_label_5d"] = (df["forward_5d_return"] >= 0.05).fillna(False).astype(bool)
    df["crash_label_5d"] = (df["forward_5d_return"] <= -0.05).fillna(False).astype(bool)
    df["jump_label_21d"] = (df["forward_21d_return"] >= 0.10).fillna(False).astype(bool)
    df["crash_label_21d"] = (df["forward_21d_return"] <= -0.10).fillna(False).astype(bool)

    # 9. Active Trading & Delisting Flag
    vol_valid = df["stock_volume"].fillna(0.0) > 0.0
    price_valid = df["stock_close"].fillna(0.0) > 0.0
    df["is_actively_trading"] = (vol_valid & price_valid).astype(bool)

    return df


def enrich_single_day_option_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate compact option signals, variance proxy, generalized term slopes, and robust coalesced skew/RR metrics."""
    out = df.copy()

    # 1. Compact Option Metrics & Multi-source ATM IV Coalescing
    has_call_iv = "avg_call_iv" in out.columns
    has_put_iv = "avg_put_iv" in out.columns

    call_iv = out["avg_call_iv"] if has_call_iv else pd.Series(np.nan, index=out.index)
    put_iv = out["avg_put_iv"] if has_put_iv else pd.Series(np.nan, index=out.index)

    # Row-level averaging: if both valid take mean; otherwise take whichever side is available
    avg_both = (call_iv + put_iv) / 2.0
    out["iv_level"] = avg_both.fillna(call_iv).fillna(put_iv).astype("float32")
    out["iv_gap"] = (call_iv - put_iv).astype("float32")

    # 25-delta proxy for ATM when exact ATM contracts are missing
    call_25 = out["call_25_delta_iv"] if "call_25_delta_iv" in out.columns else pd.Series(np.nan, index=out.index)
    put_25 = out["put_25_delta_iv"] if "put_25_delta_iv" in out.columns else pd.Series(np.nan, index=out.index)
    mid_25 = ((call_25 + put_25) / 2.0).fillna(call_25).fillna(put_25)

    # 30d ATM IV
    atm_30 = out["atm_iv_30d"] if "atm_iv_30d" in out.columns else pd.Series(np.nan, index=out.index)

    # Coalesce ATM IV across Stage 2 and Stage 1 with wing proxy fallback
    if "atm_iv" in out.columns:
        out["atm_iv"] = (
            out["atm_iv"]
            .fillna(out["iv_level"])
            .fillna(call_iv)
            .fillna(put_iv)
            .fillna(atm_30)
            .fillna(mid_25)
            .astype("float32")
        )
    else:
        out["atm_iv"] = out["iv_level"].fillna(atm_30).fillna(mid_25).astype("float32")

    # If iv_level was missing, fallback to atm_iv
    out["iv_level"] = out["iv_level"].fillna(out["atm_iv"]).astype("float32")

    # 2. Variance Risk Premium Proxy & Dimensionless IV/RV Ratio
    if "tvol_20d" in out.columns:
        iv_for_var = out["iv_level"].fillna(out["atm_iv"])
        out["variance_proxy"] = (
            (iv_for_var ** 2) - (out["tvol_20d"] ** 2)
        ).astype("float32")
        out["iv_to_rv_ratio"] = (
            iv_for_var / np.maximum(out["tvol_20d"], 1e-4)
        ).astype("float32")
    else:
        out["variance_proxy"] = np.nan
        out["iv_to_rv_ratio"] = np.nan

    # 3. Generalized Multi-Point Term Slopes (normalized to 60-day baseline)
    if "atm_iv_30d" in out.columns and "atm_iv_60d" in out.columns and "atm_iv_90d" in out.columns:
        s_90_30 = out["atm_iv_90d"] - out["atm_iv_30d"]
        s_60_30 = (out["atm_iv_60d"] - out["atm_iv_30d"]) * 2.0
        s_90_60 = (out["atm_iv_90d"] - out["atm_iv_60d"]) * 2.0
        out["atm_term_slope"] = s_90_30.fillna(s_60_30).fillna(s_90_60).astype("float32")

    if "put_25_delta_iv_30d" in out.columns and "put_25_delta_iv_60d" in out.columns and "put_25_delta_iv_90d" in out.columns:
        ps_90_30 = out["put_25_delta_iv_90d"] - out["put_25_delta_iv_30d"]
        ps_60_30 = (out["put_25_delta_iv_60d"] - out["put_25_delta_iv_30d"]) * 2.0
        ps_90_60 = (out["put_25_delta_iv_90d"] - out["put_25_delta_iv_60d"]) * 2.0
        out["put_skew_term_slope"] = ps_90_30.fillna(ps_60_30).fillna(ps_90_60).astype("float32")

    if "risk_reversal_25_delta_30d" in out.columns and "risk_reversal_25_delta_60d" in out.columns and "risk_reversal_25_delta_90d" in out.columns:
        rr_90_30 = out["risk_reversal_25_delta_90d"] - out["risk_reversal_25_delta_30d"]
        rr_60_30 = (out["risk_reversal_25_delta_60d"] - out["risk_reversal_25_delta_30d"]) * 2.0
        rr_90_60 = (out["risk_reversal_25_delta_90d"] - out["risk_reversal_25_delta_60d"]) * 2.0
        out["rr_term_slope"] = rr_90_30.fillna(rr_60_30).fillna(rr_90_60).astype("float32")

    # 4. Robust Skew & Risk Reversal Coalescing (Multi-tier fallback)
    rr_surf = out["risk_reversal_25_delta"] if "risk_reversal_25_delta" in out.columns else pd.Series(np.nan, index=out.index)
    rr_10 = out["risk_reversal_10_delta"] if "risk_reversal_10_delta" in out.columns else pd.Series(np.nan, index=out.index)
    iv_gap = out["iv_gap"] if "iv_gap" in out.columns else pd.Series(np.nan, index=out.index)
    skew_proj = (0.4033 * iv_gap - 0.0179).astype("float32")
    scaled_10 = (rr_10 * 0.60).astype("float32")

    rr_30 = out["risk_reversal_25_delta_30d"] if "risk_reversal_25_delta_30d" in out.columns else pd.Series(np.nan, index=out.index)
    rr_60 = out["risk_reversal_25_delta_60d"] if "risk_reversal_25_delta_60d" in out.columns else pd.Series(np.nan, index=out.index)
    rr_90 = out["risk_reversal_25_delta_90d"] if "risk_reversal_25_delta_90d" in out.columns else pd.Series(np.nan, index=out.index)

    diff_25_surf = (out["call_25_delta_iv"] - out["put_25_delta_iv"]).astype("float32") if "call_25_delta_iv" in out.columns and "put_25_delta_iv" in out.columns else pd.Series(np.nan, index=out.index)
    diff_25_30 = (out["call_25_delta_iv_30d"] - out["put_25_delta_iv_30d"]).astype("float32") if "call_25_delta_iv_30d" in out.columns and "put_25_delta_iv_30d" in out.columns else pd.Series(np.nan, index=out.index)
    diff_25_60 = (out["call_25_delta_iv_60d"] - out["put_25_delta_iv_60d"]).astype("float32") if "call_25_delta_iv_60d" in out.columns and "put_25_delta_iv_60d" in out.columns else pd.Series(np.nan, index=out.index)
    diff_25_90 = (out["call_25_delta_iv_90d"] - out["put_25_delta_iv_90d"]).astype("float32") if "call_25_delta_iv_90d" in out.columns and "put_25_delta_iv_90d" in out.columns else pd.Series(np.nan, index=out.index)

    out["risk_reversal_25_delta_30d_robust"] = (
        rr_30
        .fillna(diff_25_30)
        .fillna(rr_surf)
        .fillna(diff_25_surf)
        .fillna(rr_60)
        .fillna(diff_25_60)
        .fillna(rr_90)
        .fillna(diff_25_90)
        .fillna(scaled_10)
        .fillna(skew_proj)
        .astype("float32")
    )
    out["risk_reversal_25_delta_60d_robust"] = (
        rr_60
        .fillna(diff_25_60)
        .fillna(rr_surf)
        .fillna(diff_25_surf)
        .fillna(rr_30)
        .fillna(diff_25_30)
        .fillna(rr_90)
        .fillna(diff_25_90)
        .fillna(scaled_10)
        .fillna(skew_proj)
        .astype("float32")
    )

    # Robust ATM IV coalescing
    atm_surf = out["atm_iv"] if "atm_iv" in out.columns else pd.Series(np.nan, index=out.index)
    atm_30 = out["atm_iv_30d"] if "atm_iv_30d" in out.columns else pd.Series(np.nan, index=out.index)
    atm_60 = out["atm_iv_60d"] if "atm_iv_60d" in out.columns else pd.Series(np.nan, index=out.index)
    atm_90 = out["atm_iv_90d"] if "atm_iv_90d" in out.columns else pd.Series(np.nan, index=out.index)
    iv_lev = out["iv_level"] if "iv_level" in out.columns else pd.Series(np.nan, index=out.index)

    out["atm_iv_30d_robust"] = (
        atm_30.fillna(atm_surf).fillna(atm_60).fillna(atm_90).fillna(iv_lev).astype("float32")
    )

    # 5. Cross-Sectional Daily Percentile Ranks (bounded between 0.0 and 1.0)
    has_dates = "trade_date" in out.columns and out["trade_date"].nunique() >= 1
    for base_col, rank_col in [
        ("iv_level", "iv_level_cs_rank"),
        ("risk_reversal_25_delta_30d_robust", "risk_reversal_30d_cs_rank"),
        ("variance_proxy", "variance_proxy_cs_rank"),
    ]:
        if base_col in out.columns:
            if has_dates:
                out[rank_col] = out.groupby("trade_date")[base_col].rank(pct=True).astype("float32")
            else:
                out[rank_col] = out[base_col].rank(pct=True).astype("float32")
        else:
            out[rank_col] = np.nan

    # 6. Investable Options Universe Filter Flag
    is_sp500 = out["is_sp500"].fillna(False).astype(bool) if "is_sp500" in out.columns else pd.Series(False, index=out.index)
    is_active = out["is_actively_trading"].fillna(False).astype(bool) if "is_actively_trading" in out.columns else pd.Series(False, index=out.index)
    close_ok = (out["stock_close"].fillna(0.0) >= 5.0) if "stock_close" in out.columns else pd.Series(False, index=out.index)
    mcap_ok = (out["market_cap"].fillna(0.0) >= 2.0e9) if "market_cap" in out.columns else pd.Series(False, index=out.index)
    vol_ok = (out["stock_volume"].fillna(0.0) >= 100_000) if "stock_volume" in out.columns else pd.Series(False, index=out.index)
    contracts_ok = (out["n_contracts_total"].fillna(0.0) >= 10) if "n_contracts_total" in out.columns else pd.Series(False, index=out.index)
    iv_ok = out["iv_level"].notna() if "iv_level" in out.columns else pd.Series(False, index=out.index)

    out["is_options_investable"] = (
        is_sp500 & is_active & close_ok & mcap_ok & vol_ok & contracts_ok & iv_ok
    ).astype(bool)

    return out


def filter_investable_universe(
    df: pd.DataFrame,
    min_contracts: int = 10,
    min_market_cap: float = 2.0e9,
    min_stock_volume: float = 100_000,
    min_price: float = 5.0,
) -> pd.DataFrame:
    """Filter dataset to liquid, institutional, options-investable universe.

    Eliminates micro-caps, penny stocks, illiquid option chains (<10 contracts),
    and inactive trading days where option pricing is unreliable/stale.
    """
    if "is_options_investable" in df.columns:
        if min_contracts == 10 and min_market_cap == 2.0e9 and min_stock_volume == 100_000 and min_price == 5.0:
            return df[df["is_options_investable"] == True].copy()

    is_sp500 = (df["is_sp500"] == True) if "is_sp500" in df.columns else pd.Series(True, index=df.index)
    is_active = (df["is_actively_trading"] == True) if "is_actively_trading" in df.columns else pd.Series(True, index=df.index)
    close_ok = (df["stock_close"] >= min_price) if "stock_close" in df.columns else pd.Series(True, index=df.index)
    mcap_ok = (df["market_cap"] >= min_market_cap) if "market_cap" in df.columns else pd.Series(True, index=df.index)
    vol_ok = (df["stock_volume"] >= min_stock_volume) if "stock_volume" in df.columns else pd.Series(True, index=df.index)
    contracts_ok = (df["n_contracts_total"] >= min_contracts) if "n_contracts_total" in df.columns else pd.Series(True, index=df.index)
    iv_ok = df["iv_level"].notna() if "iv_level" in df.columns else pd.Series(True, index=df.index)

    mask = is_sp500 & is_active & close_ok & mcap_ok & vol_ok & contracts_ok & iv_ok
    return df[mask].copy()


def enrich_full_table_option_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate multi-day rolling changes in option signals across the full chronological table."""
    out = df.sort_values(["underlying", "trade_date"]).reset_index(drop=True)

    if "iv_gap" in out.columns:
        out["delta_5d_iv_gap"] = out.groupby("underlying")["iv_gap"].diff(5).astype("float32")
        out["delta_20d_iv_gap"] = out.groupby("underlying")["iv_gap"].diff(20).astype("float32")
    else:
        out["delta_5d_iv_gap"] = np.nan
        out["delta_20d_iv_gap"] = np.nan

    if "risk_reversal_25_delta_60d_robust" in out.columns:
        out["delta_5d_rr_60d"] = out.groupby("underlying")["risk_reversal_25_delta_60d_robust"].diff(5).astype("float32")
    elif "risk_reversal_25_delta" in out.columns:
        out["delta_5d_rr_60d"] = out.groupby("underlying")["risk_reversal_25_delta"].diff(5).astype("float32")
    else:
        out["delta_5d_rr_60d"] = np.nan

    # Re-apply cross-sectional percentile ranks across the full table
    if "trade_date" in out.columns:
        for base_col, rank_col in [
            ("iv_level", "iv_level_cs_rank"),
            ("risk_reversal_25_delta_30d_robust", "risk_reversal_30d_cs_rank"),
            ("variance_proxy", "variance_proxy_cs_rank"),
        ]:
            if base_col in out.columns:
                out[rank_col] = out.groupby("trade_date")[base_col].rank(pct=True).astype("float32")

    return out

