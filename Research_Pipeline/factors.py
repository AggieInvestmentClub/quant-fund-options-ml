"""Fama-French five-factor plus momentum regression with Newey-West errors."""

from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_COLUMNS = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF")


def load_monthly_factors(path) -> pd.DataFrame:
    """Read an explicitly versioned factor file; returns must be decimal, not percent."""
    frame = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    required = {"period", "source_version", *FACTOR_COLUMNS}
    if not required.issubset(frame):
        raise ValueError(f"factor file missing {sorted(required - set(frame))}")
    if frame.source_version.isna().any() or frame.source_version.astype(str).str.strip().eq("").any():
        raise ValueError("factor source_version required")
    frame["period"] = pd.PeriodIndex(frame.period.astype(str), freq="M")
    if frame.period.duplicated().any():
        raise ValueError("duplicate factor month")
    for column in FACTOR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"nonfinite factor {column}")
    return frame


def newey_west_alpha(monthly: pd.DataFrame, factors: pd.DataFrame,
                     *, return_column: str = "gross_return", lags: int | None = None) -> dict:
    """Monthly OLS alpha and Bartlett HAC t statistic for the held return series."""
    if not {"period", return_column}.issubset(monthly):
        raise ValueError("monthly portfolio is missing its period or return")
    series = monthly[["period", return_column]].copy()
    # A signal formed at month t enters on the first close of t+1. Monthly
    # French factors are calendar-month returns, so this alignment is approximate.
    series["factor_period"] = pd.PeriodIndex(series.period.astype(str), freq="M") + 1
    joined = series.merge(factors, left_on="factor_period", right_on="period",
                          how="left", validate="many_to_one", suffixes=("", "_factor"))
    if joined[list(FACTOR_COLUMNS)].isna().any().any():
        raise ValueError("missing factor months for portfolio return intervals")
    n = len(joined)
    x = np.column_stack([np.ones(n), joined[list(FACTOR_COLUMNS[:-1])].to_numpy(dtype=float)])
    # The portfolio has a dollar long and a dollar short leg, so its raw
    # long-short return is the dependent variable in the paper's regression.
    y = joined[return_column].to_numpy(dtype=float)
    if n <= x.shape[1] + 1:
        raise ValueError("too few months for factor regression")
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ coefficients
    xtx_inverse = np.linalg.pinv(x.T @ x)
    lag_count = min(n - 1, lags if lags is not None else max(1, int(4 * (n / 100) ** (2 / 9))))
    if lag_count < 0:
        raise ValueError("lags cannot be negative")
    scores = x * residual[:, None]
    meat = scores.T @ scores
    for lag in range(1, lag_count + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lag_count + 1)) * (cross + cross.T)
    covariance = xtx_inverse @ meat @ xtx_inverse * n / (n - x.shape[1])
    se = float(np.sqrt(max(covariance[0, 0], 0)))
    alpha = float(coefficients[0])
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "months": n, "alpha_monthly": alpha, "alpha_annualized_linear": 12 * alpha,
        "alpha_newey_west_t": alpha / se if se > 0 else None,
        "newey_west_lags": lag_count,
        "factor_betas": dict(zip(FACTOR_COLUMNS[:-1], map(float, coefficients[1:]))),
        "r_squared": 1 - float(np.sum(residual ** 2)) / total if total > 0 else None,
        "factor_alignment": "calendar month after signal; approximate for first-close-to-first-close holdings",
    }


def newey_west_paired_mean(candidate: pd.DataFrame, baseline: pd.DataFrame,
                          *, column: str = "gross_return", lags: int | None = None) -> dict:
    """HAC t statistic for a model's monthly return difference from baseline."""
    paired = candidate[["period", column]].merge(
        baseline[["period", column]], on="period", how="outer", validate="one_to_one",
        suffixes=("_candidate", "_baseline"), indicator=True,
    ).sort_values("period")
    if not paired._merge.eq("both").all():
        raise ValueError("candidate and baseline months differ")
    difference = (paired[f"{column}_candidate"] - paired[f"{column}_baseline"]).to_numpy(dtype=float)
    if not np.isfinite(difference).all():
        raise ValueError("nonfinite paired monthly return")
    n = len(difference)
    if n < 3:
        raise ValueError("need at least three paired months")
    mean = float(difference.mean())
    centered = difference - mean
    lag_count = min(n - 1, lags if lags is not None else max(1, int(4 * (n / 100) ** (2 / 9))))
    if lag_count < 0:
        raise ValueError("lags cannot be negative")
    long_run_variance = float(centered @ centered / n)
    for lag in range(1, lag_count + 1):
        long_run_variance += 2 * (1 - lag / (lag_count + 1)) * float(
            centered[lag:] @ centered[:-lag] / n
        )
    standard_error = float(np.sqrt(max(long_run_variance, 0) / n))
    return {
        "months": n, "mean_monthly_difference": mean,
        "annualized_arithmetic_mean_difference": 12 * mean,
        "newey_west_t": mean / standard_error if standard_error > 0 else None,
        "newey_west_lags": lag_count,
    }
