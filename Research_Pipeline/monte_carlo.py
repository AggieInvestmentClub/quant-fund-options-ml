"""Random-ranking null and moving-block uncertainty checks on held-out returns."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .backtest import backtest_one, portfolio_metrics


def _prepared_months(predictions: pd.DataFrame) -> list[dict]:
    months = []
    for _, frame in predictions.groupby("period", sort=True):
        frame = frame.sort_values("security_id")
        n = len(frame)
        cap = frame.market_cap.to_numpy(dtype=float)
        months.append({
            "ids": frame.security_id.to_numpy(dtype=str),
            "caps": cap,
            "returns": frame.stock_return.to_numpy(dtype=float),
            "jump": frame.p_jump.to_numpy(dtype=float),
            "crash": frame.p_crash.to_numpy(dtype=float),
            "bins": np.array_split(np.argsort(cap, kind="stable"), min(10, n)),
            "n_side": max(1, math.ceil(n * 0.05)),
        })
    return months


def randomized_mean_monthly_return(
    predictions: pd.DataFrame, rng: np.random.Generator, *, randomize: bool = True,
) -> float:
    """Fast equivalent of the portfolio rule for the repeated null simulation."""
    months = _prepared_months(predictions)
    return _mean_from_prepared(months, rng, randomize=randomize)


def _mean_from_prepared(months: list[dict], rng: np.random.Generator, *, randomize: bool) -> float:
    previous: dict[str, float] = {}
    net_returns = []
    for month_index, data in enumerate(months):
        ids, caps, stock_returns = data["ids"], data["caps"], data["returns"]
        jump, crash = data["jump"].copy(), data["crash"].copy()
        if randomize:
            for bucket in data["bins"]:
                if len(bucket) > 1:
                    donor = rng.permutation(bucket)
                    jump[bucket] = data["jump"][donor]
                    crash[bucket] = data["crash"][donor]
        n_side = data["n_side"]
        long_order = np.lexsort((ids, -jump))
        short_order = np.lexsort((ids, -crash))
        longs = set(long_order[:n_side])
        shorts = set(short_order[:n_side])
        for index in longs & shorts:
            if jump[index] >= crash[index]:
                shorts.remove(index)
            else:
                longs.remove(index)
        for index in long_order:
            if len(longs) >= n_side:
                break
            if index not in longs and index not in shorts:
                longs.add(index)
        for index in short_order:
            if len(shorts) >= n_side:
                break
            if index not in longs and index not in shorts:
                shorts.add(index)
        long_index, short_index = np.fromiter(sorted(longs), int), np.fromiter(sorted(shorts), int)
        weights = {
            **{ids[index]: float(caps[index] / caps[long_index].sum()) for index in long_index},
            **{ids[index]: float(-caps[index] / caps[short_index].sum()) for index in short_index},
        }
        turnover = sum(abs(weights.get(key, 0.0) - previous.get(key, 0.0))
                       for key in weights.keys() | previous.keys())
        if month_index == len(months) - 1:
            turnover += sum(abs(value) for value in weights.values())
        gross = sum(weight * stock_returns[np.searchsorted(ids, key)] for key, weight in weights.items())
        net_returns.append(gross - 0.003 * turnover)
        previous = weights
    return float(np.mean(net_returns))


def shuffle_scores_within_month_and_size(predictions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    shuffled = predictions.reset_index(drop=True).copy()
    for _, month in shuffled.groupby("period", sort=True):
        bins = pd.qcut(month.market_cap.rank(method="first"), q=min(10, len(month)), labels=False)
        for bin_id in bins.unique():
            target = month.index[bins.eq(bin_id)]
            if len(target) > 1:
                donor = rng.permutation(target)
                shuffled.loc[target, ["p_jump", "p_normal", "p_crash"]] = (
                    shuffled.loc[donor, ["p_jump", "p_normal", "p_crash"]].to_numpy()
                )
    return shuffled


def moving_block_sample(values: np.ndarray, rng: np.random.Generator, block_size: int = 3) -> np.ndarray:
    n = len(values)
    if n < block_size:
        raise ValueError(f"need at least {block_size} months for block resampling")
    starts = rng.integers(0, n - block_size + 1, size=int(np.ceil(n / block_size)))
    return np.concatenate([values[start:start + block_size] for start in starts])[:n]


def monte_carlo_one(
    predictions: pd.DataFrame, *, iterations: int = 2_000, seed: int = 17,
) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    _, observed_monthly = backtest_one(predictions)
    observed = portfolio_metrics(observed_monthly)
    observed_mean = observed["mean_monthly_return"]
    months = _prepared_months(predictions)
    null_means = np.empty(iterations)
    for index in range(iterations):
        null_means[index] = _mean_from_prepared(months, rng, randomize=True)
    series = observed_monthly.net_return_0pct_borrow.to_numpy(dtype=float)
    boot_annual, boot_sharpe = [], []
    # With one three-month block, every resample is the original series.
    # Suppress that degenerate interval rather than presenting it as certainty.
    if len(series) >= 6:
        for _ in range(iterations):
            sampled = moving_block_sample(series, rng)
            measures = portfolio_metrics(pd.DataFrame({"net_return_0pct_borrow": sampled}))
            if measures["annualized_return"] is not None:
                boot_annual.append(measures["annualized_return"])
            if measures["sharpe"] is not None:
                boot_sharpe.append(measures["sharpe"])
    return {
        "iterations": iterations, "seed": seed, "observed": observed,
        "random_signal_one_sided_p": float((1 + np.sum(null_means >= observed_mean)) / (iterations + 1)),
        "random_signal_mean_monthly_return_95pct": np.quantile(null_means, [0.025, 0.975]).tolist(),
        "block_bootstrap_annualized_return_95pct": np.quantile(boot_annual, [0.025, 0.975]).tolist() if boot_annual else None,
        "block_bootstrap_sharpe_95pct": np.quantile(boot_sharpe, [0.025, 0.975]).tolist() if boot_sharpe else None,
        "block_bootstrap_status": "available" if len(series) >= 6 else "unavailable: need at least two three-month blocks",
        "method": "paired probability shuffle within month and market-cap decile; three-month moving-block bootstrap",
    }


def monte_carlo_all(attached_predictions: pd.DataFrame, *, iterations: int = 2_000, seed: int = 17) -> dict:
    result = {}
    for (features, model), frame in attached_predictions.groupby(["feature_set", "model"]):
        for universe in ("full", "sp500"):
            key = f"{features}/{model}/{universe}"
            if universe == "sp500":
                if frame.sp500_member.isna().any():
                    result[key] = {"status": "unavailable: missing or stale membership snapshot"}
                    continue
                cohort = frame[frame.sp500_member]
                if cohort.period.nunique() != frame.period.nunique():
                    result[key] = {"status": "unavailable: empty S&P month"}
                    continue
            else:
                cohort = frame
            result[key] = monte_carlo_one(cohort, iterations=iterations, seed=seed)
    for universe in ("full", "sp500"):
        keys = [key for key, value in result.items()
                if key.endswith(f"/{universe}") and "random_signal_one_sided_p" in value]
        for key in keys:
            result[key]["random_signal_familywise_p_bonferroni"] = min(
                1.0, result[key]["random_signal_one_sided_p"] * len(keys),
            )
    return result
