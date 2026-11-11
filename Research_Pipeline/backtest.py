"""Monthly, delayed-execution long/short backtest with explicit turnover costs."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

def _next_rebalance(period: pd.Period, prices: pd.DataFrame) -> pd.Timestamp:
    future = prices[prices.trade_date.dt.to_period("M").eq(period + 1)]
    if future.empty:
        raise ValueError(f"no stock prices in month after {period}")
    next_month_end = future.trade_date.max()
    next_trade = prices.loc[prices.trade_date > next_month_end, "trade_date"].min()
    if pd.isna(next_trade):
        raise ValueError(f"no tradable close after {next_month_end.date()}")
    return next_trade


def attach_forward_trade_returns(predictions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach return between consecutive post-publication rebalance closes."""
    result = predictions.copy()
    result["exit_date"] = result.period.map({
        period: _next_rebalance(period, prices)
        for period in result.period.unique()
    })
    price_lookup = prices.set_index(["trade_date", "security_id"])["adjusted_close"]
    entry_keys = pd.MultiIndex.from_frame(result[["rebalance_date", "security_id"]].rename(columns={"rebalance_date": "trade_date"}))
    exit_keys = pd.MultiIndex.from_frame(result[["exit_date", "security_id"]].rename(columns={"exit_date": "trade_date"}))
    result["entry_adjusted_close"] = price_lookup.reindex(entry_keys).to_numpy()
    result["exit_adjusted_close"] = price_lookup.reindex(exit_keys).to_numpy()
    missing = result.entry_adjusted_close.isna() | result.exit_adjusted_close.isna()
    if missing.any():
        sample = result.loc[missing, ["period", "security_id", "rebalance_date", "exit_date"]].head(3).to_dict("records")
        raise ValueError(f"missing entry/exit stock prices for {int(missing.sum())} predictions: {sample}")
    if (result.entry_adjusted_close <= 0).any():
        raise ValueError("cannot enter a position at a nonpositive adjusted close")
    result["stock_return"] = result.exit_adjusted_close / result.entry_adjusted_close - 1
    return result


def _pick_sides(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = max(1, math.ceil(len(frame) * 0.05))
    longs = frame.sort_values(["p_jump", "security_id"], ascending=[False, True]).head(n)
    shorts = frame.sort_values(["p_crash", "security_id"], ascending=[False, True]).head(n)
    overlap = set(longs.security_id) & set(shorts.security_id)
    if overlap:
        long_ids = set(longs.security_id)
        short_ids = set(shorts.security_id)
        for security_id in overlap:
            item = frame.loc[frame.security_id.eq(security_id)].iloc[0]
            if item.p_jump >= item.p_crash:
                short_ids.remove(security_id)
            else:
                long_ids.remove(security_id)
        for item in frame.sort_values(["p_jump", "security_id"], ascending=[False, True]).itertuples():
            if len(long_ids) >= n:
                break
            if item.security_id not in long_ids | short_ids:
                long_ids.add(item.security_id)
        for item in frame.sort_values(["p_crash", "security_id"], ascending=[False, True]).itertuples():
            if len(short_ids) >= n:
                break
            if item.security_id not in long_ids | short_ids:
                short_ids.add(item.security_id)
        longs = frame[frame.security_id.isin(long_ids)]
        shorts = frame[frame.security_id.isin(short_ids)]
    if longs.empty or shorts.empty:
        raise ValueError("cannot form both long and short sides from this universe")
    return longs, shorts


def _positions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.security_id.duplicated().any():
        raise ValueError("duplicate stock prediction in a rebalance month")
    longs, shorts = _pick_sides(frame)
    pieces = []
    for side, selected, sign in (("long", longs, 1.0), ("short", shorts, -1.0)):
        if selected.market_cap.isna().any() or (selected.market_cap <= 0).any():
            raise ValueError("selected stock has missing market cap")
        picked = selected.copy()
        picked["side"] = side
        picked["weight"] = sign * picked.market_cap / picked.market_cap.sum()
        pieces.append(picked)
    return pd.concat(pieces, ignore_index=True)


def backtest_one(predictions: pd.DataFrame, *, cost_bps: float = 30.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not len(predictions):
        raise ValueError("empty predictions")
    if "stock_return" not in predictions:
        raise ValueError("call attach_forward_trade_returns first")
    trades, returns = [], []
    previous_weights: dict[str, float] = {}
    grouped = list(predictions.groupby("period", sort=True))
    for index, (period, frame) in enumerate(grouped):
        chosen = _positions(frame)
        weights = dict(zip(chosen.security_id, chosen.weight))
        turnover = sum(abs(weights.get(key, 0.0) - previous_weights.get(key, 0.0))
                       for key in set(weights) | set(previous_weights))
        if index == len(grouped) - 1:
            turnover += sum(abs(value) for value in weights.values())
        cost = turnover * cost_bps / 10_000
        gross = float((chosen.weight * chosen.stock_return).sum())
        long_return = float((chosen.loc[chosen.side.eq("long"), "weight"] *
                             chosen.loc[chosen.side.eq("long"), "stock_return"]).sum())
        short_return = float((chosen.loc[chosen.side.eq("short"), "weight"] *
                              chosen.loc[chosen.side.eq("short"), "stock_return"]).sum())
        borrow_3 = float(abs(chosen.loc[chosen.side.eq("short"), "weight"].sum()) * 0.03 / 12)
        returns.append({
            "period": period, "rebalance_date": chosen.rebalance_date.iloc[0],
            "exit_date": chosen.exit_date.iloc[0], "gross_return": gross,
            "long_return": long_return, "short_return": short_return,
            "transaction_cost": cost, "borrow_cost_3pct": borrow_3,
            "net_return_0pct_borrow": gross - cost,
            "net_return_3pct_borrow": gross - cost - borrow_3,
            "turnover": turnover,
            "n_long": int(chosen.side.eq("long").sum()),
            "n_short": int(chosen.side.eq("short").sum()),
        })
        trades.append(chosen[[
            "period", "security_id", "side", "weight", "p_jump", "p_crash",
            "rebalance_date", "exit_date", "stock_return",
        ]])
        previous_weights = weights
    return pd.concat(trades, ignore_index=True), pd.DataFrame(returns)


def portfolio_metrics(monthly: pd.DataFrame, return_col: str = "net_return_0pct_borrow") -> dict:
    values = monthly[return_col].to_numpy(dtype=float)
    if not len(values):
        return {"months": 0}
    wealth = np.cumprod(1 + values)
    annual_return = float(wealth[-1] ** (12 / len(values)) - 1) if np.all(wealth > 0) else None
    standard_deviation = np.std(values, ddof=1) if len(values) > 1 else 0.0
    sharpe = float(np.sqrt(12) * np.mean(values) / standard_deviation) if standard_deviation > 0 else None
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])
    drawdowns = np.r_[1.0, wealth] / peaks - 1
    return {
        "months": len(values), "annualized_return": annual_return,
        "annualized_arithmetic_mean": float(12 * np.mean(values)),
        "annualized_volatility": float(np.sqrt(12) * standard_deviation),
        "sharpe": sharpe, "max_drawdown": float(drawdowns.min()),
        "mean_monthly_return": float(np.mean(values)),
        "mean_monthly_turnover": float(monthly.turnover.mean()) if "turnover" in monthly else None,
    }


def backtest_all(predictions: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    attached = attach_forward_trade_returns(predictions, prices)
    all_trades, all_returns, report = [], [], {}
    for (features, model), frame in attached.groupby(["feature_set", "model"]):
        for universe in ("full", "sp500"):
            if universe == "sp500":
                if frame.sp500_member.isna().any():
                    report[f"{features}/{model}/sp500"] = {"status": "unavailable: missing or stale membership snapshot"}
                    continue
                chosen = frame[frame.sp500_member]
                if chosen.period.nunique() != frame.period.nunique():
                    report[f"{features}/{model}/sp500"] = {"status": "unavailable: empty S&P month"}
                    continue
            else:
                chosen = frame
            trades, monthly = backtest_one(chosen)
            for output in (trades, monthly):
                output["feature_set"] = features
                output["model"] = model
                output["universe"] = universe
            all_trades.append(trades)
            all_returns.append(monthly)
            report[f"{features}/{model}/{universe}"] = {
                "gross": portfolio_metrics(monthly, "gross_return"),
                "net_0pct_borrow": portfolio_metrics(monthly, "net_return_0pct_borrow"),
                "net_3pct_borrow": portfolio_metrics(monthly, "net_return_3pct_borrow"),
            }
    return pd.concat(all_trades, ignore_index=True), pd.concat(all_returns, ignore_index=True), report
