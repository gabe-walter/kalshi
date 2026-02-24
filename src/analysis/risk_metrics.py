"""
Risk metrics module.

Computes Sharpe, Sortino, drawdown analysis, Value at Risk, and Kelly
fraction analysis from trading data.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0,
                  annualization: float = 1.0) -> float:
    """Compute Sharpe ratio.

    Args:
        returns: Array of per-trade or per-period returns.
        risk_free: Risk-free rate per period.
        annualization: Factor to annualize (e.g. sqrt(252) for daily).
    """
    excess = returns - risk_free
    if len(excess) < 2 or np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess, ddof=1) * annualization)


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0,
                   annualization: float = 1.0) -> float:
    """Compute Sortino ratio (only penalizes downside volatility)."""
    excess = returns - risk_free
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("inf") if np.mean(excess) > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    return float(np.mean(excess) / downside_std * annualization)


def max_drawdown_analysis(equity_series: pd.DataFrame) -> dict:
    """Comprehensive drawdown analysis.

    Expects DataFrame with columns: timestamp, bankroll.
    Returns dict with max_drawdown, max_drawdown_pct, duration info.
    """
    if equity_series.empty or "bankroll" not in equity_series.columns:
        return {}

    df = equity_series.sort_values("timestamp").reset_index(drop=True)
    bankroll = df["bankroll"].values
    timestamps = df["timestamp"].values

    peak = np.maximum.accumulate(bankroll)
    drawdown = peak - bankroll
    drawdown_pct = drawdown / peak * 100

    max_dd_idx = np.argmax(drawdown)
    max_dd = float(drawdown[max_dd_idx])
    max_dd_pct = float(drawdown_pct[max_dd_idx])

    # Find the peak before max drawdown
    peak_idx = np.argmax(bankroll[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

    # Find recovery point (where bankroll exceeds previous peak)
    recovery_idx = None
    peak_val = bankroll[peak_idx]
    for i in range(max_dd_idx, len(bankroll)):
        if bankroll[i] >= peak_val:
            recovery_idx = i
            break

    result = {
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "peak_value": float(bankroll[peak_idx]),
        "trough_value": float(bankroll[max_dd_idx]),
        "peak_time": str(timestamps[peak_idx]),
        "trough_time": str(timestamps[max_dd_idx]),
    }

    if recovery_idx is not None:
        result["recovery_time"] = str(timestamps[recovery_idx])
        result["recovered"] = True
        duration = pd.Timestamp(timestamps[recovery_idx]) - pd.Timestamp(timestamps[peak_idx])
        result["drawdown_duration_hours"] = duration.total_seconds() / 3600
    else:
        result["recovered"] = False

    return result


def value_at_risk(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical Value at Risk.

    Returns the loss at the given confidence level (as a positive number).
    """
    if len(returns) < 2:
        return 0.0
    return float(-np.percentile(returns, (1 - confidence) * 100))


def expected_shortfall(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Expected Shortfall (Conditional VaR).

    Average loss beyond the VaR threshold.
    """
    if len(returns) < 2:
        return 0.0
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= threshold]
    if len(tail_losses) == 0:
        return 0.0
    return float(-np.mean(tail_losses))


def kelly_analysis(positions_df: pd.DataFrame) -> dict:
    """Analyze position sizing relative to Kelly criterion.

    For binary bets: f* = (bp - q) / b
    where b = odds, p = win prob, q = 1-p.

    Returns dict with theoretical Kelly, actual sizing stats.
    """
    if positions_df.empty:
        return {}

    closed = positions_df[positions_df["pnl"].notna()].copy()
    if closed.empty:
        return {}

    # Empirical win rate
    win_rate = (closed["pnl"] > 0).mean()
    loss_rate = 1 - win_rate

    # Average payoff ratio
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] <= 0]

    avg_win = wins["pnl"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl"].mean()) if len(losses) > 0 else 1

    # Kelly fraction: f* = (b*p - q) / b where b = avg_win/avg_loss
    b = avg_win / avg_loss if avg_loss > 0 else 0
    if b > 0:
        kelly_f = (b * win_rate - loss_rate) / b
    else:
        kelly_f = 0

    # Actual bet sizing (cost as fraction of bankroll at time of trade)
    actual_fractions = []
    if "cost" in closed.columns and "bankroll_after" in closed.columns:
        for _, row in closed.iterrows():
            bankroll = row.get("bankroll_after", 10000)
            cost = row.get("cost", 0)
            if bankroll > 0:
                actual_fractions.append(cost / bankroll)

    return {
        "empirical_win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "payoff_ratio": float(b),
        "kelly_fraction": float(kelly_f),
        "half_kelly": float(kelly_f / 2),
        "quarter_kelly": float(kelly_f / 4),
        "three_quarter_kelly": float(kelly_f * 0.75),
        "avg_actual_fraction": float(np.mean(actual_fractions)) if actual_fractions else None,
        "n_trades": len(closed),
    }


def exposure_over_time(equity_df: pd.DataFrame) -> pd.DataFrame:
    """Extract exposure and idle capital over time.

    Expects columns: timestamp, bankroll, open_exposure.
    Returns DataFrame with: timestamp, bankroll, exposure, idle, exposure_pct.
    """
    if equity_df.empty:
        return pd.DataFrame()

    cols = ["timestamp", "bankroll"]
    if "open_exposure" in equity_df.columns:
        cols.append("open_exposure")
    else:
        return pd.DataFrame()

    df = equity_df[cols].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["idle"] = df["bankroll"] - df["open_exposure"]
    df["exposure_pct"] = df["open_exposure"] / df["bankroll"] * 100
    return df


def compute_risk_summary(positions_df: pd.DataFrame,
                          equity_df: pd.DataFrame) -> dict:
    """Compute all risk metrics in one call.

    Returns a dict of all key risk statistics.
    """
    result = {}

    # Trade-level returns
    closed = positions_df[positions_df["pnl"].notna()].copy() if not positions_df.empty else pd.DataFrame()
    if not closed.empty:
        returns = closed["pnl"].values
        costs = (closed["size"] * closed["entry_price"] / 100).values
        pct_returns = returns / costs  # Return on cost basis

        result["sharpe"] = sharpe_ratio(pct_returns)
        result["sortino"] = sortino_ratio(pct_returns)
        result["var_95"] = value_at_risk(returns, 0.95)
        result["var_99"] = value_at_risk(returns, 0.99)
        result["es_95"] = expected_shortfall(returns, 0.95)
        result["win_rate"] = float((returns > 0).mean() * 100)
        result["n_trades"] = len(closed)
        result["total_pnl"] = float(returns.sum())
        result["avg_pnl"] = float(returns.mean())

    # Drawdown from equity curve
    if not equity_df.empty:
        dd = max_drawdown_analysis(equity_df)
        result["max_drawdown"] = dd.get("max_drawdown", 0)
        result["max_drawdown_pct"] = dd.get("max_drawdown_pct", 0)
        result["drawdown_recovered"] = dd.get("recovered", False)

    # Kelly
    kelly = kelly_analysis(positions_df)
    if kelly:
        result["kelly_fraction"] = kelly["kelly_fraction"]
        result["three_quarter_kelly"] = kelly["three_quarter_kelly"]

    return result
