"""
P&L analysis module.

Computes equity curves, drawdowns, per-trade returns, and bucketed
performance statistics from paper trading data.
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_equity_curve(equity_df: pd.DataFrame) -> pd.DataFrame:
    """Add drawdown columns to an equity curve DataFrame.

    Expects columns: timestamp, bankroll.
    Returns DataFrame with added: peak, drawdown, drawdown_pct.
    """
    if equity_df.empty:
        return equity_df

    df = equity_df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["peak"] = df["bankroll"].cummax()
    df["drawdown"] = df["peak"] - df["bankroll"]
    df["drawdown_pct"] = df["drawdown"] / df["peak"] * 100
    return df


def compute_trade_returns(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trade returns from closed positions.

    Expects columns: entry_price, exit_price, size, side, pnl.
    Returns DataFrame of closed positions with added return columns.
    """
    if positions_df.empty:
        return positions_df

    closed = positions_df[positions_df["exit_time"].notna()].copy()
    if closed.empty:
        return closed

    # Cost basis
    closed["cost"] = closed["size"] * closed["entry_price"] / 100
    # Return on cost
    closed["return_pct"] = closed["pnl"] / closed["cost"] * 100
    # Win/loss flag
    closed["is_win"] = closed["pnl"] > 0

    return closed.sort_values("exit_time").reset_index(drop=True)


def pnl_by_bucket(closed_positions: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    """Aggregate P&L statistics grouped by a categorical column.

    Args:
        closed_positions: DataFrame from compute_trade_returns
        bucket_col: column name to group by (e.g. 'asset', 'side')

    Returns:
        DataFrame with: bucket, n_trades, total_pnl, avg_pnl, win_rate, avg_edge
    """
    if closed_positions.empty or bucket_col not in closed_positions.columns:
        return pd.DataFrame()

    groups = closed_positions.groupby(bucket_col)
    rows = []
    for name, g in groups:
        rows.append({
            "bucket": name,
            "n_trades": len(g),
            "total_pnl": g["pnl"].sum(),
            "avg_pnl": g["pnl"].mean(),
            "win_rate": g["is_win"].mean() * 100,
            "avg_edge": g["edge"].mean() * 100 if "edge" in g.columns else None,
            "avg_return_pct": g["return_pct"].mean() if "return_pct" in g.columns else None,
        })
    return pd.DataFrame(rows)


def pnl_by_sigma_bucket(closed_positions: pd.DataFrame,
                         bins: Optional[list] = None) -> pd.DataFrame:
    """Aggregate P&L by sigma distance buckets."""
    if closed_positions.empty or "sigma_distance" not in closed_positions.columns:
        # Try model_prob as proxy if sigma_distance not available
        return pd.DataFrame()

    df = closed_positions.copy()
    if bins is None:
        bins = [0, 1.5, 2.0, 2.5, 3.0, float("inf")]

    # Use absolute sigma distance
    if "sigma_distance" in df.columns:
        df["abs_sigma"] = df["sigma_distance"].abs()
        df["sigma_bucket"] = pd.cut(df["abs_sigma"], bins=bins, right=False)
        return pnl_by_bucket(df, "sigma_bucket")
    return pd.DataFrame()


def pnl_by_hour(closed_positions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate P&L by hour of day (UTC) of entry."""
    if closed_positions.empty or "entry_time" not in closed_positions.columns:
        return pd.DataFrame()

    df = closed_positions.copy()
    df["hour"] = df["entry_time"].dt.hour
    return pnl_by_bucket(df, "hour")


def cumulative_pnl_series(closed_positions: pd.DataFrame) -> pd.DataFrame:
    """Build a cumulative P&L time series from closed positions.

    Returns DataFrame with: exit_time, pnl, cumulative_pnl.
    """
    if closed_positions.empty:
        return pd.DataFrame()

    df = closed_positions.sort_values("exit_time").copy()
    df["cumulative_pnl"] = df["pnl"].cumsum()
    return df[["exit_time", "pnl", "cumulative_pnl"]].reset_index(drop=True)


def expected_vs_realized_edge(closed_positions: pd.DataFrame) -> dict:
    """Compare expected edge (from model) vs realized P&L.

    Returns dict with summary statistics.
    """
    if closed_positions.empty:
        return {}

    df = closed_positions.copy()
    total_cost = (df["size"] * df["entry_price"] / 100).sum()
    total_pnl = df["pnl"].sum()
    realized_edge = total_pnl / total_cost if total_cost > 0 else 0

    avg_expected_edge = df["edge"].mean() if "edge" in df.columns else None

    return {
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "realized_edge_pct": realized_edge * 100,
        "expected_edge_pct": avg_expected_edge * 100 if avg_expected_edge is not None else None,
        "n_trades": len(df),
        "win_rate": (df["pnl"] > 0).mean() * 100,
    }
