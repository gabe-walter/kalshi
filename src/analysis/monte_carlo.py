"""
Monte Carlo simulation module.

Resamples historical trades to simulate thousands of equity paths,
producing confidence intervals, ruin probabilities, and drawdown
distributions for robustness testing.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results."""
    n_paths: int
    n_trades_per_path: int
    initial_bankroll: float

    # Equity paths: shape (n_paths, n_trades_per_path+1)
    equity_paths: np.ndarray

    # Derived statistics
    final_bankrolls: np.ndarray
    max_drawdowns: np.ndarray
    max_drawdown_pcts: np.ndarray

    def percentile_paths(self, percentiles: list[float] = None) -> dict[float, np.ndarray]:
        """Get percentile equity paths for fan chart."""
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        return {p: np.percentile(self.equity_paths, p, axis=0) for p in percentiles}

    def final_bankroll_stats(self) -> dict:
        return {
            "mean": float(np.mean(self.final_bankrolls)),
            "median": float(np.median(self.final_bankrolls)),
            "std": float(np.std(self.final_bankrolls)),
            "p5": float(np.percentile(self.final_bankrolls, 5)),
            "p25": float(np.percentile(self.final_bankrolls, 25)),
            "p75": float(np.percentile(self.final_bankrolls, 75)),
            "p95": float(np.percentile(self.final_bankrolls, 95)),
            "min": float(np.min(self.final_bankrolls)),
            "max": float(np.max(self.final_bankrolls)),
        }

    def ruin_probability(self, threshold_pct: float = 50.0) -> float:
        """P(bankroll drops below threshold % of initial at any point)."""
        threshold = self.initial_bankroll * (1 - threshold_pct / 100)
        min_per_path = np.min(self.equity_paths, axis=1)
        return float(np.mean(min_per_path < threshold))

    def drawdown_stats(self) -> dict:
        return {
            "mean": float(np.mean(self.max_drawdown_pcts)),
            "median": float(np.median(self.max_drawdown_pcts)),
            "p75": float(np.percentile(self.max_drawdown_pcts, 75)),
            "p90": float(np.percentile(self.max_drawdown_pcts, 90)),
            "p95": float(np.percentile(self.max_drawdown_pcts, 95)),
            "p99": float(np.percentile(self.max_drawdown_pcts, 99)),
            "max": float(np.max(self.max_drawdown_pcts)),
        }


def simulate(positions_df: pd.DataFrame,
             n_paths: int = 10000,
             n_trades: Optional[int] = None,
             initial_bankroll: float = 10000.0,
             edge_multiplier: float = 1.0,
             seed: Optional[int] = None) -> MonteCarloResult:
    """Run Monte Carlo simulation by resampling historical trades.

    Each path draws n_trades with replacement from the historical trade
    P&L distribution. Optionally scale the edge to stress-test.

    Args:
        positions_df: DataFrame of closed positions with 'pnl' column.
        n_paths: Number of simulation paths.
        n_trades: Trades per path. Defaults to len(closed_positions).
        initial_bankroll: Starting bankroll.
        edge_multiplier: Scale factor on P&L (1.0=actual, 0.5=half edge).
        seed: Random seed for reproducibility.
    """
    closed = positions_df[positions_df["pnl"].notna()].copy()
    if closed.empty:
        raise ValueError("No closed positions with P&L to simulate from")

    pnls = closed["pnl"].values * edge_multiplier

    if n_trades is None:
        n_trades = len(pnls)

    rng = np.random.default_rng(seed)

    # Resample: shape (n_paths, n_trades)
    sampled_pnls = rng.choice(pnls, size=(n_paths, n_trades), replace=True)

    # Build equity paths: shape (n_paths, n_trades+1)
    equity_paths = np.zeros((n_paths, n_trades + 1))
    equity_paths[:, 0] = initial_bankroll
    equity_paths[:, 1:] = initial_bankroll + np.cumsum(sampled_pnls, axis=1)

    # Final bankrolls
    final_bankrolls = equity_paths[:, -1]

    # Max drawdown per path
    max_drawdowns = np.zeros(n_paths)
    max_drawdown_pcts = np.zeros(n_paths)
    for i in range(n_paths):
        path = equity_paths[i]
        peak = np.maximum.accumulate(path)
        dd = peak - path
        dd_pct = dd / peak * 100
        max_drawdowns[i] = np.max(dd)
        max_drawdown_pcts[i] = np.max(dd_pct)

    return MonteCarloResult(
        n_paths=n_paths,
        n_trades_per_path=n_trades,
        initial_bankroll=initial_bankroll,
        equity_paths=equity_paths,
        final_bankrolls=final_bankrolls,
        max_drawdowns=max_drawdowns,
        max_drawdown_pcts=max_drawdown_pcts,
    )


def sensitivity_analysis(positions_df: pd.DataFrame,
                          edge_multipliers: Optional[list[float]] = None,
                          n_paths: int = 5000,
                          n_trades: Optional[int] = None,
                          initial_bankroll: float = 10000.0,
                          seed: int = 42) -> pd.DataFrame:
    """Run Monte Carlo at different edge assumptions.

    Tests: "What if my edge is only 50% of what I think?"

    Returns DataFrame with rows per multiplier and summary statistics.
    """
    if edge_multipliers is None:
        edge_multipliers = [0.25, 0.5, 0.75, 1.0, 1.25]

    rows = []
    for mult in edge_multipliers:
        try:
            result = simulate(
                positions_df, n_paths=n_paths, n_trades=n_trades,
                initial_bankroll=initial_bankroll,
                edge_multiplier=mult, seed=seed,
            )
            stats = result.final_bankroll_stats()
            dd_stats = result.drawdown_stats()
            rows.append({
                "edge_multiplier": mult,
                "edge_label": f"{mult*100:.0f}%",
                "mean_final": stats["mean"],
                "median_final": stats["median"],
                "p5_final": stats["p5"],
                "p95_final": stats["p95"],
                "ruin_prob_50pct": result.ruin_probability(50),
                "ruin_prob_25pct": result.ruin_probability(25),
                "median_max_dd_pct": dd_stats["median"],
                "p95_max_dd_pct": dd_stats["p95"],
            })
        except ValueError:
            continue

    return pd.DataFrame(rows)


def fan_chart_data(result: MonteCarloResult,
                    percentiles: Optional[list[float]] = None) -> pd.DataFrame:
    """Convert MonteCarloResult into a long-format DataFrame for plotting.

    Returns DataFrame with: trade_number, percentile, bankroll.
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    paths = result.percentile_paths(percentiles)
    n_steps = result.n_trades_per_path + 1
    trade_numbers = np.arange(n_steps)

    rows = []
    for pct, values in paths.items():
        for i, val in enumerate(values):
            rows.append({
                "trade_number": int(trade_numbers[i]),
                "percentile": f"p{int(pct)}",
                "bankroll": float(val),
            })

    return pd.DataFrame(rows)
