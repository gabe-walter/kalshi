"""
Backtesting framework for Kalshi crypto above/below arbitrage strategy.

This module compares option-implied probabilities to Kalshi prices
and simulates trading performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    ticker: str
    side: str  # 'buy_yes' or 'buy_no'
    price: float  # Price paid (0-1)
    size: int  # Number of contracts
    option_implied_prob: float  # What options market implies
    kalshi_prob: float  # Kalshi price
    edge: float  # Expected edge
    expiry: datetime
    strike: float
    settled_price: Optional[float] = None  # 1 if yes, 0 if no
    pnl: Optional[float] = None

    def settle(self, outcome: bool):
        """Settle the trade given the outcome."""
        self.settled_price = 1.0 if outcome else 0.0

        if self.side == "buy_yes":
            self.pnl = (self.settled_price - self.price) * self.size
        else:  # buy_no
            self.pnl = ((1 - self.settled_price) - self.price) * self.size


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    min_edge: float = 0.05  # Minimum edge to trade (5%)
    max_position_per_contract: int = 100  # Max contracts per market
    kalshi_fee_rate: float = 0.0  # Kalshi fee (currently 0)
    slippage: float = 0.01  # Assumed slippage (1 cent)
    capital: float = 10000  # Starting capital


@dataclass
class BacktestResult:
    """Results from a backtest."""
    trades: List[Trade]
    config: BacktestConfig
    start_time: datetime
    end_time: datetime

    # Computed metrics
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    avg_edge: float = 0.0
    realized_edge: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    def compute_metrics(self):
        """Compute performance metrics from trades."""
        if not self.trades:
            return

        self.total_trades = len(self.trades)

        settled_trades = [t for t in self.trades if t.pnl is not None]
        if not settled_trades:
            return

        self.winning_trades = sum(1 for t in settled_trades if t.pnl > 0)
        self.total_pnl = sum(t.pnl for t in settled_trades)
        self.avg_edge = np.mean([t.edge for t in self.trades])

        # Realized edge vs expected
        total_bet = sum(t.price * t.size for t in settled_trades)
        if total_bet > 0:
            self.realized_edge = self.total_pnl / total_bet

        # PnL series for Sharpe and drawdown
        pnls = [t.pnl for t in settled_trades]
        if len(pnls) > 1:
            pnl_std = np.std(pnls)
            if pnl_std > 0:
                self.sharpe_ratio = np.mean(pnls) / pnl_std * np.sqrt(len(pnls))

            # Max drawdown
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            self.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    def summary(self) -> str:
        """Generate summary string."""
        self.compute_metrics()

        win_rate = self.winning_trades / max(1, self.total_trades) * 100

        return f"""
Backtest Results
================
Period: {self.start_time.strftime('%Y-%m-%d')} to {self.end_time.strftime('%Y-%m-%d')}
Total Trades: {self.total_trades}
Win Rate: {win_rate:.1f}%

PnL Summary:
  Total PnL: ${self.total_pnl:.2f}
  Avg Edge (expected): {self.avg_edge:.2%}
  Realized Edge: {self.realized_edge:.2%}

Risk Metrics:
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Max Drawdown: ${self.max_drawdown:.2f}
"""


class Backtester:
    """
    Backtester for Kalshi crypto above/below strategy.

    Compares option-implied probabilities to Kalshi prices and
    simulates trading when sufficient edge exists.
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []

    def find_opportunities(
        self,
        kalshi_price: float,
        option_prob: float,
        option_prob_lower: float = None,
        option_prob_upper: float = None
    ) -> Optional[Tuple[str, float]]:
        """
        Determine if there's a trading opportunity.

        Args:
            kalshi_price: Kalshi yes price (0-1)
            option_prob: Option-implied probability of yes
            option_prob_lower: Lower bound of probability estimate
            option_prob_upper: Upper bound of probability estimate

        Returns:
            Tuple of (side, edge) if opportunity exists, None otherwise
        """
        # Use conservative estimate if bounds available
        if option_prob_lower is not None and option_prob_upper is not None:
            # Use the bound that's least favorable to the trade
            pass  # For now, use point estimate

        # Edge on buying yes: option_prob - kalshi_price
        # Edge on buying no: (1 - option_prob) - (1 - kalshi_price) = kalshi_price - option_prob
        yes_edge = option_prob - kalshi_price - self.config.slippage
        no_edge = kalshi_price - option_prob - self.config.slippage

        if yes_edge >= self.config.min_edge:
            return ("buy_yes", yes_edge)
        elif no_edge >= self.config.min_edge:
            return ("buy_no", no_edge)

        return None

    def simulate_trade(
        self,
        timestamp: datetime,
        ticker: str,
        kalshi_price: float,
        option_prob: float,
        expiry: datetime,
        strike: float,
        kalshi_bid: float = None,
        kalshi_ask: float = None
    ) -> Optional[Trade]:
        """
        Simulate a trade if opportunity exists.

        Args:
            timestamp: Time of the trade
            ticker: Kalshi market ticker
            kalshi_price: Mid price (yes side)
            option_prob: Option-implied probability
            expiry: Contract expiration
            strike: Strike price
            kalshi_bid: Best bid (yes side)
            kalshi_ask: Best ask (yes side)

        Returns:
            Trade object if opportunity found, None otherwise
        """
        opportunity = self.find_opportunities(kalshi_price, option_prob)

        if opportunity is None:
            return None

        side, edge = opportunity

        # Determine execution price
        if side == "buy_yes":
            exec_price = kalshi_ask if kalshi_ask else kalshi_price + self.config.slippage
        else:
            # Buying no is equivalent to selling yes
            # No price = 1 - yes_bid
            yes_bid = kalshi_bid if kalshi_bid else kalshi_price - self.config.slippage
            exec_price = 1 - yes_bid

        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            side=side,
            price=exec_price,
            size=self.config.max_position_per_contract,
            option_implied_prob=option_prob,
            kalshi_prob=kalshi_price,
            edge=edge,
            expiry=expiry,
            strike=strike
        )

        self.trades.append(trade)
        return trade

    def settle_trades(self, settlement_data: Dict[str, bool]):
        """
        Settle trades based on outcomes.

        Args:
            settlement_data: Dict mapping ticker to outcome (True if price > strike)
        """
        for trade in self.trades:
            if trade.pnl is None and trade.ticker in settlement_data:
                outcome = settlement_data[trade.ticker]
                trade.settle(outcome)

    def run_backtest(
        self,
        data: pd.DataFrame,
        settlement_data: Dict[str, bool] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with columns:
                - timestamp, ticker, kalshi_price, option_prob, expiry, strike
                - optional: kalshi_bid, kalshi_ask, option_prob_lower, option_prob_upper
            settlement_data: Dict mapping ticker to outcome

        Returns:
            BacktestResult with all trades and metrics
        """
        self.trades = []

        for _, row in data.iterrows():
            self.simulate_trade(
                timestamp=row["timestamp"],
                ticker=row["ticker"],
                kalshi_price=row["kalshi_price"],
                option_prob=row["option_prob"],
                expiry=row["expiry"],
                strike=row["strike"],
                kalshi_bid=row.get("kalshi_bid"),
                kalshi_ask=row.get("kalshi_ask")
            )

        if settlement_data:
            self.settle_trades(settlement_data)

        result = BacktestResult(
            trades=self.trades,
            config=self.config,
            start_time=data["timestamp"].min(),
            end_time=data["timestamp"].max()
        )
        result.compute_metrics()

        return result

    def plot_results(self, result: BacktestResult, save_path: str = None):
        """Plot backtest results."""
        if not result.trades:
            print("No trades to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Cumulative PnL
        settled_trades = [t for t in result.trades if t.pnl is not None]
        if settled_trades:
            timestamps = [t.timestamp for t in settled_trades]
            pnls = [t.pnl for t in settled_trades]
            cumulative_pnl = np.cumsum(pnls)

            axes[0, 0].plot(timestamps, cumulative_pnl, "b-", linewidth=1.5)
            axes[0, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            axes[0, 0].set_title("Cumulative PnL")
            axes[0, 0].set_xlabel("Date")
            axes[0, 0].set_ylabel("PnL ($)")
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Edge distribution
        edges = [t.edge for t in result.trades]
        axes[0, 1].hist(edges, bins=30, edgecolor="black", alpha=0.7)
        axes[0, 1].axvline(x=result.avg_edge, color="red", linestyle="--", label=f"Mean: {result.avg_edge:.2%}")
        axes[0, 1].set_title("Edge Distribution")
        axes[0, 1].set_xlabel("Edge")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Kalshi vs Option-implied probability
        kalshi_probs = [t.kalshi_prob for t in result.trades]
        option_probs = [t.option_implied_prob for t in result.trades]

        axes[1, 0].scatter(kalshi_probs, option_probs, alpha=0.5, s=20)
        axes[1, 0].plot([0, 1], [0, 1], "r--", label="No edge line")
        axes[1, 0].set_title("Kalshi vs Option-Implied Probability")
        axes[1, 0].set_xlabel("Kalshi Price")
        axes[1, 0].set_ylabel("Option-Implied Probability")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Trade outcomes by edge
        if settled_trades:
            trade_edges = [t.edge for t in settled_trades]
            trade_pnls = [t.pnl for t in settled_trades]
            colors = ["green" if pnl > 0 else "red" for pnl in trade_pnls]

            axes[1, 1].scatter(trade_edges, trade_pnls, c=colors, alpha=0.5, s=20)
            axes[1, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            axes[1, 1].set_title("Trade PnL vs Edge")
            axes[1, 1].set_xlabel("Expected Edge")
            axes[1, 1].set_ylabel("Realized PnL ($)")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save_path}")

        plt.show()


def generate_synthetic_backtest_data(
    num_observations: int = 500,
    base_price: float = 100000,
    volatility: float = 0.02,
    kalshi_noise: float = 0.03,
    edge_present: bool = True
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """
    Generate synthetic data for testing the backtester.

    Args:
        num_observations: Number of data points
        base_price: Starting price
        volatility: Daily volatility
        kalshi_noise: Noise in Kalshi prices relative to true prob
        edge_present: If True, adds systematic edge

    Returns:
        Tuple of (data DataFrame, settlement dict)
    """
    np.random.seed(42)

    timestamps = pd.date_range(
        start="2024-01-01",
        periods=num_observations,
        freq="1H"
    )

    # Simulate price path
    returns = np.random.normal(0, volatility / np.sqrt(24), num_observations)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate strikes around current price
    strikes = np.round(prices * (1 + np.random.uniform(-0.1, 0.1, num_observations)), -3)

    # True probabilities (simplified)
    moneyness = strikes / prices
    true_probs = 1 - norm_cdf_approx(moneyness - 1, 0, 0.05)

    # Option-implied probabilities (close to true with some noise)
    option_probs = true_probs + np.random.normal(0, 0.02, num_observations)
    option_probs = np.clip(option_probs, 0.01, 0.99)

    # Kalshi prices (with noise and potential systematic bias)
    if edge_present:
        # Kalshi prices slightly biased away from true probability
        kalshi_prices = true_probs + np.random.normal(0.03, kalshi_noise, num_observations)
    else:
        kalshi_prices = true_probs + np.random.normal(0, kalshi_noise, num_observations)

    kalshi_prices = np.clip(kalshi_prices, 0.01, 0.99)

    # Generate tickers
    tickers = [f"KXBTC-{ts.strftime('%Y%m%d')}-{int(s)}" for ts, s in zip(timestamps, strikes)]

    # Expiry (4 hours after observation for simplicity)
    expiries = timestamps + timedelta(hours=4)

    data = pd.DataFrame({
        "timestamp": timestamps,
        "ticker": tickers,
        "kalshi_price": kalshi_prices,
        "option_prob": option_probs,
        "expiry": expiries,
        "strike": strikes,
        "underlying_price": prices,
        "true_prob": true_probs
    })

    # Settlement data
    final_prices = prices * np.exp(np.random.normal(0, volatility, num_observations))
    settlement = {
        ticker: final_price > strike
        for ticker, final_price, strike in zip(tickers, final_prices, strikes)
    }

    return data, settlement


def norm_cdf_approx(x, mean=0, std=1):
    """Approximate normal CDF."""
    z = (x - mean) / std
    return 0.5 * (1 + np.tanh(z * 0.7978845608))


if __name__ == "__main__":
    print("Running synthetic backtest...")

    # Generate synthetic data
    data, settlements = generate_synthetic_backtest_data(
        num_observations=500,
        edge_present=True
    )

    print(f"Generated {len(data)} observations")
    print(f"\nSample data:")
    print(data[["timestamp", "ticker", "kalshi_price", "option_prob", "strike"]].head())

    # Run backtest
    config = BacktestConfig(
        min_edge=0.03,
        max_position_per_contract=10,
        slippage=0.01
    )

    backtester = Backtester(config)
    result = backtester.run_backtest(data, settlements)

    print(result.summary())

    # Plot results
    backtester.plot_results(result, save_path="data/backtest_results.png")
