#!/usr/bin/env python3
"""
Paper Trader V3 for Kalshi KXBTCD Strategy

Inspired by @Argona0x's Polymarket bot:
- IV Surface: Interpolated from Deribit options, converted to probabilities via Black-Scholes
- Sigma filter: Only trade strikes beyond ±1.75σ from spot (tail-risk focus)
- 5¢ mispricing threshold: Simple cents-based edge instead of percentage
- Faster polling: 30 second intervals instead of 5 minutes

Position sizing: Three-quarter Kelly (3x optimal from Monte Carlo)
Exit: Hold to expiration
"""

import sys
sys.path.insert(0, "src")

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import pandas as pd

from data.deribit_client import DeribitClient
from data.kalshi_client import KalshiClient
from models.iv_surface_2d import build_iv_surface_2d, kalshi_settlement_in_grid


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AssetConfig:
    """Configuration for a single asset."""
    symbol: str                        # BTC or ETH
    kalshi_series: str                 # KXBTCD or KXETHD


# Supported assets
ASSETS = {
    "BTC": AssetConfig(symbol="BTC", kalshi_series="KXBTCD"),
    "ETH": AssetConfig(symbol="ETH", kalshi_series="KXETHD"),
}


@dataclass
class Config:
    """Paper trading configuration."""
    initial_bankroll: float = 10000.0  # Starting capital in dollars
    kelly_fraction: float = 0.75       # Three-quarter Kelly (3x from 0.25)
    min_mispricing_cents: int = 5      # 5¢ minimum mispricing to trade
    max_per_strike: float = 0.15       # 15% max per single strike (3x from 0.05)
    max_per_settlement: float = 0.50   # 50% max per settlement time (3x from 0.20, capped)
    max_total_open: float = 0.80       # 80% max total exposure (increased from 0.60)
    min_bid_cents: int = 5             # Minimum bid for liquidity
    max_spread_cents: int = 20         # Maximum spread to trade
    data_dir: Path = Path("data/paper_trading_v3")

    # V3 Filters
    min_sigma: float = 1.75            # Only trade strikes beyond ±1.75σ from spot

    # Assets to trade
    assets: tuple = ("BTC", "ETH")     # Which assets to scan


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Position:
    """Represents an open or closed position."""
    id: str
    ticker: str
    asset: str  # BTC or ETH
    strike: float
    settlement_time: datetime
    side: str  # "YES" or "NO"
    entry_price: float  # Price paid (0-100 cents)
    size: int  # Number of contracts
    entry_time: datetime
    model_prob: float  # Model probability at entry
    edge: float  # Edge at entry
    deribit_expiry: str  # Which Deribit expiry was used

    # Set when position closes
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "settlement", "model_flip", "manual"
    settlement_value: Optional[float] = None  # 100 if won, 0 if lost
    pnl: Optional[float] = None  # Profit/loss in dollars

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def cost_basis(self) -> float:
        """Total cost of position in dollars."""
        return self.size * self.entry_price / 100

    def close(self, exit_price: float, exit_time: datetime, reason: str,
              settlement_value: Optional[float] = None):
        """Close the position and calculate P&L."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.settlement_value = settlement_value

        if reason == "settlement" and settlement_value is not None:
            # Settlement: we get settlement_value per contract
            self.pnl = self.size * (settlement_value - self.entry_price) / 100
        else:
            # Early exit: we sell at exit_price
            self.pnl = self.size * (exit_price - self.entry_price) / 100


@dataclass
class TradeLog:
    """Log entry for a trade action."""
    timestamp: datetime
    action: str  # "OPEN", "CLOSE", "SKIP"
    ticker: str
    strike: float
    side: str
    price: float
    size: int
    model_prob: float
    edge: float
    reason: str
    bankroll_after: float


@dataclass
class EquityPoint:
    """Point on the equity curve."""
    timestamp: datetime
    bankroll: float
    open_exposure: float
    open_positions: int
    total_pnl: float
    drawdown: float
    drawdown_pct: float


# =============================================================================
# Paper Trader V2
# =============================================================================

class PaperTraderV3:
    """
    Paper trading system V3 for Kalshi KXBTCD strategy.

    Inspired by @Argona0x's Polymarket bot:
    - Sigma filter: Only trade strikes beyond 2.1σ from spot
    - 5¢ mispricing threshold instead of percentage edge
    - 30 second polling instead of 5 minutes
    - Time ratio filter for variance scaling integrity
    """

    VERSION = "3.0"

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        self.bankroll = self.config.initial_bankroll
        self.positions: Dict[str, Position] = {}  # id -> Position
        self.trade_log: List[TradeLog] = []
        self.equity_curve: List[EquityPoint] = []
        self.position_counter = 0
        self.peak_bankroll = self.config.initial_bankroll

        # Track filtered trades for analysis
        self.filtered_trades = {
            "not_tail": 0,
            "outside_grid": 0,
            "insufficient_mispricing": 0
        }

        # API clients
        self.deribit = DeribitClient()
        self.kalshi = KalshiClient()

        # Load existing state if available
        self._load_state()

    # =========================================================================
    # State Persistence
    # =========================================================================

    def _state_file(self) -> Path:
        return self.config.data_dir / "state.json"

    def _log_file(self) -> Path:
        return self.config.data_dir / "trade_log.json"

    def _equity_file(self) -> Path:
        return self.config.data_dir / "equity_curve.json"

    def _snapshots_dir(self) -> Path:
        return self.config.data_dir / "snapshots"

    def _save_market_snapshot(self, kalshi_markets: List[dict], signals: Dict[str, dict],
                               spots: Dict[str, float], deribit_chains: Dict[str, pd.DataFrame]):
        """Save full market snapshot for later analysis (multi-asset)."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")

        snapshot_dir = self._snapshots_dir() / date_str
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save Kalshi markets with model signals
        kalshi_data = []
        for m in kalshi_markets:
            ticker = m.get('ticker')
            asset = m.get('_asset', 'BTC')
            signal = signals.get(ticker, {})

            kalshi_data.append({
                'snapshot_time': now.isoformat(),
                'asset': asset,
                'ticker': ticker,
                'event_ticker': m.get('event_ticker'),
                'strike': m.get('floor_strike'),
                'close_time': m.get('close_time'),
                'yes_bid': m.get('yes_bid'),
                'yes_ask': m.get('yes_ask'),
                'no_bid': m.get('no_bid'),
                'no_ask': m.get('no_ask'),
                'volume': m.get('volume'),
                'volume_24h': m.get('volume_24h'),
                'open_interest': m.get('open_interest'),
                'model_prob': signal.get('model_prob'),
                'sigma_distance': signal.get('sigma'),
                'mispricing_cents': signal.get('mispricing'),
                'signal_side': signal.get('side'),
                'spot_price': spots.get(asset),
                'orderbook_depth': signal.get('orderbook_depth'),
            })

        kalshi_df = pd.DataFrame(kalshi_data)
        kalshi_df.to_parquet(snapshot_dir / f"kalshi_{date_str}_{time_str}.parquet", index=False)

        # Save Deribit chains (combined)
        deribit_dfs = []
        for asset, chain in deribit_chains.items():
            if not chain.empty:
                chain = chain.copy()
                chain['snapshot_time'] = now.isoformat()
                chain['spot_price'] = spots.get(asset)
                deribit_dfs.append(chain)

        if deribit_dfs:
            combined_deribit = pd.concat(deribit_dfs, ignore_index=True)
            combined_deribit.to_parquet(snapshot_dir / f"deribit_{date_str}_{time_str}.parquet", index=False)

    def _save_state(self):
        """Save current state to disk."""
        state = {
            "version": self.VERSION,
            "bankroll": self.bankroll,
            "position_counter": self.position_counter,
            "peak_bankroll": self.peak_bankroll,
            "filtered_trades": self.filtered_trades,
            "positions": {
                pid: {
                    **asdict(pos),
                    "settlement_time": pos.settlement_time.isoformat(),
                    "entry_time": pos.entry_time.isoformat(),
                    "exit_time": pos.exit_time.isoformat() if pos.exit_time else None,
                }
                for pid, pos in self.positions.items()
            }
        }
        with open(self._state_file(), "w") as f:
            json.dump(state, f, indent=2)

        # Also save trade log
        log_data = [
            {
                **asdict(entry),
                "timestamp": entry.timestamp.isoformat()
            }
            for entry in self.trade_log
        ]
        with open(self._log_file(), "w") as f:
            json.dump(log_data, f, indent=2)

        # Save equity curve
        equity_data = [
            {
                **asdict(point),
                "timestamp": point.timestamp.isoformat()
            }
            for point in self.equity_curve
        ]
        with open(self._equity_file(), "w") as f:
            json.dump(equity_data, f, indent=2)

    def _load_state(self):
        """Load state from disk if exists."""
        if self._state_file().exists():
            with open(self._state_file()) as f:
                state = json.load(f)

            self.bankroll = state["bankroll"]
            self.position_counter = state["position_counter"]
            self.peak_bankroll = state.get("peak_bankroll", self.bankroll)
            self.filtered_trades = state.get("filtered_trades", self.filtered_trades)

            for pid, pos_data in state["positions"].items():
                pos_data["settlement_time"] = datetime.fromisoformat(pos_data["settlement_time"])
                pos_data["entry_time"] = datetime.fromisoformat(pos_data["entry_time"])
                if pos_data["exit_time"]:
                    pos_data["exit_time"] = datetime.fromisoformat(pos_data["exit_time"])
                # Backward compatibility: infer asset from ticker if missing
                if "asset" not in pos_data:
                    ticker = pos_data.get("ticker", "")
                    if ticker.startswith("KXBTCD"):
                        pos_data["asset"] = "BTC"
                    elif ticker.startswith("KXETHD"):
                        pos_data["asset"] = "ETH"
                    else:
                        pos_data["asset"] = "BTC"  # Default to BTC
                self.positions[pid] = Position(**pos_data)

        if self._log_file().exists():
            with open(self._log_file()) as f:
                log_data = json.load(f)
            self.trade_log = [
                TradeLog(
                    timestamp=datetime.fromisoformat(entry["timestamp"]),
                    **{k: v for k, v in entry.items() if k != "timestamp"}
                )
                for entry in log_data
            ]

        if self._equity_file().exists():
            with open(self._equity_file()) as f:
                equity_data = json.load(f)
            self.equity_curve = [
                EquityPoint(
                    timestamp=datetime.fromisoformat(point["timestamp"]),
                    **{k: v for k, v in point.items() if k != "timestamp"}
                )
                for point in equity_data
            ]

    # =========================================================================
    # Risk Management
    # =========================================================================

    def _get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions.values() if p.is_open]

    def _total_open_exposure(self) -> float:
        """Total dollar exposure in open positions."""
        return sum(p.cost_basis for p in self._get_open_positions())

    def _exposure_by_settlement(self, settlement_time: datetime) -> float:
        """Dollar exposure for a specific settlement time."""
        return sum(
            p.cost_basis for p in self._get_open_positions()
            if p.settlement_time == settlement_time
        )

    def _exposure_by_strike(self, strike: float, settlement_time: datetime) -> float:
        """Dollar exposure for a specific strike and settlement."""
        return sum(
            p.cost_basis for p in self._get_open_positions()
            if p.strike == strike and p.settlement_time == settlement_time
        )

    def _get_positions_for_settlement(self, settlement_time: datetime) -> List[Position]:
        """Get all open positions for a specific settlement."""
        return [p for p in self._get_open_positions() if p.settlement_time == settlement_time]

    def _would_create_conflict(self, side: str, strike: float, settlement_time: datetime) -> bool:
        """
        Check if adding this position would create a conflicting setup.

        Conflict: YES at higher strike than NO creates a "death zone" where
        if BTC lands between the strikes, BOTH positions lose.

        Good setup: YES at lower strike, NO at higher strike creates
        overlapping win zone.
        """
        existing = self._get_positions_for_settlement(settlement_time)

        for pos in existing:
            if side == "YES" and pos.side == "NO":
                # Adding YES: check if it would be above any NO
                if strike > pos.strike:
                    return True  # YES above NO = bad
            elif side == "NO" and pos.side == "YES":
                # Adding NO: check if it would be below any YES
                if strike < pos.strike:
                    return True  # NO below YES = bad (equivalent to YES above NO)

        return False

    def _calculate_position_size(
        self,
        model_prob: float,
        ask_price: float,  # in cents (0-100)
        settlement_time: datetime,
        strike: float,
        available_volume: int
    ) -> int:
        """
        Calculate position size using Kelly criterion with risk limits.

        Returns number of contracts to buy (0 if shouldn't trade).
        """
        # Kelly fraction for binary option
        # f* = (model_prob - ask/100) / (1 - ask/100)
        ask_prob = ask_price / 100
        if ask_prob >= 1:
            return 0

        kelly = (model_prob - ask_prob) / (1 - ask_prob)
        if kelly <= 0:
            return 0

        # Apply Kelly fraction
        kelly_adjusted = kelly * self.config.kelly_fraction

        # Calculate dollar amount
        kelly_dollars = self.bankroll * kelly_adjusted

        # Apply risk limits
        # 1. Max per strike
        current_strike_exposure = self._exposure_by_strike(strike, settlement_time)
        max_strike_dollars = self.bankroll * self.config.max_per_strike
        available_strike = max(0, max_strike_dollars - current_strike_exposure)

        # 2. Max per settlement
        current_settlement_exposure = self._exposure_by_settlement(settlement_time)
        max_settlement_dollars = self.bankroll * self.config.max_per_settlement
        available_settlement = max(0, max_settlement_dollars - current_settlement_exposure)

        # 3. Max total open
        current_total_exposure = self._total_open_exposure()
        max_total_dollars = self.bankroll * self.config.max_total_open
        available_total = max(0, max_total_dollars - current_total_exposure)

        # Take minimum of all constraints
        max_dollars = min(kelly_dollars, available_strike, available_settlement, available_total)

        if max_dollars <= 0:
            return 0

        # Convert to contracts
        cost_per_contract = ask_price / 100  # dollars per contract
        max_contracts = int(max_dollars / cost_per_contract)

        # Limit by available volume
        final_contracts = min(max_contracts, available_volume)

        return final_contracts

    # =========================================================================
    # V3 Filters
    # =========================================================================

    def _passes_v3_filters(self, mispricing_cents: float, sigma_distance: float) -> Tuple[bool, str]:
        """
        Apply V3 filters to determine if trade should be taken.

        Returns (passes, reason) tuple.
        """
        # Filter 1: Must be a tail strike (beyond ±2σ)
        if abs(sigma_distance) < self.config.min_sigma:
            self.filtered_trades["not_tail"] += 1
            return False, f"not_tail ({sigma_distance:+.1f}σ)"

        # Filter 2: Minimum mispricing (5¢ threshold)
        if mispricing_cents < self.config.min_mispricing_cents:
            self.filtered_trades["insufficient_mispricing"] += 1
            return False, f"mispricing ({mispricing_cents:.0f}¢ < {self.config.min_mispricing_cents}¢)"

        return True, "passed"

    # =========================================================================
    # Market Data
    # =========================================================================

    def _get_kalshi_markets(self, asset: str) -> List[dict]:
        """Fetch current Kalshi markets for an asset (BTC or ETH)."""
        asset_config = ASSETS[asset]
        series = asset_config.kalshi_series

        now = datetime.now(timezone.utc)
        est_offset = timedelta(hours=-5)
        now_est = now + est_offset

        year_short = str(now.year)[-2:]

        # Build event tickers
        # Hourly: next hour in EST
        next_hour_est = (now_est.hour + 1) % 24
        hourly_day_dt = now_est if next_hour_est > now_est.hour else now_est + timedelta(days=1)
        hourly_month_str = hourly_day_dt.strftime("%b").upper()
        hourly_event = f"{series}-{year_short}{hourly_month_str}{hourly_day_dt.day:02d}{next_hour_est:02d}"

        # Daily: 5pm EST today, or tomorrow if past 5pm
        if now_est.hour >= 17:
            daily_day_dt = now_est + timedelta(days=1)
        else:
            daily_day_dt = now_est
        daily_month_str = daily_day_dt.strftime("%b").upper()
        daily_event = f"{series}-{year_short}{daily_month_str}{daily_day_dt.day:02d}17"

        # Weekly: Friday 5pm EST
        days_until_friday = (4 - now_est.weekday()) % 7
        if days_until_friday == 0 and now_est.hour >= 17:
            days_until_friday = 7
        friday_dt = now_est + timedelta(days=days_until_friday)
        friday_month_str = friday_dt.strftime("%b").upper()
        weekly_event = f"{series}-{year_short}{friday_month_str}{friday_dt.day:02d}17"

        events = list(dict.fromkeys([hourly_event, daily_event, weekly_event]))

        markets = []
        for event_ticker in events:
            try:
                result = self.kalshi.get_markets(event_ticker=event_ticker, limit=200)
                for m in result.get('markets', []):
                    if m.get('status') == 'active':
                        m['_asset'] = asset  # Tag with asset for tracking
                        markets.append(m)
                time.sleep(0.2)
            except Exception as e:
                print(f"    Warning: Failed to fetch {event_ticker}: {e}")

        return markets

    # =========================================================================
    # Trading Logic
    # =========================================================================

    def _check_settlements(self):
        """Check for and resolve settled positions using Kalshi's actual results."""
        now = datetime.now(timezone.utc)

        for pos in self._get_open_positions():
            if pos.settlement_time <= now:
                # Check Kalshi for the actual settlement result
                try:
                    market_data = self.kalshi.get_market(pos.ticker)
                    market = market_data.get("market", {})
                    status = market.get("status")

                    # Only process if market is finalized/settled
                    if status not in ("finalized", "settled"):
                        # Not yet settled, might be slight delay
                        continue

                    result = market.get("result")  # "yes" or "no"
                    yes_payout = market.get("settlement_value", 0)  # 0 or 100 for YES side
                    expiration_value = market.get("expiration_value")  # Actual BRTI price

                    # Calculate payout based on our side
                    # YES gets yes_payout, NO gets (100 - yes_payout)
                    if pos.side == "YES":
                        our_payout = yes_payout
                        won = result == "yes"
                    else:  # NO
                        our_payout = 100 - yes_payout
                        won = result == "no"

                    pos.close(
                        exit_price=our_payout,
                        exit_time=now,
                        reason="settlement",
                        settlement_value=our_payout
                    )

                    # Update bankroll
                    self.bankroll += pos.pnl

                    outcome = "WON" if won else "LOST"
                    brti_str = f"BRTI=${expiration_value:,.2f}" if expiration_value else ""
                    print(f"  SETTLED: {pos.ticker} {pos.side} @ ${pos.strike:,.0f} "
                          f"-> {outcome} ({brti_str}, P&L: ${pos.pnl:+.2f})")

                    self._log_trade(
                        action="CLOSE",
                        ticker=pos.ticker,
                        strike=pos.strike,
                        side=pos.side,
                        price=our_payout,
                        size=pos.size,
                        model_prob=pos.model_prob,
                        edge=0,
                        reason=f"settlement_{outcome.lower()}"
                    )
                except Exception as e:
                    print(f"  Error checking settlement for {pos.ticker}: {e}")

    def _log_trade(self, action: str, ticker: str, strike: float, side: str,
                   price: float, size: int, model_prob: float, edge: float, reason: str):
        """Log a trade action."""
        self.trade_log.append(TradeLog(
            timestamp=datetime.now(timezone.utc),
            action=action,
            ticker=ticker,
            strike=strike,
            side=side,
            price=price,
            size=size,
            model_prob=model_prob,
            edge=edge,
            reason=reason,
            bankroll_after=self.bankroll
        ))

    def _record_equity(self):
        """Record current equity state."""
        # Update peak
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll

        # Calculate drawdown
        drawdown = self.peak_bankroll - self.bankroll
        drawdown_pct = (drawdown / self.peak_bankroll * 100) if self.peak_bankroll > 0 else 0

        # Calculate total realized P&L
        closed = [p for p in self.positions.values() if not p.is_open and p.pnl is not None]
        total_pnl = sum(p.pnl for p in closed)

        self.equity_curve.append(EquityPoint(
            timestamp=datetime.now(timezone.utc),
            bankroll=self.bankroll,
            open_exposure=self._total_open_exposure(),
            open_positions=len(self._get_open_positions()),
            total_pnl=total_pnl,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct
        ))

    def _get_orderbook_depth(self, ticker: str, side: str, ask_price: float) -> Optional[int]:
        """
        Get available volume at the ask from orderbook.

        Returns actual depth or None if fetch failed.
        """
        try:
            orderbook = self.kalshi.get_market_orderbook(ticker, depth=20)
            book = orderbook.get("orderbook", {})

            if side == "YES":
                no_bids = book.get("no", []) or []
                target_price = 100 - int(ask_price)
                for price, qty in no_bids:
                    if price == target_price:
                        return qty
                return sum(qty for price, qty in no_bids if price >= target_price)
            else:
                yes_bids = book.get("yes", []) or []
                target_price = 100 - int(ask_price)
                for price, qty in yes_bids:
                    if price == target_price:
                        return qty
                return sum(qty for price, qty in yes_bids if price >= target_price)
        except Exception as e:
            print(f"    Warning: Could not fetch orderbook for {ticker}: {e}")
            return None

    def _open_position(self, market: dict, asset: str, side: str, model_prob: float,
                       mispricing: float, sigma: float, settlement_time: datetime,
                       hours_to_settlement: float, orderbook_depth: int = None):
        """Open a new position."""
        ticker = market["ticker"]
        strike = market["floor_strike"]

        if side == "YES":
            ask_price = market.get("yes_ask", 0) or 0
        else:
            ask_price = market.get("no_ask", 0) or 0

        if ask_price <= 0:
            return

        # Apply V3 filters
        passes, filter_reason = self._passes_v3_filters(
            mispricing_cents=mispricing,
            sigma_distance=sigma
        )

        if not passes:
            print(f"  FILTERED: {ticker} {side} @ {ask_price}¢ - {filter_reason}")
            return

        # Use pre-fetched orderbook depth if available, otherwise fetch
        if orderbook_depth is not None:
            available_volume = orderbook_depth
        else:
            available_volume = self._get_orderbook_depth(ticker, side, ask_price)

        # Skip if we couldn't get orderbook depth
        if available_volume is None or available_volume <= 0:
            print(f"  SKIPPED: {ticker} - no orderbook depth available")
            return

        # Calculate position size
        size = self._calculate_position_size(
            model_prob=model_prob if side == "YES" else (1 - model_prob),
            ask_price=ask_price,
            settlement_time=settlement_time,
            strike=strike,
            available_volume=available_volume
        )

        if size <= 0:
            return

        # Create position
        self.position_counter += 1
        pos_id = f"pos_{self.position_counter}"

        position = Position(
            id=pos_id,
            ticker=ticker,
            asset=asset,
            strike=strike,
            settlement_time=settlement_time,
            side=side,
            entry_price=ask_price,
            size=size,
            entry_time=datetime.now(timezone.utc),
            model_prob=model_prob,
            edge=mispricing / 100,  # Store as decimal for compatibility
            deribit_expiry="2D_SURFACE"  # Using full 2D IV surface interpolation
        )

        self.positions[pos_id] = position

        cost = position.cost_basis
        print(f"  OPEN: [{asset}] {ticker} {side} {size} @ {ask_price}¢ "
              f"(${cost:.2f}, Δ{mispricing:.0f}¢, σ={sigma:+.1f}, depth={available_volume})")

        self._log_trade(
            action="OPEN",
            ticker=ticker,
            strike=strike,
            side=side,
            price=ask_price,
            size=size,
            model_prob=model_prob,
            edge=mispricing / 100,
            reason="signal_v3"
        )

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run_once(self):
        """Run a single iteration of the trading loop."""
        now = datetime.now(timezone.utc)
        print(f"\n{'='*70}")
        print(f"Paper Trader V3 - {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Bankroll: ${self.bankroll:,.2f} | Open positions: {len(self._get_open_positions())}")
        print(f"Assets: {', '.join(self.config.assets)}")
        print(f"{'='*70}")

        # 1. Check settlements
        print("\n[1] Checking settlements...")
        self._check_settlements()

        # 2. Get market data for all assets
        print("\n[2] Fetching market data...")

        iv_surfaces = {}      # asset -> IVSurface2D
        spots = {}            # asset -> spot price
        deribit_chains = {}   # asset -> DataFrame (for snapshots)
        all_markets = []      # Combined markets from all assets

        for asset in self.config.assets:
            print(f"\n  --- {asset} ---")
            try:
                spot = self.deribit.get_index_price(asset)["index_price"]
                chain = self.deribit.get_option_chain(asset)
                markets = self._get_kalshi_markets(asset)

                # Build 2D IV surface
                iv_surface = build_iv_surface_2d(chain, spot)
                if iv_surface is None:
                    print(f"    WARNING: Could not build IV surface for {asset}")
                    continue

                bounds = iv_surface.get_grid_bounds()
                print(f"    Spot: ${spot:,.2f}")
                print(f"    IV Surface: {bounds['min_time_hours']:.0f}h - {bounds['max_time_hours']:.0f}h, "
                      f"${bounds['min_strike']:,.0f} - ${bounds['max_strike']:,.0f}")
                print(f"    Kalshi markets: {len(markets)}")

                iv_surfaces[asset] = iv_surface
                spots[asset] = spot
                deribit_chains[asset] = chain
                all_markets.extend(markets)

                time.sleep(0.3)  # Rate limit between assets
            except Exception as e:
                print(f"    ERROR fetching {asset}: {e}")
                continue

        if not iv_surfaces:
            print("  ERROR: No IV surfaces built for any asset")
            return

        # 3. Analyze opportunities
        print("\n[3] Analyzing opportunities...")

        # Group markets by settlement
        by_settlement = defaultdict(list)
        for m in all_markets:
            close_time = m.get('close_time')
            if close_time:
                by_settlement[close_time].append(m)

        signals = {}  # ticker -> signal info
        opportunities = []

        for settlement_str in sorted(by_settlement.keys()):
            settlement_markets = by_settlement[settlement_str]
            settlement_time = pd.to_datetime(settlement_str, utc=True)

            if settlement_time <= now:
                continue

            hours_to_settlement = (settlement_time - now).total_seconds() / 3600
            years_to_settlement = hours_to_settlement / (365.25 * 24)

            for m in settlement_markets:
                strike = m.get('floor_strike')
                if not strike:
                    continue

                # Get the correct IV surface for this asset
                asset = m.get('_asset')
                if asset not in iv_surfaces:
                    continue
                iv_surface = iv_surfaces[asset]

                ticker = m.get('ticker')
                yes_bid = m.get('yes_bid', 0) or 0
                yes_ask = m.get('yes_ask', 0) or 0
                no_bid = m.get('no_bid', 0) or 0
                no_ask = m.get('no_ask', 0) or 0

                yes_spread = yes_ask - yes_bid
                no_spread = no_ask - no_bid

                # Check if settlement is within IV surface grid
                if not (iv_surface.min_time <= years_to_settlement <= iv_surface.max_time):
                    continue

                # Check if strike is within IV surface grid
                if not iv_surface.is_in_grid(strike, years_to_settlement):
                    continue

                # Model probability from 2D IV surface (Black-Scholes)
                model_prob = iv_surface.prob_above(strike, years_to_settlement)
                if model_prob is None:
                    continue

                # Sigma distance from spot (using ATM IV at this time)
                sigma_distance = iv_surface.sigma_distance(strike, years_to_settlement)
                if sigma_distance is None:
                    continue

                # Calculate mispricing in CENTS (not percentage)
                model_yes_cents = model_prob * 100
                model_no_cents = (1 - model_prob) * 100
                mispricing_yes = model_yes_cents - yes_ask  # positive = underpriced
                mispricing_no = model_no_cents - no_ask

                # Determine signal based on mispricing in cents
                signal_side = None
                signal_mispricing = 0

                yes_liquid = yes_bid >= self.config.min_bid_cents and yes_spread <= self.config.max_spread_cents
                no_liquid = no_bid >= self.config.min_bid_cents and no_spread <= self.config.max_spread_cents

                if yes_liquid and mispricing_yes >= self.config.min_mispricing_cents:
                    signal_side = "YES"
                    signal_mispricing = mispricing_yes

                if no_liquid and mispricing_no >= self.config.min_mispricing_cents and mispricing_no > signal_mispricing:
                    signal_side = "NO"
                    signal_mispricing = mispricing_no

                # Fetch orderbook depth for signals with mispricing
                orderbook_depth = None
                if signal_side:
                    ask_price = yes_ask if signal_side == "YES" else no_ask
                    orderbook_depth = self._get_orderbook_depth(ticker, signal_side, ask_price)

                # Store signal
                signals[ticker] = {
                    "asset": asset,
                    "side": signal_side,
                    "mispricing": signal_mispricing,
                    "sigma": sigma_distance,
                    "model_prob": model_prob,
                    "yes_bid": yes_bid,
                    "no_bid": no_bid,
                    "orderbook_depth": orderbook_depth
                }

                if signal_side:
                    opportunities.append({
                        "market": m,
                        "asset": asset,
                        "side": signal_side,
                        "mispricing": signal_mispricing,
                        "sigma": sigma_distance,
                        "model_prob": model_prob,
                        "settlement_time": settlement_time,
                        "hours_to_settlement": hours_to_settlement,
                        "orderbook_depth": orderbook_depth
                    })

        # 4. Hold to expiration
        print("\n[4] Holding all positions to expiration...")

        # 5. Open new positions (with V3 filters)
        print(f"\n[5] Found {len(opportunities)} raw opportunities")

        # Sort by mispricing descending (highest mispricing first)
        opportunities.sort(key=lambda x: x["mispricing"], reverse=True)

        # Get tickers we already have positions in
        open_tickers = {p.ticker for p in self._get_open_positions()}

        opened = 0
        filtered = 0
        for opp in opportunities:
            ticker = opp["market"]["ticker"]

            # Skip if already have position in this market
            if ticker in open_tickers:
                continue

            before_count = self.position_counter
            self._open_position(
                market=opp["market"],
                asset=opp["asset"],
                side=opp["side"],
                model_prob=opp["model_prob"],
                mispricing=opp["mispricing"],
                sigma=opp["sigma"],
                settlement_time=opp["settlement_time"],
                hours_to_settlement=opp["hours_to_settlement"],
                orderbook_depth=opp.get("orderbook_depth")
            )

            if self.position_counter > before_count:
                opened += 1
                open_tickers.add(ticker)
            else:
                filtered += 1

        print(f"    Opened: {opened}, Filtered: {filtered}")

        # 6. Save market snapshot
        print("\n[6] Saving market snapshot...")
        try:
            self._save_market_snapshot(all_markets, signals, spots, deribit_chains)
            total_options = sum(len(c) for c in deribit_chains.values())
            print(f"    Saved {len(all_markets)} Kalshi + {total_options} Deribit options")
        except Exception as e:
            print(f"    Warning: Failed to save snapshot: {e}")

        # 7. Record equity point
        self._record_equity()

        # 8. Save state
        self._save_state()

        # 9. Print summary
        self._print_summary()

    def _print_summary(self):
        """Print current state summary."""
        open_positions = self._get_open_positions()
        closed_positions = [p for p in self.positions.values() if not p.is_open]

        # Calculate drawdown
        drawdown = self.peak_bankroll - self.bankroll
        drawdown_pct = (drawdown / self.peak_bankroll * 100) if self.peak_bankroll > 0 else 0
        total_return = ((self.bankroll - self.config.initial_bankroll) / self.config.initial_bankroll) * 100

        print(f"\n{'='*70}")
        print(f"SUMMARY (V3)")
        print(f"{'='*70}")
        print(f"Bankroll:     ${self.bankroll:,.2f} ({total_return:+.2f}%)")
        print(f"Peak:         ${self.peak_bankroll:,.2f}")
        print(f"Drawdown:     ${drawdown:,.2f} ({drawdown_pct:.2f}%)")
        print(f"Open exposure: ${self._total_open_exposure():,.2f} "
              f"({self._total_open_exposure()/self.bankroll*100:.1f}%)")

        # V3 Filter stats
        print(f"\n--- V3 Filter Stats ---")
        print(f"Not tail (|σ| < {self.config.min_sigma}):    {self.filtered_trades.get('not_tail', 0)} blocked")
        print(f"Outside IV grid:          {self.filtered_trades.get('outside_grid', 0)} blocked")
        print(f"Insufficient mispricing:  {self.filtered_trades.get('insufficient_mispricing', 0)} blocked")

        if closed_positions:
            total_pnl = sum(p.pnl for p in closed_positions if p.pnl is not None)
            wins = [p for p in closed_positions if p.pnl is not None and p.pnl > 0]
            losses = [p for p in closed_positions if p.pnl is not None and p.pnl <= 0]
            win_rate = len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) > 0 else 0

            avg_win = sum(p.pnl for p in wins) / len(wins) if wins else 0
            avg_loss = sum(p.pnl for p in losses) / len(losses) if losses else 0

            print(f"\n--- Performance ---")
            print(f"Total P&L:    ${total_pnl:+,.2f}")
            print(f"Win rate:     {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
            print(f"Avg win:      ${avg_win:+,.2f}")
            print(f"Avg loss:     ${avg_loss:+,.2f}")

            # Profit factor
            gross_profit = sum(p.pnl for p in wins) if wins else 0
            gross_loss = abs(sum(p.pnl for p in losses)) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            print(f"Profit factor: {profit_factor:.2f}")
        else:
            print(f"\n--- Performance ---")
            print(f"No closed positions yet")

        print(f"\n--- Positions ---")
        print(f"Open:   {len(open_positions)}")
        print(f"Closed: {len(closed_positions)}")

        if open_positions:
            print(f"\nOpen positions:")
            for p in sorted(open_positions, key=lambda x: x.settlement_time):
                hours_left = (p.settlement_time - datetime.now(timezone.utc)).total_seconds() / 3600
                print(f"  [{p.asset}] {p.ticker} {p.side} {p.size}x @ {p.entry_price}¢ "
                      f"(${p.cost_basis:.2f}, edge={p.edge:.1%}, T-{hours_left:.1f}h)")

    def run_continuous(self, interval_seconds: int = 30):
        """Run continuously at specified interval."""
        print(f"Starting Paper Trader V3 (interval: {interval_seconds}s)")
        print("Press Ctrl+C to stop")

        while True:
            try:
                self.run_once()
                print(f"\nNext run in {interval_seconds}s...")
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("\nStopping paper trader.")
                self._save_state()
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                print(f"Retrying in {interval_seconds}s...")
                time.sleep(interval_seconds)

    def status(self):
        """Print current status without running trading logic."""
        self._print_summary()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper trade Kalshi KXBTCD strategy (V3 - Sigma Filter)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--status", action="store_true", help="Show status only")
    parser.add_argument("--interval", type=int, default=30,
                        help="Run interval in seconds (default: 30)")
    parser.add_argument("--bankroll", type=float, default=10000,
                        help="Initial bankroll (default: $10000)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset all state and start fresh")

    args = parser.parse_args()

    config = Config(initial_bankroll=args.bankroll)

    if args.reset:
        import shutil
        if config.data_dir.exists():
            shutil.rmtree(config.data_dir)
        print("V3 state reset.")

    trader = PaperTraderV3(config)

    if args.status:
        trader.status()
    elif args.once:
        trader.run_once()
    else:
        trader.run_continuous(args.interval)
