#!/usr/bin/env python3
"""
Paper Trader V2 for Kalshi KXBTCD Strategy

Changes from V1:
- Time ratio filter: Only trade when Kalshi time / Deribit time ≥ 50%
  (variance scaling breaks down with time mismatches)
- Avoid low probability trades (entry price < 20¢)
- Avoid uncertain trades (entry price 40-60¢)
- Conflict prevention: No YES above NO strikes (creates guaranteed loss zone)

Position sizing: Quarter Kelly with risk limits
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
from models.breeden_litzenberger import extract_implied_cdf, find_best_expiry


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Paper trading configuration."""
    initial_bankroll: float = 10000.0  # Starting capital in dollars
    kelly_fraction: float = 0.25       # Quarter Kelly
    min_edge: float = 0.03             # 3% minimum edge to trade
    max_per_strike: float = 0.05       # 5% max per single strike
    max_per_settlement: float = 0.20   # 20% max per settlement time
    max_total_open: float = 0.60       # 60% max total exposure
    min_bid_cents: int = 5             # Minimum bid for liquidity
    max_spread_cents: int = 20         # Maximum spread to trade
    data_dir: Path = Path("data/paper_trading_v2")

    # V2 Filters
    min_entry_price: int = 20          # Avoid < 20¢ (low probability lottery)
    avoid_price_range: Tuple[int, int] = (40, 60)  # Avoid 40-60¢ (uncertain zone)
    min_time_ratio: float = 0.50       # Kalshi time / Deribit time must be ≥50%


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Position:
    """Represents an open or closed position."""
    id: str
    ticker: str
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

class PaperTraderV2:
    """
    Paper trading system V2 for Kalshi KXBTCD strategy.

    Improvements over V1:
    - Time ratio filter: Kalshi/Deribit time ≥ 50% (variance scaling integrity)
    - Avoids low probability trades (<20¢)
    - Avoids uncertain trades (40-60¢)
    - Prevents conflicting positions that guarantee losses
    """

    VERSION = "2.0"

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
            "low_probability": 0,
            "uncertain_zone": 0,
            "conflict_prevention": 0,
            "bad_time_ratio": 0
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

    def _save_market_snapshot(self, kalshi_markets: List[dict], deribit_chain: pd.DataFrame,
                               spot: float, signals: Dict[str, dict]):
        """Save full market snapshot for later analysis."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")

        snapshot_dir = self._snapshots_dir() / date_str
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save Kalshi markets with model signals
        kalshi_data = []
        for m in kalshi_markets:
            ticker = m.get('ticker')
            signal = signals.get(ticker, {})
            kalshi_data.append({
                'snapshot_time': now.isoformat(),
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
                'signal_side': signal.get('side'),
                'signal_edge': signal.get('edge'),
                'spot_price': spot,
            })

        kalshi_df = pd.DataFrame(kalshi_data)
        kalshi_df.to_parquet(snapshot_dir / f"kalshi_{date_str}_{time_str}.parquet", index=False)

        # Save Deribit chain
        deribit_chain['snapshot_time'] = now.isoformat()
        deribit_chain['spot_price'] = spot
        deribit_chain.to_parquet(snapshot_dir / f"deribit_{date_str}_{time_str}.parquet", index=False)

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
    # V2 Filters
    # =========================================================================

    def _passes_v2_filters(self, edge: float, entry_price: float, side: str,
                           strike: float, settlement_time: datetime,
                           time_ratio: float) -> Tuple[bool, str]:
        """
        Apply V2 filters to determine if trade should be taken.

        Returns (passes, reason) tuple.
        """
        # Filter 1: Bad time ratio (Kalshi time / Deribit time < 50%)
        # When time ratio is low, variance scaling breaks down
        if time_ratio < self.config.min_time_ratio:
            self.filtered_trades["bad_time_ratio"] += 1
            return False, f"bad_time_ratio ({time_ratio:.0%})"

        # Filter 2: Low probability trades (< 20¢)
        if entry_price < self.config.min_entry_price:
            self.filtered_trades["low_probability"] += 1
            return False, f"low_probability ({entry_price}¢)"

        # Filter 3: Uncertain zone (40-60¢)
        avoid_min, avoid_max = self.config.avoid_price_range
        if avoid_min <= entry_price <= avoid_max:
            self.filtered_trades["uncertain_zone"] += 1
            return False, f"uncertain_zone ({entry_price}¢)"

        # Filter 4: Conflict prevention
        if self._would_create_conflict(side, strike, settlement_time):
            self.filtered_trades["conflict_prevention"] += 1
            return False, f"conflict_prevention (YES>{strike:.0f}>NO)"

        return True, "passed"

    # =========================================================================
    # Market Data
    # =========================================================================

    def _get_kalshi_markets(self) -> List[dict]:
        """Fetch current KXBTCD markets from Kalshi."""
        now = datetime.now(timezone.utc)
        est_offset = timedelta(hours=-5)
        now_est = now + est_offset

        year_short = str(now.year)[-2:]

        # Build event tickers
        # Hourly: next hour in EST
        next_hour_est = (now_est.hour + 1) % 24
        hourly_day_dt = now_est if next_hour_est > now_est.hour else now_est + timedelta(days=1)
        hourly_month_str = hourly_day_dt.strftime("%b").upper()
        hourly_event = f"KXBTCD-{year_short}{hourly_month_str}{hourly_day_dt.day:02d}{next_hour_est:02d}"

        # Daily: 5pm EST today, or tomorrow if past 5pm
        if now_est.hour >= 17:
            daily_day_dt = now_est + timedelta(days=1)
        else:
            daily_day_dt = now_est
        daily_month_str = daily_day_dt.strftime("%b").upper()
        daily_event = f"KXBTCD-{year_short}{daily_month_str}{daily_day_dt.day:02d}17"

        # Weekly: Friday 5pm EST
        days_until_friday = (4 - now_est.weekday()) % 7
        if days_until_friday == 0 and now_est.hour >= 17:
            days_until_friday = 7
        friday_dt = now_est + timedelta(days=days_until_friday)
        friday_month_str = friday_dt.strftime("%b").upper()
        weekly_event = f"KXBTCD-{year_short}{friday_month_str}{friday_dt.day:02d}17"

        events = list(dict.fromkeys([hourly_event, daily_event, weekly_event]))
        print(f"    Events: {events}")

        markets = []
        for event_ticker in events:
            try:
                result = self.kalshi.get_markets(event_ticker=event_ticker, limit=200)
                for m in result.get('markets', []):
                    if m.get('status') == 'active':
                        markets.append(m)
                time.sleep(0.2)
            except Exception as e:
                print(f"  Warning: Failed to fetch {event_ticker}: {e}")

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

    def _get_orderbook_depth(self, ticker: str, side: str, ask_price: float) -> int:
        """
        Get available volume at the ask from orderbook.
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
                total = sum(qty for price, qty in no_bids if price >= target_price)
                return total if total > 0 else 50
            else:
                yes_bids = book.get("yes", []) or []
                target_price = 100 - int(ask_price)
                for price, qty in yes_bids:
                    if price == target_price:
                        return qty
                total = sum(qty for price, qty in yes_bids if price >= target_price)
                return total if total > 0 else 50
        except Exception as e:
            print(f"    Warning: Could not fetch orderbook for {ticker}: {e}")
            return 50

    def _parse_deribit_expiry(self, expiry_str: str) -> datetime:
        """Parse Deribit expiry string (e.g. '26JAN24' or '6FEB24') to datetime (8:00 UTC)."""
        from datetime import datetime
        import re
        # Format: D(D)MMMYY - day can be 1 or 2 digits
        match = re.match(r'^(\d{1,2})([A-Z]{3})(\d{2})$', expiry_str)
        if not match:
            raise ValueError(f"Invalid expiry format: {expiry_str}")
        day = int(match.group(1))
        month_str = match.group(2)
        year = int("20" + match.group(3))
        month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                     "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
        month = month_map[month_str]
        # Deribit options expire at 8:00 UTC
        return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)

    def _open_position(self, market: dict, side: str, model_prob: float,
                       edge: float, deribit_expiry: str, settlement_time: datetime):
        """Open a new position."""
        ticker = market["ticker"]
        strike = market["floor_strike"]
        now = datetime.now(timezone.utc)

        if side == "YES":
            ask_price = market.get("yes_ask", 0) or 0
        else:
            ask_price = market.get("no_ask", 0) or 0

        if ask_price <= 0:
            return

        # Calculate time ratio for filter
        kalshi_hours = (settlement_time - now).total_seconds() / 3600
        deribit_expiry_dt = self._parse_deribit_expiry(deribit_expiry)
        deribit_hours = (deribit_expiry_dt - now).total_seconds() / 3600
        time_ratio = kalshi_hours / deribit_hours if deribit_hours > 0 else 0

        # Apply V2 filters
        passes, filter_reason = self._passes_v2_filters(
            edge=edge,
            entry_price=ask_price,
            side=side,
            strike=strike,
            settlement_time=settlement_time,
            time_ratio=time_ratio
        )

        if not passes:
            print(f"  FILTERED: {ticker} {side} @ {ask_price}¢ - {filter_reason}")
            return

        # Get real orderbook depth
        available_volume = self._get_orderbook_depth(ticker, side, ask_price)

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
            strike=strike,
            settlement_time=settlement_time,
            side=side,
            entry_price=ask_price,
            size=size,
            entry_time=datetime.now(timezone.utc),
            model_prob=model_prob,
            edge=edge,
            deribit_expiry=deribit_expiry
        )

        self.positions[pos_id] = position

        cost = position.cost_basis
        print(f"  OPEN: {ticker} {side} {size} @ {ask_price}¢ "
              f"(${cost:.2f}, edge={edge:.1%}, model={model_prob:.1%})")

        self._log_trade(
            action="OPEN",
            ticker=ticker,
            strike=strike,
            side=side,
            price=ask_price,
            size=size,
            model_prob=model_prob,
            edge=edge,
            reason="signal_v2"
        )

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run_once(self):
        """Run a single iteration of the trading loop."""
        now = datetime.now(timezone.utc)
        print(f"\n{'='*70}")
        print(f"Paper Trader V2 - {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Bankroll: ${self.bankroll:,.2f} | Open positions: {len(self._get_open_positions())}")
        print(f"{'='*70}")

        # 1. Check settlements
        print("\n[1] Checking settlements...")
        self._check_settlements()

        # 2. Get market data
        print("\n[2] Fetching market data...")
        spot = self.deribit.get_index_price("BTC")["index_price"]
        chain = self.deribit.get_option_chain("BTC")
        available_expiries = sorted(chain["expiry_str"].unique())
        markets = self._get_kalshi_markets()

        print(f"    Spot: ${spot:,.2f}")
        print(f"    Kalshi markets: {len(markets)}")

        # 3. Analyze opportunities
        print("\n[3] Analyzing opportunities...")

        # Group markets by settlement
        by_settlement = defaultdict(list)
        for m in markets:
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

            # Find Deribit expiry
            best_expiry = find_best_expiry(available_expiries, settlement_time)
            if not best_expiry:
                continue

            # Extract implied CDF
            implied_cdf = extract_implied_cdf(chain, spot, best_expiry)
            if implied_cdf is None:
                continue

            for m in settlement_markets:
                strike = m.get('floor_strike')
                if not strike:
                    continue

                if strike < implied_cdf.strikes.min() or strike > implied_cdf.strikes.max():
                    continue

                ticker = m.get('ticker')
                yes_bid = m.get('yes_bid', 0) or 0
                yes_ask = m.get('yes_ask', 0) or 0
                no_bid = m.get('no_bid', 0) or 0
                no_ask = m.get('no_ask', 0) or 0

                yes_spread = yes_ask - yes_bid
                no_spread = no_ask - no_bid

                # Model probability
                model_prob = implied_cdf.prob_above_at_time(strike, years_to_settlement)

                # Calculate edges
                yes_ask_prob = yes_ask / 100
                no_ask_prob = no_ask / 100
                edge_yes = model_prob - yes_ask_prob
                edge_no = (1 - model_prob) - no_ask_prob

                # Determine signal
                signal_side = None
                signal_edge = 0

                yes_liquid = yes_bid >= self.config.min_bid_cents and yes_spread <= self.config.max_spread_cents
                no_liquid = no_bid >= self.config.min_bid_cents and no_spread <= self.config.max_spread_cents

                if yes_liquid and edge_yes >= self.config.min_edge:
                    signal_side = "YES"
                    signal_edge = edge_yes

                if no_liquid and edge_no >= self.config.min_edge and edge_no > signal_edge:
                    signal_side = "NO"
                    signal_edge = edge_no

                # Store signal
                signals[ticker] = {
                    "side": signal_side,
                    "edge": signal_edge,
                    "model_prob": model_prob,
                    "yes_bid": yes_bid,
                    "no_bid": no_bid
                }

                if signal_side:
                    opportunities.append({
                        "market": m,
                        "side": signal_side,
                        "edge": signal_edge,
                        "model_prob": model_prob,
                        "deribit_expiry": best_expiry,
                        "settlement_time": settlement_time
                    })

        # 4. Hold to expiration
        print("\n[4] Holding all positions to expiration...")

        # 5. Open new positions (with V2 filters)
        print(f"\n[5] Found {len(opportunities)} raw opportunities")

        # Sort by edge descending
        opportunities.sort(key=lambda x: x["edge"], reverse=True)

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
                side=opp["side"],
                model_prob=opp["model_prob"],
                edge=opp["edge"],
                deribit_expiry=opp["deribit_expiry"],
                settlement_time=opp["settlement_time"]
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
            self._save_market_snapshot(markets, chain, spot, signals)
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
        print(f"SUMMARY (V2)")
        print(f"{'='*70}")
        print(f"Bankroll:     ${self.bankroll:,.2f} ({total_return:+.2f}%)")
        print(f"Peak:         ${self.peak_bankroll:,.2f}")
        print(f"Drawdown:     ${drawdown:,.2f} ({drawdown_pct:.2f}%)")
        print(f"Open exposure: ${self._total_open_exposure():,.2f} "
              f"({self._total_open_exposure()/self.bankroll*100:.1f}%)")

        # V2 Filter stats
        print(f"\n--- V2 Filter Stats ---")
        print(f"Bad time ratio (<50%):   {self.filtered_trades.get('bad_time_ratio', 0)} blocked")
        print(f"Low probability (<20¢):  {self.filtered_trades['low_probability']} blocked")
        print(f"Uncertain zone (40-60¢): {self.filtered_trades['uncertain_zone']} blocked")
        print(f"Conflict prevention:     {self.filtered_trades['conflict_prevention']} blocked")

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
                print(f"  {p.ticker} {p.side} {p.size}x @ {p.entry_price}¢ "
                      f"(${p.cost_basis:.2f}, edge={p.edge:.1%}, T-{hours_left:.1f}h)")

    def run_continuous(self, interval_seconds: int = 300):
        """Run continuously at specified interval."""
        print(f"Starting Paper Trader V2 (interval: {interval_seconds}s)")
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

    parser = argparse.ArgumentParser(description="Paper trade Kalshi KXBTCD strategy (V2)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--status", action="store_true", help="Show status only")
    parser.add_argument("--interval", type=int, default=300,
                        help="Run interval in seconds (default: 300)")
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
        print("V2 state reset.")

    trader = PaperTraderV2(config)

    if args.status:
        trader.status()
    elif args.once:
        trader.run_once()
    else:
        trader.run_continuous(args.interval)
