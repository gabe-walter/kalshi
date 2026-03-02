#!/usr/bin/env python3
"""
Live Trader for Kalshi Crypto Options (Demo Environment)

Runs the V3 IV Surface strategy with real order execution on Kalshi's demo API.
Streams real-time data via WebSocket, computes signals from Deribit IV surface,
and places limit orders when mispricings are detected.

Usage:
    python live_trader.py                    # Run on demo (default)
    python live_trader.py --live             # Run on production (real money!)
    python live_trader.py --once             # Single scan, then exit
    python live_trader.py --status           # Show current positions/balance
"""

import sys
sys.path.insert(0, "src")

import asyncio
import json
import time
import uuid
import base64
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import pandas as pd
import aiohttp

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

from data.deribit_client import DeribitClient
from data.kalshi_client import KalshiClient
from models.iv_surface_2d import build_iv_surface_2d


# =============================================================================
# Configuration
# =============================================================================

CRYPTO_PREFIXES = ("KXBTCD", "KXETHD")
ASSET_FOR_PREFIX = {"KXBTCD": "BTC", "KXETHD": "ETH"}


@dataclass
class LiveConfig:
    """Live trading configuration."""
    # Strategy params (V3 defaults)
    kelly_fraction: float = 0.75
    min_mispricing_cents: int = 5
    min_sigma: float = 1.75
    min_bid_cents: int = 5
    max_spread_cents: int = 20

    # Risk limits
    max_per_strike: float = 0.15       # 15% of bankroll per strike
    max_per_settlement: float = 0.50   # 50% per settlement
    max_total_open: float = 0.80       # 80% total exposure

    # Execution
    scan_interval: int = 30            # Seconds between signal scans
    deribit_interval: int = 60         # Deribit IV refresh interval
    market_refresh_interval: int = 300 # Market metadata refresh
    stale_order_seconds: int = 120     # Cancel unfilled orders after this

    # Assets
    assets: tuple = ("BTC", "ETH")

    # Environment
    demo: bool = True
    data_dir: Path = Path("data/live_trader")


# =============================================================================
# Live Trader
# =============================================================================

class LiveTrader:
    """
    Live trading system for Kalshi crypto options.

    Combines real-time WS data streaming with the V3 IV Surface strategy,
    placing actual limit orders via the Kalshi API.
    """

    PROD_WS = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    DEMO_WS = "wss://demo-api.kalshi.co/trade-api/ws/v2"

    def __init__(self, config: LiveConfig = None):
        self.config = config or LiveConfig()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # API clients
        self.deribit = DeribitClient()
        self.kalshi = KalshiClient(demo=self.config.demo)
        self.ws_url = self.DEMO_WS if self.config.demo else self.PROD_WS

        # Auth for WS
        self.api_key_id = os.getenv("KALSHI_API_KEY_ID")
        self.private_key = None
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if key_path:
            with open(key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )

        # Market state (from WS)
        self.latest_ticker: Dict[str, dict] = {}
        self.market_meta: Dict[str, dict] = {}
        self.orderbooks: Dict[str, dict] = {}

        # IV surface state
        self.iv_surfaces: Dict[str, object] = {}
        self.spots: Dict[str, float] = {}

        # Order tracking
        self.active_orders: Dict[str, dict] = {}   # order_id -> order info
        self.filled_tickers: set = set()            # tickers we've already traded

        # Trade log (local record)
        self.trade_log: List[dict] = []

        # WS state
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._sub_id = 0
        self._running = False
        self._last_message_time: Optional[datetime] = None
        self._reconnect_count = 0

        # Stats
        self.start_time: Optional[datetime] = None
        self.ticker_count = 0
        self.orders_placed = 0
        self.orders_filled = 0
        self.orders_cancelled = 0

        # Load previous state if exists
        self._load_state()

    # =========================================================================
    # Auth
    # =========================================================================

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        if self.private_key is None:
            raise RuntimeError("No private key. Set KALSHI_PRIVATE_KEY_PATH.")
        message = f"{timestamp_ms}{method}{path}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    # =========================================================================
    # WebSocket Connection
    # =========================================================================

    async def _ws_connect(self) -> aiohttp.ClientWebSocketResponse:
        timestamp = str(int(time.time() * 1000))
        path = "/trade-api/ws/v2"
        signature = self._sign(timestamp, "GET", path)

        headers = {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(
            self.ws_url, headers=headers, heartbeat=30,
        )
        return self._ws

    async def _ws_subscribe(self, channels: List[str]):
        self._sub_id += 1
        msg = {
            "id": self._sub_id,
            "cmd": "subscribe",
            "params": {"channels": channels},
        }
        await self._ws.send_json(msg)

    async def _ws_close(self):
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    # =========================================================================
    # WS Message Handlers
    # =========================================================================

    def _on_ticker(self, msg: dict):
        ticker = msg.get("market_ticker", "")
        if not any(ticker.startswith(p) for p in CRYPTO_PREFIXES):
            return

        self._last_message_time = datetime.now(timezone.utc)
        meta = self.market_meta.get(ticker, {})
        asset = meta.get("asset")

        model_data = self._compute_model_prob(ticker)
        model_prob = model_data.get("model_prob")
        sigma_distance = model_data.get("sigma_distance")

        yes_ask = msg.get("yes_ask")
        yes_bid = msg.get("yes_bid")

        mispricing_yes = None
        mispricing_no = None
        if model_prob is not None and yes_ask is not None and yes_ask > 0:
            mispricing_yes = model_prob * 100 - yes_ask
        if model_prob is not None and yes_bid is not None and yes_bid > 0:
            no_ask = 100 - yes_bid
            mispricing_no = (1 - model_prob) * 100 - no_ask

        record = {
            "received_at": self._last_message_time.isoformat(),
            "market_ticker": ticker,
            "asset": asset,
            "strike": meta.get("floor_strike"),
            "close_time": meta.get("close_time"),
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "last_price": msg.get("price"),
            "volume": msg.get("volume"),
            "open_interest": msg.get("open_interest"),
            "spot_price": self.spots.get(asset) if asset else None,
            "model_prob": model_prob,
            "sigma_distance": sigma_distance,
            "mispricing_yes": mispricing_yes,
            "mispricing_no": mispricing_no,
        }

        self.latest_ticker[ticker] = record
        self.ticker_count += 1

    def _on_orderbook_snapshot(self, msg: dict):
        ticker = msg.get("market_ticker", "")
        if not any(ticker.startswith(p) for p in CRYPTO_PREFIXES):
            return
        self._last_message_time = datetime.now(timezone.utc)

        yes_levels = {}
        no_levels = {}
        for price, qty in (msg.get("yes", []) or []):
            if qty > 0:
                yes_levels[price] = qty
        for price, qty in (msg.get("no", []) or []):
            if qty > 0:
                no_levels[price] = qty
        self.orderbooks[ticker] = {"yes": yes_levels, "no": no_levels}

    def _on_orderbook_delta(self, msg: dict):
        ticker = msg.get("market_ticker", "")
        if not any(ticker.startswith(p) for p in CRYPTO_PREFIXES):
            return
        self._last_message_time = datetime.now(timezone.utc)

        book = self.orderbooks.get(ticker)
        if book is None:
            return

        for price, qty in (msg.get("yes", []) or []):
            if qty <= 0:
                book["yes"].pop(price, None)
            else:
                book["yes"][price] = qty
        for price, qty in (msg.get("no", []) or []):
            if qty <= 0:
                book["no"].pop(price, None)
            else:
                book["no"][price] = qty

    # =========================================================================
    # Model / Signal Computation
    # =========================================================================

    def _compute_model_prob(self, ticker: str) -> dict:
        meta = self.market_meta.get(ticker)
        if not meta:
            return {"model_prob": None, "sigma_distance": None}

        asset = meta.get("asset")
        strike = meta.get("floor_strike")
        close_time_str = meta.get("close_time")

        if not all([asset, strike, close_time_str]):
            return {"model_prob": None, "sigma_distance": None}

        surface = self.iv_surfaces.get(asset)
        if surface is None:
            return {"model_prob": None, "sigma_distance": None}

        now = datetime.now(timezone.utc)
        try:
            settlement = pd.to_datetime(close_time_str, utc=True)
        except Exception:
            return {"model_prob": None, "sigma_distance": None}

        years_to_settlement = (settlement - now).total_seconds() / (365.25 * 24 * 3600)
        if years_to_settlement <= 0:
            return {"model_prob": None, "sigma_distance": None}

        if not surface.is_in_grid(strike, years_to_settlement):
            return {"model_prob": None, "sigma_distance": None}

        model_prob = surface.prob_above(strike, years_to_settlement)
        sigma_distance = surface.sigma_distance(strike, years_to_settlement)
        return {"model_prob": model_prob, "sigma_distance": sigma_distance}

    def _get_available_depth(self, ticker: str, side: str, price_cents: int) -> int:
        """Get available depth from live orderbook at or better than price."""
        book = self.orderbooks.get(ticker)
        if not book:
            return 0

        if side == "yes":
            # To buy YES, we need NO sellers (or equivalently, depth at the ask)
            # The orderbook 'no' side shows no bids; to buy YES at X,
            # we match against NO bids at (100 - X) or higher
            no_bids = book.get("no", {})
            target = 100 - price_cents
            return sum(qty for p, qty in no_bids.items() if p >= target)
        else:
            # To buy NO, match against YES bids at (100 - price) or higher
            yes_bids = book.get("yes", {})
            target = 100 - price_cents
            return sum(qty for p, qty in yes_bids.items() if p >= target)

    # =========================================================================
    # Risk Management (mirrors paper_trader_v3)
    # =========================================================================

    def _get_api_balance(self) -> float:
        """Get available balance in dollars from Kalshi API."""
        try:
            resp = self.kalshi.get_balance()
            return resp.get("balance", 0) / 100.0  # cents -> dollars
        except Exception as e:
            print(f"  [Risk] Error fetching balance: {e}")
            return 0.0

    def _get_api_positions(self) -> List[dict]:
        """Get current positions from Kalshi API."""
        try:
            resp = self.kalshi.get_positions(settlement_status="unsettled")
            positions = []
            for ep in resp.get("event_positions", []):
                for mp in ep.get("market_positions", []):
                    positions.append(mp)
            return positions
        except Exception as e:
            print(f"  [Risk] Error fetching positions: {e}")
            return []

    def _total_open_exposure(self, positions: List[dict]) -> float:
        """Total exposure in dollars from API positions."""
        total = 0.0
        for p in positions:
            # position > 0 means long YES, < 0 means short YES (long NO)
            pos = p.get("position", 0)
            if pos > 0:
                # Long YES: exposure = position * avg_price
                avg_cost = p.get("total_cost", 0) / 100.0 if p.get("total_cost") else 0
                total += abs(avg_cost)
            elif pos < 0:
                avg_cost = p.get("total_cost", 0) / 100.0 if p.get("total_cost") else 0
                total += abs(avg_cost)
        return total

    def _exposure_for_ticker(self, positions: List[dict], ticker: str) -> float:
        """Exposure for a specific ticker."""
        for p in positions:
            if p.get("ticker") == ticker:
                return abs(p.get("total_cost", 0)) / 100.0
        return 0.0

    def _exposure_for_settlement(self, positions: List[dict], settlement_str: str) -> float:
        """Exposure for all positions sharing a settlement time."""
        total = 0.0
        for p in positions:
            meta = self.market_meta.get(p.get("ticker"), {})
            if meta.get("close_time") == settlement_str:
                total += abs(p.get("total_cost", 0)) / 100.0
        return total

    def _has_position_in(self, positions: List[dict], ticker: str) -> bool:
        """Check if we already have a position in this ticker."""
        for p in positions:
            if p.get("ticker") == ticker and p.get("position", 0) != 0:
                return True
        return False

    def _calculate_position_size(
        self,
        model_prob: float,
        ask_price: int,  # cents
        bankroll: float,
        current_strike_exposure: float,
        current_settlement_exposure: float,
        current_total_exposure: float,
        available_depth: int,
    ) -> int:
        """Calculate position size using Kelly criterion with risk limits."""
        ask_prob = ask_price / 100.0
        if ask_prob >= 1:
            return 0

        kelly = (model_prob - ask_prob) / (1 - ask_prob)
        if kelly <= 0:
            return 0

        kelly_adjusted = kelly * self.config.kelly_fraction
        kelly_dollars = bankroll * kelly_adjusted

        # Risk limits
        max_strike = max(0, bankroll * self.config.max_per_strike - current_strike_exposure)
        max_settle = max(0, bankroll * self.config.max_per_settlement - current_settlement_exposure)
        max_total = max(0, bankroll * self.config.max_total_open - current_total_exposure)

        max_dollars = min(kelly_dollars, max_strike, max_settle, max_total)
        if max_dollars <= 0:
            return 0

        cost_per_contract = ask_price / 100.0
        max_contracts = int(max_dollars / cost_per_contract)

        return min(max_contracts, available_depth)

    # =========================================================================
    # Order Execution
    # =========================================================================

    def _place_order(self, ticker: str, side: str, price_cents: int, count: int,
                     signal_info: dict) -> Optional[str]:
        """Place a limit order via Kalshi API. Returns order_id or None."""
        client_order_id = str(uuid.uuid4())

        try:
            kwargs = {
                "ticker": ticker,
                "side": side,
                "action": "buy",
                "count": count,
                "order_type": "limit",
                "client_order_id": client_order_id,
            }
            if side == "yes":
                kwargs["yes_price"] = price_cents
            else:
                kwargs["no_price"] = price_cents

            resp = self.kalshi.create_order(**kwargs)
            order = resp.get("order", {})
            order_id = order.get("order_id")

            if order_id:
                self.active_orders[order_id] = {
                    "order_id": order_id,
                    "ticker": ticker,
                    "side": side,
                    "price": price_cents,
                    "count": count,
                    "placed_at": datetime.now(timezone.utc).isoformat(),
                    "status": order.get("status", "resting"),
                    "signal": signal_info,
                }
                self.orders_placed += 1

                cost = count * price_cents / 100
                print(f"  ORDER PLACED: {ticker} BUY {side.upper()} {count}x @ {price_cents}c "
                      f"(${cost:.2f}) [id={order_id[:8]}...]")

                self._log_trade("ORDER_PLACED", ticker, side, price_cents, count, signal_info)
                return order_id
            else:
                print(f"  ORDER FAILED: No order_id in response for {ticker}")
                return None

        except Exception as e:
            print(f"  ORDER ERROR: {ticker} - {e}")
            self._log_trade("ORDER_ERROR", ticker, side, price_cents, count,
                            {**signal_info, "error": str(e)})
            return None

    async def _check_orders(self):
        """Check status of active orders, handle fills, cancel stale orders."""
        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)

        to_remove = []

        for order_id, info in list(self.active_orders.items()):
            try:
                resp = await loop.run_in_executor(
                    None, lambda oid=order_id: self.kalshi.get_order(oid)
                )
                order = resp.get("order", {})
                status = order.get("status", "")
                fill_count = order.get("fill_count", 0)
                remaining = order.get("remaining_count", 0)

                if status == "executed" or (fill_count > 0 and remaining == 0):
                    # Fully filled
                    self.orders_filled += 1
                    self.filled_tickers.add(info["ticker"])
                    to_remove.append(order_id)
                    print(f"  FILLED: {info['ticker']} {info['side'].upper()} "
                          f"{fill_count}x @ {info['price']}c [id={order_id[:8]}...]")
                    self._log_trade("FILLED", info["ticker"], info["side"],
                                    info["price"], fill_count, info.get("signal", {}))

                elif status == "canceled" or status == "cancelled":
                    to_remove.append(order_id)
                    if fill_count > 0:
                        self.orders_filled += 1
                        self.filled_tickers.add(info["ticker"])
                        print(f"  PARTIAL FILL: {info['ticker']} {fill_count}/{info['count']} "
                              f"filled before cancel")
                    self.orders_cancelled += 1

                elif status == "resting":
                    # Check if order is stale
                    placed_at = datetime.fromisoformat(info["placed_at"])
                    age = (now - placed_at).total_seconds()
                    if age > self.config.stale_order_seconds:
                        try:
                            await loop.run_in_executor(
                                None, lambda oid=order_id: self.kalshi.cancel_order(oid)
                            )
                            to_remove.append(order_id)
                            self.orders_cancelled += 1
                            print(f"  CANCELLED STALE: {info['ticker']} "
                                  f"(age={age:.0f}s) [id={order_id[:8]}...]")
                            self._log_trade("CANCELLED_STALE", info["ticker"],
                                            info["side"], info["price"], info["count"],
                                            info.get("signal", {}))
                        except Exception as e:
                            print(f"  Cancel error for {order_id[:8]}: {e}")

            except Exception as e:
                print(f"  Order check error for {order_id[:8]}: {e}")

        for oid in to_remove:
            self.active_orders.pop(oid, None)

    # =========================================================================
    # Signal Scanning
    # =========================================================================

    async def _scan_signals(self):
        """Scan all markets for trading signals and place orders."""
        now = datetime.now(timezone.utc)
        loop = asyncio.get_event_loop()

        # Get current state from API
        bankroll = await loop.run_in_executor(None, self._get_api_balance)
        positions = await loop.run_in_executor(None, self._get_api_positions)
        total_exposure = self._total_open_exposure(positions)

        if bankroll <= 0:
            print(f"  [Scan] No balance available (${bankroll:.2f})")
            return

        # Tickers we already have positions in or active orders for
        skip_tickers = set()
        for p in positions:
            if p.get("position", 0) != 0:
                skip_tickers.add(p.get("ticker"))
        for info in self.active_orders.values():
            skip_tickers.add(info["ticker"])

        opportunities = []

        for ticker, data in self.latest_ticker.items():
            if ticker in skip_tickers:
                continue

            meta = self.market_meta.get(ticker, {})
            asset = meta.get("asset")
            close_time = meta.get("close_time")
            strike = meta.get("floor_strike")

            if not all([asset, close_time, strike]):
                continue

            # Check settlement time
            try:
                settlement = pd.to_datetime(close_time, utc=True)
            except Exception:
                continue
            if settlement <= now:
                continue

            hours_to_settlement = (settlement - now).total_seconds() / 3600

            # Get model data
            model_data = self._compute_model_prob(ticker)
            model_prob = model_data.get("model_prob")
            sigma_distance = model_data.get("sigma_distance")

            if model_prob is None or sigma_distance is None:
                continue

            # Sigma filter
            if abs(sigma_distance) < self.config.min_sigma:
                continue

            # Get prices
            yes_bid = data.get("yes_bid") or 0
            yes_ask = data.get("yes_ask") or 0
            no_bid = 100 - yes_ask if yes_ask > 0 else 0
            no_ask = 100 - yes_bid if yes_bid > 0 else 0

            # Calculate mispricings
            model_yes_cents = model_prob * 100
            model_no_cents = (1 - model_prob) * 100
            mispricing_yes = model_yes_cents - yes_ask if yes_ask > 0 else -999
            mispricing_no = model_no_cents - no_ask if no_ask > 0 else -999

            # Determine signal
            yes_spread = yes_ask - yes_bid if yes_ask > 0 and yes_bid > 0 else 999
            no_spread = no_ask - no_bid if no_ask > 0 and no_bid > 0 else 999

            yes_liquid = yes_bid >= self.config.min_bid_cents and yes_spread <= self.config.max_spread_cents
            no_liquid = no_bid >= self.config.min_bid_cents and no_spread <= self.config.max_spread_cents

            signal_side = None
            signal_mispricing = 0
            signal_price = 0

            if yes_liquid and mispricing_yes >= self.config.min_mispricing_cents:
                signal_side = "yes"
                signal_mispricing = mispricing_yes
                signal_price = yes_ask

            if no_liquid and mispricing_no >= self.config.min_mispricing_cents and mispricing_no > signal_mispricing:
                signal_side = "no"
                signal_mispricing = mispricing_no
                signal_price = no_ask

            if signal_side and signal_price > 0:
                opportunities.append({
                    "ticker": ticker,
                    "asset": asset,
                    "side": signal_side,
                    "price": signal_price,
                    "mispricing": signal_mispricing,
                    "sigma": sigma_distance,
                    "model_prob": model_prob,
                    "strike": strike,
                    "close_time": close_time,
                    "hours_to_settlement": hours_to_settlement,
                })

        # Sort by mispricing (best first)
        opportunities.sort(key=lambda x: x["mispricing"], reverse=True)

        if opportunities:
            print(f"\n  [Scan] {len(opportunities)} signals found "
                  f"(bankroll=${bankroll:.2f}, exposure=${total_exposure:.2f})")

        placed = 0
        for opp in opportunities:
            ticker = opp["ticker"]
            side = opp["side"]
            price = opp["price"]

            # Get depth from live orderbook
            depth = self._get_available_depth(ticker, side, price)
            if depth <= 0:
                continue

            # Get exposure info for risk checks
            strike_exposure = self._exposure_for_ticker(positions, ticker)
            settle_exposure = self._exposure_for_settlement(positions, opp["close_time"])

            # Calculate size
            model_prob_for_side = opp["model_prob"] if side == "yes" else (1 - opp["model_prob"])
            size = self._calculate_position_size(
                model_prob=model_prob_for_side,
                ask_price=price,
                bankroll=bankroll,
                current_strike_exposure=strike_exposure,
                current_settlement_exposure=settle_exposure,
                current_total_exposure=total_exposure,
                available_depth=depth,
            )

            if size <= 0:
                continue

            signal_info = {
                "asset": opp["asset"],
                "strike": opp["strike"],
                "sigma": opp["sigma"],
                "mispricing": opp["mispricing"],
                "model_prob": opp["model_prob"],
                "depth": depth,
                "hours_to_settlement": opp["hours_to_settlement"],
            }

            order_id = self._place_order(ticker, side, price, size, signal_info)
            if order_id:
                placed += 1
                total_exposure += size * price / 100

        if placed > 0:
            print(f"  [Scan] Placed {placed} orders")

    # =========================================================================
    # Background Loops
    # =========================================================================

    async def _refresh_markets(self):
        loop = asyncio.get_event_loop()
        for asset in self.config.assets:
            series = f"KX{asset}D"
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda s=series: self.kalshi.get_markets(
                        series_ticker=s, status="open", limit=1000
                    ),
                )
                for m in result.get("markets", []):
                    ticker = m.get("ticker", "")
                    if m.get("status") == "active":
                        self.market_meta[ticker] = {
                            "floor_strike": m.get("floor_strike"),
                            "close_time": m.get("close_time"),
                            "event_ticker": m.get("event_ticker"),
                            "asset": asset,
                            "status": m.get("status"),
                        }
            except Exception as e:
                print(f"  [Markets] Error fetching {asset}: {e}")

        print(f"  [Markets] Tracking {len(self.market_meta)} markets")

    async def _refresh_markets_loop(self):
        while self._running:
            try:
                await self._refresh_markets()
            except Exception as e:
                print(f"  [Markets] Error: {e}")
            await asyncio.sleep(self.config.market_refresh_interval)

    async def _refresh_deribit(self):
        loop = asyncio.get_event_loop()
        for asset in self.config.assets:
            try:
                spot = await loop.run_in_executor(
                    None,
                    lambda a=asset: self.deribit.get_index_price(a)["index_price"],
                )
                chain = await loop.run_in_executor(
                    None,
                    lambda a=asset: self.deribit.get_option_chain(a),
                )

                surface = build_iv_surface_2d(chain, spot)
                if surface:
                    self.iv_surfaces[asset] = surface
                    self.spots[asset] = spot
                    bounds = surface.get_grid_bounds()
                    print(f"  [Deribit] {asset} spot=${spot:,.2f}, "
                          f"IV grid: {bounds['min_time_hours']:.0f}h-{bounds['max_time_hours']:.0f}h")
            except Exception as e:
                print(f"  [Deribit] Error fetching {asset}: {e}")

    async def _refresh_deribit_loop(self):
        while self._running:
            try:
                await self._refresh_deribit()
            except Exception as e:
                print(f"  [Deribit] Error: {e}")
            await asyncio.sleep(self.config.deribit_interval)

    async def _trading_loop(self):
        """Main trading loop: scan signals and manage orders."""
        # Wait for initial data
        await asyncio.sleep(5)

        while self._running:
            try:
                # Check existing orders
                if self.active_orders:
                    await self._check_orders()

                # Scan for new signals
                await self._scan_signals()

                # Save state
                self._save_state()

            except Exception as e:
                print(f"  [Trading] Error: {e}")
                import traceback
                traceback.print_exc()

            await asyncio.sleep(self.config.scan_interval)

    async def _stats_loop(self):
        while self._running:
            await asyncio.sleep(120)
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            mins = elapsed / 60

            # Get balance
            try:
                balance = self._get_api_balance()
                positions = self._get_api_positions()
                n_pos = sum(1 for p in positions if p.get("position", 0) != 0)
            except Exception:
                balance = 0
                n_pos = 0

            env = "DEMO" if self.config.demo else "LIVE"
            print(
                f"\n  [{env} Stats] {mins:.0f}m elapsed | "
                f"Balance: ${balance:,.2f} | Positions: {n_pos} | "
                f"Orders: {self.orders_placed} placed, {self.orders_filled} filled, "
                f"{self.orders_cancelled} cancelled | "
                f"Tickers: {self.ticker_count} updates\n"
            )

    async def _health_check_loop(self):
        stale_threshold = 120
        while self._running:
            await asyncio.sleep(30)
            if self._last_message_time:
                since_last = (datetime.now(timezone.utc) - self._last_message_time).total_seconds()
                if since_last > stale_threshold:
                    print(f"  [Health] WARNING: No WS messages for {since_last:.0f}s")

    # =========================================================================
    # State Persistence
    # =========================================================================

    def _state_file(self) -> Path:
        return self.config.data_dir / "state.json"

    def _log_file(self) -> Path:
        return self.config.data_dir / "trade_log.json"

    def _log_trade(self, action: str, ticker: str, side: str, price: int,
                   count: int, signal: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "ticker": ticker,
            "side": side,
            "price": price,
            "count": count,
            "signal": signal,
        }
        self.trade_log.append(entry)

        # Append to log file
        try:
            self.config.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self._log_file(), "w") as f:
                json.dump(self.trade_log, f, indent=2)
        except Exception:
            pass

    def _save_state(self):
        state = {
            "active_orders": self.active_orders,
            "filled_tickers": list(self.filled_tickers),
            "orders_placed": self.orders_placed,
            "orders_filled": self.orders_filled,
            "orders_cancelled": self.orders_cancelled,
            "start_time": self.start_time.isoformat() if self.start_time else None,
        }
        try:
            with open(self._state_file(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def _load_state(self):
        if self._state_file().exists():
            try:
                with open(self._state_file()) as f:
                    state = json.load(f)
                self.active_orders = state.get("active_orders", {})
                self.filled_tickers = set(state.get("filled_tickers", []))
                self.orders_placed = state.get("orders_placed", 0)
                self.orders_filled = state.get("orders_filled", 0)
                self.orders_cancelled = state.get("orders_cancelled", 0)
            except Exception:
                pass

        if self._log_file().exists():
            try:
                with open(self._log_file()) as f:
                    self.trade_log = json.load(f)
            except Exception:
                pass

    # =========================================================================
    # Status / One-Shot
    # =========================================================================

    def print_status(self):
        """Print current positions and balance from the API."""
        env = "DEMO" if self.config.demo else "PRODUCTION"
        print(f"\n{'='*70}")
        print(f"Live Trader Status ({env})")
        print(f"{'='*70}")

        try:
            balance = self._get_api_balance()
            print(f"Balance: ${balance:,.2f}")
        except Exception as e:
            print(f"Balance: Error - {e}")

        try:
            positions = self._get_api_positions()
            active = [p for p in positions if p.get("position", 0) != 0]

            if active:
                print(f"\nOpen positions ({len(active)}):")
                for p in active:
                    ticker = p.get("ticker", "?")
                    pos = p.get("position", 0)
                    side = "YES" if pos > 0 else "NO"
                    qty = abs(pos)
                    meta = self.market_meta.get(ticker, {})
                    strike = meta.get("floor_strike", "?")
                    asset = meta.get("asset", "?")
                    print(f"  [{asset}] {ticker} {side} {qty}x "
                          f"(strike=${strike})")
            else:
                print("\nNo open positions")
        except Exception as e:
            print(f"Positions: Error - {e}")

        try:
            resp = self.kalshi.get_orders(status="resting")
            orders = resp.get("orders", [])
            if orders:
                print(f"\nResting orders ({len(orders)}):")
                for o in orders:
                    print(f"  {o.get('ticker')} {o.get('side')} {o.get('remaining_count')}x "
                          f"@ {o.get('yes_price') or o.get('no_price')}c "
                          f"[{o.get('order_id', '')[:8]}...]")
            else:
                print("\nNo resting orders")
        except Exception as e:
            print(f"Orders: Error - {e}")

        # Local stats
        print(f"\nSession stats:")
        print(f"  Orders placed: {self.orders_placed}")
        print(f"  Orders filled: {self.orders_filled}")
        print(f"  Orders cancelled: {self.orders_cancelled}")
        print(f"{'='*70}\n")

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def run(self):
        self._running = True
        self.start_time = datetime.now(timezone.utc)

        env = "DEMO" if self.config.demo else "PRODUCTION"
        print(f"\n{'='*70}")
        print(f"Live Trader - {env} Environment")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Assets: {', '.join(self.config.assets)}")
        print(f"Strategy: V3 IV Surface (sigma>{self.config.min_sigma}, "
              f"mispricing>{self.config.min_mispricing_cents}c, "
              f"kelly={self.config.kelly_fraction})")
        print(f"Scan interval: {self.config.scan_interval}s")
        print(f"Stale order timeout: {self.config.stale_order_seconds}s")
        print(f"Data dir: {self.config.data_dir}")
        print(f"{'='*70}")

        if not self.config.demo:
            print("\n  *** WARNING: RUNNING WITH REAL MONEY ***")
            print("  Press Ctrl+C within 10 seconds to abort...")
            await asyncio.sleep(10)

        # 1. Initial data fetch
        print("\n[1] Fetching initial Deribit IV surfaces...")
        await self._refresh_deribit()

        print("\n[2] Fetching market metadata...")
        await self._refresh_markets()

        # 3. Show initial status
        self.print_status()

        # 4. Connect WS with reconnect loop
        while self._running:
            try:
                await self._run_ws_session()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._reconnect_count += 1
                print(f"  [WS] Disconnected: {e} (reconnect #{self._reconnect_count})")
                if self._running:
                    wait = min(5 * (2 ** min(self._reconnect_count - 1, 3)), 60)
                    print(f"  [WS] Reconnecting in {wait}s...")
                    await asyncio.sleep(wait)

        # Final status
        print("\nFinal status:")
        self.print_status()

    async def _run_ws_session(self):
        print("\n[3] Connecting to Kalshi WebSocket...")
        ws = await self._ws_connect()
        self._reconnect_count = 0
        self._last_message_time = datetime.now(timezone.utc)

        print("[4] Subscribing to ticker + orderbook_delta channels...")
        await self._ws_subscribe(["ticker"])
        await self._ws_subscribe(["orderbook_delta"])
        print("  Subscribed. Starting trading loop...\n")

        tasks = [
            asyncio.create_task(self._refresh_deribit_loop()),
            asyncio.create_task(self._refresh_markets_loop()),
            asyncio.create_task(self._trading_loop()),
            asyncio.create_task(self._stats_loop()),
            asyncio.create_task(self._health_check_loop()),
        ]

        try:
            async for msg in ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")

                    if msg_type == "ticker":
                        self._on_ticker(data.get("msg", {}))
                    elif msg_type == "orderbook_snapshot":
                        self._on_orderbook_snapshot(data.get("msg", {}))
                    elif msg_type == "orderbook_delta":
                        self._on_orderbook_delta(data.get("msg", {}))
                    elif msg_type == "subscribed":
                        channel = data.get("msg", {}).get("channel", "?")
                        print(f"  [WS] Confirmed: {channel}")
                    elif msg_type == "error":
                        print(f"  [WS] Error: {data}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"  [WS] Error: {ws.exception()}")
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    break
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            await self._ws_close()

    def shutdown(self):
        self._running = False


# =============================================================================
# CLI
# =============================================================================

async def main(args):
    config = LiveConfig(
        demo=not args.live,
        scan_interval=args.scan_interval,
        min_sigma=args.min_sigma,
        min_mispricing_cents=args.min_mispricing,
        kelly_fraction=args.kelly,
        assets=tuple(args.assets.split(",")),
    )

    trader = LiveTrader(config)

    if args.status:
        trader.print_status()
        return

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(
        __import__("signal").SIGINT,
        trader.shutdown,
    )
    loop.add_signal_handler(
        __import__("signal").SIGTERM,
        trader.shutdown,
    )

    await trader.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Live trader for Kalshi crypto options (V3 IV Surface strategy)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Use PRODUCTION environment (real money!). Default is demo.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current positions/balance and exit",
    )
    parser.add_argument(
        "--scan-interval", type=int, default=30,
        help="Seconds between signal scans (default: 30)",
    )
    parser.add_argument(
        "--min-sigma", type=float, default=1.75,
        help="Minimum sigma distance filter (default: 1.75)",
    )
    parser.add_argument(
        "--min-mispricing", type=int, default=5,
        help="Minimum mispricing in cents (default: 5)",
    )
    parser.add_argument(
        "--kelly", type=float, default=0.75,
        help="Kelly fraction (default: 0.75)",
    )
    parser.add_argument(
        "--assets", default="BTC,ETH",
        help="Comma-separated assets (default: BTC,ETH)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
