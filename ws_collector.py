#!/usr/bin/env python3
"""
WebSocket Market Data Collector for Kalshi Crypto Options

Streams real-time ticker and trade data via Kalshi's WebSocket API,
enriched with Deribit IV surface model probabilities.

Improvements over V3 REST polling (30s intervals):
- Captures every trade execution (not just periodic snapshots)
- Records last_trade_price even on illiquid contracts
- Sub-second bid/ask granularity
- Single persistent connection instead of thousands of REST calls/day

Data output (parquet files in data/ws_collector/snapshots/YYYYMMDD/):
- ws_tickers_HHMMSS.parquet: raw ticker updates from websocket
- ws_trades_HHMMSS.parquet: every trade execution
- enriched_HHMMSS.parquet: latest state per contract + model_prob from IV surface
- deribit_{asset}_HHMMSS.parquet: Deribit option chain snapshots
"""

import sys
sys.path.insert(0, "src")

import asyncio
import json
import time
import base64
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import aiohttp

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

from data.deribit_client import DeribitClient
from data.kalshi_client import KalshiClient
from models.iv_surface_2d import build_iv_surface_2d


CRYPTO_PREFIXES = ("KXBTCD", "KXETHD")
ASSET_FOR_PREFIX = {"KXBTCD": "BTC", "KXETHD": "ETH"}


class WSCollector:
    """
    WebSocket-based market data collector.

    Streams real-time Kalshi ticker and trade data, periodically enriches
    with Deribit IV surface model probabilities, and writes to parquet.
    """

    WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

    def __init__(
        self,
        data_dir: Path = Path("data/ws_collector"),
        deribit_interval: int = 60,
        market_refresh_interval: int = 300,
        flush_interval: int = 300,
        assets: tuple = ("BTC", "ETH"),
    ):
        self.data_dir = data_dir
        self.deribit_interval = deribit_interval
        self.market_refresh_interval = market_refresh_interval
        self.flush_interval = flush_interval
        self.assets = assets

        # API clients
        self.deribit = DeribitClient()
        self.kalshi_rest = KalshiClient()

        # Auth
        self.api_key_id = os.getenv("KALSHI_API_KEY_ID")
        self.private_key = None
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if key_path:
            with open(key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )

        # Data buffers
        self.ticker_buffer: List[dict] = []
        self.trade_buffer: List[dict] = []

        # Latest ticker state per market (for enriched snapshots)
        self.latest_ticker: Dict[str, dict] = {}

        # Market metadata cache: ticker -> {floor_strike, close_time, event_ticker, status}
        self.market_meta: Dict[str, dict] = {}

        # IV surface state
        self.iv_surfaces: Dict[str, object] = {}
        self.spots: Dict[str, float] = {}

        # Orderbook depth cache: ticker -> {yes_levels: [...], no_levels: [...], fetched_at: str}
        self.orderbook_cache: Dict[str, dict] = {}
        self.orderbook_interval = flush_interval  # sync with flush cycle

        # Stats
        self.ticker_count = 0
        self.trade_count = 0
        self.start_time: Optional[datetime] = None

        # WebSocket state
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._sub_id = 0
        self._running = False
        self._subscribed_tickers: set = set()

        # Connection health tracking
        self._last_message_time: Optional[datetime] = None
        self._reconnect_count = 0
        self._connection_log_path = self.data_dir / "connection_log.txt"

    def _log_connection_event(self, event: str, details: str = ""):
        """Log connection events to both console and file."""
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        msg = f"[{timestamp}] {event}"
        if details:
            msg += f" - {details}"
        print(f"  [Connection] {msg}")

        # Append to log file
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self._connection_log_path, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"  [Connection] Failed to write log: {e}")

    # =========================================================================
    # WebSocket Auth
    # =========================================================================

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        if self.private_key is None:
            raise RuntimeError(
                "No private key loaded. Set KALSHI_PRIVATE_KEY_PATH environment variable."
            )
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
            self.WS_URL,
            headers=headers,
            heartbeat=30,
        )
        return self._ws

    async def _ws_subscribe(self, channels: List[str], market_tickers: List[str] = None):
        self._sub_id += 1
        msg = {
            "id": self._sub_id,
            "cmd": "subscribe",
            "params": {"channels": channels},
        }
        if market_tickers:
            msg["params"]["market_tickers"] = market_tickers
        await self._ws.send_json(msg)

    async def _ws_update_subscription(self, sid: int, action: str, market_tickers: List[str]):
        """Add or remove tickers from an existing subscription."""
        self._sub_id += 1
        msg = {
            "id": self._sub_id,
            "cmd": "update_subscription",
            "params": {
                "sids": [sid],
                "action": action,
                "market_tickers": market_tickers,
            },
        }
        await self._ws.send_json(msg)

    async def _ws_close(self):
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    # =========================================================================
    # Message Handlers
    # =========================================================================

    def _on_ticker(self, msg: dict):
        ticker = msg.get("market_ticker", "")

        # Filter to crypto markets only
        if not any(ticker.startswith(p) for p in CRYPTO_PREFIXES):
            return

        received_at = datetime.now(timezone.utc)
        self._last_message_time = received_at

        # Get market metadata
        meta = self.market_meta.get(ticker, {})
        asset = meta.get("asset")

        # Compute model probability using cached IV surface and spot
        model_data = self._compute_model_prob(ticker)
        model_prob = model_data.get("model_prob")
        sigma_distance = model_data.get("sigma_distance")

        # Compute mispricing if we have model_prob and prices
        yes_ask = msg.get("yes_ask")
        yes_bid = msg.get("yes_bid")
        mispricing_yes = None
        mispricing_no = None
        if model_prob is not None and yes_ask is not None and yes_ask > 0:
            model_yes_cents = model_prob * 100
            mispricing_yes = model_yes_cents - yes_ask
        if model_prob is not None and yes_bid is not None and yes_bid > 0:
            model_no_cents = (1 - model_prob) * 100
            no_ask = 100 - yes_bid
            mispricing_no = model_no_cents - no_ask

        # Get current spot price for reference
        spot = self.spots.get(asset) if asset else None

        record = {
            "received_at": received_at.isoformat(),
            "market_ticker": ticker,
            "asset": asset,
            "strike": meta.get("floor_strike"),
            "close_time": meta.get("close_time"),
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "yes_bid_dollars": msg.get("yes_bid_dollars"),
            "yes_ask_dollars": msg.get("yes_ask_dollars"),
            "last_price": msg.get("price"),
            "price_dollars": msg.get("price_dollars"),
            "volume": msg.get("volume"),
            "volume_fp": msg.get("volume_fp"),
            "open_interest": msg.get("open_interest"),
            "open_interest_fp": msg.get("open_interest_fp"),
            "dollar_volume": msg.get("dollar_volume"),
            "dollar_open_interest": msg.get("dollar_open_interest"),
            "ts": msg.get("ts"),
            # Enrichment fields
            "spot_price": spot,
            "model_prob": model_prob,
            "sigma_distance": sigma_distance,
            "mispricing_yes": mispricing_yes,
            "mispricing_no": mispricing_no,
        }

        self.ticker_buffer.append(record)
        self.latest_ticker[ticker] = record
        self.ticker_count += 1

    def _on_trade(self, msg: dict):
        ticker = msg.get("market_ticker", "")
        if not any(ticker.startswith(p) for p in CRYPTO_PREFIXES):
            return

        received_at = datetime.now(timezone.utc)

        record = {
            "received_at": received_at.isoformat(),
            "trade_id": msg.get("trade_id"),
            "market_ticker": ticker,
            "yes_price": msg.get("yes_price"),
            "no_price": msg.get("no_price"),
            "yes_price_dollars": msg.get("yes_price_dollars"),
            "no_price_dollars": msg.get("no_price_dollars"),
            "count": msg.get("count"),
            "count_fp": msg.get("count_fp"),
            "taker_side": msg.get("taker_side"),
            "ts": msg.get("ts"),
        }

        self.trade_buffer.append(record)
        self.trade_count += 1

    # =========================================================================
    # Periodic REST Fetches
    # =========================================================================

    async def _refresh_markets_loop(self):
        while self._running:
            try:
                await self._refresh_markets()
            except Exception as e:
                print(f"  [Markets] Error: {e}")
            await asyncio.sleep(self.market_refresh_interval)

    async def _refresh_markets(self):
        """Fetch active crypto markets from Kalshi REST API and update metadata cache."""
        loop = asyncio.get_event_loop()

        new_tickers = set()
        for asset in self.assets:
            series = f"KX{asset}D"
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda s=series: self.kalshi_rest.get_markets(
                        series_ticker=f"KX{asset}D", status="open", limit=1000
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
                        new_tickers.add(ticker)
            except Exception as e:
                print(f"  [Markets] Error fetching {asset}: {e}")

        print(f"  [Markets] Tracking {len(self.market_meta)} markets ({len(new_tickers)} active)")

    async def _refresh_deribit_loop(self):
        while self._running:
            try:
                await self._refresh_deribit()
            except Exception as e:
                print(f"  [Deribit] Error: {e}")
            await asyncio.sleep(self.deribit_interval)

    async def _refresh_deribit(self):
        """Fetch Deribit option chains and rebuild IV surfaces."""
        loop = asyncio.get_event_loop()

        for asset in self.assets:
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
                    print(
                        f"  [Deribit] {asset} spot=${spot:,.2f}, "
                        f"IV grid: {bounds['min_time_hours']:.0f}h-{bounds['max_time_hours']:.0f}h"
                    )

                    # Save Deribit snapshot
                    now = datetime.now(timezone.utc)
                    time_str = now.strftime("%H%M%S")
                    chain_copy = chain.copy()
                    chain_copy["snapshot_time"] = now.isoformat()
                    chain_copy["spot_price"] = spot
                    path = self._snapshots_dir() / f"deribit_{asset}_{time_str}.parquet"
                    await loop.run_in_executor(
                        None, lambda df=chain_copy, p=path: df.to_parquet(p, index=False)
                    )
            except Exception as e:
                print(f"  [Deribit] Error fetching {asset}: {e}")

    # =========================================================================
    # Orderbook Depth
    # =========================================================================

    async def _refresh_orderbooks_loop(self):
        while self._running:
            try:
                await self._refresh_orderbooks()
            except Exception as e:
                print(f"  [Orderbook] Error: {e}")
            await asyncio.sleep(self.orderbook_interval)

    async def _refresh_orderbooks(self):
        """Fetch orderbook depth for markets within the IV grid."""
        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)

        # Only fetch for markets that have model_prob (within IV grid)
        candidates = []
        for ticker in self.latest_ticker:
            model = self._compute_model_prob(ticker)
            if model.get("model_prob") is not None:
                candidates.append(ticker)

        if not candidates:
            return

        fetched = 0
        errors = 0
        for ticker in candidates:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda t=ticker: self.kalshi_rest.get_market_orderbook(t, depth=20),
                )
                book = result.get("orderbook", {})
                self.orderbook_cache[ticker] = {
                    "yes_levels": book.get("yes", []) or [],
                    "no_levels": book.get("no", []) or [],
                    "fetched_at": now.isoformat(),
                }
                fetched += 1
                await asyncio.sleep(0.15)  # rate limit: ~7 req/s
            except Exception:
                errors += 1

        print(f"  [Orderbook] Fetched {fetched}/{len(candidates)} books "
              f"({errors} errors)")

    def _get_orderbook_depth(self, ticker: str) -> dict:
        """
        Extract depth summary from cached orderbook.

        Returns dict with:
          yes_best_bid_depth: contracts available at best yes bid
          no_best_bid_depth: contracts available at best no bid
          yes_total_depth: total yes bid depth across all levels
          no_total_depth: total no bid depth across all levels
          yes_levels: number of price levels with bids
          no_levels: number of price levels with bids
        """
        cache = self.orderbook_cache.get(ticker)
        if not cache:
            return {
                "yes_best_bid_depth": None,
                "no_best_bid_depth": None,
                "yes_total_depth": None,
                "no_total_depth": None,
                "yes_levels": None,
                "no_levels": None,
            }

        yes_levels = cache.get("yes_levels", [])
        no_levels = cache.get("no_levels", [])

        yes_best = yes_levels[0][1] if yes_levels else None
        no_best = no_levels[0][1] if no_levels else None
        yes_total = sum(qty for _, qty in yes_levels) if yes_levels else None
        no_total = sum(qty for _, qty in no_levels) if no_levels else None

        return {
            "yes_best_bid_depth": yes_best,
            "no_best_bid_depth": no_best,
            "yes_total_depth": yes_total,
            "no_total_depth": no_total,
            "yes_levels": len(yes_levels),
            "no_levels": len(no_levels),
        }

    # =========================================================================
    # Enrichment
    # =========================================================================

    def _compute_model_prob(self, ticker: str) -> dict:
        """Compute model_prob and sigma_distance for a market using IV surface."""
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

    # =========================================================================
    # Storage
    # =========================================================================

    def _snapshots_dir(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        d = self.data_dir / "snapshots" / date_str
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def _flush_loop(self):
        while self._running:
            await asyncio.sleep(self.flush_interval)
            try:
                await self._flush_buffers()
            except Exception as e:
                print(f"  [Flush] Error: {e}")

    async def _flush_buffers(self):
        """Write buffered data to parquet and create enriched snapshot."""
        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)
        time_str = now.strftime("%H%M%S")
        snapshot_dir = self._snapshots_dir()

        # Flush ticker buffer
        if self.ticker_buffer:
            data = self.ticker_buffer.copy()
            self.ticker_buffer.clear()
            df = pd.DataFrame(data)
            path = snapshot_dir / f"ws_tickers_{time_str}.parquet"
            await loop.run_in_executor(None, lambda: df.to_parquet(path, index=False))
            print(f"  [Flush] {len(df)} ticker updates -> {path.name}")

        # Flush trade buffer
        if self.trade_buffer:
            data = self.trade_buffer.copy()
            self.trade_buffer.clear()
            df = pd.DataFrame(data)
            path = snapshot_dir / f"ws_trades_{time_str}.parquet"
            await loop.run_in_executor(None, lambda: df.to_parquet(path, index=False))
            print(f"  [Flush] {len(df)} trades -> {path.name}")

        # Write enriched snapshot
        await self._write_enriched_snapshot(time_str, snapshot_dir)

    async def _write_enriched_snapshot(self, time_str: str, snapshot_dir: Path):
        """Combine latest ticker state + market metadata + model probabilities."""
        if not self.latest_ticker:
            return

        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)

        rows = []
        for ticker, ticker_data in self.latest_ticker.items():
            meta = self.market_meta.get(ticker, {})
            model = self._compute_model_prob(ticker)

            asset = meta.get("asset")
            strike = meta.get("floor_strike")
            close_time = meta.get("close_time")

            # Parse yes_bid/yes_ask from either cents or dollar fields
            yes_bid = ticker_data.get("yes_bid")
            yes_ask = ticker_data.get("yes_ask")
            last_price = ticker_data.get("last_price")

            # Compute mispricing if we have model_prob and market prices
            model_prob = model.get("model_prob")
            mispricing_yes = None
            mispricing_no = None
            if model_prob is not None:
                model_yes_cents = model_prob * 100
                if yes_ask is not None and yes_ask > 0:
                    mispricing_yes = model_yes_cents - yes_ask
                no_ask = (100 - yes_bid) if yes_bid is not None and yes_bid > 0 else None
                if no_ask is not None and no_ask > 0:
                    mispricing_no = (1 - model_prob) * 100 - no_ask

            hours_to_settlement = None
            if close_time:
                try:
                    settlement = pd.to_datetime(close_time, utc=True)
                    hours_to_settlement = (settlement - now).total_seconds() / 3600
                except Exception:
                    pass

            rows.append({
                "snapshot_time": now.isoformat(),
                "market_ticker": ticker,
                "asset": asset,
                "strike": strike,
                "close_time": close_time,
                "hours_to_settlement": hours_to_settlement,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "last_price": last_price,
                "yes_bid_dollars": ticker_data.get("yes_bid_dollars"),
                "yes_ask_dollars": ticker_data.get("yes_ask_dollars"),
                "price_dollars": ticker_data.get("price_dollars"),
                "volume": ticker_data.get("volume"),
                "open_interest": ticker_data.get("open_interest"),
                "spot_price": self.spots.get(asset),
                "model_prob": model_prob,
                "sigma_distance": model.get("sigma_distance"),
                "mispricing_yes": mispricing_yes,
                "mispricing_no": mispricing_no,
                **self._get_orderbook_depth(ticker),
            })

        if rows:
            df = pd.DataFrame(rows)
            path = snapshot_dir / f"enriched_{time_str}.parquet"
            await loop.run_in_executor(None, lambda: df.to_parquet(path, index=False))
            n_with_model = df["model_prob"].notna().sum()
            n_with_price = df["last_price"].notna().sum()
            print(
                f"  [Flush] Enriched snapshot: {len(df)} markets, "
                f"{n_with_model} with model_prob, {n_with_price} with last_price"
            )

    # =========================================================================
    # Stats
    # =========================================================================

    async def _stats_loop(self):
        while self._running:
            await asyncio.sleep(60)
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            mins = elapsed / 60
            print(
                f"\n  [Stats] {mins:.0f}m elapsed | "
                f"Tickers: {self.ticker_count} ({self.ticker_count / max(mins, 1):.0f}/min) | "
                f"Trades: {self.trade_count} ({self.trade_count / max(mins, 1):.0f}/min) | "
                f"Markets: {len(self.latest_ticker)} tracked, "
                f"{len(self.market_meta)} in metadata cache\n"
            )

    async def _health_check_loop(self):
        """Monitor connection health - log warning if no messages received."""
        stale_threshold = 120  # seconds without messages before warning
        while self._running:
            await asyncio.sleep(30)
            if self._last_message_time:
                since_last = (datetime.now(timezone.utc) - self._last_message_time).total_seconds()
                if since_last > stale_threshold:
                    self._log_connection_event(
                        "STALE_CONNECTION",
                        f"No messages for {since_last:.0f}s (threshold: {stale_threshold}s)"
                    )

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def run(self):
        self._running = True
        self.start_time = datetime.now(timezone.utc)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        print(f"{'=' * 70}")
        print(f"WebSocket Market Data Collector")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Assets: {', '.join(self.assets)}")
        print(f"Deribit refresh: every {self.deribit_interval}s")
        print(f"Market metadata refresh: every {self.market_refresh_interval}s")
        print(f"Flush to disk: every {self.flush_interval}s")
        print(f"Orderbook depth: every {self.orderbook_interval}s (markets in IV grid only)")
        print(f"Data dir: {self.data_dir}")
        print(f"{'=' * 70}")

        # 1. Initial data fetch
        print("\n[1] Initial Deribit fetch...")
        await self._refresh_deribit()

        print("\n[2] Initial market metadata fetch...")
        await self._refresh_markets()

        # Log startup
        self._log_connection_event("STARTUP", f"Collector started, assets={self.assets}")

        # 2. Connect websocket with reconnect loop
        while self._running:
            try:
                await self._run_websocket_session()
            except asyncio.CancelledError:
                self._log_connection_event("SHUTDOWN", "Cancelled")
                break
            except Exception as e:
                self._reconnect_count += 1
                self._log_connection_event(
                    "DISCONNECTED",
                    f"Error: {e}, reconnect #{self._reconnect_count}"
                )
                if self._running:
                    # Exponential backoff: 5s, 10s, 20s, max 60s
                    wait_time = min(5 * (2 ** min(self._reconnect_count - 1, 3)), 60)
                    self._log_connection_event("RECONNECTING", f"Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)

        # Final flush
        print("\nFinal flush...")
        await self._flush_buffers()

        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        print(f"\nSession complete: {elapsed / 60:.1f} minutes")
        print(f"Total ticker updates: {self.ticker_count}")
        print(f"Total trades: {self.trade_count}")
        print(f"Markets tracked: {len(self.latest_ticker)}")

    async def _run_websocket_session(self):
        """Single websocket session (reconnects on disconnect)."""
        print("\n[3] Connecting to Kalshi WebSocket...")
        ws = await self._ws_connect()
        self._log_connection_event("CONNECTED", f"Reconnect count was {self._reconnect_count}")
        self._reconnect_count = 0  # Reset on successful connection
        self._last_message_time = datetime.now(timezone.utc)

        # Subscribe to all ticker and trade updates (filter in handler)
        print("[4] Subscribing to ticker + trade channels (all markets)...")
        await self._ws_subscribe(["ticker"])
        await self._ws_subscribe(["trade"])
        print("  Subscribed. Listening for messages...\n")

        # Start background tasks
        tasks = [
            asyncio.create_task(self._refresh_deribit_loop()),
            asyncio.create_task(self._refresh_markets_loop()),
            asyncio.create_task(self._refresh_orderbooks_loop()),
            asyncio.create_task(self._flush_loop()),
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
                    elif msg_type == "trade":
                        self._on_trade(data.get("msg", {}))
                    elif msg_type == "subscribed":
                        sid = data.get("sid")
                        channel = data.get("msg", {}).get("channel", "?")
                        print(f"  [WS] Confirmed subscription: {channel} (sid={sid})")
                    elif msg_type == "error":
                        print(f"  [WS] Error: {data}")
                        self._log_connection_event("WS_ERROR", str(data))

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self._log_connection_event("WS_ERROR", str(ws.exception()))
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    self._log_connection_event("CLOSED_BY_SERVER", "Connection closed by server")
                    break
        finally:
            for t in tasks:
                t.cancel()
            # Wait for task cancellation
            for t in tasks:
                try:
                    await t
                except asyncio.CancelledError:
                    pass
            await self._ws_close()

    def shutdown(self):
        self._running = False


async def main(args):
    collector = WSCollector(
        data_dir=Path(args.data_dir),
        deribit_interval=args.deribit_interval,
        market_refresh_interval=args.market_refresh,
        flush_interval=args.flush_interval,
        assets=tuple(args.assets.split(",")),
    )

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(
        __import__("signal").SIGINT,
        collector.shutdown,
    )
    loop.add_signal_handler(
        __import__("signal").SIGTERM,
        collector.shutdown,
    )

    await collector.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="WebSocket-based Kalshi market data collector"
    )
    parser.add_argument(
        "--data-dir", default="data/ws_collector",
        help="Output directory (default: data/ws_collector)",
    )
    parser.add_argument(
        "--deribit-interval", type=int, default=60,
        help="Deribit IV surface refresh interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--market-refresh", type=int, default=300,
        help="Kalshi market metadata refresh interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--flush-interval", type=int, default=300,
        help="Flush to parquet interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--assets", default="BTC,ETH",
        help="Comma-separated assets to track (default: BTC,ETH)",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
