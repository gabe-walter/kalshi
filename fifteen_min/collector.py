#!/usr/bin/env python3
"""
15-Minute Up/Down Market WebSocket Collector

Streams real-time ticker data for 15-minute crypto up/down markets via WebSocket.
Tracks each window from open to close, calculating arb metrics in real-time.

Key metrics tracked:
- Opening prices when window starts
- All price updates (every tick)
- Min/max prices during window
- Whether arb opportunity existed (YES ask + NO ask < 100¢)
- Best arb profit available
"""

import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../src")

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


# 15-minute market prefixes
PREFIXES_15M = ("KXBTC15M", "KXETH15M", "KXSOL15M")


class FifteenMinCollector:
    """
    WebSocket-based collector for 15-minute up/down markets.
    """

    WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    REST_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        data_dir: Path = Path("data"),
        flush_interval: int = 60,
    ):
        self.data_dir = data_dir
        self.flush_interval = flush_interval

        # Auth
        self.api_key_id = os.getenv("KALSHI_API_KEY_ID")
        self.private_key = None
        key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        if key_path:
            with open(key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None, backend=default_backend()
                )

        # WebSocket state
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._sub_id = 0
        self._running = False

        # Connection health
        self._last_message_time: Optional[datetime] = None
        self._reconnect_count = 0

        # Data structures
        # window_key = ticker -> window data (one market per 15-min window)
        self.active_windows: Dict[str, dict] = {}
        self.completed_windows: List[dict] = []
        self.tick_buffer: List[dict] = []

        # Stats
        self.ticker_count = 0
        self.start_time: Optional[datetime] = None

        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # WebSocket Auth
    # =========================================================================

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        if self.private_key is None:
            raise RuntimeError("No private key loaded.")
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
    # Message Handlers
    # =========================================================================

    def _on_ticker(self, msg: dict):
        ticker = msg.get("market_ticker", "")

        # Filter to 15-minute markets only
        if not any(ticker.startswith(p) for p in PREFIXES_15M):
            return

        received_at = datetime.now(timezone.utc)
        self._last_message_time = received_at
        self.ticker_count += 1

        yes_bid = msg.get("yes_bid") or 0
        yes_ask = msg.get("yes_ask") or 0
        no_bid = 100 - yes_ask if yes_ask > 0 else 0
        no_ask = 100 - yes_bid if yes_bid > 0 else 0
        total_cost = yes_ask + no_ask if (yes_ask > 0 and no_ask > 0) else None

        # Initialize or update window
        if ticker not in self.active_windows:
            self._init_window(ticker, yes_bid, yes_ask, no_bid, no_ask, received_at)
        else:
            self._update_window(ticker, yes_bid, yes_ask, no_bid, no_ask, total_cost, received_at)

        # Buffer tick
        self.tick_buffer.append({
            "received_at": received_at,
            "ticker": ticker,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
            "total_cost": total_cost,
            "volume": msg.get("volume", 0),
        })

    def _init_window(self, ticker: str, yes_bid: int, yes_ask: int,
                     no_bid: int, no_ask: int, received_at: datetime):
        """Initialize tracking for a new 15-minute window."""
        total_cost = yes_ask + no_ask if (yes_ask > 0 and no_ask > 0) else None

        self.active_windows[ticker] = {
            "ticker": ticker,
            "series": ticker.split("-")[0],
            "first_seen": received_at.isoformat(),

            # Opening prices
            "open_yes_bid": yes_bid,
            "open_yes_ask": yes_ask,
            "open_no_bid": no_bid,
            "open_no_ask": no_ask,
            "open_total_cost": total_cost,

            # Track min/max
            "min_yes_ask": yes_ask if yes_ask > 0 else 999,
            "max_yes_ask": yes_ask,
            "min_no_ask": no_ask if no_ask > 0 else 999,
            "max_no_ask": no_ask,
            "min_total_cost": total_cost if total_cost else 999,

            # Arb tracking
            "arb_possible": total_cost is not None and total_cost < 100,
            "arb_best_cost": total_cost if (total_cost and total_cost < 100) else None,
            "arb_best_time": received_at.isoformat() if (total_cost and total_cost < 100) else None,

            # Tick count
            "tick_count": 1,
        }

        status = "ARB!" if (total_cost and total_cost < 100) else ""
        print(f"[{received_at.strftime('%H:%M:%S')}] NEW: {ticker} | "
              f"YES {yes_ask}¢, NO {no_ask}¢, Total {total_cost}¢ {status}")

    def _update_window(self, ticker: str, yes_bid: int, yes_ask: int,
                       no_bid: int, no_ask: int, total_cost: Optional[int],
                       received_at: datetime):
        """Update window tracking with new price data."""
        w = self.active_windows[ticker]
        w["tick_count"] += 1

        # Update min/max
        if yes_ask > 0:
            w["min_yes_ask"] = min(w["min_yes_ask"], yes_ask)
            w["max_yes_ask"] = max(w["max_yes_ask"], yes_ask)
        if no_ask > 0:
            w["min_no_ask"] = min(w["min_no_ask"], no_ask)
            w["max_no_ask"] = max(w["max_no_ask"], no_ask)
        if total_cost:
            w["min_total_cost"] = min(w["min_total_cost"], total_cost)

        # Check arb
        if total_cost and total_cost < 100:
            w["arb_possible"] = True
            if w["arb_best_cost"] is None or total_cost < w["arb_best_cost"]:
                w["arb_best_cost"] = total_cost
                w["arb_best_time"] = received_at.isoformat()
                print(f"[{received_at.strftime('%H:%M:%S')}] ARB: {ticker} | "
                      f"YES {yes_ask}¢ + NO {no_ask}¢ = {total_cost}¢ | "
                      f"Profit: {100 - total_cost}¢")

    # =========================================================================
    # Window Cleanup (check for closed markets)
    # =========================================================================

    async def _cleanup_loop(self):
        """Periodically check for markets that have closed."""
        while self._running:
            await asyncio.sleep(30)  # Check every 30 seconds

            now = datetime.now(timezone.utc)
            to_remove = []

            for ticker, window in self.active_windows.items():
                first_seen = datetime.fromisoformat(window["first_seen"].replace("Z", "+00:00"))
                age_minutes = (now - first_seen).total_seconds() / 60

                # 15-min markets should close after ~15 mins from open
                # Give some buffer for late messages
                if age_minutes > 20:
                    to_remove.append(ticker)

            for ticker in to_remove:
                window = self.active_windows.pop(ticker)
                self._finalize_window(window)

    def _finalize_window(self, window: dict):
        """Finalize a completed window."""
        window["closed_at"] = datetime.now(timezone.utc).isoformat()

        print(f"[CLOSED] {window['ticker']} | Ticks: {window['tick_count']} | "
              f"Open: YES {window['open_yes_ask']}¢ NO {window['open_no_ask']}¢ | "
              f"Range: YES {window['min_yes_ask']}-{window['max_yes_ask']}¢, "
              f"NO {window['min_no_ask']}-{window['max_no_ask']}¢")

        self.completed_windows.append(window)

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
                print(f"[Flush] Error: {e}")

    async def _flush_buffers(self):
        """Write buffered data to parquet."""
        loop = asyncio.get_event_loop()
        now = datetime.now(timezone.utc)
        time_str = now.strftime("%H%M%S")
        snapshot_dir = self._snapshots_dir()

        # Flush completed windows
        if self.completed_windows:
            data = self.completed_windows.copy()
            self.completed_windows.clear()
            df = pd.DataFrame(data)
            path = snapshot_dir / f"windows_{time_str}.parquet"
            await loop.run_in_executor(None, lambda: df.to_parquet(path, index=False))
            print(f"[Flush] {len(df)} windows -> {path.name}")

        # Flush ticks
        if self.tick_buffer:
            data = self.tick_buffer.copy()
            self.tick_buffer.clear()
            df = pd.DataFrame(data)
            path = snapshot_dir / f"ticks_{time_str}.parquet"
            await loop.run_in_executor(None, lambda: df.to_parquet(path, index=False))
            print(f"[Flush] {len(df)} ticks -> {path.name}")

    # =========================================================================
    # Stats
    # =========================================================================

    async def _stats_loop(self):
        while self._running:
            await asyncio.sleep(60)
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            mins = elapsed / 60
            print(
                f"\n[Stats] {mins:.0f}m | "
                f"Ticks: {self.ticker_count} ({self.ticker_count / max(mins, 1):.0f}/min) | "
                f"Active windows: {len(self.active_windows)}\n"
            )

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def run(self):
        self._running = True
        self.start_time = datetime.now(timezone.utc)

        print("=" * 60)
        print("15-Minute Market WebSocket Collector")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Tracking: {', '.join(PREFIXES_15M)}")
        print(f"Data dir: {self.data_dir}")
        print("=" * 60)

        while self._running:
            try:
                await self._run_websocket_session()
            except asyncio.CancelledError:
                print("[Shutdown] Cancelled")
                break
            except Exception as e:
                self._reconnect_count += 1
                print(f"[Disconnected] {e}, reconnect #{self._reconnect_count}")
                if self._running:
                    wait_time = min(5 * (2 ** min(self._reconnect_count - 1, 3)), 60)
                    print(f"[Reconnecting] Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)

        # Final flush
        print("\nFinal flush...")
        await self._flush_buffers()

        print(f"\nSession complete: {(datetime.now(timezone.utc) - self.start_time).total_seconds() / 60:.1f} min")
        print(f"Total ticks: {self.ticker_count}")

    async def _run_websocket_session(self):
        """Single websocket session."""
        print("\n[Connecting] Kalshi WebSocket...")
        ws = await self._ws_connect()
        print("[Connected]")
        self._reconnect_count = 0
        self._last_message_time = datetime.now(timezone.utc)

        # Subscribe to ticker channel
        print("[Subscribing] ticker channel...")
        await self._ws_subscribe(["ticker"])
        print("[Subscribed] Listening for 15-min market updates...\n")

        # Background tasks
        tasks = [
            asyncio.create_task(self._flush_loop()),
            asyncio.create_task(self._stats_loop()),
            asyncio.create_task(self._cleanup_loop()),
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
                    elif msg_type == "subscribed":
                        channel = data.get("msg", {}).get("channel", "?")
                        print(f"[WS] Confirmed: {channel}")
                    elif msg_type == "error":
                        print(f"[WS] Error: {data}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[WS] Error: {ws.exception()}")
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    print("[WS] Connection closed")
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


async def main():
    collector = FifteenMinCollector(
        data_dir=Path("data"),
        flush_interval=60,
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
    asyncio.run(main())
