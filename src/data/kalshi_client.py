"""
Kalshi API client for fetching market data.

Kalshi API docs: https://docs.kalshi.com/
"""

import requests
import time
import base64
from datetime import datetime, timezone
from typing import Optional
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
import pandas as pd
import os


class KalshiClient:
    """Client for interacting with Kalshi's API."""

    PROD_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"

    def __init__(self, api_key_id: str = None, private_key_path: str = None, demo: bool = False):
        """
        Initialize the Kalshi client.

        Args:
            api_key_id: Your Kalshi API key ID
            private_key_path: Path to your RSA private key file
            demo: Use demo environment if True
        """
        self.base_url = self.DEMO_BASE_URL if demo else self.PROD_BASE_URL
        self.api_key_id = api_key_id or os.getenv("KALSHI_API_KEY_ID")
        self.private_key = None

        if private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH"):
            key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")
            with open(key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )

        self.session = requests.Session()

    def _sign_request(self, method: str, path: str, timestamp: str) -> str:
        """Sign a request using RSA-PSS with SHA256 (Kalshi's required method)."""
        if not self.private_key or not self.api_key_id:
            return None

        # Message format: timestamp + method + path (path without query params)
        message = f"{timestamp}{method}{path}".encode('utf-8')
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def _request(self, method: str, endpoint: str, params: dict = None, data: dict = None) -> dict:
        """Make a request to the Kalshi API."""
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        # Add authentication if credentials are available
        if self.api_key_id and self.private_key:
            timestamp = str(int(time.time() * 1000))
            # Sign the full path (e.g., /trade-api/v2/markets), not just endpoint
            full_path = f"/trade-api/v2{endpoint}"
            signature = self._sign_request(method.upper(), full_path, timestamp)
            headers["KALSHI-ACCESS-KEY"] = self.api_key_id
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp

        response = self.session.request(method, url, params=params, json=data, headers=headers)
        response.raise_for_status()
        return response.json()

    def get_markets(
        self,
        status: str = None,
        series_ticker: str = None,
        event_ticker: str = None,
        limit: int = 100,
        cursor: str = None
    ) -> dict:
        """
        Get list of markets.

        Args:
            status: Filter by status ('unopened', 'open', 'closed', 'settled')
            series_ticker: Filter by series (e.g., 'KXBTC' for Bitcoin)
            event_ticker: Filter by event
            limit: Max results per page (max 1000)
            cursor: Pagination cursor
        """
        params = {"limit": limit}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor

        return self._request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        """Get details for a specific market."""
        return self._request("GET", f"/markets/{ticker}")

    def get_market_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Get orderbook for a market."""
        return self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_events(self, series_ticker: str = None, status: str = None, limit: int = 100) -> dict:
        """Get list of events."""
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        return self._request("GET", "/events", params=params)

    def get_event(self, event_ticker: str) -> dict:
        """Get details for a specific event."""
        return self._request("GET", f"/events/{event_ticker}")

    def get_series(self, series_ticker: str) -> dict:
        """Get details for a series."""
        return self._request("GET", f"/series/{series_ticker}")

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        period_minutes: int = 1,
        start_ts: int = None,
        end_ts: int = None
    ) -> dict:
        """
        Get candlestick data for a market.

        Args:
            series_ticker: Series ticker (e.g., 'KXBTC')
            ticker: Market ticker
            period_minutes: Candle period (1, 60, or 1440 minutes)
            start_ts: Start timestamp (seconds)
            end_ts: End timestamp (seconds)
        """
        params = {"period_interval": period_minutes}
        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts

        return self._request(
            "GET",
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params=params
        )

    def get_trades(self, ticker: str = None, limit: int = 100, cursor: str = None) -> dict:
        """Get recent trades."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets/trades", params=params)

    def get_crypto_above_below_markets(self, crypto: str = "BTC") -> pd.DataFrame:
        """
        Get all crypto above/below markets for a given cryptocurrency.

        Args:
            crypto: 'BTC' or 'ETH'

        Returns:
            DataFrame with market details including strike prices and expiries
        """
        # Kalshi uses series tickers like 'KXBTC' for Bitcoin above/below
        series_ticker = f"KX{crypto.upper()}"

        all_markets = []
        cursor = None

        while True:
            result = self.get_markets(series_ticker=series_ticker, cursor=cursor, limit=1000)
            markets = result.get("markets", [])
            all_markets.extend(markets)

            cursor = result.get("cursor")
            if not cursor or not markets:
                break

        if not all_markets:
            return pd.DataFrame()

        df = pd.DataFrame(all_markets)

        # Parse relevant fields
        if not df.empty and "close_time" in df.columns:
            df["close_datetime"] = pd.to_datetime(df["close_time"], utc=True)

        return df

    def get_market_history(self, ticker: str, series_ticker: str = None) -> pd.DataFrame:
        """
        Get price history for a market.

        Args:
            ticker: Market ticker
            series_ticker: Series ticker (will try to infer if not provided)
        """
        if not series_ticker:
            # Try to get series from market details
            market = self.get_market(ticker)
            series_ticker = market.get("market", {}).get("series_ticker")

        if not series_ticker:
            raise ValueError("Could not determine series_ticker")

        # Get candlesticks
        candles = self.get_candlesticks(series_ticker, ticker, period_minutes=1)

        if not candles.get("candles"):
            return pd.DataFrame()

        df = pd.DataFrame(candles["candles"])
        df["timestamp"] = pd.to_datetime(df["end_period_ts"], unit="s", utc=True)
        return df


def fetch_crypto_markets(crypto: str = "BTC", output_path: str = None):
    """Fetch current crypto above/below markets and optionally save."""
    client = KalshiClient()
    df = client.get_crypto_above_below_markets(crypto)

    if output_path and not df.empty:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{output_path}/{crypto}_kalshi_markets_{timestamp}.parquet"
        df.to_parquet(filename, index=False)
        print(f"Saved {len(df)} markets to {filename}")

    return df


if __name__ == "__main__":
    client = KalshiClient()

    # Get BTC above/below markets
    print("Fetching BTC above/below markets...")
    markets = client.get_crypto_above_below_markets("BTC")

    if not markets.empty:
        print(f"\nFound {len(markets)} BTC markets")
        print("\nSample columns:")
        print(markets.columns.tolist())
        print("\nSample data:")
        cols = ["ticker", "subtitle", "yes_bid", "yes_ask", "close_time"]
        available_cols = [c for c in cols if c in markets.columns]
        print(markets[available_cols].head(10))
    else:
        print("No markets found (may need API authentication)")
