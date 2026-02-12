"""
Deribit API client for fetching options data.

Deribit public API docs: https://docs.deribit.com/
"""

import requests
from datetime import datetime, timezone
from typing import Optional
import pandas as pd


class DeribitClient:
    """Client for interacting with Deribit's public API."""

    BASE_URL = "https://www.deribit.com/api/v2"

    def __init__(self):
        self.session = requests.Session()

    def _request(self, method: str, params: dict = None) -> dict:
        """Make a request to the Deribit API."""
        url = f"{self.BASE_URL}/public/{method}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "result" in data:
            return data["result"]
        raise ValueError(f"Unexpected response: {data}")

    def get_currencies(self) -> list:
        """Get list of supported currencies."""
        return self._request("get_currencies")

    def get_index_price(self, currency: str) -> dict:
        """Get current index price for a currency."""
        return self._request("get_index_price", {"index_name": f"{currency.lower()}_usd"})

    def get_instruments(self, currency: str, kind: str = "option", expired: bool = False) -> list:
        """
        Get all instruments for a currency.

        Args:
            currency: BTC or ETH
            kind: 'option', 'future', or 'spot'
            expired: Include expired instruments
        """
        params = {
            "currency": currency.upper(),
            "kind": kind,
            "expired": str(expired).lower()
        }
        return self._request("get_instruments", params)

    def get_order_book(self, instrument_name: str, depth: int = 10) -> dict:
        """Get order book for an instrument."""
        return self._request("get_order_book", {
            "instrument_name": instrument_name,
            "depth": depth
        })

    def get_book_summary_by_currency(self, currency: str, kind: str = "option") -> list:
        """Get book summaries for all instruments of a currency."""
        return self._request("get_book_summary_by_currency", {
            "currency": currency.upper(),
            "kind": kind
        })

    def get_option_chain(self, currency: str) -> pd.DataFrame:
        """
        Get the full option chain for a currency as a DataFrame.

        Returns DataFrame with columns:
            - instrument_name, strike, expiry, option_type (call/put)
            - bid, ask, mark_price, underlying_price
            - iv (implied volatility), delta, gamma, vega, theta
        """
        instruments = self.get_instruments(currency, kind="option")
        summaries = self.get_book_summary_by_currency(currency, kind="option")

        # Create lookup by instrument name
        summary_map = {s["instrument_name"]: s for s in summaries}

        rows = []
        for inst in instruments:
            name = inst["instrument_name"]
            summary = summary_map.get(name, {})

            # Parse instrument name: BTC-28MAR25-100000-C
            parts = name.split("-")
            if len(parts) >= 4:
                option_type = "call" if parts[-1] == "C" else "put"
                strike = float(parts[-2])
                expiry_str = parts[1]
            else:
                continue

            rows.append({
                "instrument_name": name,
                "currency": currency,
                "strike": strike,
                "expiry_str": expiry_str,
                "expiry_timestamp": inst.get("expiration_timestamp"),
                "option_type": option_type,
                "bid": summary.get("bid_price"),
                "ask": summary.get("ask_price"),
                "mark_price": summary.get("mark_price"),
                "mark_iv": summary.get("mark_iv"),
                "underlying_price": summary.get("underlying_price"),
                "volume_24h": summary.get("volume"),
                "open_interest": summary.get("open_interest"),
                "timestamp": datetime.now(timezone.utc)
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["expiry_datetime"] = pd.to_datetime(df["expiry_timestamp"], unit="ms", utc=True)
        return df

    def get_options_for_expiry(self, currency: str, expiry_str: str) -> pd.DataFrame:
        """
        Get options for a specific expiry date.

        Args:
            currency: BTC or ETH
            expiry_str: Expiry string like '28MAR25'
        """
        chain = self.get_option_chain(currency)
        if chain.empty:
            return chain
        return chain[chain["expiry_str"] == expiry_str.upper()]

    def get_historical_volatility(self, currency: str) -> dict:
        """Get historical volatility for a currency."""
        return self._request("get_historical_volatility", {"currency": currency.upper()})


def fetch_and_save_snapshot(currency: str, output_path: str):
    """Fetch current option chain and save to parquet."""
    client = DeribitClient()
    df = client.get_option_chain(currency)
    if not df.empty:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{output_path}/{currency}_options_{timestamp}.parquet"
        df.to_parquet(filename, index=False)
        print(f"Saved {len(df)} options to {filename}")
        return filename
    return None


if __name__ == "__main__":
    # Test the client
    client = DeribitClient()

    # Get current BTC index price
    price = client.get_index_price("BTC")
    print(f"BTC Index Price: ${price['index_price']:,.2f}")

    # Get option chain
    chain = client.get_option_chain("BTC")
    print(f"\nFetched {len(chain)} BTC options")
    print(f"Expiries: {sorted(chain['expiry_str'].unique())}")
    print(f"\nSample data:")
    print(chain[["instrument_name", "strike", "option_type", "bid", "ask", "mark_iv"]].head(10))
