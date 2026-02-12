"""Configuration settings for the trading system."""

# API Endpoints
DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"
KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

# Supported assets
SUPPORTED_ASSETS = ["BTC", "ETH"]

# Kalshi API credentials (set via environment variables)
import os
KALSHI_EMAIL = os.getenv("KALSHI_EMAIL", "")
KALSHI_PASSWORD = os.getenv("KALSHI_PASSWORD", "")

# Data storage paths
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed"
