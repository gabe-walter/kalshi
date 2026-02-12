#!/usr/bin/env python3
"""
Data Recorder for Backtesting

Continuously snapshots:
1. Kalshi KXBTCD markets (all bid/ask/volume)
2. Deribit BTC option chain (all strikes)

Saves to parquet files for later backtesting.

Usage:
    python data_recorder.py --interval 300  # Snapshot every 5 minutes
    python data_recorder.py --once          # Single snapshot
"""

import sys
sys.path.insert(0, "src")

import os
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from data.deribit_client import DeribitClient
from data.kalshi_client import KalshiClient


DATA_DIR = Path("data/snapshots")


def get_kalshi_snapshot(kalshi: KalshiClient) -> pd.DataFrame:
    """Snapshot all KXBTCD markets with bid/ask/volume."""

    now = datetime.now(timezone.utc)
    snapshot_time = now.isoformat()

    # Build event tickers for the 3 active settlements we care about:
    # 1. Next hourly, 2. Daily 5pm EST, 3. Weekly Friday 5pm EST
    from datetime import timedelta

    est_offset = timedelta(hours=-5)
    now_est = now + est_offset

    year_short = str(now.year)[-2:]
    month_str = now.strftime("%b").upper()

    # Next hour EST
    next_hour_est = (now_est.hour + 1) % 24
    hourly_day = now_est.day if next_hour_est > now_est.hour else now_est.day + 1
    hourly_event = f"KXBTCD-{year_short}{month_str}{hourly_day:02d}{next_hour_est:02d}"

    # 5pm EST today
    daily_event = f"KXBTCD-{year_short}{month_str}{now_est.day:02d}17"

    # Friday 5pm EST
    days_until_friday = (4 - now.weekday()) % 7
    if days_until_friday == 0 and now_est.hour >= 17:
        days_until_friday = 7
    friday = now + timedelta(days=days_until_friday)
    weekly_event = f"KXBTCD-{year_short}{month_str}{friday.day:02d}17"

    events_to_fetch = list(dict.fromkeys([hourly_event, daily_event, weekly_event]))

    all_markets = []
    for event_ticker in events_to_fetch:
        try:
            markets = kalshi.get_markets(event_ticker=event_ticker, limit=200)
            for m in markets.get('markets', []):
                if m.get('status') == 'active':
                    all_markets.append({
                        'snapshot_time': snapshot_time,
                        'ticker': m.get('ticker'),
                        'event_ticker': event_ticker,
                        'strike': m.get('floor_strike'),
                        'close_time': m.get('close_time'),
                        'yes_bid': m.get('yes_bid'),
                        'yes_ask': m.get('yes_ask'),
                        'no_bid': m.get('no_bid'),
                        'no_ask': m.get('no_ask'),
                        'volume': m.get('volume'),
                        'volume_24h': m.get('volume_24h'),
                        'open_interest': m.get('open_interest'),
                        'last_price': m.get('last_price'),
                    })
            time.sleep(0.2)  # Small delay to avoid rate limits
        except Exception as e:
            print(f"    Warning: Failed to fetch {event_ticker}: {e}")
            continue

    return pd.DataFrame(all_markets)


def get_deribit_snapshot(deribit: DeribitClient) -> pd.DataFrame:
    """Snapshot full BTC option chain."""

    chain = deribit.get_option_chain("BTC")

    # Add snapshot timestamp
    chain['snapshot_time'] = datetime.now(timezone.utc).isoformat()

    # Get spot price
    spot_data = deribit.get_index_price("BTC")
    chain['spot_price'] = spot_data.get('index_price')

    return chain


def save_snapshot(df: pd.DataFrame, source: str, timestamp: datetime):
    """Save snapshot to parquet file."""

    date_str = timestamp.strftime("%Y%m%d")
    time_str = timestamp.strftime("%H%M%S")

    dir_path = DATA_DIR / source / date_str
    dir_path.mkdir(parents=True, exist_ok=True)

    file_path = dir_path / f"{source}_{date_str}_{time_str}.parquet"
    df.to_parquet(file_path, index=False)

    return file_path


def run_snapshot():
    """Run a single snapshot of both sources."""

    timestamp = datetime.now(timezone.utc)
    print(f"\n[{timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}] Taking snapshot...")

    kalshi = KalshiClient()
    deribit = DeribitClient()

    # Kalshi snapshot
    print("  Kalshi KXBTCD...", end=" ")
    try:
        kalshi_df = get_kalshi_snapshot(kalshi)
        if not kalshi_df.empty:
            path = save_snapshot(kalshi_df, "kalshi", timestamp)
            print(f"{len(kalshi_df)} markets → {path}")
        else:
            print("No data")
    except Exception as e:
        print(f"Error: {e}")

    # Deribit snapshot
    print("  Deribit options...", end=" ")
    try:
        deribit_df = get_deribit_snapshot(deribit)
        if not deribit_df.empty:
            path = save_snapshot(deribit_df, "deribit", timestamp)
            print(f"{len(deribit_df)} options → {path}")
        else:
            print("No data")
    except Exception as e:
        print(f"Error: {e}")

    return timestamp


def run_continuous(interval_seconds: int):
    """Run continuous snapshots at specified interval."""

    print(f"Starting data recorder (interval: {interval_seconds}s)")
    print(f"Data directory: {DATA_DIR.absolute()}")
    print("Press Ctrl+C to stop")

    while True:
        try:
            run_snapshot()
            print(f"  Next snapshot in {interval_seconds}s...")
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nStopping data recorder.")
            break
        except Exception as e:
            print(f"  Error during snapshot: {e}")
            print(f"  Retrying in {interval_seconds}s...")
            time.sleep(interval_seconds)


def show_stats():
    """Show statistics about collected data."""

    print("\n=== Data Collection Stats ===\n")

    for source in ["kalshi", "deribit"]:
        source_dir = DATA_DIR / source
        if not source_dir.exists():
            print(f"{source}: No data collected yet")
            continue

        files = list(source_dir.rglob("*.parquet"))
        if not files:
            print(f"{source}: No data collected yet")
            continue

        # Get date range
        dates = sorted(set(f.parent.name for f in files))
        total_size = sum(f.stat().st_size for f in files)

        print(f"{source}:")
        print(f"  Files: {len(files)}")
        print(f"  Date range: {dates[0]} to {dates[-1]}")
        print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")

        # Sample latest file
        latest = max(files, key=lambda f: f.name)
        df = pd.read_parquet(latest)
        print(f"  Latest snapshot: {len(df)} records")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record market data for backtesting")
    parser.add_argument("--interval", type=int, default=300,
                        help="Snapshot interval in seconds (default: 300 = 5 min)")
    parser.add_argument("--once", action="store_true",
                        help="Take a single snapshot and exit")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics about collected data")

    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.once:
        run_snapshot()
        print("\nDone.")
    else:
        run_continuous(args.interval)
