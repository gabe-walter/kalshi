#!/usr/bin/env python3
"""
Analyze 15-minute market data.

Just loads and summarizes the raw data - strategy analysis comes later.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/snapshots")


def load_all_windows() -> pd.DataFrame:
    """Load all completed window data."""
    frames = []
    for parquet_file in DATA_DIR.rglob("windows_*.parquet"):
        df = pd.read_parquet(parquet_file)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_all_ticks() -> pd.DataFrame:
    """Load all raw tick data."""
    frames = []
    for parquet_file in DATA_DIR.rglob("ticks_*.parquet"):
        df = pd.read_parquet(parquet_file)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main():
    print("=" * 60)
    print("15-Minute Market Data Summary")
    print("=" * 60)

    windows = load_all_windows()
    ticks = load_all_ticks()

    print(f"\nData loaded:")
    print(f"  Windows: {len(windows)}")
    print(f"  Ticks: {len(ticks)}")

    if windows.empty:
        print("\nNo data yet. Run the collector first!")
        return

    print("\n" + "-" * 40)
    print("OPENING PRICES")
    print("-" * 40)
    print(f"  Mean YES ask at open: {windows['open_yes_ask'].mean():.1f}¢")
    print(f"  Mean NO ask at open: {windows['open_no_ask'].mean():.1f}¢")

    windows["open_total"] = windows["open_yes_ask"] + windows["open_no_ask"]
    print(f"  Mean total (spread): {windows['open_total'].mean():.1f}¢")

    # How often is open near 50/50?
    near_50 = windows[(windows["open_yes_ask"] >= 45) & (windows["open_yes_ask"] <= 55)]
    print(f"  Windows with YES open 45-55¢: {len(near_50)} ({len(near_50)/len(windows)*100:.1f}%)")

    print("\n" + "-" * 40)
    print("PRICE RANGES DURING WINDOWS")
    print("-" * 40)
    windows["yes_range"] = windows["max_yes_ask"] - windows["min_yes_ask"]
    windows["no_range"] = windows["max_no_ask"] - windows["min_no_ask"]
    print(f"  Mean YES range: {windows['yes_range'].mean():.1f}¢")
    print(f"  Mean NO range: {windows['no_range'].mean():.1f}¢")
    print(f"  Max YES range: {windows['yes_range'].max()}¢")
    print(f"  Max NO range: {windows['no_range'].max()}¢")

    print("\n" + "-" * 40)
    print("BY ASSET")
    print("-" * 40)
    for series in ["KXBTC15M", "KXETH15M", "KXSOL15M"]:
        asset = windows[windows["series"] == series]
        if asset.empty:
            continue
        name = series.replace("KX", "").replace("15M", "")
        print(f"\n  {name}: {len(asset)} windows")
        print(f"    Mean open YES: {asset['open_yes_ask'].mean():.1f}¢")
        print(f"    Mean YES range: {asset['yes_range'].mean():.1f}¢")

    print("\n" + "-" * 40)
    print("RECENT WINDOWS")
    print("-" * 40)
    for _, row in windows.tail(10).iterrows():
        ticker = row["ticker"]
        print(f"  {ticker}")
        print(f"    Open: YES {row['open_yes_ask']}¢, NO {row['open_no_ask']}¢")
        print(f"    Range: YES {row['min_yes_ask']}-{row['max_yes_ask']}¢, NO {row['min_no_ask']}-{row['max_no_ask']}¢")
        print(f"    Ticks: {row['tick_count']}")


if __name__ == "__main__":
    main()
