"""
Data loader for analysis dashboard.

Loads and merges parquet snapshots, trade logs, and equity curves
from both the ws_collector and paper_trader_v3 data directories.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


# Default data directories
WS_COLLECTOR_DIR = Path("data/ws_collector")
PAPER_TRADER_DIR = Path("data/paper_trading_v3")


def _find_snapshot_dirs(base_dir: Path, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> list[Path]:
    """Find snapshot date directories within a base directory."""
    snapshots = base_dir / "snapshots"
    if not snapshots.exists():
        return []

    dirs = sorted(d for d in snapshots.iterdir() if d.is_dir() and d.name.isdigit())

    if start_date:
        dirs = [d for d in dirs if d.name >= start_date]
    if end_date:
        dirs = [d for d in dirs if d.name <= end_date]

    return dirs


def load_enriched_snapshots(base_dir: Path = WS_COLLECTOR_DIR,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
    """Load all enriched snapshot parquets into a single DataFrame."""
    dirs = _find_snapshot_dirs(base_dir, start_date, end_date)
    frames = []
    for d in dirs:
        for f in sorted(d.glob("enriched_*.parquet")):
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                continue

    if not frames:
        # Fallback: try paper_trader_v3 kalshi snapshots
        return _load_kalshi_snapshots_as_enriched(start_date, end_date)

    df = pd.concat(frames, ignore_index=True)
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    return df


def _load_kalshi_snapshots_as_enriched(start_date: Optional[str] = None,
                                        end_date: Optional[str] = None) -> pd.DataFrame:
    """Fallback: load paper_trader_v3 kalshi snapshots and normalize column names."""
    dirs = _find_snapshot_dirs(PAPER_TRADER_DIR, start_date, end_date)
    frames = []
    for d in dirs:
        for f in sorted(d.glob("kalshi_*.parquet")):
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # Normalize column names to match enriched schema
    rename = {"ticker": "market_ticker"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    # Compute asset from ticker prefix
    if "asset" not in df.columns and "market_ticker" in df.columns:
        df["asset"] = df["market_ticker"].apply(
            lambda t: "BTC" if t.startswith("KXBTCD") else ("ETH" if t.startswith("KXETHD") else None)
        )
    return df


def load_ws_trades(base_dir: Path = WS_COLLECTOR_DIR,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """Load all websocket trade parquets."""
    dirs = _find_snapshot_dirs(base_dir, start_date, end_date)
    frames = []
    for d in dirs:
        for f in sorted(d.glob("ws_trades_*.parquet")):
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    return df


def load_ws_tickers(base_dir: Path = WS_COLLECTOR_DIR,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
    """Load all websocket ticker update parquets."""
    dirs = _find_snapshot_dirs(base_dir, start_date, end_date)
    frames = []
    for d in dirs:
        for f in sorted(d.glob("ws_tickers_*.parquet")):
            try:
                frames.append(pd.read_parquet(f))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    return df


def load_deribit_snapshots(base_dir: Path = WS_COLLECTOR_DIR,
                           asset: str = "BTC",
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
    """Load Deribit option chain snapshots for a given asset."""
    dirs = _find_snapshot_dirs(base_dir, start_date, end_date)
    if not dirs:
        dirs = _find_snapshot_dirs(PAPER_TRADER_DIR, start_date, end_date)

    frames = []
    for d in dirs:
        for f in sorted(d.glob(f"deribit_{asset}_*.parquet")) or sorted(d.glob("deribit_*.parquet")):
            try:
                df = pd.read_parquet(f)
                # Filter to requested asset if loaded from combined files
                if "currency" in df.columns:
                    df = df[df["currency"] == asset]
                frames.append(df)
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_spot_prices(base_dir: Path = WS_COLLECTOR_DIR,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
    """Extract spot price time series from Deribit snapshots or enriched data."""
    # Try enriched snapshots first (has spot_price column)
    enriched = load_enriched_snapshots(base_dir, start_date, end_date)
    if not enriched.empty and "spot_price" in enriched.columns:
        spots = (
            enriched[enriched["spot_price"].notna()]
            .groupby(["snapshot_time", "asset"])["spot_price"]
            .first()
            .reset_index()
        )
        spots = spots.rename(columns={"snapshot_time": "timestamp", "spot_price": "price"})
        return spots.sort_values("timestamp").reset_index(drop=True)
    return pd.DataFrame()


def load_trade_log(data_dir: Path = PAPER_TRADER_DIR) -> pd.DataFrame:
    """Load paper trader trade log from JSON."""
    path = data_dir / "trade_log.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        trades = json.load(f)
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_equity_curve(data_dir: Path = PAPER_TRADER_DIR) -> pd.DataFrame:
    """Load paper trader equity curve from JSON."""
    path = data_dir / "equity_curve.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        data = json.load(f)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_positions(data_dir: Path = PAPER_TRADER_DIR) -> pd.DataFrame:
    """Load all positions (open + closed) from state.json."""
    path = data_dir / "state.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        state = json.load(f)
    positions = state.get("positions", {})
    if not positions:
        return pd.DataFrame()

    rows = []
    for pos_id, pos in positions.items():
        rows.append(pos)
    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
    if "settlement_time" in df.columns:
        df["settlement_time"] = pd.to_datetime(df["settlement_time"], utc=True)
    return df
