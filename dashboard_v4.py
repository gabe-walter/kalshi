#!/usr/bin/env python3
"""
Live Dashboard V4 for Kalshi Crypto Paper Trader

Multi-asset (BTC + ETH), real-time mark-to-market P&L, IV surface visualization,
signal monitor, risk dashboard, performance analytics, trade journal, and alerts.

Run with: streamlit run dashboard_v4.py

Requires paper_trader_v3 (or v4) to be running and writing to data/paper_trading_v3.
The trader must also write the NEW files specified in this dashboard:
  - data/paper_trading_v3/signals.json      (current scan signals)
  - data/paper_trading_v3/iv_snapshot.json   (current IV surface grid)
  - data/paper_trading_v3/orderbook_cache.json (latest orderbook mid-prices)
"""

import sys
sys.path.insert(0, "src")

import json
import time
import math
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data.deribit_client import DeribitClient
from data.kalshi_client import KalshiClient

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data/paper_trading_v3")
REFRESH_INTERVAL = 15  # seconds
INITIAL_BANKROLL = 10_000.0
ASSETS = ("BTC", "ETH")
ASSET_COLORS = {"BTC": "#F7931A", "ETH": "#627EEA", "combined": "#00D4AA"}

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Crypto Paper Trader V4",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject auto-refresh via meta tag (Streamlit-compatible approach)
st.markdown(
    f"""
    <meta http-equiv="refresh" content="{REFRESH_INTERVAL}">
    """,
    unsafe_allow_html=True,
)

# Custom CSS for tighter layout and alert styling
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    div[data-testid="stMetric"] {background: #0e1117; border: 1px solid #262730;
        border-radius: 8px; padding: 12px 16px;}
    .alert-red {background: #3d1111; border-left: 4px solid #ff4444;
        padding: 10px 14px; margin: 6px 0; border-radius: 4px; color: #ff9999;}
    .alert-yellow {background: #3d3211; border-left: 4px solid #ffaa00;
        padding: 10px 14px; margin: 6px 0; border-radius: 4px; color: #ffdd88;}
    .alert-green {background: #113d11; border-left: 4px solid #44ff44;
        padding: 10px 14px; margin: 6px 0; border-radius: 4px; color: #99ff99;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_resource
def get_deribit_client():
    return DeribitClient()


@st.cache_resource
def get_kalshi_client():
    return KalshiClient()


def load_json(filename: str):
    """Load a JSON file from DATA_DIR. Returns None if missing."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_state() -> Optional[dict]:
    return load_json("state.json")


def load_equity_curve() -> list:
    return load_json("equity_curve.json") or []


def load_trade_log() -> list:
    return load_json("trade_log.json") or []


def load_signals() -> Optional[dict]:
    """Load the latest signal scan output.

    NEW FILE the trader must write. Structure:
    {
        "scan_time": "2026-...",
        "spots": {"BTC": 97500.0, "ETH": 3200.0},
        "markets_scanned": 142,
        "signals": [
            {
                "ticker": "KXBTCD-...",
                "asset": "BTC",
                "strike": 99000,
                "settlement_time": "2026-...",
                "hours_to_settlement": 3.5,
                "model_prob": 0.12,
                "sigma_distance": -2.3,
                "kalshi_yes_bid": 10, "kalshi_yes_ask": 14,
                "kalshi_no_bid": 86, "kalshi_no_ask": 90,
                "model_yes_cents": 12.0,
                "mispricing_yes": -2.0, "mispricing_no": 2.0,
                "best_side": "NO", "best_mispricing": 2.0,
                "filter_result": "passed" | "not_tail" | "insufficient_mispricing" | ...,
                "filter_reason": "human-readable reason",
                "would_trade": true/false,
                "orderbook_depth": 150,
                "position_size_if_traded": 25
            },
            ...
        ]
    }
    """
    return load_json("signals.json")


def load_iv_snapshot() -> Optional[dict]:
    """Load the latest IV surface snapshot.

    NEW FILE the trader must write. Structure:
    {
        "snapshot_time": "2026-...",
        "surfaces": {
            "BTC": {
                "spot": 97500.0,
                "expiries": ["7FEB26", "8FEB26", ...],
                "grid": [
                    {"strike": 90000, "time_years": 0.001, "iv": 0.65, "model_prob": 0.95},
                    ...
                ],
                "atm_iv_term": [
                    {"expiry": "7FEB26", "time_hours": 12.5, "atm_iv": 0.55},
                    ...
                ]
            },
            "ETH": { ... }
        }
    }
    """
    return load_json("iv_snapshot.json")


def load_orderbook_cache() -> Optional[dict]:
    """Load latest orderbook mid-prices for mark-to-market.

    NEW FILE the trader must write. Structure:
    {
        "update_time": "2026-...",
        "prices": {
            "KXBTCD-26FEB1217-T97499.99": {"yes_bid": 45, "yes_ask": 50, "no_bid": 50, "no_ask": 55},
            ...
        }
    }
    """
    return load_json("orderbook_cache.json")


def get_live_prices() -> Dict[str, Optional[float]]:
    """Get current spot prices for all assets from Deribit."""
    prices = {}
    client = get_deribit_client()
    for asset in ASSETS:
        try:
            data = client.get_index_price(asset)
            prices[asset] = data.get("index_price", 0)
        except Exception:
            prices[asset] = None
    return prices


def get_price_history(asset: str) -> Optional[pd.DataFrame]:
    """Get 24h of 5-minute candles for an asset."""
    try:
        client = get_deribit_client()
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (24 * 60 * 60 * 1000)
        perp = f"{asset}-PERPETUAL"
        data = client._request("get_tradingview_chart_data", {
            "instrument_name": perp,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "resolution": "5"
        })
        if data and "close" in data:
            return pd.DataFrame({
                "time": pd.to_datetime(data["ticks"], unit="ms"),
                "price": data["close"]
            })
    except Exception:
        pass
    return None


# =============================================================================
# Helpers
# =============================================================================

def parse_positions(state: dict) -> Tuple[List[dict], List[dict]]:
    """Split positions into open and closed lists."""
    positions = state.get("positions", {})
    open_pos = [p for p in positions.values() if p.get("exit_time") is None]
    closed_pos = [p for p in positions.values() if p.get("exit_time") is not None]
    return open_pos, closed_pos


def mark_to_market_pnl(position: dict, orderbook_cache: Optional[dict]) -> Optional[float]:
    """Calculate mark-to-market P&L for an open position using orderbook mid.

    If no orderbook cache is available, returns None (fall back to cost-basis only).
    """
    if orderbook_cache is None:
        return None

    prices = orderbook_cache.get("prices", {})
    ticker = position["ticker"]
    book = prices.get(ticker)
    if book is None:
        return None

    side = position["side"]
    if side == "YES":
        bid = book.get("yes_bid", 0) or 0
        ask = book.get("yes_ask", 0) or 0
    else:
        bid = book.get("no_bid", 0) or 0
        ask = book.get("no_ask", 0) or 0

    if bid <= 0 and ask <= 0:
        return None

    mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (bid or ask)
    entry = position["entry_price"]
    size = position["size"]
    return size * (mid - entry) / 100


def time_remaining_str(settlement_iso: str) -> str:
    """Human-readable time remaining."""
    settlement = datetime.fromisoformat(settlement_iso)
    now = datetime.now(timezone.utc)
    delta = settlement - now
    hours = delta.total_seconds() / 3600
    if hours < 0:
        return "EXPIRED"
    if hours < 1:
        return f"{int(delta.total_seconds() / 60)}m"
    if hours < 24:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def sharpe_ratio(returns: pd.Series, periods_per_year: float = 365 * 24) -> float:
    """Annualized Sharpe ratio from a series of periodic returns."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


# =============================================================================
# Section 1: Header + Global KPIs
# =============================================================================

def render_header(state: dict, spots: Dict[str, Optional[float]],
                  open_pos: list, closed_pos: list, orderbook_cache: Optional[dict]):
    """Top bar: spot prices, bankroll, mark-to-market equity, drawdown, exposure."""

    st.markdown("### Crypto Paper Trader V4")

    if state is None:
        st.warning("No trader state found. Is paper_trader_v3/v4 running?")
        return

    bankroll = state.get("bankroll", INITIAL_BANKROLL)
    peak = state.get("peak_bankroll", bankroll)

    # --- Realized P&L ---
    realized_pnl = sum(p.get("pnl", 0) for p in closed_pos if p.get("pnl") is not None)

    # --- Unrealized (mark-to-market) P&L ---
    unrealized = 0.0
    mtm_available = False
    for p in open_pos:
        mtm = mark_to_market_pnl(p, orderbook_cache)
        if mtm is not None:
            unrealized += mtm
            mtm_available = True

    # Mark-to-market equity = bankroll + unrealized
    mtm_equity = bankroll + unrealized if mtm_available else bankroll
    total_return_pct = (mtm_equity - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100

    # Drawdown (use mtm equity if available, else bankroll)
    dd_ref = max(peak, mtm_equity)
    drawdown = dd_ref - mtm_equity
    drawdown_pct = (drawdown / dd_ref * 100) if dd_ref > 0 else 0

    open_exposure = sum(p["size"] * p["entry_price"] / 100 for p in open_pos)

    # --- Layout: 7 columns ---
    cols = st.columns(7)

    with cols[0]:
        btc = spots.get("BTC")
        st.metric("BTC", f"${btc:,.0f}" if btc else "N/A")
    with cols[1]:
        eth = spots.get("ETH")
        st.metric("ETH", f"${eth:,.0f}" if eth else "N/A")
    with cols[2]:
        st.metric("Bankroll (realized)", f"${bankroll:,.2f}",
                   f"{(bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100:+.2f}%")
    with cols[3]:
        label = "MTM Equity" if mtm_available else "Equity (no MTM)"
        st.metric(label, f"${mtm_equity:,.2f}", f"{total_return_pct:+.2f}%")
    with cols[4]:
        st.metric("Unrealized P&L",
                   f"${unrealized:+,.2f}" if mtm_available else "N/A",
                   delta_color="normal")
    with cols[5]:
        st.metric("Drawdown", f"${drawdown:,.0f}",
                   f"-{drawdown_pct:.1f}%" if drawdown_pct > 0 else "0%",
                   delta_color="inverse")
    with cols[6]:
        st.metric("Open / Exposure",
                   f"{len(open_pos)} pos",
                   f"${open_exposure:,.0f}")


# =============================================================================
# Section 2: Price Charts (Multi-Asset)
# =============================================================================

def render_price_charts(spots: Dict[str, Optional[float]], open_pos: list):
    """Two side-by-side price charts with strike overlays."""

    st.subheader("Spot Prices (24h)")
    cols = st.columns(2)

    for idx, asset in enumerate(ASSETS):
        with cols[idx]:
            history = get_price_history(asset)
            spot = spots.get(asset)

            # Positions for this asset
            asset_positions = [p for p in open_pos if p.get("asset", "BTC") == asset]

            if history is None or history.empty:
                st.info(f"Loading {asset} price data...")
                continue

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=history["time"], y=history["price"],
                mode="lines", name=f"{asset} Price",
                line=dict(color=ASSET_COLORS[asset], width=2)
            ))

            # Strike lines for open positions
            if asset_positions:
                strikes = sorted(set(p["strike"] for p in asset_positions))
                for strike in strikes:
                    # Find the side for label color
                    sides = [p["side"] for p in asset_positions if p["strike"] == strike]
                    color = "#00cc66" if "YES" in sides else "#cc3333"
                    fig.add_hline(
                        y=strike, line_dash="dash", line_color=color,
                        annotation_text=f"${strike:,.0f}",
                        annotation_position="right"
                    )

            # Current spot marker
            if spot:
                fig.add_hline(y=spot, line_dash="dot", line_color="white",
                              line_width=1, opacity=0.5)

            fig.update_layout(
                title=f"{asset} ${spot:,.0f}" if spot else asset,
                height=320, margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="", yaxis_title="",
                template="plotly_dark",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Section 3: Real-Time P&L Tracking
# =============================================================================

def render_mtm_positions(open_pos: list, spots: Dict[str, Optional[float]],
                         orderbook_cache: Optional[dict]):
    """Open positions table with mark-to-market values."""

    st.subheader("Open Positions (Mark-to-Market)")

    if not open_pos:
        st.info("No open positions")
        return

    now = datetime.now(timezone.utc)
    rows = []

    for p in sorted(open_pos, key=lambda x: x["settlement_time"]):
        asset = p.get("asset", "BTC")
        spot = spots.get(asset)
        strike = p["strike"]
        side = p["side"]

        # ITM/OTM status
        if spot:
            if side == "YES":
                is_itm = spot > strike
            else:
                is_itm = spot < strike
            dist_pct = (spot - strike) / strike * 100
            status = "ITM" if is_itm else "OTM"
        else:
            status = "?"
            dist_pct = 0

        # Mark-to-market
        mtm = mark_to_market_pnl(p, orderbook_cache)
        cost = p["size"] * p["entry_price"] / 100
        max_profit = p["size"] * (100 - p["entry_price"]) / 100

        # Current mid price
        mid_str = "N/A"
        if orderbook_cache and orderbook_cache.get("prices", {}).get(p["ticker"]):
            book = orderbook_cache["prices"][p["ticker"]]
            if side == "YES":
                bid = book.get("yes_bid", 0) or 0
                ask = book.get("yes_ask", 0) or 0
            else:
                bid = book.get("no_bid", 0) or 0
                ask = book.get("no_ask", 0) or 0
            if bid > 0 or ask > 0:
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (bid or ask)
                mid_str = f"{mid:.0f}c"

        rows.append({
            "Asset": asset,
            "Ticker": p["ticker"].split("-", 1)[1] if "-" in p["ticker"] else p["ticker"],
            "Side": side,
            "Strike": f"${strike:,.0f}",
            "Qty": p["size"],
            "Entry": f"{p['entry_price']}c",
            "Mid": mid_str,
            "Cost": f"${cost:.2f}",
            "MTM P&L": f"${mtm:+.2f}" if mtm is not None else "N/A",
            "Max Profit": f"${max_profit:.2f}",
            "Edge": f"{p.get('edge', 0):.1%}",
            "Sigma": f"{p.get('sigma_distance', 0):+.1f}" if "sigma_distance" in p else "?",
            "Status": status,
            "Dist": f"{dist_pct:+.1f}%",
            "Expires": time_remaining_str(p["settlement_time"]),
        })

    df = pd.DataFrame(rows)

    # Color the MTM P&L column via Streamlit dataframe
    st.dataframe(df, use_container_width=True, hide_index=True, height=min(400, 40 + 35 * len(rows)))

    # Summary metrics below the table
    total_cost = sum(p["size"] * p["entry_price"] / 100 for p in open_pos)
    total_mtm = sum(mark_to_market_pnl(p, orderbook_cache) or 0 for p in open_pos)
    total_max = sum(p["size"] * (100 - p["entry_price"]) / 100 for p in open_pos)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Cost Basis", f"${total_cost:,.2f}")
    with c2:
        st.metric("Unrealized P&L", f"${total_mtm:+,.2f}")
    with c3:
        st.metric("Max Potential Profit", f"${total_max:,.2f}")
    with c4:
        itm = 0
        for p in open_pos:
            asset = p.get("asset", "BTC")
            spot = spots.get(asset)
            if spot:
                if (p["side"] == "YES" and spot > p["strike"]) or \
                   (p["side"] == "NO" and spot < p["strike"]):
                    itm += 1
        st.metric("ITM / Total", f"{itm} / {len(open_pos)}")


# =============================================================================
# Section 4: IV Surface Visualization
# =============================================================================

def render_iv_surface(iv_snapshot: Optional[dict], spots: Dict[str, Optional[float]]):
    """3D IV surface + model vs market comparison scatter."""

    st.subheader("IV Surface & Model vs Market")

    if iv_snapshot is None:
        st.info(
            "No IV snapshot available. The trader must write `iv_snapshot.json` each cycle. "
            "See docstring in `load_iv_snapshot()` for the required schema."
        )
        return

    surfaces = iv_snapshot.get("surfaces", {})
    asset_tabs = st.tabs(list(ASSETS))

    for tab, asset in zip(asset_tabs, ASSETS):
        with tab:
            surface_data = surfaces.get(asset)
            if not surface_data:
                st.info(f"No IV surface data for {asset}")
                continue

            spot = surface_data.get("spot", spots.get(asset, 0))
            grid = surface_data.get("grid", [])

            if not grid:
                st.info(f"Empty grid for {asset}")
                continue

            # --- 3D Surface Plot ---
            grid_df = pd.DataFrame(grid)
            if {"strike", "time_years", "iv"}.issubset(grid_df.columns):
                # Convert time to hours for readability
                grid_df["time_hours"] = grid_df["time_years"] * 365.25 * 24

                fig_surface = go.Figure(data=[go.Mesh3d(
                    x=grid_df["time_hours"],
                    y=grid_df["strike"],
                    z=grid_df["iv"] * 100,  # show as percentage
                    intensity=grid_df["iv"] * 100,
                    colorscale="Viridis",
                    opacity=0.7,
                    name="IV Surface"
                )])
                fig_surface.update_layout(
                    title=f"{asset} IV Surface",
                    scene=dict(
                        xaxis_title="Hours to Expiry",
                        yaxis_title="Strike",
                        zaxis_title="IV (%)"
                    ),
                    height=450,
                    template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_surface, use_container_width=True)

            # --- ATM Term Structure ---
            atm_data = surface_data.get("atm_iv_term", [])
            if atm_data:
                atm_df = pd.DataFrame(atm_data)
                fig_atm = go.Figure()
                fig_atm.add_trace(go.Scatter(
                    x=atm_df["time_hours"], y=atm_df["atm_iv"].apply(lambda v: v * 100),
                    mode="lines+markers", name="ATM IV",
                    line=dict(color=ASSET_COLORS[asset])
                ))
                fig_atm.update_layout(
                    title=f"{asset} ATM IV Term Structure",
                    xaxis_title="Hours to Expiry",
                    yaxis_title="IV (%)",
                    height=280, template="plotly_dark",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_atm, use_container_width=True)

            # --- Model vs Market scatter ---
            # This requires signal data that includes both model_prob and kalshi price
            st.caption(
                "Model vs Market: see Signal Monitor tab for per-market model probability "
                "vs Kalshi mid-price comparison."
            )


# =============================================================================
# Section 5: Signal Monitor
# =============================================================================

def render_signal_monitor(signal_data: Optional[dict]):
    """Live view of all scanned markets: what passed/failed and why."""

    st.subheader("Signal Monitor")

    if signal_data is None:
        st.info(
            "No signal data available. The trader must write `signals.json` each cycle. "
            "See docstring in `load_signals()` for the required schema."
        )
        return

    scan_time = signal_data.get("scan_time", "unknown")
    total_scanned = signal_data.get("markets_scanned", 0)
    signals = signal_data.get("signals", [])

    st.caption(f"Last scan: {scan_time} | Markets scanned: {total_scanned}")

    if not signals:
        st.info("No signals in latest scan")
        return

    # --- Filter controls ---
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        asset_filter = st.selectbox("Asset", ["All"] + list(ASSETS), key="sig_asset")
    with fc2:
        status_filter = st.selectbox("Filter Result",
                                      ["All", "passed", "not_tail",
                                       "insufficient_mispricing", "outside_grid",
                                       "no_liquidity"],
                                      key="sig_status")
    with fc3:
        sort_by = st.selectbox("Sort by",
                                ["best_mispricing", "sigma_distance", "hours_to_settlement"],
                                key="sig_sort")
    with fc4:
        sort_desc = st.checkbox("Descending", value=True, key="sig_desc")

    # Apply filters
    filtered = signals
    if asset_filter != "All":
        filtered = [s for s in filtered if s.get("asset") == asset_filter]
    if status_filter != "All":
        filtered = [s for s in filtered if s.get("filter_result") == status_filter]

    # Sort
    filtered.sort(
        key=lambda s: abs(s.get(sort_by, 0)) if sort_by == "sigma_distance" else s.get(sort_by, 0),
        reverse=sort_desc
    )

    # --- Summary bar ---
    passed = [s for s in signals if s.get("filter_result") == "passed"]
    blocked_tail = sum(1 for s in signals if s.get("filter_result") == "not_tail")
    blocked_misprice = sum(1 for s in signals if s.get("filter_result") == "insufficient_mispricing")
    blocked_other = sum(1 for s in signals
                        if s.get("filter_result") not in ("passed", "not_tail", "insufficient_mispricing"))

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Passed Filters", len(passed))
    with mc2:
        st.metric("Blocked: Not Tail", blocked_tail)
    with mc3:
        st.metric("Blocked: Mispricing", blocked_misprice)
    with mc4:
        st.metric("Blocked: Other", blocked_other)

    # --- Model vs Market scatter (all signals) ---
    if passed:
        scatter_df = pd.DataFrame(passed)
        if "model_yes_cents" in scatter_df.columns and "kalshi_yes_ask" in scatter_df.columns:
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=scatter_df["kalshi_yes_ask"],
                y=scatter_df["model_yes_cents"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=scatter_df["best_mispricing"],
                    colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="Mispricing (c)")
                ),
                text=scatter_df.apply(
                    lambda r: f"{r.get('asset','?')} {r.get('ticker','')}<br>"
                              f"sigma={r.get('sigma_distance',0):.1f}",
                    axis=1
                ),
                hoverinfo="text+x+y"
            ))
            fig_scatter.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                                   line=dict(dash="dash", color="gray"))
            fig_scatter.update_layout(
                title="Model Prob (cents) vs Kalshi Ask",
                xaxis_title="Kalshi YES Ask (cents)",
                yaxis_title="Model YES (cents)",
                height=350, template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Signals table ---
    rows = []
    for s in filtered[:100]:  # cap at 100 rows
        rows.append({
            "Asset": s.get("asset", "?"),
            "Ticker": s.get("ticker", "?"),
            "Strike": f"${s.get('strike', 0):,.0f}",
            "T-Exp": f"{s.get('hours_to_settlement', 0):.1f}h",
            "Sigma": f"{s.get('sigma_distance', 0):+.1f}",
            "Model YES": f"{s.get('model_yes_cents', 0):.1f}c",
            "Kalshi YES": f"{s.get('kalshi_yes_bid', 0)}/{s.get('kalshi_yes_ask', 0)}",
            "Kalshi NO": f"{s.get('kalshi_no_bid', 0)}/{s.get('kalshi_no_ask', 0)}",
            "Best Side": s.get("best_side", ""),
            "Mispricing": f"{s.get('best_mispricing', 0):+.0f}c",
            "Filter": s.get("filter_result", "?"),
            "Reason": s.get("filter_reason", ""),
            "Depth": s.get("orderbook_depth", "?"),
            "Size": s.get("position_size_if_traded", ""),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                      height=min(500, 40 + 35 * len(rows)))


# =============================================================================
# Section 6: Risk Dashboard
# =============================================================================

def render_risk_dashboard(state: dict, open_pos: list, equity_data: list,
                          spots: Dict[str, Optional[float]]):
    """Exposure breakdown, concentration charts, drawdown over time."""

    st.subheader("Risk Dashboard")

    if not open_pos and not equity_data:
        st.info("No positions or equity history to display")
        return

    bankroll = state.get("bankroll", INITIAL_BANKROLL) if state else INITIAL_BANKROLL

    # --- Exposure breakdown ---
    rc1, rc2 = st.columns(2)

    with rc1:
        # By asset
        exposure_by_asset = defaultdict(float)
        for p in open_pos:
            cost = p["size"] * p["entry_price"] / 100
            exposure_by_asset[p.get("asset", "BTC")] += cost

        if exposure_by_asset:
            fig_asset = go.Figure(data=[go.Pie(
                labels=list(exposure_by_asset.keys()),
                values=list(exposure_by_asset.values()),
                marker_colors=[ASSET_COLORS.get(a, "#888") for a in exposure_by_asset.keys()],
                hole=0.4
            )])
            fig_asset.update_layout(
                title="Exposure by Asset",
                height=280, template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_asset, use_container_width=True)
        else:
            st.info("No open exposure")

    with rc2:
        # By settlement time
        exposure_by_settle = defaultdict(float)
        for p in open_pos:
            cost = p["size"] * p["entry_price"] / 100
            settle = p["settlement_time"][:13]  # truncate to hour
            exposure_by_settle[settle] += cost

        if exposure_by_settle:
            settle_df = pd.DataFrame({
                "Settlement": list(exposure_by_settle.keys()),
                "Exposure": list(exposure_by_settle.values())
            }).sort_values("Settlement")

            fig_settle = go.Figure(data=[go.Bar(
                x=settle_df["Settlement"],
                y=settle_df["Exposure"],
                marker_color="#00D4AA"
            )])
            fig_settle.update_layout(
                title="Exposure by Settlement",
                xaxis_title="", yaxis_title="$ Exposure",
                height=280, template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_settle, use_container_width=True)
        else:
            st.info("No open exposure")

    # --- Concentration limits ---
    if open_pos:
        st.markdown("**Concentration vs Limits**")
        total_exposure = sum(p["size"] * p["entry_price"] / 100 for p in open_pos)
        max_total = bankroll * 0.80

        # Per-strike max
        strike_exposures = defaultdict(float)
        for p in open_pos:
            key = f"{p.get('asset', 'BTC')}|{p['strike']}|{p['settlement_time'][:13]}"
            strike_exposures[key] += p["size"] * p["entry_price"] / 100
        max_strike_exposure = max(strike_exposures.values()) if strike_exposures else 0
        max_strike_limit = bankroll * 0.15

        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            pct = total_exposure / max_total * 100
            color = "normal" if pct < 70 else "inverse"
            st.metric("Total Exposure",
                       f"${total_exposure:,.0f} / ${max_total:,.0f}",
                       f"{pct:.0f}% of limit",
                       delta_color=color)
        with lc2:
            pct2 = max_strike_exposure / max_strike_limit * 100 if max_strike_limit > 0 else 0
            st.metric("Max Single Strike",
                       f"${max_strike_exposure:,.0f} / ${max_strike_limit:,.0f}",
                       f"{pct2:.0f}% of limit")
        with lc3:
            # Directional tilt
            yes_exposure = sum(p["size"] * p["entry_price"] / 100 for p in open_pos if p["side"] == "YES")
            no_exposure = sum(p["size"] * p["entry_price"] / 100 for p in open_pos if p["side"] == "NO")
            st.metric("YES / NO Split",
                       f"${yes_exposure:,.0f} / ${no_exposure:,.0f}")

    # --- Drawdown chart ---
    if equity_data:
        eq_df = pd.DataFrame(equity_data)
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=eq_df["timestamp"], y=-eq_df["drawdown_pct"],
            fill="tozeroy", fillcolor="rgba(255,68,68,0.2)",
            line=dict(color="#ff4444", width=1),
            name="Drawdown %"
        ))
        fig_dd.update_layout(
            title="Drawdown Over Time",
            xaxis_title="", yaxis_title="Drawdown (%)",
            height=250, template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_dd, use_container_width=True)


# =============================================================================
# Section 7: Performance Analytics
# =============================================================================

def render_performance(closed_pos: list, equity_data: list):
    """Win rate breakdowns, Sharpe, profit factor, equity curve."""

    st.subheader("Performance Analytics")

    if not closed_pos:
        st.info("No closed positions yet")
        return

    # Build DataFrame of closed trades
    trades_df = pd.DataFrame(closed_pos)
    trades_df["pnl"] = trades_df["pnl"].fillna(0)
    trades_df["win"] = trades_df["pnl"] > 0

    if "exit_time" in trades_df.columns:
        trades_df["exit_dt"] = pd.to_datetime(trades_df["exit_time"])
    if "entry_time" in trades_df.columns:
        trades_df["entry_dt"] = pd.to_datetime(trades_df["entry_time"])
    if "asset" not in trades_df.columns:
        # Infer from ticker
        trades_df["asset"] = trades_df["ticker"].apply(
            lambda t: "ETH" if "KXETHD" in t else "BTC"
        )

    # --- Overall stats ---
    total_trades = len(trades_df)
    wins = trades_df["win"].sum()
    losses = total_trades - wins
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_pnl = trades_df["pnl"].sum()
    avg_win = trades_df.loc[trades_df["win"], "pnl"].mean() if wins > 0 else 0
    avg_loss = trades_df.loc[~trades_df["win"], "pnl"].mean() if losses > 0 else 0
    gross_profit = trades_df.loc[trades_df["win"], "pnl"].sum() if wins > 0 else 0
    gross_loss = abs(trades_df.loc[~trades_df["win"], "pnl"].sum()) if losses > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy
    expectancy = total_pnl / total_trades if total_trades > 0 else 0

    mc = st.columns(6)
    with mc[0]:
        st.metric("Total P&L", f"${total_pnl:+,.2f}")
    with mc[1]:
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{int(wins)}W / {int(losses)}L")
    with mc[2]:
        st.metric("Avg Win / Loss", f"${avg_win:+,.2f} / ${avg_loss:+,.2f}")
    with mc[3]:
        st.metric("Profit Factor", f"{profit_factor:.2f}")
    with mc[4]:
        st.metric("Expectancy", f"${expectancy:+,.2f} / trade")
    with mc[5]:
        # Sharpe from equity curve
        if equity_data and len(equity_data) > 2:
            eq_df = pd.DataFrame(equity_data)
            returns = eq_df["bankroll"].pct_change().dropna()
            sr = sharpe_ratio(returns)
            st.metric("Sharpe (ann.)", f"{sr:.2f}")
        else:
            st.metric("Sharpe", "N/A")

    # --- Win rate by asset ---
    st.markdown("---")
    bc1, bc2 = st.columns(2)

    with bc1:
        # By asset
        asset_stats = trades_df.groupby("asset").agg(
            total=("pnl", "count"),
            wins=("win", "sum"),
            pnl=("pnl", "sum")
        ).reset_index()
        asset_stats["win_rate"] = asset_stats["wins"] / asset_stats["total"] * 100

        fig_asset = go.Figure()
        fig_asset.add_trace(go.Bar(
            x=asset_stats["asset"], y=asset_stats["win_rate"],
            marker_color=[ASSET_COLORS.get(a, "#888") for a in asset_stats["asset"]],
            text=asset_stats.apply(lambda r: f"{r['win_rate']:.0f}% ({int(r['wins'])}/{int(r['total'])})", axis=1),
            textposition="auto"
        ))
        fig_asset.update_layout(
            title="Win Rate by Asset",
            yaxis_title="%", height=280, template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_asset, use_container_width=True)

    with bc2:
        # By sigma bucket
        if "model_prob" in trades_df.columns:
            # Derive approximate sigma from model_prob
            # If sigma_distance is not stored, we can bucket by edge or model_prob
            if "edge" in trades_df.columns:
                trades_df["edge_bucket"] = pd.cut(
                    trades_df["edge"] * 100,
                    bins=[0, 3, 5, 8, 12, 100],
                    labels=["0-3c", "3-5c", "5-8c", "8-12c", "12c+"]
                )
                bucket_stats = trades_df.groupby("edge_bucket", observed=True).agg(
                    total=("pnl", "count"),
                    wins=("win", "sum"),
                    pnl=("pnl", "sum")
                ).reset_index()
                bucket_stats["win_rate"] = bucket_stats["wins"] / bucket_stats["total"] * 100

                fig_bucket = go.Figure()
                fig_bucket.add_trace(go.Bar(
                    x=bucket_stats["edge_bucket"].astype(str),
                    y=bucket_stats["pnl"],
                    marker_color=bucket_stats["pnl"].apply(
                        lambda v: "#00cc66" if v > 0 else "#cc3333"
                    ),
                    text=bucket_stats.apply(
                        lambda r: f"${r['pnl']:+,.0f} ({r['win_rate']:.0f}%)", axis=1
                    ),
                    textposition="auto"
                ))
                fig_bucket.update_layout(
                    title="P&L by Edge Bucket",
                    yaxis_title="P&L ($)", height=280, template="plotly_dark",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_bucket, use_container_width=True)

    # --- Rolling metrics over time ---
    if len(trades_df) >= 5 and "exit_dt" in trades_df.columns:
        trades_sorted = trades_df.sort_values("exit_dt")
        trades_sorted["cum_pnl"] = trades_sorted["pnl"].cumsum()
        trades_sorted["rolling_wr"] = trades_sorted["win"].rolling(20, min_periods=5).mean() * 100

        # Cumulative P&L + rolling win rate
        fig_rolling = make_subplots(specs=[[{"secondary_y": True}]])

        fig_rolling.add_trace(go.Scatter(
            x=trades_sorted["exit_dt"], y=trades_sorted["cum_pnl"],
            mode="lines", name="Cumulative P&L",
            line=dict(color="#00D4AA", width=2)
        ), secondary_y=False)

        fig_rolling.add_trace(go.Scatter(
            x=trades_sorted["exit_dt"], y=trades_sorted["rolling_wr"],
            mode="lines", name="Rolling WR (20)",
            line=dict(color="#627EEA", width=1, dash="dash")
        ), secondary_y=True)

        fig_rolling.update_layout(
            title="Cumulative P&L & Rolling Win Rate",
            height=300, template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        fig_rolling.update_yaxes(title_text="Cum P&L ($)", secondary_y=False)
        fig_rolling.update_yaxes(title_text="Win Rate (%)", secondary_y=True)
        st.plotly_chart(fig_rolling, use_container_width=True)

    # --- Profit factor over time (trailing 20 trades) ---
    if len(trades_df) >= 10 and "exit_dt" in trades_df.columns:
        trades_sorted = trades_df.sort_values("exit_dt").reset_index(drop=True)
        pf_values = []
        for i in range(19, len(trades_sorted)):
            window = trades_sorted.iloc[i - 19:i + 1]
            gp = window.loc[window["win"], "pnl"].sum()
            gl = abs(window.loc[~window["win"], "pnl"].sum())
            pf = gp / gl if gl > 0 else 10.0  # cap at 10
            pf_values.append({"idx": i, "exit_dt": trades_sorted.iloc[i]["exit_dt"], "pf": min(pf, 10)})

        if pf_values:
            pf_df = pd.DataFrame(pf_values)
            fig_pf = go.Figure()
            fig_pf.add_trace(go.Scatter(
                x=pf_df["exit_dt"], y=pf_df["pf"],
                mode="lines", name="Profit Factor (20)",
                line=dict(color="#F7931A", width=2)
            ))
            fig_pf.add_hline(y=1.0, line_dash="dash", line_color="red",
                              annotation_text="Breakeven")
            fig_pf.update_layout(
                title="Rolling Profit Factor (20 trades)",
                yaxis_title="Profit Factor",
                height=250, template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_pf, use_container_width=True)


# =============================================================================
# Section 8: Trade Journal
# =============================================================================

def render_trade_journal(trade_log: list, state: dict):
    """Detailed log of every trade with entry rationale and model state."""

    st.subheader("Trade Journal")

    if not trade_log:
        st.info("No trade log entries")
        return

    # --- Filters ---
    jc1, jc2, jc3 = st.columns(3)
    with jc1:
        action_filter = st.selectbox("Action", ["All", "OPEN", "CLOSE", "SKIP"], key="journal_action")
    with jc2:
        journal_asset_filter = st.selectbox("Asset", ["All", "BTC", "ETH"], key="journal_asset")
    with jc3:
        n_entries = st.slider("Entries to show", 10, 200, 50, key="journal_n")

    # Filter
    entries = trade_log[::-1]  # most recent first
    if action_filter != "All":
        entries = [e for e in entries if e.get("action") == action_filter]
    if journal_asset_filter != "All":
        series = "KXBTCD" if journal_asset_filter == "BTC" else "KXETHD"
        entries = [e for e in entries if series in e.get("ticker", "")]

    entries = entries[:n_entries]

    if not entries:
        st.info("No matching entries")
        return

    rows = []
    for e in entries:
        ticker = e.get("ticker", "")
        asset = "ETH" if "KXETHD" in ticker else "BTC"
        rows.append({
            "Time": e.get("timestamp", "")[:19],
            "Action": e.get("action", ""),
            "Asset": asset,
            "Ticker": ticker.split("-", 1)[1] if "-" in ticker else ticker,
            "Side": e.get("side", ""),
            "Strike": f"${e.get('strike', 0):,.0f}",
            "Price": f"{e.get('price', 0)}c",
            "Qty": e.get("size", 0),
            "Model P": f"{e.get('model_prob', 0):.2f}",
            "Edge": f"{e.get('edge', 0):.1%}",
            "Reason": e.get("reason", ""),
            "Bankroll After": f"${e.get('bankroll_after', 0):,.2f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                  height=min(600, 40 + 35 * len(rows)))

    # --- Expandable detail for recent OPENs ---
    recent_opens = [e for e in entries if e.get("action") == "OPEN"][:5]
    if recent_opens:
        st.markdown("**Recent Trade Details**")
        for e in recent_opens:
            ticker = e.get("ticker", "")
            with st.expander(
                f"{e.get('timestamp', '')[:16]} | {e.get('side','')} "
                f"{ticker} @ {e.get('price', 0)}c x {e.get('size', 0)}"
            ):
                dc1, dc2, dc3 = st.columns(3)
                with dc1:
                    st.write(f"**Model probability:** {e.get('model_prob', 0):.4f}")
                    st.write(f"**Edge:** {e.get('edge', 0):.2%}")
                with dc2:
                    st.write(f"**Entry price:** {e.get('price', 0)} cents")
                    st.write(f"**Size:** {e.get('size', 0)} contracts")
                    cost = e.get("size", 0) * e.get("price", 0) / 100
                    st.write(f"**Cost:** ${cost:.2f}")
                with dc3:
                    st.write(f"**Reason:** {e.get('reason', '')}")
                    st.write(f"**Bankroll after:** ${e.get('bankroll_after', 0):,.2f}")

                # If we can find matching position in state, show current status
                if state:
                    positions = state.get("positions", {})
                    match = None
                    for pos in positions.values():
                        if pos.get("ticker") == ticker and abs(pos.get("entry_price", 0) - e.get("price", 0)) < 1:
                            match = pos
                            break
                    if match:
                        if match.get("exit_time"):
                            st.write(f"**Outcome:** P&L ${match.get('pnl', 0):+.2f} "
                                     f"({match.get('exit_reason', '')})")
                        else:
                            st.write(f"**Status:** Still open, expires "
                                     f"{time_remaining_str(match['settlement_time'])}")


# =============================================================================
# Section 9: Alerts
# =============================================================================

def render_alerts(state: dict, open_pos: list, equity_data: list,
                  spots: Dict[str, Optional[float]], signal_data: Optional[dict]):
    """Highlight unusual conditions that need attention."""

    st.subheader("Alerts")

    if state is None:
        return

    alerts = []  # (severity, message)
    bankroll = state.get("bankroll", INITIAL_BANKROLL)
    peak = state.get("peak_bankroll", bankroll)

    # --- Drawdown alerts ---
    drawdown_pct = (peak - bankroll) / peak * 100 if peak > 0 else 0
    if drawdown_pct > 15:
        alerts.append(("red", f"CRITICAL DRAWDOWN: {drawdown_pct:.1f}% from peak "
                                f"(${peak:,.0f} -> ${bankroll:,.0f})"))
    elif drawdown_pct > 10:
        alerts.append(("yellow", f"Elevated drawdown: {drawdown_pct:.1f}% from peak"))

    # --- Exposure alerts ---
    total_exposure = sum(p["size"] * p["entry_price"] / 100 for p in open_pos)
    exposure_pct = total_exposure / bankroll * 100 if bankroll > 0 else 0
    if exposure_pct > 70:
        alerts.append(("yellow", f"High exposure: {exposure_pct:.0f}% of bankroll "
                                   f"(${total_exposure:,.0f} / ${bankroll:,.0f})"))

    # --- Concentration alerts ---
    settle_exposures = defaultdict(float)
    for p in open_pos:
        settle_exposures[p["settlement_time"]] += p["size"] * p["entry_price"] / 100
    for settle_time, exp in settle_exposures.items():
        pct = exp / bankroll * 100 if bankroll > 0 else 0
        if pct > 40:
            alerts.append(("yellow", f"High settlement concentration: {pct:.0f}% on "
                                       f"{settle_time[:16]}"))

    # --- Positions expiring soon ---
    now = datetime.now(timezone.utc)
    for p in open_pos:
        settlement = datetime.fromisoformat(p["settlement_time"])
        hours_left = (settlement - now).total_seconds() / 3600
        spot = spots.get(p.get("asset", "BTC"))
        if spot and hours_left < 1:
            strike = p["strike"]
            distance_pct = abs(spot - strike) / strike * 100
            if distance_pct < 0.5:
                alerts.append(("red", f"CLOSE CALL: {p['ticker']} expires in "
                                        f"{hours_left * 60:.0f}min, spot only "
                                        f"{distance_pct:.2f}% from strike"))

    # --- Model disagreement (from signals) ---
    if signal_data:
        signals = signal_data.get("signals", [])
        big_disagreements = [
            s for s in signals
            if abs(s.get("best_mispricing", 0)) > 15
        ]
        if big_disagreements:
            alerts.append(("yellow", f"Large model disagreements: {len(big_disagreements)} markets "
                                       f"with >15c mispricing (possible model/market dislocation)"))

    # --- Losing streak ---
    positions = state.get("positions", {})
    closed = sorted(
        [p for p in positions.values() if p.get("exit_time") is not None],
        key=lambda x: x.get("exit_time", ""),
        reverse=True
    )
    if closed:
        streak = 0
        for p in closed:
            if p.get("pnl", 0) <= 0:
                streak += 1
            else:
                break
        if streak >= 5:
            alerts.append(("red", f"LOSING STREAK: Last {streak} trades were losses"))
        elif streak >= 3:
            alerts.append(("yellow", f"Losing streak: {streak} consecutive losses"))

    # --- Stale data ---
    # Check if trader has updated recently
    if equity_data:
        last_update = equity_data[-1].get("timestamp", "")
        if last_update:
            last_dt = datetime.fromisoformat(last_update)
            staleness_min = (now - last_dt).total_seconds() / 60
            if staleness_min > 5:
                alerts.append(("yellow", f"Stale data: trader last updated {staleness_min:.0f} min ago"))
            if staleness_min > 15:
                alerts.append(("red", f"TRADER POSSIBLY DOWN: no update for {staleness_min:.0f} min"))

    # --- Render alerts ---
    if not alerts:
        st.markdown('<div class="alert-green">All clear. No alerts.</div>', unsafe_allow_html=True)
    else:
        for severity, msg in alerts:
            st.markdown(f'<div class="alert-{severity}">{msg}</div>', unsafe_allow_html=True)


# =============================================================================
# Section 10: Equity Curve
# =============================================================================

def render_equity_curve(equity_data: list):
    """Full-width equity curve with exposure overlay."""

    st.subheader("Equity Curve")

    if not equity_data:
        st.info("No equity history yet")
        return

    df = pd.DataFrame(equity_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Equity line
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["bankroll"],
        mode="lines", name="Bankroll",
        line=dict(color="#00D4AA", width=2),
        fill="tonexty" if "open_exposure" in df.columns else None
    ), secondary_y=False)

    # Starting bankroll reference
    fig.add_hline(y=INITIAL_BANKROLL, line_dash="dash", line_color="gray",
                   annotation_text=f"Start ${INITIAL_BANKROLL:,.0f}")

    # Open exposure area
    if "open_exposure" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["open_exposure"],
            mode="lines", name="Open Exposure",
            line=dict(color="#F7931A", width=1, dash="dot")
        ), secondary_y=True)

    fig.update_layout(
        height=350, template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Bankroll ($)", secondary_y=False)
    fig.update_yaxes(title_text="Exposure ($)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Main
# =============================================================================

def main():
    # --- Load all data ---
    state = load_state()
    equity_data = load_equity_curve()
    trade_log = load_trade_log()
    signal_data = load_signals()
    iv_snapshot = load_iv_snapshot()
    orderbook_cache = load_orderbook_cache()
    spots = get_live_prices()

    # Derive open/closed positions
    if state:
        open_pos, closed_pos = parse_positions(state)
    else:
        open_pos, closed_pos = [], []

    # =========================================================================
    # LAYOUT
    # =========================================================================

    # --- Header KPIs (always visible) ---
    render_header(state, spots, open_pos, closed_pos, orderbook_cache)
    st.divider()

    # --- Alerts (always visible at top, before tabs) ---
    render_alerts(state, open_pos, equity_data, spots, signal_data)
    st.divider()

    # --- Tabbed sections ---
    tab_positions, tab_signals, tab_risk, tab_perf, tab_iv, tab_journal = st.tabs([
        "Positions & Prices",
        "Signal Monitor",
        "Risk",
        "Performance",
        "IV Surface",
        "Trade Journal",
    ])

    with tab_positions:
        render_price_charts(spots, open_pos)
        st.divider()
        render_mtm_positions(open_pos, spots, orderbook_cache)

    with tab_signals:
        render_signal_monitor(signal_data)

    with tab_risk:
        render_risk_dashboard(state, open_pos, equity_data, spots)

    with tab_perf:
        render_performance(closed_pos, equity_data)

    with tab_iv:
        render_iv_surface(iv_snapshot, spots)

    with tab_journal:
        render_trade_journal(trade_log, state)

    # --- Equity curve (always visible below tabs) ---
    st.divider()
    render_equity_curve(equity_data)

    # --- Footer ---
    st.divider()
    st.caption(
        f"Last dashboard refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | "
        f"Auto-refresh: {REFRESH_INTERVAL}s | "
        f"Trader version: {state.get('version', '?') if state else '?'}"
    )


if __name__ == "__main__":
    main()
