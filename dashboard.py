#!/usr/bin/env python3
"""
Live Dashboard for Kalshi KXBTCD Paper Trader V2

Run with: streamlit run dashboard.py
"""

import sys
sys.path.insert(0, "src")

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

from data.deribit_client import DeribitClient

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data/paper_trading_v2")
REFRESH_INTERVAL = 10  # seconds

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="KXBTCD Paper Trader",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Auto-refresh
st.markdown(
    f"""
    <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {REFRESH_INTERVAL * 1000});
    </script>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# Data Loading
# =============================================================================

@st.cache_resource
def get_deribit_client():
    return DeribitClient()

def load_state():
    """Load paper trader state."""
    state_file = DATA_DIR / "state.json"
    if not state_file.exists():
        return None
    with open(state_file) as f:
        return json.load(f)

def load_equity_curve():
    """Load equity curve data."""
    equity_file = DATA_DIR / "equity_curve.json"
    if not equity_file.exists():
        return []
    with open(equity_file) as f:
        return json.load(f)

def get_live_btc_price():
    """Get current BTC price from Deribit."""
    try:
        client = get_deribit_client()
        data = client.get_index_price("BTC")
        return data.get("index_price", 0)
    except Exception as e:
        st.error(f"Failed to fetch BTC price: {e}")
        return None

def get_btc_price_history():
    """Get BTC price history for chart."""
    try:
        client = get_deribit_client()
        # Get 24h of 5-minute candles
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - (24 * 60 * 60 * 1000)  # 24 hours ago

        data = client._request("get_tradingview_chart_data", {
            "instrument_name": "BTC-PERPETUAL",
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "resolution": "5"  # 5 minute candles
        })

        if data and "close" in data:
            df = pd.DataFrame({
                "time": pd.to_datetime(data["ticks"], unit="ms"),
                "price": data["close"]
            })
            return df
    except Exception as e:
        st.error(f"Failed to fetch price history: {e}")
    return None

# =============================================================================
# UI Components
# =============================================================================

def render_header(state, btc_price):
    """Render the header with key metrics."""
    st.title("₿ KXBTCD Paper Trader V2")

    if state is None:
        st.warning("No paper trader state found. Is V2 running?")
        return

    bankroll = state.get("bankroll", 10000)
    initial = 10000
    pnl_pct = (bankroll - initial) / initial * 100
    peak = state.get("peak_bankroll", bankroll)
    drawdown = peak - bankroll
    drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0

    # Open positions
    positions = state.get("positions", {})
    open_positions = [p for p in positions.values() if p.get("exit_time") is None]
    open_exposure = sum(p["size"] * p["entry_price"] / 100 for p in open_positions)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("BTC Price", f"${btc_price:,.2f}" if btc_price else "N/A")

    with col2:
        st.metric("Bankroll", f"${bankroll:,.2f}", f"{pnl_pct:+.2f}%")

    with col3:
        st.metric("Drawdown", f"${drawdown:,.2f}", f"-{drawdown_pct:.1f}%", delta_color="inverse")

    with col4:
        st.metric("Open Positions", len(open_positions))

    with col5:
        st.metric("Open Exposure", f"${open_exposure:,.2f}")

def render_btc_chart(price_history, open_positions=None):
    """Render BTC price chart with strike lines."""
    st.subheader("BTC Price (24h)")

    if price_history is None or price_history.empty:
        st.info("Loading price data...")
        return

    # Calculate y-axis range including both price data and strikes
    min_price = price_history["price"].min()
    max_price = price_history["price"].max()

    if open_positions:
        strikes = [p["strike"] for p in open_positions]
        min_price = min(min_price, min(strikes))
        max_price = max(max_price, max(strikes))

    padding = (max_price - min_price) * 0.05
    y_domain = [min_price - padding, max_price + padding]

    # Build the chart
    price_line = alt.Chart(price_history).mark_line(color="#00D4AA", strokeWidth=2).encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("price:Q", title="BTC Price", scale=alt.Scale(domain=y_domain))
    )

    # Add strike lines if we have positions
    if open_positions:
        strikes = list(set(p["strike"] for p in open_positions))
        strike_df = pd.DataFrame({
            "strike": strikes,
            "label": [f"${s:,.0f}" for s in strikes]
        })

        strike_lines = alt.Chart(strike_df).mark_rule(
            color="orange",
            strokeDash=[5, 5],
            strokeWidth=1.5
        ).encode(
            y=alt.Y("strike:Q", scale=alt.Scale(domain=y_domain))
        )

        chart = alt.layer(price_line, strike_lines).properties(height=350)
    else:
        chart = price_line.properties(height=350)

    st.altair_chart(chart, use_container_width=True)

def get_position_status(position, btc_price):
    """Get ITM/OTM status and distance from strike."""
    strike = position["strike"]
    side = position["side"]

    distance = btc_price - strike
    distance_pct = distance / strike * 100

    if side == "YES":
        # YES wins if BTC > strike at expiry
        is_itm = btc_price > strike
    else:
        # NO wins if BTC < strike at expiry
        is_itm = btc_price < strike

    return is_itm, distance, distance_pct

def render_positions(state, btc_price):
    """Render open positions table."""
    st.subheader("Open Positions")

    if state is None:
        return

    positions = state.get("positions", {})
    open_positions = [p for p in positions.values() if p.get("exit_time") is None]

    if not open_positions:
        st.info("No open positions")
        return

    now = datetime.now(timezone.utc)

    rows = []
    for p in sorted(open_positions, key=lambda x: x["settlement_time"]):
        settlement = datetime.fromisoformat(p["settlement_time"])
        time_left = settlement - now
        hours_left = time_left.total_seconds() / 3600

        # Format time remaining
        if hours_left < 1:
            time_str = f"{int(time_left.total_seconds() / 60)}m"
        elif hours_left < 24:
            time_str = f"{hours_left:.1f}h"
        else:
            time_str = f"{hours_left / 24:.1f}d"

        # Get status
        strike = p["strike"]
        if btc_price:
            is_itm, distance, distance_pct = get_position_status(p, btc_price)
            status = "✅ ITM" if is_itm else "❌ OTM"
            dist_str = f"{distance_pct:+.1f}%"
        else:
            status = "?"
            dist_str = "?"

        # Max profit/loss for this position
        cost = p['size'] * p['entry_price'] / 100
        max_profit = p['size'] * (100 - p['entry_price']) / 100

        rows.append({
            "Ticker": p["ticker"].replace("KXBTCD-", "").replace("-T", " $"),
            "Side": p["side"],
            "Strike": f"${p['strike']:,.0f}",
            "Size": p["size"],
            "Entry": f"{p['entry_price']}¢",
            "Cost": f"${cost:.2f}",
            "Max Profit": f"${max_profit:.2f}",
            "Edge": f"{p['edge']:.1%}",
            "Status": status,
            "Dist": dist_str,
            "Expires": time_str
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Summary
    total_cost = sum(p["size"] * p["entry_price"] / 100 for p in open_positions)
    total_max_profit = sum(p["size"] * (100 - p["entry_price"]) / 100 for p in open_positions)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cost", f"${total_cost:.2f}")
    with col2:
        st.metric("Max Profit", f"${total_max_profit:.2f}")
    with col3:
        itm_count = sum(1 for p in open_positions if
                       (p["side"] == "YES" and btc_price and btc_price > p["strike"]) or
                       (p["side"] == "NO" and btc_price and btc_price < p["strike"]))
        st.metric("Positions ITM", f"{itm_count}/{len(open_positions)}")

def render_settlements(state):
    """Render upcoming settlements."""
    st.subheader("Upcoming Settlements")

    if state is None:
        return

    positions = state.get("positions", {})
    open_positions = [p for p in positions.values() if p.get("exit_time") is None]

    if not open_positions:
        st.info("No upcoming settlements")
        return

    now = datetime.now(timezone.utc)

    # Group by settlement time
    by_settlement = {}
    for p in open_positions:
        settle_time = p["settlement_time"]
        if settle_time not in by_settlement:
            by_settlement[settle_time] = []
        by_settlement[settle_time].append(p)

    for settle_time in sorted(by_settlement.keys()):
        positions_at_time = by_settlement[settle_time]
        settlement = datetime.fromisoformat(settle_time)
        time_left = settlement - now
        hours_left = time_left.total_seconds() / 3600

        # Format
        if hours_left < 1:
            time_str = f"{int(time_left.total_seconds() / 60)} minutes"
        elif hours_left < 24:
            time_str = f"{hours_left:.1f} hours"
        else:
            time_str = f"{hours_left / 24:.1f} days"

        exposure = sum(p["size"] * p["entry_price"] / 100 for p in positions_at_time)

        with st.expander(f"📅 {settlement.strftime('%b %d %H:%M')} UTC - {len(positions_at_time)} positions (${exposure:.2f}) - {time_str}"):
            for p in positions_at_time:
                st.write(f"  • {p['side']} ${p['strike']:,.0f} - {p['size']}x @ {p['entry_price']}¢")

def render_equity_curve(equity_data):
    """Render equity curve chart."""
    st.subheader("Equity Curve")

    if not equity_data:
        st.info("No equity history yet")
        return

    df = pd.DataFrame(equity_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    st.line_chart(df["bankroll"], use_container_width=True)

def render_filter_stats(state):
    """Render V2 filter statistics."""
    st.subheader("Filter Stats")

    if state is None:
        return

    filtered = state.get("filtered_trades", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Bad Time Ratio", filtered.get("bad_time_ratio", 0))
    with col2:
        st.metric("Low Probability", filtered.get("low_probability", 0))
    with col3:
        st.metric("Uncertain Zone", filtered.get("uncertain_zone", 0))
    with col4:
        st.metric("Conflicts", filtered.get("conflict_prevention", 0))

def render_closed_positions(state):
    """Render closed positions summary."""
    st.subheader("Closed Positions")

    if state is None:
        return

    positions = state.get("positions", {})
    closed = [p for p in positions.values() if p.get("exit_time") is not None]

    if not closed:
        st.info("No closed positions yet")
        return

    wins = [p for p in closed if p.get("pnl", 0) > 0]
    losses = [p for p in closed if p.get("pnl", 0) <= 0]

    total_pnl = sum(p.get("pnl", 0) for p in closed)
    win_rate = len(wins) / len(closed) * 100 if closed else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total P&L", f"${total_pnl:+,.2f}")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Wins", len(wins))
    with col4:
        st.metric("Losses", len(losses))

    # Recent trades table
    if closed:
        recent = sorted(closed, key=lambda x: x.get("exit_time", ""), reverse=True)[:10]
        rows = []
        for p in recent:
            rows.append({
                "Ticker": p["ticker"].replace("KXBTCD-", ""),
                "Side": p["side"],
                "Entry": f"{p['entry_price']}¢",
                "Size": p["size"],
                "P&L": f"${p.get('pnl', 0):+.2f}",
                "Result": "✅ Win" if p.get("pnl", 0) > 0 else "❌ Loss"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# =============================================================================
# Main
# =============================================================================

def main():
    # Load data
    state = load_state()
    equity_data = load_equity_curve()
    btc_price = get_live_btc_price()
    price_history = get_btc_price_history()

    # Render header
    render_header(state, btc_price)

    st.divider()

    # Get open positions for chart
    open_positions = None
    if state:
        positions = state.get("positions", {})
        open_positions = [p for p in positions.values() if p.get("exit_time") is None]

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        render_btc_chart(price_history, open_positions)
        render_positions(state, btc_price)

    with col2:
        render_settlements(state)
        render_filter_stats(state)
        render_closed_positions(state)

    # Equity curve full width
    st.divider()
    render_equity_curve(equity_data)

    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | Auto-refresh: {REFRESH_INTERVAL}s")

if __name__ == "__main__":
    main()
