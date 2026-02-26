#!/usr/bin/env python3
"""
Post-Hoc Analysis Dashboard for Kalshi Crypto Options Strategy

Offline analysis of collected data: P&L, model calibration, risk metrics,
and Monte Carlo robustness testing.

Run with: streamlit run analysis_dashboard.py
"""

import sys
sys.path.insert(0, "src")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from analysis.data_loader import (
    load_enriched_snapshots,
    load_ws_tickers,
    load_trade_log,
    load_equity_curve,
    load_positions,
    load_spot_prices,
    WS_COLLECTOR_DIR,
    PAPER_TRADER_DIR,
)
from analysis.pnl import (
    compute_equity_curve,
    compute_trade_returns,
    pnl_by_bucket,
    cumulative_pnl_series,
    expected_vs_realized_edge,
)
from analysis.calibration import (
    build_calibration_dataset,
    calibration_curve,
    calibration_summary,
    mispricing_distribution,
    model_vs_market_scatter,
    edge_decay_by_time,
)
from analysis.risk_metrics import (
    compute_risk_summary,
    max_drawdown_analysis,
    kelly_analysis,
    exposure_over_time,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
    expected_shortfall,
)
from analysis.monte_carlo import (
    simulate,
    sensitivity_analysis,
    fan_chart_data,
)

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Strategy Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Sidebar — Data Source Selection
# =============================================================================

st.sidebar.title("Data Configuration")

data_source = st.sidebar.selectbox(
    "Data Source",
    ["Paper Trader V3", "WebSocket Collector"],
)

if data_source == "WebSocket Collector":
    base_dir = WS_COLLECTOR_DIR
else:
    base_dir = PAPER_TRADER_DIR

start_date = st.sidebar.text_input("Start Date (YYYYMMDD)", value="")
end_date = st.sidebar.text_input("End Date (YYYYMMDD)", value="")
start_date = start_date if start_date else None
end_date = end_date if end_date else None

# =============================================================================
# Data Loading (cached)
# =============================================================================

@st.cache_data(ttl=300)
def load_all_data(base_dir_str, start_date, end_date):
    base = Path(base_dir_str)
    enriched = load_enriched_snapshots(base, start_date, end_date)
    # Load trade data from the selected data source
    trade_log = load_trade_log(base)
    equity = load_equity_curve(base)
    positions = load_positions(base)
    spots = load_spot_prices(base, start_date, end_date)
    return enriched, trade_log, equity, positions, spots


enriched, trade_log, equity_raw, positions, spots = load_all_data(
    str(base_dir), start_date, end_date
)

# Pre-compute common derivatives
equity = compute_equity_curve(equity_raw)
closed_positions = compute_trade_returns(positions) if not positions.empty else pd.DataFrame()

# =============================================================================
# Navigation
# =============================================================================

page = st.sidebar.radio(
    "Page",
    ["P&L Analysis", "Model Calibration", "Risk Metrics", "Monte Carlo"],
)

# =============================================================================
# PAGE 1: P&L Analysis
# =============================================================================

if page == "P&L Analysis":
    st.title("P&L Analysis")

    # ---- Header metrics ----
    if not closed_positions.empty:
        edge_stats = expected_vs_realized_edge(closed_positions)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total P&L", f"${edge_stats.get('total_pnl', 0):,.2f}")
        col2.metric("Trades", edge_stats.get("n_trades", 0))
        col3.metric("Win Rate", f"{edge_stats.get('win_rate', 0):.1f}%")
        col4.metric(
            "Realized Edge",
            f"{edge_stats.get('realized_edge_pct', 0):.2f}%",
        )
        exp_edge = edge_stats.get("expected_edge_pct")
        col5.metric(
            "Expected Edge",
            f"{exp_edge:.2f}%" if exp_edge is not None else "N/A",
        )
    else:
        st.info("No closed positions found. Run the paper trader to generate trade data.")

    st.divider()

    # ---- Equity Curve ----
    st.subheader("Equity Curve")
    if not equity.empty:
        eq_chart = (
            alt.Chart(equity)
            .mark_area(opacity=0.15, color="#00D4AA")
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("bankroll:Q", title="Bankroll ($)",
                         scale=alt.Scale(zero=False)),
            )
        )
        eq_line = (
            alt.Chart(equity)
            .mark_line(color="#00D4AA", strokeWidth=2)
            .encode(
                x="timestamp:T",
                y="bankroll:Q",
            )
        )
        st.altair_chart((eq_chart + eq_line).properties(height=350),
                        use_container_width=True)

        # Drawdown subplot
        st.subheader("Drawdown")
        dd_data = equity[equity["drawdown"] > 0]
        if not dd_data.empty:
            dd_chart = (
                alt.Chart(dd_data)
                .mark_area(color="#FF4444", opacity=0.4)
                .encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("drawdown_pct:Q", title="Drawdown %",
                             scale=alt.Scale(reverse=True)),
                )
                .properties(height=200)
            )
            st.altair_chart(dd_chart, use_container_width=True)
        else:
            st.info("No drawdowns recorded yet.")
    else:
        st.info("No equity curve data available.")

    st.divider()

    # ---- Cumulative P&L from trades ----
    st.subheader("Cumulative P&L (per trade)")
    if not closed_positions.empty:
        cum_pnl = cumulative_pnl_series(closed_positions)
        if not cum_pnl.empty:
            pnl_chart = (
                alt.Chart(cum_pnl)
                .mark_line(color="#00D4AA", strokeWidth=2)
                .encode(
                    x=alt.X("exit_time:T", title="Settlement Time"),
                    y=alt.Y("cumulative_pnl:Q", title="Cumulative P&L ($)"),
                )
                .properties(height=300)
            )
            zero_line = (
                alt.Chart(pd.DataFrame({"y": [0]}))
                .mark_rule(color="gray", strokeDash=[5, 5])
                .encode(y="y:Q")
            )
            st.altair_chart((pnl_chart + zero_line), use_container_width=True)

    st.divider()

    # ---- Bucketed Analysis ----
    st.subheader("Performance by Category")
    if not closed_positions.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**By Asset**")
            by_asset = pnl_by_bucket(closed_positions, "asset")
            if not by_asset.empty:
                st.dataframe(by_asset, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("**By Side**")
            by_side = pnl_by_bucket(closed_positions, "side")
            if not by_side.empty:
                st.dataframe(by_side, use_container_width=True, hide_index=True)

        # Trade details table
        st.subheader("Trade Log")
        display_cols = ["ticker", "asset", "side", "strike", "entry_price",
                        "size", "pnl", "edge", "entry_time", "exit_time"]
        available = [c for c in display_cols if c in closed_positions.columns]
        st.dataframe(
            closed_positions[available].sort_values("exit_time", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# =============================================================================
# PAGE 2: Model Calibration
# =============================================================================

elif page == "Model Calibration":
    st.title("Model Calibration")

    # Build calibration dataset
    calib_df = build_calibration_dataset(enriched, positions)

    has_settlements = not calib_df.empty and "settled_yes" in calib_df.columns and calib_df["settled_yes"].notna().any()

    # ---- Calibration Curve ----
    if has_settlements:
        st.subheader("Calibration Curve")
        st.caption("Predicted probability vs actual settlement frequency. "
                   "Perfect calibration lies on the diagonal.")

        pred = calib_df.loc[calib_df["settled_yes"].notna(), "model_prob"].values
        actual = calib_df.loc[calib_df["settled_yes"].notna(), "settled_yes"].values

        cal_data = calibration_curve(pred, actual, n_bins=10)
        if not cal_data.empty:
            scatter = (
                alt.Chart(cal_data)
                .mark_circle(size=100, color="#00D4AA")
                .encode(
                    x=alt.X("avg_predicted:Q", title="Model Predicted Probability",
                             scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("avg_actual:Q", title="Actual YES Rate",
                             scale=alt.Scale(domain=[0, 1])),
                    size=alt.Size("count:Q", legend=None),
                    tooltip=["bin_label", "avg_predicted", "avg_actual", "count", "bias"],
                )
            )
            diagonal = (
                alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                .mark_line(color="gray", strokeDash=[5, 5])
                .encode(x="x:Q", y="y:Q")
            )
            st.altair_chart((scatter + diagonal).properties(height=400),
                            use_container_width=True)

            # Summary stats
            summary = calibration_summary(pred, actual)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Brier Score", f"{summary['brier_score']:.4f}")
            col2.metric("Log Loss", f"{summary['log_loss']:.4f}")
            col3.metric("Avg Predicted", f"{summary['avg_predicted']:.4f}")
            col4.metric("Bias", f"{summary['bias']:+.4f}")

            # Calibration table
            st.dataframe(cal_data, use_container_width=True, hide_index=True)
    else:
        st.info("No settlement outcomes available for calibration curves. "
                "Requires enriched snapshots with contracts that have settled.")

    st.divider()

    # ---- Model vs Market Scatter ----
    st.subheader("Model Probability vs Market Price")
    st.caption("Points above diagonal = model thinks it's more likely than market (underpriced, good to buy)")
    scatter_data = model_vs_market_scatter(enriched)

    if not scatter_data.empty:
        # model_prob_no and market_prob_no are now computed correctly in model_vs_market_scatter()
        # market_prob_no = (100 - yes_bid) / 100 = no_ask / 100

        # Subsample if too many points
        if len(scatter_data) > 5000:
            scatter_data = scatter_data.sample(5000, random_state=42)

        diagonal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(color="red", strokeDash=[5, 5])
            .encode(x="x:Q", y="y:Q")
        )

        st.markdown("**Buy YES opportunities**")
        scatter_yes = (
            alt.Chart(scatter_data)
            .mark_circle(size=10, opacity=0.3)
            .encode(
                x=alt.X("market_prob:Q", title="Market (yes_ask/100)",
                         scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("model_prob:Q", title="Model P(YES)",
                         scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("asset:N", legend=alt.Legend(title="Asset")),
                tooltip=["market_ticker", "model_prob", "market_prob", "asset"],
            )
        )
        st.altair_chart(
            (scatter_yes + diagonal).properties(height=400),
            use_container_width=True,
        )

        st.markdown("**Buy NO opportunities**")
        scatter_no = (
            alt.Chart(scatter_data)
            .mark_circle(size=10, opacity=0.3)
            .encode(
                x=alt.X("market_prob_no:Q", title="Market (no_ask/100)",
                         scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("model_prob_no:Q", title="Model P(NO)",
                         scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("asset:N", legend=alt.Legend(title="Asset")),
                tooltip=["market_ticker", "model_prob_no", "market_prob_no", "asset"],
            )
        )
        st.altair_chart(
            (scatter_no + diagonal).properties(height=400),
            use_container_width=True,
        )
    else:
        st.info("No enriched data with model probabilities and market prices.")

    st.divider()

    # ---- Single Market Time Series ----
    st.subheader("Single Market Edge Over Time")
    st.caption("Track how model vs market divergence evolves for a specific contract. "
               "Persistent gaps = difference of opinion. Brief spikes = transient edge.")

    # Try to load raw ticker data (more granular) - falls back to enriched snapshots
    raw_tickers = load_ws_tickers(base_dir, start_date, end_date)
    use_raw_data = (not raw_tickers.empty and "model_prob" in raw_tickers.columns
                    and raw_tickers["model_prob"].notna().any())

    if use_raw_data:
        time_series_data = raw_tickers.copy()
        time_series_data = time_series_data.rename(columns={"received_at": "snapshot_time"})
        data_source_label = "raw ticker updates (second-by-second)"
    elif not enriched.empty and "market_ticker" in enriched.columns:
        time_series_data = enriched.copy()
        data_source_label = "enriched snapshots (5-min intervals)"
    else:
        time_series_data = pd.DataFrame()
        data_source_label = None

    if not time_series_data.empty and "market_ticker" in time_series_data.columns:
        # Get markets that have multiple data points
        market_counts = time_series_data.groupby("market_ticker").size()
        markets_with_history = market_counts[market_counts > 1].index.tolist()

        if markets_with_history:
            st.caption(f"Data source: {data_source_label}")

            selected_market = st.selectbox(
                "Select market to analyze",
                options=sorted(markets_with_history),
                index=0
            )

            market_history = time_series_data[time_series_data["market_ticker"] == selected_market].copy()
            market_history = market_history.sort_values("snapshot_time")

            # Filter to rows with model_prob
            market_history = market_history[market_history["model_prob"].notna()]

            if not market_history.empty:
                # Compute all the probability series
                market_history["model_yes"] = market_history["model_prob"] * 100
                market_history["market_yes"] = market_history["yes_ask"]
                market_history["model_no"] = (1 - market_history["model_prob"]) * 100
                market_history["market_no"] = 100 - market_history["yes_bid"]

                # Melt for plotting
                plot_data = market_history[["snapshot_time", "model_yes", "market_yes", "model_no", "market_no"]].melt(
                    id_vars=["snapshot_time"],
                    var_name="series",
                    value_name="price"
                )

                # YES side chart
                yes_data = plot_data[plot_data["series"].isin(["model_yes", "market_yes"])]
                if not yes_data.empty:
                    st.markdown("**YES Side (model vs yes_ask)**")
                    yes_chart = (
                        alt.Chart(yes_data)
                        .mark_line(point=False)
                        .encode(
                            x=alt.X("snapshot_time:T", title="Time"),
                            y=alt.Y("price:Q", title="Price (cents)", scale=alt.Scale(zero=False)),
                            color=alt.Color("series:N", legend=alt.Legend(title=""),
                                           scale=alt.Scale(domain=["model_yes", "market_yes"],
                                                          range=["#00D4AA", "#FF6B6B"])),
                            tooltip=["snapshot_time:T", "series", "price"]
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(yes_chart, use_container_width=True)

                # NO side chart
                no_data = plot_data[plot_data["series"].isin(["model_no", "market_no"])]
                if not no_data.empty:
                    st.markdown("**NO Side (model vs no_ask)**")
                    no_chart = (
                        alt.Chart(no_data)
                        .mark_line(point=False)
                        .encode(
                            x=alt.X("snapshot_time:T", title="Time"),
                            y=alt.Y("price:Q", title="Price (cents)", scale=alt.Scale(zero=False)),
                            color=alt.Color("series:N", legend=alt.Legend(title=""),
                                           scale=alt.Scale(domain=["model_no", "market_no"],
                                                          range=["#00D4AA", "#FF6B6B"])),
                            tooltip=["snapshot_time:T", "series", "price"]
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(no_chart, use_container_width=True)

                # Show data point count
                st.caption(f"Showing {len(market_history)} data points from "
                          f"{market_history['snapshot_time'].min()} to {market_history['snapshot_time'].max()}")
            else:
                st.info("No model probabilities available for this market yet.")
        else:
            st.info("Need multiple data points for the same market to show time series.")
    else:
        st.info("No data available for time series analysis.")

    st.divider()

    # ---- Mispricing Distribution ----
    st.subheader("Mispricing Distribution (Buy Opportunities)")
    st.caption("Positive = underpriced (good to buy). Negative = overpriced (avoid).")

    st.markdown("**Buy YES Mispricing**")
    misp_yes = mispricing_distribution(enriched, "mispricing_yes")
    if not misp_yes.empty:
        hist_yes = (
            alt.Chart(misp_yes)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("mispricing_yes:Q", bin=alt.Bin(maxbins=50),
                         title="model_yes - yes_ask (cents)"),
                y=alt.Y("count()", title="Count"),
                color=alt.Color("asset:N"),
            )
            .properties(height=300)
        )
        zero_line = (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(x="x:Q")
        )
        st.altair_chart((hist_yes + zero_line), use_container_width=True)
        st.metric("Mean", f"{misp_yes['mispricing_yes'].mean():.2f}¢")

    st.markdown("**Buy NO Mispricing**")
    misp_no = mispricing_distribution(enriched, "mispricing_no")
    if not misp_no.empty:
        hist_no = (
            alt.Chart(misp_no)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("mispricing_no:Q", bin=alt.Bin(maxbins=50),
                         title="model_no - no_ask (cents)"),
                y=alt.Y("count()", title="Count"),
                color=alt.Color("asset:N"),
            )
            .properties(height=300)
        )
        zero_line = (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(x="x:Q")
        )
        st.altair_chart((hist_no + zero_line), use_container_width=True)
        st.metric("Mean", f"{misp_no['mispricing_no'].mean():.2f}¢")

    if misp_yes.empty and misp_no.empty:
        st.info("No mispricing data available.")

    st.divider()

    # ---- Edge Decay ----
    st.subheader("Mispricing vs Time to Settlement")
    st.caption("Does the model's edge shrink as contracts approach expiry?")
    edge_decay = edge_decay_by_time(enriched)
    if not edge_decay.empty:
        st.dataframe(edge_decay, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 3: Risk Metrics
# =============================================================================

elif page == "Risk Metrics":
    st.title("Risk Metrics")

    # ---- Summary Cards ----
    risk = compute_risk_summary(positions, equity_raw)

    if risk:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe Ratio", f"{risk.get('sharpe', 0):.2f}")
        col2.metric("Sortino Ratio", f"{risk.get('sortino', 0):.2f}")
        col3.metric("Max Drawdown", f"${risk.get('max_drawdown', 0):,.2f}")
        col4.metric("Max DD %", f"{risk.get('max_drawdown_pct', 0):.2f}%")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VaR 95%", f"${risk.get('var_95', 0):,.2f}")
        col2.metric("VaR 99%", f"${risk.get('var_99', 0):,.2f}")
        col3.metric("Exp. Shortfall 95%", f"${risk.get('es_95', 0):,.2f}")
        col4.metric("Win Rate", f"{risk.get('win_rate', 0):.1f}%")
    else:
        st.info("No closed positions to compute risk metrics from.")

    st.divider()

    # ---- Drawdown Analysis ----
    st.subheader("Drawdown Analysis")
    dd_analysis = max_drawdown_analysis(equity_raw)
    if dd_analysis:
        col1, col2, col3 = st.columns(3)
        col1.metric("Peak", f"${dd_analysis.get('peak_value', 0):,.2f}")
        col2.metric("Trough", f"${dd_analysis.get('trough_value', 0):,.2f}")
        col3.metric("Recovered", "Yes" if dd_analysis.get("recovered") else "No")

        if dd_analysis.get("drawdown_duration_hours") is not None:
            st.caption(f"Drawdown duration: {dd_analysis['drawdown_duration_hours']:.1f} hours")

    st.divider()

    # ---- Kelly Analysis ----
    st.subheader("Kelly Criterion Analysis")
    kelly = kelly_analysis(positions)
    if kelly:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Win Rate", f"{kelly['empirical_win_rate']*100:.1f}%")
        col2.metric("Payoff Ratio", f"{kelly['payoff_ratio']:.2f}")
        col3.metric("Kelly f*", f"{kelly['kelly_fraction']*100:.1f}%")
        col4.metric("3/4 Kelly", f"{kelly['three_quarter_kelly']*100:.1f}%")

        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Avg Win | ${kelly['avg_win']:.2f} |
        | Avg Loss | ${kelly['avg_loss']:.2f} |
        | Full Kelly | {kelly['kelly_fraction']*100:.1f}% of bankroll |
        | Half Kelly | {kelly['half_kelly']*100:.1f}% |
        | 3/4 Kelly (current) | {kelly['three_quarter_kelly']*100:.1f}% |
        """)

        if kelly.get("avg_actual_fraction") is not None:
            st.metric("Avg Actual Bet Size (% of bankroll)",
                      f"{kelly['avg_actual_fraction']*100:.2f}%")
    else:
        st.info("Not enough trade data for Kelly analysis.")

    st.divider()

    # ---- Exposure Over Time ----
    st.subheader("Capital Exposure Over Time")
    exp_df = exposure_over_time(equity_raw)
    if not exp_df.empty:
        base = alt.Chart(exp_df).encode(x=alt.X("timestamp:T", title="Time"))
        bankroll_line = base.mark_line(color="#00D4AA", strokeWidth=2).encode(
            y=alt.Y("bankroll:Q", title="$ Amount"),
        )
        exposure_area = base.mark_area(color="#FF6B6B", opacity=0.3).encode(
            y=alt.Y("open_exposure:Q"),
        )
        st.altair_chart(
            (bankroll_line + exposure_area).properties(height=300),
            use_container_width=True,
        )

        # Exposure percentage
        exp_pct_chart = (
            alt.Chart(exp_df)
            .mark_area(color="#FF6B6B", opacity=0.4)
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("exposure_pct:Q", title="Exposure %"),
            )
            .properties(height=200)
        )
        st.altair_chart(exp_pct_chart, use_container_width=True)
    else:
        st.info("No exposure data available.")

    st.divider()

    # ---- P&L Distribution ----
    st.subheader("P&L Distribution")
    if not closed_positions.empty:
        pnl_hist = (
            alt.Chart(closed_positions)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("pnl:Q", bin=alt.Bin(maxbins=30), title="P&L ($)"),
                y=alt.Y("count()", title="Count"),
                color=alt.condition(
                    alt.datum.pnl > 0,
                    alt.value("#00D4AA"),
                    alt.value("#FF4444"),
                ),
            )
            .properties(height=300)
        )
        st.altair_chart(pnl_hist, use_container_width=True)

# =============================================================================
# PAGE 4: Monte Carlo Simulation
# =============================================================================

elif page == "Monte Carlo":
    st.title("Monte Carlo Simulation")
    st.caption("Resamples historical trades to simulate thousands of equity paths "
               "for robustness testing.")

    if closed_positions.empty or closed_positions["pnl"].isna().all():
        st.warning("Need closed positions with P&L data. Run the paper trader first.")
    else:
        # Controls
        col1, col2, col3 = st.columns(3)
        n_paths = col1.slider("Simulation Paths", 1000, 50000, 10000, step=1000)
        n_trades = col2.slider("Trades per Path", 10, 500,
                                min(100, len(closed_positions) * 2), step=10)
        edge_mult = col3.slider("Edge Multiplier", 0.25, 1.5, 1.0, step=0.25)

        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running Monte Carlo simulation..."):
                result = simulate(
                    closed_positions,
                    n_paths=n_paths,
                    n_trades=n_trades,
                    edge_multiplier=edge_mult,
                    seed=42,
                )

            st.divider()

            # ---- Summary ----
            st.subheader("Simulation Results")
            stats = result.final_bankroll_stats()
            dd_stats = result.drawdown_stats()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Median Final Bankroll", f"${stats['median']:,.2f}")
            col2.metric("Mean Final Bankroll", f"${stats['mean']:,.2f}")
            col3.metric("5th Percentile", f"${stats['p5']:,.2f}")
            col4.metric("95th Percentile", f"${stats['p95']:,.2f}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("P(Ruin >50%)", f"{result.ruin_probability(50)*100:.1f}%")
            col2.metric("P(Ruin >25%)", f"{result.ruin_probability(25)*100:.1f}%")
            col3.metric("Median Max DD", f"{dd_stats['median']:.1f}%")
            col4.metric("95th %ile Max DD", f"{dd_stats['p95']:.1f}%")

            st.divider()

            # ---- Fan Chart ----
            st.subheader("Equity Path Fan Chart")
            fan_df = fan_chart_data(result, [5, 25, 50, 75, 95])

            # Build layered chart
            band_5_95 = (
                alt.Chart(
                    fan_df[fan_df["percentile"].isin(["p5", "p95"])]
                    .pivot(index="trade_number", columns="percentile", values="bankroll")
                    .reset_index()
                )
                .mark_area(opacity=0.15, color="#00D4AA")
                .encode(
                    x=alt.X("trade_number:Q", title="Trade #"),
                    y=alt.Y("p5:Q", title="Bankroll ($)"),
                    y2="p95:Q",
                )
            )
            band_25_75 = (
                alt.Chart(
                    fan_df[fan_df["percentile"].isin(["p25", "p75"])]
                    .pivot(index="trade_number", columns="percentile", values="bankroll")
                    .reset_index()
                )
                .mark_area(opacity=0.25, color="#00D4AA")
                .encode(
                    x="trade_number:Q",
                    y="p25:Q",
                    y2="p75:Q",
                )
            )
            median_line = (
                alt.Chart(fan_df[fan_df["percentile"] == "p50"])
                .mark_line(color="#00D4AA", strokeWidth=2)
                .encode(
                    x="trade_number:Q",
                    y=alt.Y("bankroll:Q"),
                )
            )
            baseline = (
                alt.Chart(pd.DataFrame({"y": [result.initial_bankroll]}))
                .mark_rule(color="gray", strokeDash=[5, 5])
                .encode(y="y:Q")
            )

            st.altair_chart(
                (band_5_95 + band_25_75 + median_line + baseline).properties(height=400),
                use_container_width=True,
            )

            st.divider()

            # ---- Final Bankroll Distribution ----
            st.subheader("Final Bankroll Distribution")
            final_df = pd.DataFrame({"final_bankroll": result.final_bankrolls})
            hist = (
                alt.Chart(final_df)
                .mark_bar(opacity=0.7, color="#00D4AA")
                .encode(
                    x=alt.X("final_bankroll:Q", bin=alt.Bin(maxbins=50),
                             title="Final Bankroll ($)"),
                    y=alt.Y("count()", title="Count"),
                )
                .properties(height=300)
            )
            initial_line = (
                alt.Chart(pd.DataFrame({"x": [result.initial_bankroll]}))
                .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=2)
                .encode(x="x:Q")
            )
            st.altair_chart((hist + initial_line), use_container_width=True)

            st.divider()

            # ---- Max Drawdown Distribution ----
            st.subheader("Max Drawdown Distribution")
            dd_df = pd.DataFrame({"max_dd_pct": result.max_drawdown_pcts})
            dd_hist = (
                alt.Chart(dd_df)
                .mark_bar(opacity=0.7, color="#FF6B6B")
                .encode(
                    x=alt.X("max_dd_pct:Q", bin=alt.Bin(maxbins=50),
                             title="Max Drawdown %"),
                    y=alt.Y("count()", title="Count"),
                )
                .properties(height=300)
            )
            st.altair_chart(dd_hist, use_container_width=True)

            st.divider()

            # ---- Sensitivity Analysis ----
            st.subheader("Edge Sensitivity Analysis")
            st.caption("What if your edge is smaller than you think?")

            with st.spinner("Running sensitivity analysis..."):
                sens = sensitivity_analysis(
                    closed_positions,
                    edge_multipliers=[0.25, 0.5, 0.75, 1.0, 1.25],
                    n_paths=5000,
                    n_trades=n_trades,
                    seed=42,
                )

            if not sens.empty:
                st.dataframe(
                    sens.style.format({
                        "mean_final": "${:,.2f}",
                        "median_final": "${:,.2f}",
                        "p5_final": "${:,.2f}",
                        "p95_final": "${:,.2f}",
                        "ruin_prob_50pct": "{:.1%}",
                        "ruin_prob_25pct": "{:.1%}",
                        "median_max_dd_pct": "{:.1f}%",
                        "p95_max_dd_pct": "{:.1f}%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                # Sensitivity chart
                sens_chart = (
                    alt.Chart(sens)
                    .mark_bar(opacity=0.7)
                    .encode(
                        x=alt.X("edge_label:N", title="Edge Assumption",
                                 sort=alt.EncodingSortField(field="edge_multiplier")),
                        y=alt.Y("median_final:Q", title="Median Final Bankroll ($)"),
                        color=alt.condition(
                            alt.datum.median_final > 10000,
                            alt.value("#00D4AA"),
                            alt.value("#FF4444"),
                        ),
                        tooltip=["edge_label", "median_final", "ruin_prob_50pct",
                                 "p95_max_dd_pct"],
                    )
                    .properties(height=300)
                )
                baseline = (
                    alt.Chart(pd.DataFrame({"y": [10000]}))
                    .mark_rule(color="gray", strokeDash=[5, 5])
                    .encode(y="y:Q")
                )
                st.altair_chart((sens_chart + baseline), use_container_width=True)

# =============================================================================
# Footer
# =============================================================================

st.divider()
data_summary = []
if not enriched.empty:
    data_summary.append(f"Enriched: {len(enriched):,} rows")
if not closed_positions.empty:
    data_summary.append(f"Closed trades: {len(closed_positions)}")
if not equity.empty:
    data_summary.append(f"Equity points: {len(equity)}")

st.caption(f"Data loaded: {' | '.join(data_summary) if data_summary else 'No data'}")
