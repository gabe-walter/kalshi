#!/usr/bin/env python3
"""
Kalshi vs Options-Implied Probability Comparison

V5: Uses proper Breeden-Litzenberger method
- Extract risk-neutral CDF directly from option prices
- No lognormal assumption
- Captures skew, kurtosis, and fat tails as priced by the market
- Variance scaling to adjust for time mismatch between Deribit expiry and Kalshi settlement
"""

import sys
sys.path.insert(0, "src")

from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd
import numpy as np

from data.deribit_client import DeribitClient
from data.kalshi_client import KalshiClient
from models.breeden_litzenberger import extract_implied_cdf, find_best_expiry, parse_deribit_expiry


def run_comparison(
    min_edge: float = 0.03,
    min_bid_cents: int = 5,
    max_spread_cents: int = 20
):
    """Run comparison using Breeden-Litzenberger implied distribution."""

    print("=" * 85)
    print("Kalshi vs Options (Breeden-Litzenberger Method)")
    print(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Filters: min_bid >= {min_bid_cents}¢, max_spread <= {max_spread_cents}¢")
    print("=" * 85)

    now = datetime.now(timezone.utc)

    deribit = DeribitClient()
    kalshi = KalshiClient()

    # Get Deribit data
    print("\n[1] Deribit data...")
    spot = deribit.get_index_price("BTC")["index_price"]
    chain = deribit.get_option_chain("BTC")
    available_expiries = sorted(chain["expiry_str"].unique())

    print(f"    Spot: ${spot:,.2f}")
    print(f"    Option expiries: {available_expiries[:5]}")

    # Get Kalshi KXBTCD markets - fetch the 3 active settlement times
    print("\n[2] Kalshi KXBTCD markets...")

    # Build event tickers based on current time
    # Kalshi KXBTCD events: KXBTCD-YYMONDDTT where TT is hour in EST (0-23)
    # We want: hourly (next hour EST), daily (5pm EST today), weekly (5pm EST Friday)

    from datetime import timedelta

    # Convert to EST (UTC - 5)
    est_offset = timedelta(hours=-5)
    now_est = now + est_offset

    # Next hourly: round up to next hour in EST
    next_hour_est = (now_est.hour + 1) % 24
    hourly_day = now_est.day if next_hour_est > now_est.hour else now_est.day + 1

    # Format: 26JAN1812 means year 26, JAN, day 18, hour 12 (EST)
    year_short = str(now.year)[-2:]  # "26" for 2026
    month_str = now.strftime("%b").upper()  # "JAN"

    # Build the 3 event tickers
    hourly_event = f"KXBTCD-{year_short}{month_str}{hourly_day:02d}{next_hour_est:02d}"
    daily_event = f"KXBTCD-{year_short}{month_str}{now_est.day:02d}17"  # 5pm EST = hour 17

    # Friday's event - find next Friday
    days_until_friday = (4 - now.weekday()) % 7  # Friday = 4
    if days_until_friday == 0 and now_est.hour >= 17:
        days_until_friday = 7  # Next Friday if past 5pm
    friday = now + timedelta(days=days_until_friday)
    weekly_event = f"KXBTCD-{year_short}{month_str}{friday.day:02d}17"

    events_to_fetch = [hourly_event, daily_event, weekly_event]
    # Remove duplicates (e.g., if daily and weekly are same)
    events_to_fetch = list(dict.fromkeys(events_to_fetch))

    print(f"    Target events: {events_to_fetch}")

    # Fetch all markets for these events
    markets = []
    for event_ticker in events_to_fetch:
        try:
            event_markets = kalshi.get_markets(event_ticker=event_ticker, limit=200)
            fetched = event_markets.get('markets', [])
            # Only include active markets
            active = [m for m in fetched if m.get('status') == 'active']
            markets.extend(active)
            print(f"    {event_ticker}: {len(active)} active markets")
        except Exception as e:
            print(f"    {event_ticker}: Failed - {e}")

    print(f"    Total: {len(markets)} contracts")

    # Group by settlement
    by_settlement = defaultdict(list)
    for m in markets:
        close_time = m.get('close_time')
        if close_time:
            by_settlement[close_time].append(m)

    print(f"    Settlement times: {len(by_settlement)}")

    # Analyze each settlement
    print("\n[3] Analysis using Breeden-Litzenberger implied CDF...")
    all_opportunities = []

    for settlement_str in sorted(by_settlement.keys()):
        settlement_markets = by_settlement[settlement_str]
        settlement_time = pd.to_datetime(settlement_str, utc=True)

        if settlement_time <= now:
            continue

        hours_to_settlement = (settlement_time - now).total_seconds() / 3600

        # Find appropriate Deribit expiry
        best_expiry = find_best_expiry(available_expiries, settlement_time)
        if not best_expiry:
            print(f"\n  No valid expiry for settlement {settlement_time}")
            continue

        # Extract implied CDF using Breeden-Litzenberger
        implied_cdf = extract_implied_cdf(chain, spot, best_expiry)
        if implied_cdf is None:
            print(f"\n  Failed to extract CDF for {best_expiry}")
            continue

        # Calculate time to settlement in years for variance scaling
        years_to_settlement = hours_to_settlement / (365.25 * 24)
        time_scale = np.sqrt(implied_cdf.time_to_expiry / years_to_settlement) if years_to_settlement > 0 else 1.0

        print(f"\n{'='*85}")
        print(f"Settlement: {settlement_time.strftime('%Y-%m-%d %H:%M')} UTC (T-{hours_to_settlement:.1f}h)")
        print(f"Using Deribit {best_expiry} options (T-{implied_cdf.time_to_expiry*365.25*24:.1f}h) | Time scale: {time_scale:.2f}x")
        print(f"Strike range: ${implied_cdf.strikes.min():,.0f} - ${implied_cdf.strikes.max():,.0f}")
        print("-" * 85)
        print(f"{'Strike':>10} {'YES B/A':>12} {'NO B/A':>12} {'Model':>8} {'YES Edge':>10} {'NO Edge':>10} {'Signal':<10}")
        print("-" * 85)

        # Sort by strike descending
        sorted_markets = sorted(settlement_markets, key=lambda x: x.get('floor_strike', 0), reverse=True)

        for m in sorted_markets:
            strike = m.get('floor_strike')
            if not strike:
                continue

            # Check if strike is within our CDF range
            if strike < implied_cdf.strikes.min() or strike > implied_cdf.strikes.max():
                continue

            # Get both YES and NO prices
            yes_bid = m.get('yes_bid', 0) or 0
            yes_ask = m.get('yes_ask', 0) or 0
            no_bid = m.get('no_bid', 0) or 0
            no_ask = m.get('no_ask', 0) or 0

            yes_spread = yes_ask - yes_bid
            no_spread = no_ask - no_bid

            # Convert to probabilities (prices are in cents)
            yes_ask_prob = yes_ask / 100.0
            no_ask_prob = no_ask / 100.0

            # Model probability from Breeden-Litzenberger CDF with time adjustment
            model_prob = implied_cdf.prob_above_at_time(strike, years_to_settlement)
            model_prob_no = 1 - model_prob

            # Calculate edges (what we'd pay vs what model says it's worth)
            # BUY YES: pay yes_ask, worth model_prob
            # BUY NO: pay no_ask, worth (1 - model_prob)
            edge_yes = model_prob - yes_ask_prob
            edge_no = model_prob_no - no_ask_prob

            # Determine signal based on which side has tradeable edge
            signal = "-"
            best_edge = 0
            trade_side = None

            # Check YES side: need liquidity (yes_bid > 0) and reasonable spread
            if yes_bid >= min_bid_cents and yes_spread <= max_spread_cents and edge_yes >= min_edge:
                signal = "BUY YES"
                best_edge = edge_yes
                trade_side = "YES"

            # Check NO side: need liquidity (no_bid > 0) and reasonable spread
            if no_bid >= min_bid_cents and no_spread <= max_spread_cents and edge_no >= min_edge:
                # If both sides have edge, pick the better one
                if edge_no > best_edge:
                    signal = "BUY NO"
                    best_edge = edge_no
                    trade_side = "NO"

            # Skip if neither side passes liquidity filter
            yes_liquid = yes_bid >= min_bid_cents and yes_spread <= max_spread_cents
            no_liquid = no_bid >= min_bid_cents and no_spread <= max_spread_cents
            if not yes_liquid and not no_liquid:
                continue

            # Format output
            yes_ba_str = f"{yes_bid:>2}/{yes_ask:<2}¢"
            no_ba_str = f"{no_bid:>2}/{no_ask:<2}¢"
            edge_yes_str = f"{edge_yes:>+7.1%}" if yes_liquid else "  illiq"
            edge_no_str = f"{edge_no:>+7.1%}" if no_liquid else "  illiq"

            print(f"${strike:>9,.0f} {yes_ba_str:>12} {no_ba_str:>12} {model_prob:>7.1%} "
                  f"{edge_yes_str:>10} {edge_no_str:>10} {signal:<10}")

            if signal != "-":
                all_opportunities.append({
                    'ticker': m.get('ticker'),
                    'settlement': settlement_time,
                    'strike': strike,
                    'yes_bid': yes_bid,
                    'yes_ask': yes_ask,
                    'no_bid': no_bid,
                    'no_ask': no_ask,
                    'model_prob': model_prob,
                    'edge': best_edge,
                    'signal': signal,
                    'hours': hours_to_settlement,
                    'deribit_expiry': best_expiry,
                    'time_scale': time_scale
                })

    # Summary
    print("\n" + "=" * 85)
    print("SUMMARY")
    print("=" * 85)
    print(f"Opportunities found (>= {min_edge:.0%} edge): {len(all_opportunities)}")

    if all_opportunities:
        print("\nTop opportunities:")
        for opp in sorted(all_opportunities, key=lambda x: x['edge'], reverse=True)[:10]:
            if opp['signal'] == "BUY YES":
                price_str = f"YES @ {opp['yes_ask']}¢"
            else:
                price_str = f"NO @ {opp['no_ask']}¢"
            print(f"  ${opp['strike']:,.0f}: {price_str} | "
                  f"Model {opp['model_prob']:.1%} | Edge {opp['edge']:+.1%} → {opp['signal']}")

    return all_opportunities


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--min-bid", type=int, default=5)
    parser.add_argument("--max-spread", type=int, default=20)
    args = parser.parse_args()

    run_comparison(args.min_edge, args.min_bid, args.max_spread)
