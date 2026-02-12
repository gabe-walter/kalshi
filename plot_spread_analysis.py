#!/usr/bin/env python3
"""
Plot the spread between Kalshi and Deribit implied probabilities over time,
alongside estimated BTC spot price.

Uses trade log data to:
1. Estimate BTC spot price from strike/probability relationships
2. Show the Kalshi vs Deribit probability spread (edge) over time
3. Overlay to test hypothesis: bigger spreads during big BTC moves
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load trade logs
def load_trades(path):
    with open(path) as f:
        trades = json.load(f)
    for t in trades:
        t['dt'] = datetime.fromisoformat(t['timestamp'])
    return trades

v1_trades = load_trades('data/paper_trading/trade_log.json')
v2_trades = load_trades('data/paper_trading_v2/trade_log.json')

all_trades = v1_trades + v2_trades
all_trades.sort(key=lambda t: t['dt'])

# Only OPEN trades (not closes/settlements)
opens = [t for t in all_trades if t['action'] == 'OPEN']

print(f"V1 trades: {len(v1_trades)}, V2 trades: {len(v2_trades)}")
print(f"Total OPEN trades: {len(opens)}")
print(f"Date range: {opens[0]['dt'].date()} to {opens[-1]['dt'].date()}")

# ── Estimate BTC spot price at each timestamp ──
# Strategy: at each timestamp, we have trades at various strikes with model_prob.
# The strike where model_prob ≈ 0.5 is roughly where BTC spot is.
# We interpolate between the two closest strikes to 0.5 probability.
# Group trades by approximate time window (round to nearest 5 minutes)

def round_time(dt, minutes=30):
    """Round datetime to nearest N minutes for grouping."""
    ts = dt.timestamp()
    rounded = round(ts / (minutes * 60)) * (minutes * 60)
    return datetime.fromtimestamp(rounded, tz=timezone.utc)

# Group trades into time windows
time_groups = defaultdict(list)
for t in opens:
    key = round_time(t['dt'], minutes=30)
    time_groups[key].append(t)

# Estimate spot from each group
spot_estimates = []
for time_key, group in sorted(time_groups.items()):
    # Get unique strikes with their model probabilities
    strike_probs = {}
    for t in group:
        s = t['strike']
        p = t['model_prob']
        # If we have both YES and NO for same strike, use model_prob directly
        strike_probs[s] = p

    if len(strike_probs) < 2:
        continue

    strikes = sorted(strike_probs.keys())
    probs = [strike_probs[s] for s in strikes]

    # Find where probability crosses 0.5 (spot estimate)
    # model_prob = P(BTC > strike), so it decreases as strike increases
    # Find pair where prob goes from >0.5 to <0.5
    spot = None
    for i in range(len(strikes) - 1):
        p1, p2 = probs[i], probs[i + 1]
        s1, s2 = strikes[i], strikes[i + 1]
        if (p1 >= 0.5 and p2 < 0.5) or (p1 < 0.5 and p2 >= 0.5):
            # Linear interpolation
            if abs(p1 - p2) > 0.01:
                frac = (0.5 - p2) / (p1 - p2)
                spot = s2 + frac * (s1 - s2)
            else:
                spot = (s1 + s2) / 2
            break

    if spot is None:
        # Fallback: use weighted average of strikes by closeness to 0.5
        weights = [1 / (abs(p - 0.5) + 0.01) for p in probs]
        total_w = sum(weights)
        spot = sum(s * w for s, w in zip(strikes, weights)) / total_w

    spot_estimates.append((time_key, spot))

# ── Prepare plot data ──
trade_times = [t['dt'] for t in opens]
kalshi_probs = [t['price'] / 100.0 for t in opens]
deribit_probs = [t['model_prob'] for t in opens]
edges = [t['edge'] for t in opens]
sides = [t['side'] for t in opens]

# Signed edge: positive = model says Kalshi is cheap
signed_edges = []
for t in opens:
    if t['side'] == 'YES':
        # Bought YES: model_prob > kalshi_yes_price
        signed_edges.append(t['model_prob'] - t['price'] / 100.0)
    else:
        # Bought NO: (1 - model_prob) > kalshi_no_price
        signed_edges.append((1 - t['model_prob']) - t['price'] / 100.0)

spot_times = [s[0] for s in spot_estimates]
spot_prices = [s[1] for s in spot_estimates]

# ── Create the plot ──
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1.5, 1]})

# Panel 1: Estimated BTC Price
ax1 = axes[0]
ax1.plot(spot_times, spot_prices, color='#f7931a', linewidth=1.5, alpha=0.9, label='Est. BTC Spot')
ax1.fill_between(spot_times, spot_prices, alpha=0.1, color='#f7931a')

# Mark the big drop on Feb 7 (strikes drop from ~88K to ~67K)
ax1.set_ylabel('Estimated BTC Price ($)', fontsize=12)
ax1.set_title('BTC Price vs Kalshi-Deribit Probability Spread\n(from paper trading V1 + V2 trade logs)',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Compute price changes between spot estimates
if len(spot_prices) > 1:
    pct_changes = []
    for i in range(1, len(spot_prices)):
        pct = abs(spot_prices[i] - spot_prices[i-1]) / spot_prices[i-1] * 100
        pct_changes.append((spot_times[i], pct))

    # Highlight big moves (>2% change between windows)
    big_moves = [(t, p) for t, p in pct_changes if p > 3]
    for t, p in big_moves:
        ax1.axvline(x=t, color='red', alpha=0.2, linewidth=8)

# Panel 2: Probability spread (edge) over time
ax2 = axes[1]

# Color by side
yes_mask = [s == 'YES' for s in sides]
no_mask = [s == 'NO' for s in sides]

yes_times = [t for t, m in zip(trade_times, yes_mask) if m]
yes_edges = [e for e, m in zip(edges, yes_mask) if m]
no_times = [t for t, m in zip(trade_times, no_mask) if m]
no_edges = [e for e, m in zip(edges, no_mask) if m]

ax2.scatter(yes_times, yes_edges, c='#2ecc71', s=20, alpha=0.7, label='YES trades (edge)', zorder=3)
ax2.scatter(no_times, no_edges, c='#e74c3c', s=20, alpha=0.7, label='NO trades (edge)', zorder=3)

# Rolling mean of edge
if len(trade_times) > 5:
    # Sort and compute rolling average
    sorted_idx = np.argsort([t.timestamp() for t in trade_times])
    sorted_times = [trade_times[i] for i in sorted_idx]
    sorted_edges = [edges[i] for i in sorted_idx]

    window = min(10, len(sorted_edges))
    rolling_edge = np.convolve(sorted_edges, np.ones(window)/window, mode='valid')
    rolling_times = sorted_times[window-1:]
    ax2.plot(rolling_times, rolling_edge, color='white', linewidth=2, alpha=0.9,
             label=f'Rolling avg (n={window})', zorder=4)

ax2.set_ylabel('Edge (Spread)', fontsize=12)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.05, color='yellow', linestyle='--', alpha=0.4, label='5% threshold')

# Highlight big move periods on this panel too
for t, p in big_moves:
    ax2.axvline(x=t, color='red', alpha=0.2, linewidth=8)

# Panel 3: Trade density (proxy for opportunity frequency)
ax3 = axes[2]

# Bin trades by hour
from collections import Counter
hour_bins = Counter()
for t in trade_times:
    # Round to nearest 6 hours
    hour_key = t.replace(minute=0, second=0, microsecond=0)
    hour_key = hour_key.replace(hour=(t.hour // 6) * 6)
    hour_bins[hour_key] += 1

bin_times = sorted(hour_bins.keys())
bin_counts = [hour_bins[t] for t in bin_times]

ax3.bar(bin_times, bin_counts, width=0.2, color='#3498db', alpha=0.7, label='Trades per 6h window')
ax3.set_ylabel('Trade Count', fontsize=12)
ax3.set_xlabel('Date (UTC)', fontsize=12)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

# Highlight big move periods
for t, p in big_moves:
    ax3.axvline(x=t, color='red', alpha=0.2, linewidth=8)

# Format x-axis
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.xticks(rotation=45)

# Style
for ax in axes:
    ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#0d0d1a')
for ax in axes:
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white') if ax.get_title() else None
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

plt.tight_layout()
plt.savefig('data/spread_analysis.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"\nPlot saved to data/spread_analysis.png")

# Print summary stats
print(f"\n{'='*60}")
print(f"SPREAD ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Total trades analyzed: {len(opens)}")
print(f"Average edge (all): {np.mean(edges)*100:.1f}%")
print(f"Median edge (all): {np.median(edges)*100:.1f}%")
print(f"Max edge: {max(edges)*100:.1f}%")

# Split by period
jan_trades = [t for t in opens if t['dt'].month == 1]
feb_trades = [t for t in opens if t['dt'].month == 2]
jan_edges = [t['edge'] for t in jan_trades]
feb_edges = [t['edge'] for t in feb_trades]

if jan_edges:
    print(f"\nJan (BTC ~$85-90K range):")
    print(f"  Trades: {len(jan_trades)}, Avg edge: {np.mean(jan_edges)*100:.1f}%, Median: {np.median(jan_edges)*100:.1f}%")
if feb_edges:
    print(f"\nFeb (BTC drop to ~$67K):")
    print(f"  Trades: {len(feb_trades)}, Avg edge: {np.mean(feb_edges)*100:.1f}%, Median: {np.median(feb_edges)*100:.1f}%")

# Edge by size bucket
small_edge = [e for e in edges if e < 0.05]
med_edge = [e for e in edges if 0.05 <= e < 0.10]
large_edge = [e for e in edges if e >= 0.10]
print(f"\nEdge distribution:")
print(f"  <5%: {len(small_edge)} trades ({len(small_edge)/len(edges)*100:.0f}%)")
print(f"  5-10%: {len(med_edge)} trades ({len(med_edge)/len(edges)*100:.0f}%)")
print(f"  >10%: {len(large_edge)} trades ({len(large_edge)/len(edges)*100:.0f}%)")

plt.show()
