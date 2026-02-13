#!/usr/bin/env python3
"""
Model Calibration Analysis for Kalshi Crypto Binary Options Trading Strategy

Compares model-predicted probabilities vs realized settlement frequencies to determine
if the model is well-calibrated or systematically biased. Also compares model calibration
against market-implied probabilities.
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

SNAPSHOTS_DIR = '/home/user/kalshi/data/paper_trading_v3/snapshots'

# ==============================================================================
# STEP 1: Load all data and determine settlement outcomes
# ==============================================================================
print("=" * 100)
print("KALSHI CRYPTO BINARY OPTIONS - MODEL CALIBRATION ANALYSIS")
print("=" * 100)

# First, gather all unique (event_ticker, close_time) pairs and find the
# last snapshot before each close_time to determine settlement outcomes.
# Then gather model predictions from snapshots well before settlement.

print("\n[1] Loading snapshot data across all days...")

all_days = sorted(glob.glob(f'{SNAPSHOTS_DIR}/20260*'))
print(f"    Found {len(all_days)} days: {[os.path.basename(d) for d in all_days]}")

# Strategy:
# 1) For each day, load ALL kalshi files (they're small parquets)
# 2) For each contract (ticker), find the last snapshot before close_time => settlement outcome
# 3) For each contract, find snapshots 1-2 hours before close_time => model prediction / market price

def parse_snapshot_time(s):
    """Parse snapshot_time string to datetime."""
    return pd.to_datetime(s)

def parse_close_time(s):
    """Parse close_time string to datetime."""
    return pd.to_datetime(s)

# Load all data into one big DataFrame per day, then concatenate
# We'll be selective about which days to load to manage memory

all_data = []
for day_dir in all_days:
    day_name = os.path.basename(day_dir)
    files = sorted(glob.glob(f'{day_dir}/kalshi_*.parquet'))
    print(f"    Loading {day_name}: {len(files)} files...", end=" ")

    day_frames = []
    for f in files:
        df = pd.read_parquet(f)
        day_frames.append(df)

    if day_frames:
        day_df = pd.concat(day_frames, ignore_index=True)
        all_data.append(day_df)
        print(f"{len(day_df)} rows, {day_df['ticker'].nunique()} unique tickers")
    else:
        print("no files")

print("\n    Concatenating all data...")
df_all = pd.concat(all_data, ignore_index=True)
print(f"    Total: {len(df_all)} rows, {df_all['ticker'].nunique()} unique tickers")

# Parse times
print("\n[2] Parsing timestamps...")
df_all['snapshot_dt'] = pd.to_datetime(df_all['snapshot_time'], utc=True)
df_all['close_dt'] = pd.to_datetime(df_all['close_time'], utc=True)

# Compute time_to_settle in hours
df_all['time_to_settle_hrs'] = (df_all['close_dt'] - df_all['snapshot_dt']).dt.total_seconds() / 3600

# ==============================================================================
# STEP 2: Determine settlement outcomes
# ==============================================================================
print("\n[3] Determining settlement outcomes...")

# For each contract (ticker), the settlement is YES if spot >= strike at close_time.
# We approximate by taking the LAST snapshot BEFORE close_time for each contract.

# Filter to rows where snapshot is before close_time (time_to_settle > 0)
df_presettlement = df_all[df_all['time_to_settle_hrs'] > 0].copy()

# For each ticker, get the row with the minimum time_to_settle_hrs (last snapshot before settlement)
idx_last_before_settle = df_presettlement.groupby('ticker')['time_to_settle_hrs'].idxmin()
df_settle = df_presettlement.loc[idx_last_before_settle].copy()

# Determine settlement outcome
df_settle['settled_yes'] = (df_settle['spot_price'] >= df_settle['strike']).astype(int)
df_settle['minutes_before_settle'] = df_settle['time_to_settle_hrs'] * 60

print(f"    Contracts with settlement determination: {len(df_settle)}")
print(f"    Minutes before settlement (median): {df_settle['minutes_before_settle'].median():.1f}")
print(f"    Minutes before settlement (mean): {df_settle['minutes_before_settle'].mean():.1f}")
print(f"    Minutes before settlement (min): {df_settle['minutes_before_settle'].min():.1f}")
print(f"    Minutes before settlement (max): {df_settle['minutes_before_settle'].max():.1f}")
print(f"    Settled YES: {df_settle['settled_yes'].sum()} ({df_settle['settled_yes'].mean()*100:.1f}%)")
print(f"    Settled NO:  {(1-df_settle['settled_yes']).sum()} ({(1-df_settle['settled_yes']).mean()*100:.1f}%)")

# Create settlement lookup: ticker -> settled_yes
settlement_lookup = df_settle.set_index('ticker')['settled_yes'].to_dict()

# Also store the settlement spot_price for reference
settle_spot_lookup = df_settle.set_index('ticker')['spot_price'].to_dict()

# Filter to only contracts that settled within our observation window
# (contracts whose close_time is before the last snapshot we have)
last_snapshot_time = df_all['snapshot_dt'].max()
print(f"\n    Last snapshot time: {last_snapshot_time}")
print(f"    Only considering contracts that settled before: {last_snapshot_time}")

# Only keep contracts that actually settled (close_time < last snapshot time)
settled_tickers = set(df_settle[df_settle['close_dt'] < last_snapshot_time]['ticker'].values)
print(f"    Contracts that settled within observation window: {len(settled_tickers)}")

# ==============================================================================
# STEP 3: Build calibration dataset
# ==============================================================================
print("\n[4] Building calibration dataset...")

# For each settled contract, get the model prediction and market price from
# a snapshot well before settlement. We'll use multiple time horizons.

# Strategy: for each ticker, get rows from 30min-2hr before settlement with valid model_prob
df_calib_raw = df_all[
    (df_all['ticker'].isin(settled_tickers)) &
    (df_all['time_to_settle_hrs'] > 0.5) &  # at least 30 min before
    (df_all['time_to_settle_hrs'] < 3.0) &   # within 3 hours
    (df_all['model_prob'].notna())
].copy()

print(f"    Rows with valid model_prob, 30min-3hr before settlement: {len(df_calib_raw)}")

# For each ticker, take the observation closest to 1 hour before settlement
df_calib_raw['dist_from_1hr'] = abs(df_calib_raw['time_to_settle_hrs'] - 1.0)
idx_best = df_calib_raw.groupby('ticker')['dist_from_1hr'].idxmin()
df_calib = df_calib_raw.loc[idx_best].copy()

# Add settlement outcome
df_calib['settled_yes'] = df_calib['ticker'].map(settlement_lookup)
df_calib['settle_spot'] = df_calib['ticker'].map(settle_spot_lookup)

# Compute market-implied probability (midpoint of yes_bid and yes_ask, in [0,1])
df_calib['market_mid'] = (df_calib['yes_bid'] + df_calib['yes_ask']) / 2.0 / 100.0
# Also use yes_ask as conservative market probability for buying YES
df_calib['market_ask'] = df_calib['yes_ask'] / 100.0

# Handle edge cases: if yes_bid=0 and yes_ask=0, market implies P(YES)~0
# If yes_bid=100 and yes_ask=100, market implies P(YES)~1
# If spread is too wide, midpoint may be unreliable; flag these
df_calib['market_spread'] = (df_calib['yes_ask'] - df_calib['yes_bid'])
df_calib['has_valid_market'] = (
    (df_calib['yes_ask'] > 0) &
    (df_calib['yes_bid'] < 100) &
    (df_calib['market_spread'] <= 20)  # reasonable spread
)

print(f"    Contracts with model predictions for calibration: {len(df_calib)}")
print(f"    Contracts with valid market prices: {df_calib['has_valid_market'].sum()}")
print(f"    Time to settlement range: {df_calib['time_to_settle_hrs'].min():.2f} - {df_calib['time_to_settle_hrs'].max():.2f} hrs")
print(f"    Assets: {df_calib['asset'].value_counts().to_dict()}")
print(f"    Settled YES: {df_calib['settled_yes'].sum()} ({df_calib['settled_yes'].mean()*100:.1f}%)")

# ==============================================================================
# STEP 4: Model Calibration Analysis
# ==============================================================================
print("\n" + "=" * 100)
print("MODEL CALIBRATION ANALYSIS")
print("=" * 100)

def calibration_table(predicted, actual, n_bins=10, label="Model"):
    """Create a calibration table binning predicted probs vs actual outcomes."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{bin_edges[i]*100:.0f}-{bin_edges[i+1]*100:.0f}%" for i in range(n_bins)]

    results = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        if i == n_bins - 1:
            mask = (predicted >= lo) & (predicted <= hi)
        else:
            mask = (predicted >= lo) & (predicted < hi)

        n = mask.sum()
        if n > 0:
            avg_pred = predicted[mask].mean()
            avg_actual = actual[mask].mean()
            bias = avg_pred - avg_actual
            results.append({
                'Bin': bin_labels[i],
                'N': n,
                f'Avg {label} Pred': avg_pred,
                'Actual YES Rate': avg_actual,
                'Bias (Pred-Actual)': bias,
            })
        else:
            results.append({
                'Bin': bin_labels[i],
                'N': 0,
                f'Avg {label} Pred': np.nan,
                'Actual YES Rate': np.nan,
                'Bias (Pred-Actual)': np.nan,
            })

    return pd.DataFrame(results)

# 4a: Overall model calibration
print("\n--- 4a. Overall Model Calibration (10 bins) ---")
model_cal = calibration_table(
    df_calib['model_prob'].values,
    df_calib['settled_yes'].values,
    n_bins=10,
    label="Model"
)
print(model_cal.to_string(index=False))

# Summary statistics
valid_mask = df_calib['model_prob'].notna() & df_calib['settled_yes'].notna()
pred = df_calib.loc[valid_mask, 'model_prob'].values.astype(np.float64)
actual = df_calib.loc[valid_mask, 'settled_yes'].values.astype(np.float64)

brier_score = np.mean((pred - actual) ** 2)
log_loss_vals = -(actual * np.log(np.clip(pred, 1e-10, 1-1e-10)) +
                  (1-actual) * np.log(np.clip(1-pred, 1e-10, 1-1e-10)))
avg_log_loss = np.mean(log_loss_vals)
avg_pred = np.mean(pred)
avg_actual = np.mean(actual)
overall_bias = avg_pred - avg_actual

print(f"\n  Overall Statistics:")
print(f"    N contracts:        {len(pred)}")
print(f"    Brier Score:        {brier_score:.6f}  (lower is better; baseline naive = 0.25)")
print(f"    Log Loss:           {avg_log_loss:.6f}")
print(f"    Avg Model Pred:     {avg_pred:.4f}")
print(f"    Avg Actual YES:     {avg_actual:.4f}")
print(f"    Overall Bias:       {overall_bias:+.4f}  ({'over-predicting YES' if overall_bias > 0 else 'under-predicting YES'})")

# 4b: Finer-grained calibration (20 bins)
print("\n--- 4b. Model Calibration (20 bins, finer granularity) ---")
model_cal_20 = calibration_table(pred, actual, n_bins=20, label="Model")
print(model_cal_20.to_string(index=False))

# ==============================================================================
# STEP 5: Market Calibration Analysis
# ==============================================================================
print("\n" + "=" * 100)
print("MARKET CALIBRATION ANALYSIS")
print("=" * 100)

# Use midpoint for market-implied probability
df_market = df_calib[df_calib['has_valid_market']].copy()
print(f"\n  Contracts with valid market prices: {len(df_market)}")

if len(df_market) > 0:
    print("\n--- 5a. Market Calibration (midpoint, 10 bins) ---")
    market_cal = calibration_table(
        df_market['market_mid'].values,
        df_market['settled_yes'].values,
        n_bins=10,
        label="Market"
    )
    print(market_cal.to_string(index=False))

    mkt_pred = df_market['market_mid'].values.astype(np.float64)
    mkt_actual = df_market['settled_yes'].values.astype(np.float64)
    mkt_brier = np.mean((mkt_pred - mkt_actual) ** 2)
    mkt_logloss_vals = -(mkt_actual * np.log(np.clip(mkt_pred, 1e-10, 1-1e-10)) +
                         (1-mkt_actual) * np.log(np.clip(1-mkt_pred, 1e-10, 1-1e-10)))
    mkt_avg_logloss = np.mean(mkt_logloss_vals)
    mkt_avg_pred = np.mean(mkt_pred)
    mkt_avg_actual = np.mean(mkt_actual)
    mkt_bias = mkt_avg_pred - mkt_avg_actual

    print(f"\n  Market Overall Statistics:")
    print(f"    N contracts:        {len(mkt_pred)}")
    print(f"    Brier Score:        {mkt_brier:.6f}")
    print(f"    Log Loss:           {mkt_avg_logloss:.6f}")
    print(f"    Avg Market Mid:     {mkt_avg_pred:.4f}")
    print(f"    Avg Actual YES:     {mkt_avg_actual:.4f}")
    print(f"    Overall Bias:       {mkt_bias:+.4f}")

    # Compare model vs market on the SAME set of contracts
    print("\n--- 5b. Model vs Market Comparison (same contract set) ---")
    model_pred_same = df_market['model_prob'].values.astype(np.float64)
    model_brier_same = np.mean((model_pred_same - mkt_actual) ** 2)
    model_logloss_same_vals = -(mkt_actual * np.log(np.clip(model_pred_same, 1e-10, 1-1e-10)) +
                                (1-mkt_actual) * np.log(np.clip(1-model_pred_same, 1e-10, 1-1e-10)))
    model_avg_logloss_same = np.mean(model_logloss_same_vals)

    comparison = pd.DataFrame({
        'Metric': ['Brier Score', 'Log Loss', 'Avg Prediction', 'Avg Actual', 'Bias'],
        'Model': [model_brier_same, model_avg_logloss_same, np.mean(model_pred_same), mkt_avg_actual, np.mean(model_pred_same) - mkt_avg_actual],
        'Market': [mkt_brier, mkt_avg_logloss, mkt_avg_pred, mkt_avg_actual, mkt_bias],
        'Model Better?': [
            'YES' if model_brier_same < mkt_brier else 'NO',
            'YES' if model_avg_logloss_same < mkt_avg_logloss else 'NO',
            '-', '-',
            'YES' if abs(np.mean(model_pred_same) - mkt_avg_actual) < abs(mkt_bias) else 'NO'
        ]
    })
    print(comparison.to_string(index=False))

# ==============================================================================
# STEP 6: Signal-specific analysis (NO signals = our trades)
# ==============================================================================
print("\n" + "=" * 100)
print("SIGNAL-SPECIFIC ANALYSIS (Contracts Where We Had Signals)")
print("=" * 100)

# Look at ALL snapshots where signal_side is not NaN for settled contracts
df_signals_all = df_all[
    (df_all['ticker'].isin(settled_tickers)) &
    (df_all['signal_side'].notna()) &
    (df_all['time_to_settle_hrs'] > 0) &
    (df_all['model_prob'].notna())
].copy()

df_signals_all['settled_yes'] = df_signals_all['ticker'].map(settlement_lookup)

print(f"\n  Total signal observations (all snapshots): {len(df_signals_all)}")
print(f"  Signal side distribution:")
print(f"    {df_signals_all['signal_side'].value_counts().to_string()}")

# For unique contract analysis, take the FIRST signal observation for each ticker
if len(df_signals_all) > 0:
    idx_first_signal = df_signals_all.groupby('ticker')['snapshot_dt'].idxmin()
    df_signals = df_signals_all.loc[idx_first_signal].copy()

    print(f"\n  Unique contracts with signals: {len(df_signals)}")
    print(f"  Signal side distribution (unique contracts):")
    print(f"    {df_signals['signal_side'].value_counts().to_string()}")

    # For NO signals
    df_no_signals = df_signals[df_signals['signal_side'] == 'NO'].copy()
    df_no_signals['model_prob_no'] = 1 - df_no_signals['model_prob']
    df_no_signals['settled_no'] = 1 - df_no_signals['settled_yes']

    print(f"\n--- 6a. NO Signal Contracts ---")
    print(f"    Count: {len(df_no_signals)}")
    if len(df_no_signals) > 0:
        print(f"    Avg model P(NO):    {df_no_signals['model_prob_no'].mean():.4f}")
        print(f"    Actual NO rate:     {df_no_signals['settled_no'].mean():.4f}")
        print(f"    Win rate:           {df_no_signals['settled_no'].mean()*100:.1f}%")
        print(f"    Bias (pred-actual): {df_no_signals['model_prob_no'].mean() - df_no_signals['settled_no'].mean():+.4f}")
        print(f"    Avg mispricing:     {df_no_signals['mispricing_cents'].mean():.2f} cents")
        print(f"    Time to settle:     {df_no_signals['time_to_settle_hrs'].mean():.1f} hrs")

        print(f"\n    NO Signal Calibration (5 bins):")
        no_cal = calibration_table(
            df_no_signals['model_prob_no'].values,
            df_no_signals['settled_no'].values,
            n_bins=5,
            label="Model P(NO)"
        )
        print(f"    {no_cal.to_string(index=False)}")

    # For YES signals
    df_yes_signals = df_signals[df_signals['signal_side'] == 'YES'].copy()
    if len(df_yes_signals) > 0:
        print(f"\n--- 6b. YES Signal Contracts ---")
        print(f"    Count: {len(df_yes_signals)}")
        print(f"    Avg model P(YES):   {df_yes_signals['model_prob'].mean():.4f}")
        print(f"    Actual YES rate:    {df_yes_signals['settled_yes'].mean():.4f}")
        print(f"    Win rate:           {df_yes_signals['settled_yes'].mean()*100:.1f}%")
        print(f"    Bias (pred-actual): {df_yes_signals['model_prob'].mean() - df_yes_signals['settled_yes'].mean():+.4f}")
        print(f"    Avg mispricing:     {df_yes_signals['mispricing_cents'].mean():.2f} cents")

# ==============================================================================
# STEP 7: Calibration by asset (BTC vs ETH)
# ==============================================================================
print("\n" + "=" * 100)
print("CALIBRATION BY ASSET")
print("=" * 100)

for asset in ['BTC', 'ETH']:
    df_asset = df_calib[df_calib['asset'] == asset]
    if len(df_asset) == 0:
        continue

    print(f"\n--- {asset} ({len(df_asset)} contracts) ---")

    a_pred = df_asset['model_prob'].values
    a_actual = df_asset['settled_yes'].values
    a_brier = np.mean((a_pred - a_actual) ** 2)
    a_bias = np.mean(a_pred) - np.mean(a_actual)

    cal = calibration_table(a_pred, a_actual, n_bins=10, label="Model")
    print(cal.to_string(index=False))
    print(f"  Brier: {a_brier:.6f}, Bias: {a_bias:+.4f}, Avg pred: {np.mean(a_pred):.4f}, Avg actual: {np.mean(a_actual):.4f}")

# ==============================================================================
# STEP 8: Calibration by time to settlement
# ==============================================================================
print("\n" + "=" * 100)
print("CALIBRATION BY TIME TO SETTLEMENT")
print("=" * 100)

# Check at different time horizons
for min_hrs, max_hrs, label in [(0.5, 1.0, "30-60 min"), (1.0, 2.0, "1-2 hrs"), (2.0, 4.0, "2-4 hrs")]:
    df_time = df_all[
        (df_all['ticker'].isin(settled_tickers)) &
        (df_all['time_to_settle_hrs'] >= min_hrs) &
        (df_all['time_to_settle_hrs'] < max_hrs) &
        (df_all['model_prob'].notna())
    ].copy()

    if len(df_time) == 0:
        print(f"\n--- {label}: No data ---")
        continue

    # Take one obs per ticker (closest to midpoint of range)
    mid_hr = (min_hrs + max_hrs) / 2
    df_time['dist_mid'] = abs(df_time['time_to_settle_hrs'] - mid_hr)
    idx = df_time.groupby('ticker')['dist_mid'].idxmin()
    df_time_uniq = df_time.loc[idx].copy()
    df_time_uniq['settled_yes'] = df_time_uniq['ticker'].map(settlement_lookup)
    df_time_uniq = df_time_uniq.dropna(subset=['settled_yes'])

    if len(df_time_uniq) < 10:
        print(f"\n--- {label}: Only {len(df_time_uniq)} contracts, skipping ---")
        continue

    t_pred = df_time_uniq['model_prob'].values
    t_actual = df_time_uniq['settled_yes'].values
    t_brier = np.mean((t_pred - t_actual) ** 2)
    t_bias = np.mean(t_pred) - np.mean(t_actual)

    print(f"\n--- {label} before settlement ({len(df_time_uniq)} contracts) ---")
    cal = calibration_table(t_pred, t_actual, n_bins=10, label="Model")
    print(cal.to_string(index=False))
    print(f"  Brier: {t_brier:.6f}, Bias: {t_bias:+.4f}")

# ==============================================================================
# STEP 9: Calibration by sigma_distance
# ==============================================================================
print("\n" + "=" * 100)
print("CALIBRATION BY SIGMA DISTANCE (Moneyness)")
print("=" * 100)

df_sigma = df_calib[df_calib['sigma_distance'].notna()].copy()
if len(df_sigma) > 0:
    for lo_s, hi_s, label in [(-999, -2, "Deep ITM (sigma < -2)"),
                                (-2, -1, "ITM (-2 < sigma < -1)"),
                                (-1, -0.5, "Slight ITM (-1 < sigma < -0.5)"),
                                (-0.5, 0.5, "ATM (-0.5 < sigma < 0.5)"),
                                (0.5, 1, "Slight OTM (0.5 < sigma < 1)"),
                                (1, 2, "OTM (1 < sigma < 2)"),
                                (2, 999, "Deep OTM (sigma > 2)")]:
        mask = (df_sigma['sigma_distance'] >= lo_s) & (df_sigma['sigma_distance'] < hi_s)
        subset = df_sigma[mask]
        if len(subset) < 5:
            continue
        s_pred = subset['model_prob'].values
        s_actual = subset['settled_yes'].values
        s_bias = np.mean(s_pred) - np.mean(s_actual)
        s_brier = np.mean((s_pred - s_actual) ** 2)
        print(f"  {label:40s}  N={len(subset):5d}  AvgPred={np.mean(s_pred):.4f}  AvgActual={np.mean(s_actual):.4f}  Bias={s_bias:+.4f}  Brier={s_brier:.4f}")

# ==============================================================================
# STEP 10: Detailed bias analysis
# ==============================================================================
print("\n" + "=" * 100)
print("DETAILED BIAS ANALYSIS")
print("=" * 100)

# Check if model is systematically biased in specific probability ranges
print("\n--- Systematic Bias by Prediction Range ---")
for lo, hi, label in [(0, 0.1, "Very Low (0-10%)"),
                       (0.1, 0.3, "Low (10-30%)"),
                       (0.3, 0.5, "Below Mid (30-50%)"),
                       (0.5, 0.7, "Above Mid (50-70%)"),
                       (0.7, 0.9, "High (70-90%)"),
                       (0.9, 1.01, "Very High (90-100%)")]:
    if hi > 1:
        mask = (df_calib['model_prob'] >= lo) & (df_calib['model_prob'] <= hi)
    else:
        mask = (df_calib['model_prob'] >= lo) & (df_calib['model_prob'] < hi)
    subset = df_calib[mask]
    if len(subset) < 3:
        continue
    avg_p = subset['model_prob'].mean()
    avg_a = subset['settled_yes'].mean()
    bias = avg_p - avg_a
    direction = "OVER-predicts YES" if bias > 0.02 else ("UNDER-predicts YES" if bias < -0.02 else "Well calibrated")
    print(f"  {label:25s}  N={len(subset):5d}  Pred={avg_p:.4f}  Actual={avg_a:.4f}  Bias={bias:+.4f}  => {direction}")

# Check calibration of the NO probability for contracts where model says likely NO
print("\n--- P(NO) Calibration for Model-Predicted NO-likely Contracts ---")
df_calib['model_prob_no'] = 1 - df_calib['model_prob']
df_calib['settled_no'] = 1 - df_calib['settled_yes']

for lo, hi, label in [(0.5, 0.7, "P(NO) 50-70%"),
                       (0.7, 0.9, "P(NO) 70-90%"),
                       (0.9, 1.01, "P(NO) 90-100%")]:
    if hi > 1:
        mask = (df_calib['model_prob_no'] >= lo) & (df_calib['model_prob_no'] <= 1.0)
    else:
        mask = (df_calib['model_prob_no'] >= lo) & (df_calib['model_prob_no'] < hi)
    subset = df_calib[mask]
    if len(subset) < 3:
        continue
    avg_p = subset['model_prob_no'].mean()
    avg_a = subset['settled_no'].mean()
    bias = avg_p - avg_a
    print(f"  {label:20s}  N={len(subset):5d}  Pred P(NO)={avg_p:.4f}  Actual NO Rate={avg_a:.4f}  Bias={bias:+.4f}")

# ==============================================================================
# STEP 11: Reliability Diagram Data (for potential plotting)
# ==============================================================================
print("\n" + "=" * 100)
print("RELIABILITY DIAGRAM DATA (Model vs Market)")
print("=" * 100)

# Side-by-side calibration for both model and market on same contracts
df_both = df_calib[df_calib['has_valid_market']].copy()
if len(df_both) > 0:
    bin_edges = np.linspace(0, 1, 11)
    print(f"\n{'Bin':>12s} {'N_model':>8s} {'Model_Pred':>11s} {'N_mkt':>8s} {'Mkt_Pred':>11s} {'Actual':>8s} {'Model_Bias':>11s} {'Mkt_Bias':>11s}")
    print("-" * 90)

    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i+1]
        lbl = f"{lo*100:.0f}-{hi*100:.0f}%"

        # Model bins
        if i == 9:
            m_mask = (df_both['model_prob'] >= lo) & (df_both['model_prob'] <= hi)
        else:
            m_mask = (df_both['model_prob'] >= lo) & (df_both['model_prob'] < hi)
        m_n = m_mask.sum()
        m_pred = df_both.loc[m_mask, 'model_prob'].mean() if m_n > 0 else np.nan
        m_actual = df_both.loc[m_mask, 'settled_yes'].mean() if m_n > 0 else np.nan
        m_bias = m_pred - m_actual if m_n > 0 else np.nan

        # Market bins
        if i == 9:
            k_mask = (df_both['market_mid'] >= lo) & (df_both['market_mid'] <= hi)
        else:
            k_mask = (df_both['market_mid'] >= lo) & (df_both['market_mid'] < hi)
        k_n = k_mask.sum()
        k_pred = df_both.loc[k_mask, 'market_mid'].mean() if k_n > 0 else np.nan
        k_actual = df_both.loc[k_mask, 'settled_yes'].mean() if k_n > 0 else np.nan
        k_bias = k_pred - k_actual if k_n > 0 else np.nan

        print(f"{lbl:>12s} {m_n:>8d} {m_pred:>11.4f} {k_n:>8d} {k_pred:>11.4f} {m_actual:>8.4f} {m_bias:>+11.4f} {k_bias:>+11.4f}"
              if (m_n > 0 and k_n > 0) else f"{lbl:>12s} {m_n:>8d} {'N/A':>11s} {k_n:>8d} {'N/A':>11s} {'N/A':>8s} {'N/A':>11s} {'N/A':>11s}")

# ==============================================================================
# STEP 12: Summary and Conclusions
# ==============================================================================
print("\n" + "=" * 100)
print("SUMMARY AND CONCLUSIONS")
print("=" * 100)

print(f"""
DATA SCOPE:
  - Days analyzed: {[os.path.basename(d) for d in all_days]}
  - Total snapshots loaded: {len(df_all):,}
  - Unique contracts (tickers): {df_all['ticker'].nunique():,}
  - Settled contracts used for calibration: {len(df_calib):,}
  - Contracts with valid market prices: {df_calib['has_valid_market'].sum():,}

MODEL CALIBRATION:
  - Brier Score: {brier_score:.6f}
  - Log Loss: {avg_log_loss:.6f}
  - Average Model Prediction: {avg_pred:.4f}
  - Average Actual Settlement: {avg_actual:.4f}
  - Overall Bias: {overall_bias:+.4f} ({'model OVER-predicts YES' if overall_bias > 0 else 'model UNDER-predicts YES'})
""")

if len(df_market) > 0:
    print(f"""MARKET CALIBRATION (same contracts with valid prices):
  - Brier Score: {mkt_brier:.6f}
  - Log Loss: {mkt_avg_logloss:.6f}
  - Overall Bias: {mkt_bias:+.4f}
  - Model vs Market: {'Model is better calibrated' if model_brier_same < mkt_brier else 'Market is better calibrated'} (by Brier Score)
""")

if len(df_no_signals) > 0:
    no_win_rate = df_no_signals['settled_no'].mean()
    no_bias = df_no_signals['model_prob_no'].mean() - no_win_rate
    print(f"""TRADING SIGNAL ANALYSIS (NO signals):
  - Number of NO signal contracts: {len(df_no_signals)}
  - Model avg P(NO): {df_no_signals['model_prob_no'].mean():.4f}
  - Actual NO settlement rate: {no_win_rate:.4f} ({no_win_rate*100:.1f}%)
  - Signal bias: {no_bias:+.4f}
  - Average mispricing claimed: {df_no_signals['mispricing_cents'].mean():.2f} cents
""")

# Detect specific biases
print("BIAS PATTERNS DETECTED:")
bias_patterns = []

# Check overall
if overall_bias > 0.03:
    bias_patterns.append(f"  - Model systematically OVER-predicts P(YES) by {overall_bias:.4f}")
elif overall_bias < -0.03:
    bias_patterns.append(f"  - Model systematically UNDER-predicts P(YES) by {abs(overall_bias):.4f}")
else:
    bias_patterns.append(f"  - Model overall bias is small ({overall_bias:+.4f}), appears reasonably calibrated in aggregate")

# Check tails
low_pred = df_calib[df_calib['model_prob'] < 0.2]
high_pred = df_calib[df_calib['model_prob'] > 0.8]

if len(low_pred) > 10:
    low_bias = low_pred['model_prob'].mean() - low_pred['settled_yes'].mean()
    if abs(low_bias) > 0.03:
        bias_patterns.append(f"  - LOW probability tail (P<20%): bias={low_bias:+.4f} (pred={low_pred['model_prob'].mean():.4f}, actual={low_pred['settled_yes'].mean():.4f})")

if len(high_pred) > 10:
    high_bias = high_pred['model_prob'].mean() - high_pred['settled_yes'].mean()
    if abs(high_bias) > 0.03:
        bias_patterns.append(f"  - HIGH probability tail (P>80%): bias={high_bias:+.4f} (pred={high_pred['model_prob'].mean():.4f}, actual={high_pred['settled_yes'].mean():.4f})")

# Check mid-range
mid_pred = df_calib[(df_calib['model_prob'] >= 0.3) & (df_calib['model_prob'] <= 0.7)]
if len(mid_pred) > 10:
    mid_bias = mid_pred['model_prob'].mean() - mid_pred['settled_yes'].mean()
    if abs(mid_bias) > 0.03:
        bias_patterns.append(f"  - MID-RANGE (30-70%): bias={mid_bias:+.4f} (pred={mid_pred['model_prob'].mean():.4f}, actual={mid_pred['settled_yes'].mean():.4f})")

for p in bias_patterns:
    print(p)

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
