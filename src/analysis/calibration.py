"""
Model calibration analysis module.

Compares model-predicted probabilities against realized settlement outcomes
and market-implied probabilities. Produces calibration curves, Brier scores,
and mispricing distribution data.
"""

import numpy as np
import pandas as pd
from typing import Optional


def build_calibration_dataset(enriched_df: pd.DataFrame,
                               positions_df: pd.DataFrame) -> pd.DataFrame:
    """Build a dataset pairing model predictions with settlement outcomes.

    Uses enriched snapshots for model/market probabilities and
    positions for known settlement results.

    If positions data has settlement outcomes, we use those.
    Otherwise we attempt to infer from the last enriched snapshot
    before each contract's close_time.
    """
    if enriched_df.empty:
        return pd.DataFrame()

    # Get settlement outcomes from positions
    settled = positions_df[positions_df["exit_time"].notna()].copy() if not positions_df.empty else pd.DataFrame()

    settlement_lookup = {}
    if not settled.empty and "settlement_value" in settled.columns:
        for _, row in settled.iterrows():
            ticker = row.get("ticker")
            # settlement_value of 100 means the side won
            # For NO side: settlement_value=100 means NO won (YES=0), so settled_yes=0
            # For YES side: settlement_value=100 means YES won, so settled_yes=1
            side = row.get("side", "")
            sv = row.get("settlement_value", 0)
            if side == "YES":
                settlement_lookup[ticker] = 1 if sv == 100 else 0
            else:  # NO
                settlement_lookup[ticker] = 0 if sv == 100 else 1

    df = enriched_df.copy()

    # Filter to rows with model_prob
    df = df[df["model_prob"].notna()].copy()

    # If we have settlement data, add it
    if settlement_lookup:
        df["settled_yes"] = df["market_ticker"].map(settlement_lookup)
    elif "spot_price" in df.columns and "strike" in df.columns:
        # Infer from last snapshot: for each ticker, find the last snapshot
        # and check if spot >= strike
        df = _infer_settlement_from_snapshots(df)

    return df


def _infer_settlement_from_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Infer settlement outcomes from last-known spot vs strike."""
    if "close_time" not in df.columns:
        return df

    # For each ticker, find rows where snapshot_time < close_time
    df_pre = df[df["snapshot_time"] < df["close_time"]].copy()
    if df_pre.empty:
        return df

    # For each ticker, get the last snapshot before close_time
    if "hours_to_settlement" in df_pre.columns:
        idx_last = df_pre.groupby("market_ticker")["hours_to_settlement"].idxmin()
        df_settle = df_pre.loc[idx_last]
        settle_map = {}
        for _, row in df_settle.iterrows():
            if pd.notna(row.get("spot_price")) and pd.notna(row.get("strike")):
                settle_map[row["market_ticker"]] = int(row["spot_price"] >= row["strike"])
        df["settled_yes"] = df["market_ticker"].map(settle_map)

    return df


def calibration_curve(predicted: np.ndarray, actual: np.ndarray,
                       n_bins: int = 10) -> pd.DataFrame:
    """Compute calibration curve data.

    Bins predicted probabilities and computes the actual frequency in each bin.

    Returns DataFrame with: bin_center, avg_predicted, avg_actual, count, bias.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (predicted >= lo) & (predicted < hi) if i < n_bins - 1 else (predicted >= lo) & (predicted <= hi)
        n = mask.sum()
        if n > 0:
            avg_pred = predicted[mask].mean()
            avg_act = actual[mask].mean()
            rows.append({
                "bin_center": (lo + hi) / 2,
                "bin_label": f"{lo*100:.0f}-{hi*100:.0f}%",
                "avg_predicted": avg_pred,
                "avg_actual": avg_act,
                "count": int(n),
                "bias": avg_pred - avg_act,
            })

    return pd.DataFrame(rows)


def brier_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute Brier score (lower is better)."""
    return float(np.mean((predicted - actual) ** 2))


def log_loss(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute average log loss."""
    p = np.clip(predicted, 1e-10, 1 - 1e-10)
    return float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))


def calibration_summary(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """Compute summary calibration statistics."""
    return {
        "n": len(predicted),
        "brier_score": brier_score(predicted, actual),
        "log_loss": log_loss(predicted, actual),
        "avg_predicted": float(np.mean(predicted)),
        "avg_actual": float(np.mean(actual)),
        "bias": float(np.mean(predicted) - np.mean(actual)),
    }


def mispricing_distribution(enriched_df: pd.DataFrame,
                             column: str = "mispricing_yes") -> pd.DataFrame:
    """Extract mispricing values for histogram plotting.

    Returns DataFrame with the mispricing column and metadata.
    """
    if enriched_df.empty or column not in enriched_df.columns:
        return pd.DataFrame()

    df = enriched_df[enriched_df[column].notna()].copy()
    return df[["snapshot_time", "market_ticker", "asset", column,
               "model_prob", "yes_ask", "sigma_distance"]].copy()


def model_vs_market_scatter(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Build data for model_prob vs market implied prob scatter plot.

    Market implied prob = yes_ask / 100.
    Returns DataFrame with: model_prob, market_prob, mispricing, asset.
    """
    if enriched_df.empty:
        return pd.DataFrame()

    df = enriched_df[
        enriched_df["model_prob"].notna() &
        enriched_df["yes_ask"].notna() &
        (enriched_df["yes_ask"] > 0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["market_prob"] = df["yes_ask"] / 100.0
    cols = ["snapshot_time", "market_ticker", "asset", "model_prob", "market_prob"]
    if "mispricing_yes" in df.columns:
        cols.append("mispricing_yes")
    if "sigma_distance" in df.columns:
        cols.append("sigma_distance")
    if "hours_to_settlement" in df.columns:
        cols.append("hours_to_settlement")

    return df[cols].copy()


def edge_decay_by_time(enriched_df: pd.DataFrame,
                        time_bins: Optional[list] = None) -> pd.DataFrame:
    """Compute average mispricing by hours-to-settlement bucket.

    Shows whether edge shrinks as contracts approach expiry.
    """
    if enriched_df.empty or "hours_to_settlement" not in enriched_df.columns:
        return pd.DataFrame()

    df = enriched_df[
        enriched_df["mispricing_yes"].notna() &
        enriched_df["hours_to_settlement"].notna() &
        (enriched_df["hours_to_settlement"] > 0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    if time_bins is None:
        time_bins = [0, 1, 2, 4, 8, 16, 24, 48, float("inf")]

    df["time_bucket"] = pd.cut(df["hours_to_settlement"], bins=time_bins, right=False)

    result = df.groupby("time_bucket", observed=True).agg(
        avg_mispricing_yes=("mispricing_yes", "mean"),
        avg_abs_mispricing=("mispricing_yes", lambda x: x.abs().mean()),
        count=("mispricing_yes", "count"),
    ).reset_index()

    return result
