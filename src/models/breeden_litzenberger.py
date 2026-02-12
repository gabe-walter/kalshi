"""
Breeden-Litzenberger Implied Distribution

Extract the risk-neutral probability distribution directly from option prices.

The key insight:
- The risk-neutral PDF is the second derivative of call prices w.r.t. strike
- f(K) = e^(rT) × ∂²C/∂K²

This captures the full implied distribution including skew, kurtosis, and fat tails
without assuming any parametric form (no lognormal assumption).

For binary options (what Kalshi trades):
- P(S_T > K) = -e^(rT) × ∂C/∂K

This is because a digital call = -∂C/∂K (in the risk-neutral measure).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.integrate import cumulative_trapezoid
from dataclasses import dataclass
from typing import Optional, Tuple, List
from datetime import datetime, timezone


@dataclass
class ImpliedCDF:
    """
    Risk-neutral CDF extracted from option prices via Breeden-Litzenberger.
    """
    strikes: np.ndarray          # Strike prices
    call_prices: np.ndarray      # Call prices in USD
    cdf_values: np.ndarray       # P(S_T <= K) for each strike
    pdf_values: np.ndarray       # Probability density f(K)
    spot: float                  # Current spot price
    expiry: str                  # Expiry used
    time_to_expiry: float        # Years to expiry

    # Spline for interpolation
    _cdf_spline: Optional[CubicSpline] = None

    def prob_above(self, strike: float) -> float:
        """P(S_T > K) = 1 - CDF(K) at the option expiry time T."""
        if strike <= self.strikes.min():
            return 1.0
        if strike >= self.strikes.max():
            return 0.0

        if self._cdf_spline is not None:
            cdf_val = float(self._cdf_spline(strike))
            return max(0.0, min(1.0, 1.0 - cdf_val))

        # Linear interpolation fallback
        cdf_val = np.interp(strike, self.strikes, self.cdf_values)
        return max(0.0, min(1.0, 1.0 - cdf_val))

    def prob_above_at_time(self, strike: float, target_time_years: float) -> float:
        """
        P(S_t > K) at a different time t using variance scaling.

        The B-L CDF is extracted for time T (option expiry). To get probabilities
        at time t (Kalshi settlement), we use variance scaling:

        Under risk-neutral dynamics, variance scales linearly with time.
        A strike K at time t corresponds to an "equivalent" strike K' at time T:
            ln(K'/S) = ln(K/S) × √(T/t)
            K' = S × (K/S)^(√(T/t))

        This preserves the shape of the distribution while adjusting its width.

        Args:
            strike: The strike price to evaluate
            target_time_years: Time to Kalshi settlement in years

        Returns:
            P(S_t > K) adjusted for time t
        """
        T = self.time_to_expiry  # Option expiry time
        t = target_time_years     # Target (Kalshi settlement) time

        # Handle edge cases
        if t <= 0:
            return 1.0 if self.spot >= strike else 0.0
        if T <= 0:
            return self.prob_above(strike)

        # If times are very close, no adjustment needed
        if abs(T - t) < 1e-6:
            return self.prob_above(strike)

        # Compute time scaling factor
        # When t < T, scale > 1, pushing K' further from spot
        # When t > T, scale < 1, pulling K' closer to spot
        scale = np.sqrt(T / t)

        # Transform strike: K' = S × (K/S)^scale
        # In log space: ln(K'/S) = scale × ln(K/S)
        if strike <= 0:
            return 1.0

        log_moneyness = np.log(strike / self.spot)
        adjusted_log_moneyness = log_moneyness * scale
        adjusted_strike = self.spot * np.exp(adjusted_log_moneyness)

        # Look up probability at the adjusted strike
        return self.prob_above(adjusted_strike)

    def prob_below(self, strike: float) -> float:
        """P(S_T <= K) = CDF(K)"""
        return 1.0 - self.prob_above(strike)

    def prob_between(self, low: float, high: float) -> float:
        """P(low < S_T <= high)"""
        return self.prob_below(high) - self.prob_below(low)

    def summary(self) -> str:
        """Human-readable summary."""
        # Find approximate percentiles
        p10 = self.strikes[np.searchsorted(self.cdf_values, 0.10)]
        p50 = self.strikes[np.searchsorted(self.cdf_values, 0.50)]
        p90 = self.strikes[np.searchsorted(self.cdf_values, 0.90)]

        return (
            f"ImpliedCDF (Breeden-Litzenberger):\n"
            f"  Spot: ${self.spot:,.2f}\n"
            f"  Expiry: {self.expiry}\n"
            f"  Strike range: ${self.strikes.min():,.0f} - ${self.strikes.max():,.0f}\n"
            f"  10th percentile: ${p10:,.0f}\n"
            f"  Median (50th): ${p50:,.0f}\n"
            f"  90th percentile: ${p90:,.0f}\n"
            f"  P(S > spot): {self.prob_above(self.spot):.1%}"
        )


def extract_implied_cdf(
    option_chain: pd.DataFrame,
    spot: float,
    expiry_str: str,
    risk_free_rate: float = 0.0,  # Assume 0 for crypto
    smoothing: float = 0.0        # Spline smoothing (0 = interpolating)
) -> Optional[ImpliedCDF]:
    """
    Extract the risk-neutral CDF from option prices using Breeden-Litzenberger.

    Method:
    1. Get call prices C(K) at all strikes for the given expiry
    2. Convert to USD (Deribit quotes in BTC terms)
    3. Fit smooth spline through C(K)
    4. Compute -dC/dK to get survival function (P(S > K))
    5. CDF = 1 - survival function

    Args:
        option_chain: Deribit option chain DataFrame
        spot: Current spot price
        expiry_str: Which expiry to use (e.g., '19JAN26')
        risk_free_rate: Risk-free rate (annualized)
        smoothing: Spline smoothing factor (0 = exact interpolation)

    Returns:
        ImpliedCDF object with prob_above() method
    """
    # Filter for this expiry and calls only
    exp_chain = option_chain[
        (option_chain["expiry_str"] == expiry_str) &
        (option_chain["option_type"] == "call")
    ].copy()

    if len(exp_chain) < 5:
        print(f"Not enough call options for {expiry_str}: {len(exp_chain)}")
        return None

    # Get underlying price for this expiry (forward price)
    underlying = exp_chain["underlying_price"].iloc[0]
    if pd.isna(underlying) or underlying <= 0:
        underlying = spot

    # Get time to expiry
    expiry_ts = exp_chain["expiry_timestamp"].iloc[0]
    if pd.notna(expiry_ts):
        expiry_dt = pd.to_datetime(expiry_ts, unit="ms", utc=True)
        now = datetime.now(timezone.utc)
        time_to_expiry = max(0, (expiry_dt - now).total_seconds() / (365.25 * 24 * 3600))
    else:
        time_to_expiry = 1/365.25  # Default to 1 day

    # Extract strikes and call prices
    # Use mid price if available, otherwise mark price
    exp_chain["mid_price"] = (exp_chain["bid"].fillna(0) + exp_chain["ask"].fillna(0)) / 2
    exp_chain["price_to_use"] = exp_chain["mid_price"].where(
        exp_chain["mid_price"] > 0,
        exp_chain["mark_price"]
    )

    # Filter out invalid prices
    valid = exp_chain[
        (exp_chain["price_to_use"].notna()) &
        (exp_chain["price_to_use"] > 0) &
        (exp_chain["strike"].notna())
    ].copy()

    if len(valid) < 5:
        print(f"Not enough valid prices for {expiry_str}: {len(valid)}")
        return None

    # Sort by strike
    valid = valid.sort_values("strike")

    strikes = valid["strike"].values
    # Convert prices from BTC to USD
    call_prices_btc = valid["price_to_use"].values
    call_prices_usd = call_prices_btc * underlying

    # Ensure call prices are monotonically decreasing (arbitrage-free)
    # Call price must decrease as strike increases
    for i in range(1, len(call_prices_usd)):
        if call_prices_usd[i] > call_prices_usd[i-1]:
            call_prices_usd[i] = call_prices_usd[i-1] * 0.999

    # Ensure call prices are positive
    call_prices_usd = np.maximum(call_prices_usd, 1e-8)

    # =========================================================================
    # Fit spline to call prices
    # =========================================================================
    try:
        if smoothing > 0:
            # Smoothing spline
            spline = UnivariateSpline(strikes, call_prices_usd, s=smoothing, k=3)
        else:
            # Interpolating spline
            spline = CubicSpline(strikes, call_prices_usd, bc_type='natural')
    except Exception as e:
        print(f"Spline fitting failed: {e}")
        return None

    # =========================================================================
    # Compute derivative: -dC/dK = P(S > K) (discounted)
    # =========================================================================
    # Use finer grid for smooth output
    K_fine = np.linspace(strikes.min(), strikes.max(), 200)
    C_fine = spline(K_fine)

    # First derivative: dC/dK
    dC_dK = spline(K_fine, 1)  # First derivative

    # Survival function: P(S > K) = -e^(rT) * dC/dK
    # For r ≈ 0 (crypto), this simplifies to -dC/dK
    discount = np.exp(risk_free_rate * time_to_expiry)
    survival = -dC_dK * discount

    # Clamp to [0, 1]
    survival = np.clip(survival, 0, 1)

    # CDF = 1 - survival
    cdf_values = 1 - survival

    # Ensure CDF is monotonically increasing
    for i in range(1, len(cdf_values)):
        if cdf_values[i] < cdf_values[i-1]:
            cdf_values[i] = cdf_values[i-1]

    # =========================================================================
    # Compute PDF: f(K) = e^(rT) * d²C/dK²
    # =========================================================================
    d2C_dK2 = spline(K_fine, 2)  # Second derivative
    pdf_values = d2C_dK2 * discount
    pdf_values = np.maximum(pdf_values, 0)  # PDF must be non-negative

    # Create CDF spline for interpolation
    cdf_spline = CubicSpline(K_fine, cdf_values)

    result = ImpliedCDF(
        strikes=K_fine,
        call_prices=C_fine,
        cdf_values=cdf_values,
        pdf_values=pdf_values,
        spot=spot,
        expiry=expiry_str,
        time_to_expiry=time_to_expiry,
        _cdf_spline=cdf_spline
    )

    return result


def parse_deribit_expiry(expiry_str: str) -> datetime:
    """Parse Deribit expiry string to datetime."""
    try:
        return datetime.strptime(expiry_str, "%d%b%y").replace(
            hour=8, minute=0, tzinfo=timezone.utc
        )
    except:
        return None


def find_best_expiry(
    available_expiries: List[str],
    settlement_time: datetime
) -> Optional[str]:
    """Find Deribit expiry that's after the settlement time."""
    valid = []
    for exp_str in available_expiries:
        exp_dt = parse_deribit_expiry(exp_str)
        if exp_dt and exp_dt > settlement_time:
            valid.append((exp_str, exp_dt))

    if not valid:
        return available_expiries[0] if available_expiries else None

    valid.sort(key=lambda x: x[1])
    return valid[0][0]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.deribit_client import DeribitClient

    client = DeribitClient()
    spot = client.get_index_price("BTC")["index_price"]
    chain = client.get_option_chain("BTC")

    print(f"Spot: ${spot:,.2f}")
    print()

    # Test for each expiry
    for expiry in sorted(chain["expiry_str"].unique())[:3]:
        print(f"{'='*60}")
        print(f"Expiry: {expiry}")
        print(f"{'='*60}")

        cdf = extract_implied_cdf(chain, spot, expiry)

        if cdf is None:
            print("Failed to extract CDF")
            continue

        print(cdf.summary())
        print()

        # Test some probabilities
        test_strikes = [
            spot * 0.95,  # 5% below
            spot * 0.98,  # 2% below
            spot,         # ATM
            spot * 1.02,  # 2% above
            spot * 1.05,  # 5% above
        ]

        print("Strike          P(S > K)")
        print("-" * 30)
        for k in test_strikes:
            p = cdf.prob_above(k)
            pct = (k / spot - 1) * 100
            print(f"${k:>10,.0f} ({pct:+.1f}%)  {p:>6.1%}")
        print()
