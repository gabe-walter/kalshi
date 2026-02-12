"""
Implied Volatility Surface Model

Interpolates Deribit option IVs to create a smooth volatility surface,
then uses Black-Scholes to derive probabilities for any strike.
"""

import numpy as np
from scipy import interpolate
from scipy.stats import norm
from typing import Optional, Tuple
import pandas as pd


class IVSurface:
    """
    Implied volatility surface interpolated from Deribit options.

    Provides:
    - Interpolated IV for any strike
    - Black-Scholes probability P(S > K)
    - Sigma distance calculations
    """

    def __init__(self, spot: float, strikes: np.ndarray, ivs: np.ndarray,
                 time_to_expiry: float, expiry_str: str):
        """
        Args:
            spot: Current spot price
            strikes: Array of option strikes
            ivs: Array of implied volatilities (as decimals, e.g., 0.5 for 50%)
            time_to_expiry: Time to expiry in years
            expiry_str: Expiry string for reference (e.g., '7FEB26')
        """
        self.spot = spot
        self.time_to_expiry = time_to_expiry
        self.expiry_str = expiry_str

        # Sort by strike
        sort_idx = np.argsort(strikes)
        self.strikes = strikes[sort_idx]
        self.ivs = ivs[sort_idx]

        # Log-moneyness for interpolation (more stable than raw strikes)
        self.log_moneyness = np.log(self.strikes / spot)

        # Build interpolator (cubic spline with flat extrapolation)
        self._iv_interp = interpolate.interp1d(
            self.log_moneyness, self.ivs,
            kind='cubic',
            bounds_error=False,
            fill_value=(self.ivs[0], self.ivs[-1])  # Flat extrapolation
        )

        # ATM IV (interpolated at spot)
        self.atm_iv = float(self._iv_interp(0))

        # Strike range
        self.min_strike = self.strikes.min()
        self.max_strike = self.strikes.max()

    def get_iv(self, strike: float) -> float:
        """Get interpolated IV for a given strike."""
        log_m = np.log(strike / self.spot)
        return float(self._iv_interp(log_m))

    def prob_above(self, strike: float, time_years: Optional[float] = None) -> float:
        """
        Calculate P(S_T > K) using Black-Scholes.

        Args:
            strike: Strike price
            time_years: Time to expiry in years (uses surface's time if None)

        Returns:
            Probability that spot will be above strike at expiry
        """
        T = time_years if time_years is not None else self.time_to_expiry
        if T <= 0:
            return 1.0 if self.spot > strike else 0.0

        iv = self.get_iv(strike)
        if iv <= 0:
            return 1.0 if self.spot > strike else 0.0

        # Black-Scholes d2 (assuming r=0 for crypto)
        # d2 = (ln(S/K) - 0.5*σ²*T) / (σ*√T)
        sqrt_t = np.sqrt(T)
        d2 = (np.log(self.spot / strike) - 0.5 * iv**2 * T) / (iv * sqrt_t)

        # P(S > K) = N(d2)
        return float(norm.cdf(d2))

    def sigma_distance(self, strike: float) -> float:
        """
        Calculate how many standard deviations the strike is from spot.

        Uses ATM IV to define the expected move.

        Returns:
            Number of standard deviations (positive = OTM call, negative = OTM put)
        """
        if self.time_to_expiry <= 0 or self.atm_iv <= 0:
            return 0.0

        # Expected 1-sigma move in log-space
        sigma_move = self.atm_iv * np.sqrt(self.time_to_expiry)

        # Log distance from spot
        log_distance = np.log(strike / self.spot)

        # Sigma distance (signed)
        return log_distance / sigma_move if sigma_move > 0 else 0.0

    def strike_at_sigma(self, num_sigma: float) -> float:
        """
        Get the strike price that is num_sigma standard deviations from spot.

        Args:
            num_sigma: Number of sigmas (positive = above spot, negative = below)

        Returns:
            Strike price
        """
        if self.time_to_expiry <= 0 or self.atm_iv <= 0:
            return self.spot

        sigma_move = self.atm_iv * np.sqrt(self.time_to_expiry)
        return self.spot * np.exp(num_sigma * sigma_move)

    def expected_move_pct(self, num_sigma: float = 1.0) -> float:
        """Get expected percentage move for given sigma."""
        if self.time_to_expiry <= 0 or self.atm_iv <= 0:
            return 0.0

        sigma_move = self.atm_iv * np.sqrt(self.time_to_expiry)
        return np.exp(num_sigma * sigma_move) - 1


def extract_iv_surface(chain: pd.DataFrame, spot: float, expiry_str: str,
                       min_delta: float = 0.05, max_delta: float = 0.95) -> Optional[IVSurface]:
    """
    Extract an IV surface from Deribit option chain data.

    Args:
        chain: DataFrame with option chain data
        spot: Current spot price
        expiry_str: Expiry to extract (e.g., '7FEB26')
        min_delta: Minimum delta to include (filters far OTM)
        max_delta: Maximum delta to include (filters deep ITM)

    Returns:
        IVSurface object or None if insufficient data
    """
    # Filter to expiry
    exp_chain = chain[chain['expiry_str'] == expiry_str].copy()
    if len(exp_chain) == 0:
        return None

    # Get time to expiry from the chain
    if 'time_to_expiry' in exp_chain.columns:
        time_to_expiry = exp_chain['time_to_expiry'].iloc[0]
    else:
        # Estimate from expiry string
        time_to_expiry = 1/365  # Default to 1 day

    # Filter by delta if available (removes illiquid far OTM options)
    if 'delta' in exp_chain.columns:
        exp_chain = exp_chain[
            (exp_chain['delta'].abs() >= min_delta) &
            (exp_chain['delta'].abs() <= max_delta)
        ]

    # Get unique strikes with their IVs
    # Use mark_iv, prefer calls for OTM calls and puts for OTM puts
    strikes = []
    ivs = []

    for strike in sorted(exp_chain['strike'].unique()):
        strike_data = exp_chain[exp_chain['strike'] == strike]

        # Get call and put IVs
        calls = strike_data[strike_data['option_type'] == 'call']
        puts = strike_data[strike_data['option_type'] == 'put']

        # Use OTM option IV (more liquid/reliable)
        if strike > spot and len(calls) > 0:
            iv = calls['mark_iv'].iloc[0]
        elif strike <= spot and len(puts) > 0:
            iv = puts['mark_iv'].iloc[0]
        elif len(calls) > 0:
            iv = calls['mark_iv'].iloc[0]
        elif len(puts) > 0:
            iv = puts['mark_iv'].iloc[0]
        else:
            continue

        # mark_iv from Deribit is in percentage (e.g., 50 for 50%)
        if iv > 0:
            strikes.append(strike)
            ivs.append(iv / 100)  # Convert to decimal

    if len(strikes) < 3:
        return None

    return IVSurface(
        spot=spot,
        strikes=np.array(strikes),
        ivs=np.array(ivs),
        time_to_expiry=time_to_expiry,
        expiry_str=expiry_str
    )


def find_best_expiry(available_expiries: list, target_time) -> Optional[str]:
    """
    Find the best Deribit expiry for a given Kalshi settlement time.

    Prefers expiries that are after the target time but as close as possible.
    """
    from datetime import datetime, timezone
    import pandas as pd

    if isinstance(target_time, str):
        target_time = pd.to_datetime(target_time, utc=True)

    target_dt = target_time.to_pydatetime() if hasattr(target_time, 'to_pydatetime') else target_time

    def parse_expiry(exp_str: str) -> datetime:
        """Parse Deribit expiry string to datetime."""
        import re
        match = re.match(r'^(\d{1,2})([A-Z]{3})(\d{2})$', exp_str)
        if not match:
            return datetime.max.replace(tzinfo=timezone.utc)

        day = int(match.group(1))
        month_str = match.group(2)
        year = int("20" + match.group(3))

        month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                     "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
        month = month_map.get(month_str, 1)

        # Deribit options expire at 8:00 UTC
        return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)

    best_expiry = None
    best_diff = float('inf')

    for exp_str in available_expiries:
        exp_dt = parse_expiry(exp_str)

        # Must be after target (or same day)
        if exp_dt < target_dt:
            continue

        diff = (exp_dt - target_dt).total_seconds()
        if diff < best_diff:
            best_diff = diff
            best_expiry = exp_str

    return best_expiry
