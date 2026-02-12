"""
2D Implied Volatility Surface Model

Builds a full vol surface from Deribit options across all expiries,
then interpolates for any (strike, time) within the grid.
Uses Black-Scholes to derive probabilities.
"""

import numpy as np
from scipy import interpolate
from scipy.stats import norm
from typing import Optional, Tuple, List
from datetime import datetime, timezone
import pandas as pd
import re


class IVSurface2D:
    """
    2D Implied volatility surface interpolated from Deribit options.

    Interpolates across both strike (log-moneyness) and time dimensions.
    Only provides values within the grid - no extrapolation.
    """

    def __init__(self, spot: float,
                 times: np.ndarray,  # Times to expiry in years
                 strikes_grid: List[np.ndarray],  # Strikes for each expiry
                 ivs_grid: List[np.ndarray],  # IVs for each expiry
                 expiry_strs: List[str]):
        """
        Args:
            spot: Current spot price
            times: Array of times to expiry (years) for each expiry
            strikes_grid: List of strike arrays for each expiry
            ivs_grid: List of IV arrays for each expiry (as decimals)
            expiry_strs: List of expiry strings for reference
        """
        self.spot = spot
        self.expiry_strs = expiry_strs

        # Time bounds
        self.times = np.array(times)
        self.min_time = self.times.min()
        self.max_time = self.times.max()

        # Build unified grid for interpolation
        # We'll use log-moneyness (log(K/S)) for strike dimension
        all_log_m = []
        for strikes in strikes_grid:
            log_m = np.log(strikes / spot)
            all_log_m.extend(log_m)

        # Get common log-moneyness grid
        self.log_m_min = min(all_log_m)
        self.log_m_max = max(all_log_m)

        # Create a regular grid for 2D interpolation
        # Use 100 points in each dimension
        n_log_m = 100
        n_time = len(times)

        self.log_m_grid = np.linspace(self.log_m_min, self.log_m_max, n_log_m)
        self.time_grid = np.sort(times)

        # Build IV matrix: rows = log_moneyness, cols = time
        iv_matrix = np.zeros((n_log_m, n_time))

        for t_idx, (t, strikes, ivs) in enumerate(zip(times, strikes_grid, ivs_grid)):
            # Sort by strike
            sort_idx = np.argsort(strikes)
            strikes_sorted = strikes[sort_idx]
            ivs_sorted = ivs[sort_idx]

            # Convert to log-moneyness
            log_m = np.log(strikes_sorted / spot)

            # Interpolate to common grid (1D for this expiry)
            if len(log_m) >= 2:
                interp_1d = interpolate.interp1d(
                    log_m, ivs_sorted,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(ivs_sorted[0], ivs_sorted[-1])
                )
                iv_matrix[:, t_idx] = interp_1d(self.log_m_grid)
            else:
                iv_matrix[:, t_idx] = ivs_sorted[0] if len(ivs_sorted) > 0 else 0.5

        # Build 2D interpolator
        # RectBivariateSpline for smooth interpolation
        self._iv_interp = interpolate.RectBivariateSpline(
            self.log_m_grid, self.time_grid, iv_matrix,
            kx=1, ky=1  # Linear interpolation (safe, no overshooting)
        )

        # ATM IV at each expiry (log_m = 0)
        self.atm_ivs = {}
        for t, exp_str in zip(times, expiry_strs):
            self.atm_ivs[exp_str] = float(self._iv_interp(0, t)[0, 0])

        # Strike bounds (in actual prices)
        self.min_strike = spot * np.exp(self.log_m_min)
        self.max_strike = spot * np.exp(self.log_m_max)

    def is_in_grid(self, strike: float, time_years: float) -> bool:
        """Check if (strike, time) is within the interpolation grid."""
        log_m = np.log(strike / self.spot)

        in_strike = self.log_m_min <= log_m <= self.log_m_max
        in_time = self.min_time <= time_years <= self.max_time

        return in_strike and in_time

    def get_iv(self, strike: float, time_years: float) -> Optional[float]:
        """
        Get interpolated IV for a given strike and time.

        Returns None if outside the grid.
        """
        if not self.is_in_grid(strike, time_years):
            return None

        log_m = np.log(strike / self.spot)
        return float(self._iv_interp(log_m, time_years)[0, 0])

    def get_atm_iv(self, time_years: float) -> Optional[float]:
        """Get ATM IV for a given time to expiry."""
        if not (self.min_time <= time_years <= self.max_time):
            return None
        return float(self._iv_interp(0, time_years)[0, 0])

    def prob_above(self, strike: float, time_years: float) -> Optional[float]:
        """
        Calculate P(S_T > K) using Black-Scholes.

        Returns None if outside the grid.
        """
        iv = self.get_iv(strike, time_years)
        if iv is None or iv <= 0 or time_years <= 0:
            return None

        # Black-Scholes d2 (assuming r=0 for crypto)
        sqrt_t = np.sqrt(time_years)
        d2 = (np.log(self.spot / strike) - 0.5 * iv**2 * time_years) / (iv * sqrt_t)

        return float(norm.cdf(d2))

    def sigma_distance(self, strike: float, time_years: float) -> Optional[float]:
        """
        Calculate how many standard deviations the strike is from spot.

        Uses ATM IV at the given time to define expected move.
        Returns None if outside the grid.
        """
        atm_iv = self.get_atm_iv(time_years)
        if atm_iv is None or atm_iv <= 0 or time_years <= 0:
            return None

        # Expected 1-sigma move in log-space
        sigma_move = atm_iv * np.sqrt(time_years)

        # Log distance from spot
        log_distance = np.log(strike / self.spot)

        return log_distance / sigma_move if sigma_move > 0 else 0.0

    def strike_at_sigma(self, num_sigma: float, time_years: float) -> Optional[float]:
        """Get strike at num_sigma standard deviations for given time."""
        atm_iv = self.get_atm_iv(time_years)
        if atm_iv is None or time_years <= 0:
            return None

        sigma_move = atm_iv * np.sqrt(time_years)
        return self.spot * np.exp(num_sigma * sigma_move)

    def get_grid_bounds(self) -> dict:
        """Get the bounds of the interpolation grid."""
        return {
            "min_time_years": self.min_time,
            "max_time_years": self.max_time,
            "min_time_hours": self.min_time * 365.25 * 24,
            "max_time_hours": self.max_time * 365.25 * 24,
            "min_strike": self.min_strike,
            "max_strike": self.max_strike,
            "spot": self.spot,
            "expiries": self.expiry_strs
        }


def parse_deribit_expiry(exp_str: str) -> Optional[datetime]:
    """Parse Deribit expiry string to datetime (8:00 UTC)."""
    match = re.match(r'^(\d{1,2})([A-Z]{3})(\d{2})$', exp_str)
    if not match:
        return None

    day = int(match.group(1))
    month_str = match.group(2)
    year = int("20" + match.group(3))

    month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                 "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    month = month_map.get(month_str, 1)

    return datetime(year, month, day, 8, 0, 0, tzinfo=timezone.utc)


def build_iv_surface_2d(chain: pd.DataFrame, spot: float) -> Optional[IVSurface2D]:
    """
    Build a 2D IV surface from the full Deribit option chain.

    Args:
        chain: DataFrame with option chain data (all expiries)
        spot: Current spot price

    Returns:
        IVSurface2D object or None if insufficient data
    """
    now = datetime.now(timezone.utc)

    expiries = sorted(chain['expiry_str'].unique())

    times = []
    strikes_grid = []
    ivs_grid = []
    valid_expiries = []

    for exp_str in expiries:
        exp_dt = parse_deribit_expiry(exp_str)
        if exp_dt is None:
            continue

        # Time to expiry in years
        time_to_exp = (exp_dt - now).total_seconds() / (365.25 * 24 * 3600)
        if time_to_exp <= 0:
            continue  # Skip expired

        # Get data for this expiry
        exp_chain = chain[chain['expiry_str'] == exp_str].copy()

        # Extract strikes and IVs
        strikes = []
        ivs = []

        for strike in sorted(exp_chain['strike'].unique()):
            strike_data = exp_chain[exp_chain['strike'] == strike]

            # Use OTM option IV (more reliable)
            calls = strike_data[strike_data['option_type'] == 'call']
            puts = strike_data[strike_data['option_type'] == 'put']

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

            if iv > 0:
                strikes.append(strike)
                ivs.append(iv / 100)  # Convert from percentage

        if len(strikes) >= 3:  # Need at least 3 strikes
            times.append(time_to_exp)
            strikes_grid.append(np.array(strikes))
            ivs_grid.append(np.array(ivs))
            valid_expiries.append(exp_str)

    if len(times) < 2:  # Need at least 2 expiries
        return None

    return IVSurface2D(
        spot=spot,
        times=np.array(times),
        strikes_grid=strikes_grid,
        ivs_grid=ivs_grid,
        expiry_strs=valid_expiries
    )


def kalshi_settlement_in_grid(settlement_time: datetime, surface: IVSurface2D) -> bool:
    """Check if a Kalshi settlement time falls within the IV surface grid."""
    now = datetime.now(timezone.utc)
    time_years = (settlement_time - now).total_seconds() / (365.25 * 24 * 3600)

    return surface.min_time <= time_years <= surface.max_time
