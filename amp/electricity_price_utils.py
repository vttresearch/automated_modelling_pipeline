"""
Utility functions for generating and working with electricity prices.

This module provides functions to:
1. Generate synthetic electricity price profiles
2. Load real price data (if available)
3. Analyze price patterns
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


def generate_day_ahead_prices(
    time_indices: Union[int, np.ndarray, 'pd.DatetimeIndex'] = 24,
    base_price: float = 0.12,
    peak_multiplier: float = 1.8,
    off_peak_multiplier: float = 0.7,
    price_volatility: float = 0.15,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic day-ahead electricity prices with realistic patterns.
    
    This creates a typical daily price profile with:
    - Low prices during night (off-peak): 23:00 - 06:00
    - High prices during morning/evening peaks: 07:00-09:00, 17:00-21:00
    - Medium prices during day: 10:00-16:00
    - Random volatility to simulate market fluctuations
    
    Args:
        time_indices: One of:
            - int: number of hourly timesteps to generate starting from hour 0.
            - pandas.DatetimeIndex or array of datetime64: hour-of-day is read
              directly from the timestamps (hour + minute/60 + second/3600).
            - 1-D float/int array: treated as fractional hour-of-day values (0–24).
        base_price: Base electricity price in $/kWh or €/kWh (default: 0.12)
        peak_multiplier: Multiplier for peak hours (default: 1.8x base)
        off_peak_multiplier: Multiplier for off-peak hours (default: 0.7x base)
        price_volatility: Random volatility as fraction of price (default: 0.15 = ±15%)
        random_seed: Seed for reproducibility
    
    Returns:
        prices: Array of electricity prices (len(time_indices),) in $/kWh or €/kWh
    
    Example:
        >>> prices = generate_day_ahead_prices(24, base_price=0.12)
        >>> print(f"Night price: ${prices[2]:.3f}/kWh")  # Around 02:00
        >>> print(f"Peak price: ${prices[18]:.3f}/kWh")  # Around 18:00
        >>> # Pass a DatetimeIndex for correct time-of-day alignment:
        >>> prices = generate_day_ahead_prices(df.index, base_price=0.12)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    hour_of_day = _to_hour_of_day(time_indices, data_freq_minutes=60)

    multiplier = np.ones(len(hour_of_day))
    multiplier[(hour_of_day < 6) | (hour_of_day >= 23)] = off_peak_multiplier
    multiplier[((hour_of_day >= 7) & (hour_of_day <= 9)) | ((hour_of_day >= 17) & (hour_of_day <= 21))] = peak_multiplier
    multiplier[((hour_of_day >= 6) & (hour_of_day < 7)) | ((hour_of_day >= 22) & (hour_of_day < 23))] = (off_peak_multiplier + peak_multiplier) / 2

    prices = base_price * multiplier
    noise = np.random.normal(0, price_volatility * prices)
    prices = np.maximum(0.01, prices + noise)

    return prices


def _to_hour_of_day(
    time_input: Union[int, np.ndarray, 'pd.DatetimeIndex'],
    data_freq_minutes: int = 60
) -> np.ndarray:
    """
    Convert various time input types to fractional hour-of-day (0 <= h < 24).

    Parameters
    ----------
    time_input : int, pd.DatetimeIndex, array of datetime64, or float array
        - int n: generates n evenly-spaced hourly steps from hour 0, using
          ``data_freq_minutes`` to convert steps to hours.
        - pd.DatetimeIndex / array of datetime64: hour extracted directly from
          timestamps as ``hour + minute/60 + second/3600``.
        - numeric array: treated as fractional hour-of-day values directly
          (modulo 24 applied for safety).
    data_freq_minutes : int
        Step size in minutes; only used when ``time_input`` is an int.

    Returns
    -------
    hour_of_day : np.ndarray
        Fractional hours in [0, 24).
    """
    if isinstance(time_input, (int, np.integer)):
        steps = np.arange(int(time_input), dtype=float)
        return (steps * data_freq_minutes / 60.0) % 24

    if isinstance(time_input, pd.DatetimeIndex):
        return (time_input.hour + time_input.minute / 60.0 + time_input.second / 3600.0).to_numpy(dtype=float)

    arr = np.asarray(time_input)
    if np.issubdtype(arr.dtype, np.datetime64):
        idx = pd.DatetimeIndex(arr)
        return (idx.hour + idx.minute / 60.0 + idx.second / 3600.0).to_numpy(dtype=float)

    # Numeric array — treat as fractional hour-of-day
    return arr.astype(float) % 24


def analyze_price_profile(prices: np.ndarray) -> dict:
    """
    Analyze an electricity price profile.
    
    Args:
        prices: Array of electricity prices
    
    Returns:
        analysis: Dictionary with price statistics
    """
    return {
        'mean': np.mean(prices),
        'std': np.std(prices),
        'min': np.min(prices),
        'max': np.max(prices),
        'min_hour': np.argmin(prices),
        'max_hour': np.argmax(prices),
        'range': np.max(prices) - np.min(prices),
        'coefficient_of_variation': np.std(prices) / np.mean(prices)
    }


def print_price_summary(prices: np.ndarray, title: str = "Electricity Prices"):
    """
    Print a summary of electricity prices.
    
    Args:
        prices: Array of electricity prices
        title: Title for the summary
    """
    stats = analyze_price_profile(prices)
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Mean price:     ${stats['mean']:.3f}/kWh")
    print(f"Std deviation:  ${stats['std']:.3f}/kWh")
    print(f"Min price:      ${stats['min']:.3f}/kWh at hour {stats['min_hour']}")
    print(f"Max price:      ${stats['max']:.3f}/kWh at hour {stats['max_hour']}")
    print(f"Price range:    ${stats['range']:.3f}/kWh")
    print(f"Volatility:     {stats['coefficient_of_variation']:.1%}")
    print(f"{'='*60}\n")


class PriceProfileGenerator:
    """
    Generator for synthetic electricity price profiles with various patterns.
    
    Supports multiple price profile types for testing different scenarios:
    - 'constant': Flat price (no variation)
    - 'day_ahead': Realistic day-ahead market with peaks
    - 'high_volatility': Extreme price swings
    - 'random': Uniform random prices
    - 'linear_ramp': Linearly increasing prices
    - 'sine_wave': Smooth sinusoidal variation
    - 'two_tier': Simple day/night pricing
    
    Parameters
    ----------
    profile_type : str
        Type of price profile (see above). Default: 'day_ahead'
    base_price : float
        Base price in €/MWh. Default: 50.0
    data_freq_minutes : int
        Data resolution in minutes (default: 60 for hourly). For sub-hourly data
        (e.g., 15 or 30), hourly price patterns are repeated to match the resolution.
    **kwargs : dict
        Additional parameters for specific profile types (see generate() method)
    
    Examples
    --------
    >>> # Realistic day-ahead market
    >>> gen = PriceProfileGenerator('day_ahead', base_price=50.0, 
    ...                              peak_multiplier=2.0, price_volatility=0.2)
    >>> prices = gen.generate(96)
    
    >>> # High volatility for stress testing
    >>> gen = PriceProfileGenerator('high_volatility', base_price=50.0, volatility=0.8)
    >>> prices = gen.generate(48)
    
    >>> # Simple day/night pricing
    >>> gen = PriceProfileGenerator('two_tier', day_price=60.0, night_price=30.0)
    >>> prices = gen.generate(24)
    """
    
    VALID_PROFILES = [
        'constant', 'day_ahead', 'high_volatility', 'random',
        'linear_ramp', 'sine_wave', 'two_tier'
    ]
    
    def __init__(self, profile_type: str = 'day_ahead', base_price: float = 50.0, data_freq_minutes: int = 60, **kwargs):
        """
        Initialize price profile generator.
        
        Parameters
        ----------
        profile_type : str
            Type of price profile. See class docstring for options.
        base_price : float
            Base price in €/MWh
        data_freq_minutes : int
            Data resolution in minutes (default: 60 for hourly). For sub-hourly
            resolutions (e.g., 15 or 30), hourly price patterns are repeated to
            fill each hour with the same price value.
        **kwargs : dict
            Profile-specific parameters (see generate() method)
        """
        self.profile_type = profile_type.lower()
        self.base_price = base_price
        self.data_freq_minutes = data_freq_minutes
        self.params = kwargs
        
        if self.profile_type not in self.VALID_PROFILES:
            raise ValueError(
                f"Unknown profile_type '{profile_type}'. "
                f"Choose from: {', '.join(self.VALID_PROFILES)}"
            )
    
    def generate(self, time_input: Union[int, np.ndarray, 'pd.DatetimeIndex']) -> np.ndarray:
        """
        Generate price profile for the given time input.
        
        Parameters
        ----------
        time_input : int, pd.DatetimeIndex, array of datetime64, or float array
            If int: generates that many timesteps using ``data_freq_minutes`` to
            map steps to hours (backward compatible).
            If pd.DatetimeIndex or array of datetime64: hour-of-day is read
            directly from the timestamps.
            If numeric array: treated as fractional hour-of-day values (0–24).
        
        Returns
        -------
        prices : np.ndarray
            Array of electricity prices (len(time_indices),) in €/MWh
        
        Profile-Specific Parameters
        ----------------------------
        For 'day_ahead':
            peak_multiplier : float
                Peak hour multiplier (default: 1.8)
            off_peak_multiplier : float
                Off-peak multiplier (default: 0.7)
            price_volatility : float
                Random volatility fraction (default: 0.15)
            random_seed : int
                Seed for reproducibility
        
        For 'high_volatility':
            volatility : float
                Volatility level (default: 0.5)
            random_seed : int
                Seed for reproducibility
        
        For 'random':
            min_price : float
                Minimum price (default: base_price * 0.5)
            max_price : float
                Maximum price (default: base_price * 2.0)
            random_seed : int
                Seed for reproducibility
        
        For 'linear_ramp':
            start_price : float
                Starting price (default: base_price * 0.5)
            end_price : float
                Ending price (default: base_price * 2.0)
        
        For 'sine_wave':
            amplitude : float
                Wave amplitude (default: base_price * 0.5)
            period_hours : float
                Period in hours (default: 24.0)
        
        For 'two_tier':
            day_price : float
                Daytime price (default: base_price * 1.2)
            night_price : float
                Nighttime price (default: base_price * 0.8)
            day_start : int
                Day start hour (default: 7)
            day_end : int
                Day end hour (default: 22)
        """
        hour_of_day = _to_hour_of_day(time_input, self.data_freq_minutes)

        if self.profile_type == 'constant':
            return self._generate_constant(hour_of_day)
        elif self.profile_type == 'day_ahead':
            return self._generate_day_ahead(hour_of_day)
        elif self.profile_type == 'high_volatility':
            return self._generate_high_volatility(hour_of_day)
        elif self.profile_type == 'random':
            return self._generate_random(hour_of_day)
        elif self.profile_type == 'linear_ramp':
            return self._generate_linear_ramp(hour_of_day)
        elif self.profile_type == 'sine_wave':
            return self._generate_sine_wave(hour_of_day)
        elif self.profile_type == 'two_tier':
            return self._generate_two_tier(hour_of_day)
    
    def _generate_constant(self, time_indices: np.ndarray) -> np.ndarray:
        """Flat price - no variation."""
        return np.full(len(time_indices), self.base_price)
    
    def _generate_day_ahead(self, hour_of_day: np.ndarray) -> np.ndarray:
        """Realistic day-ahead market pattern."""
        peak_mult = self.params.get('peak_multiplier', 1.8)
        off_peak_mult = self.params.get('off_peak_multiplier', 0.7)
        volatility = self.params.get('price_volatility', 0.15)
        seed = self.params.get('random_seed', None)

        if seed is not None:
            np.random.seed(seed)

        multiplier = np.ones(len(hour_of_day))
        multiplier[(hour_of_day < 6) | (hour_of_day >= 23)] = off_peak_mult
        multiplier[((hour_of_day >= 7) & (hour_of_day <= 9)) | ((hour_of_day >= 17) & (hour_of_day <= 21))] = peak_mult
        multiplier[((hour_of_day >= 6) & (hour_of_day < 7)) | ((hour_of_day >= 22) & (hour_of_day < 23))] = (off_peak_mult + peak_mult) / 2

        prices = self.base_price * multiplier
        noise = np.random.normal(0, volatility * prices)
        prices = np.maximum(self.base_price * 0.01, prices + noise)

        return prices
    
    def _generate_high_volatility(self, hour_of_day: np.ndarray) -> np.ndarray:
        """Extreme price swings for stress testing."""
        volatility = self.params.get('volatility', 0.5)
        seed = self.params.get('random_seed', None)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Start with day-ahead pattern then add extreme noise
        # hour_of_day is passed directly (already converted)
        base_gen = PriceProfileGenerator(
            'day_ahead',
            base_price=self.base_price,
            price_volatility=0.0,
            random_seed=seed
        )
        base_pattern = base_gen.generate(hour_of_day)
        
        # Add large random shocks
        shocks = np.random.normal(0, volatility * self.base_price, len(hour_of_day))
        prices = np.maximum(base_pattern + shocks, self.base_price * 0.1)  # Floor at 10% of base
        
        return prices
    
    def _generate_random(self, hour_of_day: np.ndarray) -> np.ndarray:
        """Uniform random prices."""
        min_price = self.params.get('min_price', self.base_price * 0.5)
        max_price = self.params.get('max_price', self.base_price * 2.0)
        seed = self.params.get('random_seed', None)
        
        if seed is not None:
            np.random.seed(seed)
        
        return np.random.uniform(min_price, max_price, len(hour_of_day))
    
    def _generate_linear_ramp(self, hour_of_day: np.ndarray) -> np.ndarray:
        """Linearly increasing prices."""
        start_price = self.params.get('start_price', self.base_price * 0.5)
        end_price = self.params.get('end_price', self.base_price * 2.0)
        
        return np.linspace(start_price, end_price, len(hour_of_day))
    
    def _generate_sine_wave(self, hour_of_day: np.ndarray) -> np.ndarray:
        """Smooth sinusoidal variation."""
        amplitude = self.params.get('amplitude', self.base_price * 0.5)
        period_hours = self.params.get('period_hours', 24.0)

        return self.base_price + amplitude * np.sin(2 * np.pi * hour_of_day / period_hours)
    
    def _generate_two_tier(self, hour_of_day: np.ndarray) -> np.ndarray:
        """Simple day/night pricing."""
        day_price = self.params.get('day_price', self.base_price * 1.2)
        night_price = self.params.get('night_price', self.base_price * 0.8)
        day_start = self.params.get('day_start', 7)
        day_end = self.params.get('day_end', 22)

        prices = np.where((hour_of_day >= day_start) & (hour_of_day < day_end), day_price, night_price)

        return prices
    
    def __repr__(self):
        return (f"PriceProfileGenerator(profile_type='{self.profile_type}', "
                f"base_price={self.base_price}, data_freq_minutes={self.data_freq_minutes}, "
                f"params={self.params})")


def generate_price_profile(
    time_indices: Union[int, np.ndarray],
    profile_type: str = 'day_ahead',
    base_price: float = 50.0,
    **kwargs
) -> np.ndarray:
    """
    Convenience function to generate synthetic electricity price profiles.
    
    This is a functional wrapper around PriceProfileGenerator class.
    For cleaner code when generating multiple profiles, use the class directly.
    
    Parameters
    ----------
    time_indices : int or np.ndarray
        If int: number of timesteps to generate (starts from step 0).
        If np.ndarray: explicit step indices for each output position,
        allowing correct price alignment when data does not start at midnight.
    profile_type : str
        Type of price profile (see PriceProfileGenerator for options)
    base_price : float
        Base price in €/MWh (default: 50.0)
    **kwargs : dict
        Additional parameters for specific profile types
    
    Returns
    -------
    prices : np.ndarray
        Array of electricity prices (len(time_indices),) in €/MWh
    
    See Also
    --------
    PriceProfileGenerator : Class-based API with full documentation
    
    Examples
    --------
    >>> # Functional API (quick one-off generation)
    >>> prices = generate_price_profile(24, 'constant', base_price=50.0)
    
    >>> # Class-based API (cleaner when generating multiple profiles)
    >>> gen = PriceProfileGenerator('day_ahead', base_price=50.0)
    >>> prices = gen.generate(96)
    
    >>> # Pass explicit step indices for correct time-of-day alignment:
    >>> prices = generate_price_profile(np.arange(48, 96), 'day_ahead', base_price=50.0)
    """
    generator = PriceProfileGenerator(profile_type, base_price, **kwargs)
    return generator.generate(time_indices)
