"""
Feature generators for creating derived features from raw data.

This module provides a plugin-based architecture for generating features that don't
exist in the raw data. Inspired by amp.efp.preprocessing patterns but made more
modular and reusable across different model types.

Key components:
- FeatureGenerator: Abstract base class for all generators
- TemporalFeatureGenerator: Creates raw temporal features (hour, weekday, etc.)
- CyclicalFeatureGenerator: Creates sin/cos encodings of temporal features
- PriceFeatureGenerator: Generates synthetic electricity price profiles
- MultiplicationFeatureGenerator: Creates interaction features via multiplication
- CompositeFeatureGenerator: Chains multiple generators together

Example
-------
>>> from amp.feature_generators import TemporalFeatureGenerator, CyclicalFeatureGenerator, PriceFeatureGenerator
>>> 
>>> # Generate raw temporal features, then encode cyclically
>>> temporal_gen = TemporalFeatureGenerator()
>>> cyclical_gen = CyclicalFeatureGenerator(features=['hour', 'weekday'])
>>> price_gen = PriceFeatureGenerator('day_ahead', base_price=50.0, random_seed=42)
>>> 
>>> df = temporal_gen.generate(df)  # Adds: hour, weekday, month, holiday
>>> df = cyclical_gen.generate(df)  # Adds: hour_sin, hour_cos, weekday_sin, weekday_cos
>>> df = price_gen.generate(df)     # Adds: spot_price
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from attr import dataclass
import pandas
import numpy
import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureGenerator(ABC):
    """
    Base class for feature generators that can add columns to DataFrames.
    
    Feature generators are responsible for creating derived features from raw data,
    such as temporal encodings, lagged features, or physics-based calculations.
    They can be chained together and integrated into DataLoader pipelines.
    
    Design Philosophy
    -----------------
    - Single Responsibility: Each generator handles one type of feature
    - Composable: Generators can be combined via CompositeFeatureGenerator
    - Stateless: Generate method should be idempotent (can be called multiple times)
    - Explicit: get_feature_names() makes capabilities discoverable
    
    Examples
    --------
    >>> class MyGenerator(FeatureGenerator):
    ...     def get_feature_names(self):
    ...         return ['my_feature']
    ...     
    ...     def generate(self, df):
    ...         df = df.copy()
    ...         df['my_feature'] = df['input'] * 2
    ...         return df
    """
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Return list of feature names this generator can create.
        
        Returns
        -------
        list of str
            Names of all features this generator creates.
            
        Notes
        -----
        This method should return all possible feature names, even if some
        might not be generated depending on input data or configuration.
        """
        pass
    
    @abstractmethod
    def generate(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Generate features and add to DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame. Should have DatetimeIndex for temporal generators.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with added features. Original df should not be modified.
            
        Notes
        -----
        - Always copy the input DataFrame to avoid side effects
        - Log warnings for missing dependencies or invalid inputs
        - Raise ValueError only for unrecoverable errors
        """
        pass
    
    def can_generate(self, feature_name: str) -> bool:
        """
        Check if this generator can create the given feature.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature to check.
            
        Returns
        -------
        bool
            True if this generator can create the feature.
        """
        return feature_name in self.get_feature_names()
    
    def validate_input(self, df: pandas.DataFrame, require_datetime_index: bool = True):
        """
        Validate that input DataFrame is suitable for this generator.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame to validate.
        require_datetime_index : bool, default=True
            Whether to require a DatetimeIndex.
            
        Raises
        ------
        ValueError
            If validation fails.
        """
        if not isinstance(df, pandas.DataFrame):
            raise ValueError(
                f"{self.__class__.__name__} requires pandas DataFrame. "
                f"Got {type(df)}"
            )
        
        if require_datetime_index and not isinstance(df.index, pandas.DatetimeIndex):
            raise ValueError(
                f"{self.__class__.__name__} requires DatetimeIndex. "
                f"Got {type(df.index)}"
            )


class TemporalFeatureGenerator(FeatureGenerator):
    """
    Generates raw temporal features from datetime index.
    
    Creates features like hour, weekday, month, and holiday indicators from the
    DataFrame's DatetimeIndex. Logic reused from amp.efp.preprocessing.add_temporal_columns()
    but made more configurable.
    
    Parameters
    ----------
    features : list of str, optional
        Which features to generate. If None, generates all available features.
        Options: 'hour', 'weekday', 'month', 'holiday', 'day_of_year'
    country : str, default='FI'
        Country code for holiday calendar (e.g., 'FI', 'US', 'GB').
        Uses the 'holidays' package.
        
    Attributes
    ----------
    features : list of str
        Features to generate.
    country : str
        Country code for holidays.
        
    Examples
    --------
    >>> gen = TemporalFeatureGenerator()
    >>> df = gen.generate(df)  # Adds hour, weekday, month, holiday
    
    >>> # Only generate specific features
    >>> gen = TemporalFeatureGenerator(features=['hour', 'weekday'])
    >>> df = gen.generate(df)
    
    Notes
    -----
    Requires DatetimeIndex. Holiday feature requires 'holidays' package.
    """
    
    AVAILABLE_FEATURES = ['hour', 'weekday', 'month', 'holiday', 'day_of_year']
    
    def __init__(
        self, 
        features: Optional[List[str]] = None,
        country: str = 'FI'
    ):
        self.features = features or ['hour', 'weekday', 'month', 'holiday']
        self.country = country
        
        # Validate requested features
        invalid = set(self.features) - set(self.AVAILABLE_FEATURES)
        if invalid:
            raise ValueError(
                f"Invalid features requested: {invalid}. "
                f"Available: {self.AVAILABLE_FEATURES}"
            )
    
    def get_feature_names(self) -> List[str]:
        """Return list of temporal features this generator creates."""
        return self.features
    
    def generate(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Add temporal columns based on datetime index.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame with DatetimeIndex.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with added temporal features.
            
        Raises
        ------
        ValueError
            If df does not have DatetimeIndex.
        ImportError
            If 'holidays' package not installed and 'holiday' feature requested.
        """
        self.validate_input(df, require_datetime_index=True)
        df = df.copy()
        
        if 'hour' in self.features:
            df['hour'] = df.index.hour
            logger.debug("Generated 'hour' feature")
            
        if 'weekday' in self.features:
            df['weekday'] = df.index.dayofweek
            logger.debug("Generated 'weekday' feature")
            
        if 'month' in self.features:
            df['month'] = df.index.month
            logger.debug("Generated 'month' feature")
        
        if 'day_of_year' in self.features:
            df['day_of_year'] = df.index.dayofyear
            logger.debug("Generated 'day_of_year' feature")
            
        if 'holiday' in self.features:
            try:
                import holidays
                # Dynamically get holiday calendar
                holiday_calendar = getattr(holidays, self.country, holidays.US)()
                df['holiday'] = [int(x in holiday_calendar) for x in df.index.date]
                logger.debug(f"Generated 'holiday' feature (country={self.country})")
            except ImportError:
                logger.error(
                    "Cannot generate 'holiday' feature: 'holidays' package not installed. "
                    "Install with: pip install holidays"
                )
                raise
        
        generated = [f for f in self.features if f in df.columns]
        logger.info(f"TemporalFeatureGenerator: Generated {len(generated)} features: {generated}")
        
        return df


class CyclicalFeatureGenerator(FeatureGenerator):
    """
    Generates sin/cos cyclical encodings of temporal or periodic features.
    
    Creates sine and cosine transformations of features to preserve their cyclical
    nature (e.g., hour 23 is close to hour 0). Logic reused from 
    amp.efp.pipe_processing.SincosTransformer but operates directly on DataFrames
    instead of within sklearn pipelines.
    
    For a feature 'hour' with max value 24:
    - hour_sin = sin(2π * hour / 24)
    - hour_cos = cos(2π * hour / 24)
    
    Parameters
    ----------
    features : list of str, optional
        Base features to encode cyclically. If None, will auto-detect from
        available columns when generate() is called.
    max_values : dict, optional
        Custom max values for features. If not provided, uses defaults
        (hour=24, weekday=7, month=12, etc.) or infers from data.
        
    Attributes
    ----------
    features : list of str or None
        Base features to encode.
    max_values : dict
        Max values for cyclical encoding (merged with defaults).
        
    Examples
    --------
    >>> # Encode specific features with defaults
    >>> gen = CyclicalFeatureGenerator(features=['hour', 'weekday'])
    >>> df = gen.generate(df)  # Adds: hour_sin, hour_cos, weekday_sin, weekday_cos
    
    >>> # Custom max value for day of year
    >>> gen = CyclicalFeatureGenerator(
    ...     features=['day_of_year'],
    ...     max_values={'day_of_year': 365}
    ... )
    >>> df = gen.generate(df)
    
    >>> # Auto-detect features to encode
    >>> gen = CyclicalFeatureGenerator()  # Will encode any recognized temporal features
    >>> df = gen.generate(df)
    
    Notes
    -----
    - Input features must already exist in the DataFrame
    - For auto-detection, only encodes features with known max values
    - If a feature has NaN values, the encoding will also be NaN
    """
    
    # Default max values for common temporal features
    # Based on amp.efp.pipe_processing.SincosTransformer
    DEFAULT_MAX_VALUES = {
        'hour': 24,
        'weekday': 7,
        'dayofweek': 7,  # Alias
        'month': 12,
        'day_of_year': 365,
        'dayofyear': 365,  # Alias
    }
    
    def __init__(
        self,
        features: Optional[List[str]] = None,
        max_values: Optional[Dict[str, int]] = None
    ):
        self.features = features
        self.max_values = {**self.DEFAULT_MAX_VALUES}
        if max_values:
            self.max_values.update(max_values)
    
    def get_feature_names(self) -> List[str]:
        """
        Return both base features and their sin/cos variants.
        
        Returns
        -------
        list of str
            Feature names in format: [base_sin, base_cos, ...]
            
        Notes
        -----
        If features=None (auto-detect mode), returns empty list since
        output depends on input DataFrame.
        """
        if self.features is None:
            # Can't determine names without knowing input
            # In auto-detect mode, check DEFAULT_MAX_VALUES keys
            names = []
            for feat in self.DEFAULT_MAX_VALUES.keys():
                names.extend([f'{feat}_sin', f'{feat}_cos'])
            return names
        
        names = []
        for feat in self.features:
            names.extend([f'{feat}_sin', f'{feat}_cos'])
        return names
    
    def can_generate(self, feature_name: str) -> bool:
        """
        Check if this generator can create the given feature.
        
        Handles both exact matches and base feature names.
        """
        # Check if it's a sin/cos variant
        if feature_name.endswith('_sin') or feature_name.endswith('_cos'):
            base_feature = feature_name.rsplit('_', 1)[0]
            
            if self.features is not None:
                return base_feature in self.features
            else:
                # Auto-detect mode: can generate if we know the max value
                return base_feature in self.max_values
        
        return False
    
    def _get_max_value(self, feature_name: str, df: pandas.DataFrame) -> int:
        """
        Get max value for cyclical encoding.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature.
        df : pandas.DataFrame
            DataFrame containing the feature.
            
        Returns
        -------
        int
            Max value for cyclical encoding.
            
        Raises
        ------
        ValueError
            If max value cannot be determined.
        """
        # Check if we have a predefined max value
        if feature_name in self.max_values:
            return self.max_values[feature_name]
        
        # Otherwise, infer from data
        if feature_name in df.columns:
            max_val = int(df[feature_name].max()) + 1
            logger.info(
                f"Inferred max value for '{feature_name}': {max_val} "
                f"(from data range)"
            )
            return max_val
        
        raise ValueError(
            f"Cannot determine max value for '{feature_name}'. "
            f"Provide it in max_values or ensure feature exists in DataFrame."
        )
    
    def generate(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Generate sin/cos encodings for temporal features.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing features to encode.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with added sin/cos encoded features.
        """
        self.validate_input(df, require_datetime_index=False)
        df = df.copy()
        
        # Auto-detect features if not specified
        features_to_encode = self.features
        if features_to_encode is None:
            # Look for known temporal features in columns
            features_to_encode = [
                col for col in df.columns 
                if col in self.DEFAULT_MAX_VALUES
            ]
            
            if features_to_encode:
                logger.debug(
                    f"Auto-detected features to encode: {features_to_encode}"
                )
        
        if not features_to_encode:
            logger.warning(
                f"{self.__class__.__name__}: No features to encode. "
                f"Available columns: {list(df.columns)}"
            )
            return df
        
        encoded_count = 0
        for feature in features_to_encode:
            if feature not in df.columns:
                logger.warning(
                    f"Feature '{feature}' not found in DataFrame. "
                    f"Skipping cyclical encoding. Available: {list(df.columns)}"
                )
                continue
            
            try:
                max_val = self._get_max_value(feature, df)
                
                # Create sin/cos encoding
                # Logic from amp.efp.pipe_processing.SincosTransformer
                values = df[feature].values
                df[f'{feature}_sin'] = numpy.sin(2 * numpy.pi * values / max_val)
                df[f'{feature}_cos'] = numpy.cos(2 * numpy.pi * values / max_val)
                
                logger.debug(
                    f"Encoded '{feature}' cyclically (max={max_val}): "
                    f"{feature}_sin, {feature}_cos"
                )
                encoded_count += 1
                
            except Exception as e:
                logger.error(
                    f"Failed to encode '{feature}': {e}"
                )
        
        logger.info(
            f"CyclicalFeatureGenerator: Encoded {encoded_count} features "
            f"(+{encoded_count * 2} columns)"
        )
        
        return df


class PriceFeatureGenerator(FeatureGenerator):
    """
    Generate electricity price features using synthetic profiles.
    
    This generator creates spot_price columns when not present in the data,
    supporting multiple price profile types for testing and simulation.
    Uses amp.electricity_price_utils.PriceProfileGenerator internally.
    
    Parameters
    ----------
    profile_type : str
        Type of price profile. Options:
        - 'constant': Flat price (no variation)
        - 'day_ahead': Realistic day-ahead market with peaks (default)
        - 'high_volatility': Extreme price swings for stress testing
        - 'random': Uniform random prices
        - 'linear_ramp': Linearly increasing prices
        - 'sine_wave': Smooth sinusoidal variation
        - 'two_tier': Simple day/night pricing
    base_price : float
        Base price in €/MWh (default: 50.0)
    price_column : str
        Name of the price column to generate (default: 'spot_price')
    skip_if_exists : bool
        If True, skip generation if price_column already exists (default: True)
    **profile_params : dict
        Additional parameters passed to PriceProfileGenerator.
        See amp.electricity_price_utils.PriceProfileGenerator for details.
        Common parameters:
        - peak_multiplier: For 'day_ahead' (default: 1.8)
        - volatility: For 'high_volatility' (default: 0.5)
        - random_seed: For reproducibility
        
    Attributes
    ----------
    profile_type : str
        Type of price profile
    base_price : float
        Base price in €/MWh
    price_column : str
        Name of price column
    skip_if_exists : bool
        Whether to skip if column exists
    profile_params : dict
        Additional profile parameters
    price_generator : PriceProfileGenerator
        Underlying price generator instance
        
    Examples
    --------
    >>> from amp.feature_generators import PriceFeatureGenerator
    >>> 
    >>> # Generate day-ahead prices if not present
    >>> price_gen = PriceFeatureGenerator(
    ...     'day_ahead', 
    ...     base_price=50.0,
    ...     peak_multiplier=1.8, 
    ...     random_seed=42
    ... )
    >>> df = price_gen.generate(df)  # Adds 'spot_price' column
    
    >>> # Force regenerate prices even if column exists
    >>> price_gen = PriceFeatureGenerator(
    ...     'constant', 
    ...     base_price=60.0,
    ...     skip_if_exists=False
    ... )
    >>> df = price_gen.generate(df)
    
    >>> # Use in a pipeline
    >>> from amp.feature_generators import CompositeFeatureGenerator, TemporalFeatureGenerator
    >>> pipeline = CompositeFeatureGenerator([
    ...     TemporalFeatureGenerator(),
    ...     PriceFeatureGenerator('day_ahead', base_price=50.0)
    ... ])
    >>> df = pipeline.generate(df)
    
    Notes
    -----
    - Generates prices based on DataFrame length (number of timesteps)
    - Price profiles repeat every 24 hours (assumes hourly data)
    - For realistic scenarios, use 'day_ahead' with appropriate base_price
    - For testing edge cases, use 'high_volatility' or 'random'
    """
    
    def __init__(
        self, 
        profile_type: str = 'day_ahead',
        base_price: float = 50.0,
        price_column: str = 'spot_price',
        skip_if_exists: bool = True,
        **profile_params
    ):
        """
        Initialize PriceFeatureGenerator.
        
        Parameters
        ----------
        profile_type : str
            Type of price profile (default: 'day_ahead')
        base_price : float
            Base price in €/MWh (default: 50.0)
        price_column : str
            Name of price column to generate (default: 'spot_price')
        skip_if_exists : bool
            Skip generation if column exists (default: True)
        **profile_params : dict
            Additional parameters for PriceProfileGenerator
        """
        from amp.electricity_price_utils import PriceProfileGenerator
        
        self.profile_type = profile_type
        self.base_price = base_price
        self.price_column = price_column
        self.skip_if_exists = skip_if_exists
        self.profile_params = profile_params
        
        # Initialize price generator
        self.price_generator = PriceProfileGenerator(
            profile_type=profile_type,
            base_price=base_price,
            **profile_params
        )
        
        logger.info(
            f"Initialized PriceFeatureGenerator: type={profile_type}, "
            f"base={base_price} €/MWh, column={price_column}"
        )
    
    def get_feature_names(self) -> List[str]:
        """
        Return the price column name.
        
        Returns
        -------
        list of str
            List containing the price column name.
        """
        return [self.price_column]
    
    def can_generate(self, feature_name: str) -> bool:
        """
        Check if this generator can create the given feature.
        
        Parameters
        ----------
        feature_name : str
            Name of feature to check.
            
        Returns
        -------
        bool
            True if feature_name matches price_column.
        """
        return feature_name == self.price_column
    
    def generate(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Generate price column if not present.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with price column added (or unchanged if skip_if_exists=True
            and column already exists)
            
        Notes
        -----
        - Does not modify input DataFrame (returns copy)
        - Generates hourly prices and repeats for sub-hourly data
        - If price column exists and skip_if_exists=True, returns df unchanged
        """
        # Check if price already exists
        if self.price_column in df.columns:
            if self.skip_if_exists:
                logger.info(
                    f"Price column '{self.price_column}' already exists. "
                    f"Skipping generation (skip_if_exists=True)."
                )
                return df
            else:
                logger.warning(
                    f"Overwriting existing '{self.price_column}' column "
                    f"(skip_if_exists=False)"
                )
        
        # Pass the DatetimeIndex directly; hour-of-day is extracted from timestamps
        prices = self.price_generator.generate(df.index)
        
        # Add to DataFrame
        df = df.copy()
        df[self.price_column] = prices
        
        logger.info(
            f"Generated {self.price_column} ({self.profile_type}): "
            f"{prices.min():.2f} - {prices.max():.2f} €/MWh "
            f"(mean: {prices.mean():.2f}, {len(df)} timesteps)"
        )
        
        return df


class CompositeFeatureGenerator(FeatureGenerator):
    """
    Combines multiple generators in sequence.
    
    Useful for creating feature generation pipelines, such as:
    1. Generate raw temporal features (hour, weekday)
    2. Encode them cyclically (hour_sin, hour_cos, weekday_sin, weekday_cos)
    
    Parameters
    ----------
    generators : list of FeatureGenerator
        Generators to apply in sequence.
        
    Attributes
    ----------
    generators : list of FeatureGenerator
        Ordered list of generators.
        
    Examples
    --------
    >>> # Create a pipeline: temporal -> cyclical
    >>> temporal_gen = TemporalFeatureGenerator()
    >>> cyclical_gen = CyclicalFeatureGenerator(features=['hour', 'weekday'])
    >>> pipeline = CompositeFeatureGenerator([temporal_gen, cyclical_gen])
    >>> 
    >>> df = pipeline.generate(df)  # Applies both in sequence
    
    Notes
    -----
    - Generators are applied left-to-right
    - Each generator receives the output of the previous one
    - If any generator fails, the error is propagated
    """
    
    def __init__(self, generators: List[FeatureGenerator]):
        if not generators:
            raise ValueError("CompositeFeatureGenerator requires at least one generator")
        
        self.generators = generators
    
    def get_feature_names(self) -> List[str]:
        """Return combined feature names from all generators."""
        names = []
        for gen in self.generators:
            names.extend(gen.get_feature_names())
        return names
    
    def generate(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Apply generators in sequence.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with all generators applied.
        """
        for gen in self.generators:
            try:
                df = gen.generate(df)
            except Exception as e:
                logger.error(
                    f"Generator {gen.__class__.__name__} failed in composite pipeline: {e}"
                )
                raise
        
        return df
    
    def can_generate(self, feature_name: str) -> bool:
        """Check if any generator in pipeline can create feature."""
        return any(gen.can_generate(feature_name) for gen in self.generators)


class Building:
    """Helper class to store building parameters for solar calculations."""
    
    def __init__(self, latitude, longitude_loc, longitude_std, 
                 window_direction=None, window_tilt=None):
        self.longitude_std = longitude_std
        self.longitude_loc = longitude_loc
        self.latitude = latitude
        # Handle optional window parameters (set to NaN if not provided)
        self.window_tilt = window_tilt if window_tilt is not None else numpy.nan
        self.window_direction = window_direction if window_direction is not None else numpy.nan

    def params(self):
        """Return building parameters in format expected by solar calculations."""
        return (self.longitude_std, self.longitude_loc, 
                numpy.deg2rad(self.latitude), 
                numpy.deg2rad(self.window_tilt),
                numpy.deg2rad(self.window_direction))

class SolarGainFeatureGenerator(FeatureGenerator):
    """
    Generates solar irradiation features based on geographic location and time.
    
    Uses the existing SolarGains computation logic but provides a FeatureGenerator
    interface for pipeline integration. The generate() method computes all solar
    features, then selects only the requested ones.
    
    Parameters
    ----------
    latitude : float
        Latitude in degrees (e.g., 60.1699 for Helsinki)
    longitude_loc : float
        Local longitude in degrees (e.g., 24.9384 for Helsinki)
    longitude_std : float
        Standard meridian longitude for time zone (e.g., 30 for UTC+2/+3)
    window_direction : float, optional
        Window azimuth angle in degrees (0=South). If None, building-specific
        irradiation will not be calculated.
    window_tilt : float, optional
        Window tilt angle in degrees (90=vertical). Required if window_direction provided.
    optical_thickness : float, default=0.2
        Atmospheric optical thickness parameter
    features : list of str, optional
        Which solar features to include in output. Options:
        - 'sun_angle', 'declination_angle', 'hour_angle', 'sun_azimuth'
        - 't_solar', 'total_irradiation', 'direct_irradiation', 'diffuse_irradiation'
        - 'building_irradiation' (requires window parameters)
        If None, includes all computed features.
        
    Examples
    --------
    >>> # Generate only irradiation features
    >>> solar_gen = SolarFeatureGenerator(
    ...     latitude=60.1699,
    ...     longitude_loc=24.9384,
    ...     longitude_std=30,
    ...     features=['direct_irradiation', 'diffuse_irradiation']
    ... )
    >>> df = solar_gen.generate(df)
    
    >>> # Generate all features including building-specific
    >>> solar_gen = SolarFeatureGenerator(
    ...     latitude=60.1699,
    ...     longitude_loc=24.9384,
    ...     longitude_std=30,
    ...     window_direction=0,  # South
    ...     window_tilt=90,  # Vertical
    ...     features=None  # All features
    ... )
    >>> df = solar_gen.generate(df)
    
    >>> # Integrate with pipeline
    >>> from amp.feature_generators import CompositeFeatureGenerator, TemporalFeatureGenerator
    >>> pipeline = CompositeFeatureGenerator([
    ...     TemporalFeatureGenerator(),
    ...     solar_gen
    ... ])
    >>> df = pipeline.generate(df)
    """
    
    # Available features that can be selected
    AVAILABLE_FEATURES = [
        'sun_angle', 'declination_angle', 'hour_angle', 'sun_azimuth',
        't_solar', 'total_irradiation', 'direct_irradiation', 
        'diffuse_irradiation', 'building_irradiation'
    ]

    k = 0.2  # optical thickness of atmosphere, ~0.2
    G_sc = 1367  # Solar constant [W/m2]
    
    def __init__(
        self,
        latitude: float,
        longitude_loc: float,
        longitude_std: float,
        window_direction: Optional[float] = None,
        window_tilt: Optional[float] = None,
        optical_thickness: float = 0.2,
        features: Optional[List[str]] = None
    ):

        # Store as building_constants dict for SolarGains compatibility
        self.building = Building(
            latitude=latitude,
            longitude_loc=longitude_loc,
            longitude_std=longitude_std,
            window_direction=window_direction,
            window_tilt=window_tilt
        )
        self.optical_thickness = optical_thickness
        
        # Determine which features to include in output
        if features is None:
            # If window params provided, include building_irradiation
            if window_direction is not None and window_tilt is not None:
                self.features = self.AVAILABLE_FEATURES.copy()
            else:
                # Exclude building_irradiation if no window params
                self.features = [f for f in self.AVAILABLE_FEATURES 
                                if f != 'building_irradiation']
        else:
            # Validate requested features
            invalid = set(features) - set(self.AVAILABLE_FEATURES)
            if invalid:
                raise ValueError(
                    f"Invalid features requested: {invalid}. "
                    f"Available: {self.AVAILABLE_FEATURES}"
                )
            
            # Check if building_irradiation requested without params
            if 'building_irradiation' in features:
                if window_direction is None or window_tilt is None:
                    raise ValueError(
                        "window_direction and window_tilt required for building_irradiation"
                    )
            
            self.features = features
    
    def get_feature_names(self) -> List[str]:
        """Return list of features that will be included in output."""
        return self.features
    
    def get_time_arrays(self, idx: pandas.DatetimeIndex) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Extract day index and hour arrays from DatetimeIndex.
        
        Parameters
        ----------
        idx : pandas.DatetimeIndex
            DateTime index to extract time information from
            
        Returns
        -------
        day_index : numpy.ndarray
            Day of year (1-365/366)
        hour : numpy.ndarray
            Hour of day as float (includes minutes/seconds as fractions)
        """
        day_index = idx.to_series().apply(
            lambda x: datetime.datetime.strftime(x, '%j')
        ).astype(float)
        hour = idx.hour + idx.minute / 60 + idx.second / 3600
        return day_index.to_numpy(), hour.to_numpy()
    
    def generate(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Generate selected solar features and add to DataFrame.
        
        This method:
        1. Computes all solar features using compute_features()
        2. Selects only the requested features (self.features)
        3. Merges them into the input DataFrame
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame with DatetimeIndex.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with selected solar features added.
            
        Notes
        -----
        - Requires DatetimeIndex or 'day_index'/'hour' columns
        - Sets negative sun angles to zero irradiation (handled in compute_features)
        """
        self.validate_input(df, require_datetime_index=True)
        df = df.copy()
        
        # Compute all solar features
        solar_features = self.compute_features(df.index)
        
        # Select only requested features
        available_features = [f for f in self.features if f in solar_features.columns]
        
        if len(available_features) < len(self.features):
            missing = set(self.features) - set(solar_features.columns)
            logger.warning(
                f"Could not compute {len(missing)} requested features: {missing}. "
                f"This may be due to missing window parameters for building_irradiation."
            )
        
        # Add selected features to DataFrame
        for feature in available_features:
            df[feature] = solar_features[feature].values
        
        logger.info(
            f"SolarFeatureGenerator: Added {len(available_features)} solar features to DataFrame"
        )
        
        return df
    
    def compute_features(self, df):
        """
        Compute all solar features from input data.
        
        Parameters
        ----------
        df : pandas.DataFrame or pandas.DatetimeIndex
            Either a DataFrame with 'day_index' and 'hour' columns,
            or a DatetimeIndex to extract time information from.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with all computed solar features and MultiIndex
        """
        if isinstance(df, pandas.DataFrame):
            n = df.loc[:, 'day_index'].to_numpy()
            hour = df.loc[:, 'hour'].to_numpy()
        elif isinstance(df, pandas.DatetimeIndex):
            n, hour = self.get_time_arrays(df)
        else:
            logger.error('Wrong input data type for compute_features')
            raise ValueError(
                f"Expected pandas.DataFrame or pandas.DatetimeIndex, got {type(df)}"
            )
            
        longitude_std, longitude_loc, latitude, window_tilt, window_direction = self.building.params()

        # Calculate all intermediate values
        B = self.B(n)
        declination_angle = self.declination_angle(B)
        E = self.E(B)
        t_solar = self.t_solar(hour, longitude_std, longitude_loc, E)
        hour_angle = self.hour_angle(t_solar)
        sun_angle = self.sun_angle(latitude, declination_angle, hour_angle)
        gamma = self.gamma(declination_angle, hour_angle, sun_angle)
        G0 = self.G0(self.G_sc, n)
        m = self.m(sun_angle)
        I_dir = self.I_dir(G0, self.k, m)
        I_diff = self.I_diff(G0, self.k, m)

        # Set irradiation to zero when sun below horizon
        I_dir[sun_angle < 0] = 0
        I_diff[sun_angle < 0] = 0

        # Calculate building-specific irradiation if window parameters provided
        building_irradiation = numpy.zeros_like(I_dir)
        if (not numpy.isnan(window_tilt)) & (not numpy.isnan(window_direction)):
            F_dir = self.F_dir(sun_angle, window_direction, gamma)
            F_diff = self.F_diff(window_tilt)
            building_irradiation = I_dir * F_dir + I_diff * F_diff

        # Create MultiIndex for returned DataFrame
        idx = pandas.MultiIndex.from_arrays([n, hour], names=['day_index', 'hour'])
        
        return pandas.DataFrame({
            'sun_angle': numpy.rad2deg(sun_angle),
            'declination_angle': numpy.rad2deg(declination_angle),
            'hour_angle': numpy.rad2deg(hour_angle),
            'sun_azimuth': numpy.rad2deg(gamma),
            't_solar': t_solar,
            'total_irradiation': G0,
            'direct_irradiation': I_dir,
            'diffuse_irradiation': I_diff,
            'building_irradiation': building_irradiation
        }, index=idx)

    # Solar time
    @staticmethod
    def t_solar(h, longitude_std, longitude_loc, E):
        return h + (longitude_std - longitude_loc) / 15 + E / 60

    # Time alignment
    @staticmethod
    def E(B):
        return 9.87 * numpy.sin(2 * B) - 7.53 * numpy.cos(B) - 1.5 * numpy.sin(B)

    # Normalized day index
    @staticmethod
    def B(n):
        return numpy.deg2rad(360 * (n - 81) / 365)

    # Sun declination angle
    @staticmethod
    def declination_angle(B):
        return numpy.deg2rad(23.45 * numpy.sin(B))

    @staticmethod
    def gamma(declination_angle, hour_angle, sun_angle):
        return numpy.arcsin(numpy.cos(declination_angle) * numpy.sin(hour_angle) / numpy.cos(sun_angle))

    @staticmethod
    def sun_angle(latitude, declination_angle, hour_angle):
        return numpy.arcsin(numpy.sin(latitude) * numpy.sin(declination_angle) + numpy.cos(latitude) * numpy.cos(
            declination_angle) * numpy.cos(hour_angle))

    # Hour angle
    @staticmethod
    def hour_angle(t_solar):
        return numpy.deg2rad(15 * (t_solar - 12))

    # Total irradiation
    @staticmethod
    def I_total(A, g, I_dir, F_dir, I_diff, F_diff):
        return A * g * (I_dir * F_dir + I_diff * F_diff)

    @staticmethod
    def _total_rad(k, m):
        return numpy.clip(numpy.exp(-k * m), 0, 1)

    # Direct irradiation
    def I_dir(self, G0, k, m):
        return G0 * self._total_rad(k, m)

    # Diffuse irradiation
    def I_diff(self, G0, k, m):
        return G0 * (1 - self._total_rad(k, m))

    # Coefficient of direct irradiation
    @staticmethod
    def F_dir(sun_angle, beta, gamma):
        return numpy.cos(sun_angle) * numpy.cos(beta - gamma)

    # Coefficient of diffuse irradiation
    @staticmethod
    def F_diff(phi):
        return (1 + numpy.cos(phi)) / 2

    # Solar irradiation over the earths atmosphere
    @staticmethod
    def G0(G_sc, n):
        return G_sc * (1 + 0.033 * numpy.cos(numpy.deg2rad(360 / 365 * n)))

    # Mass coefficient of atmosphere
    @staticmethod
    def m(sun_angle):
        return 1 / numpy.sin(sun_angle)


class MultiplicationFeatureGenerator(FeatureGenerator):
    """
    Generates new features by multiplying existing features together.
    
    This generator creates interaction features by computing element-wise products
    of specified feature pairs or groups. Useful for capturing non-linear 
    interactions in linear models.
    
    Parameters
    ----------
    feature_multiplications : list of tuple or list of dict
        Specifications for features to multiply. Can be:
        - List of tuples: [(feature1, feature2), ...] 
          Output names will be 'feature1_x_feature2'
        - List of dicts: [{'features': [f1, f2, ...], 'output_name': 'name'}, ...]
          Allows custom output names
    
    Attributes
    ----------
    multiplications : list of dict
        Internal representation of multiplication specs with 'features' and 'output_name'
        
    Examples
    --------
    >>> # Simple pairwise multiplications (auto-named)
    >>> gen = MultiplicationFeatureGenerator([
    ...     ('t_out', 'ws_10min'),  # Creates 't_out_x_ws_10min'
    ...     ('setpoint', 'hour_sin')  # Creates 'setpoint_x_hour_sin'
    ... ])
    >>> df = gen.generate(df)
    
    >>> # Multiple features with custom names
    >>> gen = MultiplicationFeatureGenerator([
    ...     {'features': ['t_out', 'ws_10min'], 'output_name': 'windchill_proxy'},
    ...     {'features': ['t_out', 'hour_sin', 'hour_cos'], 'output_name': 'temp_time_interaction'}
    ... ])
    >>> df = gen.generate(df)
    
    >>> # Three-way interaction
    >>> gen = MultiplicationFeatureGenerator([
    ...     ('feature1', 'feature2', 'feature3')  # Creates 'feature1_x_feature2_x_feature3'
    ... ])
    
    Notes
    -----
    - Missing features are logged as warnings but don't raise errors
    - NaN values in input features propagate to output features
    - Multiplication is element-wise (not matrix multiplication)
    - Features must already exist in the DataFrame before generation
    """
    
    def __init__(self, feature_multiplications: List[Union[Tuple[str, ...], Dict[str, Any]]]):
        if not feature_multiplications:
            raise ValueError("feature_multiplications must contain at least one specification")
        
        # Normalize all specifications to dict format
        self.multiplications = []
        for spec in feature_multiplications:
            if isinstance(spec, tuple):
                # Convert tuple to dict with auto-generated name
                features = list(spec)
                output_name = '_x_'.join(features)
                self.multiplications.append({
                    'features': features,
                    'output_name': output_name
                })
            elif isinstance(spec, dict):
                # Validate dict format
                if 'features' not in spec or 'output_name' not in spec:
                    raise ValueError(
                        "Dict specifications must have 'features' and 'output_name' keys"
                    )
                if len(spec['features']) < 2:
                    raise ValueError(
                        f"Must specify at least 2 features to multiply, got {len(spec['features'])}"
                    )
                self.multiplications.append(spec)
            else:
                raise TypeError(
                    f"feature_multiplications must contain tuples or dicts, got {type(spec)}"
                )
    
    def get_feature_names(self) -> List[str]:
        """Return list of output feature names that will be created."""
        return [mult['output_name'] for mult in self.multiplications]
    
    def generate(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Generate multiplication features and add to DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing features to multiply.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with added multiplication features.
            
        Notes
        -----
        - Missing input features are logged as warnings
        - NaN values propagate through multiplications
        - Original DataFrame is not modified
        """
        df = df.copy()
        
        for mult_spec in self.multiplications:
            features = mult_spec['features']
            output_name = mult_spec['output_name']
            
            # Check if all features exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.warning(
                    f"Cannot create '{output_name}': missing features {missing_features}"
                )
                continue
            
            # Compute element-wise product
            result = df[features[0]].copy()
            for feature in features[1:]:
                result = result * df[feature]
            
            df[output_name] = result
            
            logger.debug(
                f"Created multiplication feature '{output_name}' from {features}"
            )
        
        return df
    
    def can_generate(self, feature_name: str) -> bool:
        """Check if this generator can create the specified feature."""
        return feature_name in self.get_feature_names()