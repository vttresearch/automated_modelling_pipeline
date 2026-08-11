""" Dataloader related functionality. """


from typing import List, Tuple, Optional, Dict, Any, Union
from abc import ABC
import logging
import numpy
import pandas
import datetime
import logging
from dataclasses import dataclass

from amp.utils import create_dataset
from amp.feature_generators import FeatureGenerator

logger = logging.getLogger(__name__)


def create_samples(
        df: pandas.DataFrame,
        features: List[str],
        targets: Union[str, List[str]],
        input_window: Tuple[int, int],
        output_format: str = 'list',
        stride: int = 1,
        normalize: bool = False,
        norm_params: dict = None,
        resample_freq: int = 1,
        period_name: str = 'data',
        **kwargs
    ) -> Union[List[Dict], List[pandas.DataFrame], numpy.ndarray, 'torch.Tensor']:
        """
        Create samples dynamically from continuous timeseries data.
        
        Generates sliding window samples without caching. Samples are filtered to
        exclude those with NaN values or time gaps.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with timeseries data to create samples from
        features : list of str
            Feature column names to include
        targets : str or list of str
            Target column name(s) to predict
        input_window : tuple of (int, int)
            Window boundaries as (start_offset, end_offset) in timesteps
        output_format : str, default='list'
            Output format:
            - 'list': List of dicts with 'data', 'start_time', 'end_time' keys
            - 'dataframes': List of DataFrames (one per sample)
            - 'array': Numpy array of shape (n_samples, window_size, n_features)
            - 'tensor': PyTorch tensor (requires torch)
        stride : int, default=1
            Stride between consecutive samples
        normalize : bool, default=False
            Whether to normalize the sample data
        norm_params : dict, optional
            Normalization parameters with 'mean' and 'std' keys (required if normalize=True)
        resample_freq : int, default=1
            Expected frequency in minutes for gap detection
        period_name : str, default='data'
            Name of the period for logging purposes
        **kwargs
            Additional keyword arguments (reserved for future use)
            
        Returns
        -------
        samples : list, numpy.ndarray, or torch.Tensor
            Samples in requested format (only accepted samples)
            
        Raises
        ------
        ValueError
            If df is None/empty or normalization requested without norm_params
        ImportError
            If output_format='tensor' but torch is not installed
        """
        if df is None or len(df) == 0:
            raise ValueError(
                f"No data available for period '{period_name}'. "
                "Provide a valid DataFrame."
            )
        
        if normalize and norm_params is None:
            raise ValueError("norm_params must be provided when normalize=True")
        
        # Normalize targets to list
        if isinstance(targets, str):
            targets = [targets]
        
        required_columns = list(set(list(features) + targets))
        
        available_columns = [col for col in required_columns if col in df.columns]
        if len(available_columns) < len(required_columns):
            missing = set(required_columns) - set(available_columns)
            logger.warning(
                f"Missing {len(missing)} columns in {period_name} data: {missing}"
            )
        
        df_filtered = df[available_columns].copy()
        
        # Diagnostic: Check for gaps in the entire DataFrame
        full_gaps = find_gap_indices(df_filtered, tolerance=1.0, freq=resample_freq)
        if len(full_gaps) > 0:
            gap_percentage = (len(full_gaps) / len(df_filtered)) * 100
            logger.warning(
                f"Detected {len(full_gaps)} gaps in {period_name} data "
                f"({gap_percentage:.2f}% of {len(df_filtered)} timesteps). "
                f"This may affect sample acceptance rate."
            )
        else:
            logger.info(f"No gaps detected in {period_name} data ({len(df_filtered)} timesteps)")
        
        min_offset, max_offset = input_window[0], input_window[1]
        
        start_i = abs(min_offset)
        end_i = len(df_filtered) - max_offset  # window_end = i + max_offset + 1, so max i is len - max_offset - 1
        
        if start_i >= end_i:
            raise ValueError(
                f"Cannot create samples: data length ({len(df_filtered)}) too short for "
                f"window range ({min_offset}, {max_offset}). Need at least {abs(min_offset) + max_offset} samples."
            )
                
        samples = []
        rejected_nan = 0
        rejected_gap = 0
        
        for i in range(start_i, end_i, stride):
            window_start = i + min_offset
            window_end = i + max_offset + 1  # +1 for inclusive indexing
            sample_data = df_filtered.iloc[window_start:window_end].copy()
            
            if sample_data.isnull().any().any():
                rejected_nan += 1
                continue
            
            if len(sample_data) > 1 and len(find_gap_indices(sample_data, tolerance=1.0, freq=resample_freq)) > 0:
                rejected_gap += 1
                continue
            
            if normalize:
                sample_data = normalize_data(sample_data, norm_params)
            
            sample = {
                'data': sample_data,
                'start_idx': window_start,
                'end_idx': window_end - 1,
                'start_time': df_filtered.index[window_start],
                'end_time': df_filtered.index[window_end - 1],
                'window_size': window_end - window_start,
                'center_idx': i
            }
            samples.append(sample)
        
        total_possible = len(range(start_i, end_i, stride))
        rejected_count = rejected_nan + rejected_gap
        acceptance_rate = (len(samples) / total_possible * 100) if total_possible > 0 else 0
        
        logger.info(
            f"Created {len(samples)} samples for {period_name} "
            f"(acceptance rate: {acceptance_rate:.1f}%, rejected: {rejected_count} "
            f"[NaN: {rejected_nan}, gaps: {rejected_gap}])"
        )
        
        # Convert to requested output format
        if output_format == 'list':
            return samples
        elif output_format == 'dataframes':
            return [s['data'] for s in samples]
        elif output_format == 'array':
            return numpy.array([s['data'].values for s in samples])
        elif output_format == 'tensor':
            try:
                import torch
                return torch.FloatTensor(numpy.array([s['data'].values for s in samples]))
            except ImportError:
                raise ImportError(
                    "PyTorch is required for output_format='tensor'. "
                    "Install it with: pip install torch"
                )
        else:
            raise ValueError(
                f"Invalid output_format '{output_format}'. "
                f"Must be one of: 'list', 'dataframes', 'array', 'tensor'"
            )

def find_gap_indices(df: pandas.DataFrame, tolerance: float = None, freq: int = 1) -> pandas.Index:
        """
        Find indices where time gaps occur in a DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with DatetimeIndex to analyze
        tolerance : float, optional
            Tolerance factor for expected frequency (e.g., 1.1 = 10% tolerance).
            If None, uses exact frequency matching with no tolerance.
            Default is None.
        freq : int, default=1
            Expected frequency in minutes
            
        Returns
        -------
        pandas.Index
            Index values where gaps occur (empty if no gaps)
        """
        if df is None or len(df) <= 1:
            return pandas.Index([])
        
        # Calculate time differences between consecutive rows
        time_diffs = df.index.to_series().diff()
        
        # Expected frequency and maximum allowed gap
        expected_freq = pandas.Timedelta(minutes=freq)
        max_gap = expected_freq * tolerance if tolerance is not None else expected_freq
        
        # Find indices where gaps occur
        gap_indices = time_diffs[time_diffs > max_gap].index
        
        return gap_indices


def normalize_data(df: pandas.DataFrame, norm_params: dict, columns: Optional[List[str]] = None) -> pandas.DataFrame:
        """
        Normalize data using fitted parameters.
        
        Applies z-score normalization: (x - mean) / std
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data to normalize.
        norm_params : dict
            Normalization parameters containing 'mean' and 'std' for each column.
        columns : list of str, optional
            Specific columns to normalize. If None, normalize all columns in the DataFrame.
            
        Returns
        -------
        pandas.DataFrame
            Normalized data. If normalize=False or normalization not fitted, 
            returns data unchanged.
            
        Notes
        -----
        - Only normalizes columns present in norm_params
        - Columns not in norm_params are left unchanged
        """

        df_norm = df.copy()

        if columns is None:
            columns = df.columns
        
        # Only normalize specified columns that exist in norm_params
        for col in columns:
            if col in df.columns and col in norm_params['mean'].index:
                df_norm[col] = (df[col] - norm_params['mean'][col]) / norm_params['std'][col]
        
        return df_norm
    
def denormalize_data(df: pandas.DataFrame, norm_params: dict, columns: Optional[List[str]] = None) -> pandas.DataFrame:
    """
    Denormalize data back to original scale.
    
    Reverses z-score normalization: x * std + mean
    
    Parameters
    ----------
    df : pandas.DataFrame
        Normalized data to denormalize.
    norm_params : dict
        Normalization parameters containing 'mean' and 'std' for each column.
    columns : list of str, optional
        Specific columns to denormalize. If None, denormalize all columns in the DataFrame.
        
    Returns
    -------
    pandas.DataFrame
        Denormalized data. If normalize=False or normalization not fitted,
        returns data unchanged.
        
    Notes
    -----
    - Only denormalizes columns present in norm_params
    - Columns not in norm_params are left unchanged
    
    """
        
    df_denorm = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col in norm_params['mean'].index:
            df_denorm[col] = (df[col] * norm_params['std'][col]) + norm_params['mean'][col]
    
    return df_denorm


def output_method(func):
    """
    Decorator to mark a method as a data output method that should be tested for data leaks.
    
    When a method is decorated with @output_method, it will be automatically registered
    in the DataLoader's data_output_methods list and included in leak detection tests.
    
    This decorator should be applied to any method that produces data that will be
    consumed by the model during training, validation, or prediction.
    
    Parameters
    ----------
    func : callable
        The method to mark as an output method.
        
    Returns
    -------
    callable
        The decorated method with _is_output_method attribute set to True.
        
    Examples
    --------
    >>> class CustomDataLoader(DataLoader):
    ...     @output_method
    ...     def prepare_training_data(self, training_set, validation_set):
    ...         return training_set, validation_set
    ...
    ...     @output_method
    ...     def prepare_prediction_data(self, df):
    ...         return df
    
    Notes
    -----
    - Methods are automatically discovered during __init__
    - All decorated methods will be tested for data leaks when test_all_output_methods() is called
    - The decorator does not modify the method's behavior, only marks it for registration
    """
    func._is_output_method = True
    return func


class DataLoader(ABC):
    """
    Base class for data loading, preprocessing, and formatting for time-series forecasting.
    
    Provides functionality for:
    - Loading data from CSV files or DataFrames
    - Preprocessing (normalization, resampling, feature engineering)
    - Splitting data into training/validation/testing sets
    - Creating samples with sliding windows
    - Detecting and handling time gaps
    - Feature indexing for mapping column names to integer positions
    - Data leak detection for model validation
    
    Parameters
    ----------
    update_freq : int
        Update rate of the forecast (in minutes)
    targets : str or list of str
        Target column name(s) to predict
    normalize : bool, default=True
        Whether to apply z-score normalization
    resample_freq : int, optional
        Resampling frequency in minutes. If None, no resampling is applied.
    train_split : float, default=0.7
        Proportion of data for training when using ratio-based splitting
    val_split : float, default=0.15
        Proportion of data for validation when using ratio-based splitting
    feature_generators : list of FeatureGenerator, optional
        Feature generators to apply for creating derived features.
        If None, no automatic feature generation is performed.
    """
    
    def __init__(
        self,
        update_freq: int,
        targets: Union[str, List[str]],
        normalize: bool = True,
        resample_freq: Optional[int] = 60,
        train_split: float = 0.7,
        val_split: float = 0.15,
        feature_generators: Optional[List['FeatureGenerator']] = None,
        additional_generated_features: Optional[List[str]] = None,
    ):
        self.update_freq = update_freq
        self.update_rate = update_freq // resample_freq  # in steps
        self.targets = targets
        self.normalize = normalize
        self.resample_freq = resample_freq
        self.train_split = train_split
        self.val_split = val_split
        
        # Feature generation setup - explicit only
        self.feature_generators = feature_generators or []
        self.additional_generated_features = additional_generated_features or []
        
        # Storage for data and splits
        self.df = None  # Full timeseries data
        self.data_periods = None  # Dictionary with 'training', 'validation', 'testing' periods
        
        # Sampled data storage
        self.samples = None  # List of all samples (dicts with data and metadata)
        self.accepted_samples = None  # List of accepted samples
        self.rejected_samples = None  # List of rejected samples
        self.sample_acceptance_rate = None  # Percentage of accepted samples
        
        # Split datasets - sampled data
        self.training_samples = None  # List of training samples
        self.validation_samples = None  # List of validation samples
        self.testing_samples = None  # List of testing samples
        
        # Split datasets - List of continuous timeseries
        self.training_segments = None  # List of training DataFrames
        self.validation_segments = None  # List of validation DataFrames
        self.testing_segments = None  # List of testing DataFrames
        
        # Continuous timeseries for each split
        self.training_timeseries = None  # Continuous timeseries for training period
        self.validation_timeseries = None  # Continuous timeseries for validation period
        self.testing_timeseries = None  # Continuous timeseries for testing period
        
        # Normalization parameters
        self.norm_params = {}  # Stores mean and std for each feature
        self.is_fitted = False  # Whether normalization params have been computed
        
        # Feature indexing - maps integer position to column name
        self.feature_index = None  # Dict mapping {0: 'col_name', 1: 'col_name', ...}
        self.index_to_feature = None  # Alias for feature_index (clearer naming)
        self.feature_to_index = None  # Reverse mapping {col_name: 0, ...}
        
        # Automatically discover methods decorated with @output_method
        self.data_output_methods = self._discover_output_methods()
    
    def _discover_output_methods(self) -> List[str]:
        """
        Discover all methods decorated with @output_method.
        
        This method scans the class instance for methods that have been marked
        with the @output_method decorator and returns their names as a list.
        
        Returns
        -------
        list of str
            Names of all methods marked as output methods.
            
        Notes
        -----
        - Only non-private methods (not starting with '_') are included
        - Methods must be callable and have the _is_output_method attribute
        - This is called automatically during __init__
        """
        methods = []
        for name in dir(self):
            # Skip private/protected methods
            if name.startswith('_'):
                continue
            try:
                attr = getattr(self, name)
                # Check if it's callable and marked as output method
                if callable(attr) and hasattr(attr, '_is_output_method'):
                    methods.append(name)
            except AttributeError:
                # Skip if attribute cannot be accessed
                continue
        
        if methods:
            logger.debug(
                f"{self.__class__.__name__} discovered {len(methods)} output methods: {methods}"
            )
        
        return methods
    
    def load_csv(
        self,
        filepath: str,
        index_col: Optional[Union[int, str]] = 0,
        parse_dates: bool = True,
        date_parser: Optional[callable] = None,
        **kwargs
    ) -> pandas.DataFrame:
        """
        Load data from a CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        index_col : int or str, optional
            Column to use as the row index. Default is 0 (first column).
        parse_dates : bool, default=True
            Whether to parse the index as dates.
        date_parser : callable, optional
            Function to parse date strings. If None, pandas default parser is used.
        **kwargs
            Additional keyword arguments passed to pandas.read_csv().
            
        Returns
        -------
        pandas.DataFrame
            Loaded dataframe with datetime index.
            
        Examples
        --------
        >>> df = DataLoader.load_csv('data.csv')
        >>> df = DataLoader.load_csv('data.csv', index_col='timestamp')
        >>> df = DataLoader.load_csv('data.csv', sep=';', decimal=',')
        """
        # Set default arguments
        read_kwargs = {
            'index_col': index_col,
            'parse_dates': parse_dates if index_col is not None else False,
        }
        
        if date_parser is not None:
            read_kwargs['date_parser'] = date_parser
        
        # Merge with user-provided kwargs
        read_kwargs.update(kwargs)
        
        # Load the CSV
        df = pandas.read_csv(filepath, **read_kwargs)
        
        if parse_dates and index_col is not None:
            if not isinstance(df.index, pandas.DatetimeIndex):
                try:
                    df.index = pandas.to_datetime(df.index)
                except Exception as e:
                    logger.warning(f"Could not convert index to datetime: {e}")
            
            # Always convert to UTC for consistency
            if isinstance(df.index, pandas.DatetimeIndex):
                if df.index.tz is None:
                    # Localize timezone-naive datetime to UTC
                    df.index = df.index.tz_localize('UTC')
                    logger.debug("Localized timezone-naive index to UTC")
                elif df.index.tz != pandas.DatetimeTZDtype.construct_from_string("datetime64[ns, UTC]").tz:
                    # Convert to UTC if it has a different timezone
                    df.index = df.index.tz_convert('UTC')
                    logger.debug(f"Converted index from {df.index.tz} to UTC")
        
        df = df.sort_index()
        
        logger.info(f"Loaded data from {filepath}: shape {df.shape}, "
                   f"date range {df.index[0]} to {df.index[-1]}")
        
        if hasattr(self, 'resample_freq') and self.resample_freq is not None and self.resample_freq > 0:
            original_len = len(df)
            df = df.resample(f'{self.resample_freq}min').mean()
            logger.info(f"Resampled from {original_len} to {len(df)} rows at {self.resample_freq}min frequency")
        
        return df
    
    def load_multiple_csv(
        self,
        filepaths: List[str],
        **kwargs
    ) -> Union[pandas.DataFrame, List[pandas.DataFrame]]:
        """
        Load data from multiple CSV files.
        
        Parameters
        ----------
        filepaths : list of str
            List of paths to CSV files.
        **kwargs
            Additional keyword arguments passed to load_csv().
            
        Returns
        -------
        pandas.DataFrame or list of pandas.DataFrame
            List of dataframes.
            
        Examples
        --------
        >>> df = DataLoader.load_multiple_csv(['data1.csv', 'data2.csv'])
        >>> df_list = DataLoader.load_multiple_csv(['data1.csv', 'data2.csv'])
        """
        dfs = []
        for filepath in filepaths:
            df = self.load_csv(filepath, **kwargs)
            dfs.append(df)
        
        return dfs
    
    def load_data(
        self,
        source: Union[str, List[str], Dict[str, Any], pandas.DataFrame],
        features: Optional[List[str]] = None,
        **kwargs
    ) -> pandas.DataFrame:
        """
        Load data from various sources and return a single concatenated DataFrame.
        
        This method handles different input types:
        - Single CSV filepath
        - Multiple CSV filepaths (concatenated row-wise or column-wise)
        - Dictionary with multiple data sources (with merge strategies)
        - Existing DataFrame
        
        Parameters
        ----------
        source : str, list of str, dict, or pandas.DataFrame
            Data source. Can be:
            - str: Path to a single CSV file
            - list of str: Paths to multiple CSV files (concatenated row-wise by default)
            - dict: Multiple data sources with merge strategies (see examples)
            - pandas.DataFrame: Already loaded dataframe (returned as-is)
        features : list of str, optional
            List of feature columns to keep. If provided:
            - Only these columns are retained
            - Rows with NaNs in these columns are dropped
            - Helps ensure clean data for modeling
        **kwargs
            Additional keyword arguments passed to load_csv() or load_multiple_csv().
            
        Returns
        -------
        pandas.DataFrame
            Single concatenated DataFrame with optional feature filtering.
            
        Examples
        --------
        >>> # Single CSV
        >>> df = loader.load_data('data.csv')
        
        >>> # Multiple CSVs (row-wise concatenation)
        >>> df = loader.load_data(['winter_2022.csv', 'winter_2023.csv'])
        
        >>> # Dictionary-based multi-source loading with merge strategies
        >>> df = loader.load_data({
        ...     'building': {
        ...         'paths': ['building_1.csv', 'building_2.csv'],
        ...         'concat_axis': 0  # Row-wise (default)
        ...     },
        ...     'weather': {
        ...         'paths': 'weather.csv',
        ...         'merge_on_index': True,
        ...         'merge_how': 'left'
        ...     }
        ... })
        
        >>> # With feature filtering
        >>> df = loader.load_data('data.csv', features=['t_in', 't_out', 'power'])
        """
        # Handle different source types
        if isinstance(source, pandas.DataFrame):
            logger.info(f"Using provided DataFrame: shape {source.shape}")
            df = source.copy()
            
            # Ensure index is in UTC
            if isinstance(df.index, pandas.DatetimeIndex):
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                    logger.debug("Localized timezone-naive DataFrame index to UTC")
                elif df.index.tz != pandas.DatetimeTZDtype.construct_from_string("datetime64[ns, UTC]").tz:
                    df.index = df.index.tz_convert('UTC')
                    logger.debug(f"Converted DataFrame index from {df.index.tz} to UTC")
            
        elif isinstance(source, str):
            df = self.load_csv(source, **kwargs)
            
        elif isinstance(source, list):
            dfs = self.load_multiple_csv(source, **kwargs)
            logger.info(f"Concatenating {len(dfs)} DataFrames row-wise")
            df = pandas.concat(dfs, axis=0, ignore_index=False)
            logger.info(f"Concatenated DataFrame shape: {df.shape}")
            
        elif isinstance(source, dict):
            # Dictionary-based multi-source loading
            df = self._load_from_dict_source(source, **kwargs)
            
        else:
            raise TypeError(
                f"Unsupported data source type: {type(source)}. "
                f"Expected str, list, dict, or pandas.DataFrame"
            )
        
        # Feature filtering and NaN handling
        if features is not None:
            original_shape = df.shape
            
            # Check which features exist in the data
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                logger.warning(
                    f"Requested features not found in data: {missing_features}. "
                    f"Available columns: {list(df.columns)}"
                )
            
            # Keep only existing requested features
            existing_features = [f for f in features if f in df.columns]
            if existing_features:
                df = df[existing_features]
                logger.info(f"Filtered to {len(existing_features)} features: {existing_features}")
            
            # Drop rows with NaNs in the requested features
            nan_counts_before = df.isna().sum()
            if nan_counts_before.any():
                logger.info(f"NaN counts before dropping:\n{nan_counts_before[nan_counts_before > 0]}")
            
            df = df.dropna()
            
            rows_dropped = original_shape[0] - df.shape[0]
            if rows_dropped > 0:
                logger.info(
                    f"Dropped {rows_dropped} rows with NaNs "
                    f"({rows_dropped/original_shape[0]*100:.1f}% of data). "
                    f"Final shape: {df.shape}"
                )
            else:
                logger.info("No rows with NaNs found - data is clean")
        
        return df
    
    def _load_from_dict_source(
        self,
        source_dict: Dict[str, Any],
        **kwargs
    ) -> pandas.DataFrame:
        """
        Load data from dictionary-based source specification.
        
        Parameters
        ----------
        source_dict : dict
            Dictionary where each key is a source name and value contains:
            - 'paths': str or list of str (file path(s))
            - 'concat_axis': int, optional (0=rows, 1=columns, default=0)
            - 'merge_on_index': bool, optional (if True, merge this source on index)
            - 'merge_how': str, optional ('left', 'right', 'outer', 'inner', default='left')
        **kwargs
            Additional arguments passed to load_csv()
            
        Returns
        -------
        pandas.DataFrame
            Merged/concatenated DataFrame from all sources
            
        Examples
        --------
        >>> source_dict = {
        ...     'building': {
        ...         'paths': ['building_1.csv', 'building_2.csv'],
        ...         'concat_axis': 0
        ...     },
        ...     'weather': {
        ...         'paths': 'weather.csv',
        ...         'merge_on_index': True,
        ...         'merge_how': 'left'
        ...     }
        ... }
        """
        logger.info(f"Loading data from {len(source_dict)} sources: {list(source_dict.keys())}")
        
        loaded_sources = {}
        
        # Load each source
        for source_name, source_config in source_dict.items():
            if not isinstance(source_config, dict) or 'paths' not in source_config:
                raise ValueError(
                    f"Source '{source_name}' must be a dict with 'paths' key. "
                    f"Got: {source_config}"
                )
            
            paths = source_config['paths']
            concat_axis = source_config.get('concat_axis', 0)
            
            # Load the data
            if isinstance(paths, str):
                df_source = self.load_csv(paths, **kwargs)
            elif isinstance(paths, list):
                dfs = self.load_multiple_csv(paths, **kwargs)
                if len(dfs) == 1:
                    df_source = dfs[0]
                else:
                    logger.info(f"  {source_name}: Concatenating {len(dfs)} files on axis={concat_axis}")
                    df_source = pandas.concat(dfs, axis=concat_axis, ignore_index=False)
            else:
                raise TypeError(
                    f"Source '{source_name}' paths must be str or list, got {type(paths)}"
                )
            
            logger.info(f"  {source_name}: Loaded shape {df_source.shape}")
            loaded_sources[source_name] = {
                'data': df_source,
                'config': source_config
            }
        
        # Merge all sources
        # Start with the first source, then merge others
        source_names = list(loaded_sources.keys())
        result_df = loaded_sources[source_names[0]]['data'].copy()
        logger.info(f"Starting with '{source_names[0]}' as base: shape {result_df.shape}")
        
        for source_name in source_names[1:]:
            source_info = loaded_sources[source_name]
            df_to_merge = source_info['data']
            config = source_info['config']
            
            merge_on_index = config.get('merge_on_index', True)
            merge_how = config.get('merge_how', 'left')
            
            if merge_on_index:
                logger.info(
                    f"  Merging '{source_name}' on index with how='{merge_how}': "
                    f"shape {df_to_merge.shape}"
                )
                result_df = result_df.join(df_to_merge, how=merge_how, rsuffix=f'_{source_name}')
            else:
                # Could add support for merge on specific columns in the future
                logger.warning(
                    f"  Source '{source_name}' has merge_on_index=False, "
                    f"but column-based merge not yet implemented. Falling back to index merge."
                )
                result_df = result_df.join(df_to_merge, how=merge_how, rsuffix=f'_{source_name}')
            
            logger.info(f"  Result after merging '{source_name}': shape {result_df.shape}")
        
        nan_counts = result_df.isna().sum()
        if nan_counts.any():
            logger.warning(
                f"NaN values after merging sources:\n{nan_counts[nan_counts > 0]}"
            )
        
        return result_df
           
    def create_samples(
        self,
        period: str,
        features: List[str],
        targets: Union[str, List[str]],
        input_window: Tuple[int, int],
        output_format: str = 'list',
        stride: int = 1,
        **kwargs
    ) -> Union[List[Dict], List[pandas.DataFrame], numpy.ndarray, 'torch.Tensor']:
        """
        Create samples dynamically from continuous timeseries data.
        
        Generates sliding window samples without caching. Samples are filtered to
        exclude those with NaN values or time gaps.
        
        Parameters
        ----------
        period : str
            Data period: 'training', 'validation', or 'testing'
        features : list of str
            Feature column names to include
        targets : str or list of str
            Target column name(s) to predict
        input_window : tuple of (int, int)
            Window boundaries as (start_offset, end_offset) in timesteps
        output_format : str, default='list'
            Output format:
            - 'list': List of dicts with 'data', 'start_time', 'end_time' keys
            - 'dataframes': List of DataFrames (one per sample)
            - 'array': Numpy array of shape (n_samples, window_size, n_features)
            - 'tensor': PyTorch tensor (requires torch)
        stride : int, default=1
            Stride between consecutive samples
        **kwargs
            Additional keyword arguments (reserved for future use)
            
        Returns
        -------
        samples : list, numpy.ndarray, or torch.Tensor
            Samples in requested format (only accepted samples)
            
        Raises
        ------
        ValueError
            If period is invalid or no data available for that period
        ImportError
            If output_format='tensor' but torch is not installed
        """
        # Validate period and get corresponding timeseries
        valid_periods = {'training', 'validation', 'testing'}
        if period not in valid_periods:
            raise ValueError(f"period must be one of {valid_periods}, got '{period}'")
        
        # Get the appropriate timeseries
        timeseries_map = {
            'training': self.training_timeseries,
            'validation': self.validation_timeseries,
            'testing': self.testing_timeseries
        }
        
        df = timeseries_map[period]
        
        # Use the standalone function with instance-specific parameters
        return create_samples(
            df=df,
            features=features,
            targets=targets,
            input_window=input_window,
            output_format=output_format,
            stride=stride,
            normalize=self.normalize and self.is_fitted,
            norm_params=self.norm_params if self.is_fitted else None,
            resample_freq=self.resample_freq,
            period_name=period,
            **kwargs
        )
    
    def is_sample_valid(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a sample is valid for training/evaluation.
        
        This method checks for:
        - Missing values (NaN/None) in the sample data
        - Sufficient data length
        - Any custom validation criteria (override in subclasses)
        
        Since samples are filtered to only contain required features during creation,
        this method now simply checks all columns present in the sample.
        
        Parameters
        ----------
        sample : dict
            Sample dictionary with 'data' key containing DataFrame.
            
        Returns
        -------
        tuple
            (is_valid, reason) - Boolean indicating validity and string reason if invalid.
            
        Examples
        --------
        >>> is_valid, reason = loader.is_sample_valid(sample)
        >>> if not is_valid:
        ...     print(f"Sample rejected: {reason}")
        
        Notes
        -----
        Override _custom_sample_validation() in subclasses to implement custom logic.
        """
        data = sample['data']
        
        # Custom validation first
        is_valid, reason = self._custom_sample_validation(sample)
        if not is_valid:
            return False, reason
        
        # Check for missing values in all columns (since sample only has required features)
        if data.isnull().any().any():
            # Find which columns have NaN
            nan_cols = data.columns[data.isnull().any()].tolist()
            return False, f"Contains NaN values in columns: {nan_cols}"
        
        # Length validation is already handled by create_samples()
        # which computes valid indices based on feature windows
        
        return True, "Valid"
    
    def _custom_sample_validation(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Custom sample validation logic to be overridden by subclasses.
        
        Parameters
        ----------
        sample : dict
            Sample dictionary.
            
        Returns
        -------
        tuple
            (is_valid, reason) - Boolean and reason string.
        """
        return True, "Valid"
    
    def filter_samples(
        self,
        samples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Filter samples into accepted and rejected based on validity.
        
        Parameters
        ----------
        samples : list of dict
            List of all samples to filter.
            
        Returns
        -------
        tuple
            (accepted_samples, rejected_samples, acceptance_rate)
            - accepted_samples: List of valid samples
            - rejected_samples: List of invalid samples with rejection reasons
            - acceptance_rate: Percentage of accepted samples (0-100)
            
        Examples
        --------
        >>> accepted, rejected, rate = loader.filter_samples(samples)
        >>> print(f"Acceptance rate: {rate:.1f}%")
        >>> print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")
        """
        accepted = []
        rejected = []
        
        for sample in samples:
            is_valid, reason = self.is_sample_valid(sample)
            if is_valid:
                sample['accepted'] = True
                sample['rejection_reason'] = None
                accepted.append(sample)
            else:
                sample['accepted'] = False
                sample['rejection_reason'] = reason
                rejected.append(sample)
        
        total = len(samples)
        acceptance_rate = (len(accepted) / total * 100) if total > 0 else 0.0
        
        logger.info(
            f"Sample filtering:\n"
            f"  Total samples: {total}\n"
            f"  Accepted samples: {len(accepted)} ({acceptance_rate:.1f}%)\n"
            f"  Rejected samples: {len(rejected)} ({100-acceptance_rate:.1f}%)"
        )
        
        return accepted, rejected, acceptance_rate
    
    def _normalize_data_periods(
        self,
        data_periods: Dict[str, Union[Tuple, List[Tuple[pandas.Timestamp, pandas.Timestamp]]]]
    ) -> Dict[str, List[Tuple[pandas.Timestamp, pandas.Timestamp]]]:
        """
        Normalize data_periods to ensure all values are lists of timestamp tuples.
        
        Accepts both formats:
        - Single tuple: ('2023-01-01', '2023-06-30')
        - List of tuples: [('2023-01-01', '2023-03-31'), ('2023-04-01', '2023-06-30')]
        
        Parameters
        ----------
        data_periods : dict
            Dictionary with keys like 'training', 'validation', 'testing'.
            Values can be either:
            - A single tuple of (start, end)
            - A list of (start, end) tuples
            
        Returns
        -------
        dict
            Normalized dictionary where all values are lists of (Timestamp, Timestamp) tuples.
            
        Examples
        --------
        >>> # Single tuple format
        >>> periods = {'training': ('2023-01-01', '2023-06-30')}
        >>> normalized = loader._normalize_data_periods(periods)
        >>> # Returns: {'training': [(Timestamp('2023-01-01'), Timestamp('2023-06-30'))]}
        
        >>> # List of tuples format (already normalized)
        >>> periods = {'training': [('2023-01-01', '2023-03-31'), ('2023-04-01', '2023-06-30')]}
        >>> normalized = loader._normalize_data_periods(periods)
        >>> # Returns as-is with Timestamps
        """
        normalized = {}
        
        for key, period in data_periods.items():
            if period is None or (isinstance(period, list) and len(period) == 0):
                # Empty or None - set to empty list
                normalized[key] = []
            elif isinstance(period, tuple) and len(period) == 2:
                # Single tuple: convert to list with one element
                start, end = period
                # Convert to Timestamp if not already
                if not isinstance(start, pandas.Timestamp):
                    start = pandas.Timestamp(start, tz='UTC')
                if not isinstance(end, pandas.Timestamp):
                    end = pandas.Timestamp(end, tz='UTC')
                normalized[key] = [(start, end)]
            elif isinstance(period, list):
                # List of tuples: ensure all are Timestamps
                normalized_list = []
                for item in period:
                    if not isinstance(item, tuple) or len(item) != 2:
                        raise ValueError(
                            f"Invalid period format for '{key}': each item must be a (start, end) tuple. "
                            f"Got: {item}"
                        )
                    start, end = item
                    # Convert to Timestamp if not already
                    if not isinstance(start, pandas.Timestamp):
                        start = pandas.Timestamp(start, tz='UTC')
                    if not isinstance(end, pandas.Timestamp):
                        end = pandas.Timestamp(end, tz='UTC')
                    normalized_list.append((start, end))
                normalized[key] = normalized_list
            else:
                raise ValueError(
                    f"Invalid format for '{key}' period. Must be either:\n"
                    f"  - A single tuple: ('2023-01-01', '2023-06-30')\n"
                    f"  - A list of tuples: [('2023-01-01', '2023-03-31'), ('2023-04-01', '2023-06-30')]\n"
                    f"Got: {type(period)} - {period}"
                )
        
        return normalized
    
    def _apply_feature_generators(
        self,
        df: pandas.DataFrame,
        required_features: Optional[List[str]] = None
    ) -> pandas.DataFrame:
        """
        Apply feature generators to add missing features.
        
        This method applies registered feature generators to create derived features
        that don't exist in the raw data. Common use cases:
        - Generate temporal features (hour, weekday) from DatetimeIndex
        - Create cyclical encodings (hour_sin, hour_cos) for periodic features
        - Add lagged features or physics-based calculations
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input data.
        required_features : list of str, optional
            Features that must be present in the output. If None, generates
            all possible features from all generators. If provided, only
            generates missing required features.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with generated features added.
            
        Examples
        --------
        >>> # Generate all possible features
        >>> df = loader._apply_feature_generators(df)
        
        >>> # Only generate specific missing features
        >>> required = ['hour_sin', 'hour_cos', 't_out']
        >>> df = loader._apply_feature_generators(df, required)
        
        Notes
        -----
        - Original DataFrame is not modified (returns a copy)
        - Logs warnings for features that cannot be generated
        - If required_features is provided and some cannot be generated,
          logs errors but continues (downstream code will catch missing columns)
        """
        if not self.feature_generators:
            return df
        
        df_copy = df.copy()
        
        if required_features is None:
            # Generate all possible features from all generators
            logger.info("Applying all feature generators (no specific requirements)")
            for gen in self.feature_generators:
                try:
                    df_copy = gen.generate(df_copy)
                except Exception as e:
                    logger.warning(
                        f"Generator {gen.__class__.__name__} failed: {e}"
                    )
        else:
            # Only generate missing required features
            missing = [f for f in required_features if f not in df_copy.columns]
            
            if not missing:
                logger.debug("All required features already present in data")
                return df_copy
            
            logger.info(
                f"Attempting to generate {len(missing)} missing features: "
                f"{missing}"
            )
            
            # Run ALL generators in sequence to handle dependencies
            # E.g., TemporalFeatureGenerator creates 'hour', then CyclicalFeatureGenerator creates 'hour_sin'
            # Don't check can_generate() per feature - generators may create intermediate dependencies
            generated_any = False
            for gen in self.feature_generators:
                try:
                    before_cols = set(df_copy.columns)
                    df_copy = gen.generate(df_copy)
                    after_cols = set(df_copy.columns)
                    new_cols = after_cols - before_cols
                    
                    if new_cols:
                        generated_any = True
                        # Log which required features were generated
                        generated_required = [f for f in missing if f in new_cols]
                        if generated_required:
                            logger.info(
                                f"{gen.__class__.__name__} generated required features: "
                                f"{generated_required}"
                            )
                        # Also log intermediate features (dependencies)
                        intermediate_cols = new_cols - set(generated_required)
                        if intermediate_cols:
                            logger.debug(
                                f"{gen.__class__.__name__} also generated intermediate features: "
                                f"{intermediate_cols}"
                            )
                        # Update missing list
                        missing = [f for f in missing if f not in df_copy.columns]
                        
                except Exception as e:
                    logger.warning(
                        f"Failed to run {gen.__class__.__name__}: {e}"
                    )
            
            # Check if any required features are still missing
            still_missing = [f for f in required_features if f not in df_copy.columns]
            if still_missing:
                logger.error(
                    f"Could not generate {len(still_missing)} required features: "
                    f"{still_missing}. No generator available or all generators failed."
                )
            
            if generated_any:
                final_missing = [f for f in required_features if f not in df_copy.columns]
                if final_missing:
                    logger.warning(
                        f"After feature generation, still missing {len(final_missing)} features: "
                        f"{final_missing}"
                    )
                else:
                    logger.info("Successfully generated all missing required features")
        
        return df_copy
    
    def _get_required_features(self) -> Optional[List[str]]:
        """
        Get list of required features for this dataloader.
        
        Override this in subclasses to specify which features are needed.
        For example, TorchLKFDataLoader might return control_features + 
        disturbance_features + observation_features.
        
        Returns
        -------
        list of str or None
            List of required feature names, or None if not applicable.
            
        Examples
        --------
        >>> class MyDataLoader(DataLoader):
        ...     def __init__(self, control_features, disturbance_features, **kwargs):
        ...         self.control_features = control_features
        ...         self.disturbance_features = disturbance_features
        ...         super().__init__(**kwargs)
        ...     
        ...     def _get_required_features(self):
        ...         return self.control_features + self.disturbance_features
        """
        return None
    
    def load_and_split_data(
        self,
        source: Union[str, List[str], Dict[str, Any], pandas.DataFrame],
        data_periods: Optional[Dict[str, List[Tuple[pandas.Timestamp, pandas.Timestamp]]]] = None,
        split_by_ratio: bool = False,
        features: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[List[pandas.DataFrame], List[pandas.DataFrame], List[pandas.DataFrame]]:
        """
        Load data from source, split into training/validation/testing periods, and fit normalization.
        
        Splits data by specific date periods or ratio-based splitting. Samples are NOT created
        here - they are generated on-demand via create_samples() with model-specific parameters.
        
        Parameters
        ----------
        source : str, list of str, dict, or pandas.DataFrame
            Data source to load. Can be:
            - str: Path to a single CSV file
            - list of str: Paths to multiple CSV files (concatenated row-wise)
            - dict: Multiple data sources with merge strategies (see load_data() for format)
            - pandas.DataFrame: Already loaded dataframe
        data_periods : dict, optional
            Dictionary with keys 'training', 'validation', 'testing' and values as
            lists of (start, end) timestamp tuples. If provided, split_by_ratio is ignored.
        split_by_ratio : bool, default=False
            If True and data_periods is None, use ratio-based splitting
        features : list of str, optional
            List of feature columns to keep. If provided:
            - Only these columns are retained
            - Rows with NaNs in these columns are dropped during loading
        train_split : float, default=0.7
            Training proportion when using ratio-based splitting
        val_split : float, default=0.15
            Validation proportion when using ratio-based splitting
        **kwargs
            Additional arguments passed to load_data()
            
        Returns
        -------
        tuple of (list of DataFrames, list of DataFrames, list of DataFrames)
            (training_segments, validation_segments, testing_segments)
            
        Examples
        --------
        >>> # Simple CSV loading
        >>> loader.load_and_split_data('data.csv', data_periods={...})
        
        >>> # Multiple CSVs with row-wise concatenation
        >>> loader.load_and_split_data(['winter_2022.csv', 'winter_2023.csv'], data_periods={...})
        
        >>> # Dictionary-based multi-source loading (building + weather data)
        >>> loader.load_and_split_data(
        ...     source={
        ...         'building': {'paths': ['building_1.csv', 'building_2.csv']},
        ...         'weather': {'paths': 'weather.csv', 'merge_on_index': True, 'merge_how': 'left'}
        ...     },
        ...     features=['t_in', 't_out', 'power'],
        ...     data_periods={...}
        ... )
        """
        # Load data - now returns a single DataFrame with optional feature filtering and NaN handling
        self.df = self.load_data(source, features=features, **kwargs)
        
        # Apply feature generators AFTER loading but BEFORE splitting/normalization
        # This ensures generated features are available in all splits
        if self.feature_generators:
            required_features = self._get_required_features()
            logger.info("Applying feature generators to loaded data...")
            logger.info(f"Columns BEFORE feature generation: {list(self.df.columns)}")
            self.df = self._apply_feature_generators(self.df, required_features)
            logger.info(f"Columns AFTER feature generation: {list(self.df.columns)}")
            # Only filter columns if features list was provided
            if features is not None:
                self.df = self.df.loc[:, [f for f in features]]
                logger.info(f"Columns AFTER feature filtering: {list(self.df.columns)}")

        if data_periods is not None:
            normalized_periods = self._normalize_data_periods(data_periods)
            self.data_periods = normalized_periods
            
            # Use self.df (with generated features) for splitting
            self.training_timeseries, self.validation_timeseries, self.testing_timeseries = \
                self._split_with_periods(self.df, normalized_periods)
            
            logger.info(f"Training split columns: {list(self.training_timeseries.columns) if self.training_timeseries is not None else 'None'}")
            logger.info(f"Validation split columns: {list(self.validation_timeseries.columns) if self.validation_timeseries is not None else 'None'}")
            
        elif split_by_ratio:
            # Use self.df (with generated features) for splitting
            self.training_timeseries, self.validation_timeseries, self.testing_timeseries = \
                self._split_by_ratio(self.df, self.train_split, self.val_split)
            
        else:
            raise ValueError(
                "Either 'data_periods' must be provided or 'split_by_ratio' must be True"
            )
                
        if self.normalize:
            dfs_for_norm = []
            if self.training_timeseries is not None:
                dfs_for_norm.append(self.training_timeseries)
            if self.validation_timeseries is not None:
                dfs_for_norm.append(self.validation_timeseries)
            
            if dfs_for_norm:
                combined_df = pandas.concat(dfs_for_norm, axis=0)
                self.fit_normalization(combined_df)
                logger.info(f"Fitted normalization parameters on training + validation data ({len(combined_df)} samples)")
        
        self._create_segments_from_timeseries()
        
        self._create_feature_index()
        self._create_feature_index()  ## TODO: Why is this line duplicated?

    def set_split_data(
        self,
        training_set,
        validation_set=None,
        testing_set=None,
    ):
        """
        Set pre-split data directly, bypassing CSV loading.

        Each argument can be a single ``pd.DataFrame`` or a **list of
        DataFrames** representing multiple disjoint periods (e.g. the output
        of ``create_dataset()`` in ``amp.utils``).  Lists are concatenated
        into a single time-series and also stored as segments so that
        ``create_samples()`` rejects any sliding window that spans a gap.

        After calling this method the DataLoader is ready to produce PyTorch
        DataLoaders via ``get_dataloaders()``.

        Parameters
        ----------
        training_set : pd.DataFrame or list of pd.DataFrame
            Training data.
        validation_set : pd.DataFrame or list of pd.DataFrame, optional
            Validation data.
        testing_set : pd.DataFrame or list of pd.DataFrame, optional
            Testing data.
        """
        def _to_parts(data):
            """Return (timeseries_df, segments_list) for single df or list."""
            if data is None:
                return None, []
            if isinstance(data, list):
                if len(data) == 0:
                    return None, []
                ts = pandas.concat(data).sort_index()
                ts = ts[~ts.index.duplicated(keep='first')]
                return ts, list(data)
            return data, [data]

        self.training_timeseries, self.training_segments = _to_parts(training_set)
        self.validation_timeseries, self.validation_segments = _to_parts(validation_set)
        self.testing_timeseries, self.testing_segments = _to_parts(testing_set)

        # Build a combined df for feature index creation
        dfs = [ts for ts in [
            self.training_timeseries,
            self.validation_timeseries,
            self.testing_timeseries,
        ] if ts is not None]
        if dfs:
            self.df = pandas.concat(dfs).sort_index()
            self.df = self.df[~self.df.index.duplicated(keep='first')]

        self._create_feature_index()

        if self.normalize:
            norm_dfs = [ts for ts in [
                self.training_timeseries,
                self.validation_timeseries,
            ] if ts is not None]
            if norm_dfs:
                combined_df = pandas.concat(norm_dfs, axis=0)
                self.fit_normalization(combined_df)
        else:
            self.is_fitted = True

    def fit_normalization(self, df: pandas.DataFrame):
        """
        Fit normalization parameters (mean and std) on the provided data.
        
        Computes and stores mean and standard deviation for each numeric column
        in the dataframe. These parameters are later used to normalize/denormalize data.
        
        Typically called with combined training + validation data to get better
        statistical estimates that represent the full distribution seen during training.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data to compute statistics from (typically training + validation combined).
            
        Notes
        -----
        - Only numeric columns are normalized
        - Standard deviation is replaced with 1 if it's 0 to avoid division by zero
        - Sets self.is_fitted to True after computing parameters
        - In load_and_split_data(), this is called with training + validation data combined
        
        Examples
        --------
        >>> loader.fit_normalization(training_df)
        >>> print(loader.norm_params['mean'])
        >>> print(loader.norm_params['std'])
        """
        self.norm_params = {
            'mean': df.mean(),
            'std': df.std().replace(0, 1)  # Avoid division by zero
        }
        self.is_fitted = True
        logger.debug(f"Normalization parameters fitted on {len(df)} samples")
    
        self.is_fitted = True
        logger.debug(f"Normalization parameters fitted on {len(df)} samples")
    
    def _find_gap_indices(self, df: pandas.DataFrame, tolerance: float = None) -> pandas.Index:
        """
        Find indices where time gaps occur in a DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with DatetimeIndex to analyze
        tolerance : float, optional
            Tolerance factor for expected frequency (e.g., 1.1 = 10% tolerance).
            If None, uses exact frequency matching with no tolerance.
            Default is None.
            
        Returns
        -------
        pandas.Index
            Index values where gaps occur (empty if no gaps)
            
        Examples
        --------
        >>> gap_indices = loader._find_gap_indices(df)
        >>> print(f"Found {len(gap_indices)} gaps")
        >>> gap_indices_strict = loader._find_gap_indices(df, tolerance=None)
        """
        return find_gap_indices(df, tolerance=tolerance, freq=self.resample_freq)
    
    def _create_segments_from_timeseries(self):
        """
        Split timeseries into continuous segments for training, validation, and testing.
        
        Takes the `*_timeseries` attributes (single DataFrames per period) and splits them
        into lists of continuous segments by detecting time gaps. Each segment is a continuous
        DataFrame without time gaps.
        
        This method is called automatically by load_and_split_data() after data splitting
        and normalization fitting.
        
        Sets the following attributes:
        - self.training_segments: List of continuous training DataFrames
        - self.validation_segments: List of continuous validation DataFrames
        - self.testing_segments: List of continuous testing DataFrames
        
        Notes
        -----
        - Detects gaps based on self.resample_freq with 0% tolerance
        - Empty or None timeseries result in empty segment lists
        - Single-row timeseries become single-element segment lists
        """
        self.training_segments = self._split_timeseries_into_segments(self.training_timeseries)
        self.validation_segments = self._split_timeseries_into_segments(self.validation_timeseries)
        self.testing_segments = self._split_timeseries_into_segments(self.testing_timeseries)
        
        logger.info(
            f"Created segments: training={len(self.training_segments)}, "
            f"validation={len(self.validation_segments)}, "
            f"testing={len(self.testing_segments)}"
        )
    
    def _split_timeseries_into_segments(self, df: pandas.DataFrame) -> List[pandas.DataFrame]:
        """
        Split a single timeseries DataFrame into continuous segments.
        
        Identifies gaps in the time series (where time difference exceeds expected
        frequency) and splits the DataFrame at those gaps. Returns a list of
        continuous DataFrames without time gaps.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with DatetimeIndex to split. Can be None.
            
        Returns
        -------
        list of pandas.DataFrame
            List of continuous time segments. Empty list if df is None or empty.
            
        Examples
        --------
        >>> # DataFrame with gap between 2023-01-15 and 2023-02-01
        >>> segments = loader._split_timeseries_into_segments(df)
        >>> # Returns [df1 (Jan 1-15), df2 (Feb 1-28)]
        """
        if df is None or len(df) == 0:
            return []
        
        if len(df) == 1:
            return [df]
        
        gap_indices = self._find_gap_indices(df, tolerance=1.0)  # no tolerance
        
        if len(gap_indices) == 0:
            return [df]
        
        segments = []
        start_idx = 0
        
        for gap_idx in gap_indices:
            gap_pos = df.index.get_loc(gap_idx)
            segment = df.iloc[start_idx:gap_pos]
            if len(segment) > 0:
                segments.append(segment)
            start_idx = gap_pos
        
        final_segment = df.iloc[start_idx:]
        if len(final_segment) > 0:
            segments.append(final_segment)
        
        return segments
    
    def _split_with_periods(
        self,
        df: pandas.DataFrame,
        data_periods: Dict[str, List[Tuple[pandas.Timestamp, pandas.Timestamp]]]
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        """
        Split concatenated data by specific time periods.
        
        Extracts training, validation, and testing data based on provided
        timestamp ranges. Multiple periods per split are concatenated.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Concatenated data from all segments
        data_periods : dict
            Dictionary with 'training', 'validation', 'testing' keys,
            each containing list of (start, end) timestamp tuples
            
        Returns
        -------
        tuple
            (training_df, validation_df, testing_df) - concatenated DataFrames for each split
            
        Examples
        --------
        >>> periods = {
        ...     'training': [(pandas.Timestamp('2023-01-01'), pandas.Timestamp('2023-06-30'))],
        ...     'validation': [(pandas.Timestamp('2023-07-01'), pandas.Timestamp('2023-09-30'))],
        ...     'testing': [(pandas.Timestamp('2023-10-01'), pandas.Timestamp('2023-12-31'))]
        ... }
        >>> train_df, val_df, test_df = loader._split_with_periods(df, periods)
        """
        # Extract periods for each split
        training_df = self._extract_periods(df, data_periods.get('training', []))
        validation_df = self._extract_periods(df, data_periods.get('validation', []))
        testing_df = self._extract_periods(df, data_periods.get('testing', []))
        
        logger.info(
            f"Split with periods: training={len(training_df)}, "
            f"validation={len(validation_df)}, testing={len(testing_df)} rows"
        )
        
        return training_df, validation_df, testing_df
    
    def _split_by_ratio(
        self,
        df: pandas.DataFrame,
        train_split: float,
        val_split: float
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        """
        Split data by ratio into training, validation, and testing sets.
        
        Splits the concatenated data based on proportions. Test split is
        automatically calculated as remainder (1 - train_split - val_split).
        Splits by row count, treating all data as one continuous sequence
        (gaps in time index are maintained but don't affect split points).
        
        Parameters
        ----------
        df : pandas.DataFrame
            Concatenated data from all segments
        train_split : float
            Proportion for training (e.g., 0.7 for 70%)
        val_split : float
            Proportion for validation (e.g., 0.15 for 15%)
            
        Returns
        -------
        tuple
            (training_df, validation_df, testing_df) - DataFrames for each split
            
        Examples
        --------
        >>> train_df, val_df, test_df = loader._split_by_ratio(df, 0.7, 0.15)
        >>> # Results in 70% train, 15% val, 15% test by row count
        """
        total_len = len(df)
        train_end_idx = int(total_len * train_split)
        val_end_idx = int(total_len * (train_split + val_split))
        
        training_df = df.iloc[:train_end_idx]
        validation_df = df.iloc[train_end_idx:val_end_idx]
        testing_df = df.iloc[val_end_idx:]
        
        # Store computed periods for reference
        self.data_periods = {
            'training': [(training_df.index[0], training_df.index[-1])] if len(training_df) > 0 else [],
            'validation': [(validation_df.index[0], validation_df.index[-1])] if len(validation_df) > 0 else [],
            'testing': [(testing_df.index[0], testing_df.index[-1])] if len(testing_df) > 0 else []
        }
        
        logger.info(
            f"Split by ratio ({train_split:.0%}/{val_split:.0%}/{1-train_split-val_split:.0%}): "
            f"training={len(training_df)}, validation={len(validation_df)}, testing={len(testing_df)} rows"
        )
        
        return training_df, validation_df, testing_df
    
    def _extract_periods(
        self,
        df: pandas.DataFrame,
        periods: List[Tuple[pandas.Timestamp, pandas.Timestamp]]
    ) -> pandas.DataFrame:
        """
        Extract and concatenate data for multiple time periods.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Source DataFrame with datetime index
        periods : list of tuple
            List of (start, end) timestamp tuples
            
        Returns
        -------
        pandas.DataFrame
            Concatenated DataFrame containing data from all specified periods.
            Returns empty DataFrame if no periods provided or no data found.
            
        Examples
        --------
        >>> periods = [
        ...     (pandas.Timestamp('2023-01-01'), pandas.Timestamp('2023-03-31')),
        ...     (pandas.Timestamp('2023-07-01'), pandas.Timestamp('2023-09-30'))
        ... ]
        >>> extracted = loader._extract_periods(df, periods)
        """
        if not periods:
            return pandas.DataFrame()
        
        period_dfs = []
        for start, end in periods:
            mask = (df.index >= start) & (df.index <= end)
            segment = df[mask]
            if len(segment) > 0:
                period_dfs.append(segment)
            else:
                logger.warning(f"No data found for period: {start} to {end}")
        
        if len(period_dfs) == 0:
            logger.warning(f"No data found for any of {len(periods)} periods")
            return pandas.DataFrame()
        
        return pandas.concat(period_dfs, axis=0)
    
    def normalize_data(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Normalize data using fitted parameters.
        
        Applies z-score normalization: (x - mean) / std
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data to normalize.
            
        Returns
        -------
        pandas.DataFrame
            Normalized data. If normalize=False or normalization not fitted, 
            returns data unchanged.
            
        Notes
        -----
        - Only normalizes columns present in norm_params
        - Columns not in norm_params are left unchanged
        - If self.normalize is False or self.is_fitted is False, returns original data
        
        Examples
        --------
        >>> loader.fit_normalization(train_df)
        >>> normalized_val = loader.normalize_data(val_df)
        """
        if not self.normalize or not self.is_fitted:
            return df
        
        return normalize_data(df, self.norm_params)
    
    def denormalize_data(self, df: pandas.DataFrame, columns: Optional[List[str]] = None) -> pandas.DataFrame:
        """
        Denormalize data back to original scale.
        
        Reverses z-score normalization: x * std + mean
        
        Parameters
        ----------
        df : pandas.DataFrame
            Normalized data to denormalize.
        columns : list of str, optional
            Specific columns to denormalize. If None, denormalize all columns in the DataFrame.
            
        Returns
        -------
        pandas.DataFrame
            Denormalized data. If normalize=False or normalization not fitted,
            returns data unchanged.
            
        Notes
        -----
        - Only denormalizes columns present in norm_params
        - Columns not in norm_params are left unchanged
        - If self.normalize is False or self.is_fitted is False, returns original data
        
        Examples
        --------
        >>> predictions_original_scale = loader.denormalize_data(predictions_normalized, columns=['temperature'])
        """
        if not self.normalize or not self.is_fitted:
            return df
        
        return denormalize_data(df, self.norm_params, columns=columns)
    
    def _create_feature_index(self):
        """
        Create feature index mapping from integer positions to column names.
        
        This method creates bidirectional mappings between column positions and names
        for all datasets (training_set, validation_set, testing_set, samples, etc.).
        
        The mappings are:
        - self.feature_index / self.index_to_feature: {0: 'col_name', 1: 'col_name', ...}
        - self.feature_to_index: {'col_name': 0, 'col_name': 1, ...}
        
        Since all splits should have the same columns (just different time periods),
        we use the columns from self.df or the first available dataset.
        
        Called automatically at the end of load_and_split_data().
        
        Notes
        -----
        - All datasets (training_set, validation_set, testing_set) have the same columns
        - The integer indices correspond to DataFrame column positions
        - This allows models to reference features by name even when working with arrays/tensors
        
        Examples
        --------
        >>> loader.load_and_split_data('data.csv', data_periods={...})
        >>> print(loader.feature_index)
        {0: 'timestamp', 1: 'temp_outdoor', 2: 'temp_indoor', 3: 'heating_power'}
        >>> print(loader.feature_to_index['temp_indoor'])
        2
        """
        # Get columns from the first available data source
        columns = self.df.columns.tolist()
        
        if columns is None:
            logger.warning("No data available to create feature index")
            self.feature_index = {}
            self.index_to_feature = {}
            self.feature_to_index = {}
            return
        
        # Create forward mapping (index -> feature name)
        self.feature_index = {i: col for i, col in enumerate(columns)}
        self.index_to_feature = self.feature_index  # Alias for clarity
        
        # Create reverse mapping (feature name -> index)
        self.feature_to_index = {col: i for i, col in enumerate(columns)}
        
        logger.info(f"Created feature index with {len(columns)} columns")
        logger.debug(f"Feature index: {self.feature_index}")
    
    def get_feature_index(self, feature_name: str) -> Optional[int]:
        """
        Get the integer index for a feature name.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature/column
            
        Returns
        -------
        int or None
            Integer index of the feature, or None if not found
            
        Examples
        --------
        >>> idx = loader.get_feature_index('temp_indoor')
        >>> print(idx)
        2
        """
        if self.feature_to_index is None:
            logger.warning("Feature index not created yet. Call load_and_split_data() first.")
            return None
        return self.feature_to_index.get(feature_name)
    
    def get_feature_name(self, index: int) -> Optional[str]:
        """
        Get the feature name for an integer index.
        
        Parameters
        ----------
        index : int
            Integer index of the feature
            
        Returns
        -------
        str or None
            Name of the feature/column, or None if index out of range
            
        Examples
        --------
        >>> name = loader.get_feature_name(2)
        >>> print(name)
        'temp_indoor'
        """
        if self.feature_index is None:
            logger.warning("Feature index not created yet. Call load_and_split_data() first.")
            return None
        return self.feature_index.get(index)
    
    def get_feature_indices(self, feature_names: List[str]) -> Dict[str, int]:
        """
        Get integer indices for a list of feature names.
        
        Useful when a model uses only a subset of available features.
        
        Parameters
        ----------
        feature_names : list of str
            List of feature/column names
            
        Returns
        -------
        dict
            Dictionary mapping feature names to their indices {feature_name: index}
            Features not found in the index are omitted with a warning.
            
        Examples
        --------
        >>> indices = loader.get_feature_indices(['temp_indoor', 'temp_outdoor'])
        >>> print(indices)
        {'temp_indoor': 2, 'temp_outdoor': 1}
        """
        if self.feature_to_index is None:
            logger.warning("Feature index not created yet. Call load_and_split_data() first.")
            return {}
        
        indices = {}
        missing = []
        
        for name in feature_names:
            idx = self.feature_to_index.get(name)
            if idx is not None:
                indices[name] = idx
            else:
                missing.append(name)
        
        if missing:
            logger.warning(f"Features not found in index: {missing}")
        
        return indices
    
    def _split_samples_by_period(
        self,
        samples: List[Dict[str, Any]],
        periods: List[Tuple[pandas.Timestamp, pandas.Timestamp]]
    ) -> List[Dict[str, Any]]:
        """
        Split samples by time periods.
        
        A sample belongs to a period if its start_time falls within that period.
        
        Parameters
        ----------
        samples : list of dict
            List of samples to split.
        periods : list of tuple
            List of (start, end) timestamp tuples.
            
        Returns
        -------
        list of dict
            Samples that fall within the specified periods.
        """
        period_samples = []
        
        for sample in samples:
            sample_start = sample['start_time']
            for period_start, period_end in periods:
                if period_start <= sample_start <= period_end:
                    period_samples.append(sample)
                    break
        
        return period_samples
    
    def split_data(
        self,
        df: pandas.DataFrame,
        data_periods: Dict[str, List[Tuple[pandas.Timestamp, pandas.Timestamp]]],
        max_lag: int,
        fcast_len: int,
        lead_time: int,
        data_freq: int
    ) -> Tuple[List[pandas.DataFrame], List[pandas.DataFrame], List[pandas.DataFrame]]:
        """
        Split data into training, validation, and testing sets with proper extensions
        for lagged features and forecast horizon.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The full dataset.
        data_periods : dict
            Dictionary with keys 'training', 'validation', 'testing' containing
            lists of (start, end) timestamp tuples.
        max_lag : int
            Maximum lag of the model (negative value, e.g., -24)
        fcast_len : int
            Forecast length in steps
        lead_time : int
            Lead time in steps
        data_freq : int
            Data frequency in minutes
            
        Returns
        -------
        tuple
            (training_set, validation_set, testing_set) - each is a list of DataFrames
        """
        training_set = create_dataset(
            df, data_periods['training'], max_lag, 
            fcast_len, lead_time, data_freq
        )
        validation_set = create_dataset(
            df, data_periods['validation'], max_lag,
            fcast_len, lead_time, data_freq
        )
        testing_set = create_dataset(
            df, data_periods['testing'], max_lag,
            fcast_len, lead_time, data_freq
        )
        
        return training_set, validation_set, testing_set
       
        
    def check_for_data_leaks(
        self,
        df: pandas.DataFrame,
        fcast_len: int,
        lead_time: int,
        method_name: str = 'output',
        raise_on_leak: bool = True
    ) -> Dict[str, Any]:
        """
        Check if the dataframe contains target features in the prediction horizon,
        which would constitute a data leak.
        
        This method checks if target column values exist for future timesteps
        (lead_time to lead_time + fcast_len) that should be predicted, not observed.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe to check for data leaks.
        fcast_len : int
            Forecast length in steps
        lead_time : int
            Lead time in steps
        method_name : str, default='output'
            Name of the method being checked (for logging purposes).
        raise_on_leak : bool, default=True
            If True, raise ValueError when data leak is detected.
            If False, log warning and return leak information.
            
        Returns
        -------
        dict
            Dictionary with leak detection results:
            - 'has_leak': bool indicating if leak was found
            - 'leak_columns': list of column names with leaks
            - 'leak_indices': list of indices where leaks occur
            - 'method_name': name of the method checked
            
        Raises
        ------
        ValueError
            If raise_on_leak=True and data leak is detected.
            
        Examples
        --------
        >>> loader = DataFrameDataLoader(...)
        >>> result = loader.check_for_data_leaks(df, fcast_len=24, lead_time=0, method_name='prepare_training_data')
        >>> if result['has_leak']:
        ...     print(f"Data leak found in columns: {result['leak_columns']}")
        """
        if not self.targets or self.targets not in df.columns:
            return {
                'has_leak': False,
                'leak_columns': [],
                'leak_indices': [],
                'method_name': method_name
            }
        
        leak_columns = []
        leak_indices = []
        
        prediction_horizon_start = lead_time
        prediction_horizon_end = lead_time + fcast_len
        
        for col in df.columns:
            if col == self.targets:
                leak_columns.append(col)
                leak_indices.extend(df.index.tolist())
            
            if col.startswith(f"{self.targets}_"):
                try:
                    suffix = col.split(f"{self.targets}_")[1]
                    if suffix.isdigit():
                        step = int(suffix)
                        if prediction_horizon_start <= step < prediction_horizon_end:
                            leak_columns.append(col)
                            non_null_idx = df[col].notna()
                            if non_null_idx.any():
                                leak_indices.extend(df[non_null_idx].index.tolist())
                except (ValueError, IndexError):
                    pass
        
        leak_columns = list(set(leak_columns))
        leak_indices = list(set(leak_indices))
        
        has_leak = len(leak_columns) > 0
        
        result = {
            'has_leak': has_leak,
            'leak_columns': leak_columns,
            'leak_indices': leak_indices,
            'method_name': method_name
        }
        
        if has_leak:
            msg = (
                f"Data leak detected in '{method_name}'!\n"
                f"  Columns with target values in prediction horizon: {leak_columns}\n"
                f"  Prediction horizon: steps {prediction_horizon_start} to {prediction_horizon_end}\n"
                f"  Number of affected indices: {len(leak_indices)}\n"
                f"  Target column: {self.targets}\n"
                f"  This means the model would have access to future target values it should predict."
            )
            
            if raise_on_leak:
                raise ValueError(msg)
            else:
                logger.warning(msg)
        
        return result
    
    def test_all_output_methods(
        self,
        sample_data: Union[pandas.DataFrame, List[pandas.DataFrame]],
        raise_on_leak: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Test all registered data output methods for data leaks.
        
        This method iterates through all methods listed in self.data_output_methods
        and checks their outputs for data leaks.
        
        Parameters
        ----------
        sample_data : pandas.DataFrame or list of pandas.DataFrame
            Sample data to test with. Should be representative of actual data.
        raise_on_leak : bool, default=True
            If True, raise ValueError on first detected leak.
            If False, collect all leaks and return summary.
            
        Returns
        -------
        dict
            Dictionary mapping method names to their leak detection results.
            
        Examples
        --------
        >>> loader = DataFrameDataLoader(...)
        >>> results = loader.test_all_output_methods(sample_df, raise_on_leak=False)
        >>> for method, result in results.items():
        ...     if result['has_leak']:
        ...         print(f"{method} has data leak!")
        """
        if not self.data_output_methods:
            logger.info("No data output methods registered to test.")
            return {}
        
        results = {}
        
        for method_name in self.data_output_methods:
            if not hasattr(self, method_name):
                logger.warning(f"Method '{method_name}' not found in {self.__class__.__name__}")
                continue
            
            try:
                method = getattr(self, method_name)
                
                # Call the method with sample data
                # Handle different signatures
                if method_name in ['prepare_training_data']:
                    if isinstance(sample_data, list):
                        output = method(sample_data, [])
                    else:
                        output = method([sample_data], [])
                elif method_name in ['prepare_prediction_data']:
                    if isinstance(sample_data, list):
                        output = method(sample_data[0] if sample_data else sample_data)
                    else:
                        output = method(sample_data)
                else:
                    # Generic call
                    output = method(sample_data)
                
                # Check output for leaks
                # Handle different output types
                dfs_to_check = []
                
                if isinstance(output, pandas.DataFrame):
                    dfs_to_check = [output]
                elif isinstance(output, list):
                    for item in output:
                        if isinstance(item, pandas.DataFrame):
                            dfs_to_check.append(item)
                elif isinstance(output, tuple):
                    for item in output:
                        if isinstance(item, pandas.DataFrame):
                            dfs_to_check.append(item)
                        elif isinstance(item, list):
                            dfs_to_check.extend([x for x in item if isinstance(x, pandas.DataFrame)])
                
                # Check each DataFrame for leaks
                method_has_leak = False
                leak_info = {
                    'has_leak': False,
                    'leak_columns': [],
                    'leak_indices': [],
                    'method_name': method_name
                }
                
                for df in dfs_to_check:
                    result = self.check_for_data_leaks(df, method_name, raise_on_leak=False)
                    if result['has_leak']:
                        method_has_leak = True
                        leak_info['leak_columns'].extend(result['leak_columns'])
                        leak_info['leak_indices'].extend(result['leak_indices'])
                
                leak_info['has_leak'] = method_has_leak
                leak_info['leak_columns'] = list(set(leak_info['leak_columns']))
                leak_info['leak_indices'] = list(set(leak_info['leak_indices']))
                
                results[method_name] = leak_info
                
                if method_has_leak and raise_on_leak:
                    msg = (
                        f"Data leak detected in method '{method_name}'!\n"
                        f"  Columns: {leak_info['leak_columns']}\n"
                        f"  Affected indices: {len(leak_info['leak_indices'])}"
                    )
                    raise ValueError(msg)
                
            except Exception as e:
                if raise_on_leak:
                    raise
                else:
                    logger.error(f"Error testing method '{method_name}': {e}")
                    results[method_name] = {
                        'has_leak': None,
                        'error': str(e),
                        'method_name': method_name
                    }
        
        # Log summary
        total_methods = len(results)
        methods_with_leaks = sum(1 for r in results.values() if r.get('has_leak', False))
        
        if methods_with_leaks > 0:
            logger.warning(
                f"Data leak test summary: {methods_with_leaks}/{total_methods} methods have data leaks"
            )
        else:
            logger.info(
                f"Data leak test summary: All {total_methods} methods passed (no leaks detected)"
            )
        
        return results
          
    @output_method
    def get_evaluation_dataframes(self, period: str, min_length: Optional[int] = None) -> List[pandas.DataFrame]:
        """
        Get evaluation DataFrames for the specified period.
        
        Returns continuous timeseries DataFrames that were created during
        load_and_split_data(). These represent the original time periods
        without windowing/sampling, suitable for model evaluation.
        
        Parameters
        ----------
        period : str
            The period for which evaluation dataframes are requested.
            Valid values: 'training', 'validation', 'testing'
        min_length : int, optional
            Minimum number of timesteps required for a segment to be included.
            Segments shorter than this will be filtered out with a warning.
            If None, no length filtering is applied. Default is None.
        
        Returns
        -------
        list of pandas.DataFrame
            List of continuous timeseries DataFrames for evaluation.
            Each DataFrame represents one continuous time period (e.g., one winter season).
            If min_length is specified, only segments with sufficient length are returned.
            
        Raises
        ------
        ValueError
            If period is invalid or data has not been loaded yet.
            
        Examples
        --------
        >>> loader.load_and_split_data('data.csv', data_periods={...})
        >>> train_dfs = loader.get_evaluation_dataframes('training')
        >>> # Returns list of DataFrames, e.g., [winter_2022_df, winter_2023_df]
        >>> 
        >>> # Filter segments shorter than window size
        >>> test_dfs = loader.get_evaluation_dataframes('testing', min_length=54)
        >>> # Returns only segments with at least 54 timesteps
        """
        if period not in ['training', 'validation', 'testing']:
            raise ValueError(
                f"Invalid period '{period}'. Must be one of: 'training', 'validation', 'testing'"
            )
        
        # Map period to the corresponding dataset attribute
        period_map = {
            'training': self.training_segments,
            'validation': self.validation_segments,
            'testing': self.testing_segments
        }
        
        dataset = period_map[period]
        
        if dataset is None:
            raise ValueError(
                f"No data available for period '{period}'. "
                f"Call load_and_split_data() first to load and split the data."
            )
        
        if len(dataset) == 0:
            logger.warning(f"No DataFrames available for period '{period}'. Returning empty list.")
            return []
        
        # Apply minimum length filtering if specified
        if min_length is not None:
            filtered_dataset = []
            filtered_count = 0
            
            for df in dataset:
                if len(df) >= min_length:
                    filtered_dataset.append(df)
                else:
                    filtered_count += 1
                    logger.warning(
                        f"Filtered out segment with {len(df)} timesteps "
                        f"(min required: {min_length}). "
                        f"Time range: {df.index[0]} to {df.index[-1]}"
                    )
            
            if filtered_count > 0:
                logger.info(
                    f"Filtered {filtered_count} short segment(s) from '{period}' period. "
                    f"Kept {len(filtered_dataset)} valid segment(s)."
                )
            
            dataset = filtered_dataset
        
        logger.info(
            f"Retrieved {len(dataset)} continuous timeseries DataFrame(s) for '{period}' period"
        )
        
        return dataset

    def _get_required_features(self) -> Optional[List[str]]:
        """
        Get the list of required features for data preprocessing.
        
        This method returns all features and targets that should be kept during
        preprocessing (sample creation, filtering). It combines the feature names
        with the target columns.
        
        Returns
        -------
        list of str
            List of required feature names (features + targets combined).
            
        Notes
        -----
        This is different from model-specific feature requirements. This method
        defines what the dataloader needs for preprocessing (combined from all models
        in the pipeline), while individual models specify their requirements via the
        `required_features` parameter in get_dataloaders().
        """
        # Get features from instance if available (subclasses may set this)
        feature_names = []
        if hasattr(self, 'features'):
            feature_names = list(self.features) if self.features else []
        
        target_names = list(self.targets) if self.targets else []
        
        return list(set(feature_names + target_names))
    
    def __repr__(self) -> str:
        data_loaded = "data_loaded=True" if self.df is not None else "data_loaded=False"
        samples_info = ""
        if self.samples is not None:
            samples_info = (
                f", samples={len(self.samples)}, "
                f"accepted={len(self.accepted_samples) if self.accepted_samples else 0}, "
                f"acceptance_rate={self.sample_acceptance_rate:.1f}%" 
                if self.sample_acceptance_rate is not None else ", samples=created"
            )
        
        feature_info = ""
        if self.feature_index is not None:
            feature_info = f", features={len(self.feature_index)}"
        
        return (
            f"{self.__class__.__name__}("
            f"update_rate={self.update_rate}, "
            f"normalize={self.normalize}, "
            f"{data_loaded}{feature_info}{samples_info})"
        )
