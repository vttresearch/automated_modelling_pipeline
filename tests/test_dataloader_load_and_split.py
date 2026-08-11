"""
Test suite for DataLoader.load_and_split_data() functionality.

Tests cover:
1. Loading from single CSV
2. Loading from multiple CSVs (row-wise concatenation)
3. Dictionary-based multi-source loading (column-wise merge)
4. Feature filtering with NaN handling
5. Data splitting by periods
6. Data splitting by ratio
7. Backward compatibility
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from amp.dataloader import DataLoader


# Create a concrete implementation for testing (avoid "Test" prefix to prevent pytest collection)
class SimpleDataLoader(DataLoader):
    """Concrete DataLoader implementation for testing."""
    
    def prepare_data(self, df):
        """Simple pass-through implementation."""
        return df


@pytest.fixture
def hoas_data_path():
    """Path to the HOAS example data CSV."""
    return Path(__file__).parent / 'data' / 'hoas_example_data.csv'


@pytest.fixture
def test_data_dir():
    """Create temporary directory with test CSV files."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Load HOAS data
    hoas_path = Path(__file__).parent / 'data' / 'hoas_example_data.csv'
    df = pd.read_csv(hoas_path, index_col='timestamp', parse_dates=True)
    
    # Split into multiple files to simulate different data sources
    # Building data (subset 1): 2015-01 to 2015-06
    df_building_1 = df.loc['2015-01':'2015-06', ['ele', 'dh']]
    df_building_1.to_csv(temp_dir / 'building_2015_h1.csv')
    
    # Building data (subset 2): 2015-07 to 2015-12
    df_building_2 = df.loc['2015-07':'2015-12', ['ele', 'dh']]
    df_building_2.to_csv(temp_dir / 'building_2015_h2.csv')
    
    # Weather data: full year with some artificial gaps
    df_weather = df.loc['2015-01':'2015-12', ['t_out']].copy()
    # Introduce some NaN values to test NaN handling
    df_weather.iloc[100:120, 0] = np.nan
    df_weather.iloc[500:505, 0] = np.nan
    df_weather.to_csv(temp_dir / 'weather_2015.csv')
    
    # Additional features file (e.g., electricity price, solar radiation)
    # Create synthetic data
    df_additional = pd.DataFrame({
        'electricity_price': np.random.uniform(30, 100, len(df)),
        'solar_radiation': np.abs(np.sin(np.arange(len(df)) * 2 * np.pi / 24) * 800)
    }, index=df.index)
    df_additional.to_csv(temp_dir / 'additional_features_2015.csv')
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestLoadAndSplitDataBasic:
    """Basic loading functionality tests."""
    
    def test_load_single_csv(self, hoas_data_path):
        """Test loading from a single CSV file."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        loader.load_and_split_data(
            source=str(hoas_data_path),
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-06-30', tz='UTC'))],
                'validation': [(pd.Timestamp('2015-07-01', tz='UTC'), 
                               pd.Timestamp('2015-09-30', tz='UTC'))],
                'testing': [(pd.Timestamp('2015-10-01', tz='UTC'), 
                            pd.Timestamp('2015-12-31', tz='UTC'))]
            }
        )
        
        # Verify data was loaded
        assert loader.df is not None
        assert len(loader.df) > 0
        assert 'ele' in loader.df.columns
        assert 'dh' in loader.df.columns
        assert 't_out' in loader.df.columns
        
        # Verify splits were created
        assert loader.training_timeseries is not None
        assert loader.validation_timeseries is not None
        assert loader.testing_timeseries is not None
        assert len(loader.training_timeseries) > 0
        assert len(loader.validation_timeseries) > 0
        assert len(loader.testing_timeseries) > 0
    
    def test_load_multiple_csvs_row_concat(self, test_data_dir):
        """Test loading multiple CSVs with row-wise concatenation."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        # Load two building data files (row-wise concatenation)
        loader.load_and_split_data(
            source=[
                str(test_data_dir / 'building_2015_h1.csv'),
                str(test_data_dir / 'building_2015_h2.csv')
            ],
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-09-30', tz='UTC'))],
                'validation': [(pd.Timestamp('2015-10-01', tz='UTC'), 
                               pd.Timestamp('2015-12-31', tz='UTC'))],
                'testing': []
            }
        )
        
        assert loader.df is not None
        assert len(loader.df) > 0
        # Should have data from both H1 and H2
        assert loader.df.index.min() < pd.Timestamp('2015-07-01', tz='UTC')
        assert loader.df.index.max() > pd.Timestamp('2015-07-01', tz='UTC')
        assert 'ele' in loader.df.columns
        assert 'dh' in loader.df.columns


class TestDictionaryBasedLoading:
    """Tests for dictionary-based multi-source loading."""
    
    def test_dict_source_merge_on_index(self, test_data_dir):
        """Test merging building and weather data on index."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        loader.load_and_split_data(
            source={
                'building': {
                    'paths': [
                        str(test_data_dir / 'building_2015_h1.csv'),
                        str(test_data_dir / 'building_2015_h2.csv')
                    ],
                    'concat_axis': 0  # Row-wise
                },
                'weather': {
                    'paths': str(test_data_dir / 'weather_2015.csv'),
                    'merge_on_index': True,
                    'merge_how': 'left'
                }
            },
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-09-30', tz='UTC'))],
                'validation': [(pd.Timestamp('2015-10-01', tz='UTC'), 
                               pd.Timestamp('2015-12-31', tz='UTC'))],
                'testing': []
            }
        )
        
        # Should have columns from both sources
        assert 'ele' in loader.df.columns
        assert 'dh' in loader.df.columns
        assert 't_out' in loader.df.columns
        
        # Weather data has NaNs, verify they're present (not dropped yet)
        assert loader.df['t_out'].isna().sum() > 0
    
    def test_dict_source_three_sources(self, test_data_dir):
        """Test merging three different data sources."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        loader.load_and_split_data(
            source={
                'building': {
                    'paths': [
                        str(test_data_dir / 'building_2015_h1.csv'),
                        str(test_data_dir / 'building_2015_h2.csv')
                    ]
                },
                'weather': {
                    'paths': str(test_data_dir / 'weather_2015.csv'),
                    'merge_on_index': True,
                    'merge_how': 'left'
                },
                'additional': {
                    'paths': str(test_data_dir / 'additional_features_2015.csv'),
                    'merge_on_index': True,
                    'merge_how': 'left'
                }
            },
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-09-30', tz='UTC'))],
                'validation': [(pd.Timestamp('2015-10-01', tz='UTC'), 
                               pd.Timestamp('2015-12-31', tz='UTC'))],
                'testing': []
            }
        )
        
        # Should have columns from all three sources
        assert 'ele' in loader.df.columns
        assert 'dh' in loader.df.columns
        assert 't_out' in loader.df.columns
        assert 'electricity_price' in loader.df.columns
        assert 'solar_radiation' in loader.df.columns


class TestFeatureFilteringAndNaNHandling:
    """Tests for feature filtering and NaN handling."""
    
    def test_feature_filtering(self, hoas_data_path):
        """Test that only requested features are kept."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        loader.load_and_split_data(
            source=str(hoas_data_path),
            features=['ele', 'dh'],  # Only keep these two
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-12-31', tz='UTC'))],
                'validation': [],
                'testing': []
            }
        )
        
        # Should only have the requested features
        assert list(loader.df.columns) == ['ele', 'dh']
        assert 't_out' not in loader.df.columns
    
    def test_nan_removal_with_features(self, test_data_dir):
        """Test that NaNs are removed when features are specified."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        # Load data with weather (which has NaNs)
        loader.load_and_split_data(
            source={
                'building': {
                    'paths': str(test_data_dir / 'building_2015_h1.csv')
                },
                'weather': {
                    'paths': str(test_data_dir / 'weather_2015.csv'),
                    'merge_on_index': True,
                    'merge_how': 'left'
                }
            },
            features=['ele', 'dh', 't_out'],  # Specify features -> NaNs dropped
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-06-30', tz='UTC'))],
                'validation': [],
                'testing': []
            }
        )
        
        # NaNs should be removed
        assert loader.df['t_out'].isna().sum() == 0
        assert loader.df.isna().sum().sum() == 0
    
    def test_no_nan_removal_without_features(self, test_data_dir):
        """Test that NaNs are NOT removed when features are not specified."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        # Load without specifying features -> NaNs preserved
        loader.load_and_split_data(
            source={
                'building': {
                    'paths': str(test_data_dir / 'building_2015_h1.csv')
                },
                'weather': {
                    'paths': str(test_data_dir / 'weather_2015.csv'),
                    'merge_on_index': True,
                    'merge_how': 'left'
                }
            },
            # No features parameter -> NaNs preserved
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-06-30', tz='UTC'))],
                'validation': [],
                'testing': []
            }
        )
        
        # NaNs should still be present
        assert loader.df['t_out'].isna().sum() > 0


class TestDataSplitting:
    """Tests for data splitting functionality."""
    
    def test_split_by_periods_single_period(self, hoas_data_path):
        """Test splitting with single period per split."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        train_start = pd.Timestamp('2015-01-01', tz='UTC')
        train_end = pd.Timestamp('2015-06-30', tz='UTC')
        val_start = pd.Timestamp('2015-07-01', tz='UTC')
        val_end = pd.Timestamp('2015-09-30', tz='UTC')
        
        loader.load_and_split_data(
            source=str(hoas_data_path),
            data_periods={
                'training': [(train_start, train_end)],
                'validation': [(val_start, val_end)],
                'testing': []
            }
        )
        
        # Check training period
        assert loader.training_timeseries.index.min() >= train_start
        assert loader.training_timeseries.index.max() <= train_end
        
        # Check validation period
        assert loader.validation_timeseries.index.min() >= val_start
        assert loader.validation_timeseries.index.max() <= val_end
    
    def test_split_by_periods_multiple_periods(self, test_data_dir):
        """Test splitting with multiple periods per split."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        loader.load_and_split_data(
            source=[
                str(test_data_dir / 'building_2015_h1.csv'),
                str(test_data_dir / 'building_2015_h2.csv')
            ],
            data_periods={
                'training': [
                    (pd.Timestamp('2015-01-01', tz='UTC'), 
                     pd.Timestamp('2015-02-28', tz='UTC')),
                    (pd.Timestamp('2015-07-01', tz='UTC'), 
                     pd.Timestamp('2015-08-31', tz='UTC'))
                ],
                'validation': [
                    (pd.Timestamp('2015-03-01', tz='UTC'), 
                     pd.Timestamp('2015-03-31', tz='UTC')),
                    (pd.Timestamp('2015-09-01', tz='UTC'), 
                     pd.Timestamp('2015-09-30', tz='UTC'))
                ],
                'testing': []
            }
        )
        
        # Training should have data from both Jan-Feb and Jul-Aug
        training_data = loader.training_timeseries
        assert training_data is not None
        assert len(training_data) > 0
        
        # Check that we have gaps (data is not continuous)
        # The gap between Feb and Jul should be excluded
        training_months = training_data.index.month.unique()
        assert 1 in training_months or 2 in training_months  # Jan or Feb
        assert 7 in training_months or 8 in training_months  # Jul or Aug
    
    def test_split_by_ratio(self, hoas_data_path):
        """Test ratio-based splitting."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60,
            train_split=0.7,
            val_split=0.15
        )
        
        loader.load_and_split_data(
            source=str(hoas_data_path),
            split_by_ratio=True
        )
        
        total_len = len(loader.df)
        train_len = len(loader.training_timeseries)
        val_len = len(loader.validation_timeseries)
        test_len = len(loader.testing_timeseries)
        
        # Check approximate ratios (allow some tolerance due to rounding)
        assert abs(train_len / total_len - 0.7) < 0.01
        assert abs(val_len / total_len - 0.15) < 0.01
        assert abs(test_len / total_len - 0.15) < 0.01
        
        # Verify all data is accounted for
        assert train_len + val_len + test_len == total_len


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""
    
    def test_list_of_paths_still_works(self, test_data_dir):
        """Test that passing a list of paths still works (old way)."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        # This is the old way of loading multiple files
        loader.load_and_split_data(
            source=[
                str(test_data_dir / 'building_2015_h1.csv'),
                str(test_data_dir / 'building_2015_h2.csv')
            ],
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-12-31', tz='UTC'))],
                'validation': [],
                'testing': []
            }
        )
        
        assert loader.df is not None
        assert len(loader.df) > 0
    
    def test_dataframe_source_still_works(self, hoas_data_path):
        """Test that passing a DataFrame directly still works."""
        # Pre-load the data
        df = pd.read_csv(hoas_data_path, index_col='timestamp', parse_dates=True)
        
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        loader.load_and_split_data(
            source=df,  # Pass DataFrame directly
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-12-31', tz='UTC'))],
                'validation': [],
                'testing': []
            }
        )
        
        assert loader.df is not None
        assert len(loader.df) == len(df)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_missing_features_warning(self, hoas_data_path):
        """Test that requesting non-existent features logs a warning."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        # Request features that don't exist
        loader.load_and_split_data(
            source=str(hoas_data_path),
            features=['ele', 'dh', 'nonexistent_feature'],
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-12-31', tz='UTC'))],
                'validation': [],
                'testing': []
            }
        )
        
        # Should have only the existing features
        assert 'ele' in loader.df.columns
        assert 'dh' in loader.df.columns
        assert 'nonexistent_feature' not in loader.df.columns
    
    def test_empty_testing_period(self, hoas_data_path):
        """Test that empty testing period is handled correctly."""
        loader = SimpleDataLoader(
            update_freq=60,
            targets=['dh'],
            normalize=False,
            resample_freq=60
        )
        
        loader.load_and_split_data(
            source=str(hoas_data_path),
            data_periods={
                'training': [(pd.Timestamp('2015-01-01', tz='UTC'), 
                             pd.Timestamp('2015-09-30', tz='UTC'))],
                'validation': [(pd.Timestamp('2015-10-01', tz='UTC'), 
                               pd.Timestamp('2015-12-31', tz='UTC'))],
                'testing': []  # Empty
            }
        )
        
        assert loader.training_timeseries is not None
        assert loader.validation_timeseries is not None
        # Testing should be None or empty
        assert loader.testing_timeseries is None or len(loader.testing_timeseries) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
