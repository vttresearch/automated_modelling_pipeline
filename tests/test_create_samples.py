import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from amp.dataloader import create_samples, normalize_data, denormalize_data, find_gap_indices


# Setup - load test data
TEST_DATA_PATH = Path(__file__).parent / 'data' / 'hoas_example_data.csv'


@pytest.fixture
def sample_data():
    """Load HOAS example data for testing."""
    df = pd.read_csv(TEST_DATA_PATH, parse_dates=['timestamp'], index_col='timestamp')
    # Use first 1000 rows for faster testing
    return df.head(1000)


@pytest.fixture
def norm_params(sample_data):
    """Create normalization parameters from sample data."""
    return {
        'mean': sample_data.mean(),
        'std': sample_data.std()
    }


def test_create_samples_basic(sample_data):
    """Test basic sample creation with default parameters."""
    features = ['t_out', 'ele']
    targets = 'dh'
    input_window = (-24, 23)  # 24 hours past, 24 hours future
    
    samples = create_samples(
        df=sample_data,
        features=features,
        targets=targets,
        input_window=input_window,
        output_format='list',
        stride=1,
        normalize=False,
        resample_freq=60  # hourly data
    )
    
    assert isinstance(samples, list)
    assert len(samples) > 0
    
    # Check first sample structure
    sample = samples[0]
    assert 'data' in sample
    assert 'start_idx' in sample
    assert 'end_idx' in sample
    assert 'start_time' in sample
    assert 'end_time' in sample
    assert 'window_size' in sample
    assert 'center_idx' in sample
    
    # Check sample data shape
    assert isinstance(sample['data'], pd.DataFrame)
    assert len(sample['data']) == 48  # 24 past + 24 future
    assert all(col in sample['data'].columns for col in features + [targets])


def test_create_samples_output_formats(sample_data):
    """Test different output formats."""
    features = ['t_out']
    targets = 'dh'
    input_window = (-10, 9)  # 10 hours past, 10 hours future
    
    # Test list format
    samples_list = create_samples(
        df=sample_data,
        features=features,
        targets=targets,
        input_window=input_window,
        output_format='list',
        stride=10,
        normalize=False,
        resample_freq=60
    )
    assert isinstance(samples_list, list)
    assert len(samples_list) > 0
    
    # Test dataframes format
    samples_dfs = create_samples(
        df=sample_data,
        features=features,
        targets=targets,
        input_window=input_window,
        output_format='dataframes',
        stride=10,
        normalize=False,
        resample_freq=60
    )
    assert isinstance(samples_dfs, list)
    assert all(isinstance(df, pd.DataFrame) for df in samples_dfs)
    
    # Test array format
    samples_array = create_samples(
        df=sample_data,
        features=features,
        targets=targets,
        input_window=input_window,
        output_format='array',
        stride=10,
        normalize=False,
        resample_freq=60
    )
    assert isinstance(samples_array, np.ndarray)
    assert samples_array.ndim == 3  # (n_samples, window_size, n_features)


def test_create_samples_with_normalization(sample_data, norm_params):
    """Test sample creation with normalization."""
    features = ['t_out', 'ele']
    targets = 'dh'
    input_window = (-12, 11)
    
    samples = create_samples(
        df=sample_data,
        features=features,
        targets=targets,
        input_window=input_window,
        output_format='list',
        stride=5,
        normalize=True,
        norm_params=norm_params,
        resample_freq=60
    )
    
    assert len(samples) > 0
    
    # Check that values are normalized (should have different scale)
    sample_data_normalized = samples[0]['data']
    original_sample = sample_data.iloc[samples[0]['start_idx']:samples[0]['end_idx']+1]
    
    # Normalized data should be different from original
    assert not np.allclose(sample_data_normalized.values, original_sample[sample_data_normalized.columns].values)


def test_create_samples_stride(sample_data):
    """Test that stride parameter works correctly."""
    features = ['t_out']
    targets = 'dh'
    input_window = (-10, 9)
    
    samples_stride1 = create_samples(
        df=sample_data,
        features=features,
        targets=targets,
        input_window=input_window,
        stride=1,
        normalize=False,
        resample_freq=60
    )
    
    samples_stride10 = create_samples(
        df=sample_data,
        features=features,
        targets=targets,
        input_window=input_window,
        stride=10,
        normalize=False,
        resample_freq=60
    )
    
    # stride=10 should produce roughly 1/10 the samples
    assert len(samples_stride10) < len(samples_stride1)
    assert len(samples_stride1) > len(samples_stride10) * 8  # Allow some margin


def test_create_samples_multiple_targets(sample_data):
    """Test sample creation with multiple targets."""
    features = ['t_out']
    targets = ['dh', 'ele']  # Multiple targets
    input_window = (-5, 4)
    
    samples = create_samples(
        df=sample_data,
        features=features,
        targets=targets,
        input_window=input_window,
        normalize=False,
        resample_freq=60
    )
    
    assert len(samples) > 0
    sample = samples[0]
    
    # All targets should be in the data
    assert all(target in sample['data'].columns for target in targets)


def test_create_samples_invalid_input():
    """Test error handling for invalid inputs."""
    # Test with None dataframe
    with pytest.raises(ValueError, match="No data available"):
        create_samples(
            df=None,
            features=['t_out'],
            targets='dh',
            input_window=(-10, 9),
            resample_freq=60
        )
    
    # Test with empty dataframe
    with pytest.raises(ValueError, match="No data available"):
        create_samples(
            df=pd.DataFrame(),
            features=['t_out'],
            targets='dh',
            input_window=(-10, 9),
            resample_freq=60
        )


def test_create_samples_normalization_without_params(sample_data):
    """Test that normalization requires norm_params."""
    with pytest.raises(ValueError, match="norm_params must be provided"):
        create_samples(
            df=sample_data,
            features=['t_out'],
            targets='dh',
            input_window=(-10, 9),
            normalize=True,
            norm_params=None,
            resample_freq=60
        )


def test_find_gap_indices():
    """Test gap detection in timeseries."""
    # Create data with a gap
    dates = pd.date_range('2020-01-01', periods=10, freq='h')
    dates_with_gap = dates[:5].append(dates[7:])  # Skip 2 hours
    df = pd.DataFrame({'value': range(8)}, index=dates_with_gap)
    
    gaps = find_gap_indices(df, tolerance=1.1, freq=60)
    
    # Should find the gap
    assert len(gaps) > 0


def test_normalize_denormalize_data():
    """Test normalization and denormalization functions."""
    # Create simple test data
    df = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0, 5.0],
        'b': [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    # Create normalization parameters
    norm_params = {
        'mean': df.mean(),
        'std': df.std()
    }
    
    # Normalize
    df_normalized = normalize_data(df, norm_params)
    
    # Check that normalized data has mean ≈ 0 and std ≈ 1
    assert np.allclose(df_normalized.mean(), 0.0, atol=1e-10)
    assert np.allclose(df_normalized.std(), 1.0, atol=1e-10)
    
    # Denormalize
    df_denormalized = denormalize_data(df_normalized, norm_params)
    
    # Should get back original data
    assert np.allclose(df_denormalized.values, df.values, atol=1e-10)


def test_normalize_denormalize_specific_columns():
    """Test normalization/denormalization of specific columns."""
    df = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0, 5.0],
        'b': [10.0, 20.0, 30.0, 40.0, 50.0],
        'c': [100.0, 200.0, 300.0, 400.0, 500.0]
    })
    
    # Create normalization parameters for all columns
    norm_params = {
        'mean': df.mean(),
        'std': df.std()
    }
    
    # Normalize only columns 'a' and 'b'
    df_normalized = normalize_data(df, norm_params, columns=['a', 'b'])
    
    # Column 'c' should remain unchanged
    assert np.allclose(df_normalized['c'].values, df['c'].values)
    
    # Columns 'a' and 'b' should be normalized
    assert not np.allclose(df_normalized['a'].values, df['a'].values)
    assert not np.allclose(df_normalized['b'].values, df['b'].values)
    
    # Denormalize only column 'a'
    df_denormalized = denormalize_data(df_normalized, norm_params, columns=['a'])
    
    # Column 'a' should be back to original
    assert np.allclose(df_denormalized['a'].values, df['a'].values, atol=1e-10)
    
    # Column 'b' should still be normalized
    assert not np.allclose(df_denormalized['b'].values, df['b'].values)
