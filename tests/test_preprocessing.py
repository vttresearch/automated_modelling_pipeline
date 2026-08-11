import pandas as pd
import pytest
from amp.preprocessing import basic_features



# Setup
lead_time = 0
forecast_len = 24


def test_basic_features_default():
    # Test basic_features with default parameters

    f_list = ['lagged_target', 'weekday', 'hour', 't_out', 'holiday']

    features = basic_features(lead_time=lead_time,
                              forecast_len=forecast_len,
                              features=f_list,
                              target_lags=[24 * 7],
                              target_lag_len=24 * 7,
                              temp_lag=8)
    assert isinstance(features, dict)
    for f in f_list:
        assert f in features


