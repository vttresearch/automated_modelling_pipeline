from amp.efp.base import Forecaster
import pytest
import pandas as pd
import pathlib
import pickle

from amp.efp.models.sklearn import RandomForestForecaster, LinearForecaster, MLPForecaster, DecisionTreeForecaster, MLPMultiForecaster, SVRForecaster
from amp.efp.models.lightgbm import LightGBMForecaster
from amp.base import Target

DATA_DIR = pathlib.Path(__file__).resolve().parent / 'data'

@pytest.fixture
def sample_data():
    filepath = DATA_DIR / 'hoas_example_data.csv'
    return pd.read_csv(filepath, index_col='timestamp', parse_dates=True)

@pytest.fixture()
def prediction_data():
    with open(DATA_DIR / 'rf_predictions.pickle', 'rb') as handle:
        return pickle.load(handle)

@pytest.fixture
def forecaster_params():
    return {
        'targets': [Target(output='dh'), Target(output='ele')],
        'lead_time': 0,
        'forecast_len': 24,
        'data_freq': 60,
        'features': {
            'lagged_target': {'windows': [(-168, -1)], 'type': 'numeric'},
            'weekday': {'windows': [(0, 23)], 'type': 'onehot'},
            'hour': {'windows': [(0, 23)], 'type': 'onehot'},
            't_out': {'windows': [(-8, 23)], 'type': 'numeric'},
            'holiday': {'windows': [(0, 23)], 'type': 'onehot'}
        },
        'hyperparams': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10, 'n_jobs': -1},
        'update_freq': 1,
        'upper_limits': None,
        'lower_limits': None
    }

@pytest.fixture
def forecaster_params_simple():
    return {
        'targets': [Target(output='dh'), Target(output='ele')],
        'lead_time': 0,
        'forecast_len': 24,
        'data_freq': 60,
        'features': {
            'lagged_target': {'windows': [(-168, -1)], 'type': 'numeric'},
            'weekday': {'windows': [(0, 23)], 'type': 'onehot'},
            'hour': {'windows': [(0, 23)], 'type': 'onehot'},
            't_out': {'windows': [(-8, 23)], 'type': 'numeric'},
            'holiday': {'windows': [(0, 23)], 'type': 'onehot'}
        },
        'update_freq': 1,
        'upper_limits': None,
        'lower_limits': None
    }


@pytest.fixture
def forecaster_params_search():
    return {
        'targets': [Target(output='dh'), Target(output='ele')],
        'lead_time': 0,
        'forecast_len': 24,
        'data_freq': 60,
        'features': {'lagged_target': {'windows': [(-168, -1)], 'type': 'numeric'},
                     'weekday': {'windows': [(0, 23)], 'type': 'onehot'},
                     'hour': {'windows': [(0, 23)], 'type': 'onehot'},
                     't_out': {'windows': [(-8, 23)], 'type': 'numeric'},
                     'holiday': {'windows': [(0, 23)], 'type': 'onehot'}},
        'hyperparams': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10, 'n_jobs': -1},
        'hyperparam_search': {'hyperparam_search_method': 'grid_search',
        'hyperparam_space': {'num_leaves': [7, 14, 21]}},
        'update_freq': 1,
            'upper_limits': None,
            'lower_limits': None
        }

def test_create_rf(sample_data, forecaster_params):
    # Initialize a RandomForestForecaster instance
    forecaster = RandomForestForecaster(**forecaster_params)
    # Check if the Forecaster instance is created properly
    assert isinstance(forecaster, RandomForestForecaster)
    assert forecaster.forecast_len == forecaster_params['forecast_len']
    assert forecaster._outputs == forecaster_params['targets']


def test_create_lightgbm(sample_data, forecaster_params_search, prediction_data):

    forecaster = LightGBMForecaster(**forecaster_params_search)

    # Check if the Forecaster instance is created properly
    assert isinstance(forecaster, LightGBMForecaster)
    assert forecaster.forecast_len == forecaster_params_search['forecast_len']
    assert forecaster._outputs == forecaster_params_search['targets']


def test_fit_predict(sample_data, forecaster_params, prediction_data):

    # Initialize a RandomForestForecaster instance
    forecaster = RandomForestForecaster(**forecaster_params)

    # Test fitting and predicting with the Forecaster instance
    forecaster.fit(sample_data)
    predictions = forecaster.predict(sample_data)
    predictions['ele'] = predictions['ele'].round(2)
    predictions['dh'] = predictions['dh'].round(2)
    #compare to old predictions
    assert predictions['dh'].equals(prediction_data['dh'])
    assert predictions['ele'].equals(prediction_data['ele'])

"""
def test_fit_predict_FFN(sample_data, forecaster_params_simple):
    from amp.efp.models.tf import FFNForecaster
    

    # Initialize a RandomForestForecaster instance
    forecaster = FFNForecaster(**forecaster_params_simple)

    # Test fitting and predicting with the Forecaster instance
    forecaster.fit(sample_data)
    #predictions = forecaster.predict(sample_data)
    #predictions['ele'] = predictions['ele'].round(2)
    #predictions['dh'] = predictions['dh'].round(2)

"""

@pytest.mark.parametrize(
    "forecaster_class",
    [
        LinearForecaster,
        RandomForestForecaster,
        MLPForecaster,
        MLPMultiForecaster,
        SVRForecaster,
        DecisionTreeForecaster
    ],
)
def test_create_and_fit_forecaster(forecaster_class, sample_data, forecaster_params_simple):
    # Initialize forecaster
    forecaster = forecaster_class(**forecaster_params_simple)

    # Verify instance creation
    assert isinstance(forecaster, forecaster_class)
    assert forecaster.forecast_len == forecaster_params_simple['forecast_len']
    assert forecaster._outputs == forecaster_params_simple['targets']

    target = forecaster_params_simple['targets']

    reduced_sample = sample_data.tail(500)

    forecaster.fit(reduced_sample)
    predictions = forecaster.predict(reduced_sample)
    for key, df in predictions.items():
        assert df.shape[1] == forecaster_params_simple['forecast_len'], (f"Column count mismatch for key '{key}':"
                                                                         f" expected "
                                                                         f"{forecaster_params_simple['forecast_len']}, "
                                                                         f"got {df.shape[1]}")
