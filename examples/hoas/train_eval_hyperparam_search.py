"""
Module containing functions for forecasting and evaluation of energy consumption data.
"""

import pandas as pd  # Import library for efficient data manipulation and analysis
import pathlib  # Import library for handling file paths and directories

from amp.tools import train_eval_tool  # Import tool for training and evaluating models
from amp.baselines import MeanBaseline, LagBaseline  # Import baseline models for energy consumption forecasting
from amp.utils import param_select  # Import function for selecting hyperparameters to optimize
from amp.preprocessing import basic_features  # Import module for feature engineering of energy consumption data
from amp.efp.models.lightgbm import LightGBMForecaster  # Import LightGBM forecaster model
from amp.base import Target

# Configure pandas to raise warnings on chained assignment operations
pd.options.mode.chained_assignment = "raise"


def load_data():
    """
    Load the HOAS example dataset from a CSV file.

    Returns:
        pd.DataFrame: The loaded dataset with 'timestamp' as index and dates in True format.
    """
    filepath = pathlib.Path('data/hoas_example_data.csv')
    return pd.read_csv(filepath, index_col='timestamp', parse_dates=True)


def define_forecasters(targets, data_freq, forecast_len, lead_time, update_freq):
    """
    Define a dictionary of LightGBM forecaster models for each output variable.

    Args:
        targets (list): List of Targets to be forecasted.
        data_freq (int): Data frequency in minutes.
        forecast_len (int): Forecast length in minutes.
        lead_time (int): Lead time in minutes.
        update_freq (int): Update frequency.

    Returns:
        dict: Dictionary of LightGBM forecaster models for each output variable.
    """
    # Define basic features that are used in several models.
    features = basic_features(lead_time=lead_time,
                              forecast_len=forecast_len,
                              features=['lagged_target', 'weekday', 'hour', 't_out', 'holiday'],
                              target_lags=[24 * 7],
                              target_lag_len=24 * 7,
                              temp_lag=8)

    # Define hyperparameter search spaces for LightGBM forecaster model
    # We'll use three different search spaces with varying levels of complexity

    param_search_base_lightgbm = {
        'num_leaves': [7, 14, 21]
    }

    param_search_base_lightgbm = {
        'num_leaves': [7, 14, 21],  # Number of leaves in each tree.
        'learning_rate': [0.01, 0.05, 0.1],  # Learning rate for gradient boosting.
        'min_data_in_leaf': [10, 20, 50],  # Minimum data needed in a leaf.
        'feature_fraction': [0.6, 0.8, 1.0],  # Fraction of features used in building trees.
        'bagging_fraction': [0.6, 0.8, 1.0],  # Fraction of data used for bagging.
        'bagging_freq': [1, 5, 10]  # Frequency for bagging.
    }


    param_search_base_lightgbm_poor = {
        'num_leaves': [2],
        'learning_rate': [0.0001],
        'min_data_in_leaf': [100, 500],
        'feature_fraction': [0.1, 0.3],
        'bagging_fraction': [0.1],
        'bagging_freq': [0]
    }

    param_search_base_lightgbm_poor_and_defaults = {
        'num_leaves': [31],  # Default: 31
        'learning_rate': [0.1],  # Default: 0.1
        'min_data_in_leaf': [200, 500],  # Default: 20
        'feature_fraction': [1.0],  # Default: 1.0
        'bagging_fraction': [1.0],  # Default: 1.0
        'bagging_freq': [0],  # Default: 0 (bagging disabled)
    }


    # Select hyperparameter combination and method
    hyperparam_search_lightgbm ={
        'hyperparam_space': param_search_base_lightgbm_poor_and_defaults,
        'hyperparam_search_method': 'grid_search',
    }

    features_dict = {'f1': features}
    model_name = 'lightgbm'
    forecaster_dict = {f'{model_name}_{fname}': LightGBMForecaster(targets,
                                                                   lead_time,
                                                                   forecast_len,
                                                                   data_freq,
                                                                   features=feature,
                                                                   update_freq=update_freq,
                                                                   verbose=3,
                                                                   hyperparams={'max_depth': 16},
                                                                   hyperparam_search=hyperparam_search_lightgbm) for
                       (fname, feature) in features_dict.items()}

    forecasters = {}
    forecasters.update(forecaster_dict)

    return forecasters


def define_targets():
    """
    Define the target variables and their corresponding metadata.

    Returns:
        list: List of dictionaries containing metadata for each target variable.
    """
    dh_target = Target(output='dh')
    ele_target = Target(output='ele')

    return [dh_target, ele_target]


if __name__ == '__main__':
    """
    Main function for training and evaluating models.
    """
    targets = define_targets()
    data_freq = 60  # Data frequency in minutes
    forecast_len = 24  # Forecast length in minutes
    lead_time = 0
    update_freq = 1



    all_models = define_forecasters(targets,
                                    data_freq,
                                    forecast_len,
                                    lead_time,
                                    update_freq)

    mode, models, plot, verbose = param_select(all_models)
    df = load_data()
    test_period = ('2020-01-01', '2020-03-30')
    baselines = {'mean': MeanBaseline(),
                 'lag_4': LagBaseline(4),
                 'lag_24': LagBaseline(24)}
    train_eval_tool(mode,
                    targets,
                    models,
                    df,
                    test_period,
                    data_freq,
                    forecast_len,
                    lead_time,
                    update_freq,
                    baselines,
                    plot,
                    verbose,
                    'trained_models')
