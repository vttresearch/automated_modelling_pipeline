"""
Module Description:

This module contains functions for loading data, defining forecasters and targets,
and running a train-evaluation tool.

Functions:

* load_data: Loads the HOAS example data from a CSV file.
* define_forecasters: Defines basic features and various forecasters for different models.
* define_targets: Defines the target variables to be predicted.
* param_select: Selects the mode, models, plot, and verbose options based on user input.

Classes:

* RandomForestForecaster: A class representing a random forest forecaster model.
* LightGBMForecaster: A class representing a LightGBM forecaster model.
* MeanBaseline: A class representing a mean baseline model.
* LagBaseline: A class representing a lag-based baseline model.
"""

import pandas as pd
import pathlib

from amp.tools import train_eval_tool
from amp.baselines import MeanBaseline, LagBaseline
from amp.utils import param_select
from amp.preprocessing import basic_features
from amp.efp.models.sklearn import RandomForestForecaster
from amp.efp.models.lightgbm import LightGBMForecaster
from amp.timesfm.forecaster import TimesFMForecaster
from amp.base import Target, load_model

pd.options.mode.chained_assignment = "raise"
pass

def load_data():
    """
    Loads the HOAS example data from a CSV file.

    Returns:
        pd.DataFrame: The loaded data with timestamp as index and parse_dates=True.
    """
    filepath = pathlib.Path('data/hoas_example_data.csv')
    return pd.read_csv(filepath, index_col='timestamp', parse_dates=True)




def define_forecasters(targets, data_freq, forecast_len, lead_time, update_freq):
    """
    Defines basic features that are used in several models and various forecasters for different models.

    Args:
        targets (list): The Targets variables to be predicted.
        data_freq (int): The frequency of the data in minutes.
        forecast_len (int): The length of the forecast period.
        lead_time (int): The lead time for the prediction.
        update_freq (int): The update frequency for the model.

    Returns:
        dict: A dictionary containing the defined forecasters.
    """
    # Define basic features that are used in several models.
    features = basic_features(lead_time=lead_time,
                              forecast_len=forecast_len,
                              features=['lagged_target', 'weekday', 'hour', 't_out', 'holiday'],
                              target_lags=[24 * 7, 24],
                              target_lag_len=24,
                              temp_lag=8)

    forecasters = {}
    other = {'rf': RandomForestForecaster(targets,
                                          lead_time,
                                          forecast_len,
                                          data_freq,
                                          features=features,
                                          update_freq=update_freq,
                                          verbose=3,
                                          hyperparams={'max_depth': 16,
                                                       'n_estimators': 20},
                                          hyperparam_search=None),
             'lightgbm_user_defined_hyperparams':
                 LightGBMForecaster(targets,
                                    lead_time,
                                    forecast_len,
                                    data_freq,
                                    features=features,
                                    update_freq=update_freq,
                                    verbose=3,
                                    hyperparams={'max_depth': 16}),
             'lightgbm_default_hyperparams':
                 LightGBMForecaster(targets,
                                    lead_time,
                                    forecast_len,
                                    data_freq,
                                    features=features,
                                    update_freq=update_freq,
                                    verbose=3)
             }

    forecasters.update(other)
    return forecasters


def define_targets():
    """
    Defines the target variables to be predicted.

    Returns:
        list: A list of dictionaries containing the target variable information.
    """
    dh_target = Target(output='dh')
    ele_target = Target(output='ele')

    return [dh_target, ele_target]


if __name__ == '__main__':
    """
    Run the train-evaluation tool with predefined parameters.

    Parameters:
        targets: A list of dictionaries containing the target variable information.
        data_freq: The frequency of the data in minutes.
        forecast_len: The length of the forecast period.
        lead_time: The lead time for the prediction.
        update_freq: The update frequency for the model.
    """
    targets = define_targets()
    data_freq = 60  # Data freq in minutes
    forecast_len = 24  # Forecast length
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


