# -*- coding: utf-8 -*-
"""
models.py documentation:

Forecaster models for spesific algorithms


Project: energia
File name: models.py 
Author: Janne Takalo-Mattila
Organization: VTT Technical Research Centre of Finland
Date created: 09.05.2023
"""
import lightgbm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from amp.efp.base import Forecaster
from amp.efp.pipe_processing import preprocessor_pipe
from amp.efp.decorators import parameter_search_decorator



class LightGBMForecaster(Forecaster):
    def __init__(self, targets, lead_time, forecast_len, data_freq,
                 features=None, update_freq=1, upper_limits=None, lower_limits=None, **model_params):
        super().__init__(
            targets=targets,
            lead_time=lead_time,
            forecast_len=forecast_len,
            data_freq=data_freq,
            model=self._pipeline,
            features=features,
            hyperparams=model_params,
            update_freq=update_freq,
            upper_limits=upper_limits,
            lower_limits=lower_limits
        )

    @parameter_search_decorator
    def _pipeline(self, forecast_len, features, **hyperparams):
        # Check if 'hyperparams' key exists in the hyperparams dictionary
        if 'hyperparams' in hyperparams:
            regressor_params = hyperparams['hyperparams']
        else:
            regressor_params = {}

        # Suppress LightGBM C++ core output unless user explicitly sets verbosity
        regressor_params.setdefault('verbose', -1)

        regressor = MultiOutputRegressor(
            lightgbm.LGBMRegressor(**regressor_params)) if forecast_len > 1 else lightgbm.LGBMRegressor(**regressor_params)
        full_pipe = Pipeline([
            ('preprocessing', preprocessor_pipe(features)),
            ('model', regressor)
        ])
        return full_pipe

