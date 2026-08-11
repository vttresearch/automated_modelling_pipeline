# -*- coding: utf-8 -*-
"""
sklearn.py documentation:

Project: energia
File name: sklearn.py 
Author: Janne Takalo-Mattila
Organization: VTT Technical Research Centre of Finland
Date created: 10.05.2023
"""

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from amp.efp.decorators import parameter_search_decorator
from amp.efp.pipe_processing import preprocessor_pipe
from amp.efp.base import Forecaster


#
# Forecasters
## Common Forecaster Base Class
class BaseForecaster(Forecaster):
    def __init__(self, targets, lead_time, forecast_len, data_freq, model_cls, features=None,
                 update_freq=1, upper_limits=None, lower_limits=None, **model_params):
        self.model_cls = model_cls
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
        regressor_params = hyperparams.get('hyperparams', {})
        full_pipe = Pipeline([
            ('preprocessing', preprocessor_pipe(features)),
            ('model', self.model_cls(**regressor_params))
        ])
        return full_pipe


# Helper functions to replace lambda functions
def create_linear_model(**params):
    return MultiOutputRegressor(LinearRegression(**params))


def create_mlp_model(**params):
    return MultiOutputRegressor(MLPRegressor(**params))


def create_svr_model(**params):
    return MultiOutputRegressor(SVR(**params))


# Specific Forecaster Implementations
class LinearForecaster(BaseForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_cls=create_linear_model, **kwargs)


# TODO: For some reason this class does not accept some hyperparameters, like max_iter. The fit and evaluation run always with defaults.
# What is the difference between this and the next class?
class MLPForecaster(BaseForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_cls=MLPRegressor, **kwargs)


class MLPMultiForecaster(BaseForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_cls=create_mlp_model, **kwargs)


class SVRForecaster(BaseForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_cls=create_svr_model, **kwargs)


class RandomForestForecaster(BaseForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_cls=RandomForestRegressor, **kwargs)


class DecisionTreeForecaster(BaseForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_cls=DecisionTreeRegressor, **kwargs)