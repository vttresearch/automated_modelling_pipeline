from amp import pipeline
import pandas as pd
import pytest
from copy import deepcopy
from amp.pipeline import Pipeline
from amp.base import BaseModel

def test_forecast_feature():
    # Test with default parameters
    expected_default = {
        'model1_dh_forecast_0': {'windows': [(0, 0)], 'type': 'numeric'},
        'model1_dh_forecast_1': {'windows': [(1, 1)], 'type': 'numeric'},
        'model1_dh_forecast_2': {'windows': [(2, 2)], 'type': 'numeric'},
        'model1_ele_forecast_0': {'windows': [(0, 0)], 'type': 'numeric'},
        'model1_ele_forecast_1': {'windows': [(1, 1)], 'type': 'numeric'},
        'model1_ele_forecast_2': {'windows': [(2, 2)], 'type': 'numeric'}
    }
    assert pipeline.forecast_feature('model1', ['dh', 'ele'], lead_time=0, forecast_len=3) == expected_default



