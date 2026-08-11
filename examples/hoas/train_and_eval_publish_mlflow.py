"""
Example: Train and Evaluate Forecasting Model Using AMP Framework with MLflow Integration

This script demonstrates a full pipeline for training and evaluating a multivariate forecasting model
for district heating ('dh') and electricity ('ele') demand using the AMP (Automated Model Pipeline) framework.
It uses a LightGBM-based forecaster and includes baseline comparisons. The trained model is tracked via MLflow.

Steps covered:
1. Define forecasting parameters including horizon, frequency, and targets.
2. Load historical building-level data from a CSV file.
3. Specify engineered features using AMP's built-in `basic_features`.
4. Train a `LightGBMForecaster` using the AMP model dictionary abstraction.
5. Set up and evaluate baseline models (mean and lag-based).
6. Log all metrics, plots, and models to a local MLflow server.

Requirements:
- AMP framework installed (custom Python forecasting package)
- MLflow server running locally at http://localhost:5001
- Dataset available at: data/hoas_example_data.csv with UTC-timestamped entries

To run an MLflow server:
    $ conda activate your-env
    $ cd /path/to/mlflow_server
    $ mlflow server --host 0.0.0.0 --port 5001
    → Access MLflow UI at http://localhost:5001

After training:
- Model is saved locally under 'trained_models/'
- Evaluation results are logged under the MLflow experiment "CIASEM/heating_power_forecast"
- Use a separate script (e.g. `load_model_from_mlflow.py`) for inference

"""

import os
import pandas as pd
import pathlib
from amp.base import Target
from amp.preprocessing import basic_features
from amp.efp.models.lightgbm import LightGBMForecaster
from amp.tools import train_eval_tool
from amp.baselines import MeanBaseline, LagBaseline

""" 
Define parameters
"""

data_freq = 60  # Data freq in minutes
forecast_len = 24  # Forecast length
lead_time = 0  # Start forecast from current time
update_freq = 1  # Update forecast in every period
test_period = ('2020-01-01', '2020-03-30')
mlflow_uri = os.environ.get('AMP_MLFLOW_URI', 'http://localhost:5000')
#mlflow_uri = 'http://0.0.0.0:5002'
"""
Define MLFlow parameters
Notice that MLFlow server needs to be running:
1. Install MLFlow to conda environment, activate conda environment 
2. Run:
cd /Users/jtmjanne/Documents/projects/tools/mlflow_server
mlflow server --host 0.0.0.0 --port 5001
3. http://localhost:5001 view mlruns

"""

project_name = 'CIASEM'
task_name = 'ele_dh_power_forecast'
version = 1
experiment_name = f"{project_name}/{task_name}_v{version}"
model_name = project_name + "_" + task_name + '_model'

"""

Load dataset

"""
df = pd.read_csv(pathlib.Path('data/hoas_example_data.csv'), index_col='timestamp', parse_dates=True)


"""

Define targets to be forecasted. Outputs define model outputs and those needs to be found from dataset

"""

targets = [Target(output='dh'), Target(output='ele')]


"""

Define features that are used in forecasting

In this example:
features = {'lagged_target': {'windows': [(-168, -1)], 'type': 'numeric'}, 
            'weekday': {'windows': [(0, 23)], 'type': 'onehot'}, 
            'hour': {'windows': [(0, 23)], 'type': 'onehot'}, 
            't_out': {'windows': [(-8, 23)], 'type': 'numeric'}, 
            'holiday': {'windows': [(0, 23)], 'type': 'onehot'}
            }


"""

features = basic_features(lead_time=lead_time,
                              forecast_len=forecast_len,
                              features=['lagged_target', 'weekday', 'hour', 't_out', 'holiday', 'month'],
                              target_lags=[168, 24],
                              target_lag_len=24,
                              temp_lag=16)


"""

Create a forecaster 

In this example:
model_dict = {'lightgbm_forecaster': <amp.efp.models.lightgbm.LightGBMForecaster>}

"""

model_dict = {'lightgbm_forecaster':
              LightGBMForecaster(targets=targets,
                                 lead_time=lead_time,
                                 forecast_len=forecast_len,
                                 data_freq=data_freq,
                                 features=features,
                                 update_freq=update_freq)}

"""

Create baselines for comparison

"""

baselines = {'mean': MeanBaseline(),
             'lag_4': LagBaseline(4),
             'lag_24': LagBaseline(24)}

"""

Use train eval tool to fit a model
 
"""

train_eval_tool(mode='fit',
                targets=targets,
                models=model_dict,
                df=df,
                test_period=test_period,
                data_freq=data_freq,
                fcast_len=forecast_len,
                lead_time=lead_time,
                update_rate=update_freq,
                baselines=baselines,
                trained_folder='trained_models')

"""
Use train_eval_tool to evaluate model
"""


train_eval_tool(mode='eval',
                targets=targets,
                models=model_dict,
                df=df,
                test_period=test_period,
                data_freq=data_freq,
                fcast_len=forecast_len,
                lead_time=lead_time,
                update_rate=update_freq,
                baselines=baselines,
                plot='best',
                mlflow_experiment=experiment_name,
                mlflow_uri=mlflow_uri,
                mlflow_model_name=model_name,
                mlflow_registry_update_mode="always")

# To load the model, check example load_model_from_mlflow.py
