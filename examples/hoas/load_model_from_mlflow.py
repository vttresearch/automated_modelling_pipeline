"""
Example: Load and Run Inference with an MLflow-Registered Forecasting Model

This script demonstrates how to:
1. Connect to an MLflow server and locate the latest registered version of a forecasting model.
2. Load HVAC system data and prepare it for evaluation.
3. Load the registered model along with its metadata (e.g., feature names, input/output configuration).
4. Prepare an appropriate input window for the model using a chosen forecast timestamp.
5. Run predictions using both a full time series (output_mode='target') and a specific window (output_mode='single').

Requirements:
- MLflow server is running and accessible at http://localhost:5001
- A model is registered in MLflow with the name format: 'CIASEM_heating_power_forecast_model'
- Historical data is located in 'data/hoas_example_data.csv', timestamped and in UTC

Key Features:
- Automatically loads metadata from MLflow model registry (inputs, outputs, features, etc.)
- Flexible prediction windowing using model-defined lags and forecast horizon
- Demonstrates both single-window and time series-based inference



"""


import os
import mlflow
import pandas as pd
import pathlib
import json
import datetime
import pytz
from dateutil import parser
from amp.utils import set_random_seed, train_test_split
from amp.mlflow_utils import ForecasterWrapper
from amp.utils import floor_time

# Configuration
version = 1
project_name = 'CIASEM'
task_name = 'ele_dh_power_forecast_timesfm'
experiment_name = f"{project_name}/{task_name}_v{version}"
model_name = f"{project_name}_{task_name}_model"
mlflow_uri = os.environ.get('AMP_MLFLOW_URI', 'http://localhost:5000')
#mlflow_uri = 'http://0.0.0.0:5002'
mlflow.set_tracking_uri(mlflow_uri)  # Your MLflow server

# Load latest registered model
registered_models = mlflow.search_registered_models(filter_string=f"name='{model_name}'")
if not registered_models:
    raise ValueError(f"No registered model found with name: {model_name}")

latest_version = registered_models[0].latest_versions[0].version
model_uri = f"models:/{model_name}/{latest_version}"
print(f"Model URI: {model_uri}")

# Load data
df = pd.read_csv(pathlib.Path('data/hoas_example_data.csv'), index_col='timestamp', parse_dates=True)
df.index = df.index.tz_localize(pytz.UTC)

# Split data
test_period = ('2020-01-01', '2020-03-30')
_, test_df = train_test_split(df, test_period=test_period)

# Load model with metadata
loaded_model = ForecasterWrapper.load_model_with_metadata(model_uri)

# Optional: Access metadata directly
metadata = loaded_model.metadata.get_model_info().metadata
features = json.loads(metadata["features"])
inputs = json.loads(metadata["inputs"])
feature_types = list(inputs.keys())
outputs = json.loads(metadata["outputs"])
data_freq =  json.loads(metadata["data_freq"])
input_window = json.loads(metadata["input_window"])

# Alternative: Access metadata via attributes
print(f'Model has inputs: {loaded_model.inputs} and outputs: {loaded_model.outputs}')

# Prepare test input for prediction
max_lag, max_future = loaded_model.input_window


# Choose a prediction time from test set
now = floor_time(test_df.index[max_lag], data_freq)

input_start = now - datetime.timedelta(minutes=abs(max_lag) * data_freq)
input_end = now + datetime.timedelta(minutes=(abs(max_future)) * data_freq)

# Select the needed input data
input_df = test_df.loc[input_start:input_end, feature_types]


# Run predictions
predicted_target = loaded_model.predict(test_df, params={'output_mode': 'target'})
predicted_window = loaded_model.predict(input_df, params={'output_mode': 'single'})
pass
