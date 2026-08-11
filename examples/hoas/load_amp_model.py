"""
Example: Loading and Using a Trained HVAC Forecasting Model

This script demonstrates how to load a trained machine learning model (LightGBM in this case),
prepare input data, and run predictions using historical HVAC system data.

Key Steps:
1. Load historical data from a CSV file.
2. Split the dataset to isolate a test period.
3. Load a pre-trained model from disk.
4. Prepare the input window for a given forecast timestamp.
5. Predict both the full target time window and a single output point.

Requirements:
- Data is stored in 'data/hoas_example_data.csv', indexed by UTC timestamps.
- Model is stored in 'trained_models/lightgbm_default_hyperparams.zip'.
- Prediction is performed using features the model was trained with.
"""



import pandas as pd
import pathlib
import datetime
import pytz
from amp.base import load_model
from amp.utils import train_test_split
from amp.utils import floor_time

# Config
model_directory  = 'trained_models/'
model_name = 'lightgbm_default_hyperparams.zip'
data_freq = 60

# Load data
df = pd.read_csv(pathlib.Path('data/hoas_example_data.csv'), index_col='timestamp', parse_dates=True)
df.index = df.index.tz_localize(pytz.UTC)

# Split data
test_period = ('2020-01-01', '2020-03-30')
_, test_df = train_test_split(df, test_period=test_period)


loaded_model = load_model(model_directory+model_name)
# Prepare test input for prediction
feature_types = loaded_model.feature_types
max_lag, max_future = loaded_model.input_window

# Choose a prediction time from test set
start_point = floor_time(test_df.index[max_lag], data_freq)

input_start = start_point - datetime.timedelta(minutes=abs(max_lag) * data_freq)
input_end = start_point + datetime.timedelta(minutes=(abs(max_future)) * data_freq) # in hvac.py there is +1 in equation, why?

# Select the needed input data
input_df = test_df.loc[input_start:input_end, feature_types]

# Run predictions
predicted_target = loaded_model.predict(test_df, output_mode='target')
predicted_window = loaded_model.predict(input_df, output_mode='single')
print("\nPredictions DataFrame")
print(predicted_window)

