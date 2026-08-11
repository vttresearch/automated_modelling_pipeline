import mlflow
import joblib
import pandas as pd
from amp.efp.base import Forecaster
import numpy as np
from amp.efp.base import Forecaster  # Assuming your custom Forecaster class
from amp.constants import METRIC_NAME_TARGET_TOTAL_MSE
from amp.utils import calculate_metrics
import os
import zipfile
import json
import joblib
import datetime
import pandas as pd


def _input_df_generation(model, df):
    # Prepare test input for prediction
    feature_types = model.feature_types
    max_lag, max_future = model.input_window
    most_common_diff = df.index.to_series().diff().mode()[0]

    dominant_freq_minutes = int(most_common_diff.total_seconds() // 60)

    input_start = df.index[0]
    input_end = (input_start + datetime.timedelta(minutes=(abs(max_lag)) * dominant_freq_minutes) +
                 datetime.timedelta(minutes=(abs(max_future)) * dominant_freq_minutes))

    # Select the needed input data
    input_df = df.loc[input_start:input_end, feature_types]
    return input_df


class ForecasterWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, forecaster):
        self.forecaster = forecaster

    def predict(self, context, model_input, params=None):
        params = params or {"output_mode": "target"}
        output_mode = params.get("output_mode", "target")
        return self.forecaster.predict(model_input, output_mode=output_mode)



    @classmethod
    def save_model(cls, input, forecaster):
        """
        Save the forecaster model to a specific path.
        This method is compatible with MLflow’s `log_model`.
        """

        # Define the signature associated with the model
        params_example = {'output_mode': "single"}
        input_df_windowed = _input_df_generation(forecaster, input)
        output_example = forecaster.predict(input_df_windowed, output_mode='single')
        signature = mlflow.models.infer_signature(input_df_windowed, output_example, params=params_example)
        input_example = (input, params_example)

        f_details = _forecaster_details_dictionary(forecaster)
        metadata = f_details

        # Log the model to MLflow
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "project_model.zip")
            forecaster.save(zip_path)

            artifacts = {"project_model_zip": zip_path}

            # Log model
            mlflow.pyfunc.log_model(
                artifact_path="project_model",
                python_model=cls(forecaster),  # forecaster will be loaded from zip
                artifacts=artifacts,
                signature=signature,
                metadata={k: json.dumps(v) for k, v in metadata.items()}
            )



    @staticmethod
    def load_model_with_metadata(model_uri):
        """
        Loading the model
        """
        # Load the model using MLflow's pyfunc API
        model = mlflow.pyfunc.load_model(model_uri)
        metadata = model.metadata.get_model_info().metadata

        # Attach all metadata entries as attributes
        for k, v in metadata.items():
            try:
                setattr(model, k, json.loads(v))
            except (TypeError, json.JSONDecodeError):
                setattr(model, k, v)

        return model


def log_model_metrics_and_params(best_model, targets):
    """
    Logs model performance metrics and parameters to MLflow.

    Parameters:
    -----------
    best_model : dict
        Dictionary containing the best model's name and instance.
    targets : list
        List of target dictionaries containing actual values and computed errors.
    forecast_len : int
        Forecast horizon length.
    lead_time : int
        Lead time of the forecast.
    update_rate : int
        How often the model updates predictions.

    Returns:
    --------
    None
    """
    print('Logging current run to MLFlow')
    best_model_name = best_model['name']
    model_instance = best_model['model']
    total_target_mse = 0
    for t in targets:
        target_calc = t[best_model_name][t["column"]]
        target_mse = t[f"{best_model_name}_mse"]

        # Calculate performance metrics
        rmse, nrmse1, nrmse2 = calculate_metrics(target_mse, target_calc)

        # Log metrics to MLflow
        mlflow.log_metric(f"{t['output']}_total_mse", target_mse)
        mlflow.log_metric(f"{t['output']}_total_rmse", rmse)
        mlflow.log_metric(f"{t['output']}_total_nrmse1", nrmse1)
        mlflow.log_metric(f"{t['output']}_total_nrmse2", nrmse2)
        total_target_mse += target_mse
    mlflow.log_metric(METRIC_NAME_TARGET_TOTAL_MSE, total_target_mse)
    # Log model parameters
    inputs = _input_features(model_instance)

    f_details = _forecaster_details_dictionary(model_instance)

    mlflow.log_params(f_details)


def best_existing_model(mlflow_experiment):
    experiment = mlflow.get_experiment_by_name(mlflow_experiment)

    if experiment:
        experiment_id = experiment.experiment_id

        # Search for the best model by lowest target_total_mse within the specific experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{METRIC_NAME_TARGET_TOTAL_MSE} ASC"]
        )

        if not runs.empty:
            best_run = runs.iloc[0]  # Select the row with the lowest MSE Extract run_id
            return best_run  # Return both ID and full row

    return None  # No valid experiment or runs found

def register_model(model_name="ForecasterModel"):
    """Register model in MLflow Model Registry"""
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/project_model"
    registered_model = mlflow.register_model(model_uri, model_name)
    print(f"Model registered with version: {registered_model.version}")
    return registered_model


def print_run_summary(run):
    """Prints a structured summary of the MLflow run, including metrics and parameters."""

    print("=" * 50)
    print(f"Summary current best MLflow Run (ID: {run['run_id']})")
    print("=" * 50)
    print(f"Experiment ID: {run['experiment_id']}")
    print(f"Model: {run.get('params.Model_name', 'Unknown')}")
    print(f"Artifact URI: {run['artifact_uri']}")
    print(f"Run name in MLFlow: {run.get('tags.mlflow.runName', 'Unknown')}")
    print("=" * 50)

    # Print metrics
    print("Metrics:")
    for key, value in run.items():
        if key.startswith("metrics."):
            metric_name = key.replace("metrics.", "")
            print(f"  {metric_name}: {value}")

    print("=" * 50)
    # Print parameters
    print("Parameters:")
    for key, value in run.items():
        if key.startswith("params."):
            param_name = key.replace("params.", "")
            print(f"  {param_name}: {value}")
    print("=" * 50)


def _input_features(forecaster):
    # Generate input description per each input that is required
    inputs = {}
    for feature in forecaster.feature_types:
        if (feature in forecaster.outputs) & ('lagged_target' in forecaster.features.keys()):
            # Use lagged_target window for outputs
            inputs[feature] = {
                'type': forecaster.features['lagged_target']['type'],
                'windows': forecaster.features['lagged_target']['windows']
            }
        else:
            # Use feature-specific window
            inputs[feature] = forecaster.features.get(feature, {'type': 'unknown', 'windows': []})
    return inputs


def _forecaster_details_dictionary(forecaster):
    inputs = _input_features(forecaster)
    return {
        "inputs": inputs,
        "input_window": forecaster.input_window,
        "feature_types": forecaster.feature_types,
        "data_freq": forecaster.data_freq,
        "forecast_length": forecaster.forecast_len,
        "lead_time": forecaster.lead_time,
        "update_rate": forecaster.update_rate,
        "model_type": str(forecaster),  # Convert model instance to string to avoid errors
        "features": str(forecaster.features),
        "outputs": str(forecaster.outputs),
    }


def log_dataset_context(train_data, test_data, data_freq, dataset_info):
    """
    Logs dataset context information to MLflow, including training and testing data characteristics and data frequency.
    Parameters
    ----------
    train_data
    test_data
    data_freq

    Returns
    -------

    """



    # Handle list or single df
    if isinstance(train_data, list):
        train_concat = pd.concat(train_data)
    else:
        train_concat = train_data

    if isinstance(test_data, list):
        test_concat = pd.concat(test_data)
    else:
        test_concat = test_data

    # ---- TRAIN DATA ----
    mlflow.log_param("train_start", str(train_concat.index.min()))
    mlflow.log_param("train_end", str(train_concat.index.max()))
    mlflow.log_param("train_samples", len(train_concat))

    train_duration = (train_concat.index.max() - train_concat.index.min()).total_seconds() / 3600
    mlflow.log_param("train_duration_hours", train_duration)

    # ---- TEST DATA ----
    mlflow.log_param("test_start", str(test_concat.index.min()))
    mlflow.log_param("test_end", str(test_concat.index.max()))
    mlflow.log_param("test_samples", len(test_concat))

    # ---- DATA RESOLUTION ----
    mlflow.log_param("data_freq_minutes", data_freq)

    # ---- ADDITIONAL DATASET INFO ----
    if dataset_info:
        for key, value in dataset_info.items():
            if isinstance(value, (list, dict)):
                mlflow.log_param(key, str(value))  # MLflow needs strings
            else:
                mlflow.log_param(key, value)