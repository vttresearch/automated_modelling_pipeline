"""
Walk-Forward Validation Example

This example demonstrates walk-forward validation with rolling train/test windows
using both supported window evolution modes in the same script:

1) Sliding window (fixed-size training window)
2) Expanding window (growing training history)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pathlib

from amp.tools import train_eval_tool
from amp.baselines import MeanBaseline, LagBaseline
from amp.preprocessing import basic_features
from amp.efp.models.lightgbm import LightGBMForecaster
from amp.timesfm.forecaster import TimesFMForecaster
from amp.base import Target


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
    Defines basic features and various forecasters for walk-forward validation.

    Args:
        targets (list): The Target variables to be predicted.
        data_freq (int): The frequency of the data in minutes.
        forecast_len (int): The length of the forecast period.
        lead_time (int): The lead time for the prediction.
        update_freq (int): The update frequency for the model.

    Returns:
        dict: A dictionary containing the defined forecasters.
    """
    # Define basic features for LightGBM
    features = basic_features(lead_time=lead_time,
                              forecast_len=forecast_len,
                              features=['lagged_target', 'weekday', 'hour', 't_out'],
                              target_lags=[24 * 7, 24],
                              target_lag_len=24,
                              temp_lag=8)

    # Define features for TimesFM (uses longer history window)
    timesfm_features = {
        "lagged_target": {
            "windows": [(-1024, -1)],
            "type": "numeric"
        },
        "t_out": {
            "windows": [(-1024, 23)],
            "type": "numeric"
        },
        "hour": {
            "windows": [(-1024, 23)],
            "type": "categorical"
        },
        "weekday": {
            "windows": [(-1024, 23)],
            "type": "categorical"
        },
        "holiday": {
            "windows": [(-1024, 23)],
            "type": "categorical"
        }
    }

    forecasters = {
        'lightgbm': LightGBMForecaster(
            targets,
            lead_time,
            forecast_len,
            data_freq,
            features=features,
            update_freq=update_freq,
            verbose=1
        ),
    }

    return forecasters


def define_targets():
    """
    Defines the target variables to be predicted.

    Returns:
        list: A list of Target objects.
    """
    dh_target = Target(output='dh')
    ele_target = Target(output='ele')
    return [dh_target, ele_target]


def run_walk_forward(
    mode_name,
    targets,
    models,
    df,
    data_freq,
    forecast_len,
    lead_time,
    update_freq,
    baselines,
    train_window_size,
    test_window_size,
    step_size,
):
    walk_forward_params = {
        'train_window_size': train_window_size,
        'test_window_size': test_window_size,
        'step_size': step_size,
        'window_type': mode_name,
    }

    print(f"\n{'=' * 80}")
    print(f"RUNNING WALK-FORWARD MODE: {mode_name.upper()}")
    print(f"{'=' * 80}")
    print(f"Training window: {train_window_size} samples ({train_window_size / 24:.1f} days)")
    print(f"Test window: {test_window_size} samples ({test_window_size / 24:.1f} days)")
    print(f"Step size: {step_size} samples ({step_size / 24:.1f} days)")

    results = train_eval_tool(
        mode='walk_forward',
        targets=targets,
        models=models,
        df=df,
        data_freq=data_freq,
        fcast_len=forecast_len,
        lead_time=lead_time,
        update_rate=update_freq,
        baselines=baselines,
        verbose=False,
        plot=None,
        walk_forward_params=walk_forward_params,
    )

    print(f"Completed {mode_name} walk-forward")
    print(f"Best model: {results['best_model']}")
    print(f"Best avg MSE: {results['best_mse']:.4f}")
    return results


def plot_expanding_rmse(results, model_names, output_dir="results"):
    windows = results.get('all_windows', [])
    if not windows:
        print("No walk-forward window results available for plotting.")
        return

    target_names = []
    for window in windows:
        target_names.extend(window.get('target_metrics', {}).keys())
    target_names = list(dict.fromkeys(target_names))

    if not target_names:
        print("No per-target metrics available for plotting.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(target_names), 1, figsize=(12, max(4 * len(target_names), 6)), sharex=True)
    if len(target_names) == 1:
        axes = [axes]

    train_lengths = [window['train_length'] for window in windows]

    for axis, target_name in zip(axes, target_names):
        for model_name in model_names:
            rmse_values = []
            x_values = []
            for window in windows:
                target_metrics = window.get('target_metrics', {}).get(target_name, {})
                metric_values = target_metrics.get(model_name)
                if metric_values is None:
                    continue
                x_values.append(window['train_length'])
                rmse_values.append(metric_values['rmse'])

            if rmse_values:
                axis.plot(x_values, rmse_values, marker='o', label=model_name)

        axis.set_title(f"Target: {target_name}")
        axis.set_ylabel("RMSE")
        axis.grid(alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Training set length (samples)")
    fig.suptitle("Expanding walk-forward: RMSE vs training set length", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    figure_path = output_path / "walk_forward_expanding_rmse_vs_training_length.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Saved expanding walk-forward RMSE plot to: {figure_path}")


def plot_backward_expanding_rmse(results, model_names, output_dir="results"):
    """
    Plot RMSE vs training set length for backward expanding window mode.
    
    In this mode, test window is fixed and training history grows backward.
    This shows how test performance improves as more historical data is included.
    
    Args:
        results: Walk-forward results dict with 'all_windows' containing window metrics
        model_names: List of model names to plot
        output_dir: Directory to save the output plot
    """
    windows = results.get('all_windows', [])
    if not windows:
        print("No walk-forward window results available for plotting.")
        return

    # Extract unique target names from all windows
    target_names = []
    for window in windows:
        target_names.extend(window.get('target_metrics', {}).keys())
    target_names = list(dict.fromkeys(target_names))

    if not target_names:
        print("No per-target metrics available for plotting.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subplots: one per target
    fig, axes = plt.subplots(len(target_names), 1, figsize=(12, max(4 * len(target_names), 6)), sharex=True)
    if len(target_names) == 1:
        axes = [axes]

    # Plot for each target
    for axis, target_name in zip(axes, target_names):
        for model_name in model_names:
            rmse_values = []
            train_lengths = []
            
            # Collect RMSE values for each iteration (sorted by train_length since we walk backward)
            for window in windows:
                target_metrics = window.get('target_metrics', {}).get(target_name, {})
                metric_values = target_metrics.get(model_name)
                if metric_values is not None:
                    train_lengths.append(window['train_length'])
                    rmse_values.append(metric_values['rmse'])
            
            if rmse_values:
                # Sort by training length for proper x-axis ordering
                sorted_pairs = sorted(zip(train_lengths, rmse_values), key=lambda x: x[0])
                sorted_lengths, sorted_rmses = zip(*sorted_pairs)
                axis.plot(sorted_lengths, sorted_rmses, marker='o', label=model_name, linewidth=2)

        axis.set_title(f"Target: {target_name} — Test performance vs training history")
        axis.set_ylabel("RMSE on fixed test set")
        axis.grid(alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Training history length (samples)")
    fig.suptitle("Backward expanding: How performance improves with more historical training data", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    figure_path = output_path / "walk_forward_backward_expanding_rmse_vs_history.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"Saved backward expanding RMSE plot to: {figure_path}")


if __name__ == '__main__':
    """
    Run walk-forward validation.
    """
    # Define parameters
    targets = define_targets()
    data_freq = 60  # Data frequency in minutes
    forecast_len = 24  # Forecast length (24 hours)
    lead_time = 0  # Start forecasting from next step
    update_freq = 1  # Update forecast every step

    # Create forecasters
    all_models = define_forecasters(targets, data_freq, forecast_len, lead_time, update_freq)

    # Load data
    df = load_data()

    # Quick test mode for faster iteration during development
    quick_test = False
    if quick_test:
        quick_days = 70
        df = df.tail(24 * quick_days)

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Data frequency: {data_freq} minutes\n")

    # Define baselines for comparison
    baselines = {
        'mean': MeanBaseline(),
        'lag_24': LagBaseline(24),
    }

    # Define shared walk-forward sizes
    # Window sizes are in number of samples (hours, since data_freq=60 minutes)
    if quick_test:
        train_window_size = 24 * 21  # 21 days for training
        test_window_size = 24 * 3    # 3 days for testing
        step_size = 24 * 7           # Slide by 7 days
    else:
        train_window_size = 24 * 60  # 60 days for training
        test_window_size = 24 * 14   # 14 days for testing
        step_size = 24 * 30          # Slide by 30 days

    mode = "expanding_backward"  # Can also run "expanding", "expanding_backward" mode by changing this variable

    results = run_walk_forward(
        mode_name=mode,
        targets=targets,
        models=all_models,
        df=df,
        data_freq=data_freq,
        forecast_len=forecast_len,
        lead_time=lead_time,
        update_freq=update_freq,
        baselines=baselines,
        train_window_size=train_window_size,
        test_window_size=test_window_size,
        step_size=step_size,
    )

    print(f"\n{'=' * 80}")
    print("WALK-FORWARD MODE COMPARISON")
    print(f"{'=' * 80}")
    print(f"best:   {results['best_model']} (avg MSE {results['best_mse']:.4f})")

    if mode == "expanding":
        plot_expanding_rmse(results, model_names=list(all_models.keys()))
    elif mode == "expanding_backward":
        plot_backward_expanding_rmse(results, model_names=list(all_models.keys()))

