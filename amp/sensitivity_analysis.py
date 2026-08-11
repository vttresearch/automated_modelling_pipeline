""" Tools for performing sensitivity analysis on models"""
from amp.base import BaseModel
from amp.dataloader import create_samples
from typing import Dict, Optional, Tuple, Union, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def sweeping_analysis(
    model: BaseModel,
    input_data: Union[pd.DataFrame, List[pd.DataFrame]],
    param_ranges: Dict[str, Tuple[float, float]],
    num_sweep_points: int = 20,
    num_samples: int = 5,
    sample_indices: Optional[list] = None,
    save_path: Optional[str] = None,
    figsize_per_sample: Tuple[float, float] = (4, 4),
    random_seed: Optional[int] =None,
    context_features: Optional[List[str]] = None,
    data_freq: Optional[int] = None,
    horizon_len: Optional[int] = None,
    other_features: Optional[List[str]] = None
):
    """
    Perform sweeping sensitivity analysis on an AMP BaseModel by varying controllable parameters.
    
    This function sweeps each parameter defined in param_ranges from its minimum to maximum value,
    while keeping other features at their values from the input DataFrame. For each parameter sweep,
    predictions are made across multiple samples and visualized in a multi-panel plot.
    
    Parameters
    ----------
    model : BaseModel
        The trained AMP model to analyze. Must implement the predict() method.
    input_data : pd.DataFrame or list of pd.DataFrame
        Input DataFrame(s) containing all features needed for prediction. The DataFrame should have
        a datetime index and include sufficient historical data for the model's input window.
        If a list is provided, samples are randomly selected from all DataFrames.
    param_ranges : dict
        Dictionary mapping parameter names (column names in input_data) to (min, max) tuples
        defining the sweep range. Example: {'setpoint_temp': (18.0, 24.0), 'flow_rate': (0.5, 2.0)}
    num_sweep_points : int, optional
        Number of discrete points to evaluate in each parameter sweep (default: 20).
    num_samples : int, optional
        Number of time samples to test from input_data (default: 5).
        If input_data is a list, samples are randomly selected across all DataFrames.
    sample_indices : list of int, optional
        Explicit list of sample indices to use. If None, samples are automatically selected
        randomly from the available data. Ignored if input_data is a list of DataFrames.
    save_path : str, optional
        Path to save the sensitivity plots. If None, plots are only displayed.
        Each parameter sweep creates a separate file: {save_path}_<param_name>.png
    figsize_per_sample : tuple of (width, height), optional
        Figure size per sample column in inches (default: (12, 4)).
        Total figure size scales with num_samples.
    random_seed : int, optional
        Random seed for reproducible sample selection (default: 42).
    context_features : list of str, optional
        List of feature names to display as context (non-swept features) in the plot.
        These features will be shown in additional rows above the target predictions.
        If None, all available non-swept, non-target features are displayed.
        Use an empty list [] to hide context features.
    other_features : list of str, optional
        List of feature names to be normalized and plotted together on a single row
        positioned first in the figure. All features in this list will be normalized
        to [0, 1] range and overlaid on the same subplot for easy comparison.
        If None, no normalized feature row is added.
    
    Returns
    -------
    None
        Generates and displays (and optionally saves) sensitivity plots showing how
        model predictions vary with each parameter across different time samples.
    
    Notes
    -----
    - The model is expected to handle normalization/denormalization internally
    - Each prediction uses output_mode='single' on the model's predict() method
    - The model's forecast horizon and input window are automatically determined
      from the model's configuration (lead_time and fcast_len)
    - For each sample, the function creates a prediction by varying only the swept
      parameter while keeping all other features at their original DataFrame values
    - When input_data is a list, samples are selected randomly across all periods
    
    Examples
    --------
    >>> from amp.base import load_model
    >>> model = load_model('path/to/model.pkl')
    >>> data = pd.read_csv('data.csv', index_col=0, parse_dates=True)
    >>> param_ranges = {
    ...     'setpoint_temperature': (18.0, 24.0),
    ...     'supply_flow_rate': (0.5, 2.0)
    ... }
    >>> sweeping_analysis(
    ...     model=model,
    ...     input_data=data,
    ...     param_ranges=param_ranges,
    ...     num_sweep_points=30,
    ...     num_samples=3,
    ...     save_path='./results/sensitivity'
    ... )
    
    >>> # Or with list of DataFrames
    >>> data_list = [winter_2023_df, winter_2024_df]
    >>> sweeping_analysis(
    ...     model=model,
    ...     input_data=data_list,
    ...     param_ranges=param_ranges,
    ...     num_samples=5
    ... )
    """
    
    # Validate inputs
    if not hasattr(model, 'predict'):
        raise AttributeError("Model must have a predict() method")
    
    # Normalize input to list
    data_list = input_data if isinstance(input_data, list) else [input_data]
    
    # Validate all DataFrames have required columns
    for i, df in enumerate(data_list):
        for param_name in param_ranges.keys():
            if param_name not in df.columns:
                raise ValueError(f"Parameter '{param_name}' not found in DataFrame {i} columns")
    
    logger.info("=" * 80)
    logger.info("SWEEPING SENSITIVITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Input: {len(data_list)} DataFrame(s)")
    logger.info(f"Parameters to sweep: {list(param_ranges.keys())}")
    logger.info(f"Number of sweep points: {num_sweep_points}")
    logger.info(f"Number of samples: {num_samples}")
    
    # Get model configuration
    lead_time = model.lead_time
    fcast_len = model.forecast_len
    
    # Determine input window size
    # input_window is a tuple (lag_offset, future_offset) where:
    # - lag_offset is negative (e.g., -168 for 168 steps back)
    # - future_offset is non-negative (e.g., 23 means from step 0 to step 23, inclusive)
    input_window = list(model.input_window)
    input_window[1] = horizon_len - 1 if horizon_len is not None else input_window[1]
    lag_offset = input_window[0]
    future_offset = input_window[1]

    # Modify the models input window by updating the underlying buffer (if it's a PyTorch model)
    # For PyTorch models, input_window is a read-only property that reads from _input_window buffer
    if horizon_len is not None:
        if hasattr(model, '_input_window'):
            # PyTorch model with buffer
            import torch
            model._input_window = torch.tensor(input_window)

    # Calculate lengths as positive values
    lag_length = abs(lag_offset) if lag_offset < 0 else 0  # Length of historical data needed
    future_length = future_offset + 1 if future_offset >= 0 else 0  # Length of future data (inclusive)
    total_window_length = lag_length + future_length  # Total sample length
    
    # Calculate where forecast starts within the sample window
    # Forecast starts after historical data + lead time
    forecast_start_idx = lag_length + lead_time
    
    logger.info(f"Model configuration:")
    logger.info(f"  Lead time: {lead_time} steps")
    logger.info(f"  Forecast length: {fcast_len} steps")
    logger.info(f"  Input window: {input_window} (lag_offset, future_offset)")
    logger.info(f"  Lag length: {lag_length} steps")
    logger.info(f"  Future length: {future_length} steps (future_offset + 1)")
    logger.info(f"  Total sample window length: {total_window_length} steps")
    logger.info(f"  Forecast starts at index: {forecast_start_idx} (within sample)")
    
    # Get target names from model
    target_names = [t for t in model.outputs]
    
    # Get required features from model
    required_features = model.feature_types
    logger.info(f"Model requires {len(required_features)} features: {required_features}")
    
    # Concatenate DataFrames
    combined_df = pd.concat(data_list, axis=0) if len(data_list) > 1 else data_list[0]
    
    # Filter DataFrame to only include required features and targets
    required_columns = list(set(required_features + target_names))
    available_columns = [col for col in required_columns if col in combined_df.columns]
    
    if len(available_columns) < len(required_columns):
        missing = set(required_columns) - set(available_columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    filtered_df = combined_df[available_columns].copy()
    logger.info(f"Using {len(available_columns)} columns from data")
    
    # Create samples using create_samples function
    if data_freq is None:
        raise ValueError("data_freq is required. Provide the data frequency in minutes.")
    
    logger.info(f"Using data frequency: {data_freq} minutes")
    
    # Create samples with only required features
    samples = create_samples(
        df=filtered_df,
        features=required_features,
        targets=target_names,
        input_window=input_window,
        output_format='list',
        stride=1,
        normalize=False,
        resample_freq=data_freq,
        period_name='sensitivity_analysis'
    )
    
    if len(samples) == 0:
        raise ValueError(
            f"No valid samples found. Data too short or contains gaps. "
            f"Need at least {total_window_length} timesteps."
        )
    
    logger.info(f"Created {len(samples)} valid samples from data")
    
    # Select samples
    if sample_indices is None:
        # Randomly select samples
        if random_seed is not None:
            random.seed(random_seed)
            
        num_samples_to_select = min(num_samples, len(samples))
        selected_sample_indices = random.sample(range(len(samples)), num_samples_to_select)
        selected_sample_indices.sort()
        logger.info(f"Randomly selected {num_samples_to_select} samples from {len(samples)} valid positions")
    else:
        # Use provided sample_indices
        for idx in sample_indices:
            if idx < 0 or idx >= len(samples):
                raise ValueError(
                    f"Sample index {idx} invalid. Must be in range [0, {len(samples)-1}]"
                )
        selected_sample_indices = sample_indices
        logger.info(f"Using provided {len(sample_indices)} sample indices")
    
    selected_samples = [samples[i] for i in selected_sample_indices]
    sample_labels = [f"Sample_{i}" for i in selected_sample_indices]
    
    logger.info(f"Selected {len(selected_samples)} samples for analysis")
    
    # Perform sensitivity sweep for each parameter
    for param_name, (param_min, param_max) in param_ranges.items():
        logger.info("=" * 80)
        logger.info(f"SWEEPING PARAMETER: {param_name}")
        logger.info(f"  Range: [{param_min}, {param_max}]")
        logger.info("=" * 80)
        
        # Create sweep values
        sweep_values = np.linspace(param_min, param_max, num_sweep_points)
        
        # Store predictions for all samples
        all_sample_predictions = []
        all_sample_data = []
        
        # Process each sample
        for i, sample in enumerate(selected_samples):
            sample_label = sample_labels[i]
            logger.info(f"Processing {sample_label}...")
            
            # Extract sample data from the sample dictionary
            sample_data = sample['data'].copy()
            
            # Store predictions for this sample across all sweep values
            sample_predictions = []
            
            for j, sweep_val in enumerate(sweep_values):
                # Create modified data with swept parameter value
                modified_data = sample_data.copy()
                
                # Set the parameter to the sweep value ONLY in the forecast horizon
                # Keep historical and lead time data unchanged to ensure all sweep cases 
                # start from the same initial conditions
                modified_data.loc[modified_data.index[forecast_start_idx:], param_name] = sweep_val
                
                # Make prediction
                pred_result = model.predict(modified_data, output_mode='single')
                
                # Store prediction (pred_result is a DataFrame)
                sample_predictions.append(pred_result)
            
            all_sample_predictions.append(sample_predictions)
            all_sample_data.append(sample_data)
        
        # Create visualization
        _plot_sensitivity_sweep(
            model=model,
            input_window=input_window,
            param_name=param_name,
            sweep_values=sweep_values,
            all_sample_predictions=all_sample_predictions,
            all_sample_data=all_sample_data,
            sample_indices=sample_labels,
            target_names=target_names,
            save_path=save_path,
            figsize_per_sample=figsize_per_sample,
            context_features=context_features,
            other_features=other_features
        )
    
    logger.info("=" * 80)
    logger.info("SENSITIVITY ANALYSIS COMPLETE")
    logger.info("=" * 80)


def _plot_sensitivity_sweep(
    model: BaseModel,
    input_window: Tuple[int, int],
    param_name: str,
    sweep_values: np.ndarray,
    all_sample_predictions: list,
    all_sample_data: list,
    sample_indices: list,
    target_names: list,
    save_path: Optional[str],
    figsize_per_sample: Tuple[float, float],
    context_features: Optional[List[str]] = None,
    other_features: Optional[List[str]] = None
):
    """
    Create sensitivity sweep visualization showing how predictions vary with parameter changes.
    
    Parameters
    ----------
    model : BaseModel
        The model being analyzed
    param_name : str
        Name of the parameter being swept
    sweep_values : np.ndarray
        Array of parameter values tested
    all_sample_predictions : list
        List of prediction lists for each sample. Structure: [sample][sweep_point] -> DataFrame
    all_sample_data : list
        List of DataFrames containing the original data for each sample
    sample_indices : list
        Original indices of samples in the input data
    target_names : list
        Names of model output targets
    save_path : str or None
        Base path for saving plots
    figsize_per_sample : tuple
        Figure size per sample column
    context_features : list of str or None
        List of feature names to display as context. If None, all non-swept, non-target
        features are displayed. If empty list, no context features are shown.
    other_features : list of str or None
        List of feature names to be normalized and plotted together on a single row
        positioned first in the figure. If None, no normalized feature row is added.
    """
    num_samples = len(sample_indices)
    num_targets = len(target_names)
    
    # Identify non-swept features (features that are not being swept)
    # Get all features from the first sample data
    all_features = all_sample_data[0].columns.tolist()
    # Remove the swept parameter and target features
    non_swept_features = [f for f in all_features if f != param_name and f not in target_names]
    
    # Validate and prepare other_features (normalized features row)
    validated_other_features = []
    if other_features:
        validated_other_features = [f for f in other_features if f in all_features]
        if validated_other_features:
            logger.info(f"Will plot {len(validated_other_features)} normalized features: {validated_other_features}")
    
    # Determine which features to display as context
    if context_features is None:
        # Use all non-swept features, but exclude features already in other_features
        display_features = [f for f in non_swept_features if f not in validated_other_features]
    elif len(context_features) == 0:
        # Explicitly empty - show no context features
        display_features = []
    else:
        # Use specified features, validated and excluding duplicates
        display_features = [f for f in context_features 
                          if f in all_features and f not in validated_other_features]
    
    logger.info(f"Displaying {len(display_features)} context features")
    
    # Calculate number of rows: other_features_row (0 or 1) + context features + targets
    num_other_features_rows = 1 if validated_other_features else 0
    num_rows = num_other_features_rows + len(display_features) + num_targets
    num_cols = num_samples
    
    # Create figure with shared axes
    # sharex='col' means all subplots in a column share the same x-axis
    # sharey='row' means all subplots in a row share the same y-axis
    fig_width = figsize_per_sample[0] * num_cols
    fig_height = figsize_per_sample[1] * num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height),
                             sharex='col', sharey='row')
    
    # Handle single row or single column cases
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot all sweep values
    num_plot_lines = len(sweep_values)
    plot_indices = np.arange(len(sweep_values))
    colors = plt.cm.viridis(np.linspace(0, 1, num_plot_lines))
    
    logger.info(f"Plotting all {num_plot_lines} sweep lines")
    
    # Get model configuration for extracting forecast window
    lead_time = model.lead_time
    fcast_len = model.forecast_len
    lag_offset = input_window[0]
    future_offset = input_window[1]
    
    # Calculate where forecast starts within the sample window
    lag_length = abs(lag_offset) if lag_offset < 0 else 0
    forecast_start_idx = lag_length + lead_time
    
    # Plot for each sample
    for col_idx, (sample_idx, sample_predictions, sample_data) in enumerate(
        zip(sample_indices, all_sample_predictions, all_sample_data)
    ):
        row_idx = 0
        
        # Get the time indices for the entire sample window (historical + lead + forecast)
        # This shows the full context and makes it clear all predictions start from same point
        full_time_indices = sample_data.index
        forecast_time_indices = sample_data.index[forecast_start_idx:forecast_start_idx + fcast_len]
        
        # First row: Normalized other_features (if specified)
        if validated_other_features:
            ax = axes[row_idx, col_idx]
            
            # Plot each feature in other_features, normalized to [0, 1]
            for i, feature_name in enumerate(validated_other_features):
                # Get feature values from the entire sample window
                feature_values = sample_data[feature_name].values
                
                # Normalize to [0, 1]
                min_val = feature_values.min()
                max_val = feature_values.max()
                if max_val - min_val > 1e-10:  # Avoid division by zero
                    normalized_values = (feature_values - min_val) / (max_val - min_val)
                else:
                    normalized_values = np.zeros_like(feature_values)
                
                # Use different colors for each feature
                color = plt.cm.tab10(i % 10)
                ax.plot(full_time_indices, normalized_values, linewidth=2, label=feature_name, 
                       color=color, alpha=0.8)
            
            # Add vertical line to mark forecast start
            ax.axvline(x=full_time_indices[forecast_start_idx], color='red', 
                      linestyle='--', linewidth=1.5, alpha=0.7, label='Forecast Start')
            
            # Formatting
            if col_idx == 0:
                ax.set_ylabel('Normalized\nFeatures', fontsize=14)
            if row_idx == num_rows - 1:
                ax.set_xlabel('Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)  # Slightly expand y-limits for better visibility
            
            # Only show title (sample index) on top row
            if row_idx == 0:
                ax.set_title(f'{sample_idx}', fontsize=14)
            
            # Add legend for the normalized features
            if col_idx == 0:
                ax.legend(fontsize=10, loc='best')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            row_idx += 1
        
        # Next rows: Context features (non-swept features from input data)
        for feature_name in display_features:
            ax = axes[row_idx, col_idx]
            
            # Get feature values from the original sample data
            feature_values = sample_data[feature_name].values
            
            # Plot the feature (constant across sweeps since we don't modify it)
            ax.plot(full_time_indices, feature_values, 'k-', linewidth=2, label='Input Data')
            
            # Add vertical line to mark forecast start
            ax.axvline(x=full_time_indices[forecast_start_idx], color='red', 
                      linestyle='--', linewidth=1.5, alpha=0.7, label='Forecast Start')
            
            # Formatting
            # Only show y-label (feature name) on leftmost column
            if col_idx == 0:
                ax.set_ylabel(f'{feature_name}', fontsize=14)
            # Only show x-label on bottom row
            if row_idx == num_rows - 1:
                ax.set_xlabel('Time', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Only show title (sample index) on top row
            if row_idx == 0:
                ax.set_title(f'{sample_idx}', fontsize=14)
            
            if col_idx == 0:
                ax.legend(fontsize=10, loc='best')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
            
            row_idx += 1
        
        # Remaining rows: Target predictions with sweep variations
        for target_name in target_names:
            ax = axes[row_idx, col_idx]
            
            # Plot historical target data from input (same for all sweeps)
            historical_values = sample_data[target_name].iloc[:forecast_start_idx].values
            historical_time = sample_data.index[:forecast_start_idx]
            ax.plot(historical_time, historical_values, 'k-', linewidth=2.5, 
                   label='Historical', alpha=0.9, zorder=10)
            
            # Plot selected sweep predictions (model returns only forecast horizon)
            for i, sweep_idx in enumerate(plot_indices):
                pred_df = sample_predictions[sweep_idx]
                sweep_val = sweep_values[sweep_idx]
                
                # Extract target column from prediction
                pred_values = pred_df[target_name].values
                pred_time_indices = pred_df.index
                
                # Plot the forecast predictions
                ax.plot(
                    pred_time_indices,
                    pred_values,
                    color=colors[i],
                    linewidth=2,
                    label=f'{param_name}={sweep_val:.2f}',
                    alpha=0.8,
                    zorder=8
                )
            
            # Add vertical line to mark forecast start
            ax.axvline(x=full_time_indices[forecast_start_idx], color='red', 
                      linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Formatting
            if col_idx == 0:
                ax.set_ylabel(f'{target_name}\n(Prediction)', fontsize=14)
                ax.legend(fontsize=10, loc='best')
            if row_idx == num_rows - 1:
                ax.set_xlabel('Time', fontsize=14)
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            row_idx += 1
    
    plt.suptitle(f'Sensitivity Analysis: {param_name} Sweep', fontsize=16, y=0.998)
    # rect=[left, bottom, right, top] - increase left margin for y-axis labels
    plt.tight_layout(rect=[0.05, 0, 1, 0.99], pad=1.0, h_pad=0.5, w_pad=0.5)
    
    # Save if requested
    if save_path:
        # Create filename with parameter name
        if save_path.endswith('.png'):
            save_file = save_path.replace('.png', f'_{param_name}.png')
        else:
            save_file = f'{save_path}_{param_name}.png'
        
        # Create directory if it doesn't exist
        save_file_path = Path(save_file)
        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        logger.info(f"Sensitivity plot saved to {save_file}")
    
    plt.tight_layout()
    plt.savefig(f'{param_name}_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
