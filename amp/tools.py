import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import logging
logger = logging.getLogger()
import os
import pandas as pd
import warnings

from amp.visualize import visualize_results_plt, visualize_rmses, visualize_distribution
from amp.utils import set_random_seed, train_test_split, calculate_metrics, create_dataset, get_extension, create_eval_idx
from amp.constants import METRIC_NAME_TARGET_TOTAL_MSE
from amp.dataloader import DataLoader

# Path to save results
RESULTS_PATH = 'results'


def get_indices(df):
    """
    Get indices from dataframe or list of dataframes

    Parameters
    ----------
    df : pd.DataFrame or list of pd.DataFrame
        Dataframe or list of dataframes to extract indices from

    Returns
    -------
    list of pd.Index
        List of indices extracted from the dataframe(s)
    """
    if not isinstance(df, list):
        return [df.index]
    else:
        return [d.index for d in df]
    
def get_datasets(df, data_periods, max_lag, fcast_len, lead_time, data_freq):
    # Get extended dataframes, which include extended history and forecast horizon
    training_eval_idx, _, testing_eval_idx = [create_eval_idx(df, data_periods[k]) for k in ['training', 'validation', 'testing']]
    training_set, validation_set, testing_set = [create_dataset(df, data_periods[k], max_lag, fcast_len, lead_time, data_freq) for k in ['training', 'validation', 'testing']]
    train_args = [training_set, validation_set]
    test_df = testing_set
    # Returns arguments (dataframes) for fit method, indices for training evaluation, test dataframe and test evaluation indices
    return train_args, training_eval_idx, test_df, testing_eval_idx


def _walk_forward_split(df, train_window_size, test_window_size, step_size, data_freq, window_type='sliding'):
    """
    Generate walk-forward window specifications for rolling train/test splits.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with DatetimeIndex
    train_window_size : int
        Size of training window in number of samples
    test_window_size : int
        Size of test window in number of samples
    step_size : int
        Number of samples to step forward each iteration
    data_freq : int
        Data frequency in minutes
    window_type : {'sliding', 'expanding', 'expanding_backward'}, default='sliding'
        Defines how the training window evolves:
        - 'sliding': fixed-size training window that moves forward.
        - 'expanding': training window always starts from first sample and grows over time.
        - 'expanding_backward': test window stays fixed, training window grows backward in history.
    
    Returns
    -------
    list of dict
        List of window specifications with keys:
        - 'train_start', 'train_end': training period timestamps
        - 'test_start', 'test_end': test period timestamps
        - 'iteration': window iteration number
    """
    windows = []
    df_index = df.index
    total_len = len(df)
    
    iteration = 0
    base_train_start_idx = 0
    moving_train_start_idx = 0

    if window_type not in {'sliding', 'expanding', 'expanding_backward'}:
        raise ValueError("window_type must be one of: 'sliding', 'expanding', 'expanding_backward'")
    
    if window_type == 'expanding_backward':
        # Test window is fixed at the end; training grows backward from test start
        test_end_idx = total_len
        test_start_idx = total_len - test_window_size
        
        if test_start_idx < 0:
            raise ValueError(f"Not enough data for test window of size {test_window_size}")
        
        # Training starts at various points, always ending just before test
        max_train_start_idx = test_start_idx - train_window_size
        
        if max_train_start_idx < 0:
            raise ValueError(
                f"Not enough data for backward expanding: need at least "
                f"{train_window_size + test_window_size} samples"
            )
        
        # Walk backward: start with max training, then expand further back
        current_train_start_idx = max_train_start_idx
        while current_train_start_idx >= 0:
            train_end_idx = test_start_idx
            windows.append({
                'iteration': iteration,
                'train_start': df_index[current_train_start_idx],
                'train_end': df_index[train_end_idx - 1],
                'test_start': df_index[test_start_idx],
                'test_end': df_index[test_end_idx - 1],
                'train_start_idx': current_train_start_idx,
                'train_end_idx': train_end_idx,
                'test_start_idx': test_start_idx,
                'test_end_idx': test_end_idx,
            })
            current_train_start_idx -= step_size
            iteration += 1
        
        return windows
    
    # Standard sliding/expanding logic
    while True:
        if window_type == 'sliding':
            train_start_idx = moving_train_start_idx
        else:  # expanding
            train_start_idx = base_train_start_idx

        train_end_idx = moving_train_start_idx + train_window_size
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_window_size
        
        # Stop if test window extends beyond data
        if test_end_idx > total_len:
            break
        
        windows.append({
            'iteration': iteration,
            'train_start': df_index[train_start_idx],
            'train_end': df_index[train_end_idx - 1],
            'test_start': df_index[test_start_idx],
            'test_end': df_index[test_end_idx - 1],
            'train_start_idx': train_start_idx,
            'train_end_idx': train_end_idx,
            'test_start_idx': test_start_idx,
            'test_end_idx': test_end_idx,
        })
        
        moving_train_start_idx += step_size
        iteration += 1
    
    
    return windows


def _walk_forward_evaluate(targets, models, df, windows, data_freq, fcast_len, lead_time, 
                           update_rate, baselines, trained_folder, verbose=False):
    """
    Execute walk-forward evaluation across all windows.
    
    Parameters
    ----------
    targets : list of Target
        Target variables
    models : dict
        Model instances
    df : pd.DataFrame
        Full dataset
    windows : list of dict
        Window specifications from _walk_forward_split
    data_freq : int
        Data frequency in minutes
    fcast_len : int
        Forecast length in steps
    lead_time : int
        Lead time in steps
    update_rate : int
        Update rate for forecasts
    baselines : dict
        Baseline models
    trained_folder : str
        Folder to save trained models
    verbose : bool
        Enable verbose logging
    
    Returns
    -------
    dict
        Aggregated results across all windows
    """
    from copy import deepcopy
    
    all_window_results = []
    model_step_mses = defaultdict(list)
    
    for window in windows:
        iteration = window['iteration']
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD ITERATION {iteration + 1}/{len(windows)}")
        print(f"{'='*80}")
        print(f"Training: {window['train_start'].date()} to {window['train_end'].date()}")
        print(f"Testing:  {window['test_start'].date()} to {window['test_end'].date()}")
        print(f"{'='*80}\n")
        
        # Extract train and test data for this window
        train_df = df.loc[window['train_start']:window['train_end']]
        test_df = df.loc[window['test_start']:window['test_end']]
        
        # Fit models on training window
        window_forecasters = {}
        for name, forecaster in models.items():
            # Create a fresh copy of forecaster for this window
            window_forecaster = deepcopy(forecaster)
            
            if verbose:
                print(f"Fitting {name} on iteration {iteration + 1}")
            
            window_forecaster.fit(train_df)
            window_forecasters[name] = window_forecaster
        
        # Evaluate on test window
        test_index = test_df.index
        
        # Get extension for test data
        max_lag = min([window_forecasters[name].input_window[0] for name in window_forecasters.keys()])
        test_start_extention, test_end_extention = get_extension(max_lag, fcast_len, lead_time, data_freq)
        extended_test_start = test_index[0] - test_start_extention
        extended_test_end = test_index[-1] + test_end_extention
        
        # Prepare extended test data
        extended_test_data = pd.concat([train_df.loc[extended_test_start:], test_df])
        if extended_test_end > df.index[-1]:
            # Can't extend beyond dataset
            pass
        else:
            extended_test_data = pd.concat([extended_test_data, df.loc[test_index[-1]+datetime.timedelta(minutes=data_freq):extended_test_end]])
        
        # Evaluate models
        window_result = evaluate(
            fcast_len,
            lead_time,
            update_rate,
            test_index,
            extended_test_data,
            targets,
            window_forecasters,
            baselines=baselines,
            verbose=verbose,
            plot=None,
            store_forecasts=False,
            mlflow_experiment=None,
            mlflow_active_run=False
        )
        window_total_mses = window_result.get('total_mses', {})
        # TODO we should have total_mse for each target, but for now we just take the average across targets if it is provided

        for model_name, model_mse in window_total_mses.items():
            model_step_mses[model_name].append(model_mse)

        window_best_model = window_result['name']
        window_best_mse = window_result.get('mse', window_total_mses.get(window_best_model))

        all_window_results.append({
            'iteration': iteration,
            'train_start': window['train_start'],
            'train_end': window['train_end'],
            'train_length': window['train_end_idx'] - window['train_start_idx'],
            'test_start': window['test_start'],
            'test_end': window['test_end'],
            'best_model': window_best_model,
            'best_mse': window_best_mse,
            'total_mses': window_total_mses,
            'target_metrics': window_result.get('target_metrics', {}),
        })

    average_model_mse = {
        model_name: float(np.mean(step_mses))
        for model_name, step_mses in model_step_mses.items()
        if len(step_mses) > 0
    }
    std_model_mse = {
        model_name: float(np.std(step_mses))
        for model_name, step_mses in model_step_mses.items()
        if len(step_mses) > 0
    }

    if len(average_model_mse) == 0:
        raise RuntimeError('Walk-forward evaluation did not produce any model metrics.')

    best_model_overall = min(average_model_mse, key=average_model_mse.get)
    best_mse_overall = average_model_mse[best_model_overall]
    
    return {
        'all_windows': all_window_results,
        'average_model_mse': average_model_mse,
        'std_model_mse': std_model_mse,
        'best_model': best_model_overall,
        'best_mse': best_mse_overall,
        'num_windows': len(windows)
    }


def train_eval_tool(mode,
                    targets,
                    models,
                    df=None,
                    test_period=None,
                    data_freq=None,
                    fcast_len=None,
                    lead_time=None,
                    update_rate=None,
                    baselines={},
                    plot=None,
                    verbose=False,
                    trained_folder='trained_models',
                    mlflow_experiment=None,
                    mlflow_uri=None,
                    mlflow_model_name=None,
                    mlflow_registry_update_mode='better_only',
                    data_periods=None,
                    dataset_info=None,
                    dataloader: DataLoader=None,
                    walk_forward_params=None,
                    store_all=False
                    ):
    """
    Train, evaluate, deploy, or walk-forward validate forecasting models.

    This function is the main orchestration entry point for AMP forecasters. It
    handles dataset preparation, model fitting, model loading for evaluation,
    baseline comparison, optional MLflow logging, and walk-forward validation.

    Data can be provided in three ways:

    1. Legacy single-dataframe mode: `df` + `test_period`
    2. Dataloader mode: `dataloader` supplies split dataframes for compatible models
    3. Explicit split mode: `df` + `data_periods` with training/validation/testing periods

    Parameters
    ----------
    mode : {'fit', 'eval', 'eval_train', 'deploy', 'walk_forward'}
        Operation mode.

        - 'fit': train and save the provided `models`
        - 'eval': load saved models and evaluate on the testing split
        - 'eval_train': load saved models and evaluate on the training split
        - 'deploy': load saved models, evaluate, and select the best model for deployment flow
        - 'walk_forward': run repeated train/evaluate cycles over rolling windows
    targets : list of Target
        Forecast targets used by the models and evaluation.
    models : dict[str, BaseModel]
        Mapping from model name to AMP forecaster instance. In `fit` and
        `walk_forward` modes the instances are used directly. In `eval`,
        `eval_train`, and `deploy`, the corresponding saved models are loaded
        from `trained_folder` using the provided types.
    df : pd.DataFrame, optional
        Source dataframe used in legacy single-dataframe mode, explicit
        `data_periods` mode, and walk-forward mode.
    test_period : tuple[str | pandas.Timestamp, str | pandas.Timestamp], optional
        Testing period used in legacy single-dataframe mode.
    data_freq : int, optional
        Sampling interval in minutes.
    fcast_len : int, optional
        Forecast horizon in steps.
    lead_time : int, optional
        Lead time before the forecast horizon in steps.
    update_rate : int, optional
        Forecast update interval in steps. If equal to `fcast_len`, only one
        forecast is produced per horizon.
    baselines : dict, default={}
        Baseline models used during evaluation and walk-forward validation.
    plot : {'all', 'best'} or None, default=None
        Plot selection passed to evaluation. Used only in evaluation flows.
    verbose : bool, default=False
        If True, enables more detailed logging and evaluation output.
    trained_folder : str, default='trained_models'
        Directory for saving fitted models or loading previously saved models.
    mlflow_experiment : str, optional
        MLflow experiment name. When provided, MLflow logging is enabled.
    mlflow_uri : str, optional
        MLflow tracking URI such as `http://localhost:5001`.
    mlflow_model_name : str, optional
        Registered model name used by MLflow-enabled evaluation/deployment flows.
    mlflow_registry_update_mode : {'better_only', 'always', 'none'}, default='better_only'
        MLflow registry update policy.
    data_periods : dict, optional
        Explicit split specification. Expected keys are typically `training`,
        `validation`, and `testing`.
    dataset_info : dict, optional
        Optional dataset metadata forwarded to MLflow dataset-context logging.
    dataloader : DataLoader, optional
        External dataloader providing split datasets for compatible models.
    walk_forward_params : dict, optional
        Parameters for `mode='walk_forward'`. Required keys:

        - `train_window_size`: training window length in samples
        - `test_window_size`: testing window length in samples
        - `step_size`: shift size between iterations in samples

        Optional keys:

        - `window_type`: one of `{'sliding', 'expanding', 'expanding_backward'}`

        `expanding_backward` keeps the test window fixed and expands the
        training history backward in time.
    store_all : bool, default=False
        Passed to `evaluate()`. When forecast storage is enabled there, store
        forecasts for all evaluated models instead of only the best model.

    Returns
    -------
    dict or None
        Return value depends on `mode`.

        - `walk_forward`: returns the walk-forward summary dictionary with keys
          such as `all_windows`, `average_model_mse`, `std_model_mse`,
          `best_model`, `best_mse`, and `num_windows`
        - `fit`, `eval`, `eval_train`, `deploy`: currently return `None`
    """

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Enable logging if experiment is provided
    if mlflow_experiment:
        mlflow, ForecasterWrapper, print_run_summary, log_model_metrics_and_params, best_existing_model, register_model, log_dataset_context = _import_mlflow()

        logging.basicConfig(level=logging.DEBUG)
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment)
        mlflow.start_run()  # Start the MLflow run
        mlflow_active_run = mlflow.active_run()
    else:
        mlflow_active_run = False

    set_random_seed(7)
    result = None

    # Select folder for trained models
    if trained_folder == 'trained_models':
        folder = Path(f'{Path.cwd()}/trained_models')
    else:
        folder = Path(trained_folder)

    # Load the forecasters at this point in order to use max_lag in dataset creation
    forecasters = {}
    max_lag = 0
    if mode == 'eval' or mode == 'deploy' or mode == 'eval_train':
        for name, forecaster in models.items():
            #forecaster = type(forecaster).load(folder.joinpath(f'{name}.pkl'))
            forecaster = type(forecaster).load(folder.joinpath(f'{name}.zip'))
            lag, _ = forecaster.input_window
            if lag < max_lag:
                max_lag = lag
            forecasters[name] = forecaster

    # Else, the mode is fit
    else:
        # For 'fit' and 'walk_forward' modes, use models directly
        for name, forecaster in models.items():
            lag, _ = forecaster.input_window
            if lag < max_lag:
                max_lag = lag
            forecasters[name] = forecaster

    # If data_periods is provided
    if df is not None and data_periods is None and mode != 'walk_forward':
        train_df, test_data = train_test_split(df, test_period=test_period)
        train_args = [train_df]
        test_index = test_data.index
    elif data_periods:
        # Use data_periods to create training and test datasets and evaluation indices
        train_args, train_index, test_data, test_index = get_datasets(df, data_periods, max_lag, fcast_len, lead_time, data_freq)
        training_set = train_args[0] # The first element is the training data, while the second is validation data

    if mode == 'fit':
        for name, forecaster in forecasters.items():
            print(50*'=' + f'\nFitting {name}\n' + 50*'=')

            if hasattr(forecaster, '_get_dataloaders'):
                # Torch model: resolve training/validation/testing as lists of DataFrames
                if dataloader is not None:
                    # Extract pre-split segments from an externally provided dataloader
                    training_data = getattr(dataloader, 'training_segments', None) or (
                        [dataloader.training_timeseries] if getattr(dataloader, 'training_timeseries', None) is not None else None
                    )
                    validation_data = getattr(dataloader, 'validation_segments', None) or (
                        [dataloader.validation_timeseries] if getattr(dataloader, 'validation_timeseries', None) is not None else None
                    )
                    testing_data = getattr(dataloader, 'testing_segments', None) or (
                        [dataloader.testing_timeseries] if getattr(dataloader, 'testing_timeseries', None) is not None else None
                    )
                elif data_periods:
                    training_data = train_args[0]
                    validation_data = train_args[1] if len(train_args) > 1 else None
                    testing_data = test_data
                else:
                    # Legacy single-df path: wrap in lists
                    training_data = [train_df]
                    validation_data = None
                    testing_data = [test_data] if test_data is not None else None
                    # print train and test date ranges for legacy path
                    print(f"Training data range: {training_data[0].index.min()} to {training_data[0].index.max()}")
                    print(f"Testing data range: {testing_data[0].index.min()} to {testing_data[0].index.max()}")
                forecaster.fit(training_data, validation_data, testing_data)
            else:
                # train_args[0] may be either a list of DataFrames (data_periods
                # path) or a single DataFrame (legacy ``df`` path). Handle both.
                training_data = train_args[0]
                if isinstance(training_data, list):
                    training_data = pd.concat(training_data)
                forecaster.fit(training_data)

            # Print out the hyperparameters used
            if hasattr(forecaster, 'models'):
                for output in forecaster.outputs:
                    if hasattr(forecaster.models[output].pipe.named_steps['model'], 'best_params_'):
                        best = forecaster.models[output].pipe.named_steps['model'].best_params_
                        print(f'\nBest Hyperparameters for {output} from search:{best}\n')
                    default = forecaster.models[output].pipe.named_steps['model'].get_params()
                    print(f'\nDefault parameters for {output}: {default}\n')

            if not os.path.exists(folder):
                os.makedirs(folder)

            #forecaster.save(folder.joinpath(f'{name}.pkl'))
            forecaster.save(folder.joinpath(f'{name}.zip'))

    elif mode == 'walk_forward':
        # Walk-forward validation mode
        if walk_forward_params is None:
            raise ValueError("walk_forward_params must be provided for mode='walk_forward'")
        
        train_window_size = walk_forward_params.get('train_window_size')
        test_window_size = walk_forward_params.get('test_window_size')
        step_size = walk_forward_params.get('step_size')
        window_type = walk_forward_params.get('window_type', 'sliding')
        
        if train_window_size is None or test_window_size is None or step_size is None:
            raise ValueError("walk_forward_params must contain 'train_window_size', 'test_window_size', and 'step_size'")
        
        if df is None:
            raise ValueError("df must be provided for walk_forward mode")
        
        print(f"\n{'='*80}")
        print("WALK-FORWARD VALIDATION")
        print(f"{'='*80}")
        print(f"Train window: {train_window_size} samples")
        print(f"Test window: {test_window_size} samples")
        print(f"Step size: {step_size} samples")
        print(f"Window type: {window_type}")
        print(f"Total data points: {len(df)}")
        
        # Generate windows
        windows = _walk_forward_split(
            df,
            train_window_size,
            test_window_size,
            step_size,
            data_freq,
            window_type=window_type,
        )
        print(f"Number of windows: {len(windows)}\n")
        
        # Execute walk-forward validation
        wf_results = _walk_forward_evaluate(
            targets,
            models,
            df,
            windows,
            data_freq,
            fcast_len,
            lead_time,
            update_rate,
            baselines,
            trained_folder,
            verbose=verbose
        )
        
        # Print summary of walk-forward results
        print(f"\n{'='*80}")
        print("WALK-FORWARD SUMMARY")
        print(f"{'='*80}")
        print(f"Total iterations: {wf_results['num_windows']}")
        print("Average MSE over all walk-forward steps:")
        for model_name, avg_mse in wf_results['average_model_mse'].items():
            std_mse = wf_results['std_model_mse'][model_name]
            print(f"  - {model_name}: avg={avg_mse:.4f}, std={std_mse:.4f}")
        print(f"Best model by average MSE: {wf_results['best_model']}")
        print(f"Best average MSE: {wf_results['best_mse']:.4f}\n")
        result = wf_results

    elif mode == 'eval' or mode == 'deploy' or mode == 'eval_train':

        #TODO evaluate functionality is useful outside train_eval_tool as well

        if mode == 'eval_train':
            if df is not None and data_periods is None:
                test_data = train_df
                test_index = train_df.index
            elif dataloader:
                test_data = dataloader.get_evaluation_dataframes('training', min_length=abs(max_lag) + lead_time + fcast_len)
                test_index = get_indices(test_data)
            elif data_periods:
                test_data = training_set
                test_index = train_index
            
        else:
            if df is not None and data_periods is None:
                # extend the index with lags and forecast horizon (lead time + length) so that all forecast fit
                # the validation period.
                # TODO: It is assumed that there is enough data for the extended period. If there is not,
                # the evaluation will still run, but there are not as many prediction samples made at
                # the start or end, depending on the amount of data missing.
                test_start_extention, test_end_extention = get_extension(max_lag, fcast_len, lead_time, data_freq)
                extended_test_start = test_index[0] - test_start_extention
                extended_test_end = test_index[-1] + test_end_extention
                
                test_data = pd.concat([train_df.loc[extended_test_start:], test_data])
                test_data = pd.concat([test_data, df.loc[test_index[-1]+datetime.timedelta(minutes=data_freq):
                                                    extended_test_end]])                

            elif dataloader is not None:
                test_data = dataloader.get_evaluation_dataframes('testing', min_length=abs(max_lag) + lead_time + fcast_len)
                test_index = get_indices(test_data)


        if mlflow_experiment:
            if df is not None and data_periods is None:
                log_dataset_context(train_df, test_data, data_freq, dataset_info)
            else:
                log_dataset_context(training_set, test_data, data_freq, dataset_info)

        best = evaluate(fcast_len,
                        lead_time,
                        update_rate,
                        test_index,
                        test_data,
                        targets,
                        forecasters,
                        baselines=baselines,
                        verbose=verbose,
                        plot=plot,
                        mlflow_experiment=mlflow_experiment,
                        mlflow_model_name=mlflow_model_name,
                        mlflow_active_run=mlflow_active_run,
                        mlflow_registry_update_mode=mlflow_registry_update_mode,
                        store_all=store_all)
        result = best


        if mode == 'deploy':
            best_models_dir = 'best_models'
            print(f'Deploying best model to {best_models_dir}')

            # TODO not working yet
            #best['model'].name = 'best'
            #best['model'].save(folder=best_models_dir)
    # Finish MLflow run if experiment is set
    if mlflow_experiment:
        mlflow.end_run()

    return result


def evaluate(forecast_len, lead_time, update_rate, test_index, test_data, targets,
             forecasters, baselines, verbose=False, plot='best', store_forecasts=True,
             mlflow_experiment=None, mlflow_model_name=None, mlflow_registry_update_mode='better_only', mlflow_active_run=False, store_all=False):
    """
        Evaluates the performance of forecasting models by calculating Mean Squared Error (MSE),
        comparing them against baseline models, and visualizing the results.

        Parameters
        ----------
        forecast_len : int
            The length of the forecast (number of steps ahead to predict).
        lead_time : int
            The time step after which to make predictions.
        update_rate : int
            The rate at which the forecast model is updated (e.g., how often new predictions are made).
        test_index : pandas.Dataframe.index or [pandas.DataFrame.index]
            Index of the test dataframe (a subset of the test_data indices)
        test_data : pandas.Dataframe or [pandas.DataFrame]
            The test data containing the true values for the target variables.
        targets : list of Target
            A list of Target objects representing the variables to be predicted.
        forecasters : dict
            A dictionary of forecasters where the key is the model name and the value is the model instance.
        baselines : dict
            A dictionary of baseline models (e.g., lag models, mean models) for comparison.
        plot : str or None, optional
            The type of plot to display. Options include 'best' (best-performing model) and 'all' (all models).
            Default is 'best'.
        store_forecasts : bool, optional
            Whether to save the forecasts to CSV files. Default is True.
        store_all : bool, optional
            If store_forecasts is True, whether to store forecasts for all models or only the best model. Default is False.

        Returns
        -------
        best_model : dict
            The best model after evaluation, based on the lowest total MSE across all targets.
            Contains the model name and instance.
    """
    forecasters_keys = list(forecasters.keys()) # Take forecasters keys for plotting. If forecasters is empty, forecasters_keys is parsed from the results csv file.

    if mlflow_experiment:
        mlflow, ForecasterWrapper, print_run_summary, log_model_metrics_and_params, best_existing_model, register_model, log_dataset_context = _import_mlflow()

    # Make list of dictionaries from Targets
    targets = [target.to_dict() for target in targets]
    pred_dict = {}

    if not isinstance(test_data, list):
        test_set = [test_data]
    else:
        test_set = test_data
    if not isinstance(test_index, list):
        test_index = [test_index]

    if forecasters == {}:
        # If there are no forecasters provided, load predictions from csv files for evaluation
        logger.info('No forecasters provided, loading predictions from csv files for evaluation.')
        pred_dict = {}
        
        for f in os.listdir(RESULTS_PATH):
            if f.endswith('.csv'):    
                # read the file
                df = pd.read_csv(Path(RESULTS_PATH) / f, index_col=0, parse_dates=True)

                # last column is the target
                t_long = df.columns[-1]
                t_name = t_long.replace('_measured', '')

                # drop the column
                df = df.drop(columns=[t_long])

                # Find o_name from file name or raise error if not found
                for t in targets:
                    # If we find the output in name
                    if t_name.find(t['output']) != -1:
                        o_name = t['output']
                        break
                else:
                    raise ValueError(f'Could not find target output name in file name: {f}')

                # the model name is extracted using the target column name
                f_name = f.replace(f'_{t_name}.csv', '')
                forecasters_keys.append(f_name)

                # add the dataframe to the prediction dictionary
                target_dict = {}
                target_dict[o_name] = df
                pred_dict[f_name] = target_dict
            else:
                logger.warning(f'File {f} does not end with .csv and is skipped.')
            forecasters_keys = list(set(forecasters_keys)) # Remove duplicates if any
    else:
        # Generate predictions for all forecasters
        for f_name, model in forecasters.items():
            print(50 * '=' + f'\nEvaluating model: {f_name}\n' + 50 * '=')
            import time
            start_time = time.time()
            prediction_list = [model.predict(df, output_mode='target') for df in test_set]
            prediction = {
                o: pd.concat(
                    [p[o] for p in prediction_list], axis=0
                ) for o in model.outputs
                #).resample(f'{model.update_rate}min').first() for o in model.outputs  # resample to add timestamps between datasets
            }
            print(f"Model {f_name} prediction completed in {time.time() - start_time:.2f} seconds.")
            pred_dict[f_name] = prediction

    # make a single dataframe from given set
    test_df = pd.concat([s for s in test_set], axis=0)
    test_df = test_df[~test_df.index.duplicated(keep='first')]
    test_index = pd.Index(pd.concat([pd.Series(i) for i in test_index], axis=0))

    mse_periods = {}
    mae_periods = {}
    baselines_mse = {}
    baselines_mae = {}
    total_mses = {}
    baseline_predictions = defaultdict(dict)
    summary_metrics = defaultdict(dict)

    for target_cfg in targets:
        t_name = target_cfg['column']
        o_name = target_cfg.get('output', t_name)

        print(f'\nEvaluating target variable: {t_name}\n' + '-' * 40)

        target = test_df.loc[test_df.index.intersection(test_index), t_name] # OPTION A
        #target = test_df[t_name] # OPTION B

        for f_name, prediction in pred_dict.items():

            if f_name not in mse_periods:
                mse_periods[f_name] = {}
                mae_periods[f_name] = {}
            if target.name not in mse_periods[f_name]:
                mse_periods[f_name][target.name] = {}
                mae_periods[f_name][target.name] = {}

            target_mse = 0
            target_mae = 0
            pred = prediction[o_name]
            target_cfg[f_name] = pred.loc[pred.index.intersection(test_index)].copy()
            target_cfg[f_name][t_name] = target
            if verbose:
                print(50 * '=' + f'\nResults for model: {f_name} on target: {t_name}\n' + 50 * '=')
            for period_name in pred.columns:
                if verbose:
                    print(f'Calculating MSE for forecast period: {period_name}')
                period = pred[period_name].dropna()
                target_period = target.loc[target.index.intersection(period.index)].dropna()
                period = period.loc[target_period.index]
                target_cfg[f_name][period_name] = period

                if len(target_period) == 0:
                    warnings.warn(f'Target and prediction length is zero for [{t_name}], [{period_name}]')
                    break

                period_mse = mean_squared_error(target_period, period)
                # Calculate MAE
                period_mae = mean_absolute_error(target_period, period)
                mae_periods[f_name][target.name][period_name] = period_mae
                target_mae += period_mae / len(pred.columns)
                if verbose:
                    print(f"MSE for model {f_name}, target {t_name}, period {period_name}: {period_mse:.4f}")
                    print_results(period_mse,period_mae, target_period)
                mse_periods[f_name][target.name][period_name] = period_mse
                target_mse += period_mse / len(pred.columns)

            # Store the total MSE for this target and model
            print(50 * '=' + f'\nTotal MSE for target: {t_name} with model: {f_name}\n' + 50 * '=')
            print_results(target_mse, target_mae, target)
            target_cfg[f'{f_name}_mse'] = target_mse
            target_cfg[f'{f_name}_mae'] = target_mae
            target_rmse, target_nrmse1, target_nrmse2 = calculate_metrics(target_mse, target)
            summary_metrics[t_name][f_name] = {
                'mse': float(target_mse),
                'rmse': float(target_rmse),
                'mae': float(target_mae),
                'nrmse1': float(target_nrmse1),
                'nrmse2': float(target_nrmse2),
            }

            if f_name not in total_mses:
                total_mses[f_name] = 0
            if 'scaler' not in target_cfg:
                total_mses[f_name] += target_mse
            else:
                total_mses[f_name] += (target_mse * target_cfg['scaler'])

        # Comparison to baselines

        if len(baselines) > 0:
            print(50 * '=' + '\nComparing to baseline models\n' + 50 * '=')

            for base_name, baseline in baselines.items():
                # TODO: Should each baseline declare which feature it aims to estimate?

                if hasattr(baseline, 'feature'):
                    print(f"Baseline is using feature: {baseline.feature}")
                    baseline_pred = baseline.predict(test_index, test_df[baseline.feature])
                else:
                    # If the baseline does not have feature attribute, it is applied to all targets
                    baseline_pred = baseline.predict(test_index, test_df[t_name])

                if not baseline_pred.empty:
                    baseline_pred = baseline_pred.dropna()
                    target = test_df.loc[baseline_pred.index, t_name].dropna()
                    baseline_pred = baseline_pred.loc[target.index]

                    if len(baseline_pred) == 0 or len(target) == 0:
                        warnings.warn(f'Target and/or baseline length is zero for [{t_name}], [{base_name}]')
                    else:
                        # Calculate and print baseline MSE
                        baseline_predictions[base_name][t_name] = baseline_pred
                        base_mse = mean_squared_error(target, baseline_pred)
                        base_mae = mean_absolute_error(target, baseline_pred)

                        print(f"Baseline MSE for model {base_name} on target {t_name}: {base_mse:.4f}")
                        print(f"Baseline MAE for model {base_name} on target {t_name}: {base_mae:.4f}")
                        print_results(base_mse, base_mae, target)
                        base_rmse, base_nrmse1, base_nrmse2 = calculate_metrics(base_mse, target)
                        baseline_row_name = f'baseline_{base_name}'
                        summary_metrics[t_name][baseline_row_name] = {
                            'mse': float(base_mse),
                            'rmse': float(base_rmse),
                            'mae': float(base_mae),
                            'nrmse1': float(base_nrmse1),
                            'nrmse2': float(base_nrmse2),
                        }
                        if base_name not in baselines_mse:
                            baselines_mse[base_name] = 0
                        if base_name not in baselines_mae:
                            baselines_mae[base_name] = 0
                            baselines_mae[base_name] += base_mae
                        if 'scaler' not in target_cfg:
                            baselines_mse[base_name] += base_mse
                            baselines_mae[base_name] += base_mae
                        else:
                            baselines_mse[base_name] += (base_mse * target_cfg['scaler'])
                            baselines_mae[base_name] += (base_mae * target_cfg['scaler'])
 
            best_baseline = min(baselines_mse, key=baselines_mse.get)
            best_baseline_predictions = baseline_predictions[best_baseline]
            print(f'Best baseline was: {best_baseline} Total MSE for all targets: {baselines_mse[best_baseline]}\n' + 50 * '=')
        else:
            best_baseline = None
            best_baseline_predictions = None

    best_model = {}
    best_mse = None
    
    # Overall results summary
    print('Summary of current run results for all targets. (Average of the periods) \n' + 50 * '=')
    for f_name, model in total_mses.items():
        print(f'Model: {f_name}')
        print(f'Total MSE across all targets: {total_mses[f_name]:.4f}')

        for t in targets:
            print(f'Target: {t["column"]}\n' + '-' * 40)
            print_results(t[f"{f_name}_mse"], t[f"{f_name}_mae"], t[f_name][t["column"]])

        print(50 * '=')

        if best_mse is None or total_mses[f_name] < best_mse:
            print(f'Updating best model to: {f_name} (lower MSE found)')
            best_mse = total_mses[f_name]
            best_model['name'] = f_name
            best_model['model'] = forecasters.get(f_name, None)  # Get the model instance if available. If evaluating from csv files, then model objects are not loaded.

    best_model['mse'] = best_mse
    best_model['total_mses'] = dict(total_mses)
    best_model['target_metrics'] = {
        target_name: {
            model_name: dict(metric_values)
            for model_name, metric_values in model_metrics.items()
        }
        for target_name, model_metrics in summary_metrics.items()
    }

    metric_columns = ['mse', 'rmse', 'mae', 'nrmse1', 'nrmse2']
    for target_cfg in targets:
        t_name = target_cfg['column']
        target_rows = []

        for model_name in forecasters_keys:
            metric_values = summary_metrics.get(t_name, {}).get(model_name, {})
            target_rows.append(
                {
                    'name': model_name,
                    **{metric: float(metric_values.get(metric, np.nan)) for metric in metric_columns},
                }
            )

        for baseline_name in baselines.keys():
            row_name = f'baseline_{baseline_name}'
            metric_values = summary_metrics.get(t_name, {}).get(row_name, {})
            target_rows.append(
                {
                    'name': row_name,
                    **{metric: float(metric_values.get(metric, np.nan)) for metric in metric_columns},
                }
            )

        if target_rows:
            summary_df = pd.DataFrame(target_rows).set_index('name')
            print(f'\nFinal evaluation summary table for target: {t_name} (average over forecast periods):')
            print(summary_df.to_string(float_format=lambda value: f'{value:.6f}'))
    # This is not including baselines in the decision of best model, but the summary table above includes baselines for comparison
    print(f'\nFinal best model after evaluation (not including baselines): {best_model["name"]}\n')


    # Find best existing from MLFlow
    if mlflow_experiment:
        run = best_existing_model(mlflow_experiment)
        try:
            existing_best = run[f'metrics.{METRIC_NAME_TARGET_TOTAL_MSE}']
        except KeyError:
            existing_best = None
        if mlflow_registry_update_mode == 'better_only':
            if existing_best is None or best_mse < existing_best:
                print(
                    f"Updating registry: Current model (MSE: {best_mse:.4f}) is better than existing (MSE: {existing_best if existing_best is not None else 'None'}).")
                ForecasterWrapper.save_model(test_df, best_model['model'])
                register_model(mlflow_model_name)
            else:
                print(
                    f"Registry not updated: Current model (MSE: {best_mse:.4f}) is not better than existing (MSE: {existing_best:.4f}).")
        elif mlflow_registry_update_mode == 'always':
            print("Updating registry: Always updating the model in the registry.")
            ForecasterWrapper.save_model(test_df, best_model['model'])
            register_model(mlflow_model_name)
        elif mlflow_registry_update_mode == 'none':
            print("Registry update skipped: Only logging the experiment.")
        else:
            raise ValueError(f"Invalid value for mlflow_registry_update_mode: {mlflow_registry_update_mode}")
        log_model_metrics_and_params(best_model, targets)


    # Plot results if requested
    if plot is not None:
        if plot == 'best':
            print(f'Plotting results for best model: {best_model["name"]}')
            visualize_results_plt(best_model["name"], targets, test_df,
                                  forecast_len, lead_time, update_rate)
            visualize_distribution(best_model["name"], targets, test_df, best_baseline_predictions, best_baseline, mlflow_active_run)
            visualize_rmses(mse_periods[best_model['name']], best_model['name'])
        elif plot == 'all':
            for f_name in forecasters_keys:
                print(f'Plotting results for model: {f_name}')
                visualize_results_plt(f_name, targets, test_df, forecast_len, lead_time, update_rate)
                visualize_distribution(f_name, targets, test_df, best_baseline_predictions, best_baseline, mlflow_active_run)
                visualize_rmses(mse_periods[f_name], f_name)
        else:
            raise ValueError(f'invalid value for plotting parameter: {plot}')


    # Save forecast for the best model if requested
    if store_forecasts and best_model != {}:
        path = RESULTS_PATH
        if not os.path.exists(path):
            os.makedirs(path)
        if store_all:
            models = forecasters_keys
        else:
            # Store only best
            models = [best_model['name']]

        for m in models:
            preds = pred_dict[m]
            for t in targets: 
                pred = preds[t['output']].copy() # Avoid modifying the original predictions
                pred[f'{t["column"]}_measured'] = test_df[t["column"]]
                file_path = Path(path) / f'{m}_{t["column"]}.csv'
                pred.to_csv(file_path)
                print(f'Saved forecast results for target {t["column"]} to {file_path}')


    return best_model




def print_results(mse, mae, target):

    rmse, nrmse1, nrmse2 = calculate_metrics(mse, target)
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'Mean: {target.mean()}')
    print(f'Max: {target.max()}, Min: {target.min()}')
    print(f'RMSE: {rmse}')
    print(f'NRMSE1: {nrmse1}')
    print(f'NRMSE2: {nrmse2}')
    print(50 * '=')


def _import_mlflow():
    try:
        import mlflow
        from amp.mlflow_utils import (
            ForecasterWrapper,
            print_run_summary,
            log_model_metrics_and_params,
            best_existing_model,
            register_model,
            log_dataset_context
        )
        return mlflow, ForecasterWrapper, print_run_summary, log_model_metrics_and_params, best_existing_model, register_model, log_dataset_context
    except ImportError:
        warnings.warn(
            "MLFlow is not installed in the environment. "
            "If you wish to use MLFlow, please install it using 'pip install mlflow'."
        )
        raise