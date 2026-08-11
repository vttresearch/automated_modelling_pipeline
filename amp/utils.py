import pandas as pd
import numpy as np
import argparse
import re
import itertools
import pandas
import logging
logger = logging.getLogger(__name__)
from copy import deepcopy
import datetime

def floor_time(dt, resolution=1):
    """ Floor datatime object to any resolution (sampling rate)

    Parameters
    ----------
    dt : datetime.datetime
        Time to be floored.
    resolution : int
        Sampling rate in minutes.

    Returns
    -------
    datetime.datetime
        Floored time.

    """
    remainder = dt.minute % resolution
    return dt + datetime.timedelta(minutes=-remainder, seconds=-dt.second, microseconds=-dt.microsecond)


def check_arg_type(allowed_strings):
    def check_string_type(arg):
        pattern = re.compile(arg)
        matched_strings = [s for s in allowed_strings if pattern.fullmatch(s)]
        if len(matched_strings) == 0:
            raise argparse.ArgumentTypeError(f"{arg} is not valid argument.")
        return matched_strings

    def check_list_type(arg_list):
        for arg in arg_list:
            check_string_type(arg)
        return arg_list

    # def check_list_type(arg_list):
    #     matched_strings = []
    #     for arg in arg_list:
    #         matched_strings.extend(check_string_type(arg))
    #     return matched_strings

    def check_type(arg):
        if isinstance(arg, str):
            return check_string_type(arg)
        elif isinstance(arg, list):
            return check_list_type(arg)
        else:
            raise argparse.ArgumentTypeError(f"{arg} is not a string or list of strings")

    return check_type


def param_select(all_models):
    """
    Selects the desired parameters for model evaluation and training.

    Parameters
    ----------
    all_models : dict
        A dictionary containing all available models, with each key as a model name
        and its corresponding value as a model instance.

    Returns
    -------
    mode : str
        The selected mode of operation, which can be one of the following:
            - fit: Training phase.
            - eval: Evaluation phase.
            - eval_train: Both evaluation and training phases.
            - deploy: Deployment phase.
    models : dict
        A dictionary containing the selected models to use for the desired operation. 
        If 'all' is selected, all models from `all_models` are returned. If 'results' is selected, 
        no models are returned and results are loaded from csv files for evaluation.
    plot : str or None, optional
        The plot type to display. Can be one of the following:
            - best: Best-performing model(s).
            - all: All models.
            - None: No plot displayed.
    verbose : bool, optional
        Whether to display verbose output or not.

    Example usage:
    >>> param_select({'model1': Model1(), 'model2': Model2()})
    ('fit', {'model1': Model1()}, None, False)
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('mode',
                        choices=['fit', 'eval', 'eval_train', 'deploy'],
                        type=str)

    model_choices = list(all_models.keys()) + ['all', 'results']
    parser.add_argument('--models', '-m',
                        nargs='+',
                        default=[['all']],
                        help=f'Choices: {model_choices}',
                        type=check_arg_type(model_choices))

    parser.add_argument('--plot', '-p',
                        choices=['best', 'all', None],
                        default=None,
                        type=str)

    parser.add_argument('--verbose', '-v', action='store_true')

    if hasattr(parser, 'parse_known_args'):
        args, _ = parser.parse_known_args()  # Just to ignore unknown args
    else:
        args = parser.parse_args()

    model_names_two_d = args.models
    model_names = [elem for sublist in model_names_two_d for elem in sublist]
    if 'all' in model_names:
        selected_models = all_models
    elif 'results' in model_names:
        selected_models = {}
    else:
        selected_models = {}
        for m_name, model in all_models.items():
            if m_name in model_names:
                selected_models[m_name] = model

    # TODO: We should make the output of this function more generic to allow for new arguments.
    # For example this function could return args as the first argument, and more specific arguments after
    return args.mode, selected_models, args.plot, args.verbose

def param_select_2(all_models):
    """
    Selects the desired parameters for model evaluation and training.
    Parses input with 'input()' commands insted of command line arguments.

    Parameters
    ----------
    all_models : dict
        A dictionary containing all available models, with each key as a model name
        and its corresponding value as a model instance.

    Returns
    -------
    mode : str
        The selected mode of operation, which can be one of the following:
            - fit: Training phase.
            - eval: Evaluation phase.
            - eval_train: Both evaluation and training phases.
            - deploy: Deployment phase.
    models : dict
        A dictionary containing the selected models to use for the desired operation.
    plot : str or None, optional
        The plot type to display. Can be one of the following:
            - best: Best-performing model(s).
            - all: All models.
            - None: No plot displayed.
    verbose : bool, optional
        Whether to display verbose output or not.

    Example usage:
    >>> param_select({'model1': Model1(), 'model2': Model2()})
    ('fit', {'model1': Model1()}, None, False)
    """
    
    
    # Select mode
    modes = ['fit', 'eval', 'eval_train', 'deploy']
    print("Select mode:")
    for i, mode in enumerate(modes, 1):
        print(f"{i}. {mode}")
    mode_choice = input("Enter the number of your choice: ")
    try:
        mode = modes[int(mode_choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice. Using default: eval_train")
        mode = "eval_train"
    
    # Model choice
    models = list(all_models.keys())
    selected_models = dict()
    if len(models) ==1:
        k, v = next(iter(all_models.items()))
        print(f"\nModel \'{k}\' is selected automatically as it was the only available model.")
        selected_models[k] = v
    else:
        print("\nSelect models (comma-separated numbers, or press Enter for default):")
        enum_models = list(enumerate(models, 1))
        for i, model in enum_models:
            print(f"{i}. {model}")
        model_input = input("Enter your choices: ")
        try:
            selected_indices = [int(i.strip()) - 1 for i in model_input.split(",")]
            selected_keys = [models[i] for i in selected_indices if 0 <= i < len(models)]
            for k in selected_keys:
                selected_models[k] = all_models[k]
        except ValueError:
            print("Invalid input. Using all models as default")
            selected_models = all_models

    # What to plot
    plots = ['best', 'all', None]
    print("Do you want to plot results?")
    for i, p in enumerate(plots, 1):
        print(f"{i}. {p}")
    plot_choice = input("Enter the number of your choice: ")
    try:
        plot = plots[int(plot_choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice. Using default: None")
        plot = None

    # Verbose output
    verbose = input("Do you want verbose output? (y/n): ").strip().lower() == "y"

    return mode, selected_models, plot, verbose


def model_grid(forecaster, param_grid):
    param_dict = {}
    models_dict = {}

    for indexes in itertools.product(*[range(len(v)) for v in param_grid.values()]):
        key = "_".join(str(i) for i in indexes)
        param_dict[key] = {}
        for k, v in zip(param_grid.keys(), indexes):
            param_dict[key][k] = param_grid[k][v]

    # todo add support for several param_grids as in sklearn
    for key, params in param_dict.items():
        f = deepcopy(forecaster)
        f.name = f.name + f'_{key}'
        f.set_params(**params)
        models_dict[f.name] = f

    return models_dict


def form_multi_output(preds, outputs, lead_time, forecast_len, freq):
    columns = [f'forecast_{i}' for i in range(lead_time, lead_time + forecast_len)]
    index = pd.date_range(start=preds[0].index[0], end=preds[-1].index[-1], freq=f'{freq}min')

    # From a DF per output and store them into a dictionary with output as the key.
    forecast_dict = {}
    for o in outputs:
        forecast_df = pd.DataFrame(index=index, columns=columns)
        f_tmp_list = []
        for f in preds:
            if o in f.columns:
                f_tmp = f[[o]].T
            else:
                logger.warning(f'There is no prediction for {o} at time {f.index[0]}')
                f_tmp = pandas.DataFrame([], f.index, [o]).T
            f_tmp.index = [f_tmp.columns[0]]
            f_tmp.columns = columns
            f_tmp_list.append(f_tmp)
        f_concat = pd.concat(f_tmp_list)

        # Shift forecast to the right place.
        forecast_df.loc[f_concat.index] = f_concat
        for i in range(lead_time, lead_time + forecast_len):
            forecast_df[f'forecast_{i}'] = pd.to_numeric(forecast_df[f'forecast_{i}'].shift(i - lead_time))

        forecast_dict[o] = forecast_df
    return forecast_dict


def set_random_seed(seed=7, tensoflow_support=False):
    """

    Parameters
    ----------
    seed
    tensoflow_support: Bool
    set True to enable Tensorflow
    TODO remove when tensorflow issues are fixed (e.g. slow init, usage in Mac Rosetta)

    Returns
    -------

    """
    np.random.seed(seed)  # for numpy and sklearn
    if tensoflow_support is True:
        import tensorflow as tf
        tf.random.set_seed(seed)


def train_test_split(df, test_period):
    df_train = df[(df.index < test_period[0])]
    df_test = df[(df.index >= test_period[0]) & (df.index < test_period[1])]
    return df_train, df_test


def create_dataset(
        df: pandas.DataFrame,
        period: list,
        max_lag: int,
        fcast_len: int,
        lead_time: int,
        data_freq: int
) -> list:
    """

    Args:
        df: Data for training, validation and testing
        period: A list of tuples that contain start and end dates for data
        max_lag: Maximum lag of the model set
        fcast_len: Forecasting length
        lead_time: Lead time of the forecast
        data_freq: Data sample frequency in minutes
    Returns:

    """
    df = df.copy()
    _set = []
    if not isinstance(period, list):
        period = [period]
    
    t1_extension, t2_extension = get_extension(max_lag, fcast_len, lead_time, data_freq)
    for t1, t2 in period:
        t1_extended = t1 - t1_extension  # Extend the time period from the start 
        t2_extended = t2 + t2_extension  # and from the end
        mask = (df.index >= t1_extended) & (df.index < t2_extended)
        filtered_df = df[mask]
        if filtered_df.empty:
            logger.warning(f'No data found for period {t1} - {t2}. Skipping.')
            continue
        if (t1 - filtered_df.index[0] + datetime.timedelta(minutes=data_freq)) < t1_extension:
            logger.warning('Not enough data to extend period completely!')
        if (filtered_df.index[-1] - t2 + datetime.timedelta(minutes=data_freq)) < t2_extension:
            logger.warning('Not enough data to extend period completely!')
        _set.append(filtered_df)

        #df = df[~mask]  # drop appended data to avoid duplicate data in samples

    return _set



def get_extension(max_lag, fcast_len, lead_time, data_freq):
    """ Gets the extended evaluation period. Models might require additional data outside of the
    evaluated period. This function determines the extension to the evaluation data."""
    _start = datetime.timedelta(minutes=data_freq*(abs(max_lag) - 1 + (fcast_len+lead_time)))
    _end = datetime.timedelta(minutes=data_freq*(fcast_len))
    return _start, _end


def create_eval_idx(df, periods):
    """ Get evaluation index for given periods
    Args
        df: pandas DataFrame
        period: A list of tuples that contain start and end dates for data
    """
    if not isinstance(periods, list):
        periods = [periods]

    idx_list = []
    for t1, t2 in periods:
        idx_list.append(df.loc[(t1 <= df.index) & (t2 >= df.index)].index)
    return idx_list

def mark_dr_period(df, col, cp_col, ffill_len=4, bfill_len=0):
    df[f'{col}_dr'] = np.nan
    df['dr_mask'] = False
    df.loc[df[cp_col] < 100, 'dr_mask'] = True
    if ffill_len > 0:
        df['dr_mask'] = df['dr_mask'].ffill(limit=ffill_len)
    if bfill_len > 0:
        df['dr_mask'] = df['dr_mask'].bfill(limit=bfill_len)
    df['dr_mask'] = df['dr_mask'].fillna(False)
    df.loc[df['dr_mask'], f'{col}_dr'] = df.loc[df['dr_mask'], col]
    df = df.drop('dr_mask', axis=1)
    return df

def mark_event_period(df, col, cp_col, ffill_len=4, bfill_len=0, suffix='_dr', event_type=100):
    df[f'{col}{suffix}'] = np.nan
    df['dr_mask'] = False
    df.loc[df[cp_col] != event_type, 'dr_mask'] = True
    if ffill_len > 0:
        df['dr_mask'] = df['dr_mask'].ffill(limit=ffill_len)
    if bfill_len > 0:
        df['dr_mask'] = df['dr_mask'].bfill(limit=bfill_len)
    df['dr_mask'] = df['dr_mask'].fillna(False)
    df.loc[df['dr_mask'], f'{col}{suffix}'] = df.loc[df['dr_mask'], col]
    df = df.drop('dr_mask', axis=1)
    return df


def calculate_metrics(mse, target):
    rmse = np.sqrt(mse)
    nrmse1 = rmse / target.mean()
    nrmse2 = rmse / (target.max() - target.min())
    return rmse, nrmse1, nrmse2