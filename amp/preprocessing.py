import logging
import warnings
import pandas
import numpy
import itertools


def define_lagged_feature(target_lags, lead_time, window_len):
    """ Defines windows for a lagged target feature.

    :return:
    """
    lagged_target_windows = []

    for lag in target_lags:
        if lead_time <= lag - window_len:
            lag -= lead_time
            lagged_target_windows.append((-lag, -lag + window_len - 1))
        elif lead_time < lag:
            lag -= lead_time
            end_index = min(-lag + window_len - 1, -1)
            lagged_target_windows.append((-lag, end_index))
            warnings.warn(f'Lag data not available for all target time periods. Lag: {lag}, lead_time: {lead_time},'
                          f' window len: {window_len}')
        else:
            warnings.warn('Data not available for the lag window [lag]. Omitting the feature')

    return lagged_target_windows


def basic_features(lead_time,
                   forecast_len,
                   features=['lagged_target', 'weekday', 'hour'],
                   target_lags=[7*24, 24],
                   target_lag_len=1,
                   temp_lag=0):

    recent_data = (-1, -1)
    lags = define_lagged_feature(target_lags, lead_time, target_lag_len)
    # add recent data only if not already present.
    end_periods = [window[1] for window in lags]
    if -1 not in end_periods:
        lags += [recent_data]

    lagged_target_feature = {'windows': lags,
                             'type': 'numeric'}

    weekday_feature = {'windows': [(lead_time, lead_time + forecast_len - 1)],
                       'type': 'cyclical'}

    hour_feature = {'windows': [(lead_time, lead_time + forecast_len - 1)],
                    'type': 'cyclical'}
    
    month_feature = {'windows': [(lead_time, lead_time + forecast_len - 1)],
                    'type': 'cyclical'}

    holiday_feature = {'windows': [(lead_time, lead_time + forecast_len - 1)],
                       'type': 'onehot'}

    out_temperature_feature = {'windows': [(lead_time - temp_lag, lead_time + forecast_len - 1)],
                               'type': 'numeric'}

    features_tmp = {'lagged_target': lagged_target_feature,
                    'weekday': weekday_feature,
                    'hour': hour_feature,
                    'month': month_feature, 
                    'holiday': holiday_feature,
                    't_out': out_temperature_feature}

    features = {f: features_tmp[f] for f in features}
    return features


def split_samples(data_obj, max_input, n=1):
    """Splits a data object into parts for each prediction.
    Assumes that first dimension is time.
    Use n to pick every nth sample

    Parameters
    ----------
    data_obj: pandas.DataFrame
    max_input: (int, int)
    n: int

    Returns
    -------
    list
        a list of data objects
    """
    logger = logging.getLogger(__name__)
    data_obj = data_obj.copy()  # To not mess with the original df.

    start_index, end_index = max_input
    sample_len = end_index - start_index + 1
    sample_cnt = data_obj.shape[0] - sample_len + 1  # assumes that 0 is time dimension

    data_obj_list = []
    for i in range(0, sample_cnt, n):
        if isinstance(data_obj, pandas.DataFrame) or isinstance(data_obj, pandas.Series):
            sample = data_obj[i:i + sample_len]
        else:
            sample = data_obj[i:i + sample_len, ...]
        data_obj_list.append(sample.copy())
        # Plot only 10 times to avoid spamming the logs with too many messages.
        plot_indices = list(itertools.islice(range(sample_cnt), 0, sample_cnt, max(sample_cnt // 10, 1)))
        if i in plot_indices:
            logger.info(f'split_samples() - {(i // n)+1}/{(sample_cnt // n)+1} processed.')

    return data_obj_list


def split_samples_with_feature_windows(dataframe, features, max_input, n=1):
    """This function uses split_samples() to split given dataframe into samples, but also applies feature windows to samples

    Parameters
    ----------
    dataframe: Dataframe to split
    features: Features dictionary with list of windows
    max_input: Maximum lag and future of features

    Returns
    -------
    list
        A list of dataframes where feature windows have been applied
    """

    # TODO: this function is very inefficient with large amount of samples
    logger = logging.getLogger(__name__)
    original_column_order = dataframe.columns
    samples_out = []
    max_lag, max_future = max_input
    samples = split_samples(dataframe, max_input, n)
    sample_cnt = len(samples)
    for i, sample in enumerate(samples):
        _sample_features = []
        for f in features.keys():
            s = pandas.Series(numpy.empty(sample.shape[0]) * numpy.nan, index=sample.index, name=f)
            for window in features[f]['windows']:
                idx = slice((window[0] - max_lag), (window[1] + 1 - max_lag))
                s[idx] = sample.loc[:, f].iloc[idx]
            _sample_features.append(s)
        samples_out.append(
            pandas.concat(_sample_features, axis=1).loc[:, original_column_order]
        )
        logger.debug(f'split_samples_with_feature_windows() {i + 1}/{sample_cnt} processed.')
    return samples_out


def split_samples_from_list(data_list, max_input):
    """Same as split_samples(), but takes a list of data objects and flattens the output

    Parameters
    ----------
        data_list: [object]
        max_input: (int, int)

    Returns
    -------
    list
        a list of data objects

    """
    samples_list = []
    for data in data_list:
        samples_list.extend(
            split_samples(data, max_input)
        )
    return samples_list


def temporal_features(lead_time, forecast_len):
    weekday_feature = {'windows': [(lead_time, lead_time + forecast_len - 1)],
                       'type': 'onehot'}

    hour_feature = {'windows': [(lead_time, lead_time + forecast_len - 1)],
                    'type': 'onehot'}

    holiday_feature = {'windows': [(lead_time, lead_time + forecast_len - 1)],
                       'type': 'onehot'}

    features = {'weekday': weekday_feature,
                'hour': hour_feature,
                'holiday': holiday_feature}
    return features
