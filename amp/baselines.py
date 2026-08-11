import pandas as pd
import warnings

class LagBaseline(object):
    """

    """

    def __init__(self, lag):
        """

        Parameters
        ----------
        lag length of the lag in int
        """
        self.lag = lag

    def predict(self, window, target_series):
        """

        Parameters
        ----------
        window
        target_series

        Returns
        -------

        """
        series = target_series.shift(self.lag)
        series = series.loc[window]
        series = series.dropna()
        return series


class MeanBaseline(object):

    def predict(self, window, target_series):
        """

        Parameters
        ----------
        window
        target_series

        Returns
        -------

        """
        return pd.Series(index=window, data=target_series.mean())


class ExistingBaseline(object):

    def __init__(self, series):
        """

        Parameters
        ----------
        series
        """
        self.series = series

    def predict(self, window, target_series):
        """

        Parameters
        ----------
        window
        target_series

        Returns
        -------

        """
        return self.series.loc[window]


class FeatureBaseline:
    """
    A simple baseline predictor that uses a specified feature from the input series for prediction.
    Can be used in error correction models where one of the features
    """

    def __init__(self, feature):
        """
        Initialize the baseline predictor with the target feature.

        Parameters
        ----------
        feature : str
            The name of the target feature to use for prediction.
        """
        self.feature = feature

    def predict(self,
                window,
                target_series):
        """
        Predict the target series using the baseline method (using the same values from the input series).

        Parameters
        ----------
        window : pd.DatetimeIndex
            The datetime index for which to make predictions.
        target_series : pd.Series, optional
            Required. The input data used for prediction.

        Returns
        -------
        predicted_values : pd.Series
            A Series with the same values as `self.feature` at the corresponding indices from 'window'.
        """
        if target_series is None:
            raise ValueError("Target series cannot be None.")

        return target_series.loc[window]

