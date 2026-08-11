import joblib
import pandas as pd
from pathlib import Path
import copy
import zipfile
import os
import shutil
from amp.constants import TEMPORAL

try:
    import tensorflow as tf
    from scikeras.wrappers import KerasRegressor
    tf_support = True
except ImportError:
    tf_support = False

from amp.efp.preprocessing import form_xy, form_x, form_x_list


class SingleForecaster(object):

    def __init__(self, target, lead_time, forecast_len, pipeline,
                 features, update_rate=1, freq=60, upper_limit=None, lower_limit=None):

        self.target = target
        self.lead_time = lead_time
        self.len = forecast_len
        self.update_rate = update_rate
        self.pipe = copy.deepcopy(pipeline) # todo now that multiforester has unique pipelines deepcopy can be probably removed. test
        self._features = features
        self.freq = freq
        self._upper_limit = upper_limit
        self._lower_limit = lower_limit

    def tf_filename(self, filepath):
        return Path(f'{filepath[:-4]}_tf_model.h5')

    @property
    def input_window(self):
        window_start = None
        window_end = None

        for label, f in self._features.items():
            for window in f['windows']:
                if (window_start is None) or (window[0] < window_start):
                    window_start = window[0]
                if (window_end is None) or (window[1] > window_end):
                    window_end = window[1]

        return tuple((window_start, window_end))

    @property
    def feature_types(self):
        f_types = list(self._features.keys())
        # Drop temporal features as they are created by the model
        temporal = TEMPORAL
        for t in temporal:
            if t in f_types:
                f_types.remove(t)

        if 'lagged_target' in f_types:
            f_types.remove('lagged_target')
            f_types.append(self.target)
        return f_types

    def train(self, train_df):
        train_df = train_df.copy()  # avoid modifying the original df
        X, y = form_xy(train_df, self.target, self.lead_time, self.len, self._features, self.update_rate)
        self.pipe.fit(X, y)
        features = self.pipe.named_steps['preprocessing'].get_feature_names_out()
        print('Trained with features:', features)

    def multi_predict(self, input_df_list):
        """

        Parameters
        ----------
        input_df_list

        Returns
        -------

        """
        x_list_new = form_x_list(input_df_list, self.target, self._features, self.update_rate)

        X = pd.concat(x_list_new, axis=0, ignore_index=True)

        forecast = self.pipe.predict(X)

        if self._lower_limit is not None:
            forecast[forecast < self._lower_limit] = self._lower_limit

        if self._upper_limit is not None:
            forecast[forecast > self._upper_limit] = self._upper_limit

        # Convert the array into a list of pd.DataFrames
        forecast_df_list = [pd.DataFrame(data=row, index=input.index[-self.len:], columns=['forecast'])
                            for row, input in zip(forecast, input_df_list)]
        return forecast_df_list

    def predict(self, input_df, output_format='target'):
        """

        Parameters
        ----------
        input_df        : pd.DataFrame
                          DataFrame of input features. Different features are stored in separate columns.
        output_format   : str
                          Output format. Supported formats are 'target', 'fcast' and 'single'

        Returns
        -------
        pd.DataFrame
            Forecast in the format specified by the output_format.

        """
        input_df = input_df.copy()

        X = form_x(input_df, self.target, self._features, self.update_rate)
        if X.index.freq is None:
            X = X.asfreq(f'{self.freq}min', method='ffill')
        forecast = self.pipe.predict(X)
        forecast = forecast.reshape(-1, self.len)  # This is needed for single predictions.

        if self._lower_limit is not None:
            forecast[forecast < self._lower_limit] = self._lower_limit

        if self._upper_limit is not None:
            forecast[forecast > self._upper_limit] = self._upper_limit

        columns = [f'forecast_{i}' for i in range(self.lead_time, self.lead_time + self.len)]

        # Form a DF where the index is the forecast time. todo the lead time shift might brake this. test.
        ftime_df = pd.DataFrame(data=forecast, index=X.index, columns=columns)
        if output_format == 'fcast':
            return ftime_df
        elif output_format == 'target' or output_format == 'single':
            # Crete a DF for the forecast results. Data frequency (i.e., sampling rate) is used as the index freq
            # instead of the update freq. This means that there might be nan values if update freq is larger
            # than the sampling rate.
            fcast_index = pd.DatetimeIndex(data=pd.date_range(X.index[0],
                                                              X.index[-1]+pd.Timedelta(minutes=(self.lead_time+self.len-1)*self.freq),
                                                              freq=f'{self.freq}min'))

            forecast_df = pd.DataFrame(index=fcast_index, columns=columns)

            # Shift forecast to the right place.
            for i in range(self.lead_time, self.lead_time + self.len):
                forecast_df[f'forecast_{i}'] = ftime_df[[f'forecast_{i}']]
                forecast_df[f'forecast_{i}'] = forecast_df[f'forecast_{i}'].shift(i)

            if output_format == 'single':
                if (forecast_df.count(axis='columns') > 1).any():
                    # TODO: Need to be tested how current apps work. Check that they dont produce multiple values for row
                    # Check especially max_lag and max_future and input_df length based on those
                    raise ValueError('Forecast has overlapping values. Cannot produce single output')

                # Select the only value.
                forecast_df['forecast'] = forecast_df.max(axis=1)
                forecast_df = forecast_df[['forecast']]
        else:
            raise ValueError(f'Invalid output format: {output_format}')

        return forecast_df[self.lead_time:]

    def save(self, filepath):
        # todo it might sense to keep this simple and reimplement this method for tensorflow based models in models.tf
        #  module there is a catch though as it does not implement this method directly. Put should be possible
        #  in any case.
        model_files = {}

        pickle_file = Path(f'{str(filepath)[:-4]}.pkl')
        zip_file = Path(filepath)
        if tf_support is True:
            if isinstance(self.pipe.named_steps.model, KerasRegressor):
                tf_file = self.tf_filename(filepath)
                # This hack allows us to save the sklearn pipeline with tf.keras model.
                # They have to be saved as separate files
                tf.keras.models.save_model(self.pipe.named_steps.model.model, tf_file)
                model_files['tf_model.tf'] = tf_file

                model_tmp = self.pipe.named_steps.model.model
                self.pipe.named_steps.model.model = None
                joblib.dump(self, pickle_file)
                self.pipe.named_steps.model.model = model_tmp
            else:
                joblib.dump(self, pickle_file)
        else:
            joblib.dump(self, pickle_file)

        model_files['main.pkl'] = pickle_file

        # Add all files to a zip file.
        with zipfile.ZipFile(zip_file, 'w') as zip_file:
            for model_name, m_file in model_files.items():
                zip_file.write(m_file, arcname=model_name)

        # Remove the original files.
        for m_file in model_files.values():
            os.remove(m_file)

    @classmethod
    def load(cls, filepath):
        extracted_dir = Path('extracted_files') # Temporary directory to extract the files

        # Step 1: Extract the contents of the zip file
        with zipfile.ZipFile(filepath, 'r') as zip_file:
            zip_file.extractall(extracted_dir)

        forecaster = joblib.load(extracted_dir.joinpath('main.pkl'))

        if hasattr(forecaster.pipe.named_steps.model, 'model') and (
                forecaster.pipe.named_steps.model.model is None) and tf_support is True:  # if the model is None, this is hoas tf.keras model.
            #forecaster.pipe.named_steps.model.model = tf.keras.models.load_model(forecaster.tf_filename(filepath))
            forecaster.pipe.named_steps.model.model = tf.keras.models.load_model(extracted_dir.joinpath('tf_model.tf'))

        # Remove the extracted files.
        shutil.rmtree(extracted_dir)

        return forecaster


class MultiForecaster(SingleForecaster):

    def __init__(self, target, lead_time, forecast_len, pipelines, features,
                 update_rate=1, freq=60, upper_limit=None, lower_limit=None):

        self.target = target
        self.lead_time = lead_time
        self.len = forecast_len
        self.update_rate = update_rate
        self.freq = freq
        self.feature_list = features

        # Create single forecasters for each period.
        self.forecasters = []
        for i in range(forecast_len):
            self.forecasters.append(SingleForecaster(target,
                                                     lead_time + i,
                                                     1,
                                                     pipelines[i],
                                                     features[i],
                                                     update_rate,
                                                     freq,
                                                     upper_limit,
                                                     lower_limit))

    @property
    def feature_types(self):
        return self.forecasters[0].feature_types

    @property
    def input_window(self):
        window_start = None
        window_end = None

        for forecaster in self.forecasters:
            start, end = forecaster.input_window

            if (window_start is None) or (start < window_start):
                window_start = start
            if (window_end is None) or (end > window_end):
                window_end = end

        return tuple((window_start, window_end))

    def train(self, train_df):
        for f in self.forecasters:
            f.train(train_df)

    def predict(self, input_df, single_output=False):
        preds = []

        total_start, total_end = self.input_window

        for f in self.forecasters:
            start, end = f.input_window
            sliced_input = input_df.iloc[-total_start + start: -total_end + end]
            pred = f.predict(sliced_input, single_output)
            preds.append(pred)

        if not single_output:
            forecast = pd.concat(preds, axis=1)
        else:
            forecast = pd.concat(preds, axis=0)

        return forecast

    def save(self, filepath):

        # todo modify so that the model is stored into a single file (e.g. zip)
        # Test whether the models are tf.keras by inspecting the first forecaster.
        if (isinstance(self.forecasters[0].pipe.named_steps.model, tf.keras.wrappers.scikit_learn.KerasRegressor)
                and tf_support is True):
            # Below is hoas hack for tf.keras models
            models_tmp = []
            for f in self.forecasters:
                tf.keras.models.save_model(f.pipe.named_steps.model.model, self.tf_filename(filepath))

                models_tmp.append(f.pipe.named_steps.model.model)
                f.pipe.named_steps.model.model = None

            forecaster_file = Path(filepath)
            joblib.dump(self, forecaster_file)
            # Assign the models back so the forecaster can be used.
            for i in range(len(models_tmp)):
                self.forecasters[i].pipe.named_steps.model.model = models_tmp[i]
        else:
            forecaster_file = Path(filepath)
            joblib.dump(self, forecaster_file)

    @classmethod
    def load(cls, filepath):
        forecaster = joblib.load(filepath)
        # if the model is None, this is hoas tf.keras model and the models need to be loaded separately.
        if (hasattr(forecaster.forecasters[0].pipe.named_steps.model, 'model') and\
                (forecaster.forecasters[0].pipe.named_steps.model.model is None)
                and tf_support is True):
            for f in forecaster.forecasters:
                f.pipe.named_steps.model.model = tf.keras.models.load_model(forecaster.tf_filename(filepath))
        return forecaster
