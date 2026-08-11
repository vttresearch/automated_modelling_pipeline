import pandas as pd
from darts.models import NBEATSModel
import darts
import logging
from pathlib import Path
import joblib
import warnings
import zipfile
import os
import shutil

from amp.base import BaseModel
from amp.constants import TEMPORAL

def darts_filepath(filepath, output):
    return str(Path(f'{str(filepath)[:-4]}_{output}_darts.pt'))


# todo maybe create a Darts specific forecaster that is inherited. Can be also inherited by other classes.
class DartsForecaster(BaseModel):

    #def __init__(self, name, targets, lead_time, forecast_len, data_freq, model=None, features=None, update_freq=1):
    def __init__(self, targets, lead_time, forecast_len, data_freq, model=None, features=None, update_freq=1):
        self.logger = logging.getLogger('ema')
        #self.name = name
        self.forecast_len = forecast_len
        self.lead_time = lead_time
        self.outputs = targets
        self.data_freq = data_freq
        self.update_rate = update_freq
        self.models = {}
        self._history_len = 0
        self._features = features
        #self._model = model

    def predict(self, df, output_mode='target'):
        # todo parse df into past and future covariates.

        # todo create darts timeseries from the DF.
        for m in self.models:
            m.predict()

    def fit(self, df):
        # todo create darts timeseries from the DF.
        ts = darts.timeseries.from_dataframe(df)
        self._dmodel.fit(ts)

    def save(self, filename):
        # Todo test will not probably work with darts.
        forecaster_file = Path(f'{filename}')
        joblib.dump(self, forecaster_file)

    @classmethod
    def load(cls, filename):
        # Todo test will not probably work with darts.
        return joblib.load(filename)


#class NBEATSForecaster(BaseModel):

class NBEATSMulti(object):

    def __init__(self, outputs, lead_time, forecast_len, data_freq, past_covariates=[], update_freq=1, **model_params):
        self.logger = logging.getLogger('ema')
        self.forecast_len = forecast_len
        self.lead_time = lead_time
        self.outputs = outputs
        self.data_freq = data_freq
        self.update_rate = update_freq
        self.models = {}

        pl_trainer_kwargs = {"accelerator": "cpu"}


        if len(model_params) == 0:
            for o in outputs:
                self.models[o] = NBEATSModel(input_chunk_length=24,
                                             output_chunk_length=lead_time+forecast_len,
                                             n_epochs=1,
                                             pl_trainer_kwargs=pl_trainer_kwargs)
        else:
            for o in outputs:
                input_length = model_params.pop('input_chunk_length', 24)
                self.models[o] = NBEATSModel(input_chunk_length=input_length,
                                             output_chunk_length=lead_time+forecast_len,
                                             pl_trainer_kwargs=pl_trainer_kwargs,
                                             **model_params,)

        self._features_types = past_covariates
        self._past_covariates = past_covariates
        self._future_covariates = ['t_out']

    def _form_input(self, model, output, df):
        #series = df[self.outputs]  # this version could be used for a single model that predicts all outputs
        series = df[[output]]
        series = darts.timeseries.TimeSeries.from_dataframe(series)
        series = darts.utils.missing_values.extract_subseries(series, min_gap_size=1)
        series = [s for s in series if len(s) >= model.input_chunk_length + model.output_chunk_length]
        if len(self._past_covariates) > 0:
            past_covariates = df[self._past_covariates]
            past_covariates = darts.timeseries.TimeSeries.from_dataframe(past_covariates)
            past_cov_list = []
            for s in series:
                past_cov_list.append(past_covariates.loc[s.index])
            past_covariates = past_cov_list
        else:
            past_covariates = None

        if len(self._future_covariates) > 0:
            future_covariates = df[self._future_covariates]
            future_covariates = darts.timeseries.TimeSeries.from_dataframe(future_covariates)
        else:
            future_covariates = None

        return series, past_covariates, future_covariates

    @property
    def feature_types(self):
        f_types = self._features_types

        # Drop temporal features as they are created by the model
        temporal = TEMPORAL
        for t in temporal:
            if t in f_types:
                f_types.remove(t)
        return f_types

    @property
    def input_window(self):
        # Assumes that the input and output lenghts are the same for all outputs.
        window_start = -self.models[self.outputs[0]].input_chunk_length
        window_end = self.models[self.outputs[0]].output_chunk_length
        return tuple((window_start, window_end))

    def _single_predict(self, model, output, df, output_mode='target'):
        df = df.resample(f'{self.data_freq}min').mean()
        series, past_covariates, _ = self._form_input(model, output, df)
        # Darts does not support lead time so we have to predict the lead time + forecast len and cut of the lead time.
        predict_steps = self.lead_time + self.forecast_len
        if output_mode in ['target', 'fcast']:
            pred = model.historical_forecasts(series=series, past_covariates=past_covariates,
                                              forecast_horizon=predict_steps, retrain=False,
                                              last_points_only=False,
                                              stride=self.update_rate)

            columns = [f'forecast_{i}' for i in range(self.lead_time, self.lead_time + self.forecast_len)]
            preds = []
            for p in pred:
                pred_df_tmp = p.pd_dataframe()
                pred_df = pred_df_tmp.tail(self.forecast_len).T
                pred_df.columns = columns
                pred_df.index = [pred_df_tmp.index[0]]
                preds.append(pred_df)
            preds_df = pd.concat(preds)

            if output_mode == 'fcast':
                return preds_df
            else:
                # Crete a DF for the forecast results. Data frequency (i.e., sampling rate) is used as the index freq
                # instead of the update freq. This means that there might be nan values if update freq is larger
                # than the sampling rate.
                fcast_index = pd.DatetimeIndex(data=pd.date_range(preds_df.index[0],
                                                                  preds_df.index[-1]+pd.Timedelta(minutes=(self.lead_time+self.forecast_len-1)*self.data_freq),
                                                                  freq=f'{self.data_freq}min'))

                forecast_df = pd.DataFrame(index=fcast_index, columns=columns)

                # Shift forecast to the right place.
                for i in range(self.lead_time, self.lead_time + self.forecast_len):
                    forecast_df[f'forecast_{i}'] = preds_df[[f'forecast_{i}']]
                    forecast_df[f'forecast_{i}'] = forecast_df[f'forecast_{i}'].shift(i)

                return forecast_df.iloc[self.lead_time:]

        elif output_mode == 'single':
            pred = model.predict(n=predict_steps, series=series, past_covariates=past_covariates)
        else:
            raise ValueError(f'Invalid output mode: {output_mode}')

        return pred

    def predict(self, df, output_mode='target'):
        pred = {}
        for output in self.outputs:
            pred[output] = self._single_predict(self.models[output], output, df, output_mode)

        if output_mode == 'single':
            # todo test this
            pred = {key: data['forecast'] for (key, data) in pred.items()}
            pred = pd.DataFrame.from_dict(pred)
        return pred

    def fit(self, df):
        df = df.resample(f'{self.data_freq}min').mean()
        for o, m in self.models.items():
            series, past_covariates, _ = self._form_input(m, o, df)
            m.fit(series, past_covariates=past_covariates)

    def save(self, filepath):

        if str(filepath)[-4:] != '.zip':
            warnings.warn('Invalid file format for the model. Only zip supported.')
            filepath = str(filepath) + '.zip'

        # Darts model save does not support Path object so we have to cast it to str.
        tmp_models = {}
        model_files = {}
        for output, model in self.models.items():
            darts_file = darts_filepath(filepath, output)
            model.save(darts_file)
            tmp_models[output] = model
            self.models[output] = None
            model_files[f'{output}_model.pt'] = darts_file

            # The ckpt file seems to be needed as well so let's add it.
            model_files[f'{output}_model.pt.ckpt'] = darts_file + '.ckpt'

        pickle_file = Path(f'{str(filepath)[:-4]}.pkl')
        joblib.dump(self, pickle_file)

        model_files['main.pkl'] = pickle_file

        # Add all files to a zip file.
        with zipfile.ZipFile(filepath, 'w') as zip_file:
            for name, m_file in model_files.items():
                zip_file.write(m_file, arcname=name)

        # Remove the original files.
        for m_file in model_files.values():
            os.remove(m_file)

        # Restore the models so that they can be used after saving.
        self.models = tmp_models

    @classmethod
    def load(cls, filepath):

        if str(filepath)[-4:] != '.zip':
            warnings.warn('Invalid file format for the model. Only zip supported.')
            filepath = str(filepath) + '.zip'

        extracted_dir = Path(filepath).parent.joinpath('extracted_files')

        # Extract the contents of the zip file
        with zipfile.ZipFile(filepath, 'r') as zip_file:
            zip_file.extractall(extracted_dir)

        # Load the Forecaster object.
        forecaster = joblib.load(extracted_dir.joinpath('main.pkl'))
        forecaster.model = {}

        # Darts model save does not support Path object so we have to cast it to str.
        for output in forecaster._outputs:
            forecaster.models[output] = NBEATSModel.load(str(extracted_dir.joinpath(f'{output}_model.pt')))

        # Remove the extracted files.
        shutil.rmtree(extracted_dir)

        return forecaster
