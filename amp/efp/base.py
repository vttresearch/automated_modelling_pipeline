import warnings

import joblib
import logging
import zipfile
import os
import shutil
from pathlib import Path
import pandas as pd

from amp.efp.forecaster import SingleForecaster, MultiForecaster
from amp.base import BaseModel, ScenarioMixin
from amp.base import Target

class Forecaster(ScenarioMixin, BaseModel):

    def __init__(self, targets, lead_time, forecast_len, data_freq, model=None,
                 features=None, hyperparams=None, update_freq=1, upper_limits=None, lower_limits=None):
        """
        A forecasting model that predicts future values based on historical data and specified features.

        This class supports single and multi-step forecasting using machine learning models. It manages
        multiple target variables, feature sets, and model parameters to train and generate predictions.

        Parameters
        ----------
        targets : list of Target
        A list of `Target` objects defining the forecasted variables. Each target includes:
        - `output` (str): The name of the output variable used by the model.
        - `eval_column` (str): The dataset column corresponding to the target variable.
        - `plot_label` (str): A label used for visualization.
        - `eval_scaler` (int, default=1): A scaling factor for evaluation.

        lead_time : int
        The number of time steps ahead the forecast starts. A lead time of `0` means an immediate forecast.

        forecast_len : int
        The number of future time steps the model predicts in one run.
        **Example:** `96` (predicts 96 time steps into the future).

        data_freq : int
        The time interval between data points (e.g., `15` means data is collected every 15 minutes).

        model : method or object, optional
        The forecasting method or pipeline used for prediction.
        In this case, it's a bound method from `LightGBMForecaster._pipeline`.

        features : dict
        A dictionary defining the features used for forecasting. Example structure:
        ```
        {
            'Ti_heating': {'type': 'numeric', 'windows': [(0, 95)]},
            'lagged_target': {'type': 'numeric', 'windows': [(-24, -1)]}
        }
        ```
        - Each key represents a feature name.
        - `type` specifies the data type (e.g., numeric).
        - `windows` define historical time ranges used as input.

        hyperparams : dict, optional
        Hyperparameters for the machine learning model. Example:
        ```
        {'verbose': 3}
        ```

        update_freq : int, default=1
        The frequency at which the model updates its predictions.

        upper_limits : dict or None, optional
        Upper limit constraints for the predictions. If provided, it should be a dictionary
        mapping target names to their respective limits.

        lower_limits : dict or None, optional
        Lower limit constraints for the predictions. Similar to `upper_limits`.

        Attributes
        ----------
        models : dict
        A dictionary storing the forecasting models for each target.

        _ml_method : method or object
        The machine learning model or pipeline used for forecasting.

        _features : dict
        The features extracted from the dataset for training.

        _hyperparams : dict
        The hyperparameter configuration used for model training.

        _upper_limits : dict or None
        Upper limit constraints for the predictions.

        _lower_limits : dict or None
        Lower limit constraints for the predictions.

        Methods
        -------
        fit(df)
        Trains the forecasting model using the provided DataFrame.

        predict(df, output_mode='target')
        Generates forecasts using the trained model.

        multi_predict(df_list)
        Generates multiple forecasts from different input scenarios.

        save(filename)
        Saves the trained model and configurations into a ZIP file.

        load(filepath)
        Loads a saved model from a ZIP file.

        set_params(**params)
        Updates model parameters and reinitializes the model.

        feature_window(fname)
        Returns the time window range for a specific feature.

        input_window
        Retrieves the input window size used for feature extraction.

        """

        self.logger = logging.getLogger('ema')
        self.forecast_len = forecast_len
        self.lead_time = lead_time
        self._outputs = targets
        self.data_freq = data_freq
        self.update_rate = update_freq
        self.models = {}
        self._ml_method = model
        self._features = features
        self._hyperparams = hyperparams
        self._upper_limits = upper_limits
        self._lower_limits = lower_limits
        self._create_model()

    def _create_model(self):
        if (self._ml_method is not None) and (self._features is not None):
            for target in self.outputs:
                if self._upper_limits is not None:
                    upper_limits = self._upper_limits.get(target)
                else:
                    upper_limits = None
                if self._lower_limits is not None:
                    lower_limits = self._lower_limits.get(target)
                else:
                    lower_limits = None
                if isinstance(self._features, list) and len(self._features) > 1:
                    pipelines = [self._ml_method(1, self._features[i], **self._hyperparams) if
                                 self._hyperparams else self._ml_method(1, self._features[i]) for i in range(self.forecast_len)]
                    self.models[target] = MultiForecaster(target,
                                                          self.lead_time,
                                                          self.forecast_len,  # todo is this a bug? Should it be 1? CHECK
                                                          pipelines,
                                                          self._features,
                                                          self.update_rate,
                                                          self.data_freq,
                                                          upper_limits,
                                                          lower_limits)
                else:
                    if isinstance(self._features, list):
                        features = self._features[0]
                    else:
                        features = self._features
                    full_pipeline = self._ml_method(self.forecast_len, features, **self._hyperparams) if \
                        self._hyperparams else self._ml_method(self.forecast_len, features)
                    self.models[target] = SingleForecaster(target,
                                                           self.lead_time,
                                                           self.forecast_len,
                                                           full_pipeline,
                                                           features,
                                                           self.update_rate,
                                                           self.data_freq,
                                                           upper_limits,
                                                           lower_limits)

    def set_params(self, **params):
        """ Set parameters of the model.

        Parameters
        ----------
        params : dict

        Returns
        -------

        """
        self._features = params['features']
        self._ml_method = params['model']
        self._create_model()

    def feature_window(self, fname):
        window_start = None
        window_end = None

        # todo think how to implement without need to access hidden variable
        f = list(self.models.values())[0]._features[fname]

        for window in f['windows']:
            if (window_start is None) or (window[0] < window_start):
                window_start = window[0]
            if (window_end is None) or (window[1] > window_end):
                window_end = window[1]

        return tuple((window_start, window_end))

    @property
    def input_window(self):
        # todo this should take all the models into account. Works now as all models take the same features but this
        #  should be changed soon.
        return list(self.models.values())[0].input_window

    def predict_scenarios(self, df_list):
        return self.multi_predict(df_list)

    def multi_predict(self, df_list):
        preds = []
        for output in self.outputs:
            if isinstance(output, Target):
                output = output.output
            pred_list = self.models[output].multi_predict(df_list)

            if len(preds) == 0:
                preds = [pred.rename(columns={'forecast': output}) for pred in pred_list]
            else:
                for i in range(0, len(preds)):
                    preds[i][output] = pred_list[i]['forecast']

        return preds

    def predict(self, df, output_mode='target'):
        pred = {}
        for output in self.outputs:
            if isinstance(output, Target):
                output = output.output
            pred[output] = self.models[output].predict(df, output_mode)

        if output_mode == 'single':
            pred = {key: data['forecast'] for (key, data) in pred.items()}
            pred = pd.DataFrame.from_dict(pred)
        return pred

    def fit(self, df):
        """ Fit/train the model on the provided DataFrame.

        The method uses information about features and outputs to parse the DataFrame into samples used for fitting the
        model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing all required data for fitting the model.

        Returns
        -------

        """
        self.logger.info(f'Fitting using features:\n')
        self.logger.info(f"\n{pd.DataFrame.from_dict({k: v for k, v in self._features.items()}, orient='index')}")
        # print train data range and number of samples
        self.logger.info(f'Training data range: {df.index.min()} to {df.index.max()} ({len(df)} samples)')

        for target in self.outputs:
            self.logger.info(f'Fitting model for {target}')
            self.models[target].train(df)

    def save(self, filename):
        """ Save the model as a zip file.

        The zip file includes a pickle of this class, a pickle for each SingleForecaster/MultiForecaster and a
        model specific format (e.g. pkl, tf, pt) for the actual model.

        TODO: need to package the MultiForecaster as a zip file as well.

        Parameters
        ----------
        filename

        Returns
        -------

        """
        if str(filename)[-4:] != '.zip':
            warnings.warn('Invalid file format for the model. Only zip supported.')
            filename = str(filename) + '.zip'

        model_files = {}
        for target in self.outputs:
            sub_file = f'{str(filename)[:-4]}_{target}_model.zip'
            self.models[target].save(sub_file)
            model_files[f'{target}_model.zip'] = sub_file

        pickle_file = f'{str(filename)[:-4]}.pkl'
        joblib.dump(self, pickle_file)
        model_files['main.pkl'] = pickle_file

        # Add all files to a zip file.
        with zipfile.ZipFile(filename, 'w') as zip_file:
            for name, m_file in model_files.items():
                zip_file.write(m_file, arcname=name)

        # Remove the original files.
        for m_file in model_files.values():
            os.remove(m_file)

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

        # Load the model for each output.
        for target in forecaster.outputs:
            if isinstance(forecaster.models[target], MultiForecaster):
                forecaster.models[target] = MultiForecaster.load(extracted_dir.joinpath(f'{target}_model.zip'))
            elif isinstance(forecaster.models[target], SingleForecaster):
                forecaster.models[target] = SingleForecaster.load(extracted_dir.joinpath(f'{target}_model.zip'))
            else:
                raise ValueError('Invalid model instance')

        # Remove the extracted files.
        shutil.rmtree(extracted_dir)
        return forecaster
