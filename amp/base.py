from abc import ABC, abstractmethod
import joblib
from pathlib import Path
import zipfile
import os, stat
import warnings
import shutil
import sys
from amp.constants import TEMPORAL


class Target:
    def __init__(self, output, eval_column=None, plot_label=None, eval_scaler=None):
        """
        A class to define a forecasting target.

        Parameters
        ----------
        output : str
            The name of the output in the model.
        eval_column : str
            The column name in the DataFrame that represents the target variable.
        plot_label : str
            The name used in the plotting (real, measured value that was forecasted).
        eval_scaler : optional
            A scaler to be used in evaluation (i.e. giving weight for the target.)
        """
        self._output = output
        self._eval_column = eval_column if eval_column else output
        self._plot_label = plot_label if plot_label else f"{output}_measured"
        self._eval_scaler = eval_scaler if eval_scaler is not None else 1  # Optional scaling

    def __repr__(self):
        return (f"Target(output={self._output}, "
                f"column={self._eval_column}, "
                f"plot_label={self._plot_label}, "
                f"scaler={self._eval_scaler})")

    @property
    def output(self):
        """ Return the output name """
        return self._output

    @property
    def column(self):
        """ Return the column name """
        return self._eval_column

    @property
    def plot_label(self):
        """ Return the plot label """
        return self._plot_label

    @property
    def scaler(self):
        """ Return the scaler value """
        return self._eval_scaler

    def to_dict(self):
        """ Return target data as a dictionary """
        return {
            'column': self._eval_column,
            'output': self._output,
            'plot_label': self._plot_label,
            'scaler': self._eval_scaler
        }

def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)




def load_model(path):
    """ Load a model from the specified path.

    Parameters
    ----------
    path : str
        Path to the model file. Can be a local file path or a URL.

    Returns
    -------
    BaseModel
        An instance of the loaded model.

    Notes
    -----
    - If the path is a URL, the model is loaded from an MLflow server.
    - If the path is a local file, it must be a `.zip` file containing the model.
    """

    if path.startswith("http://") or path.startswith("https://"):
        # Parse server and model from URL
        import mlflow
        from amp.mlflow_utils import ForecasterWrapper
        from urllib.parse import urlparse
        parsed = urlparse(path)
        server_uri = f"{parsed.scheme}://{parsed.netloc}"
        model_name = parsed.fragment.lstrip('/models/')  # remove leading '/'
        mlflow.set_tracking_uri(server_uri)
        model_uri = f"models:/{model_name}"
        print(f"Loading model from MLflow: {model_uri} (server: {server_uri})")
        return ForecasterWrapper.load_model_with_metadata(model_uri)

    if str(path)[-4:] != '.zip':
        warnings.warn('Invalid file format for the model. Only zip supported.')
        path = str(path) + '.zip'

    extracted_dir = Path(path).parent.joinpath('extracted_files')

    # Extract the contents of the zip file
    with zipfile.ZipFile(path, 'r') as zip_file:
        zip_file.extractall(extracted_dir)

    # Load the Forecaster object.
    forecaster = joblib.load(extracted_dir.joinpath('main.pkl'))
    forecaster.model = {}

    # Remove the extracted files.
    # shutil does not have 'onexc' argument in older python versions
    if sys.version_info >= (3, 12):
        # TODO: is there a better fix?
        shutil.rmtree(extracted_dir, onexc=remove_readonly)
    else:
        shutil.rmtree(extracted_dir)

    # Call the actual method of the subclass
    return type(forecaster).load(Path(path))


class BaseModel(ABC):

    def __init__(self, targets, features):
        """

        Parameters
        ----------
        targets : list of Target class [Target(), Target()]
            List of outputs produced by the model.
        features : dict or list of dict
            e.g. {'lagged_target':{'windows': [(-12, -1), (-156, -133)]},
                  'weekday': {'windows': [(12, 35)]},
                  'hour': {'windows': [(12, 35)]},
                  't_out': {'windows': [(12, 35)]}}

        Returns
        -------

        """
        self._outputs = targets
        self._features = features

    @property
    def outputs(self):
        """
        Return list of output identifiers (e.g. ['dh', 'ele']).
        """
        # already a list of strings, old models
        # TODO this should be fixed from HOAS model to use Target objects and remove this check
        if isinstance(self._outputs, list) and all(isinstance(o, str) for o in self._outputs):
            return self._outputs

        # list/set of Target objects
        outputs = self._outputs
        return list({output.output for output in outputs})

    @property
    def features(self):
        return self._features

    @property
    def feature_types(self):
        """ Provides the feature types required by the model.

        Returns
        -------
        list of str
            list of feature types.

        """
        if isinstance(self._features, list):
            f_types = list(self._features[0].keys())
        else:
            f_types = list(self._features.keys())
        # Drop temporal features as they are created by the model
        temporal = TEMPORAL
        for t in temporal:
            if t in f_types:
                f_types.remove(t)

        if 'lagged_target' in f_types:          # todo should this and the above be moved to efp.base as they are used there?
            f_types.remove('lagged_target')
            f_types += self.outputs
        return f_types

    def feature_window(self, fname):
        """ Provides input window for the given feature.

        Input window is a tuple of two values. The first value defines the start of the window. The second value
        defines the end of the window. Indexing works as follows. The indexes are represented with respect to the
        forecast (current) time. The values indicate the start of the time period that is required for the forecast.

        E.g. Window (0, 23) means 24 values starting from now (0) and lasting 24 steps int to future (23 indicates the
        start of period from 23-24). With 60-min sampling rate this would mean 24 hours data. Window (-24,-1) in turn
        means 24 past values.

        Parameters
        ----------
        fname : str
            Name of the feature.

        Returns
        -------
        tuple of (int, int)

        """
        window_start = None
        window_end = None

        f = self._features[fname]
        for window in f['windows']:
            if (window_start is None) or (window[0] < window_start):
                window_start = window[0]
            if (window_end is None) or (window[1] > window_end):
                window_end = window[1]

        return tuple((window_start, window_end))

    @property
    def input_window(self):
        """ The maximum input window for the model.

        Returns the maximum input window among all features.

        Returns
        -------
        tuple of (int, int)

        """
        window_start = 0
        window_end = 0

        for label in self._features.keys():
            window = self.feature_window(label)
            if window[0] < window_start:
                window_start = window[0]
            if window[1] > window_end:
                window_end = window[1]

        return tuple((window_start, window_end))

    # todo can this be removed?
    # def set_params(self, **params):
    #     """ Set parameters of the model.
    #
    #     Parameters
    #     ----------
    #     params : dict
    #
    #     Returns
    #     -------
    #
    #     """
    #     self._features = params['features']

    @abstractmethod
    def predict(self, input_df, output_mode='target'):
        """ Makes prediction(s) based on the input data.

        Parameters
        ----------
        input_df : pd.DataFrame
            inputs (features) required for the prediction(s).
        output_mode : {'single', or 'target}
            Output mode.
            - If 'single': a single forecast is returned as a pd.DataFrame. Assumes that the input_df contains data for
            a single forecast.
            - If 'target': parses the input_df into samples and performs a forecast for each sample. The forecasts are
            returned as a dict of pd.DataFrame. Each output of the model is a separate DataFrame in the dict with keys
            being the outputs. The DataFrame index is organised based on forecast target period. I.e., every row
            contains the forecast for that period. The columns specify when the forecast was made in format
            forecast_<steps_to_to_start>. E.g. forecast_5 means that the forecast was made 6 steps in the past
            (5 steps to the start and 6 to the end). I.e., with 60-min sampling rate this means that the forecast was
            made 6-hours ago.

        Returns
        -------
        pd.DataFrame or dict of pd.dataFrame
            return type depends on the output_mode parameter as specified above.
        """
        pass

    @abstractmethod
    def fit(
            self,
            training_set,
            validation_set=[]
    ):
        """ Fit/train the model on the provided DataFrame.

        Parameters
        ----------
        training_set : [pd.DataFrame]
            DataFrame containing all required data for fitting the model.
        validation_set: [pd.DataFrame], default: None
            DataFrame containing all required data for validating the model during fitting.

        Returns
        -------

        """
        pass

    def save(self, filename):
        """ Saves the model as a zip file with the given filename.

        Zip format is used instead of pickle because of it allows all types of models to be packaged as a single file.
        This is important from the deployment point of view.

        Parameters
        ----------
        filename : str
            Filename (or path).

        Returns
        -------

        """
        if str(filename)[-4:] != '.zip':
            warnings.warn('Invalid file format for the model. Only zip supported.')
            filename = str(filename) + '.zip'

        pickle_file = Path(f'{str(filename)[:-4]}.pkl')
        zip_file = Path(filename)

        joblib.dump(self, pickle_file)

        # Add the pickle file to the zip file.
        with zipfile.ZipFile(zip_file, 'w') as zip_file:
            zip_file.write(pickle_file, 'main.pkl')

        # Remove the original file.
        os.remove(pickle_file)

    @classmethod
    def load(cls, filename):
        """ Loads a model from the specified filename or URL.

        Parameters
        ----------
        filename : str
            Path to the model file. Can be a local file path or a URL.

        Returns
        -------
        BaseModel
            An instance of the loaded model.

        Notes
        -----
        - If the filename is a URL, the model is loaded from an MLflow server.
        - If the filename is a local file, it must be a `.zip` file containing the model.
        """

        if str(filename).startswith("http://") or str(filename).startswith("https://"):
            # Parse server and model from URL
            import mlflow
            from amp.mlflow_utils import ForecasterWrapper
            from urllib.parse import urlparse
            parsed = urlparse(filename)
            server_uri = f"{parsed.scheme}://{parsed.netloc}"
            model_name = parsed.fragment.lstrip('/models/')  # remove leading '/'
            mlflow.set_tracking_uri(server_uri)
            model_uri = f"models:/{model_name}"
            print(f"Loading model from MLflow: {model_uri} (server: {server_uri})")
            return ForecasterWrapper.load_model_with_metadata(model_uri)

        if str(filename)[-4:] != '.zip':
            warnings.warn('Invalid file format for the model. Only zip supported.')
            filename = str(filename) + '.zip'

        extracted_dir = Path(filename).parent.joinpath('extracted_files')

        # Extract the contents of the zip file
        with zipfile.ZipFile(filename, 'r') as zip_file:
            zip_file.extractall(extracted_dir)

        # Load the Forecaster object.
        forecaster = joblib.load(extracted_dir.joinpath('main.pkl'))
        forecaster.model = {}

        # Remove the extracted files.
        try:
            shutil.rmtree(extracted_dir, onexc=remove_readonly)
        except TypeError:
            # if onexc argument does not exist
            shutil.rmtree(extracted_dir)

        return forecaster


class ScenarioMixin(object):

    @abstractmethod
    def predict_scenarios(self, input_df_list):
        """ Predicts response for a list of scenarios.

        The different scenarios can be for example different control strategies.
        This method assumes that the DF indexes in df_list are identical.

        Parameters
        ----------
        input_df_list : list of pd.DataFrame
                        List of model input parameters. A separate input (pd.DataFrame) for each scenario.
        Returns
        -------
        list of pd.DataFrame
            List of predictions. Each prediction uses the same format as BaseModel.predict(output_mode='single').
        """

        pass


class SetpointMixin(object):

    @abstractmethod
    def predict_setpoints(self, target, input_df, control_points):
        """ Predict set points that match a target load.

        This method is provides a prediction of set points that match the given target load.

        The purpose of this method is to enable efficient optimisation of set points inside a Resource Manager.
        It can be implemented e.g. as an inverse model predicts that predicts the set points directly from the given
        inputs or as a more efficient optimisation strategy that exploits model specific details (in constrast to the
        generic optimisation strategies available at the RM level).

        Sampling rate is the same as for the BaseModel.

        Parameters
        ----------
        target : pd.DataFrame
                 Target load profile(s) to be matched. Each energy vector (e.g. electricity and district heating) are
                 represented as separate columns.
        input_df : pd.DataFrame
                   Other parameters (in addition to the target) that are required to make the prediction.
        control_points  : list of str
                          List of control point names for which the set points are predicted.

        Returns
        -------
        pd.DataFrame
            prediction of set points for the control point(s) given as a parameter. Each control point is represented
            as a separate column in the DataFrame.

        """
        pass