import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import scipy
from amp.efp.pipe_processing import preprocessor_pipe, DenseTransformer
from amp.efp.base import Forecaster
from scikeras.wrappers import KerasRegressor
from amp.efp.decorators import parameter_search_decorator


def feature_length(feature_name, pre_pipe):
    transformer = pre_pipe.named_transformers_[feature_name]
    feature_len = 1
    if isinstance(transformer, OneHotEncoder):
        feature_len = len(transformer.categories_[0])
    elif isinstance(transformer, Pipeline):
        for operator in transformer.named_steps.values():
            if isinstance(operator, OneHotEncoder):
                feature_len = len(operator.categories_[0])
    else:
        raise ValueError(f'Invalid transformer type: {type(transformer)}')
    return feature_len


class DictTransformer(TransformerMixin):

    def __init__(self, features, pre_pipe):
        self.features = features
        self.pre_pipe = pre_pipe

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) is scipy.sparse.csr_matrix:
            X = X.todense()

        X_dict = {}
        start_index = 0
        for f_name, f_value in self.features.items():
            window_cnt = 0
            for window in f_value['windows']:
                feature_len = feature_length(f_name, self.pre_pipe)
                end_index = start_index + (abs(window[0] - window[1]) + 1) * feature_len
                X_dict[f'{f_name}_{window_cnt}_input'] = X[:, start_index:end_index]
                start_index = end_index
                window_cnt += 1

        return X_dict


class KerasModel(object):

    def __init__(self,
                 forecast_len,
                 features,
                 loss='mean_squared_error',
                 optimizer='adam',
                 activation='relu',
                 early_stopping=False,
                 validation_split=0.0,
                 epochs=10,  # todo maybe the Keras train parameters could be passed as **sk_params?
                 batch_size=32,
                 verbose=1):

        self.flen = forecast_len
        self.features = features
        self.loss = loss
        self.opt = optimizer
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.val_split = validation_split
        self.cb = []
        if early_stopping:
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=verbose,
                                                                 mode='auto', baseline=None, restore_best_weights=True)
            self.cb.append(early_stopping_cb)

    def create_model(self):
        raise NotImplementedError('This method needs to be implemented by the subclass')

    def sk_transformer(self):
        return KerasRegressor(build_fn=self.create_model,
                                                             epochs=self.epochs,
                                                             batch_size=self.batch_size,
                                                             verbose=self.verbose,
                                                             callbacks=self.cb,
                                                             validation_split=self.val_split)


class FFNN(KerasModel):

    def __init__(self,
                 forecast_len,
                 features,
                 preprocessing_pipe,
                 type='single_branch',
                 loss='mean_squared_error',
                 optimizer='adam',
                 activation='relu',
                 neurons=32,
                 hidden_layers=2,
                 early_stopping=False,
                 validation_split=0.0,
                 epochs=10,  # todo maybe the Keras train parameters could be passed as **sk_params?
                 batch_size=32,
                 verbose=1):

        super(FFNN, self).__init__(forecast_len, features, loss, optimizer,
                                   activation, early_stopping, validation_split, epochs, batch_size, verbose)
        self.preprocessing = preprocessing_pipe
        self.type = type
        self.neurons = neurons
        self.layers = hidden_layers

    def create_model(self):
        """ This method return the tf model. Meant to be used in tf.keras.wrappers.scikit_learn.KerasRegressor

        :return: model
        """
        if self.type == 'single_branch':
            model = self._single_branch()
        elif self.type == 'multi_branch':
            model = self._multi_branch()
        else:
            raise ValueError(f'Invalid FFNN type: {type}.')

        model.compile(loss=self.loss, optimizer=self.opt)
        return model

    def _single_branch(self):
        """ This method return the tf model. Meant to be used in tf.keras.wrappers.scikit_learn.KerasRegressor

        :return: model
        """
        model = tf.keras.Sequential()
        for _ in range(self.layers):
            model.add(Dense(units=self.neurons, activation=self.activation))
        model.add(Dense(units=self.flen))
        return model

    def _multi_branch(self):

        # Build hoas separate input branch for each input window
        inputs = []
        branches = []
        for f_name, f_value in self.features.items():
            i = 0
            for window in f_value['windows']:
                feature_len = feature_length(f_name, self.preprocessing)
                input_size = (abs(window[0] - window[1])+1)*feature_len
                tmp_input = Input(shape=(input_size,), name=f'{f_name}_{i}_input')
                inputs.append(tmp_input)
                branches.append(Dense(self.neurons, activation=self.activation)(tmp_input))
                i += 1

        merged = concatenate(branches)
        merged = Dense(self.neurons, activation=self.activation)(merged)
        output = Dense(self.flen)(merged)

        model = Model(inputs=inputs, outputs=output)
        return model


#
# Forecasters
#
class FFNForecaster(Forecaster):

    def __init__(self, targets, lead_time, forecast_len, data_freq, features=None,  update_freq=1,
                 upper_limits=None, lower_limits=None, **model_params):
        super().__init__(
            targets=targets,
            lead_time=lead_time,
            forecast_len=forecast_len,
            data_freq=data_freq,
            model=self._pipeline,
            features=features,
            hyperparams=model_params,
            update_freq=update_freq,
            upper_limits = upper_limits,
            lower_limits = lower_limits

        )

    @parameter_search_decorator
    def _pipeline(self, forecast_len, features, **hyperparams):
        es = hyperparams.pop('early_stopping', True)
        pre_pipe = preprocessor_pipe(features)
        full_pipe = Pipeline([
            ('preprocessing', pre_pipe),
            ('formatter', DenseTransformer()),  # validations split expects numpy array (dense)
            ('model', FFNN(forecast_len, features, pre_pipe,
                           early_stopping=es, **hyperparams).sk_transformer())
            #                       early_stopping=es, validation_split=0.2).sk_transformer())
        ])
        return full_pipe


class FFNBranchForecaster(Forecaster):

    def __init__(self, targets, lead_time, forecast_len, data_freq, features=None, update_freq=1,
                 upper_limits=None, lower_limits=None, **model_params):
        super().__init__(
            targets=targets,
            lead_time=lead_time,
            forecast_len=forecast_len,
            data_freq=data_freq,
            model=self._pipeline,
            features=features,
            hyperparams=model_params,
            update_freq=update_freq,
            upper_limits = upper_limits,
            lower_limits = lower_limits
        )

    @parameter_search_decorator
    def _pipeline(self, forecast_len, features, **hyperparams):
        pre_pipe = preprocessor_pipe(features)
        full_pipe = Pipeline([
            ('preprocessing', pre_pipe),
            ('formatter', DictTransformer(features, pre_pipe)),  # validations split expects numpy array (dense)
            ('model', FFNN(forecast_len, features, pre_pipe, type='multi_branch',
                           early_stopping=True, validation_split=0.2).sk_transformer())
        ])
        return full_pipe


