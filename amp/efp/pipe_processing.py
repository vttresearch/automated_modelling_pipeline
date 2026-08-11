from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import scipy
import numpy as np


class SincosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name=None, max_value=None, debug=False):
        self.feature_name = feature_name
        self.max_value = max_value
        self.debug = debug

    def fit(self, X, y=None):
        if self.max_value is None:
            lname = (self.feature_name or "").lower()
            if "hour" in lname:
                self.max_value = 24
            elif "month" in lname:
                self.max_value = 12
            elif "weekday" in lname or "dayofweek" in lname:
                self.max_value = 7
            else:
                self.max_value = int(np.max(X)) + 1
        return self

    def transform(self, df):
        sine = np.sin(2 * np.pi * df.iloc[:, 0].values / self.max_value)
        cosine = np.cos(2 * np.pi * df.iloc[:, 0].values / self.max_value)
        sincos = np.stack([sine, cosine], axis=-1)

        return sincos

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names."""
        if input_features is None:
            input_features = [self.feature_name or "feature"]

        names = []
        for f in input_features:
            names.append(f"{f}_sin")
            names.append(f"{f}_cos")
        return np.array(names)

class DummyEstimator(BaseEstimator):
    def fit(self): pass
    def score(self): pass


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if type(X) is scipy.sparse.csr_matrix:
            return X.todense()
        else:
            return X


def flatten_input(X):
    return X.values.reshape(X.shape[0], -1)


def preprocessor_pipe(features):
    numeric_pipe = Pipeline(steps=[
        ('imbuter', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    transformers = []
    for fname, value in features.items():
        if value['type'] == 'numeric':
            pipe = numeric_pipe
        elif value['type'] == 'onehot':
            pipe = OneHotEncoder(handle_unknown='ignore')
        elif value['type'] == 'scaler_only':
            pipe = StandardScaler()
        elif value['type'] == 'cyclical':
            # TODO not sure if this is working yet
            pipe = SincosTransformer(feature_name=fname)
        else:
            raise ValueError(f'Preprocessing pipeline not specified for feature: {fname} with type: {value["type"]}')
        transformers.append((fname, pipe, make_column_selector(pattern=f'^{fname}')))

    preprocessor = ColumnTransformer(transformers=transformers, verbose=True)
    return preprocessor


def search_features(cv_split,
                    model=MLPRegressor(verbose=True, shuffle=False, max_iter=100),
                    scoring='neg_mean_squared_error'):

    pipe = Pipeline([
        ('preprocessing', DummyEstimator()),
        ('model', model)
    ])

    search_space = [{'model': [SVR(verbose=True)]},
                    {'model': [MLPRegressor(verbose=True, shuffle=False, max_iter=100)]}]

    gs = GridSearchCV(pipe, search_space, scoring=scoring, cv=cv_split, verbose=2)
    return gs


def search_model(cv_split, preprocessing_pipe, scoring='neg_mean_squared_error'):

    pipe = Pipeline([
        ('preprocessing', preprocessing_pipe),
        ('model', DummyEstimator())
    ])

    search_space = [{'model': [SVR(verbose=True)]},
                    {'model': [MLPRegressor(verbose=True, shuffle=False, max_iter=200)]}]

    gs = GridSearchCV(pipe, search_space, scoring=scoring, cv=cv_split, verbose=2)
    return gs



