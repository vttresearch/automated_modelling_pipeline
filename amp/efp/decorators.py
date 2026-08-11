from functools import wraps

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor


def _add_prefix_to_hyperparams(model, hyperparams):
    """
    Add a prefix to each hyperparameter key in a dictionary if needed.

    Args:
        pipeline
        hyperparams (dict): Dictionary containing hyperparameters.


    Returns:
        dict: Dictionary with the specified prefix added to each key if needed.
    """
    if next(iter(hyperparams)).startswith('model'):
        # Return hyperparams as is if user has defined the structure
        return hyperparams
    else:
        if isinstance(model, MultiOutputRegressor):
            prefix = 'model1__estimator__'
        else:
            prefix = 'model1__'
        return {f"{prefix}{key}": value for key, value in hyperparams.items()}




def parameter_search_decorator(func):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        hyperparam_search = kwargs.get('hyperparam_search', None)
        if hyperparam_search is not None:

            hyperparam_search_method = hyperparam_search.get('hyperparam_search_method', None)

            # rename the pipeline step "model" to "model1"
            original_pipeline = func(*args, **kwargs)
            idx = next((idx for idx, name in enumerate(original_pipeline.named_steps.keys()) if name == 'model'), None)
            if idx is not None:
                original_pipeline.steps[idx] = ('model1', original_pipeline.steps[idx][1])
                hyperparam_space = _add_prefix_to_hyperparams(original_pipeline['model1'], hyperparam_search.get('hyperparam_space', None))

            # find is this a grid_seach or random_seach case and return the new pipeline
            if hyperparam_search_method == 'grid_search':
                full_pipe = Pipeline([
                    (
                    'model', GridSearchCV(original_pipeline, hyperparam_space, refit=True, cv=10, verbose=1, n_jobs=-1))
                ])
                return full_pipe
            if hyperparam_search_method == 'random_search':
                full_pipe = Pipeline([
                    ('model', RandomizedSearchCV(original_pipeline, hyperparam_space, n_iter=1))
                ])
                return full_pipe
        # otherwise return the old pipeline without parameter search
        return func(*args, **kwargs)

    return wrapper_func
