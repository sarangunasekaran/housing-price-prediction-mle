"""Module to provide a common place for all useful Estimators regardless of library.

This module simply lists a bunch of curated ``Estimator`` classes from various libraries.
This is useful in loading a class using the name of the regressor regarless of the library
it comes from.

.. code:: python

    from ta_lib.regression import estimators

    cls = getattr(estimators, 'LinearRegression')


If needed, we can remove this module and instead use fully qualified name to load the classes.
e.g:

.. code:: python

    regressors = {
        'linear_regressor': 'sklearn.linear_model.LinearRegression',
    }

    def load_class(qual_cls_name):
        module_name, cls_name = qual_cls_name.rsplit('.', 1)
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            logger.exception(f'Failed to import module : {module_name}')
        else:
            return getattr(mod, cls_name)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import XGBRegressor  # noqa

import ta_lib.eda.analysis as ta_analysis

# List of estimators exposed by the module
__all__ = ["SKLStatsmodelOLS"]


def _check_X(X, columns=None):
    """Change array to DataFrame."""
    if isinstance(X, (pd.DataFrame)):
        return X
    elif isinstance(X, (np.ndarray)):
        if columns is None:
            return pd.DataFrame(X)
        else:
            return pd.DataFrame(X, columns=columns)
    elif isinstance(X, pd.Series):
        return pd.DataFrame(X, columns=X.name)
    elif isinstance(X, (list, tuple)):
        X = np.array(X)
        return pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    elif hasattr(X, ("__array__")):
        data = X.__array__()
        return pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])
    return X


class SKLStatsmodelOLS(BaseEstimator, RegressorMixin):
    """sklearn wrapper estimator for the statsmodels.api.OLS Estimator.

    Parameters
    ----------
    fit_intercept: bool
        argument to specify whether to fit the intercept in the model

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.
    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    Notes
    -----
    From the implementation point of view, this is just a wrapper class that uses the standards of SKLearn and model from statsmodels api.

    Examples
    --------
    >>> import numpy as np
    >>> from ta_lib.regression.estimators import SKLStatsmodelOLS
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = SKLStatsmodelOLS().fit(X, y)
    >>> reg.coef_
    array([1., 2.])
    >>> reg.intercept_
    3.0000...
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])

    """

    def __init__(self, fit_intercept=True):
        """Initialize Estimator."""
        self.fit_intercept = fit_intercept

    def fit(self, X, y, column_names=None):
        """Build a linear regression model from the training set (X, y).

        Parameters
        ----------
        X : pd.DataFrame or np.Array
            Dataframe/2D Array consisting of independant features
        y : pd.DataFrame or pd.Series or np.Array
            Dataframe/Series/Numpy Array containing the target column
        column_names : list, optional
            List of column names for features relevant when X, Y are Arrays, by default None

        """

        temp_X, tempy = check_X_y(X, y)

        X = _check_X(X, columns=column_names)
        if column_names is None:
            if hasattr(X, "columns"):
                column_names = X.columns.to_list()
            else:
                column_names = [str(i) for i in range(X.shape[1])]

        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")
            column_names = ["intercept"] + column_names

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        X = pd.DataFrame(X)
        X.columns = column_names

        self.model_ = sm.OLS(y.astype(float), X.astype(float)).fit()
        self.feature_names_ = column_names
        self._X = X

        if self.fit_intercept:
            self.intercept_ = self.model_.params[
                (self.model_.params.index.isin(["intercept", "const"]))
            ].values[0]
        else:
            self.intercept_ = 0
        self.coef_ = self.model_.params[
            ~self.model_.params.index.isin(["intercept", "const"])
        ].values
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : pd.DataFrame or np.Array
            Dataframe/2D Array consisting of independant features

        Returns
        -------
        np.Array
            Predictions for the input data
        """
        # Check is fit had been called
        check_is_fitted(self, "is_fitted_")

        # Input validation
        X = check_array(X)
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")

        return self.model_.predict(X.astype(float))

    def get_params(self, deep=False):
        """Get the parameters of this estimator.

        Parameters
        ----------
        deep : bool, optional
            WIP, by default False

        Returns
        -------
        dict
            parameters dictionery
        """
        return {"fit_intercept": self.fit_intercept}

    def summary(self):
        """Return the model summary similar to the statsmodels OLS Summary."""
        return self.model_.summary()

    def coeff_table(self, add_vif=True):
        """Coefficient table of this model.

        Parameters
        ----------
            add_vif: bool, default is True.
                if True, the VIF values for variables are also returned.

        Returns
        -------
        coeff_table: pd.DataFrame
        """
        results_summary = self.model_.summary()
        results_as_html = results_summary.tables[1].as_html()
        coeffs_table = pd.read_html(results_as_html, header=0, index_col=0)[0]
        coeffs_table.reset_index(inplace=True)
        coeffs_table.rename(columns={"index": "variables"}, inplace=True)
        if add_vif:
            vif = ta_analysis.calc_vif(self._X)
            coeffs_table = coeffs_table.merge(vif, how="left", on=["variables"])
        coeffs_table.set_index("variables", inplace=True)
        coeffs_table.index.name = None
        coeffs_table.columns.name = None
        return coeffs_table

    def get_model_type(self):
        """Return the model type of the estimator."""
        # FIXME: These definitions ought to be defined somewhere
        # outside of tigerml
        return "linear"
