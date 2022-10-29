"""Module to provide a common place for all useful Estimators regardless of library.

This module simply lists a bunch of curated ``Estimator`` classes from various libraries.
This is useful in loading a class using the name of the regressor regarless of the library
it comes from.
.. code:: python

    from ta_lib.classification import estimators
    cls = getattr(estimators, 'LogisticRegression')

If needed, we can remove this module and instead use fully qualified name to load the classes.
e.g:

.. code:: python

    regressors = {
        'logistic_regressor': 'sklearn.linear_model.LogisticRegression',
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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import XGBClassifier  # noqa

import ta_lib.eda.analysis as ta_analysis

# List of estimators exposed by the module
__all__ = ["SKLStatsmodelLogit"]


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


class SKLStatsmodelLogit(BaseEstimator, ClassifierMixin):
    """sklearn style wrapper estimator for the statsmodels.api.OLS Estimator."""

    def __init__(self, fit_intercept=True, method="hessian"):
        """Initialize Estimator.

        Parameters
        ----------
        fit_intercept: bool
            argument to specify whether to fit the intercept in the model
        """
        self.fit_intercept = fit_intercept
        self.method = method

    def _more_tags(self):
        return {
            "binary_only": True,
        }

    def fit(self, X, y, column_names=None):
        """Build a logistic regression model from the training set (X, y).

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
        check_classification_targets(y)
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
        self.classes_ = sorted(np.unique(y))
        y_map = {v: i for i, v in enumerate(self.classes_)}
        y = np.array(list(map(lambda x: y_map[x], y)))

        X = pd.DataFrame(X)
        X.columns = column_names

        self.model_ = sm.Logit(y, X, method=self.method).fit()
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
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : pd.DataFrame or np.Array
            Dataframe/2D Array consisting of independant features
        Returns
        -------
        2D np.Array
            Prediction probabilities of classes [0,1]
        """
        # Check is fit had been called
        check_is_fitted(self, "model_")

        # Input validation
        X = check_array(X)
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")

        #         print(X.shape)
        decision_2d = np.empty(shape=(X.shape[0], 2))

        sklearn_models = self.model_.predict(X)
        decision_2d[:, 1] = sklearn_models
        decision_2d[:, 0] = 1 - sklearn_models

        return decision_2d

    def predict(self, X, threshold=0.5):
        """Predict class for X."""

        # Check is fit had been called
        check_is_fitted(self, "model_")

        # Input validation
        X = check_array(X)
        if self.fit_intercept:
            X = sm.add_constant(X, has_constant="add")

        #         print(X.shape)
        predict_probs = self.model_.predict(exog=X)

        predictions = np.where(predict_probs > threshold, 1, 0)
        reverse_map = {i: v for i, v in enumerate(self.classes_)}
        predictions = np.array(list(map(lambda x: reverse_map[x], predictions)))
        return predictions

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : pd.DataFrame or np.Array
            Dataframe/2D Array consisting of independant features
        y : pd.DataFrame or pd.Series or np.Array
            Dataframe/Series/Numpy Array containing the target column
        sample_weight: array-like of shape (n_samples,), optional
            Array of class weights for imbalanced classification tasks, by default=None


        Returns
        -------
        score (float)
            Fraction of correctly classified samples, with best performance being 1
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def get_params(self, deep=False):
        """Get the parameters of this estimator.

        This is to enable refit & scoring while genearting evaluation report

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
        pd.DataFrame
            coeff_table
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
        """Return the model type of the estimator.

        For generating SHAP values mode_type should be defined as tree or linear.
        """
        return "logistic"
