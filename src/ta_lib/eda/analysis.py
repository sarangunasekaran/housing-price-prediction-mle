"""Utility functions for computing various metrics during EDA."""

import logging

import hvplot
import hvplot.pandas  # noqa
import numpy as np

from .._ext_lib import _get_analyser, detigerify, calc_vif  # noqa
from ..core.api import silence_stdout

logger = logging.getLogger(__name__)


def get_variable_summary(df):
    """Get summary of the variables(features) in the input dataset.

    Gives datatype, unique count and samples by each variable.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing the features as columns.

    Returns
    -------
    pd.DataFrame
        The summary output with each row representing a feature.

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_variable_summary
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> # get summary of the variables(features) in the dataset(df)
    >>> get_variable_summary(df)
    """
    with silence_stdout():
        an = _get_analyser(df)
        summary = detigerify(an.variable_summary())
    return summary


def get_data_health_summary(df, return_plot=False):
    """Get a summary of dataframe health.

    Includes

        - Datatypes counts
        - Missing values
        - Duplicate variables (columns)
        - Duplicate observations (rows)

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset containing the features as columns.
    return_plot : bool, default is False
        If True returns a rendered plot object, by default ``False``

    Returns
    -------
    pd.DataFrame
        The summary output with each row representing a feature.
    hvPlot
        A rendered vizualization of the summary, returned if ``return_plot`` is True.

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_data_health_summary
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> # get summary of the health of the input dataset as DataFrame.
    >>> get_data_health_summary(df)
    >>> # get both summary and hvPlot of the health of the input dataset.
    >>> get_data_health_summary(df, return_plot=True)
    """
    with silence_stdout():
        an = _get_analyser(df)
        summary = detigerify(an._compute_data_health())
        if return_plot:
            plot = an._plot_data_health(summary)

    if return_plot:
        return summary, plot
    else:
        return summary


def get_missing_values_summary(df, return_plot=False):
    """Get missing values by column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing all the relevant features as columns.
    return_plot : bool, default is False
        If True returns a rendered plot object, by default ``False``

    Returns
    -------
    pd.DataFrame
        The summary output with each row representing a feature.
    hvPlot
        A rendered vizualization of the summary, returned if ``return_plot`` is True.

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_missing_values_summary
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> # get summary of the missing values of the input dataset.
    >>> get_missing_values_summary(df)
    >>> # get both summary and hvPlot of the missing values of the input dataset.
    >>> get_missing_values_summary(df, return_plot=True)
    """
    with silence_stdout():
        an = _get_analyser(df)
        summary = detigerify(
            an._missing_values().sort_values("No of Missing", ascending=False)
        )
        if return_plot:
            plot = an.missing_plot()

    if return_plot:
        return summary, plot
    else:
        return summary


def get_duplicate_columns(df):
    """Get the duplicate columns in a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing all the relevant features as columns.

    Returns
    -------
    pd.DataFrame
        Summary output with each row representing a column

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_duplicate_columns
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> # get the duplicate columns that are present for each feature in the input dataset.
    >>> get_duplicate_columns(df)
    """
    with silence_stdout():
        an = _get_analyser(df)
        out = an.duplicate_columns()
        if not isinstance(out, str):
            out = detigerify(out)
    return out


def get_outliers(df):
    """Return outlier information for given dataset.

    Gives the outliers for each numeric column. The outliers are identified by three methods:

        - Mean +/- 3 standard deviations
        - Using `1.5 * IQR`
        - All infinity values

    Parameters
    ----------
    df: pd.Dataframe
        The dataset to be analyzed for outliers. Each column is
        considered to be a feature.

    Returns
    -------
    pd.DataFrame
        outlier analysis table with each row representing a feature
        and columns specify the various outlier regions

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_outliers
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> # get outlier information for given outlier criterion.
    >>> get_outliers(df)
    """
    with silence_stdout():
        an = _get_analyser(df)
        out_df = detigerify(an.get_outliers_df())
    return out_df


def get_density_plots(df, cols=None):
    """Get the density plots for each of the numerical columns from the input data.

    Following plots are generated as part of the output

         - `Density plot` - A plot describing the density distribution
         - `Table` - Variable name, Mean, Median, Standard Deviation

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the columns present in cols argument
    cols : list, default is None
        Columns for which the density plots have to be generated, by default ``None``

    Returns
    -------
    hvPlot
        A rendered vizualization of the density plot summary

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_density_plots
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> # get the density plots for each of the numerical columns from the input data.
    >>> get_density_plots(df)
    >>> # get the density plots for each of the columns passed as Parameter to the function.
    >>> get_density_plots(df, cols=['Age', 'Fare'])
    """
    cols = cols or df.select_dtypes("number").columns.to_list()
    with silence_stdout():
        an = _get_analyser(df)
        out = an.density_plots(cols=cols)
    return out


def get_percentile_plots(df, cols=None):
    """Get percentile plots for each of the numeric columns.

    Following views are shown as part of the output

    - `Percentile plot` - A plot describing the percentile distribution
    - `Table 1` - Variable name, Mean, Median, Standard Deviation, Minimum and Maximum
    - `Table 2` - Bottom 5 values(0.2% to 1% range)
    - `Table 3` - Top 5 values(99.0% to 99.8% range)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the columns present in cols argument.
    cols : list, ,default is None
        List of Columns for which the percentile plots have to be generated, by default None

    Returns
    -------
    hvPlot
        A rendered plot of the percentile-plot summary

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_percentile_plots
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> # get percentile plots and summary tables for each of the numeric columns.
    >>> get_percentile_plots(df)
    >>> # get percentile plots and summary tables for each of the columns passed as Parameter to the function.
    >>> get_percentile_plots(df, cols=['Age', 'Fare'])
    """
    cols = cols or df.select_dtypes("number").columns.to_list()
    with silence_stdout():
        an = _get_analyser(df)
        out = an.percentile_plots(cols=cols)
    return out


def get_frequency_plots(df, cols=None):
    """Get bar-plots of frequency distribution for non-numeric variables in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the columns present in cols argument
    cols : list, optional
        List of columns to generate the frequency plots, by default None (generates for all non-numerical columns)

    Returns
    -------
    hvPlot
        A rendered summary of frequency plots with `X.axis` as Frequency of occurrence and `Y.axis` containing top 10 most frequent levels of the variable considered

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_frequency_plots
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> # get bar-plots of frequency distribution for non-numeric variables in the dataset.
    >>> get_frequency_plots(df)
    >>> # get bar-plots of frequency distribution for non-numeric variables that are passed to function as Parameters.
    >>> get_frequency_plots(df, cols=['Survived', 'Sex'])
    """
    cols = cols or df.select_dtypes("object").columns.to_list()
    with silence_stdout():
        an = _get_analyser(df)
        out = an.non_numeric_frequency_plot(cols=cols)
    return out


def get_correlation_table(df, x_cols=None, y_cols=None):
    """Get a correlation table for the input data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the relevant columns
    x_cols : list, optional
        List of columns to be included in X axis, by default None
    y_cols : list, optional
        List of columns to be included in y axis, by default None

    Returns
    -------
    pd.DataFrame
        plot showing the correlation between the given column combinations in the input data

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_correlation_table
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> # get a correlation table for all the columns in the input data.
    >>> get_correlation_table(df)
    >>> # get a correlation table for the only specific columns in the input data.
    >>> get_correlation_table(df, x_cols=['Sex', 'Pclass'], y_cols=['Survived'])
    """
    with silence_stdout():
        an = _get_analyser(df)
        out = detigerify(an.correlation_table(x_vars=x_cols, y_vars=y_cols))
    return out


def get_correlation_heatmap(df):
    raise NotImplementedError()


def get_bivariate_plots(df, x_cols=None, y_cols=None):
    """Get the bivariate plots for input data.

    Following plots are generated, for all the variable combinations, based on data type of features in comparison

     - `Scatter Plot`     - When both X & Y are continuous variables
     - `Grouped Bar Plot` - When both X & Y are categorical/binary variables
     - `Box Plot`         - When one of X & Y is continuous variable and the other is categorical variable
     - `Butterfly Plot`   - When one of X & Y is binary variable and the other is categorical variable

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the relevant columns
    x_cols : list, optional
        List of columns to be implemented in x axis, by default None
    y_cols : list, optional
        List of columns to be implemented in y axis, by default None

    Returns
    -------
    hvPlot
        A rendered plot showing the relevant bivariate plots

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_bivariate_plots
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> # get the bivariate plots for all the columns in the input data.
    >>> get_bivariate_plots(df)
    >>> # get the relevant bivariate plots for the specified columns in the input data.
    >>> get_bivariate_plots(df, x_cols=['Sex', 'Age', 'Fare'], y_cols=['Cabin', 'Survived'])
    """
    with silence_stdout():
        an = _get_analyser(df)
        out = detigerify(an.bivariate_plots(x_vars=x_cols, y_vars=y_cols))
    return out


def get_target_correlation(df, y, x_cols=None):
    """Get the dictionary which contains correlation and information gain from the dependent variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the relevant columns
    y : pd.series
        Target column values in the input data
    x_cols : list, optional
        List of columns to be considered as x variable, by default None

    Returns
    -------
    dict(hvPlot)
        dictionary of relevant plots for the feature correlation and information gain.

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_target_correlation
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> target = df[['Survived']]
    >>> # get the dictionary which contains correlation and information gain for all x variable from the dependent variable.
    >>> get_target_correlation(df, y=target)
    >>> # get the dictionary which contains correlation and information gain of specified x variable from the dependent variable.
    >>> get_target_correlation(df, y=target, x_cols=['Pclass', 'Sex', 'Age', 'Fare'])
    """
    with silence_stdout():
        merged_df = df.copy()
        merged_df["y"] = y.values
        an = _get_analyser(merged_df, y="y")
        out = an.get_feature_scores(features=x_cols)
    return out


def get_feature_importances(df, y, x_cols=None):
    """Get model feature importances for a DV.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the relevant columns
    y : pd.series
        Target column values in the input data
    x_cols : list, optional
        List of columns to be considered as x variable, by default None

    Returns
    -------
    hvPlot
        feature importance plot from both model and shap

    Examples
    --------
    >>> from ta_lib.eda.analysis import get_feature_importances
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> target = df[['Survived']]
    >>> # get model feature importances for all x variable in the input dataset.
    >>> get_feature_importances(df, y=target)
    >>> # get model feature importances for specified x_cols in the input dataset.
    >>> get_feature_importances(df, y=target, x_cols=['Pclass', 'Sex', 'Age', 'Fare'])
    """
    with silence_stdout():
        merged_df = df.copy()
        merged_df["y"] = y.values
        an = _get_analyser(merged_df, y="y")
        out = an.feature_importances(features=x_cols, quick=False)
    return out


def residual_plots(dataset, y_true, y_pred, kde=True, threshold=0.3, error="pe"):
    """Generate density plots based on the residuals(Over prediction/Under prediction).

    Parameters
    ----------
    dataset : pd.DataFrame
        Prediction dataframe that contains y_pred and y_true columns
    y_true  : str
        Actuals column name in dataframe.
    y_pred  : str
        Predictions column name in dataframe.
    kde : bool, default is True
         To plot density plot of errors or barplot. (True - density, False - barplot)
    threshold: numeric
        limit value above or below which it is regarded as Outlier predictions
    error: str
        whether to take percentage errors or absolute error to show in density plots

    Returns
    -------
    hvPlot:
        Rendered plots(either a kde or bar) of the residuals by features
    """
    if error == "pe":
        dataset["Residuals"] = (
            np.abs(dataset["prediction"] - dataset["actuals"]) / dataset["actuals"]
        )
    else:
        dataset["Residuals"] = np.abs(dataset["prediction"] - dataset["actuals"])
    dataset["Variable"] = np.where(
        dataset["prediction"] > (dataset["actuals"] * (1 + threshold)),
        "OverPrediction",
        np.where(
            (dataset["prediction"]) < (dataset["actuals"] * (1 - threshold)),
            "UnderPrediction",
            "Others",
        ),
    )
    if kde:
        kde_plot = dataset.hvPlot.kde(
            y="Residuals",
            by="Variable",  # Grouping by Predictions
            width=800,
            height=400,
            alpha=0.7,
            ylabel="Density",
            xlabel="Residuals(APE)",
            title=f"Residual Analysis(Error threshold = {threshold*100}%)",
            legend="top_right",
        )
        return kde_plot
    else:
        bar_df = (
            dataset["Variable"].value_counts().reset_index()
        )  # .plot.bar(x='index',y='Variable')
        bar_df.columns = ["Over/Underpredictions", "Value Distribution"]
        barplot = bar_df.hvPlot.bar(
            x="Over/Underpredictions",
            y="Value Distribution",
            title=f"Distribution of Predictions (Error threshold= {threshold*100}%)",
        )
        return barplot
