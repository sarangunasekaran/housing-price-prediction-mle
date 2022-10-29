"""Module to generate the data health summary reports."""

import logging

from .._ext_lib import _get_analyser
from ..core.api import silence_stdout

logger = logging.getLogger(__name__)


def summary_report(df, save_path):
    """Generate health analysis report that consists of summary and plot on missing values,outliers and duplicate values.

    Health analysis report in the form of HTML is generated and saves in the mentioned path.

    Parameters
    ----------
    df : pd.Dataframe
        Input dataset for which the health report is generated.
    save_path : str
        Location where the HTML file gets saved

    Raises
    ------
    RuntimeError
        When failed to generate the report

    Examples
    --------
    >>> import ta_lib.reports.api as rep
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> rep.summary_report(df,save_path = '.')

    """
    # We do not support returning an object that can be rendered on the
    # notebook. Simply reading the html and embedding as html causes
    # rendering issues.
    with silence_stdout():
        an = _get_analyser(df)
        out = an.health_analysis(save_as=".html", save_path=save_path)
        if isinstance(out, str):
            raise RuntimeError(f"Failed to create report : {out}")
    logger.info(f"Saved Health report to : {save_path}")


def feature_analysis(df, save_path):
    """Generate feature analysis report that has the summary of distribution of variables,normality rankings.

    Feature analysis report in the form of HTML is generated and saves in the mentioned path.
    It consists of:
    1)Summary Stats
    2)Distributions
    3)Feature Normality

    Parameters
    ----------
    df : pd.Dataframe
        Input dataset for which the health report is generated.
    save_path : str
        Location where the HTML file gets saved

    Raises
    ------
    RuntimeError
        When failed to generate the report

    Examples
    --------
    >>> import ta_lib.reports.api as rep
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> rep.feature_analysis(df,save_path = '.')

    """
    # We do not support returning an object that can be rendered on the
    # notebook. Simply reading the html and embedding as html causes
    # rendering issues.
    with silence_stdout():
        df_copy = df.copy()
        an = _get_analyser(df_copy)
        out = an.feature_analysis(save_as=".html", save_path=save_path)
        if isinstance(out, str):
            raise RuntimeError(f"Failed to create report : {out}")
    logger.info(f"Saved Health report to : {save_path}")


def feature_interactions(df, save_path):
    """Generate feature interaction report that has plots related to correlation and covariance.

    Feature interaction report in HTML format is generated and saved in the mentioned path.
    It consist of:
    1)Correlation Table
    2)Correlation Heatmap
    3)Covariance Heatmap
    4)Bivariate Plots (top 50 Correlations)

    Parameters
    ----------
    df : pd.Dataframe
        Input dataset for which the health report is generated.
    save_path : str
        Location where the HTML file gets saved

    Raises
    ------
    RuntimeError
        When failed to generate the report

    Examples
    --------
    >>> import ta_lib.reports.api as rep
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> rep.feature_interactions(df,save_path = '.')
    """
    # We do not support returning an object that can be rendered on the
    # notebook. Simply reading the html and embedding as html causes
    # rendering issues.
    with silence_stdout():
        df_copy = df.copy()
        an = _get_analyser(df_copy)
        out = an.feature_interactions(save_as=".html", save_path=save_path)
        if isinstance(out, str):
            raise RuntimeError(f"Failed to create report : {out}")
    logger.info(f"Saved Health report to : {save_path}")


def key_drivers(df, target, save_path, quick=True):
    """Generate key drivers report that has feature scores,feature importance from model,pca analysis and bivariate plots on correlation of target variable with independent variables.

    Key drivers report in the form of HTML is generated and saves in the mentioned path.

    Parameters
    ----------
    df : pd.Dataframe
        Input dataset for which the key drivers is generated.
    target: pd.Dataframe
        target dataset
    save_path : str
        Location where the HTML file gets saved
    quick: bool,default is True
        If True,calculates SHAP and generates bivariate plots

    Raises
    ------
    RuntimeError
        When failed to generate the report

    Examples
    --------
    >>> import ta_lib.reports.api as rep
    >>> import pandas as pd
    >>> df = pd.read_csv("titatic.csv")
    >>> target = df[['Survived']]
    >>> df = df.drop(['Survived'],axis=1)
    >>> rep.key_drivers(df, target,save_path = '.')
    """
    # We do not support returning an object that can be rendered on the
    # notebook. Simply reading the html and embedding as html causes
    # rendering issues.
    with silence_stdout():
        merged_df = df.copy()
        merged_df[target.columns[0]] = target.values
        an = _get_analyser(merged_df, y=target.columns[0])
        out = an.key_drivers(save_as=".html", save_path=save_path, quick=quick)
        if isinstance(out, str):
            raise RuntimeError(f"Failed to create report : {out}")
    logger.info(f"Saved Health report to : {save_path}")


def data_exploration(df, target, save_path, quick=True):
    """Generate data exploration report that has consolidated report on data preview,health analysis,feature ana:lysis,feature interactions and key drivers.

    Data exploration report in the form of HTML is generated and saves in the mentioned path.
    Data Exploration report consists of :
    1)Health Analysis: Summary and plot on missing values,outliers and duplicate values.
    2)Feature analysis: Summary of distribution of variables, normality rankings.
    3)Feature Interactions: Plots related to correlation and covariance.
    4)Key Drivers: Plots related to feature scores,feature importance from model,pca analysis and bivariate plots on correlation of target variable with independent variables.

    Parameters
    ----------
    df : pd.Dataframe
        Input dataset for which the complete data exploration report is generated.
    target: pd.Dataframe
        target dataset
    save_path : str
        Location where the HTML file gets saved
    quick: bool ,default is True
        If True,calculates SHAP and generates bivariate plots

    Raises
    ------
    RuntimeError
        When failed to generate the report

    Examples
    --------
    >>> import ta_lib.reports.api as rep
    >>> target = df[['Survived']]
    >>> df = df.drop(['Survived'],axis=1)
    >>> rep.data_exploration(df, target,save_path = '.')
    """
    # We do not support returning an object that can be rendered on the
    # notebook. Simply reading the html and embedding as html causes
    # rendering issues.
    with silence_stdout():
        merged_df = df.copy()
        merged_df[target.columns[0]] = target.values
        an = _get_analyser(merged_df, y=target.columns[0])
        out = an.get_report(format=".html", save_path=save_path, quick=quick)
        if isinstance(out, str):
            raise RuntimeError(f"Failed to create report : {out}")
    logger.info(f"Saved Health report to : {save_path}")
