"""Additonal stats used in projects."""

import numpy as np
import pandas as pd


def woe_iv(target_series, idv_series, target_counts=None):
    """Compute Information Value (IV) from WOE (weight of evidence).

    Parameters
    ----------
        target_series: pd.Series of target variable
        idv_series: pd.Series of categorical variable

    Returns
    -------
        information value of the categorical feature

    """
    if target_counts is None:
        target_counts = target_series.value_counts().reset_index()
        target_counts.columns = ["target", "target_counts"]
    df = pd.DataFrame({"target": target_series.values, "idv": idv_series.values})

    col_target_counts = df.groupby(["idv", "target"]).size().reset_index()
    col_target_counts.columns = ["idv", "target", "col_target_counts"]
    col_target_counts = col_target_counts.merge(target_counts, on="target")
    col_target_counts["col_target_per"] = (
        col_target_counts["col_target_counts"] / col_target_counts["target_counts"]
    )
    col_target_per = col_target_counts.pivot_table(
        index="idv", columns="target", values="col_target_per"
    )
    col_target_per.columns = ["False", "True"]
    col_target_per["WoE"] = np.log(col_target_per["False"] / col_target_per["True"])
    col_target_per["IV"] = (
        col_target_per["False"] - col_target_per["True"]
    ) * col_target_per["WoE"]
    return col_target_per["IV"].sum()


def correlation_ratio(categories, measurements):
    """Compute correlation ratio (ETA).

    Parameters
    ----------
        categories: pd.Series of categorical variable
        measurements: pd.Series of continuous variable

    Returns
    -------
        correlation ratio: float
    """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta
