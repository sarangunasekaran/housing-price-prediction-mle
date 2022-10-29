"""Collection of model evaluators for classification."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from tigerml.model_eval import (  # noqa
    ClassificationComparison,
    ClassificationReport,
)

# List of all evaluators exported by this module.
__all__ = [
    "ClassificationComparison",
    "ClassificationReport",
    "overlapping_histograms",
    "confusion_matrix_by_feature",
]


def overlapping_histograms(dataset, col_idv, col_dv):
    """Generate overlapping histogram.

    For bivariate analysis between binary dependent variable
    and continuous independent variable.

    Input: dataset : Input dataframe.
           col_idv : Continuous independent variable.
           col_dv  : Binary target variable.

    Output: The function generates an overlapping histogram
            over the dependent variable.

    """

    plt.clf()

    majority = dataset[col_dv].value_counts(ascending=False).index[0]
    minority = dataset[col_dv].value_counts(ascending=True).index[0]

    sns.distplot(
        dataset[col_idv][dataset[col_dv] == majority].values,
        hist=True,
        kde=False,
        color="red",
        label=majority,
    )
    sns.distplot(
        dataset[col_idv][dataset[col_dv] == minority].values,
        hist=True,
        kde=False,
        color="blue",
        label=minority,
    )
    plt.grid(False)
    plt.legend(loc="best")
    plt.title("Overlapping histogram of " + col_idv + " over " + col_dv, fontsize=15)
    plt.xlabel(col_idv, fontsize=13)
    plt.ylabel("Distribution", fontsize=13)
    plt.show()


def confusion_matrix_by_feature(dataset, y_true, y_pred, col_idv, categorical=True):
    """Generate distribution plots by confusion matrix segment.

    In the confusion matrix for each segment, genereate joint plot
    categorical variable and distributions for continuous variables.

    Input: dataset : Input dataframe.
           y_true  : Actuals column name in dataframe.
           y_pred  : Predictions column name in dataframe.
           col_idv : Independent variable for which plot is to be generated.
           categorical : Independent variable type. (True - categorical, False - Continuous)

    Output: The function generates confusion matrix components as
            joint plot, distributions for categorical and continuous
            respectively.

    """

    dataset["concat"] = dataset[y_true].map(str) + "_" + dataset[y_pred].map(str)

    dataset["TN"] = np.where(dataset["concat"] == "0_0", 1, 0)
    dataset["TP"] = np.where(dataset["concat"] == "1_1", 1, 0)
    dataset["FP"] = np.where(dataset["concat"] == "0_1", 1, 0)
    dataset["FN"] = np.where(dataset["concat"] == "1_0", 1, 0)

    if categorical:
        value = [str(s) for s in list(np.unique(dataset[col_idv]))]
        levels = [col_idv + "_" + str(s) for s in list(np.unique(dataset[col_idv]))]
        elements = ["TN", "TP", "FN", "FP"]

        df_confusion = dataset.groupby(col_idv)[elements].sum().T.reset_index()
        df_confusion.columns = [col_idv] + levels
        df_confusion[levels] = df_confusion[levels].apply(lambda x: x / x.sum(), axis=1)
        df_confusion = (
            df_confusion.set_index(col_idv).T.rename_axis(col_idv).reset_index()
        )

        df_category_distribution = (
            dataset[col_idv].value_counts(normalize=True).reset_index()
        )
        df_category_distribution.columns = [col_idv, "ratio"]
        df_category_distribution[col_idv] = (
            col_idv + "_" + df_category_distribution[col_idv].map(str)
        )

        df_heatmap = df_confusion.merge(df_category_distribution, on=col_idv)
        df_heatmap[elements] = (
            df_heatmap[elements].div(df_heatmap["ratio"], axis=0) * 100
        ).round()
        df_heatmap.drop("ratio", axis=1, inplace=True)
        plt.figure()
        # cmap=sns.diverging_palette(10, 220, sep=80, n=7) #'coolwarm_r'
        cmap = LinearSegmentedColormap.from_list(
            name="test", colors=["red", "white", "green"]
        )
        sns.heatmap(
            df_heatmap[elements],
            robust=True,
            annot=True,
            cmap=cmap,
            linecolor="black",
            linewidths=0.5,
            fmt="g",
            center=100,
            yticklabels=value,
        )
        plt.title(
            "Index heatmap of " + col_idv + " over Consusion Matrix components",
            fontsize=15,
        )
        plt.xlabel("Confusion Matrix components", fontsize=13)
        plt.ylabel(col_idv, fontsize=13)
        plt.show()
    else:

        dataset = pd.melt(dataset, id_vars=col_idv, value_vars=["TN", "TP", "FN", "FP"])
        dataset = dataset[dataset["value"] == 1]

        sns.distplot(
            dataset[col_idv][dataset["variable"] == "TN"],
            hist=False,
            kde=True,
            color="red",
            label="TN",
        )
        sns.distplot(
            dataset[col_idv][dataset["variable"] == "TP"],
            hist=False,
            kde=True,
            color="blue",
            label="TP",
        )
        sns.distplot(
            dataset[col_idv][dataset["variable"] == "FN"],
            hist=False,
            kde=True,
            color="green",
            label="FN",
        )
        sns.distplot(
            dataset[col_idv][dataset["variable"] == "FP"],
            hist=False,
            kde=True,
            color="black",
            label="FP",
        )

        plt.grid(False)
        plt.legend(loc="best")
        plt.title("Confusion Matrix elements by " + col_idv, fontsize=15)
        plt.xlabel(col_idv, fontsize=13)
        plt.ylabel("Distribution", fontsize=13)
        plt.show()
