"""Collection of model evaluators for regression."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .._ext_lib import RegressionComparison, RegressionReport, mape, wmape, root_mean_squared_error  # noqa

# List of all supported evaluators for regression.
__all__ = [
    "RegressionComparison",
    "RegressionReport",
]


def residual_analysis_by_feature(dataset, col_idv, flag_variable, categorical=True):
    """Generate distributions plots for elements by flag variable.

    Input: dataset : Input dataframe.
           col_idv : Independent variable for which plot is to be generated.
           flag_variable: Variable, which tells about over prediction, under prediction based on a threshold value
           categorical : Independent variable type. (True - categorical, False - Continuous)

    Output: The function generates confusion matrix components as
            joint plot, distributions for categorical and continuous
            respectively.
    """

    if categorical:
        value = [str(s) for s in list(np.unique(dataset[col_idv]))]
        levels = [col_idv + "_" + str(s) for s in list(np.unique(dataset[col_idv]))]
        elements = ["good", "over predict", "under predict"]
        data_set_dummies = pd.get_dummies(
            dataset[flag_variable], columns=[flag_variable]
        )
        dataset = dataset.merge(data_set_dummies, left_index=True, right_index=True)

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
            "Index heatmap of " + col_idv + " over Confusion Matrix components",
            fontsize=15,
        )
        plt.xlabel("Confusion Matrix components", fontsize=13)
        plt.ylabel(col_idv, fontsize=13)
        plt.show()

    else:

        return dataset.hvplot.kde(
            y=col_idv,
            by=flag_variable,  # Grouping by Predictions
            width=800,
            height=400,
            alpha=0.7,
            ylabel="density",
            xlabel=col_idv,
            title=f"{col_idv}(density)",
            legend="top_right",
        )
