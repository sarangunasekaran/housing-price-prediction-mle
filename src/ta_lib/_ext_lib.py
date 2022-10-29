# Code in this package is not meant to be modified by end-users of code-templates.
from tigerml.core.dataframe.helpers import detigerify
from tigerml.core.preprocessing import Outlier
from tigerml.core.preprocessing.text import string_cleaning
from tigerml.eda import Analyser
from tigerml.eda.base import create_report
from tigerml.core.preprocessing.feature_engg import (
    SupervisedTransformer,
    UnsupervisedTransformer,
    WoeBinningTransformer,
)
from tigerml.core.preprocessing.feature_selection import (
    FeatureSelector,
    FeatureSelectorStatistic,
)
from tigerml.model_eval import (
    RegressionComparison,
    RegressionReport,
    calc_vif,
)
from tigerml.core.scoring import (
    mape,
    wmape,
    root_mean_squared_error,
)
from tigerml.core.utils.pandas import (
    get_bool_cols,
    get_cat_cols,
    get_dt_cols,
    get_non_num_cols,
    get_num_cols,
)

# FIXME: Missing values can be customized. perhaps read from config and initialize ?
# TODO: update Analyzer, DataProcessor to be somewhat customizable
def _get_analyser(df, y=None):
    an = Analyser(df, y=y)
    return an