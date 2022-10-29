"""Useful utilities for feature engineering.

The module provides access to some useful utilities around feature selection
and ``sklearn`` style feature transformers.
"""
from .._ext_lib import (FeatureSelector, FeatureSelectorStatistic, Outlier,
                        SupervisedTransformer, UnsupervisedTransformer,
                        WoeBinningTransformer)
