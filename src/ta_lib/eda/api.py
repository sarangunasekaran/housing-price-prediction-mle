"""Utiltiies around ``Exploratory Data Analysis``.

The module provides useful utilties for generating common ``EDA`` outputs.
"""
from .analysis import (
    calc_vif,
    get_bivariate_plots,
    get_correlation_table,
    get_data_health_summary,
    get_density_plots,
    get_duplicate_columns,
    get_feature_importances,
    get_frequency_plots,
    get_missing_values_summary,
    get_outliers,
    get_percentile_plots,
    get_target_correlation,
    get_variable_summary,
    residual_plots,
)
