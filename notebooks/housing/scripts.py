"""Module for listing down additional custom functions required for the notebooks."""

import pandas as pd


def binned_median_house_value(df):
    """Bin the median_income column using quantiles."""
    return pd.qcut(df["median_income"], q=5)
