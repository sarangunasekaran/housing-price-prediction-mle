"""Module with utility functions that can be called at library import."""
import warnings


def silence_common_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The sklearn.metrics.classification module",
        category=FutureWarning,
    )
    warnings.filterwarnings("ignore", ".*optional dependency `torch`.*")
    warnings.filterwarnings(
        "ignore", ".*title_format is deprecated. Please use title instead.*"
    )
    warnings.filterwarnings(
        "ignore",
        "`should_run_async` will not call `transform_cell` automatically in the future.*",
    )
    warnings.filterwarnings(
        "ignore", "The global colormaps dictionary is no longer considered public API."
    )
