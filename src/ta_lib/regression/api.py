"""Utilities for ``Regression`` usecases.

The module provides custom ``Estimators`` and ``Evaluators`` for
regression problems.
"""

from .estimators import *
from .evaluation import *
from .._ext_lib import mape, wmape, root_mean_squared_error