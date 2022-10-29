"""Module to consolidate constants used in the package."""

import os.path as op

from .utils import get_data_dir_path, get_package_path

DEFAULT_RANDOM_SEED = 0
DEFAULT_DATA_BASE_PATH = get_data_dir_path()
DEFAULT_LOG_BASE_PATH = op.join(get_package_path(), "..", "logs")
DEFAULT_ARTIFACTS_PATH = op.join(get_package_path(), "..", "artifacts")
DEFAULT_MODEL_TRACKER_BASE_PATH = op.join(get_package_path(), "..", "mlruns")
