"""Module defining a project context.

The module defines a class ``Context`` that acts as the single source of all
configuration details, credential information, service clients (e.g. spark).
This allows us to create stateless functions that accept the context object
as an argument and can get all the required state information from the object
instead of reaching out to some shared global config or environment variables.

As a design principle, we should avoid adding end-user methods on ``Context``
and instead write modules with stateless functions that take context as an
input.

The primary purpose of the class should be to provide access to configuration
information, initialize/construct resource handles.
"""
import logging
import os.path as op
import posixpath as pp
import re
from copy import deepcopy
from functools import partial

from . import constants
from .tracking import create_client
from .utils import (
    get_fs_and_abs_path,
    get_package_version,
    initialize_random_seed,
    load_yml,
)


def create_context(config_file):
    """Create a context object from a config file path.

    Parameters
    ----------
    config_file : str
        Path for the .yml config file.

    Returns
    -------
    ta_lib.core.context.Context
        Context object generated using the config file.
    """
    ctx = Context.from_config_file(config_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Context created from {config_file}")
    logger.info(f"Package version : {get_package_version()}")
    return ctx


class Context:
    """Class to hold any stateful information for the project."""

    def __init__(self, cfg):
        """Initialize the Context.

        Parameters
        ----------
        cfg : dict
            Config dictionery oaded from a relevant config file
        """
        self._cfg = cfg
        # FIXME: add a utils.configure_logger function and ensure
        # to create folders if missing for fileloggers
        logging.config.dictConfig(cfg["logging"])
        self.random_seed = initialize_random_seed(cfg["core"]["random_seed"])

        self._model_tracker = None

    @property
    def data_catalog(self):
        """Get the Data Catalog from the current project configuration.

        Returns
        -------
        dict
        """
        return self.config["data_catalog"]

    @property
    def job_catalog(self):
        """Get the Job Catalog from the current project configuration.

        Returns
        -------
        dict
        """
        return self.config["job_catalog"]

    @property
    def config(self):
        """Get the current project configuration."""
        return deepcopy(self._cfg)

    @property
    def credentials(self):
        """Get the credentials from the current project configuration.

        Returns
        -------
        dict
        """
        return self.config.get("credentials", {})

    @property
    def model_tracker(self):
        """Get the model tracking client.

        Returns
        -------
        mlflow_client
        """
        if self._model_tracker is None:
            cfg = self.config.get("model_tracker", None)
            if cfg is None:
                self._model_tracker = None
            else:
                self._model_tracker = create_client(cfg)
        return self._model_tracker

    # ----------------
    # Construction API
    # ----------------
    @classmethod
    def from_config_file(cls, cfg_file):
        """Create the Context from a config file location path.

        Parameters
        ----------
        cfg_file : str
            Location path of the .yml config file.

        Returns
        -------
        ta_lib.core.context.Context
        """

        def _dotted_access_getter(key, dct):
            for k in key.split("."):
                dct = dct[k]
            return dct

        def _repl_fn(match_obj, getter):
            return getter(match_obj.groups()[0])

        def _interpolate(val, repl_fn):
            if isinstance(val, dict):
                return {k: _interpolate(v, repl_fn) for k, v in val.items()}
            elif isinstance(val, list):
                return [_interpolate(v, repl_fn) for v in val]
            elif isinstance(val, str):
                val = val.replace(pp.sep, op.sep)
                return re.sub(r"\$\{([\w|.]+)\}", repl_fn, val)
            else:
                return val

        # FIXME: path manipulation for s3 path
        # use yarl.URL, get parts and reconstruct URL

        fs, _ = get_fs_and_abs_path(cfg_file)
        cfg = load_yml(cfg_file, fs=fs)
        config_dir = op.dirname(cfg_file)
        app_cfg = {}
        for key, val in cfg.items():
            fpath = op.join(config_dir, key, val + ".yml")
            app_cfg[key] = load_yml(fpath, fs=fs)

        for k, v in app_cfg["core"].items():
            if v is None:
                app_cfg["core"][k] = getattr(constants, f"DEFAULT_{k.upper()}")

        app_cfg = _interpolate(
            app_cfg,
            partial(_repl_fn, getter=partial(_dotted_access_getter, dct=app_cfg)),
        )

        app_cfg["config_file"] = op.abspath(cfg_file)

        return cls(app_cfg)
