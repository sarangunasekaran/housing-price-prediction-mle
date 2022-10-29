"""Collection of utility functions."""

import hashlib
import importlib
import logging
import os
import os.path as op
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from io import BytesIO
from uuid import uuid4

import fsspec
import janitor  # noqa
import joblib
import numpy as np
import pandas as pd
import pandas_flavor as pf
import panel as pn
import yaml

import ta_lib

from .base_utils import silence_common_warnings

logger = logging.getLogger(__name__)
pd.options.mode.use_inf_as_na = True


def import_python_file(py_file_path):
    mod_name, ext = op.splitext(op.basename(op.abspath(py_file_path)))
    if ext != ".py":
        raise ValueError("Invalid file extension : {ext}. Expected a py file")
    spec = importlib.util.spec_from_file_location(mod_name, py_file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_pipeline(pipeline, loc):
    """Save an sklearn pipeline in a location.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        Pipeline object to be saved
    loc : str
        Path string of the location where the pipeline has to be saved
    """
    logger.info(f"Saving pipeline to location {loc}")

    os.makedirs(op.dirname(loc), exist_ok=True)
    joblib.dump(pipeline, loc)


def create_job_id(context):
    """Create a unique id for a job.

    Parameters
    ----------
    context : ta_lib.core.context.Context

    Returns
    -------
    string
        unique string identifier
    """
    return f"job-{uuid4()}"


def initialize_environment(debug=True, hide_warnings=True):
    """Initialize the OS Environ with relevant values.

    Parameters
    ----------
    debug: bool, optional
        Whether to set TA_DEBUG to True of False in the environment, default=True
    hide_warnings: bool, optional
        True will hide warnings, default True
    """
    # FIXME: support config
    if debug:
        os.environ["TA_DEBUG"] = "True"
    else:
        os.environ["TA_DEBUG"] = "False"

    # force tigerml to raise an exception on failure
    os.environ["TA_ALLOW_EXCEPTIONS"] = "True"

    if hide_warnings:
        silence_common_warnings()


def is_debug_mode():
    """Check if the current environ is in debug mode."""
    debug_mode = os.environ.get("TA_DEBUG", "True")
    return debug_mode.upper() == "TRUE"


@contextmanager
def timed_log(msg):
    """Log the provided ``msg`` with the execution time."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logging.info(f"{msg} : {end_time-start_time} seconds")


@contextmanager
def disable_logging(highest_level=logging.CRITICAL):
    """Disable all logs below ``highest_level``."""
    # NOTE: this is the attribute that seems to be modified
    # by the call to logging.disable. so we first save this
    # and reset it when exiting the context.
    orig_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(orig_level)
        logging.info("*" * 80)


@contextmanager
def silence_stdout():
    """Silence print stmts on the console unless in debug mode."""
    # debug mode. do nothing.
    if is_debug_mode():
        try:
            yield
        finally:
            return

    # not in debug mode. silence the output by writing to the null device.
    with open(os.devnull, "w") as fp:
        old_dunder_stdout = sys.__stdout__
        old_sys_stdout = sys.stdout
        # FIXME: This doesen't help with notebooks
        # with redirect_stdout(fp):
        #    yield
        try:
            sys.__stdout__ = fp
            sys.stdout = fp
            yield

        except Exception as e:
            sys.stderr.write("Error: {}".format(str(e)))
        finally:
            sys.__stdout__ = old_dunder_stdout
            sys.stdout = old_sys_stdout


def display_as_tabs(figs, width=300, height=300, cloud_env="local"):
    """To display multiple dataset outputs as tabbed panes.

    Parameters
    ----------
    figs: list(tuples)
        List of ('tab_name',widget) to be displayed
    width: int, optional
        width of the output, default 300
    height: int, optional
        height of the output, default 300

    Returns
    -------
    pn.Tabs()
    """
    if cloud_env == "local":
        tabs = pn.Tabs()

        plts = []
        for name, wdgt in figs:
            if isinstance(wdgt, pd.DataFrame):
                wdgt.columns = map(str, wdgt.columns)
                cols = wdgt.select_dtypes("object").columns.tolist()
                wdgt = wdgt.transform_columns(cols, str)
                wdgt = pn.widgets.DataFrame(wdgt, name=name)
            plts.append((name, wdgt))

        tabs.extend(plts)
        return tabs
    elif cloud_env == "Databricks":
        report_dict = {}
        for name, wdgt in figs:
            report_dict[name] = wdgt
        from tigerml.core.reports import create_report

        create_report(report_dict, name="temp", format=".html")
        with open("temp.html", "r") as f:
            html_string = f.read()
        return html_string
    else:
        raise Exception("Input value for cloud_env is incorrect")


def is_relative_path(path):
    """To check if `path` is a relative path or not.

    Parameters
    ----------
    path : str
        path string to be evaluated

    Returns
    -------
    bool
        True if input path is relative else False
    """
    npath = op.normpath(path)
    return op.abspath(npath) != npath


def get_package_path():
    """Get the path of the current installed ta_lib package.

    Returns
    -------
    str
        path string in the current system where the ta_lib package is loaded from
    """
    path = ta_lib.__path__
    return op.dirname(op.abspath(path[0]))


def get_package_version():
    """Return the version of the package."""
    return ta_lib.__version__


def get_data_dir_path():
    """Fetch the data directory path."""
    return op.join(get_package_path(), "..", "data")


def get_fs_and_abs_path(path, storage_options=None):
    """Get the Filesystem and paths from a urlpath and options.

    Parameters
    ----------
    path : string or iterable
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``), or globstring pointing to data.
    storage_options : dict, optional
        Additional keywords to pass to the filesystem class.

    Returns
    -------
    fsspec.FileSystem
       Filesystem Object
    list(str)
        List of paths in the input path.
    """
    fs, _, paths = fsspec.core.get_fs_token_paths(path, storage_options=storage_options)
    if len(paths) == 1:
        return fs, paths[0]
    else:
        return fs, paths


def load_yml(path, *, fs=None, **kwargs):
    """Load a yml file from the input `path`.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    dict
        dictionery of the loaded yml file
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(path, mode="r") as fp:
        return yaml.safe_load(fp, **kwargs)


def create_yml(path, config, fs=None):
    """Dump a dictionary as yaml to output `path`.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    config: dict
        config dictionary to be dumped as yaml file.
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(path, "w") as out_file:
        yaml.safe_dump(config, out_file, default_flow_style=False)


def load_csv(path, *, fs=None, **kwargs):
    """Load a csv file from the file system as specified in the path variable.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    pd.DataFrame
        Dataframe load from the input csv path
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(path, mode="r") as fp:
        return pd.read_csv(fp, **kwargs)


def load_parquet(path, *, fs=None, **kwargs):
    """Load a parquet file from the file system as specified in the path variable.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    pd.DataFrame
        Dataframe load from the input parquet path
    """
    fs = fs or fsspec.filesystem("file")
    with fs.open(path, mode="rb") as fp:
        # FIXME: can be pandas or spark
        return pd.read_parquet(fp, **kwargs)


def save_parquet(df, path, *, fs=None, **kwargs):
    """Save a parquet file in the fs as specified in the path variable.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    """
    fs = fs or fsspec.filesystem("file")
    if fs.protocol == "file":
        # FIXME: utility functions to robustify this
        fs.makedirs(op.dirname(path), exist_ok=True)

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    with fs.open(path, mode="wb") as fp:
        # FIXME: can be pandas or spark
        return df.to_parquet(fp, **kwargs)


def save_csv(df, path, *, fs=None, **kwargs):
    """Save a csv file in the fs as specified in the path variable.

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    """
    fs = fs or fsspec.filesystem("file")
    if fs.protocol == "file":
        # FIXME: utility functions to robustify this
        fs.makedirs(op.dirname(path), exist_ok=True)

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    with fs.open(path, mode="wt", newline="") as fp:
        # FIXME: can be pandas or spark
        return df.to_csv(fp, **kwargs)


def load_data(path, *, fs=None, **kwargs):
    """Load data from the given path. type of data is inferred automatically.

    ``.csv`` and ``.parquet`` are compatible now

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    Returns
    -------
    pd.DataFrame
    """
    # FIXME: Move io utils to a separate module and make things generic
    if path.endswith(".parquet"):
        return load_parquet(path, fs=fs, **kwargs)
    elif path.endswith(".csv"):
        return load_csv(path, fs=fs, **kwargs)
    else:
        raise NotImplementedError()


def save_data(data, path, *, fs=None, **kwargs):
    """Save data into the given path. type of data is inferred automatically.

    ``.csv`` and ``.parquet`` are compatible now

    Parameters
    ----------
    path : string
        Absolute or relative filepath, URL (may include protocols like
        ``s3://``).
    fs : fsspec.filesystem, optional
        Filesystem of the url, by default ``None``

    """
    # FIXME: Move io utils to a separate module and make things generic
    if path.endswith(".parquet"):
        return save_parquet(data, path, **kwargs)
    elif path.endswith(".csv"):
        return save_csv(data, path, **kwargs)
    else:
        raise NotImplementedError()


def df_to_X_y(df, target_col):
    """Create X and y training variables from the provided dataframe.

    Parameters
    ----------
    target_col : string
        column name of the dependant feature

    Returns
    -------
    pd.DataFrame
        Dataframe with only independant features
    pd.DataFrame
        DataFramw with only target_col

    NOTE: This function creates a copy of the data.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col].copy()
    return X, y


def initialize_random_seed(seed):
    """Initialise random seed using the input ``seed``.

    Parameters
    ----------
    seed : int

    Returns
    -------
    int
        seed integer
    """
    logger.info(f"Initialized Random Seed : {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    return seed


def get_fsspec_storage_options(resource_type, credentials):
    """Get storage options from the credentials based on the resource type.

    Parameters
    ----------
    resource_type : string
        'aws' or 'azure' or 'local' etc.,
    credentials : dict
        Dictionery of the credentials

    Returns
    -------
    dict
        Dictionary of the relevant storage options

    Raises
    ------
    NotImplementedError
        Raised for all resourcetype inputs other than 'aws'
    """
    if resource_type == "aws":
        return {
            "key": credentials["aws_access_key_id"],
            "secret": credentials["aws_secret_access_key"],
        }
    else:
        raise NotImplementedError(f"resource type: {resource_type}")


def setanalyse(listA, listB, simplify=True, exceptions_only=False):
    """Given two lists, returns a dictionary of set analysis.

        A-B: set(A) - set(B)
        B-A: set(B) - set(A)
        AuB: A union B
        A^B: A intersection B

    Parameters
    ----------
    listA : list
        input list 1 to be evaluated
    listB : list
        input list 2 to be evaluated
    simplify: bool
        if True, gives only len in each space False gives the entire list.
    exceptions_only: False
        if True, gives only A-B & B-A False gives all 4.
        True is efficient while dealing with large sets or analyzing expections alone.

    Returns
    -------
    dict
        dictionery with the following keys 'A-B', 'B-A', 'A^B', 'AuB'

    """
    A = set(listA)
    B = set(listB)
    output = {"A-B": A - B, "B-A": B - A}
    if ~exceptions_only:
        output["AuB"] = A.union(B)
        output["A^B"] = A.intersection(B)
    if simplify:
        for key, value in output.items():
            output[key] = len(value)
    return output


def setanalyse_df(dfA, dfB, key_cols=None, simplify=True, exceptions_only=False):
    """Given two lists, returns a dictionary of set analysis.

        A-B: set(A) - set(B)
        B-A: set(B) - set(A)
        AuB: A union B
        A^B: A intersection B

    Parameters
    ----------
    dfA : pd.DataFrame
        input list 1 to be evaluated
    dfB : pd.DataFrame
        input list 2 to be evaluated
    key_cols: list
        list of join column names. When None, common column names are used.
    simplify: bool
        if True, gives only len in each space. False gives the entire list.
    exceptions_only: bool
        if True, gives only A-B & B-A. False gives all 4.
        True is efficient while dealing with large sets or analyzing expections alone.

    Returns
    -------
    dict
        dictionery with the following keys 'A-B', 'B-A', 'A^B', 'AuB'
    """
    if key_cols is None:
        key_cols = list(set(dfA.columns).intersection(set(dfB.columns)))

    df1 = dfA.copy()
    df2 = dfB.copy()
    df1["ind"] = 1
    df2["ind"] = 1
    df1 = df1.groupby(key_cols).ind.sum().reset_index()
    df2 = df2.groupby(key_cols).ind.sum().reset_index()
    df = df1.merge(df2, how="outer", on=key_cols)
    ab = df.ind_y.isnull()
    ba = df.ind_x.isnull()
    output = {"A-B": ab, "B-A": ba}
    if ~exceptions_only:
        output["A^B"] = (~ab) & (~ba)
        output["AuB"] = True | ab
    if simplify:
        for key, value in output.items():
            output[key] = value.sum()
    else:
        for key, value in output.items():
            output[key] = df.loc[value, key_cols]

    return output


def merge_expectations(dfA, dfB, onA, onB=None, how="inner"):
    """Given merged dataframe and expectations analysis.

        expectations_result: dict

    Parameters
    ----------
    dfA : pd.DataFrame
        input list 1 to be evaluated
    dfB : pd.DataFrame
        input list 2 to be evaluated
    onA: list or column_name
        list of join column names. When None, common column names are used.
    onB: column_name
        if dfA and dfB to be merged on different column names
    how: merge_type
        {‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}

    Returns
    -------
    expectations_result: dict

    Merge Recommendations:
    - Left expectations:
        Template Expectations:
            : B-A = 0 (Warning)
            : Same data type check (Error)
            : Column existance (Error)
            : Nulls in - On column from A & B = 0 (Warning)
    - Right expectations:
        Template Expectations:
            : A-B = 0 (Warning)
            : Same data type check (Error)
            : Column existance (Error)
            : Nulls in - On column from A & B = 0 (Warning)
    - Inner expectations
        Template Expectations:
            : B-A = 0 (Warning)
            : A-B = 0 (Warning)
            : Same data type check (Error)
            : Column existance (Error)
            : Nulls in - On column from A & B = 0 (Warning)
    - Cross expectations
        Template Expectations:
            : Same data type check (Error)
            : Column existance (Error)
            : Nulls in - On column from A & B = 0 (Warning)

    """
    dfb = dfB.copy()
    if onB is not None:
        dfb = dfb.rename({onB: onA}, axis="columns")

    # Set diff validation
    expectations_result = setanalyse_df(
        dfA, dfb, key_cols=onA, simplify=True, exceptions_only=False
    )

    # Data type check
    data_type_match_dict = {}
    data_type_match_warning = False
    if type(onA) == list:
        for col in onA:
            data_type_match_dict[col] = dfA[col].dtypes == dfb[col].dtypes
            if data_type_match_dict[col] is False:
                data_type_match_warning = True
    else:
        data_type_match_dict[onA] = dfA[onA].dtypes == dfb[onA].dtypes
        if data_type_match_dict[onA] is False:
            data_type_match_warning = True

    expectations_result["data_type_check"] = data_type_match_dict

    # None check
    expectations_result["dfA_nulls"] = dfA[onA].isnull().any()
    expectations_result["dfB_nulls"] = dfb[onA].isnull().any()

    # Assertions
    expect_warnings = {}
    expect_warnings["data_type_mismatch"] = data_type_match_warning
    expect_warnings["nulls_warning"] = (
        expectations_result["dfA_nulls"][expectations_result["dfA_nulls"] is True].size
        > 0
    )
    expect_warnings["nulls_warning"] = (
        expectations_result["dfB_nulls"][expectations_result["dfB_nulls"] is True].size
        > 0
    )
    if how == "left":
        expect_warnings["data_loss_B-A"] = expectations_result["B-A"] != 0
    elif how == "right":
        expect_warnings["data_loss_A-B"] = expectations_result["A-B"] != 0
    elif how == "inner":
        expect_warnings["data_loss_B-A"] = expectations_result["B-A"] != 0
        expect_warnings["data_loss_A-B"] = expectations_result["A-B"] != 0
    else:
        pass

    expectations_result["actionable_warnings"] = expect_warnings
    return expectations_result


def get_feature_names_from_column_transformer(col_trans):
    """Get feature names from a sklearn column transformer.

    The `ColumnTransformer` class in `scikit-learn` supports taking in a
    `pd.DataFrame` object and specifying `Transformer` operations on columns.
    The output of the `ColumnTransformer` is a numpy array that can used and
    does not contain the column names from the original dataframe. The class
    provides a `get_feature_names` method for this purpose that returns the
    column names corr. to the output array. Unfortunately, not all
    `scikit-learn` classes provide this method (e.g. `Pipeline`) and still
    being actively worked upon.
    This utility function is a temporary solution until the proper fix is
    available in the `scikit-learn` library.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder as skohe

    # SimpleImputer has add_indicator attribute that distinguishes it from other transformers
    # Encoder had get_feature_names attribute that distinguishes it from other transformers
    col_name = []

    for (
        transformer_in_columns
    ) in (
        col_trans.transformers_
    ):  # the last transformer is ColumnTransformer's 'remainder'
        is_pipeline = 0
        raw_col_name = list(transformer_in_columns[2])

        if isinstance(transformer_in_columns[1], Pipeline):
            # if pipeline, get the last transformer
            transformer = transformer_in_columns[1].steps[-1][1]
            is_pipeline = 1
        else:
            transformer = transformer_in_columns[1]
        try:
            if isinstance(transformer, str):
                if transformer == "passthrough":
                    names = col_trans._feature_names_in[raw_col_name].tolist()

                elif transformer == "drop":
                    names = []

                else:
                    raise RuntimeError(
                        f"Unexpected transformer action for unaccounted cols :"
                        f"{transformer} : {raw_col_name}"
                    )

            elif isinstance(transformer, skohe):
                names = list(transformer.get_feature_names(raw_col_name))

            elif isinstance(transformer, SimpleImputer) and transformer.add_indicator:
                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [
                    raw_col_name[idx] + "_missing_flag"
                    for idx in missing_indicator_indices
                ]

                names = raw_col_name + missing_indicators

            else:
                names = list(transformer.get_feature_names())

        except AttributeError:
            names = raw_col_name
        if is_pipeline:
            names = [f"{transformer_in_columns[0]}_{col_}" for col_ in names]
        col_name.extend(names)

    return col_name


def get_dataframe(arr, feature_names):
    """Convert an an numpy array into a dataframe.

    Parameters
    ----------
    arr : np.array
        Input 2D Array
    feature_names : list(string)
        List of column names in the same order of the data in the array

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame(arr, columns=feature_names)


# Pyflav utils
@pf.register_dataframe_method
def add_column_from_dt(df, col, new_col, op):
    """Add a column to the input dataframe with the relevant `op` funtion applied.

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataframe where the new column has to be added
    col : string
        Column name of the column to be transformed
    new_col : string
        Column name of the transformed column
    op : `function`
        Operator or function to be run on the `col`. this takes pd.Series as input.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new column
    """
    mask = ~df[col].isna()
    df.loc[mask, new_col] = op(df[col][mask])
    return df


@pf.register_dataframe_method
def remove_duplicate_rows(df, col_names, keep_first=True):
    """Remove duplicate rows wrt to the input columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe where duplicates have to be removed
    col_names : list(str)
        List of column names to be used as key to drop duplicates
    keep_first : bool, optional
        option whether to keep the first row in a set of duplicates, by default True

    Returns
    -------
    pd.DataFrame
        Datsframe with duplicates removed
    """
    keep = "first" if keep_first else "last"
    mask = df.duplicated(subset=col_names, keep=keep)
    return df[~mask]


@pf.register_dataframe_method
def passthrough(df):
    """Return the input dataframe."""
    return df


def merge_info(left_df, right_df, merge_df):
    """Get the column and row comparison summary of the provided dataframes.

    The returned data is useful to quickly sanity check a merge operation by
    checking the number of rows and columns.

    Parameters
    ----------
    left_df : pd.DataFrame
    right_df : pd.DataFrame
    merge_df : pd.DataFrame
        merged dataframe of left_df and right_df

    Returns
    -------
    pd.DataFrame
        Merge columns and rows comparison summary dataframe
    """
    n_cols = [len(left_df.columns), len(right_df.columns), len(merge_df.columns)]
    n_rows = [len(left_df), len(right_df), len(merge_df)]
    index = ["left_df", "right_df", "merged_df"]
    return pd.DataFrame({"n_cols": n_cols, "n_rows": n_rows}, index=index)


def custom_train_test_split(df, splitter=None, by=None):
    """Split the provided dataset using custom criteria.

    The primary utility of this function is to be able to provide a custom
    criteria to split the datasets.

    Parameters
    ----------
    df: pd.DataFrame
        The tabular dataset to be split.

    splitter: object
        The splitter object should have a method named `split` that takes
        in a dataframe and a series. Any of the ``sklearn`` splitters listed
        here are acceptable: https://scikit-learn.org/stable/modules/classes.html#splitter-classes  # noqa

    by: str or callable
        The criteria can simply be a ``column`` in the input dataframe but
        can also be a ``function``. When the latter option is used, the
        function should take in the input ``dataframe`` and return a series
        that will then be used by the splitter object.

    Returns
    -------
    splits: list
        The list of splits generated by the splitter object.
    """

    if isinstance(by, str):
        split_by = df[by]
    elif callable(by):
        split_by = by(df)
    else:
        raise ValueError(
            "`by` must be a column name or a callable that returns a series"
        )
    split_sets = []
    for indexes in splitter.split(df, split_by):
        for index in indexes:
            split_sets.append(df.loc[index])
    return split_sets


def load_dataframe(path):
    """Load parquet file from the `path` specified.

    Parameters
    ----------
    path : string
        urlpath for the data to be loaded

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_parquet(path).reset_index()
    if "index" in df.columns:
        df = df.drop(columns="index")
    return df


def load_pipeline(path):
    """Load model pipeline from the `path` specified.

    Parameters
    ----------
    path : string
        urlpath for the pipeline to be loaded

    Returns
    -------
    pd.DataFrame
    """
    return joblib.load(path)


def hash_object(obj, expensive=False, block_size=4096):
    """Return a content based hash for the input `obj`.

    The returned hash value can be used to verify equality of two objects.
    If the hash values of two objects are equal, then they are identical and
    one can be replaced with another.

    Parameters
    ----------
    obj: Object

    Returns
    -------
    string
    """
    hasher = hashlib.sha256()

    if expensive:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fname = op.join(tmp_dir, "tmp.joblib")
            joblib.dump(obj, tmp_fname)
            with open(tmp_fname, "rb") as fp:
                while True:
                    data = fp.read(block_size)
                    if len(data) <= 0:
                        break
                    hasher.update(data)
    else:
        fp = BytesIO()
        joblib.dump(obj, fp)
        data = fp.getvalue()
        hasher.update(data)

    return hasher.hexdigest()
