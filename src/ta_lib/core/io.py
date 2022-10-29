"""Utility functions around IO operations on filesystems."""

from ta_lib.core.utils import get_fs_and_abs_path, get_fsspec_storage_options


def fs(context, path_url="", credential_id=None):
    """Return a FileSytem object for the provided path."""
    # need to find the appropriate set of credentials
    # given the path prefix
    options = None
    if credential_id is not None:
        creds = context.credentials[credential_id]
        options = get_fsspec_storage_options(creds["resource_type"], creds)
    fs, _ = get_fs_and_abs_path(path_url, storage_options=options)
    return fs
