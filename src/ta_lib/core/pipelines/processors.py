import functools
import glob
import os.path as op

from ...core.utils import import_python_file

# FIXME: should we instead track this inside context object ?
# private variable to track registerd processors
_PROCESSORS = {}


def load_job_processors(folder):
    """Load job processors defined in py files in the provided path."""
    py_files = glob.glob(op.join(folder, "*.py"))
    for path in py_files:
        if path == op.abspath(__file__):
            continue
        import_python_file(path)


def get_job_processors(job_name):
    """Return the processors available for the given job."""
    try:
        return _PROCESSORS[job_name]
    except KeyError:
        avlb_jobs = list_jobs()
        raise ValueError(
            f"Unexpected job name : {job_name}. \n\nMust be one of {avlb_jobs}"
        )


def register_processor(job_name, task_name):
    """Register the decorated function as a data processor task.

    The decorated function can be referred to in the job specifications
    using the ``job_name`` and ``task_name`` used to register it.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        _PROCESSORS.setdefault(job_name, {})[task_name] = _wrapper
        return _wrapper

    return _decorator


def list_jobs():
    return _PROCESSORS.keys()


def list_job_processors(job_name):
    return get_job_processors.keys()
