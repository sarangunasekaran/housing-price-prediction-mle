"""Joblib based job runner.

We support a very simple DAG structure that mirrors pyspark.
We will have stages which are executed sequentially. Each stage will have
a bunch of tasks which are to be executed in parallel.

We will not support any recursive defn here, i.e. a task cannot have another
stage in it. If you need more control over DAG structure, use a tool like
``Dask`` or ``Airflow``

We shall assume each pipeline module will provide a function to create a
plan (``create_job_plan``) and a ``task_runner`` function to execute tasks.

The data structures are as follows:
    * A ``Job Plan`` is simply a list of stages.
    * A ``Stage`` is a list of tasks.
    * A ``Task`` is a dictionary with information reqd. for the task.
"""
import logging
from collections import namedtuple

from joblib import Parallel, delayed, parallel_backend

from ta_lib.core.tracking import is_tracker_supported

logger = logging.getLogger(__name__)

# Helper cls to return job status
JobStatus = namedtuple("JobStatus", ["status", "msg"])


def _with_worker_initialization(self, custom_init_fn):
    """Monkey patch worker initialization with provided initialization function.

    There doesen't seem to be a way to run an initialization function on each
    worker before scheduling tasks on them. See https://github.com/joblib/joblib/issues/381

    The fix here is a suggested solution in the above issue and seems to work
    fine. Essentially we are monkey-patching a private method that is used to
    initialize workers.
    """
    if custom_init_fn is None:
        return self

    hasattr(self._backend, "_workers") or self.__enter__()

    original_init_fn = self._backend._workers._initializer

    def new_init_fn():
        original_init_fn()
        custom_init_fn()

    if callable(original_init_fn):
        self._backend._workers._initializer = new_init_fn
    else:
        self._backend._workers._initializer = custom_init_fn

    return self


def _safe_runner(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except BaseException:
        # FIXME: use logger depending on whether local/subprocess is used
        # NOTE: This should not happen in a normal run.
        # If we reach here, it should be treated as a bug and
        # handled in task_runner functions appripriately.
        import traceback

        traceback.print_exc()


def sequential_executor(context, job_plan, n_threads_per_worker=-1):
    """Execute the job plan using a single worker process."""
    stages = job_plan["stages"]
    for stage in stages:
        logger.info(f'Running Stage : {stage["name"]}')
        for task in stage["tasks"]:
            logger.info(f'Running task : {task["name"]} : {task["params"]["id"]}')
            runner = task["runner"]
            params = task["params"]
            params["context"] = context
            params["job_name"] = task["job_name"]
            if is_tracker_supported(context):
                params["__tracker_run_id"] = task["__tracker_run_id"]
                params["__tracker_experiment_name"] = task["__tracker_experiment_name"]
            out = _safe_runner(runner, params)
            logger.info(out)


def parallel_executor(
    context, job_plan, init_fn=None, n_workers=-1, n_threads_per_worker=1
):
    """Execute the job plan in parallel.

    Use ``joblib`` delayed function and futures instead of standard
    ``multiprocessing``. We use ``joblib`` API to provide a status update and also
    works well in a notebook
    """
    # FIXME: move some of the settings to config file
    with parallel_backend("loky", inner_max_num_threads=2):
        with Parallel(n_jobs=n_workers, prefer="processes") as _parallel:
            with _with_worker_initialization(_parallel, init_fn) as parallel:
                stages = job_plan["stages"]
                for stage in stages:
                    logger.info(f'Running Stage : {stage["name"]}')
                    tasks = []
                    for task in stage["tasks"]:
                        logger.info(
                            f'Running task : {task["name"]} : {task["params"]["id"]}'
                        )
                        runner = task["runner"]
                        params = task["params"]
                        params["job_name"] = task["job_name"]
                        if is_tracker_supported(context):
                            params["__tracker_run_id"] = task["__tracker_run_id"]
                            params["__tracker_experiment_name"] = task[
                                "__tracker_experiment_name"
                            ]
                        tasks.append(delayed(_safe_runner)(runner, params))
                    results = parallel(tasks)
                    print(results)


def execute_job_plan(
    context, job_plan, init_fn=None, n_workers=1, n_threads_per_worker=-1
):
    if n_workers == 1:
        if init_fn is not None:
            raise RuntimeError(
                "Initialization function is not expected for" " Sequential Executor"
            )
        sequential_executor(
            context, job_plan, n_threads_per_worker=n_threads_per_worker
        )
    else:
        parallel_executor(
            context,
            job_plan,
            init_fn=init_fn,
            n_workers=n_workers,
            n_threads_per_worker=n_threads_per_worker,
        )


def main(
    context, job_planner, job_spec, init_fn=None, n_workers=1, n_threads_per_worker=-1
):
    job_plan = job_planner(context, job_spec)
    execute_job_plan(
        context,
        job_plan,
        init_fn=init_fn,
        n_workers=n_workers,
        n_threads_per_worker=n_threads_per_worker,
    )
