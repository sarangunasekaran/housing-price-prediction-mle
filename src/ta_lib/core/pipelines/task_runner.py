import logging
import time
from collections import namedtuple
from uuid import uuid4

from ta_lib.core.context import create_context
from ta_lib.core.tracking import start_experiment, tracker

from .exceptions import PipelineError
from .processors import get_job_processors

logger = logging.getLogger(__name__)

TaskStatus = namedtuple("TaskStatus", ["status", "msg"])


def create_task_id(context):
    """Create a unique id for a task.

    Parameters
    ----------
    context : ta_lib.core.context.Context

    Returns
    -------
    string
        unique string identifier
    """
    return f"task-{uuid4()}"


def run_task(task_spec):
    """Run a task as specified in `task_spec`.

    This function should be able to run in a different process than the main
    application. ``task_spec`` should have all the information required to
    initialize context and complete the task.
    """
    # create the context object, and initialize
    if "context" in task_spec:
        context = task_spec["context"]
    else:
        cfg = task_spec["config_file"]
        context = create_context(cfg)

    logger = logging.getLogger(__name__)
    t0 = time.time()

    # FIXME: instead of processing also doing IO, can we handle them here
    # as separate steps.
    # FIXME: instead of dataframes, use a datasource object, so we can
    # save/load easily
    task_id = task_spec["id"]
    task_name = task_spec["name"]
    task_params = task_spec["params"]
    job_name = task_spec["job_name"]

    # FIXME: wrap this up in a get_task_processor function
    job_processors = get_job_processors(job_name)
    try:
        processor = job_processors[task_name]
    except KeyError:
        raise PipelineError(
            f"Invalid data cleaning task name: {task_name}.\n\n"
            f"Must be one of {job_processors.keys()}"
        )

    try:
        # execute the task
        processor(context, task_params)
    # check for expected errors (e.g. insufficient data, invalid data etc)
    except PipelineError:
        msg = f"Failed to complete task : {task_id} : Pipeline Error"
        logger.exception(msg)
        return TaskStatus("Fail", msg)
    except BaseException:
        msg = f"Failed to complete task : {task_id} : Unexpected Error"
        logger.exception(msg)
        return TaskStatus("Fail", msg)
    else:
        t1 = time.time()
        msg = f"Successfully completed task : {task_id} : {t1-t0} seconds"
        logger.info(msg)
        return TaskStatus("Success", msg)


def run_tracked_task(task_spec):
    """Run and track a task as specified in `task_spec`.

    This function should be able to run in a different process than the main
    application. ``task_spec`` should have all the information required to
    initialize context and complete the task.
    """
    # create the context object, and initialize
    if "context" in task_spec:
        context = task_spec["context"]
    else:
        cfg = task_spec["config_file"]
        context = create_context(cfg)
        task_spec["context"] = context

    task_id = task_spec["id"]
    task_name = task_spec["name"]
    job_name = task_spec["job_name"]
    expt_name = task_spec["__tracker_experiment_name"]
    parent_run_id = task_spec["__tracker_run_id"]

    try:
        with start_experiment(context, expt_name, run_id=parent_run_id, nested=True):
            with start_experiment(
                context, expt_name, run_name=f"{job_name}:{task_name}", nested=True
            ) as _:

                # execute the task
                out = run_task(task_spec)
                tracker.set_tag("mlflow.note.content", out.msg)
                return out

    except BaseException:
        msg = f"Failed to complete task : {task_id} : Unexpected Error"
        logger.exception(msg)
        tracker.set_tag("mlflow.note.content", msg)
        return TaskStatus("Fail", msg)
