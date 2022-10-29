from copy import deepcopy

from ta_lib.core.tracking import is_tracker_supported

from .task_runner import create_task_id, run_task, run_tracked_task


def create_job_plan(context, job_spec):
    """Create a job execution plan from the provide job specification.

    Parameters
    ----------
    context: Context
        The project context object.
    job_spec: dict
        Dictionary containing the job-specification.
    """
    job_spec = deepcopy(job_spec)
    job_plan = {"name": job_spec["name"], "stages": []}
    for stage_spec in job_spec["stages"]:
        stage_plan = {}
        stage_plan["name"] = stage_spec["name"]
        stage_plan["tasks"] = []
        for task_spec in stage_spec["tasks"]:
            task_params = deepcopy(task_spec)
            task_params["config_file"] = context.config["config_file"]
            task_params["id"] = create_task_id(context)

            # FIXME: should we instead add the resolve function name ?
            task_plan = {}
            task_plan["params"] = task_params
            task_plan["name"] = task_params["name"]
            task_plan["job_name"] = job_spec["name"]
            if is_tracker_supported(context):
                task_plan["runner"] = run_tracked_task
                task_plan["__tracker_run_id"] = job_spec["__tracker_run_id"]
                task_plan["__tracker_experiment_name"] = job_spec[
                    "__tracker_experiment_name"
                ]
            else:
                task_plan["runner"] = run_task

            # add the task plan to job plan
            stage_plan["tasks"].append(task_plan)
        job_plan["stages"].append(stage_plan)

    return job_plan
