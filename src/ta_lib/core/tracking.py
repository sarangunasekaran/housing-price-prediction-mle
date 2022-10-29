from contextlib import contextmanager

try:
    import mlflow as tracker

    # NOTE: The import is needed for `mlflow.sklearn.log_model` call elsewhere
    import mlflow.sklearn  # noqa
except ImportError:
    _MLFLOW_SUPPORTED = False
    tracker = None
else:
    _MLFLOW_SUPPORTED = True


def _validate_requirements():
    if not _MLFLOW_SUPPORTED:
        raise RuntimeError("The tracking features require `mlflow` to be installed.")


def _get_tracking_uri(client):
    # FIXME:
    return client._tracking_client.tracking_uri


def _get_or_create_experiment_id(client, expt_name):
    expt = client.get_experiment_by_name(expt_name)
    if expt is not None:
        expt_id = expt.experiment_id
    else:
        expt_id = client.create_experiment(
            expt_name, artifact_location=client.artifact_uri
        )
    return expt_id


def is_tracker_supported(context):
    """Return ``True`` if context supports using an experiment tracker.

    To enable ``Experiment tracking``, the tracking backend library needs
    to installed and the project configuration needs a section on
    ``model_tracker`` that has connection details for the tracking server.
    """
    try:
        client = context.model_tracker
        return client is not None
    except RuntimeError:
        return False
    else:
        return True


def create_client(cfg):
    """Create a tracking client.

    Multiple backends maybe be supported. The function picks the appropriate
    backed by using the value of ``driver`` specified in the ``model_tracker``
    configuration.
    """
    _validate_requirements()

    driver = cfg["driver"]
    if driver == "mlflow":
        client = tracker.tracking.MlflowClient(
            tracking_uri=cfg["backend_store"]["uri"],
            registry_uri=cfg["model_registry"]["uri"],
        )
        client.artifact_uri = cfg["artifact_store"]["uri"]
    else:
        raise NotImplementedError()

    return client


@contextmanager
def start_experiment(context, expt_name=None, run_id=None, run_name=None, nested=False):
    """Start an ``Experiment`` and track it using the tracking server.

    Parameters
    ----------
    context: Context
        The project context object.
    expt_name: string
        The name to associate with the current experiment.
    run_id: string
        If not ``None``, the current run would be associated with an existing
        run with this id. This can be used to update an existing run with
        additional information.
    run_name: string
        If not ``None``, a new run with the provided name is created.
    nested: bool
        If ``True``, the run is considered to be part of a hierarchy and associated
        with a parent run.

    Returns
    -------
    tracker
        A ``tracker`` object that can be used to log parameters, metrics, etc.
    """

    _validate_requirements()

    client = context.model_tracker

    expt_id = _get_or_create_experiment_id(client, expt_name)
    tracker.set_experiment(expt_name)

    # set tracking uri for the context
    old_tracking_uri = tracker.get_tracking_uri()
    tracker.set_tracking_uri(_get_tracking_uri(client))

    # FIXME: if run_id is provided and doesen't already exist, create one.

    # start experiment run
    try:
        with tracker.start_run(
            experiment_id=expt_id, run_id=run_id, run_name=run_name, nested=nested
        ) as _:
            # FIXME: return a custom tracker as has some api weirdness.

            yield tracker

    finally:
        # reset the tracking uri to the value before the start of
        # current context
        tracker.set_tracking_uri(old_tracking_uri)
