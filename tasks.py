import os
import os.path as op
import platform
from contextlib import contextmanager
import shutil
import tempfile
import time
import yaml
from invoke import Collection, UnexpectedExit, task

# Some default values
PACKAGE_NAME = "ta_lib"
ENV_PREFIX = "ta-lib"
ENV_PREFIX_PYSPARK = "ta-lib-pyspark"
NUM_RETRIES = 10
SLEEP_TIME = 1

OS = platform.system().lower()
ARCH = platform.architecture()[0][:2]
PLATFORM = f"{OS}-cpu-{ARCH}"
DEV_ENV = "dev"  # one of ['dev', 'run', 'test']

HERE = op.dirname(op.abspath(__file__))
SOURCE_FOLDER = op.join(HERE, "src", PACKAGE_NAME)
TESTS_FOLDER = op.join(HERE, 'tests')
CONDA_ENV_FOLDER = op.join(HERE, "deploy", "conda_envs")
PYSPARK_ENV_FOLDER = op.join(HERE, "deploy", "pyspark")
NOTEBOOK_FOLDER = op.join(HERE, "notebooks", "tests")

_TASK_COLLECTIONS = []

# ---------
# Utilities
# ---------
def _get_env_name(platform, env):
    # FIXME: do we need platform ?
    return f"{ENV_PREFIX}-{env}"

def _get_env_name_pyspark(platform, env):
    # FIXME: do we need platform ?
    return f"{ENV_PREFIX_PYSPARK}-{env}"

def _change_permissions_recursive(path, mode):
    for root, dirs, files in os.walk(path, topdown=False):
        for _dir in [os.path.join(root, d) for d in dirs]:
            os.chmod(_dir, mode)
        for _file in [os.path.join(root, f) for f in files]:
            os.chmod(_file, mode)


def _clean_rmtree(path):
    for _try in range(NUM_RETRIES):
        try:
            _change_permissions_recursive(path, 0o777)
            shutil.rmtree(path)
        except Exception as e:
            time.sleep(SLEEP_TIME)
            print(f"{path} Remove failed with error {e}:: Retrying ..")
            continue
        print(f"{path} Remove Success")
        break


@contextmanager
def py_env(c, env_name):
    """Activate a python env while the context is active."""

    # FIXME: This works but takes a few seconds. Perhaps find a few to robustly
    # find the python path and just set the PATH variable ?
    if OS == "windows":
        # we assume conda binary is in path
        cmd = f"conda activate {env_name}"
    else:
        cmd = f'eval "$(conda shell.bash hook)" && conda activate {env_name}'
    with c.prefix(cmd):
        yield


def _create_task_collection(name, *tasks):
    """Construct a Collection object."""
    coll = Collection(name)
    for task_ in tasks:
        coll.add_task(task_)
    _TASK_COLLECTIONS.append(coll)
    return coll


def _create_root_task_collection():
    return Collection(*_TASK_COLLECTIONS)


# ---------
# debug tasks
# ---------


@task(name="check-reqs")
def check_setup_prerequisites(c):
    _failed = []

    # check the folder has no spaces in the path
    if " " in HERE:
        raise RuntimeError("The path to the current folder has whitespaces in it.")

    for binary in ["git", "conda"]:
        try:
            out = c.run(f"{binary} --version", hide="out")
        except UnexpectedExit:
            print(
                f"ERROR: Failed to find `{binary}` in path. "
                "See `pre-requisites` section in `README.md` for some pointers."
            )
            _failed.append(binary)
        else:
            print(f"SUCCESS: Found `{binary}` in path : {out.stdout}")

    # FIXME: Figure out colored output (on windows) and make the output easier
    # to read/interpret.
    # for now, make a splash when failing and make it obvious.
    if _failed:
        raise RuntimeError(f"Failed to find the following binaries in path : {_failed}")


_create_task_collection("debug", check_setup_prerequisites)


# Dev tasks
# ---------
@task(
    help={
        "platform": (
            "Specifies the platform spec. Must be of the form "
            "``{windows|linux}-{cpu|gpu}-{64|32}``"
        ),
        "env": "Specifies the enviroment type. Must be one of ``{dev|test|run}``",
        "force": "If ``True``, any pre-existing environment with the same name will be overwritten",
    }
)
def setup_env(c, platform=PLATFORM, env=DEV_ENV, force=False):
    """Setup a new development environment.

    Creates a new conda environment with the dependencies specified in the file
    ``env/{platform}-{env}.yml``. To overwrite an existing environment with the
    same name, set the flag ``force`` to ``True``.
    """

    # run pre-checks
    check_setup_prerequisites(c)

    # FIXME: We might want to split the env.yml file into multiple
    # files: run.yml, build.yml, test.yml and support combining them for
    # different environments
    force_flag = "" if not force else "--force"

    env_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"{platform}-{env}.lock"))
    req_file = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"requirements-{platform}-{env}.txt")
    )
    req_flag = True
    if not op.isfile(env_file):
        raise ValueError(f"The conda env file is not found : {env_file}")
    if not op.isfile(req_file):
        req_flag = False
    env_name = _get_env_name(platform, env)

    out = c.run(f"conda create --name {env_name} --file {env_file}  {force_flag} -y")

    # check for jupyterlab
    with open(env_file, "r") as fp:
        env_cfg = fp.read()

    # installating jupyter lab extensions
    extensions_file = op.abspath(
        op.join(CONDA_ENV_FOLDER, "jupyterlab_extensions.yml")
    )
    with open(extensions_file) as fp:
        extensions = yaml.safe_load(fp)

    # install the code-template modules
    with py_env(c, env_name):

        # install pip requirements
        if req_flag:
            c.run(f"pip install -r {req_file} --no-deps")

        # install the current package
        c.run(f"pip install -e {HERE}")

        is_jupyter = False
        if "jupyterlab-" in env_cfg:
            is_jupyter = True
            
        if is_jupyter:
            # install jupyterlab extensions
            for extension in extensions["extensions"]:
                extn_name = "@{channel}/{name}@{version}".format(**extension)
                c.run(f"jupyter labextension install --no-build {extn_name}",)

            out = c.run("jupyter lab build")

    # FIXME: create default folders that are expected. these need to be handled
    # when convering to cookiecutter templates
    os.makedirs(op.join(HERE, "logs"), exist_ok=True)
    os.makedirs(op.join(HERE, "docs", "build","html"), exist_ok=True)
    os.makedirs(op.join(HERE, "mlruns"), exist_ok=True)
    os.makedirs(op.join(HERE, "data"), exist_ok=True)

@task(
    help={
        "platform": (
            "Specifies the platform spec. Must be of the form "
            "``{windows|linux}-{cpu|gpu}-{64|32}``"
        ),
        "env": "Specifies the enviroment type. Must be one of ``{dev|test|run}``",
        "force": "If ``True``, any pre-existing environment with the same name will be overwritten",
    }
)
def setup_env_pyspark(c, platform=PLATFORM, env=DEV_ENV, force=True):
    """Setup a new development environment.

    Creates a new conda environment with the dependencies specified in the file
    ``env/{platform}-{env}.yml``. To overwrite an existing environment with the
    same name, set the flag ``force`` to ``True``.
    """

    # run pre-checks
    check_setup_prerequisites(c)

    # FIXME: We might want to split the env.yml file into multiple
    # files: run.yml, build.yml, test.yml and support combining them for
    # different environments
    force_flag = "" if not force else "--force"

    env_file = op.abspath(op.join(PYSPARK_ENV_FOLDER, f"{platform}-{env}.yml"))
    req_file = op.abspath(
        op.join(PYSPARK_ENV_FOLDER, f"requirements-{platform}-{env}.txt")
    )
    req_flag = True
    if not op.isfile(env_file):
        raise ValueError(f"The conda env file is not found : {env_file}")
    if not op.isfile(req_file):
        req_flag = False
    env_name = _get_env_name_pyspark(platform, env)
    print(f"env_name is {env_name} \n env_file is {env_file}")  # Todo: Delete this
    # out = c.run(f"conda env create --name {env_name} -f {env_file}  {force_flag}")
    out = c.run(f"conda env create -f {env_file}  {force_flag}")

    # check for jupyterlab
    with open(env_file, "r") as fp:
        env_cfg = fp.read()

    # installing jupyter lab extensions
    extensions_file = op.abspath(
        op.join(CONDA_ENV_FOLDER, "jupyterlab_extensions.yml")
    )
    with open(extensions_file) as fp:
        extensions = yaml.safe_load(fp)

    # install the code-template modules
    with py_env(c, env_name):

        # install pip requirements
        if req_flag:
            c.run(f"pip install -r {req_file} --no-deps")

        # install the current package
        c.run(f"pip install -e {HERE}") # Todo: Alternative is a separate src folder for pyspark modules only

        is_jupyter = False
        if "jupyterlab-" in env_cfg:
            is_jupyter = True

        if is_jupyter:
            # install jupyterlab extensions
            for extension in extensions["extensions"]:
                extn_name = "@{channel}/{name}@{version}".format(**extension)
                c.run(f"jupyter labextension install --no-build {extn_name}",)

            out = c.run("jupyter lab build")

    # FIXME: create default folders that are expected. these need to be handled
    # when convering to cookiecutter templates
    os.makedirs(op.join(HERE, "logs"), exist_ok=True)
    os.makedirs(op.join(HERE, "mlruns"), exist_ok=True)
    os.makedirs(op.join(HERE, "data"), exist_ok=True)

@task(name="setup_addon")
def setup_addon(
    c,
    platform=PLATFORM,
    env=DEV_ENV,
    documentation=False,
    testing=False,
    formatting=False,
    jupyter=False,
    extras=False,
):
    """Installs add on packages related to documentation, testing or code-formatting.

    Dependencies related to documentation, testing or code-formatting can be installed on demand.
    By Specifying `--documentation`, documentation related packages get installed. Similarly to install testing
    or formatting related packages, flags `--testing` `formatting` will do the needful installations.
    """
    env_name = _get_env_name(platform, env)

    addons_documentation = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"addon-documentation-{platform}-{env}.yml")
    )
    addons_testing = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"addon-testing-{platform}-{env}.yml")
    )
    addons_formatting = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"addon-code_format-{platform}-{env}.yml")
    )
    addons_jupyter = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"addon-jupyter-{platform}-{env}.yml")
    )
    addons_extras = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"addon-extras-{platform}-{env}.yml")
    )
    with py_env(c, env_name):
        if documentation:
            c.run(
                f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_documentation}"
            )
        if testing:
            c.run(
                f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_testing}"
            )
        if formatting:
            c.run(
                f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_formatting}"
            )
        if jupyter:
            c.run(
                f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_jupyter}"
            )

            extensions_file = op.abspath(
                op.join(CONDA_ENV_FOLDER, "jupyterlab_extensions.yml")
            )
            with open(extensions_file) as fp:
                extensions = yaml.safe_load(fp)

            for extension in extensions["extensions"]:
                extn_name = "@{channel}/{name}@{version}".format(**extension)
                c.run(f"jupyter labextension install --no-build {extn_name}",)

            out = c.run("jupyter lab build")
        if extras:
            c.run(
                f"conda env update --name {ENV_PREFIX}-{DEV_ENV} --file {addons_extras}"
            )
    if documentation:
        os.makedirs(op.join(HERE, "docs/build"), exist_ok=True)
        os.makedirs(op.join(HERE, "docs/source"), exist_ok=True)
    if extras:
        os.makedirs(op.join(HERE, "mlruns"), exist_ok=True)


@task(name="format-code")
def format_code(c, platform=PLATFORM, env=DEV_ENV, path="."):
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        c.run(f"black {path}", warn=True)
        c.run(f"isort -rc {path}")


@task(name="refresh-version")
def refresh_version(c, platform=PLATFORM, env=DEV_ENV):
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        res = c.run(f"python {HERE}/setup.py --version")
    return res.stdout


@task(
    name="run-notebooks",
    help={
        "template": (
            "Specifies the template folder to run"
            "``{regression|classification|all}``"
            "If all then all the templates are run."
        ),
        "notebookid": "Specifies id of notebooks. Must be one of ``{01|02|03|all}``",
        "timeout": "Maximum execution time beyond which the notebook returns an exception. default is 600 secs.",
        "refresh": "If `True`, notebooks are run and if successful refresh is also done else, notebooks are run for errors.",
    },
)
def run_notebook(
    c,
    platform=PLATFORM,
    env=DEV_ENV,
    template="all",
    notebookid="all",
    timeout=600,
    refresh=False,
):
    """Run notebooks to checks for any errors."""
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        res = c.run(
            f"python {NOTEBOOK_FOLDER}/utils.py {template} {notebookid} {timeout} {refresh}"
        )
    return res.stdout


@task(name="info")
def setup_info(c, platform=PLATFORM, env=DEV_ENV):
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        res = c.run(f"pip list")
    return res.stdout


@task(name="build-docker")
def _build_docker_image(c):
    with tempfile.TemporaryDirectory() as tempdir:
        docker_file = op.join(HERE, "deploy", "docker", "Dockerfile")
        shutil.copyfile(docker_file, op.join(tempdir,'Dockerfile'))
        docker_file = op.join(tempdir,'Dockerfile')
        template = op.basename(HERE)
        if template == "regression-py":
            tag = "ct-reg-py"
        elif template == "classification-py":
            tag = "ct-class-py"
        elif template == "tpo-py":
            tag = "ct-tpo-py"
        elif template == "rtm-py":
            tag = "ct-rtm-py"
        else:
            raise ValueError(f"Unknown template : {template}")
        shutil.copytree(op.join(HERE, "deploy"), op.join(tempdir,"deploy"))
        shutil.copytree(op.join(HERE, "production"), op.join(tempdir,"production"))
        shutil.copytree(op.join(HERE, "src", "ta_lib"), op.join(tempdir, "src", "ta_lib"))
        shutil.copyfile(op.join(HERE, "setup.py"), op.join(tempdir, "setup.py"))
        shutil.copyfile(op.join(HERE, "setup.cfg"), op.join(tempdir, "setup.cfg"))
        build_context = tempdir
        c.run(f"docker build -f {docker_file} -t {tag} {build_context}")


@task(
    help={
        "platform": (
            "Specifies the platform spec. Must be of the form "
            "``{windows|linux}-{cpu|gpu}-{64|32}``"
        ),
        "force": "If ``True``, any pre-existing environment with the same name will be overwritten",
    }
)
def setup_ci_env(c, platform=PLATFORM, force=False):
    """Setup a new CI environment for git.

    Creates a new conda environment with the dependencies specified in the file
    ``env/{platform}-{env}.yml``. To overwrite an existing environment with the
    same name, set the flag ``force`` to ``True``.
    """
    env = 'ci'
    force_flag = "" if not force else "--force"
    env_file = op.abspath(op.join(CONDA_ENV_FOLDER, f"{platform}-{env}.lock"))
    req_file = op.abspath(
        op.join(CONDA_ENV_FOLDER, f"requirements-{platform}-{env}.txt")
    )

    req_flag = True
    if not op.isfile(env_file):
        raise ValueError(f"The conda env file is not found : {env_file}")
    if not op.isfile(req_file):
        req_flag = False
    env_name = _get_env_name(platform, env)

    out = c.run(f"conda create --name {env_name} --file {env_file}  {force_flag} -y")

    # install the code-template modules
    with py_env(c, env_name):

        # install pip requirements
        if req_flag:
            c.run(f"pip install -r {req_file} --no-deps")

        # install the current package
        c.run(f"pip install -e {HERE}")

_create_task_collection(
    "dev",
    setup_env,
    setup_env_pyspark,
    format_code,
    refresh_version,
    run_notebook,
    setup_addon,
    setup_info,
    _build_docker_image,
    setup_ci_env
)


# -------------
# Test/QC tasks
# --------------
@task(name="qc")
def run_qc_test(c, platform=PLATFORM, env=DEV_ENV, fail=False):
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        # This runs flake8, flake8-black, flake8-isort, flake8-bandit,
        # flake8-docstring
        c.run(f"python -m flake8 {SOURCE_FOLDER}", warn=(not fail))


@task(name="unittest")
def run_unit_tests(c, platform=PLATFORM, env=DEV_ENV, markers=None):
    env_name = _get_env_name(platform, env)
    markers = "" if markers is None else f"-m {markers}"
    with py_env(c, env_name):
        # FIXME: Add others, flake9-black, etc
        c.run(f"pytest -v {TESTS_FOLDER} {markers}")


@task(name="vuln")
def run_vulnerability_test(c, platform=PLATFORM, env=DEV_ENV):
    env_name = _get_env_name(platform, env)
    # FIXME: platform agnostic solution: get the output from conda and then munge in python
    with py_env(c, env_name):
        c.run(
            f'conda list | tail -n +4 | tr -s " " " " '
            '| cut -f 1,2 -d " " | sed "s/\ /==/g" '
            "| safety check --stdin",
        )


@task(name="all")
def run_all_tests(c):
    run_qc_test(c)
    run_vulnerability_test(c)
    run_unit_tests(c)
    validate_env(c)


@task(name="val-env")
def validate_env(c, platform=PLATFORM, env=DEV_ENV):
    env_name = _get_env_name(platform, env)

    env_files = [
        op.join(CONDA_ENV_FOLDER, env_file)
        for env_file in os.listdir(CONDA_ENV_FOLDER)
        if (PLATFORM in env_file) & (".yml" in env_file)
    ]
    expected_list = ["ta-lib==1.0.0"]

    for env_file in env_files:
        with open(env_file) as fp:
            env_cfg = yaml.safe_load(fp)

        for i in env_cfg["dependencies"]:
            if type(i) is not dict:
                expected_list.append(i)
            else:
                expected_list = expected_list + i["pip"]

    def clean_package_name(s):
        if "git+" in s:
            s = s.split("/")[-1].split(".git")
            s = s[0] + "==" + s[-1].split("v")[-1]
        return s.replace("_", "-").lower()

    expected_list = [
        clean_package_name(i)
        for i in expected_list
        if not ("nodejs" in i or "mlflow" in i or "tigerml" in i)
    ]

    with py_env(c, env_name):
        import pkg_resources

        installed_packages = pkg_resources.working_set
        installed_packages_list = sorted(
            ["%s==%s" % (i.key, i.version) for i in installed_packages]
        )

    package_diff = list(set(expected_list) - set(installed_packages_list))
    if len(package_diff) > 0:
        print(f"Following packages are missing or have wrong versions {package_diff}")
    else:
        print("You are all good!")


_create_task_collection(
    "test",
    run_qc_test,
    run_vulnerability_test,
    run_unit_tests,
    run_all_tests,
    validate_env,
)


# -----------
# Build tasks
# -----------
@task(name="docs")
def build_docs(c, platform=PLATFORM, env=DEV_ENV, regen_api=True, update_credits=False):
    env_name = _get_env_name(platform, env)
    with py_env(c, env_name):
        if regen_api:
            code_path = op.join(HERE, "docs", "source", "_autosummary")
            if os.path.exists(code_path):
                _clean_rmtree(code_path)
            os.makedirs(code_path, exist_ok=True)

        # FIXME: Add others, flake9-black, etc
        if update_credits:
            credits_path = op.join(HERE, "docs", "source", "_credits")
            if os.path.exists(credits_path):
                _clean_rmtree(credits_path)
            os.makedirs(credits_path, exist_ok=True)
            authors_path = op.join(HERE, "docs")
            token = os.environ['GITHUB_OAUTH_TOKEN']
            c.run(
                f"python {authors_path}/generate_authors_table.py {token} {token}"
            )
        c.run("cd docs/source && sphinx-build -T -E -W --keep-going -b html -d ../build/doctrees  . ../build/html")


_create_task_collection("build", build_docs)


# -----------
# Launch stuff
# -----------
@task(name="jupyterlab")
def start_jupyterlab(
    c, platform=PLATFORM, env=DEV_ENV, ip="localhost", port=8080, token="", password=""
):
    env_name = _get_env_name(platform, env)
    # FIXME: run as a daemon and support start/stop using pidfile
    with py_env(c, env_name):
        print(f"{'--'*20} \n Running jupyterlab with {env_name} environment \n {'--'*20}")
        c.run(
            f"jupyter lab --ip {ip} --port {port} --NotebookApp.token={token} "
            f"--NotebookApp.password={password} --no-browser"
        )

@task(name="jupyterlab_pyspark")
def start_jupyterlab_pyspark(
    c, platform=PLATFORM, env=DEV_ENV, ip="localhost", port=8081, token="", password=""
):
    env_name = _get_env_name_pyspark(platform, env)
    # FIXME: run as a daemon and support start/stop using pidfile
    with py_env(c, env_name):
        print(f"{'--'*20} \n Running jupyterlab with {env_name} environment \n {'--'*20}")  # To: Delete print statement
        c.run(
            f"jupyter lab --ip {ip} --port {port} --NotebookApp.token={token} "
            f"--NotebookApp.password={password} --no-browser"
        )


@task(name="tracker-ui")
def start_tracker_ui(c, platform=PLATFORM, env=DEV_ENV, port=8082):
    env_name = _get_env_name(platform, env)
    # FIXME: run as a daemon and support start/stop using pidfile
    # NOTE: using sqlite with more than 1 worker will cause issues
    # if scalability is needed, use a postgresql server
    with py_env(c, env_name):
        c.run(
            f"mlflow server --port {port} --default-artifact-root {HERE}/mlruns "
            f"--backend-store-uri sqlite:///{HERE}/mlruns/mlflow.db "
            f"--workers 1"
        )


@task(name="docs")
def start_docs_server(c, ip="127.0.0.1", port=8081):
    # FIXME: run as a daemon and support start/stop using pidfile
    c.run(
        f"python -m http.server --bind 127.0.0.1 " f"--directory docs/build/html {port}"
    )


@task(name="ipython")
def start_ipython_shell(c, platform=PLATFORM, env=DEV_ENV):
    env_name = _get_env_name(platform, env)
    # FIXME: run as a daemon and support start/stop using pidfile
    startup_script = op.join(HERE, "deploy", "ipython", "default_startup.py")
    with py_env(c, env_name):
        c.run(f"ipython -i {startup_script}")


_create_task_collection(
    "launch",
    start_jupyterlab,
    start_jupyterlab_pyspark,
    start_tracker_ui,
    start_docs_server,
    start_ipython_shell,
)



# --------------
# Root namespace
# --------------
# override any configuration for the tasks here
# FIXME: refactor defaults (constants) and set them as config after
# auto-detecting them
ns = _create_root_task_collection()
config = dict(pty=True, echo=True)

if OS == "windows":
    config["pty"] = False

ns.configure(config)


