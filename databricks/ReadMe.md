# Code-Templates on Databricks

This template provides an overview of getting started with pyspark classification and regression notebooks with Databricks and using DE with the help of [`databricks-connect`](https://docs.databricks.com/dev-tools/databricks-connect.html).

Tip: If you don't have a markdown viewer like atom, you can render this on chrome by following [this link](https://imagecomputing.net/damien.rohmer/teaching/general/markdown_viewer/index.html).

## Creating Databricks Cluster

You can use Databricks Connect to run Spark jobs from your notebook apps or IDEs. The Databricks Connect package will be installed on your local VM as part of the local VM setup. There are few setting to be enable for supporting databricks-connect on Databricks cluster. We will also get some information required for local setting. Please use following steps to create such a Databricks cluster.

* Login to the Databricks workspace

* Select Databricks Runtime version as 8.1 ML (without GPU).

* Please enable logging in advanced settings of cluster.

        Advanced Options -> Logging -> Destination (From drop down select DBFS)

* Navigate to your required cluster, and ensure that your Spark Config is set to ```spark.databricks.service.server.enabled true``` under

        Advanced Options -> Spark -> Spark Config

* In your account, generate a new token under

        User Settings -> Access Tokens -> Generate New Token
    Copy this token and save it for further use.

* Collect the following [details](https://docs.databricks.com/workspace/workspace-details.html) regarding your workspace:
    * Databricks host (e.g. ```https://dbc-0b606se5-478a.cloud.databricks.com/?o=4234485324337931```)
    * Databricks token (saved from the previous step)
    * Cluster ID (e.g.0815-092137-flare283)
    * Org ID
    * Port (15001)


# Local VM setup

*  Ensure [invoke](http://www.pyinvoke.org/index.html) tool and pyyaml are installed in your `base` `conda` environment. If not, run

```
(base):~$ pip install invoke
(base):~$ pip install pyyaml
```

* Download Code-Templates `Classification` or `Regressison` archives from [Knowloedge Tiger](https://sites.google.com/a/tigeranalytics.com/knowledge-tiger/IP-assets/Code-Templates)

* Rename `classification-py`/`regression-py` to `code-templates`.

* Folder structure should look like `code-templates/`, and inside this folder `databricks, data, deploy, src` etc. folders should be available.

* Edit config file `local_vm_setup_config.yml` by following the direction provided in the same file.

* Run following command to setup local VM.

```
(base):~/code-templates/databricks/vm_setup$ sh local_vm_setup.sh
```

* The above command should perform following tasks:
    * Read config file  local_vm_setup_config.yml
    * Create a conda python environment named `ta-lib-pyspark-dev`.
    * Setup databricks-cli
    * Setup databricks-connect
    * Deploy data_config.json using databricks stack CLI.
    * Deploy full_stack_config.json using databricks stack CLI.


# Local VM validations
* Check the version of Python running (should be 3.7 for the above cluster)

* Check that you have [Java SE Runtime Version 8](https://www.oracle.com/java/technologies/javase-jre8-downloads.html) installed and added in your system environment variables

* Test your setup with (After activating the environment)
```
(ta-lib-pyspark-dev):~/code-templates$ databricks-connect test
```

* Launch a Jupyter notebook within this environment to start using Spark:
```
(ta-lib-pyspark-dev):~/code-templates$ inv launch.jupyterlab-pyspark
```

After running the command, type localhost:8081 in your browser to see the launched JupyterLab.

Note: Ensure your Jupyter Lab is running from the `ta-lib-pyspark-dev` environment. In case the environment is not visible in your list of available Jupyter Lab kernels, run:
```
(ta-lib-pyspark-dev):~/code-templates$ conda install -c anaconda ipykernel
(ta-lib-pyspark-dev):~/code-templates$ python -m ipykernel install --user --name ta-lib-pyspark-dev
```

# Stack-CLI Setup
Databricks CLI is already installed and setup as part of the `ta-lib-pyspark-dev` environment.

There are two services offered by databricks, where entities can be pushed

1. dbfs: It is a file storage and all entities can be pushed/synced to this service.

2. Workspace: Only codes can be imported (for us .py or .ipynb file) to this. Rest if tried will be ignored. All supported files will be imported as Databricks notebooks.

Both the services allow syncing a directory (recursively) or a single file.

* All the deployed stacks can be identified using files present in `~/code_templates/databricks/vm_setup/stack_conf/<filename>.deployed.json`

* To sync recently edited codes or data in local to Databricks use follwing commands.
```
(ta-lib-pyspark-dev):~/code-templates/databricks/vm_setup$ databricks stack deploy -o ./stack_conf/full_stack_config.json
(ta-lib-pyspark-dev):~/code-templates/databricks/vm_setup$ databricks stack deploy -o ./stack_conf/data_config.json
```

* Additional information provided to deploy single files or directory to DBFS or single notebook or directory of notebooks to Workspace service to Databricks. Please find the same in Databricks/stack_conf/stack_templates.json. Edit this file for user_name (Databricks), stack_id, source_path and target_path.

* Two type of issues can be faced while syncing these files using Databricks stack deploy.
    * While syncing notebooks to `Workspace`, if file size is more than 10MB. In such cases removing outputs of from .ipynb notebooks might help in reducing the file size.

    * For any other type of issue, delete corresponding `<filename>.deployed.json` file and run the command again.

* To download the files from dbfs:/FileStore use the following command.
```
databricks fs cp dbfs:/FileStore/<filpath_with_name> <target_path_in_local_without_filename/.>
```


# Databricks connect
To run DE with Databricks, Databricks-Connect will be used. Databricks-connect is already installed in `ta-lib-pysark-dev`. To use this service sample notebooks has been provided in `~/code-templates/databricks/notebooks/03_reference_notebooks/04_DE.ipynb`. Please notice following things.

* Following two lines should be present to create a spark session using Databricks cluster information provided in `~/code-templates/databricks/vm_setup/local_vm_setup_config.yml`

```
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
```

* Wherever spark session will be used the handle will be transferred to Databricks. Same can be tracked using Databricks-spark-UI


# Databricks Environment Setup
The cluster created on Databricks has everything except the correct environment to run Code-Templates or TigerML codes. To do that please use following steps in Databricks.

* Start the cluster (Lets call the cluster as `test_cluster`).

* Locate `env_setup.ipynb` notebook in Databricks Workspace. It should already be present in `<user_name>/code-templates/01_setup`. Attach this notebook to `test_cluster`.

* Run first 3 cells from this notebook.

* Added init script to `test_cluster`.

    * Advanced Options -> Init Scripts
    * Add path `dbfs:/FileStore/Git/init.sh` here.

* Restart the cluster.


# Data
To sync data to Databricks & local Databricks Stack CLI can be used. Any data present in `~/code_templates/data` directory will be synced (uploaded) to `dbfs:/FileStore/Git/data` in the process of setting up the local VM. To do this stack config present at `~/code_templates/databricks/vm_setup/stack_conf/data_config.json` was used. Use the same config to resync any changes to the same. Following command can be used to do that.

```
(ta-lib-pyspark-dev):~/code_templates/databricks/vm_setup$ databricks stack deploy -o ./stack_conf/data_config.json
```

# Delta tables
Quick example to convert data to delta tables has been provided in `02_delta/delta_sample_data.ipynb`, same has been synced to Databricks at the following location `code-templates/02_delta` in `Workspace` notebook location.

# MLFlow
To start tracking models in MLFlow service, please create an experiment following the details from [`here`](https://docs.databricks.com/applications/mlflow/tracking.html#create-workspace-experiment).

An example of using MLFlow for tracking has been provided in `03_reference_notebooks/01_automl.ipynb`

# AutoML
To use AutoML, install packages tpot, deap by adding the statement `pip install -r /dbfs/FileStore/regression-py/deploy/conda_envs/requirements-databricks-linux-cpu-64-dev.txt` in init script.