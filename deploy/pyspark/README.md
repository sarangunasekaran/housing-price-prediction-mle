# Classification Pyspark Template

This template provides an overview of getting started with pyspark classification and regression notebooks with Databricks clusters using [`databricks-connect`](https://docs.databricks.com/dev-tools/databricks-connect.html).

Tip: If you don't have a markdown viewer like atom, you can render this on chrome by following [this link](https://imagecomputing.net/damien.rohmer/teaching/general/markdown_viewer/index.html).
# Getting started 

*  Ensure [invoke](http://www.pyinvoke.org/index.html) tool and pyyaml are installed in your `base` `conda` environment. If not, run

```
(base):~$ pip install invoke
(base):~$ pip install pyyaml
```

*  Set up the environment `ta-lib-dev-pyspark` as below:
```
(base):~/code-templates$ inv dev.setup-env-pyspark
```

The above command should create a conda python environment named `ta-lib-pyspark-dev`.

* Activate the environment:
```
(base):~$ conda activate ta-lib-dev-pyspark
```
# Using customer docker containers in Databricks

* Install docker: https://docs.docker.com/docker-for-windows/install/

* Docker Hub:

    * Create a Docker account: https://hub.docker.com
    * Create a private/public repository (Docker Hub provides the default flexibility to create one private repository)

* Amazon ECR:
    * [Create](https://docs.aws.amazon.com/AmazonECR/latest/userguide/repository-create.html) your ECR repository 
    * [Create your access keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html) by expanding the Access Keys (access key ID and secret access key) section under: 
    
            User -> My Security Credentials 
        
    * Also ensure you have AWS CLI installed: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-windows.html

## Docker container setup
* Build the docker image:

```
(ta-lib-dev-pyspark):~$ cd deploy/pyspark/
(ta-lib-dev-pyspark):~$ docker build -t <repository_name>:<tag> .    # example : docker build -t helloapp:v1 .
```

* Verify the installation by listing the installed images

```
(ta-lib-dev-pyspark):~$ docker images
```

* Push the docker image to your repository:
    * DockerHub
    
    ```
    (ta-lib-dev-pyspark):~$ docker login --username <username>            # enter the password as prompted
    (ta-lib-dev-pyspark):~$ docker tag <image> <username>/<repo-name>     # example : docker tag 88ddeec9d217 amritbhaskar/ta_lib
    (ta-lib-dev-pyspark):~$ docker push <username>/<repo-name>            # example : docker push amritbhaskar/ta_lib
    ```

    * ECR:
    ```
    (ta-lib-dev-pyspark):~$ aws configure                                                                      # enter the AWS Access Key ID & AWS Secret Access Key for the repository as prompted
    (ta-lib-dev-pyspark):~$ aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.region.amazonaws.com/<my-repo>
    (ta-lib-dev-pyspark):~$ docker tag <image> <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<repo-name>     # example: docker tag 88ddeec9d217 aws_account_id.dkr.ecr.us-east-1.amazonaws.com/ta_lib
    (ta-lib-dev-pyspark):~$ docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<repo-name>
    ```




## Using your docker container 

Refer to the [Databricks Container Services documentation](https://docs.databricks.com/clusters/custom-containers.html) for additional details.

* Login to the databricks workspace 

* Enable container services in the Databricks account as follows: 

        Admin Console -> Advanced -> Container Services

* [Launch](https://docs.databricks.com/clusters/create.html) your cluster using the UI

* Select the **Use your own Docker container** option
 
* Enter your custom Docker image in the Docker Image URL field and give the docker repository details.

    Example - 

        DockerHub:
            <dockerid>/<repository>:<tag>
            amritbhaskar/ta_lib:latest
            
        Elastic Container Registry:
            <aws-account-id>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>

    * Select the relevant authentication type 

        * Default - When using a public Docker repository or when using Default mode with an IAM role in ECR.
        * Username and password - Provide username and password  in case you are using a DockerHub private repository 

    * Confirm the changes and restart the cluster.

### Adding packages

* For installation of any package while working on the notebook:
```
! pip install {packagename}
```

* To add new packages to docker container.

    * Add the package to the `env.yml` file.
    * Rerun the command ```docker build -t [Repository-name]:[tag]```  to rebuild the repository.
    * Push the image to the DockerHub/ECR.
    
## Databricks Connect setup

You can use Databricks Connect to run Spark jobs from your notebook apps or IDEs. The Databricks Connect package is already installed as part ofg the `ta-lib-dev-pyspark` environment.

* Login to the Databricks workspace 

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
    
* Check the version of Python running (should be 3.7 for the above cluster)

* Check that you have [Java SE Runtime Version 8](https://www.oracle.com/java/technologies/javase-jre8-downloads.html) installed and added in your system environment variables
    
* Run the cmd: ```databricks-connect configure``` and enter the saved information when prompted

* Test your setup with ```databricks-connect test```

* Launch a Jupyter notebook within this environment to start using Spark:
```
(ta-lib-dev-pyspark):~/code_templates$ inv launch.jupyterlab-pyspark
```

After running the command, type localhost:8081 in your browser to see the launched JupyterLab. 

Note: Ensure your Jupyter Lab is running from the `ta-lib-dev-pyspark` environment. In case the environment is not visible in your list of available Jupyter Lab kernels, run: 
```
(ta-lib-dev-pyspark):~/code_templates$ conda install -c anaconda ipykernel
( ta-lib-dev-pyspark):~/code_templates$python -m ipykernel install --user --name ta-lib-dev-pyspark
```
    
