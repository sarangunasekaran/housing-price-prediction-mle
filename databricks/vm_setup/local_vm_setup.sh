#!/bin/sh

# include parse_yaml function
. ./parse_yaml.sh

# read yaml file
echo "Reading Config"
eval $(parse_yaml local_vm_setup_config.yml "")

# # Cloning TigerML
# cd $git_repo_path/src # Moving the directory where local_vm_setup_config.yml is present
echo "Setting up Local VM"

# Setup conda env & activate the same
cd ..
inv -l
inv dev.setup-env-pyspark
eval "$(conda shell.bash hook)"
conda activate ta-lib-pyspark-dev
# cd databricks

# Databricks Connect setup
echo "$databricks_host/?o=$org_id
$databricks_token
$cluster_id
$org_id
$port" | databricks-connect configure

# Databricks Stack CLI Setup
echo "$databricks_host
$databricks_token" | databricks configure --token

## Deploy Databricks Stack
# Data
databricks stack deploy -o vm_setup/stack_conf/data_config.json

# Code
databricks stack deploy -o vm_setup/stack_conf/full_stack_config.json
