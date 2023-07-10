run these commands from command promt(CMD)
mkdir MLDevops
cd MLDevops
code .

bash terminal
conda create -p env python -y
source activate ./env

cmd terminal
conda create -p env python=3.8 -y
conda activate <absolute path of env>

shortcut
conda create -p env python=3.8 -y && source activate ./env


- create requirements.txt file

In requirements.txt file type
mlflow

pip install -r requirements.txt

pip list

- create first_trial.py file

write the code in python file

python first_trial.py

mlflow ui

***********************************************************
# ML Flow
MLflow is an open-source platform for managing the machine learning lifecycle. It provides several methods and functionalities that enable users to track, reproduce, and deploy machine learning models. Here are some key methods and functionalities offered by MLflow:

1. Tracking Experiments:
   - `mlflow.start_run()`: Starts a new MLflow run to track an experiment.
   - `mlflow.log_param()`: Logs a parameter value for the current run.
   - `mlflow.log_metric()`: Logs a metric value for the current run.
   - `mlflow.log_artifact()`: Logs a file or directory as an artifact for the current run.

2. Tracking Models:
   - `mlflow.log_model()`: Logs a model as an artifact, including the model file and its metadata.
   - `mlflow.load_model()`: Loads a saved MLflow model.
   - `mlflow.register_model()`: Registers a logged model in the model registry.

3. Running Projects:
   - `mlflow.run()`: Runs an MLflow project, which includes packaging the code, dependencies, and parameters as a reproducible run.
   - `mlflow.run_local()`: Runs an MLflow project locally without using a remote execution environment.

4. Model Deployment:
   - `mlflow.deployments.create()`: Deploys a registered MLflow model to a target deployment environment, such as a cloud-based serving platform or an edge device.
   - `mlflow.deployments.get()`: Retrieves information about a deployed model.

5. Model Registry:
   - `mlflow.register_model()`: Registers a model in the model registry.
   - `mlflow.model_versions.create()`: Creates a new version of a registered model.
   - `mlflow.model_versions.transition()`: Transitions the state of a model version in the model registry, such as from "Staging" to "Production."

These are some of the common methods provided by MLflow. MLflow also supports integration with popular machine learning libraries and frameworks, allowing users to track and manage experiments across different platforms and tools.



Certainly! Here are the MLflow methods categorized based on their functionalities:

**Experiment Tracking:**
- `mlflow.start_run()`
- `mlflow.log_param()`
- `mlflow.log_metric()`
- `mlflow.log_artifact()`

**Model Tracking:**
- `mlflow.log_model()`
- `mlflow.load_model()`
- `mlflow.register_model()`

**Running Projects:**
- `mlflow.run()`
- `mlflow.run_local()`

**Model Deployment:**
- `mlflow.deployments.create()`
- `mlflow.deployments.get()`

**Model Registry:**
- `mlflow.register_model()`
- `mlflow.model_versions.create()`
- `mlflow.model_versions.transition()`

These categories help organize the MLflow methods based on their specific purposes within the machine learning lifecycle. Experiment tracking methods are used to log and track experiments, including parameters, metrics, and artifacts. Model tracking methods are used to log and load models, as well as register them in the model registry. Running project methods are used to execute MLflow projects either locally or in a remote environment. Model deployment methods facilitate the deployment of registered models to different environments. Lastly, model registry methods are used to manage the lifecycle and versions of models within the MLflow model registry.
****************************************************************************************************************************************

## **DVC**
******************************************************************************************************************************************

DVC (Data Version Control) is an open-source version control system for machine learning projects. It allows you to track and manage both code and data in a reproducible and collaborative manner. Here are some commonly used DVC commands:

1. Initializing DVC:
   - `dvc init`: Initializes DVC in the current directory.
   - `dvc remote add <remote-name> <remote-url>`: Adds a remote storage location for DVC data and models.

2. Tracking Data:
   - `dvc add <file>`: Adds a data file to DVC tracking.
   - `dvc commit`: Commits changes to the DVC repository, tracking the data file(s).

3. Tracking Code:
   - `dvc run -n <stage-name> -d <dependencies> -o <outputs> <command>`: Defines a DVC pipeline stage with input dependencies, output files, and the command to execute.

4. Running DVC Pipelines:
   - `dvc repro <stage-name>`: Runs a specific DVC pipeline stage and its dependencies.
   - `dvc pipeline show <stage-name>`: Displays the DAG (Directed Acyclic Graph) of the DVC pipeline.

5. Managing Remotes:
   - `dvc remote add <remote-name> <remote-url>`: Adds a remote storage location for DVC data and models.
   - `dvc remote modify <remote-name> <key> <value>`: Modifies a configuration value for a remote.
   - `dvc remote remove <remote-name>`: Removes a remote storage location.

6. Managing Data and Models:
   - `dvc push`: Pushes data and model files to the remote storage.
   - `dvc pull`: Pulls data and model files from the remote storage.
   - `dvc import <remote-url> <path>`: Imports a file from a remote URL to the DVC project.

7. Versioning and Branching:
   - `dvc branch <branch-name>`: Creates a new branch in the DVC repository.
   - `dvc checkout <branch-name>`: Switches to a different branch in the DVC repository.
   - `dvc tag <tag-name>`: Tags a specific commit in the DVC repository.

These are some of the commonly used DVC commands that help in managing data, code, and models in machine learning projects. DVC integrates well with Git, allowing you to combine the power of version control for both data and code in a unified manner.