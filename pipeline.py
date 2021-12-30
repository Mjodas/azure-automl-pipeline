import os

from azureml.core import Environment, Experiment, RunConfiguration, Workspace
from azureml.core.compute import ComputeTarget
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.widgets import RunDetails

ws = Workspace.from_config()

cluster_name = 'mjodas1'
environment_name = 'AzureML-sklearn-1.0-ubuntu20.04-py38-cpu'


# Get cluster
try:
    cluster = ComputeTarget(ws, name=cluster_name)
except Exception as e:
    print(e)

# Get prebuilt environment
environment = Environment.get(ws, name=environment_name)

print(type(environment))
print(environment)

# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()
pipeline_run_config.target = cluster
pipeline_run_config.environment = environment

# Get source data
house_prices = ws.datasets.get('tabular-house-prices')

prepped_data = OutputFileDatasetConfig('prepped_data')

# Crete pipeline steps

prep_step = PythonScriptStep(name="Prepare Data",
                             source_directory='steps',
                             script_name="prepare_data.py",
                             arguments=['--input-data', house_prices.as_named_input('raw_data'),
                                        '--prepared-data', prepped_data],
                             compute_target=cluster,
                             runconfig=pipeline_run_config,
                             allow_reuse=True)


train_step = PythonScriptStep(name="Train and Register Model",
                              source_directory='steps',
                              script_name="train.py",
                              arguments=['--training-data',
                                         prepped_data.as_input()],
                              compute_target=cluster,
                              runconfig=pipeline_run_config,
                              allow_reuse=True)

# Create pipeline
steps = [prep_step, train_step]
pipeline = Pipeline(ws, steps=steps)

# Create and submit experiment
experiment = Experiment(ws, name='house-prices-normalized-gbr')
run = experiment.submit(pipeline, regenerate_outputs=True)
run.wait_for_completion()
