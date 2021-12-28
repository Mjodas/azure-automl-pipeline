import argparse
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

parser = argparse.ArgumentParser()

parser.add_argument("--cluster-name", type=str, dest='name',
                    help="The name of the new compute cluster.")

parser.add_argument("--max_nodes", type=int, dest="max_nodes",
                    help="The maximum number of working nodes in the cluster", default=2)

args = parser.parse_args()

cluster_name = args.name
max_nodes = args.max_nodes

ws = Workspace.from_config()

try:
    # Check for existing compute target
    cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Cluster already exist')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='STANDARD_DS11_V2', max_nodes=max_nodes)
        cluster = ComputeTarget.create(
            ws, cluster_name, compute_config)
        cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)
