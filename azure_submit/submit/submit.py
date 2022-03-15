from azureml.core import Workspace, ComputeTarget, Environment, Experiment, Datastore, ScriptRunConfig
from azureml.core.runconfig import MpiConfiguration, RunConfiguration, DEFAULT_GPU_IMAGE, DockerConfiguration
import argparse
import sys
import os
import json


def parse_args():
    parser = argparse.ArgumentParser("Azureml Script")
    parser.add_argument("--name",type=str,help="experiment name",default="dali_test")
    parser.add_argument("--cluster",type=str,help="cluster name",default="wsussc")
    parser.add_argument("--mount_folder",type=str,help="mount blob folder",default="./")
    parser.add_argument("--command",type=str,help="command",default="")
    parser.add_argument("--node_count",type=int,help="node_count",default=2)
    return parser

args = parse_args().parse_args()
git_key = os.environ.get('GITKEY')

clusster_cfg = json.load(open('../config/videopretrain.json'))

ws = Workspace(**clusster_cfg[args.cluster])
myenv = Environment.get(workspace=ws, name="lfvideo") # ???
myenv.environment_variables['GITKEY'] = git_key

datastore_name = "sizheng"  # ???
ds = Datastore(workspace=ws, name=datastore_name)
data_ref = ds.path(args.mount_folder).as_mount()
distr_config = MpiConfiguration(process_count_per_node=8, node_count=args.node_count)

run_config = ScriptRunConfig(
  source_directory= './',
  script='begin.py',
  arguments=['--blob_mount_dir', str(data_ref), '--command', args.command],
  compute_target=args.cluster,
  environment=myenv,
  distributed_job_config=distr_config,
  docker_runtime_config=DockerConfiguration(use_docker=True, shm_size='384g')
)

run_config.run_config.data_references = {data_ref.data_reference_name: data_ref.to_config()}
# submit the run configuration to start the job
run = Experiment(ws, args.name).submit(run_config)
aml_url = run.get_portal_url()
print(aml_url)