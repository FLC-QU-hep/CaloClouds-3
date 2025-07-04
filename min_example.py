import numpy as np
import torch

# Simple shower configs for
# shower_flow_fixed_input_norms = True
# shower_flow_inputs = [ "cog_x", "cog_y", "clusters_per_layer", "energy_per_layer"]
# and
# shower_flow_num_blocks = 2
from pointcloud.config_varients import caloclouds_3_simple_shower
from pointcloud.utils import showerflow_utils
from pointcloud.models import shower_flow

configs = caloclouds_3_simple_shower.Configs()
# only if you are on a machine without CUDA
configs.device = "cpu"
# point to your events, because then functions that check the metadata check the right metadata
configs.dataset_path = "/beegfs/desy/user/akorol/data/AngularShowers_RegularDetector/hdf5_for_CC/sim-E1261AT600AP180-180_file_{}slcio.hdf5"
configs.n_dataset_files = 88
# This lets you algorithmically get the input and conditioning dimensions. Then if we change them (taking out cog_z for example) you don't need to change other code.
cond_mask = showerflow_utils.get_cond_mask(configs)
input_mask = showerflow_utils.get_input_mask(configs)
# The version is the operations inside a block, you can also see it in the saved model name
version = "alt1"
constructor = shower_flow.versions_dict[version]
# Get ourselves an untrained model with the right architecture
model, dist, transforms = constructor(
    num_blocks=2,
    num_inputs=np.sum(input_mask),
    num_cond_inputs=np.sum(cond_mask),
    device=configs.device,
)
# load the weights
model_path = (
    "/beegfs/desy/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/"
    "ShowerFlow_alt1_nb2_"  # block version and number of blocks
    "inputs8070450532247928831_"  # input mask representaiton
    "fnorms_best.pth"  # fnorms = Not normalising by layer with highest value, etc.
)
loaded = torch.load(model_path, map_location=configs.device, weights_only=False)
model.load_state_dict(loaded["model"])
# Now make up some conditioning values
context = torch.tensor(
    [[0.572, 0.333, -0.4859, 0.808], [0.00934, -0.522, -0.279, 0.8059]]
)
conditioned = dist.condition(context)
print(conditioned.sample(torch.Size([len(context)])))

