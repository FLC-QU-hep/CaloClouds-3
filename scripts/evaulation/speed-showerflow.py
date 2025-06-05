import os
import sys
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from pointcloud.config_varients import caloclouds_3_simple_shower
from pointcloud.models import shower_flow
from pointcloud.utils import showerflow_utils
from pointcloud.data.conditioning import get_cond_dim, read_raw_regaxes_withcond
from pointcloud.data.read_write import get_n_events
from pointcloud.utils import precision


config = caloclouds_3_simple_shower.Configs()
if torch.cuda.is_available():
    config.device = "cuda"
else:
    config.device = "cpu"
if os.path.exists(os.path.dirname(config.dataset_path)):
    print(f"Found dataset at {config.dataset_path}")

# Get the generator models we will be comparing with, and check their data has been generated.
existing_list = []
for fnorm in [True, False]:
    config.shower_flow_fixed_input_norms = fnorm
    for tbase in [True, False]:
        config.shower_flow_train_base = tbase
        found_here = showerflow_utils.existing_models(config)
        existing_list.append(showerflow_utils.existing_models(config))
existing_models = {
    key: sum([model[key] for model in existing_list], []) for key in existing_list[0]
}
existing_models["config"] = []

for i, path in enumerate(existing_models["paths"]):
    model_config = showerflow_utils.construct_config(config, existing_models, i)
    existing_models["config"].append(model_config)

print(f"Total models found: {len(existing_models['names'])}")

# table = [['', 'versions', 'num_blocks', 'cut_inputs', 'fixed_input_norms', 'train_base', 'best_loss']]
table = [["", "versions", "num_blocks", "cut_inputs", "fixed_input_norms", "best_loss"]]
col_width = 15
n_models = len(existing_models["paths"])
for i in range(n_models):
    table.append([str(i)] + [str(existing_models[k][i]) for k in table[0][1:]])
for row in table:
    line = ""
    for col in row:
        col = col[:col_width]
        col = " " * (col_width - len(col)) + col
        col += ","
        line += col
    print(line)


def get_loaded_flow(i):
    constructor = shower_flow.versions_dict[existing_models["versions"][i]]
    config_here = existing_models["config"][i]
    config_here.device = config.device
    num_inputs = np.sum(showerflow_utils.get_input_mask(config_here))
    num_cond = get_cond_dim(config_here, "showerflow")
    model, flow_dist, transforms = constructor(
        config_here.shower_flow_num_blocks, num_inputs, num_cond, config_here.device
    )
    return flow_dist, config_here


n_events_total = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))


def get_inputs(n, config):
    indices = np.sort(np.random.choice(n_events_total - 1, n, replace=False))
    cond, _ = read_raw_regaxes_withcond(config, indices, for_model="showerflow")
    prec = precision.get("showerflow", config)
    cond = cond.to(config.device, dtype=prec)
    return cond


def time_model(model, cfgs, items_per_input, n_batches=10, per_batch=100):
    batch_total = items_per_input * per_batch
    torch.cuda.synchronize()
    measured_times = np.zeros(n_batches)
    sample_size = torch.Size([items_per_input])
    for b in range(n_batches):
        all_inputs = get_inputs(batch_total, cfgs)
        all_inputs = [
            all_inputs[i : i + items_per_input]
            for i in range(0, batch_total, items_per_input)
        ]
        t0 = time.time()
        for inputs in all_inputs:
            model.condition(inputs).sample(sample_size)
        torch.cuda.synchronize()
        t1 = time.time()
        measured_times[b] = (t1 - t0) / batch_total
    mean = np.mean(measured_times)
    std_2 = np.std(measured_times) ** 2
    std_per_item = np.sqrt(std_2 / batch_total)
    return mean, std_per_item


for_input_sizes = [2**i for i in range(1, 10)]

total_options = len(existing_models["paths"]) * len(for_input_sizes)
# run
option_num = int(sys.argv[1])

i = option_num // len(existing_models["paths"])
j = option_num % len(existing_models["paths"])

path = existing_models["paths"][j]
input_size = for_input_sizes[i]
print(f"Timing model {j} with input size {input_size}")
try:
    model, cfg = get_loaded_flow(j)
    mean, std = time_model(model, cfg, input_size)
    exception = "None"
except Exception as e:
    print(e)
    mean, std = np.nan, np.nan
    exception = str(e)
print()
print("Saving results")
out_path = f"/data/dust/user/dayhallh/point-cloud-diffusion-data/showerFlow/sim-E1261AT600AP180-180/timing/{config.device}_{i}_{j}.npz"
np.savez(
    out_path,
    mean_time=mean,
    std_time=std,
    input_size=input_size,
    path=path,
    device=config.device,
    exception=exception,
)
print(f"Results saved to {out_path}")
