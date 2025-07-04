from pointcloud.config_varients import caloclouds_3_simple_shower
from pointcloud.models import load
from pointcloud.data.conditioning import read_raw_regaxes_withcond
import torch
import time
import numpy as np

configs = caloclouds_3_simple_shower.Configs()
model_class = load.get_model_class(configs)
model_path = "/data/dust/user/dayhallh/point-cloud-diffusion-logs/sim-E1261AT600AP180-180/Anatoliis_cc_2.pt"
state_dict = torch.load(model_path)['state_dict']
n_cond_samples = 1_000
cond, _ = read_raw_regaxes_withcond(configs, total_size=n_cond_samples, for_model='diffusion')

def next_batch_size():
    i = 0
    j = 0
    while True:
        yield (j, 2**i)
        i += 1
        if i > 14:
            j += 1
            i = 0
generator = next_batch_size()

devices = {'cpu':'cpu', 'gpu':'cuda'}
precision = {'float16': torch.float16, 'float32':torch.float32, 'float64':torch.float64}
repeats = 10

directory = "/data/dust/user/dayhallh/point-cloud-diffusion-logs/sim-E1261AT600AP180-180/timing/"

while True:
    file_increment, batch_size = next(generator)
    file_name = f"speeds_batch_{file_increment}_size_{batch_size}.npz"
    file_name = directory + file_name
    print(file_name)
    data = {}
    for device_name, device in devices.items():
        print(device_name, end="~")
        for precision_name, dtype in precision.items():
            torch.set_default_dtype(dtype)
            print(precision_name, end=" ")
            configs.device = device
            configs.diffusion_precision = dtype
            model = model_class(args=configs)
            state_dict = {k: v.to(device, dtype=dtype) for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.to(device, dtype=dtype)
            cond_here = cond[np.random.randint(0, n_cond_samples, repeats)]
            cond_here = cond_here.to(device, dtype=dtype)
            # dry run
            model.sample(cond_here[[0]], batch_size, configs)
            times = [time.time()]
            for r in range(repeats):
                model.sample(cond_here[[r]], batch_size, configs)
                times.append(time.time())
            data[f"{precision_name}_{device_name}"] = times
    np.savez(file_name, **data)

model.to('cuda')
cond = cond.to('cuda', dtype=torch.float32)
