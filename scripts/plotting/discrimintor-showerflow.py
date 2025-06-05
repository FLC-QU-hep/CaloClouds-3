# # Discriminator
#
# Run a single discrminator network to compare different renditions of showerflow.
#
# ## data prep
#
# The starting point is loading the config for this dataset.
import sys
import os
import torch
from pointcloud.config_varients import (
    caloclouds_3_simple_shower,
    caloclouds_3,
    wish,
    default,
)
from pointcloud.evaluation import discriminator
from pointcloud.utils import showerflow_utils
from pointcloud.utils.plotting import nice_hex

from sklearn import metrics
from matplotlib import pyplot as plt

default_config = default.Configs()
config = caloclouds_3_simple_shower.Configs()
if torch.cuda.is_available():
    config.device = "cuda"
else:
    config.device = "cpu"
if os.path.exists(os.path.dirname(config.dataset_path)):
    print(f"Found dataset at {config.dataset_path}")

# Then we check that the ground truth features exist for this dataset.
g4_data_folder = discriminator.locate_g4_data(config)
print(g4_data_folder)

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

for i, name in enumerate(existing_models["names"]):
    model_config = showerflow_utils.construct_config(config, existing_models, i)
    existing_models["config"].append(model_config)

print(f"Total models found: {len(existing_models['names'])}")



# ## data plotting
#
# We can go ahead and create a training object now, as this will load our datasets.
# Before we actually start training, we should plot some of the data as a sanity check.


cache = {}
def gen_training(model_idx, settings="settings12"):
    if (model_idx, settings) in cache:
        return cache[(model_idx, settings)]
    model_name = existing_models["names"][model_idx]
    model_config = existing_models["config"][model_idx]
    model_path = existing_models["paths"][model_idx]
    model_data_folder = discriminator.locate_model_data(model_config, model_path)
    feature_mask = discriminator.feature_masks[settings]
    training = discriminator.Training(
        settings,
        g4_data_folder,
        model_data_folder,
        discriminator.descriminator_params[settings],
        feature_mask,
    )
    cache[(model_idx, settings)] = model_name, training
    return model_name, training


# make some auc plots
existing_models["auc"] = [None for _ in existing_models["names"]]
for start in range(0, len(existing_models["names"])+5, 5):
    fig, ax = plt.subplots()
    for plot_for in range(start, start + 5):
        try:
            name, training = gen_training(plot_for)
            training.reload()
            labels, predictions = training.predict_test()
        except Exception:
            continue

        fpr, tpr, threasholds = metrics.roc_curve(labels, predictions)
        auc = metrics.roc_auc_score(labels, predictions)
        if auc < 0.501:
            print(f"Model {plot_for} has issues")
        existing_models["auc"][plot_for] = auc
        ax.plot(fpr, tpr, label=f"{name} AUC={auc}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    save_to = os.path.join(
        training.state_dict["generator_data_folder"], f"../auc_{start}.png"
    )
    fig.savefig(save_to)
    print(f"Saved to {save_to}")
    plt.close()


for start in range(0, len(existing_models["names"]) + 5, 5):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(start, start + 5):
        try:
            name, training = gen_training(i)
            training.reload()
            training.plots(
                axes, colour=nice_hex[int(i / 5) % 5][i % 5], legend_name=name
            )
        except Exception:
            continue
    axes[0].semilogy()
    axes[0].set_xlim(0, 30)
    axes[1].semilogy()
    axes[0].legend()
    axes[1].legend()
    save_to = os.path.join(
        training.state_dict["generator_data_folder"], f"../loss_{start}.png"
    )
    fig.savefig(save_to)
    print(f"Saved to {save_to}")
    plt.close()
