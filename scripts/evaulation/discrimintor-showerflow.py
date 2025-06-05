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


chosen = int(sys.argv[1])
print(f"We will train the {chosen}th model")

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
discriminator.create_g4_data_files(config)
# Get the generator models we will be comparing with, and check their data has been generated.

stack = []
for detailed_history in [True, False]:
    config.shower_flow_detailed_history = detailed_history
    for fixed_input_norms in [True, False]:
        config.shower_flow_fixed_input_norms = fixed_input_norms
        for weight_decay in [0., 0.1, 0.0001]:
            config.shower_flow_weight_decay = weight_decay
            stack.append(showerflow_utils.existing_models(config))
        
existing_models = {key: sum([s[key] for s in stack], []) for key in stack[0]}

existing_models["config"] = []

for i, name in enumerate(existing_models["names"]):
    model_config = showerflow_utils.construct_config(config, existing_models, i)
    existing_models["config"].append(model_config)


for i, name in enumerate(existing_models["names"]):
    if i == chosen:
        model_config = existing_models["config"][i]
        path = existing_models["paths"][i]
        print(name, path)
        discriminator.create_showerflow_data_files(model_config, path)


existing_models["paths"]

# ## data plotting
#
# We can go ahead and create a training object now, as this will load our datasets.
# Before we actually start training, we should plot some of the data as a sanity check.


def gen_training(model_idx, settings="settings12"):
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
    return model_name, training


# The data looks fine. Time to launch the training.
n_epochs = 15
print()
print(f"~~~~~~~~~ {chosen} ~~~~~~~~~~")
print()
name, training = gen_training(chosen)

training.chatty = True
try:
    training.reload()
except FileNotFoundError:
    print("No existing training found, starting from scratch")
training.chatty = True
so_far = len(training.state_dict["epochs"])
while so_far < n_epochs:
    training.train(1)
    training.save()

# make some auc plots
existing_models["auc"] = [None for _ in existing_models["names"]]
for start in range(0,len(existing_models["names"]), 5):
    fig, ax = plt.subplots()
    for plot_for in range(start, start+5):
        try:
            name, training = gen_training(plot_for)
            training.reload()
            labels, predictions = training.predict_test()
        except filenotfounderror:
            continue
        
        fpr, tpr, threasholds = metrics.roc_curve(labels, predictions)
        auc = metrics.roc_auc_score(labels, predictions)
        if auc < 0.501:
            print(f"model {plot_for} has issues")
        existing_models["auc"][plot_for] = auc
        ax.plot(fpr, tpr, label=f"{name} auc={auc}")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend()
    save_to = os.path.join(training.state_dict["generator_data_folder"], f"../auc_{start}.png")
    fig.savefig(save_to)
    print(f"saved to {save_to}")
    plt.close()



for start in range(0, len(existing_models["names"]), 5):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for plot_for in range(start, start+5):
        try:
            name, training = gen_training(plot_for)
            training.reload()
            training.plots(axes, colour=nice_hex[int(plot_for/5)%5][plot_for%5], legend_name=name)
        except FileNotFoundError:
            continue
    axes[0].semilogy()
    axes[0].set_xlim(0, 30)
    axes[1].semilogy()
    axes[0].legend()
    axes[1].legend()
    save_to = os.path.join(training.state_dict["generator_data_folder"], f"../loss_{start}.png")
    fig.savefig(save_to)
    print(f"Saved to {save_to}")
    plt.close()
