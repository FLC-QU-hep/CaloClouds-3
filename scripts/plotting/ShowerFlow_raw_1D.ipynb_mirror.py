# # Verify the performance of ShowerFlow in isolation
# 
# Using mostly accumulated data, look at the ehvaior of ShowerFlow with different configurations.
import numpy as np
from matplotlib import pyplot as plt
import torch
import os

from pointcloud.config_varients import wish, caloclouds_3
from pointcloud.models import shower_flow, fish_flow
from pointcloud.utils import stats_accumulator, metadata, showerflow_utils, showerflow_training

configs = caloclouds_3.Configs()
configs.dataset_path = "/beegfs/desy/user/akorol/data/AngularShowers_RegularDetector/"\
"hdf5_for_CC/sim-E1261AT600AP180-180_file_{}slcio.hdf5"
configs.n_dataset_files = 88
meta = metadata.Metadata(configs)


try:
    configs.device = "cuda"
    incident_energies = torch.linspace(0.1, 1.0, 5)[:, None].to(configs.device)
except Exception as e:
    configs.device = "cpu"
    incident_energies = torch.linspace(0.1, 1.0, 5)[:, None].to(configs.device)


# There are 65 values to each draw from the original ShowerFlow;
# 
# - 0. Total observed clusters / `meta.n_pts_rescale`
# - 1. Total observed energy / `meta.vis_eng_rescale`
# - 2. (Center of gravity in x - `meta.mean_cog[0]`)/`meta.std_cog[0]`
# - 3. (Center of gravity in y - `meta.mean_cog[1]`)/`meta.std_cog[1]`
# - 4. (Center of gravity in z - `meta.mean_cog[2]`)/`meta.std_cog[2]`
# - 5:35. (Observed clusters in layer i/max clusters in layer in event)
# - 35:65. (Observed energy in layer i/max energy in layer in event)
# 
# These aren't exactly the distirbutions fish is designed to produce, but they can be teased out.

# ## Dataset
# 
# Start by using the functions designed to train showerflow to draw these distributions from the dataset.

data_dir = os.path.realpath(os.path.join(configs.logdir.split("point-cloud-diffusion-logs")[0], "point-cloud-diffusion-data"))
showerflow_dir = os.path.join(data_dir, "showerFlow/sim-E1261AT600AP180-180")
print(f"Showerflow dir is {showerflow_dir}")
assert os.path.exists(showerflow_dir)
pointsE_path = showerflow_training.get_incident_npts_visible(configs, showerflow_dir)
distance_path = showerflow_training.get_gun_direction(configs, showerflow_dir)
cog_path, cog_sample = showerflow_training.get_cog(configs, showerflow_dir, local_batch_size=20)
clusters_per_layer_path = showerflow_training.get_clusters_per_layer(configs, showerflow_dir)
energy_per_layer_path = showerflow_training.get_energy_per_layer(configs, showerflow_dir)
make_train_ds = showerflow_training.train_ds_function_factory(pointsE_path, cog_path, clusters_per_layer_path, energy_per_layer_path, configs, distance_path)
n_bins = 60
cond_mask = showerflow_utils.get_cond_mask(configs)
total_cond = np.sum(cond_mask)
n_distributions = 65
bins = np.tile(np.linspace(0., 0.1, n_bins+1), (n_distributions, 1))
log_x_01 = False
if log_x_01:
    bins[0] = np.logspace(np.log10(0.05), np.log10(1.5), n_bins+1)
    bins[1] = np.logspace(np.log10(0.05), np.log10(1.5), n_bins+1)
else:
    bins[0] = np.linspace(0., 1.5, n_bins+1)
    bins[1] = np.linspace(0., 1.5, n_bins+1)
cog_extent = 2
bins[2] = np.linspace(-cog_extent, cog_extent, n_bins+1)
bins[3] = np.linspace(-cog_extent, cog_extent, n_bins+1)
bins[4] = np.linspace(-cog_extent, cog_extent, n_bins+1)
true_bins = np.copy(bins)
if log_x_01:
    true_bins[0][0] = 0.
    true_bins[1][0] = 0.
dataset_distributions = np.zeros((n_distributions, n_bins))
adaptor = fish_flow.Adaptor(configs)
use_events = 10_000
local_batch_size = 100
starts = np.arange(0, use_events, local_batch_size)
n_batches = len(starts)
dataset_n_events = 0

cond_slice = slice(0, total_cond)
idx_reached = total_cond
if "total_clusters" in configs.shower_flow_inputs:    
    sum_clusters_idx = idx_reached
    idx_reached += 1
else:
    sum_clusters_idx = None
if "total_energy" in configs.shower_flow_inputs:    
    sum_energy_idx = idx_reached
    idx_reached += 1
else:
    sum_energy_idx = None
cog_idxs = []
for c in "xyz":
    if "cog_" + c in configs.shower_flow_inputs:    
        cog_idxs.append(idx_reached)
        idx_reached += 1
    else:
        cog_idxs.append(None)
if "clusters_per_layer" in configs.shower_flow_inputs:
    clusters_per_layer_slice = slice(idx_reached,idx_reached+30)
    idx_reached += 30
else:
    clusters_per_layer_slice = None
if "energy_per_layer" in configs.shower_flow_inputs:
    energy_per_layer_slice = slice(idx_reached,idx_reached+30)
    idx_reached += 30
else:
    energy_per_layer_slice = None

dataset_cond = []
for batch_n, start in enumerate(starts):
    print(f"{batch_n/n_batches:.0%}", end='\r')
    data_matrix = make_train_ds(start, start + local_batch_size)
    data_matrix_copy = data_matrix.clone()
    start_point = total_cond + (sum_clusters_idx is not None) + (sum_energy_idx is not None)
    data_matrix_copy[:, start_point:] = adaptor.to_basis(data_matrix[:, total_cond:])
    dataset_n_events += data_matrix.shape[0]
    dataset_cond.append(data_matrix[:, :total_cond])
        
    for i, idx in enumerate([sum_clusters_idx, sum_energy_idx] + cog_idxs):
        if idx is None:
            continue
        values = data_matrix[:, idx]
        dataset_distributions[i] += np.histogram(values, bins=true_bins[i])[0]
    if clusters_per_layer_slice is not None:
        values = data_matrix_copy[:, clusters_per_layer_slice]
        for i, val in enumerate(values.T):
            dataset_distributions[i+5] += np.histogram(val, bins=true_bins[i+5])[0]
    if energy_per_layer_slice is not None:
        values = data_matrix_copy[:, energy_per_layer_slice]
        for i, val in enumerate(values.T):
            dataset_distributions[i+35] += np.histogram(val, bins=true_bins[i+35])[0]
dataset_distributions /= dataset_n_events
dataset_cond = torch.cat(dataset_cond)
        
    
# Now we need to craft a function that plots this array of distributions.
def weighted_quantiles(locations, quantiles, weights, axes, **kwargs):
    sum_weights = weights.sum(axes)
    quantile_locations = torch.tensor(quantiles)*sum_weights[:, None]
    cumsum_weights = weights.cumsum(axes)
    closest_location = (quantile_locations[..., None, :] - cumsum_weights[..., None]).abs().argmin(axes)
    return locations[closest_location].T
    
from pointcloud.utils.plotting import plot_line_with_devation, blank_axes
bin_centers = 0.5*(bins[:, 1:] + bins[:, :-1])
bin_widths = (bins[:, 1:] - bins[:, :-1])
def plot_distributions(dists, axes=None, hist_kws=None, line_kws=None, **gen_kws):
    if axes is None:
        n_rows = 2
        height_ratios = [3, 1]*n_rows
        fig, axes = plt.subplots(2*n_rows, 4, figsize=(15, 8), gridspec_kw={'height_ratios':height_ratios})
        blank_axes(axes[-1, -1])
        blank_axes(axes[-2, -1])
        ratio_axes = axes.flatten()[[4, 5, 12, 13, 14, 6, 7]]
        axes = axes.flatten()[[0, 1, 8, 9, 10, 2, 3]]
    else:
        assert len(axes) == 7
        ratio_axes = None
        fig = plt.gcf()
    hist_kws = {**gen_kws, **hist_kws} if hist_kws else gen_kws
    line_kws = {**gen_kws, **line_kws} if line_kws else gen_kws
    axes[0].set_xlabel("Total clusters")
    axes[0].hist(bin_centers[0], weights=dists[0]/bin_widths[0], bins=bins[0], **hist_kws)
    axes[1].set_xlabel("Total visible energy")
    if log_x_01:
        axes[0].semilogx()
        axes[1].semilogx()
        if ratio_axes is not None:
            ratio_axes[0].semilogx()
            ratio_axes[1].semilogx()
    axes[1].hist(bin_centers[1], weights=dists[1]/bin_widths[1], bins=bins[1], **hist_kws)
    for i, c in enumerate('XYZ', 2):
        axes[i].set_xlabel(f"Center of gravity {c}")
        axes[i].hist(bin_centers[i], weights=dists[i]/bin_widths[i], bins=bins[i], **hist_kws)
    if 'color' in line_kws:
        colour = line_kws['color']
        del line_kws['color']
    xs = np.linspace(0.5, 29.5, 30)
    if ratio_axes is not None:
        for i in range(5):
            ratio_axes[i].hlines(1, bins[i][0], bins[i][-1], color='k') 
        ratio_axes[5].hlines(1, 0, 30, color='k')
        ratio_axes[6].hlines(1, 0, 30, color='k')
    axes[5].set_xlabel("layers")
    axes[5].set_ylabel("Clusters")
    ys_down, ys, ys_up = weighted_quantiles(bin_centers[5], [0.25, 0.5, 0.75], weights=dists[5:35], axes=1)
    ys_mean = np.mean(dists[5:35], axis=1)
    ys_mean = np.sum(bin_centers[5]*dists[5:35], axis=1)/np.sum(bin_centers[5])
    plot_line_with_devation(axes[5], colour, xs, ys, ys_up, ys_down, **line_kws)
    axes[5].plot(xs, ys_mean, c=colour, ls='--')
    axes[6].set_xlabel("layers")
    axes[6].set_ylabel("Energy")
    ys_down, ys, ys_up = weighted_quantiles(bin_centers[6], [0.25, 0.5, 0.75], weights=dists[35:65], axes=1)
    ys_mean = np.sum(bin_centers[6]*dists[35:65], axis=1)/np.sum(bin_centers[6])
    plot_line_with_devation(axes[6], colour, xs, ys, ys_up, ys_down, **line_kws)
    axes[6].plot(xs, ys_mean, c=colour, ls='--')
    axes[4].legend()
    return fig, axes, ratio_axes
    

dataset_kws = {'color': 'gray', 'label': 'G4'}
fig, axes, ratio_axes = plot_distributions(dataset_distributions, **dataset_kws)
fig.tight_layout()
# ## Model varients
# 
# That done, we can draw the same values from the models, and plot them alongside.
from pointcloud.utils import showerflow_utils

saved_models = showerflow_utils.existing_models(configs)
default_input_mask = showerflow_utils.get_input_mask(configs)
max_n_inputs = len(default_input_mask)

cond_feature_mask = showerflow_utils.get_cond_mask(configs)
n_cond_features = np.sum(cond_feature_mask)

def get_distributions(saved_models):
    distributions = []
    for i, version in enumerate(saved_models["version"]):
        constructor = shower_flow.versions_dict[version]
        num_inputs = max_n_inputs - len(saved_models["cut_inputs"][i])
        model, dist = constructor(
            num_blocks = saved_models["num_blocks"][i],
            num_inputs = num_inputs,
            num_cond_inputs = n_cond_features,
            device = configs.device)
        loaded_checkpoint = torch.load(saved_models["path"][i], map_location=configs.device)
        model.load_state_dict(loaded_checkpoint["model"])
        distributions.append(dist)
    return distributions

def model_distributions(model):
    repeats = 10
    found = np.zeros((n_distributions, n_bins))
    for r in range(repeats):
        for batch_n, start in enumerate(starts):
            print(f"{(r*n_batches + batch_n)/(n_batches*repeats):.0%}", end='\r')
            cond = dataset_cond[start:start+local_batch_size].float()
            if not len(cond):
                print(f"last start, {start}")
                break
            conditioned = model.condition(cond)
            values = conditioned.sample(torch.Size([cond.size(0)]))
            values[..., 5:65] = adaptor.to_basis(values)[..., 3:63]
            for i in range(65):
                found[i] += np.histogram(values[:, i], bins=true_bins[i])[0]
    found /= (repeats*dataset_cond.shape[0])
    return found



try:
    distributions[i]
    
    print(f"Already loaded {len(distributions)}")
except Exception:
    distributions = get_distributions(saved_models)
saved_models['best_loss'] = np.array(saved_models['best_loss'])
all_inputs_mask = [i for i, x in enumerate(saved_models['cut_inputs']) if not x]
best_3 = np.array(all_inputs_mask)[np.argsort(saved_models['best_loss'][all_inputs_mask])[:3]]
len(distributions)

#best_distributions = [model_distributions(distributions[i]) for i in best_3]
try:
    all_distributions[0]
    
    print("Already run distributions")
except Exception:
    all_distributions = []
    for i, dist in enumerate(distributions):
        try:
            all_distributions.append(model_distributions(dist))
        except RuntimeError:
            print(saved_models["names"][i])

worst_dist_idx = all_inputs_mask[np.argmax(saved_models['best_loss'][all_inputs_mask])]
y_mins = 0.5*np.ones(7)
y_maxes = 1.5*np.ones(7)
def plot_ratios(ratio_axes, distribution, **dataset_kws):
    ratios = []
    xs = []
    for i in range(5):
        ratios.append(distribution[i]/dataset_distributions[i])
        xs.append(bin_centers[i])
    xs += 2*[np.linspace(0.5, 29.5, 30)]
    ds_ys,  = np.quantile(dataset_distributions[5:35], [0.5], axis=1)
    ys,  = np.quantile(distribution[5:35], [0.5], axis=1)
    ratios.append(ds_ys/ys)
    ds_ys,  = np.quantile(dataset_distributions[35:65], [0.5], axis=1)
    ys,  = np.quantile(distribution[35:65], [0.5], axis=1)
    ratios.append(ds_ys/ys)
    for i, (ratio) in enumerate(ratios):
        ratio_axes[i].plot(xs[i], ratio, **dataset_kws)
        y_mins[i] = min(max(np.nanmin(ratio), -1), y_mins[i])
        y_maxes[i] = max(min(np.nanmin(ratio), 3), y_maxes[i])
        ratio_axes[i].set_ylim((y_mins[i], y_maxes[i]))
    

dataset_kws = {'color': 'gray', 'label': 'G4'}
fig, axes, ratio_axes = plot_distributions(dataset_distributions, **dataset_kws)
from pointcloud.utils.plotting import nice_hex

dataset_hist_kws = {'histtype': 'step', 'lw': 3}
line_styles = ['-', '--', ':', '-.']
for i, model_i in enumerate(best_3[:2]):
    label = f"{saved_models['names'][model_i]}, {saved_models['best_loss'][model_i]:.1f}"
    dataset_kws = {'color': nice_hex[4][(i*2+1)%5], 'label': label, 'ls':line_styles[i%4]}
    plot_distributions(all_distributions[model_i], axes=axes, hist_kws=dataset_hist_kws, **dataset_kws)
    plot_ratios(ratio_axes, all_distributions[model_i], **dataset_kws)

label = f"{saved_models['names'][worst_dist_idx]}, {saved_models['best_loss'][worst_dist_idx]:.2f}"
dataset_kws = {'color': 'red', 'label': label}
plot_distributions(all_distributions[worst_dist_idx], axes=axes, hist_kws=dataset_hist_kws, **dataset_kws)
plot_ratios(ratio_axes, all_distributions[worst_dist_idx], **dataset_kws)


fig.tight_layout()
plt.savefig("with_worst.png")
saved_models["path"]

dataset_kws = {'color': 'gray', 'label': 'G4'}
fig, axes, ratio_axes = plot_distributions(dataset_distributions, **dataset_kws)
from pointcloud.utils.plotting import nice_hex

dataset_hist_kws = {'histtype': 'step', 'lw': 3}
line_styles = ['-', '--', ':', '-.']
for i, model_i in enumerate(best_3[:2]):
    label = f"{saved_models['names'][model_i]}, {saved_models['best_loss'][model_i]:.1f}"
    dataset_kws = {'color': nice_hex[4][(i*2+1)%5], 'label': label, 'ls':line_styles[i%4]}
    plot_distributions(all_distributions[model_i], axes=axes, hist_kws=dataset_hist_kws, **dataset_kws)
    plot_ratios(ratio_axes, all_distributions[model_i], **dataset_kws)

label = f"{saved_models['names'][worst_dist_idx]}, {saved_models['best_loss'][worst_dist_idx]:.2f}"
dataset_kws = {'color': 'red', 'label': label}
plot_distributions(all_distributions[worst_dist_idx], axes=axes, hist_kws=dataset_hist_kws, **dataset_kws)
plot_ratios(ratio_axes, all_distributions[worst_dist_idx], **dataset_kws)
for ax in axes:
    ax.semilogy()


fig.tight_layout()
plt.savefig("with_worst_logy.png")
model_names = [["original_nb4", "original_nb10"], ["alt1_nb4", "alt1_nb10"], ["original_nb4", "alt1_nb4", "alt2_nb4"]]
try:
    print(len(model_dists))
except Exception:
    model_dists = {}
for group in model_names:
    for name in group:
        if name in model_dists:
            continue
        print(name)
        idx = saved_models["names"].index(name)
        if idx in best_3:
            dist = best_distributions[list(best_3).index(idx)]
        else:
            dist = model_distributions(distributions[idx])
        model_dists[name] = dist

    
for name_group in model_names:
    idxs = [saved_models["names"].index(n) for n in name_group]
    dataset_kws = {'color': 'gray', 'label': 'G4'}
    fig, axes, ratio_axes = plot_distributions(dataset_distributions, **dataset_kws)
    from pointcloud.utils.plotting import nice_hex
    dataset_hist_kws = {'histtype': 'step', 'lw': 3}
    line_styles = ['-', '--', ':', '-.']
    for i, model_i in enumerate(idxs):
        name = name_group[i]
        dist_here = model_dists[name]
        label = f"{saved_models['names'][model_i]}, {saved_models['best_loss'][model_i]:.1f}"
        dataset_kws = {'color': nice_hex[4][(i*2+1)%5], 'label': label, 'ls':line_styles[i%4]}
        plot_distributions(dist_here, axes=axes, hist_kws=dataset_hist_kws, **dataset_kws)
        plot_ratios(ratio_axes, dist_here, **dataset_kws)



    fig.tight_layout()
    plt.savefig('.vs.'.join(name_group) + ".png")



    
for name_group in model_names:
    idxs = [saved_models["names"].index(n) for n in name_group]
    dataset_kws = {'color': 'gray', 'label': 'G4'}
    fig, axes, ratio_axes = plot_distributions(dataset_distributions, **dataset_kws)
    from pointcloud.utils.plotting import nice_hex
    dataset_hist_kws = {'histtype': 'step', 'lw': 3}
    line_styles = ['-', '--', ':', '-.']
    for i, model_i in enumerate(idxs):
        name = name_group[i]
        dist_here = model_dists[name]
        label = f"{saved_models['names'][model_i]}, {saved_models['best_loss'][model_i]:.1f}"
        dataset_kws = {'color': nice_hex[4][(i*2+1)%5], 'label': label, 'ls':line_styles[i%4]}
        plot_distributions(dist_here, axes=axes, hist_kws=dataset_hist_kws, **dataset_kws)
        plot_ratios(ratio_axes, dist_here, **dataset_kws)

    for ax in axes:
        ax.semilogy()

    fig.tight_layout()
    plt.savefig('.vs.'.join(name_group) + ".png")


all_distributions



