# # ShowerAlign
# 
# Plotting the output of AlignedStatsAccumulators.
# 
# Let us imagine that we find the right distribution from the shift, then we know how to generate the correct center points for the shower.
# The number of hits found at a given point along the shower axis could be a function of;
# 
# - The incident energy
# - The depth of the shower (size of the shift)
# - The distance from the point to the center of the shower along the shower axis.
# 
# We will start by looking at the function that defines the layer offset itself.
# 
# - Is this corrilated with incident energy?
# - What is the distribution in a slice?
# 
# This must be done with raw data.

from pointcloud.config_varients.wish import Configs
from pointcloud.config_varients.wish_maxwell import Configs as MWConfigs
from pointcloud.utils.stats_accumulator import AlignedStatsAccumulator, save_location
from pointcloud.data.dataset import dataset_class_from_config
from pointcloud.data.read_write import read_raw_regaxes, get_n_events
from matplotlib import pyplot as plt
from plotly import express as ptx
import pandas
import numpy as np
import torch
from torch.distributions import Gumbel, Exponential
from pointcloud.utils.optimisers import curve_fit
configs = MWConfigs()
configs.dataset_path = "/home/dayhallh/Data/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5"
configs.dataset_path = "/beegfs/desy/user/dayhallh/data/ILCsoftEvents/p22_th90_ph90_en10-100_joined/p22_th90_ph90_en10-100_seed{}_all_steps.hdf5"

dataset_class = dataset_class_from_config(configs)

align_center = "Mean" if True else "Peak"
align_even = f"align{align_center}Even"
align_odd = f"align{align_center}Odd"
align_all = f"align{align_center}"

acc_even = AlignedStatsAccumulator.load(save_location(configs, 1, 0, align_even))
acc_odd = AlignedStatsAccumulator.load(save_location(configs, 1, 0, align_odd))
acc_all = AlignedStatsAccumulator.load(save_location(configs, 1, 0, align_all))
acc = acc_all
total_events = get_n_events(configs.dataset_path, configs.n_dataset_files)
check_events = min(sum(total_events), 10_000)
batch_size = 1_000
shifts = np.empty((check_events, 2))
for start_idx in range(0, check_events, batch_size):
    print(f"{start_idx/check_events:.0%}", end='\r')
    end_idx = start_idx + batch_size
    energies, events = read_raw_regaxes(configs, pick_events=slice(start_idx, end_idx))
    dataset_class.normalize_xyze(events)
    shifts[start_idx:end_idx, 0] = energies
    shifts[start_idx:end_idx, 1] = acc.get_shift(events).flatten()
print("Done")
dataframe = pandas.DataFrame({"incident energy":shifts[:, 0], "shift":shifts[:, 1]})
fig = ptx.scatter(dataframe, x="incident energy", y="shift", opacity=0.3)
coeffs = np.polyfit(shifts[:, 0], shifts[:, 1], configs.poly_degree)
xs = np.linspace(np.min(shifts[:, 0]), np.max(shifts[:, 0]), 100)
ys = np.polyval(coeffs, xs)
fig.add_scatter(x=xs, y=ys, name="Mean fit")
# This has a clear trend with incident energy, so we fit this as a polynomial, and restack the shifts.
# The objective is to see a distibution of the shifts about the mean at a given incident energy.
# 
# Upon stacking these, it seems that a Gumbel distribution could be apropreate.
stacked_shifts = np.copy(shifts)
mean_at = np.polyval(coeffs, stacked_shifts[:, 0])
stacked_shifts[:, 1] -= mean_at

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax in axes:
    heights, bins, patches = ax.hist(stacked_shifts[:, 1], bins=50, histtype="step", density=True, label="measured")

bin_centers = 0.5*(bins[1:] + bins[:-1])
mean = np.sum(bin_centers*heights)/np.sum(heights)
non_zero_weights = np.sum(heights>0)
varience = (non_zero_weights/(non_zero_weights-1))*np.sum(heights*(bin_centers-mean)**2)/np.sum(heights)

beta = np.sqrt(6*varience)/np.pi
mu = mean - beta*np.euler_gamma

distribution = Gumbel(mu, beta)
probs = np.exp(distribution.log_prob(torch.Tensor(bins)))
for ax in axes:
    ax.plot(bins, probs, label="Gumbel")
    ax.set_xlabel("aligned shift")

axes[1].semilogy()
axes[0].legend()

axes[0].set_ylabel("counts")


# This will be used to pick shifts. And given the shift, we want to predict the number of points at any given position along the axis.
# So, three possible input values;
# 
# - Incident energy
# - Shift
# - Distance from center
# 
# Let's look for correlations with these.
# To start with we will check that the shifted accumulator has done something sensible.
# Pull a small sample from the data and compare.
# 
# Then we will move on to looking at how tightly corrilated the three 

ze_sample = np.zeros((check_events, configs.max_points*10, 2))

for start_idx in range(0, check_events, batch_size):
    print(f"{start_idx/check_events:.0%}", end='\r')
    end_idx = start_idx + batch_size
    energies, events = read_raw_regaxes(configs, pick_events=slice(start_idx, end_idx))
    dataset_class.normalize_xyze(events)
    e = events[:, :, 3]
    n_pts = e.shape[1]
    mask = e>0
    ze_sample[start_idx:end_idx, :n_pts, 0][mask] = events[..., 2][mask]
    ze_sample[start_idx:end_idx, :n_pts, 1][mask] = e[mask]
ze_sample[..., 0] -= shifts[:, 1, None]
print("Done")



counts = acc_even.counts_hist[1:-1] + acc_odd.counts_hist[1:-1]
counts_along_axis = np.sum(counts, axis=(2, 3))
y_values = acc.layer_bottom
incident_bins = acc.incident_bin_boundaries
inci_values = 0.5*(incident_bins[1:] + incident_bins[:-1])

color_map = plt.cm.viridis(inci_values/np.max(inci_values))
for i, e in enumerate(inci_values):
    counts_here = counts_along_axis[i]
    normalisation = np.sum(counts_here)*(y_values[1] - y_values[0])
    plt.plot(y_values, counts_here/normalisation, c=color_map[i], label=f"Incident {e}MeV")

y_values = ze_sample[..., 0][ze_sample[..., 1]>0]
_ = plt.hist(y_values, bins=50, color='grey', alpha=0.5, histtype='step', lw=3, density=True)
_ = plt.xlabel("y position after shift")
_ = plt.ylabel("Normalised counts")
_ = plt.legend()
shift_bins = acc.layer_offset_bins
incid_vs_depth = np.zeros((len(inci_values), len(shift_bins)-1))

for row, (lower, upper) in enumerate(zip(incident_bins[:-1], incident_bins[1:])):
    inci_mask = (shifts[..., 0] >= lower)*(shifts[..., 0] < upper)
    for col, (lower, upper) in enumerate(zip(shift_bins[:-1], shift_bins[1:])):
        shift_mask = (shifts[..., 1] >= lower)*(shifts[..., 1] < upper)
        mask = inci_mask*shift_mask
        energies = ze_sample[mask, :, 1]
        incid_vs_depth[row, col] = np.sum(energies>0)
plt.imshow(incid_vs_depth)
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Depth")
x_ticks = [f"{b:.2}" for b in shift_bins]
plt.xticks((np.arange(len(shift_bins))-0.5)[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_values]
plt.yticks(np.arange(len(inci_values))[::2], y_ticks[::2])
cbar = plt.colorbar()
cbar.set_label("N hits")
plt.tight_layout()

shift_bins = acc.layer_offset_bins
incid_vs_depth = np.zeros((len(inci_values), len(shift_bins)-1))

for row, (lower, upper) in enumerate(zip(incident_bins[:-1], incident_bins[1:])):
    inci_mask = (shifts[..., 0] >= lower)*(shifts[..., 0] < upper)
    for col, (lower, upper) in enumerate(zip(shift_bins[:-1], shift_bins[1:])):
        shift_mask = (shifts[..., 1] >= lower)*(shifts[..., 1] < upper)
        mask = inci_mask*shift_mask
        energies = ze_sample[mask, :, 1]
        incid_vs_depth[row, col] = np.sum(energies>0)/np.sum(mask)
plt.imshow(incid_vs_depth)
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Depth")
x_ticks = [f"{b:.2}" for b in shift_bins]
plt.xticks((np.arange(len(shift_bins))-0.5)[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_values]
plt.yticks(np.arange(len(inci_values))[::2], y_ticks[::2])
cbar = plt.colorbar()
cbar.set_label("N hits per event")
plt.tight_layout()

shift_bins = acc.layer_offset_bins
incid_vs_shift = np.zeros((len(inci_values), len(shift_bins)-1))



for row, (lower, upper) in enumerate(zip(incident_bins[:-1], incident_bins[1:])):
    print(f"{row/len(inci_values):.0%}", end='\r')
    inci_mask = (shifts[..., 0] >= lower)*(shifts[..., 0] < upper)
    subsample = ze_sample[inci_mask]
    for col, (lower, upper) in enumerate(zip(shift_bins[:-1], shift_bins[1:])):
        mask = (subsample[..., 0] >= lower)*(subsample[..., 0] < upper)*(subsample[..., 1] > 0)
        # the shifting means that some bins are only poulated by a subset of events
        # if the shift makes the center more than one unit away ten the event does not intersect
        num_intersecting = ((shifts[:, 1]+1) > upper)*((shifts[:, 1]-1) < lower)*inci_mask
        incid_vs_shift[row, col] = np.sum(mask)/np.sum(num_intersecting)
plt.imshow(incid_vs_shift)
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in shift_bins]
plt.xticks((np.arange(len(shift_bins))-0.5)[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_values]
plt.yticks(np.arange(len(inci_values))[::2], y_ticks[::2])
cbar = plt.colorbar()
cbar.set_label("N hits per event")
plt.tight_layout()

shift_bins = acc.layer_offset_bins
depth_vs_shift = np.zeros((len(shift_bins)-1, len(shift_bins)-1))

for row, (lower, upper) in enumerate(zip(shift_bins[:-1], shift_bins[1:])):
    print(f"{row/len(shift_bins):.0%}", end='\r')
    shift_mask = (shifts[..., 1] >= lower)*(shifts[..., 1] < upper)
    subsample = ze_sample[shift_mask]
    for col, (lower, upper) in enumerate(zip(shift_bins[:-1], shift_bins[1:])):
        mask = (subsample[..., 0] >= lower)*(subsample[..., 0] < upper)*(subsample[..., 1] > 0)
        # the shifting means that some bins are only poulated by a subset of events
        # if the shift makes the center more than one unit away ten the event does not intersect
        num_intersecting = ((shifts[:, 1]+1) > upper)*((shifts[:, 1]-1) < lower)*shift_mask
        depth_vs_shift[row, col] = np.sum(mask)/np.sum(num_intersecting)
plt.imshow(depth_vs_shift)
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in shift_bins]
plt.xticks((np.arange(len(shift_bins))-0.5)[::5], x_ticks[::5])
plt.ylabel("Depth")
plt.yticks((np.arange(len(shift_bins))-0.5)[::5], x_ticks[::5])
cbar = plt.colorbar()
cbar.set_label("N hits")
plt.tight_layout()

# ## Conculsions for fitting
# 
# There are no reliable strong corrilations here. For a simple model, we will ignore the impact of changing the depth of the shower,
# and consider point density to be only dependent on the incident energy and the distance from the center of the shower. 
# 
# Given this, we wish to calculate a mean, and a standard devation for the number of points at each incident energy and distance from the center.
# Once we have these values, two things must be done;
# 
# 1. Find a function (not a pdf) that can fit the mean and standard devation in two dimensions;
# 
#      - This can start by plotting slices to decide if the change could be a simple rescale in either direction
# 2. Use the mean and standard devation to rescale density functions for all bins of incident energy and distance, check if it's reasonable to fit them all to the same pdf after rescaling.
# 3. When set up this way, it might be possible to write a super fast wish that draws all layers at once.
# 
# ### Starting with mean

inci_bins = 0.5*(acc.incident_bin_boundaries[:-1] + acc.incident_bin_boundaries[1:])
layer_bins = acc.layer_bottom
events_per_bin = acc.total_events[1:-2]
atleast_three = events_per_bin>3

counts_per_bin = np.sum(acc.counts_hist[1:-1], axis=(2, 3))
counts_per_event = counts_per_bin/events_per_bin

plt.imshow(counts_per_event)
plt.title("Measured")
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
cbar = plt.colorbar()
cbar.set_label("Mean N hits per event")
plt.tight_layout()


n_rows, n_cols = counts_per_event.shape
fig, (row_ax, col_ax) = plt.subplots(1, 2, figsize=(10, 4))
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, colour in enumerate(colours):
    row_ax.plot(layer_bins, counts_per_event[row], c=colour)
row_ax.set_xlabel("Distance to center")
row_ax.set_ylabel("mean N hits per event")
abs_to_center = np.abs(layer_bins)
order_abs_to_center = np.argsort(abs_to_center)
colours = plt.cm.inferno(np.linspace(0, 1, n_cols))
for col, colour in zip(order_abs_to_center, colours):
    ls = '--' if layer_bins[col] < 0 else '-'
    col_ax.plot(inci_bins, counts_per_event[:, col], c=colour, ls=ls)
col_ax.set_xlabel("Incident energy")
col_ax.set_ylabel("mean N hits per event")


# So for mean n pts per bin, the incident energy is nearly a linear fit. Treat this as a rescaling, and the distance to the center gets to be normally disributed.

evts_per_inci_bin = np.sum(events_per_bin, axis=1)
counts_per_inci_bin = np.sum(acc.counts_hist[1:-1], axis=(1, 2, 3))
inci_linear_fit = np.polyfit(inci_bins, counts_per_inci_bin/evts_per_inci_bin, 1)

inci_scalings = np.polyval(inci_linear_fit, inci_bins)
scaled_counts_per_event = counts_per_event/inci_scalings[:, np.newaxis]
scaled_mean_counts_per_layer = np.mean(scaled_counts_per_event, axis=0)

n_pts_dist_mean = np.nansum(scaled_mean_counts_per_layer*layer_bins)/np.nansum(scaled_mean_counts_per_layer)
non_zero_weights = np.sum(scaled_mean_counts_per_layer>0)
n_pts_dist_varience = (non_zero_weights/(non_zero_weights-1))*np.nansum(scaled_mean_counts_per_layer*(layer_bins-mean)**2)/np.nansum(scaled_mean_counts_per_layer)


def gaussian(xs, mu, varience, height):
    return (height/(2*np.pi))*np.exp(-(xs-mu)**2/(2*varience))
    
height_1_pred = gaussian(layer_bins, n_pts_dist_mean, n_pts_dist_varience, 1.)
ratios = scaled_mean_counts_per_layer/height_1_pred
n_pts_dist_height = np.nansum(ratios*height_1_pred)/np.nansum(height_1_pred)
predicted_counts_per_event = gaussian(layer_bins, n_pts_dist_mean, n_pts_dist_varience, n_pts_dist_height)


predicted_bins = np.tile(predicted_counts_per_event, (counts_per_event.shape[0], 1))
predicted_bins *= inci_scalings[:, None]


n_rows, n_cols = scaled_counts_per_event.shape
fig, (row_ax, col_ax) = plt.subplots(1, 2, figsize=(10, 4))
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, colour in enumerate(colours):
    row_ax.plot(layer_bins, scaled_counts_per_event[row], c=colour)
    pass
row_ax.plot(layer_bins, scaled_mean_counts_per_layer, c="blue", lw=5, alpha=0.2)
row_ax.plot(layer_bins, predicted_counts_per_event, c="blue", ls='--', lw=5, alpha=0.5)
row_ax.set_xlabel("Distance to center")
row_ax.set_ylabel("rescaled mean N hits per event")


abs_to_center = np.abs(layer_bins)
order_abs_to_center = np.argsort(abs_to_center)
colours = plt.cm.inferno(np.linspace(0, 1, n_cols))
for col, colour in zip(order_abs_to_center, colours):
    ls = '--' if layer_bins[col] < 0 else '-'
    col_ax.plot(inci_bins, counts_per_event[:, col], c=colour, ls=ls)
    col_ax.plot(inci_bins, predicted_bins[:, col], c=colour, ls=ls, lw=5, alpha=0.5)
col_ax.set_xlabel("Incident energy")
col_ax.set_ylabel("mean N hits per event")


fig, ax = plt.subplots()
plt.title("Predictions")
plt.imshow(predicted_bins)
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
cbar = plt.colorbar()
cbar.set_label("Mean N hits per event")
plt.tight_layout()


fig, ax = plt.subplots()
plt.title("Predictions - measured")
plt.imshow(predicted_bins-counts_per_event)
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
cbar = plt.colorbar()
cbar.set_label("Mean N hits per event")
plt.tight_layout()

# This is an acceptably good repeproduction of the mean.
# ## Moving to standard devation
# 
# or actually, coefficient of variation which is
# $$ CV = \sigma/\mu $$
# continue with the previous accumulator
counts_per_bin = np.sum(acc.counts_hist[1:-1], axis=(2, 3))
counts_per_event = counts_per_bin/events_per_bin

counts_sq_per_event = acc.evt_counts_sq_hist[1:-1]/events_per_bin
stddev_counts = np.sqrt(counts_sq_per_event - counts_per_event**2)
cv_counts = stddev_counts/counts_per_event


plt.imshow(cv_counts)
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
cbar = plt.colorbar()
cbar.set_label("coefficient of variation N hits per event")
plt.tight_layout()


n_rows, n_cols = cv_counts.shape
fig, (row_ax, col_ax) = plt.subplots(1, 2, figsize=(10, 4))
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, colour in enumerate(colours):
    row_ax.plot(layer_bins, cv_counts[row], c=colour)
row_ax.set_xlabel("distance to center")
row_ax.set_ylabel("cv n hits per event")
abs_to_center = np.abs(layer_bins)
order_abs_to_center = np.argsort(abs_to_center)
colours = plt.cm.inferno(np.linspace(0, 1, n_cols))
for col, colour in zip(order_abs_to_center, colours):
    ls = '--' if layer_bins[col] < 0 else '-'
    col_ax.plot(inci_bins, cv_counts[:, col], c=colour, ls=ls)
col_ax.set_xlabel("Incident energy")
col_ax.set_ylabel("CV N hits per event")


# The coefficient of variation is more intresting than the standard devation, since the standard devation very closely follows the mean. It has no significant change in incident energy, but wants a clipped, polynomial fit in distance to center.
mean_cv = np.nansum(cv_counts*evts_per_inci_bin[:, np.newaxis], axis=0)/np.nansum(evts_per_inci_bin)
mask = mean_cv<0
poly_order = 12
fit_ys = np.copy(mean_cv)
fit_ys[mask] = 0

p0 = 10*np.ones(poly_order)
p0[1] = -1

# we need to force the fit to go down at the edges
# simplest way is to give the fit the clipping
def sad_poly(xs, *coeffs):
    return np.clip(np.polyval(coeffs, xs), 0, None)
#def sad_poly(xs, *coeffs):
#    return np.clip(np.polynomial.chebyshev.chebval(xs, coeffs), 0, None)
    
cv_coeffs, _ = curve_fit(sad_poly, layer_bins, fit_ys, p0=p0, n_attempts=3, quiet=True)
predictions = sad_poly(layer_bins, *cv_coeffs)

fig, (row_ax, col_ax) = plt.subplots(1, 2, figsize=(10, 4))
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, colour in enumerate(colours):
    row_ax.plot(layer_bins, cv_counts[row], c=colour)
row_ax.set_xlabel("distance to center")
row_ax.set_ylabel("cv n hits per event")

row_ax.plot(layer_bins, predictions, lw=5, c="blue", alpha=0.5)

col_ax.plot(layer_bins, predictions, lw=5, c="blue", alpha=0.5)

col_ax.plot(layer_bins, fit_ys, lw=1, c="red")
col_ax.set_xlabel("distance to center")
col_ax.set_ylabel("cv n hits per event")
# This is not a beautiful fit, but it is suficient for this model. This can be used with the mean value to find the standard devation. 
# 
# 
# ## Per bin distribution
# 
# From this we must create a function that can return a mean and standard devation, given an incident energy and a distance to the center. Then the distribution of the number of hist in each bin will be rescaled, to have the same mean and standard devation, and the distributions will be stacked. Hopfully they all have the sameish Weibull distribution after rescaling.
def stats_for_n_pts(incident_energies, distance_to_center):
    inci_scalings = np.polyval(inci_linear_fit, incident_energies)
    predicted_counts_per_event = gaussian(distance_to_center, n_pts_dist_mean, n_pts_dist_varience, n_pts_dist_height)
    means = predicted_counts_per_event * inci_scalings
    cv = sad_poly(distance_to_center, *cv_coeffs) 
    stds = means*cv
    return means, stds

def normalise_distribution(points, mean, std):
    normed_points = points - mean
    normed_points /= std
    return normed_points
    

raw_distributions = [[None for _ in incident_bins[1:]] for _ in shift_bins[1:]]
normed_distributions = [[None for _ in incident_bins[1:]] for _ in shift_bins[1:]]

for row, (lower, upper) in enumerate(zip(incident_bins[:-1], incident_bins[1:])):
    print(f"{row/len(inci_values):.0%}", end='\r')
    mean_inci = 0.5*(upper + lower)
    inci_mask = (shifts[..., 0] >= lower)*(shifts[..., 0] < upper)
    # the subsample is the points in the events in the incident energy bin
    subsample = ze_sample[inci_mask]
    for col, (lower, upper) in enumerate(zip(shift_bins[:-1], shift_bins[1:])):
        # this mask will select both the righ layer, and the non-padding
        mask = (subsample[..., 0] >= lower)*(subsample[..., 0] < upper)*(subsample[..., 1] > 0)
        # the shifting means that some bins are only poulated by a subset of events
        # if the shift makes the center more than one unit away ten the event does not intersect
        num_intersecting = ((shifts[:, 1]+1) > upper)*((shifts[:, 1]-1) < lower)*inci_mask
        # sum over the real points in this layer in each event, and divide by all events that could contribute to this layer
        n_pts_in_layer_per_event = np.sum(mask, axis=1)/np.sum(num_intersecting)
        raw_distributions[col][row] = n_pts_in_layer_per_event
        mean, std = stats_for_n_pts(mean_inci, 0.5*(upper+lower))
        normed_distributions[col][row] = normalise_distribution(n_pts_in_layer_per_event, mean, std)
n_bins = 50

x_shift = 0.001
min_x = np.log10(x_shift*0.001)
max_x = np.log10(x_shift*1000)

from scipy.stats import gaussian_kde

n_slices = 5
slices = np.linspace(0, len(raw_distributions)-1, n_slices).astype(int)

fig, axarr = plt.subplots(n_slices, 2, figsize=(14, 4*n_slices))
n_inci_bins = len(inci_bins)
for slice_n, (ax_raw, ax_norm) in zip(slices, axarr):
    for i in range(0, n_inci_bins):
        color = plt.cm.magma(i/n_inci_bins)
        
        dist_min = raw_distributions[slice_n][i].min() - x_shift
        dist_max = raw_distributions[slice_n][i].max() - x_shift
        log_distribution = np.log10(raw_distributions[slice_n][i] - dist_min)
        density = gaussian_kde(log_distribution, bw_method=0.5)
        x_lin = np.linspace(min_x, dist_max*3, n_bins)
        estimates = (10**density(x_lin))
        x_log = np.logspace(min_x, dist_max*3, n_bins)
        #ax_raw.plot(x_log, estimates, c=color)
        ax_raw.hist(raw_distributions[slice_n][i] - dist_min, bins=x_log, alpha=0.5, color=color)
    
        dist_min = normed_distributions[slice_n][i].min() - x_shift
        log_distribution = np.log10(normed_distributions[slice_n][i] - dist_min)
        density = gaussian_kde(log_distribution, bw_method=0.5)
        estimates = (10**density(x_lin))
        #ax_norm.plot(x_log, estimates, c=color)
        ax_norm.hist(normed_distributions[slice_n][i] - dist_min, bins=x_log, alpha=0.5, color=color)
    
    ax_raw.loglog()
    ax_raw.set_xlabel("(counts in layer)/(events that intersect layer)")
    ax_raw.set_ylabel(f"Density, layer {slice_n}")
    ax_norm.loglog()
    ax_norm.set_xlabel("normed (counts in layer)/(events that intersect layer)")

# This is far from a perfect fix, but it's still a significant improvement.
# 
# ## Energy
# 
# Similar manipulations will allow us to obtain mean point energy distributions.


# chose one accumulator
inci_bins = 0.5*(acc.incident_bin_boundaries[:-1] + acc.incident_bin_boundaries[1:])
layer_bins = acc.layer_bottom
events_per_bin = acc.total_events[1:-2]
atleast_three = events_per_bin>3

event_mean_e = acc.evt_mean_E_hist[1:-1]/events_per_bin


plt.imshow(event_mean_e)
plt.title("Measured")
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
cbar = plt.colorbar()
cbar.set_label("Mean hit energy")
plt.tight_layout()


n_rows, n_cols = event_mean_e.shape
fig, (row_ax, col_ax) = plt.subplots(1, 2, figsize=(10, 4))
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, colour in enumerate(colours):
    row_ax.plot(layer_bins, event_mean_e[row], c=colour)
row_ax.set_xlabel("Distance to center")
row_ax.set_ylabel("mean hit energy")
abs_to_center = np.abs(layer_bins)
order_abs_to_center = np.argsort(abs_to_center)
colours = plt.cm.inferno(np.linspace(0, 1, n_cols))
for col, colour in zip(order_abs_to_center, colours):
    ls = '--' if layer_bins[col] < 0 else '-'
    col_ax.plot(inci_bins, event_mean_e[:, col], c=colour, ls=ls)
col_ax.set_xlabel("Incident energy")
col_ax.set_ylabel("mean hit energy")


n_rows, n_cols = event_mean_e.shape
fig, (row_ax, col_ax) = plt.subplots(1, 2, figsize=(10, 4))
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, colour in enumerate(colours):
    row_ax.plot(layer_bins, event_mean_e[row], c=colour)
row_ax.set_xlabel("Distance to center")
row_ax.set_ylabel("mean hit energy")
abs_to_center = np.abs(layer_bins)
order_abs_to_center = np.argsort(abs_to_center)
colours = plt.cm.inferno(np.linspace(0, 1, n_cols))
for col, colour in zip(order_abs_to_center, colours):
    ls = '--' if layer_bins[col] < 0 else '-'
    col_ax.plot(inci_bins, event_mean_e[:, col], c=colour, ls=ls)
col_ax.set_xlabel("Incident energy")
col_ax.set_ylabel("mean hit energy")

def gumble(xs, mu, beta, height, lift):
    z = (xs - mu)/beta
    return lift + height*np.exp(-z - np.exp(-z))

p0 = [0., 1., 1., 0.]
bounds = [[-10, 0., 0., -10], [10, np.inf, 10., 10]]
found_params = np.zeros((len(inci_bins), len(p0)))
mask = (layer_bins>-1) * (layer_bins<1.)
xs = layer_bins[mask]
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, mean_e in enumerate(event_mean_e):
    found_params[row], _ = curve_fit(gumble, xs, mean_e[mask], bounds=bounds, p0=p0, n_attempts=3, quiet=True)
    predictions = gumble(layer_bins, *found_params[row])
    row_ax.plot(layer_bins, predictions, c=colours[row], lw=5, alpha=0.6)
row_ax.set_ylim(0.1, 1)
row_ax.semilogy()

fig, ax_arr = plt.subplots(2, 2, figsize=(10, 5))
ax_arr = ax_arr.flatten()

for i, name in enumerate(["mu", "beta", "height", "lift"]):
    ax = ax_arr[i]
    ax.plot(inci_values, found_params[:, i])
    ax.set_ylabel(name)
    ax.set_xlabel("Incident energy")
fig.tight_layout()

meanE_vs_inci_mu = np.polyfit(inci_values, found_params[:, 0], 2)
meanE_vs_inci_beta = np.polyfit(inci_values, found_params[:, 1], 2)
meanE_vs_inci_height = np.polyfit(inci_values, found_params[:, 2], 1)
meanE_vs_inci_lift = np.polyfit(inci_values, found_params[:, 3], 1)

ax_arr[0].plot(inci_values, np.polyval(meanE_vs_inci_mu, inci_values), c='blue', lw=5, alpha=0.5)
ax_arr[1].plot(inci_values, np.polyval(meanE_vs_inci_beta, inci_values), c='blue', lw=5, alpha=0.5)
ax_arr[2].plot(inci_values, np.polyval(meanE_vs_inci_height, inci_values), c='blue', lw=5, alpha=0.5)
ax_arr[3].plot(inci_values, np.polyval(meanE_vs_inci_lift, inci_values), c='blue', lw=5, alpha=0.5)

def predict_event_mean_e(distances, incidents):
    mu = np.polyval(meanE_vs_inci_mu, incidents)
    beta = np.polyval(meanE_vs_inci_beta, incidents)
    height = np.polyval(meanE_vs_inci_height, incidents)
    lift = np.polyval(meanE_vs_inci_lift, incidents)
    distances = distances.reshape(-1, 1)
    predictions = gumble(distances, mu, beta, height, lift)
    return predictions.T

predicted_event_mean_e = predict_event_mean_e(layer_bins, inci_values)
fig, ax = plt.subplots(figsize=(10, 3))
image = ax.imshow(predicted_event_mean_e)
cbar = plt.colorbar(image)
cbar.set_label("Mean hit energy")
ax.set_title("Predicted")
ax.set_xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
plt.tight_layout()
    

fig, ax = plt.subplots(figsize=(10, 3))
image = ax.imshow(predicted_event_mean_e - event_mean_e)
cbar = plt.colorbar(image)
cbar.set_label("Mean hit energy")
ax.set_title("Predicted - measured")
ax.set_xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
plt.tight_layout()
    
# This is actually a really nice fit. We will continue to assume that the distribution in each bin is a log normal. For this, we also need the standard devation, or the coefficient of variation. Let's start by calculating the ocefficient of variation in each bin.


event_mean_e_sq = acc.evt_mean_E_sq_hist[1:-1]/events_per_bin
stddev_event_mean_e = np.sqrt(event_mean_e_sq - event_mean_e**2)

plt.imshow(stddev_event_mean_e)
plt.title("Measured")
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
cbar = plt.colorbar()
cbar.set_label("std dev Mean hit energy")
plt.tight_layout()


n_rows, n_cols = event_mean_e.shape
fig, (row_ax, col_ax) = plt.subplots(1, 2, figsize=(10, 4))
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, colour in enumerate(colours):
    row_ax.plot(layer_bins, stddev_event_mean_e[row], c=colour)
row_ax.set_xlabel("Distance to center")
row_ax.set_ylabel("std dev mean hit energy")
abs_to_center = np.abs(layer_bins)
order_abs_to_center = np.argsort(abs_to_center)
colours = plt.cm.inferno(np.linspace(0, 1, n_cols))
for col, colour in zip(order_abs_to_center, colours):
    ls = '--' if layer_bins[col] < 0 else '-'
    col_ax.plot(inci_bins, stddev_event_mean_e[:, col], c=colour, ls=ls)
col_ax.set_xlabel("Incident energy")
col_ax.set_ylabel("std dev mean hit energy")

# This seems like it could have the same treatment as for the coefficient of variation of the number of points, only it's the standard devation itself.

mean_std = np.nansum(stddev_event_mean_e*evts_per_inci_bin[:, np.newaxis], axis=0)/np.nansum(evts_per_inci_bin)
mask = mean_std<0
poly_order = 12
fit_ys = np.copy(mean_std)
fit_ys[mask] = 0

p0 = 10*np.ones(poly_order)
p0[1] = -1

# we need to force the fit to go down at the edges
# simplest way is to give the fit the clipping
def sad_poly(xs, *coeffs):
    return np.clip(np.polyval(coeffs, xs), 0, None)
#def sad_poly(xs, *coeffs):
#    return np.clip(np.polynomial.chebyshev.chebval(xs, coeffs), 0, None)
    
std_event_mean_e_coeffs, _ = curve_fit(sad_poly, layer_bins, fit_ys, p0=p0, n_attempts=3, quiet=True)
predictions = sad_poly(layer_bins, *std_event_mean_e_coeffs)

fig, (row_ax, col_ax) = plt.subplots(1, 2, figsize=(10, 4))
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, colour in enumerate(colours):
    row_ax.plot(layer_bins, stddev_event_mean_e[row], c=colour)
row_ax.set_xlabel("distance to center")
row_ax.set_ylabel("std dev mean hit energy")

row_ax.plot(layer_bins, predictions, lw=5, c="blue", alpha=0.5)

col_ax.plot(layer_bins, predictions, lw=5, c="blue", alpha=0.5)

col_ax.plot(layer_bins, fit_ys, lw=1, c="red")
col_ax.set_xlabel("distance to center")
col_ax.set_ylabel("std dev mean hit energy")
# Good enough.
# 
# ## Restacked event mean hit energy
# 
# Lets use these distributions to try an restack the event mean hit energy distributions and see if they are lining up better.

def stats_for_event_mean_e(incident_energies, distance_to_center):
    mu = np.polyval(meanE_vs_inci_mu, incident_energies)
    beta = np.polyval(meanE_vs_inci_beta, incident_energies)
    height = np.polyval(meanE_vs_inci_height, incident_energies)
    lift = np.polyval(meanE_vs_inci_lift, incident_energies)
    means = gumble(distance_to_center, mu, beta, height, lift)
    stds = sad_poly(distance_to_center, *std_event_mean_e_coeffs) 
    return means, stds


raw_mean_e_dists = [[None for _ in incident_bins[1:]] for _ in shift_bins[1:]]
normed_mean_e_dists = [[None for _ in incident_bins[1:]] for _ in shift_bins[1:]]

for row, (lower, upper) in enumerate(zip(incident_bins[:-1], incident_bins[1:])):
    print(f"{row/len(inci_values):.0%}", end='\r')
    mean_inci = 0.5*(upper + lower)
    inci_mask = (shifts[..., 0] >= lower)*(shifts[..., 0] < upper)
    # the subsample is the points in the events in the incident energy bin
    subsample = ze_sample[inci_mask]
    for col, (lower, upper) in enumerate(zip(shift_bins[:-1], shift_bins[1:])):
        # this mask will select both the righ layer, and the non-padding
        mask = (subsample[..., 0] >= lower)*(subsample[..., 0] < upper)*(subsample[..., 1] > 0)
        # the shifting means that some bins are only poulated by a subset of events
        # if the shift makes the center more than one unit away ten the event does not intersect
        num_intersecting = ((shifts[:, 1]+1) > upper)*((shifts[:, 1]-1) < lower)*inci_mask
        # sum over the real points in this layer in each event, and divide by all events that could contribute to this layer
        mean_e_in_layer_per_event = np.sum(subsample[..., 1]*mask, axis=1)/(np.sum(num_intersecting)*np.sum(mask))
        raw_mean_e_dists[col][row] = mean_e_in_layer_per_event
        mean, std = stats_for_event_mean_e(mean_inci, 0.5*(upper+lower))
        normed_mean_e_dists[col][row] = normalise_distribution(mean_e_in_layer_per_event, mean, std)
n_bins = 50

x_shift = 0.00001
min_x = np.log10(x_shift*0.1)
max_x = np.log10(x_shift*1000)

from scipy.stats import gaussian_kde

n_slices = 5
slices = np.linspace(0, len(raw_distributions)-1, n_slices).astype(int)

fig, axarr = plt.subplots(n_slices, 2, figsize=(14, 4*n_slices))
n_inci_bins = len(inci_bins)
for slice_n, (ax_raw, ax_norm) in zip(slices, axarr):
    for i in range(0, n_inci_bins):
        color = plt.cm.magma(i/n_inci_bins)
        
        dist_min = raw_mean_e_dists[slice_n][i].min() - x_shift
        dist_max = raw_mean_e_dists[slice_n][i].max() - x_shift
        log_distribution = np.log10(raw_mean_e_dists[slice_n][i] - dist_min)
        density = gaussian_kde(log_distribution, bw_method=0.5)
        x_lin = np.linspace(min_x, dist_max - 0.9995, n_bins)
        estimates = (10**density(x_lin))
        #x_log = np.logspace(min_x, dist_max, n_bins)
        x_log = np.logspace(min_x, np.log10((dist_max-dist_min)*10), n_bins)
        #ax_raw.plot(x_log, estimates, c=color)
        ax_raw.hist(raw_mean_e_dists[slice_n][i] - dist_min - x_shift, bins=x_log, alpha=0.5, color=color)
    
        dist_min = normed_mean_e_dists[slice_n][i].min() - x_shift
        dist_max = normed_mean_e_dists[slice_n][i].max() - x_shift
        log_distribution = np.log10(normed_mean_e_dists[slice_n][i] - dist_min)
        density = gaussian_kde(log_distribution, bw_method=0.5)
        estimates = (10**density(x_lin))
        x_log = np.logspace(min_x, np.log10((dist_max-dist_min)*10), n_bins)
        #ax_norm.plot(x_log, estimates, c=color)
        ax_norm.hist(normed_mean_e_dists[slice_n][i] - dist_min, bins=x_log, alpha=0.5, color=color)

    ax_raw.loglog()
    ax_raw.set_xlabel("(mean hit energy in layer)/(events that intersect layer)")
    ax_raw.set_ylabel(f"Density, layer {slice_n}")
    ax_norm.loglog()
    ax_norm.set_xlabel("normed (mean hit energy in layer)/(events that intersect layer)")
# Again, not super convincing, but it will have to do.
# 
# ## Standard devation of hit energy in event
# 
# There are two remaining distributions to be fit;
# 
# - The standard devation of hit energy for each event.
# - The radial distribution of the hits.
# 
# The next step will be the standard devation of the hit energy, leaving the radial distribution till last.
# By now, we have a good process for generating the right fits.


inci_bins = 0.5*(acc.incident_bin_boundaries[:-1] + acc.incident_bin_boundaries[1:])
layer_bins = acc.layer_bottom
events_per_bin = acc.total_events[1:-2]
atleast_three = events_per_bin>3

pnt_mean_E_sq = acc.pnt_mean_E_sq_hist[1:-1]/events_per_bin
mean_pnt_E_sq = acc.evt_mean_E_sq_hist[1:-1]/events_per_bin
std_point_E = np.sqrt(pnt_mean_E_sq - mean_pnt_E_sq)

plt.imshow(std_point_E)
plt.title("Measured")
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
cbar = plt.colorbar()
cbar.set_label("Standard devation of energy in event")
plt.tight_layout()


n_rows, n_cols = std_point_E.shape
fig, (row_ax, col_ax) = plt.subplots(1, 2, figsize=(10, 4))
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, colour in enumerate(colours):
    row_ax.plot(layer_bins, std_point_E[row], c=colour)
row_ax.set_xlabel("Distance to center")
row_ax.set_ylabel("Standard devation of energy in event")
abs_to_center = np.abs(layer_bins)
order_abs_to_center = np.argsort(abs_to_center)
colours = plt.cm.inferno(np.linspace(0, 1, n_cols))
for col, colour in zip(order_abs_to_center, colours):
    ls = '--' if layer_bins[col] < 0. else '-'
    col_ax.plot(inci_bins, std_point_E[:, col], c=colour, ls=ls)
col_ax.set_xlabel("Incident energy")
col_ax.set_ylabel("Standard devation of energy in event")
# While it looks well like this can be treated like n-pts, linear fits to incident energy then a distance to center prediction, this won't work quite the same. The distribution is clearly not a gaussian.



#TOP

evts_per_inci_bin = np.nansum(events_per_bin, axis=1)
inci_bin_weights = evts_per_inci_bin/np.sum(evts_per_inci_bin)
stdEevt_per_inci_bin = np.nansum(std_point_E * inci_bin_weights[:, None], axis=1)
stdE_inci_linear_fit = np.polyfit(inci_bins, stdEevt_per_inci_bin, 1)

stdE_inci_scalings = np.polyval(stdE_inci_linear_fit, inci_bins)
scaled_stdE_per_event = std_point_E/stdE_inci_scalings[:, np.newaxis]
scaled_mean_stdE_per_layer = np.nanmean(scaled_stdE_per_event, axis=0)

# mask out the odd values when makeing the fits
mask = (layer_bins > -1) * (layer_bins < 1)
masked_layer_bins = layer_bins[mask]
# decide on a level of lift and the height that normalises the ys
masked_ys = scaled_mean_stdE_per_layer[mask]
masked_ys = np.nan_to_num(masked_ys)


stdE_dist_mean = np.nansum(masked_ys*masked_layer_bins)/np.nansum(masked_ys)
non_zero_weights = np.nansum(masked_ys>0)
stdE_dist_varience = (non_zero_weights/(non_zero_weights-1))*np.nansum(masked_ys*(masked_layer_bins-mean)**2)/np.nansum(masked_ys)

stdE_gumbel_beta = np.sqrt(6*stdE_dist_varience)/np.pi
stdE_gumbel_mu = stdE_dist_mean - stdE_gumbel_beta*np.euler_gamma


def gumble(xs, mu, beta, height, lift):
    z = (xs - mu)/beta
    return lift + height*np.exp(-z - np.exp(-z))/beta

p0 = [stdE_gumbel_mu, stdE_gumbel_beta, 1., 0]
bounds = [[-10, 0., 0., -5.], [10., 10., 100., 0.3]]
stdE_gumble_params, _ = curve_fit(gumble, masked_layer_bins, masked_ys, p0=p0, bounds=bounds, n_attempts=5, quiet=True)
stdE_gumbel_mu, stdE_gumbel_beta, stdE_gumbel_height, stdE_gumbel_lift = stdE_gumble_params

predicted_stdE_per_event = gumble(layer_bins, stdE_gumbel_mu, stdE_gumbel_beta, stdE_gumbel_height, stdE_gumbel_lift)


predicted_bins = np.tile(predicted_stdE_per_event, (counts_per_event.shape[0], 1))
predicted_bins *= stdE_inci_scalings[:, None]


n_rows, n_cols = scaled_stdE_per_event.shape
fig, (row_ax, col_ax) = plt.subplots(1, 2, figsize=(10, 4))
colours = plt.cm.inferno(np.linspace(0, 1, n_rows))
for row, colour in enumerate(colours):
    row_ax.plot(layer_bins, scaled_stdE_per_event[row], c=colour)
    pass
row_ax.plot(masked_layer_bins, masked_ys, c="blue", lw=10, alpha=0.2)
row_ax.plot(layer_bins, scaled_mean_stdE_per_layer, c="blue", lw=10, alpha=0.2)
row_ax.plot(layer_bins, predicted_stdE_per_event, c="blue", ls='--', lw=5, alpha=0.5)
row_ax.set_xlabel("Distance to center")
row_ax.set_ylabel("rescaled std energy in event")


abs_to_center = np.abs(layer_bins)
order_abs_to_center = np.argsort(abs_to_center)
colours = plt.cm.inferno(np.linspace(0, 1, n_cols))
for col, colour in zip(order_abs_to_center, colours):
    ls = '--' if layer_bins[col] < 0 else '-'
    col_ax.plot(inci_bins, std_point_E[:, col], c=colour, ls=ls)
    col_ax.plot(inci_bins, predicted_bins[:, col], c=colour, ls=ls, lw=5, alpha=0.5)
col_ax.set_xlabel("Incident energy")
col_ax.set_ylabel("std energy in event")


fig, ax = plt.subplots()
plt.title("Predictions")
plt.imshow(predicted_bins)
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
cbar = plt.colorbar()
cbar.set_label("Mean std energy in event")
plt.tight_layout()


fig, ax = plt.subplots()
plt.title("Predictions - measured")
plt.imshow(predicted_bins-std_point_E)
fig = plt.gcf()
fig.set_size_inches((10, 3))
plt.xlabel("Distance to center")
x_ticks = [f"{b:.2}" for b in layer_bins]
x_locs = np.arange(len(layer_bins))
plt.xticks(x_locs[::5], x_ticks[::5])
plt.ylabel("Incident energy")
y_ticks = [f"{b:.0f}" for b in inci_bins]
y_locs = np.arange(len(inci_bins))-0.5
plt.yticks(y_locs[::5], y_ticks[::5])
cbar = plt.colorbar()
cbar.set_label("Mean std energy in event")
plt.tight_layout()
# Showing the shifted distributions hasn't been that helpful so far, so I will skip that.
# Let's just do the radial distribution, then put it all together.
# 
# ## Radial distribution
# 
# The radial distributions are complicated by two factors; 1 the gun isn't in the center, 2 the rectangular cells need a jacobean if we are 
# We will start with locating the center of the distribution. Then we will caculate the radial distance to each cell, and the jacobean at each cell. From this, we can make a radial distribution to fit.
summed_xy_hits = np.sum(acc.counts_hist, axis=(0, 1))
summed_x_hits = np.sum(acc.counts_hist, axis=(0, 1, 3))
summed_y_hits = np.sum(acc.counts_hist, axis=(0, 1, 2))
x_bin_centers, y_bin_centers, _ = acc._get_bin_centers()
bin_area = acc.lateral_bin_size**2
x_mean = np.sum(x_bin_centers*summed_x_hits)/np.sum(summed_x_hits)
y_mean = np.sum(y_bin_centers*summed_y_hits)/np.sum(summed_y_hits)
print(f"x mean = {x_mean}, y mean = {y_mean}")
y_mean = 0.05
shifted_y_bin_centers = y_bin_centers - y_mean
bin_radii = np.sqrt(x_bin_centers[:, None]**2 + shifted_y_bin_centers**2)
bin_jacobean = np.pi*bin_radii/bin_area
flat_radii = bin_radii.flatten()
all_counts = acc.counts_hist[1:-1]
flat_counts = all_counts.reshape(*all_counts.shape[:2], -1)
flat_counts_sq = acc.counts_sq_hist[1:-1].reshape(*all_counts.shape[:2], -1)
unnormed_flat_probs = bin_jacobean.flatten()*flat_counts
flat_probs = unnormed_flat_probs/np.sum(unnormed_flat_probs, axis=2)[:, :, np.newaxis]

fig, (counts_ax, ax) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Changing incident energy")
incident_flat_counts = np.sum(flat_counts, axis=1)
incident_flat_probs = bin_jacobean.flatten()*incident_flat_counts
# it's not possible to normalise this properly, as we don't have data to infinity
# but we should still divide through by the total counts, as it's all measured in the same area
incident_flat_probs /= np.sum(incident_flat_probs, axis=-1)[:, np.newaxis]
colours = plt.cm.magma(np.linspace(0.01, 0.99, len(incident_flat_probs)))
for i, c in enumerate(colours):
    counts_ax.scatter(flat_radii, incident_flat_counts[i], c=[c], alpha=0.2)
    ax.scatter(flat_radii, incident_flat_probs[i], c=[c], alpha=0.2)
ax.loglog()
counts_ax.loglog()
counts_ax.set_ylabel("Counts")
ax.set_ylabel("Probability")
counts_ax.set_xlabel("Radius")
ax.set_xlabel("Radius")

fig, (counts_ax, ax) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Changing shift from centerpoint")
layer_flat_counts = np.sum(flat_counts, axis=0)
# it's not possible to normalise this, as we don't have data to infinity
layer_flat_probs = bin_jacobean.flatten()*layer_flat_counts
# but we should still divide through by the total counts, as it's all measured in the same area
layer_flat_probs /= np.sum(layer_flat_probs, axis=-1)[:, np.newaxis]
unnormed_flat_sigma = np.sqrt(np.sum(flat_counts_sq - flat_counts**2, axis=0))
layer_flat_sigma = bin_jacobean.flatten()*unnormed_flat_sigma/np.sum(layer_flat_probs, axis=-1)[:, np.newaxis]
colours = plt.cm.viridis(np.linspace(0.01, 0.99, len(layer_flat_probs)))
for i, c in enumerate(colours):
    counts_ax.scatter(flat_radii, layer_flat_counts[i], c=[c], alpha=0.2)
    ax.scatter(flat_radii, layer_flat_probs[i], c=[c], alpha=0.2)
ax.loglog()
counts_ax.loglog()
counts_ax.set_xlabel("Radius")
ax.set_xlabel("Radius")
counts_ax.set_ylabel("Counts")
ax.set_ylabel("Probability")
# This distribution appears to be almost not energy dependent, but quite dependant on the distance from the center point.
# Typically, a two part distribution is used, one part models the center of the distribution, the other part models the tails.
# We can try the GFlash distribution, or an exponential decay. The exponential works better.
from pointcloud.models.custom_torch_distributions import GFlashRadial

def to_fit(xs, Rc, Rt_extend, p, normalisation):
    distribution = GFlashRadial(Rc, Rt_extend, p)
    found = distribution.prob(torch.Tensor(xs))
    return found.detach().numpy()*normalisation

def make_exponential(core, to_extention, p):
    core = np.atleast_1d(core)
    to_extention = np.atleast_1d(to_extention)
    p = np.atleast_1d(p)
    rates = np.vstack([1/core, 1/(core + to_extention)]).T
    factors = np.vstack([p, (1-p)]).T
    exponential = torch.distributions.Exponential(torch.Tensor(rates))
    mix = torch.distributions.Categorical(torch.Tensor(factors))
    combined = torch.distributions.MixtureSameFamily(mix, exponential)
    return combined
    

def to_fit(xs, core, to_extention, p, normalisation):
    distribution = make_exponential(core, to_extention, p)
    found = np.exp(distribution.log_prob(torch.Tensor(xs)))
    return found.detach().numpy()*normalisation

p0 = [0.01, 0.1, 0.5, 0.1]
bounds = [[1e-10, 1e-10, 1e-5, 0.], [10., 1000., 1-1e-5, 0.2]]
params_per_layer = np.zeros((len(layer_flat_probs), len(p0)))
for i, probs in enumerate(layer_flat_probs):
    print(f"{i/len(layer_flat_probs):.0%}", end='\r')
    if np.any(np.isnan(probs)):
        params_per_layer[i] = -1
        continue
    sigma = np.nan_to_num(layer_flat_sigma[i])
    sigma[sigma <=0 ] = 1.
    params, _ = curve_fit(to_fit, flat_radii, probs, sigma=sigma, bounds=bounds, p0=p0, n_attempts=200, quiet=True)
    
    params_per_layer[i] = params

standard_radii = np.linspace(0, np.max(flat_radii), 20)

fig, (data_ax, preds_ax) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Changing shift from centerpoint")
colours = plt.cm.viridis(np.linspace(0.01, 0.99, len(layer_flat_probs)))
for i, c in enumerate(colours):
    data_ax.scatter(flat_radii, layer_flat_probs[i], c=[c], alpha=0.2)
    params = params_per_layer[i]
    if np.any(params>0):
        predictions = to_fit(standard_radii, *params)
        preds_ax.plot(standard_radii, predictions, c=c)
data_ax.loglog()
preds_ax.loglog()
preds_ax.set_xlabel("Radius")
preds_ax.set_xlabel("Radius")
data_ax.set_ylabel("Probability")
preds_ax.set_ylabel("Predictions")
preds_ax.set_ylim(1e-7, 1.)


fig, (data_ax, preds_ax) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Changing shift from centerpoint")
colours = plt.cm.viridis(np.linspace(0.01, 0.99, len(layer_flat_probs)))
for i, c in enumerate(colours):
    data_ax.scatter(flat_radii, layer_flat_probs[i], c=[c], alpha=0.2)
    params = params_per_layer[i]
    if np.any(params>0):
        predictions = to_fit(standard_radii, *params)
        preds_ax.plot(standard_radii, predictions, c=c)
data_ax.loglog()
preds_ax.loglog()
preds_ax.set_xlabel("Radius")
preds_ax.set_xlabel("Radius")
data_ax.set_ylabel("Probability")
preds_ax.set_ylabel("Predictions")
preds_ax.set_ylim(1e-7, 1.)
data_ax.set_ylim(1e-7, 1.)

fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
for i, name in enumerate(["core", "to_extended", "p", "normalisation"]):
    ax = axarr.flatten()[i]
    ax.plot(layer_bins, params_per_layer[:, i])
    ax.set_xlabel("Distance to center")
    ax.set_ylabel(name)

#first_deriv = params_per_layer[1:] - params_per_layer[:-1]
#abs_second_deriv = np.abs(first_deriv[1:] - first_deriv[:-1])
#limit = np.mean(abs_second_deriv)/16
#over_limit = np.any(abs_second_deriv>limit, axis=1)
#bin_zero = np.argmin(np.abs(layer_bins))
#half_bins_width = np.min(np.abs(np.where(over_limit)[0] - bin_zero))
#
#print(f"Behavior is ok for {half_bins_width} bins either side of zero")
mask = np.arange(len(layer_bins))
#mask = (mask >= (bin_zero-half_bins_width-1))*(mask <= (bin_zero+half_bins_width+1))

n_radial_poly_coeffs = 2
fitted_radial_poly = np.zeros((params_per_layer.shape[1], n_radial_poly_coeffs+1))
masked_bins = layer_bins[mask]
for i, name in enumerate(["core", "to_extended", "p", "normalisation"]):
    ax = axarr.flatten()[i]
    params = params_per_layer[:, i]
    weights = np.ones_like(layer_bins)
    weights[~mask] = 0.01
    weights[params < 0] = 0
    polycoeff = np.polyfit(layer_bins, params, n_radial_poly_coeffs, w=weights)
    fitted_radial_poly[i] = polycoeff
    ax.scatter(layer_bins[mask][[0, -1]], params[mask][[0, -1]], c="blue", alpha=0.5, s=50)
    ax.set_ylim(np.min(params[mask*(params>0)])*0.5, np.max(params[mask*(params>0)])*1.5)

    predictions = np.polyval(polycoeff, layer_bins)
    ax.plot(layer_bins, predictions, lw=5, c='blue', alpha=0.5)


def draw_radial(distance_to_center, n_pts):
    distance_to_center = np.atleast_1d(distance_to_center)
    core = np.polyval(fitted_radial_poly[0], distance_to_center)
    to_extend = np.polyval(fitted_radial_poly[1], distance_to_center)
    p = np.polyval(fitted_radial_poly[2], distance_to_center)
    distributions = make_exponential(core, to_extend, p)
    if isinstance(n_pts, int):
        n_pts = torch.Size([n_pts])
    return distributions.sample(n_pts).T


def draw_xz(distance_to_center, n_pts):
    if isinstance(n_pts, int):
        n_pts = torch.Size([n_pts])
    radii = draw_radial(distance_to_center, n_pts)
    angular_dist = torch.distributions.Uniform(torch.tensor([-np.pi]), torch.tensor([np.pi]))
    angles = angular_dist.sample(n_pts).T
    xs = torch.cos(angles)*radii
    zs = torch.sin(angles)*radii - z_mean
    return xs, zs
    


xs, zs = draw_xz(layer_bins, 100000)


n_slices = 15
slices = np.linspace(0, len(layer_bins), n_slices).astype(int)
from matplotlib.colors import LogNorm
norm = LogNorm(vmin=1e-7, vmax=1e-2)
fig, axarr = plt.subplots(n_slices, 3, figsize=(10, n_slices*4))
for n, (ax_real, ax, diff_ax) in enumerate(axarr):
    hist, xedges, yedges = np.histogram2d(xs[n], zs[n], bins=[acc.lateral_x_bin_boundaries, acc.lateral_z_bin_boundaries])
    hist /= np.sum(hist)
    ax.imshow(hist, norm=norm)
    real_counts = np.sum(acc.counts_hist, axis=0)[n]
    if np.sum(real_counts):
        real_counts /= np.sum(real_counts)
    ax_real.imshow(real_counts, norm=norm)
    diff_ax.imshow(np.abs(hist-real_counts), norm=norm)
    

xs.shape


