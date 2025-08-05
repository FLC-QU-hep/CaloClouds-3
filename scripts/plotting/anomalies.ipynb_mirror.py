import numpy as np
from pointcloud.config_varients.wish import Configs
from pointcloud.data import trees
from pointcloud.anomalies.detect import detect, save_path_for_model
from matplotlib import pyplot as plt
from IPython.display import display_markdown
import matplotlib
import os

config = Configs()
matplotlib.rcParams["figure.figsize"] = 5, 5

model_path = (
    "/home/dayhallh/training/point-cloud-diffusion-logs/anomalies/24-05-27.23_13496.pt"
)
scores_path = save_path_for_model(model_path)

recalc_scores = False
if not os.path.exists(scores_path) or recalc_scores:
    detect(model_path, config)
scores = np.load(scores_path)

dat = trees.DataAsTrees(config)
plot_args = {"xlims": (-150, 150), "ylims": (-200, 100), "zlims": (-2, 32)}


# # displaying trees
#
# Just show a single tree to check the axis ranges.
# Also, define the ability to make a double plot and show it.
plot_num = 0

plots_folder = os.path.join(config.image_dir, "anomaly_nb")
if not os.path.exists(plots_folder):
    os.mkdir(plots_folder)


def make_double(i):
    plt.close()
    score = scores[i, 1]
    points_only = os.path.join(plots_folder, f"points_evt{i}.jpg")
    points_only = f"points_evt{i}.jpg"
    fig, ax = trees.plot_tree(dat, i, line_alpha=0, show=False, **plot_args)
    ax.set_title(f"Event {i}, Anomaly score {score:.4g}")
    plt.tight_layout()
    plt.savefig(points_only)
    plt.close()
    with_lines = os.path.join(plots_folder, f"lines_evt{i}.jpg")
    with_lines = f"lines_evt{i}.jpg"
    fig, ax = trees.plot_tree(dat, i, line_alpha=0.6, show=False, **plot_args)
    ax.set_title(f"Event {i}, Anomaly score {score:.4g}")
    plt.tight_layout()
    plt.savefig(with_lines)
    plt.close()
    return f"![Points]({points_only})  ![Lines]({with_lines})"


display_markdown(make_double(0), raw=True)
from matplotlib import pyplot as plt

hist = plt.hist(scores[:, 1], bins=50)
plt.semilogy()
plt.xlabel("Reconstruction difficulty")
plt.ylabel("N showers")
high_anomaly = np.argsort(scores[:, 1])[-10:]
low_anomaly = np.argsort(scores[:, 1])[:10]
print((high_anomaly, low_anomaly))
for i in high_anomaly:
    display_markdown(make_double(i), raw=True)
for i in low_anomaly:
    display_markdown(make_double(i), raw=True)
from pointcloud.data.read_write import read_raw_regaxes

pick_events = np.arange(5000)

incidents, events = read_raw_regaxes(config, pick_events=pick_events)


xs = np.linspace(10, 90, 50)
c_hits = 190
m_hits = 35
ys_hits = xs * m_hits + c_hits
c_energy = 0
m_energy = 0.015
ys_energy = xs * m_energy + c_energy

import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

sum_data = {"incident energy": incidents, "idx": list(range(len(incidents)))}

# energy
energies = events[:, :, 3]

sum_data["energy"] = np.sum(energies[:, :], axis=1)

# hits
sum_data["hits"] = np.sum(energies[:, :] > 0, axis=1)

event_scores = np.log(scores[pick_events][:, 1])
sum_data["anomaly_score"] = event_scores / np.max(event_scores) - 0.5

for k in sum_data:
    print((k, len(sum_data[k])))

sum_dataframe = pd.DataFrame(data=sum_data)

fig = make_subplots(rows=1, cols=2)
fig.update_layout(plot_bgcolor="black")
sub_fig_e = px.scatter(
    sum_dataframe,
    x="incident energy",
    y=["energy", "energy"],
    color="anomaly_score",  # "idx",
    color_continuous_scale=px.colors.diverging.balance,
    opacity=0.8,
)
sub_fig_hits = px.scatter(
    sum_dataframe,
    x="incident energy",
    y=["hits", "hits"],
    color="anomaly_score",
    color_continuous_scale=px.colors.sequential.Blackbody,
    opacity=0.8,
)

sub_fig_e.add_scatter(x=xs, y=ys_energy, fillcolor="red", name="cutoff")
sub_fig_hits.add_scatter(x=xs, y=ys_hits, fillcolor="red", name="cutoff")
# add each trace (or traces) to its specific subplot
for i in sub_fig_e.data:
    fig.add_trace(i, row=1, col=1)

for i in sub_fig_hits.data:
    fig.add_trace(i, row=1, col=2)
fig["layout"]["xaxis"]["title"] = "Incident energy"
fig["layout"]["yaxis"]["title"] = "Observed energy"
fig["layout"]["xaxis2"]["title"] = "Incident energy"
fig["layout"]["yaxis2"]["title"] = "Observed hits"
import plotly.io as pio

pio.renderers.default = "iframe"

fig.show()
down = 1.0
below_hits = sum_data["hits"] < (incidents * m_hits + c_hits) * down
below_hits = np.where(below_hits)[0]
print(below_hits)
below_energy = sum_data["energy"] < (incidents * m_energy + c_energy) * down
below_energy = np.where(below_energy)[0]
print(below_energy)

below_both = [i for i in below_hits if i in below_energy]
only_e = [i for i in below_energy if i not in below_hits]
print(only_e)
only_h = [i for i in below_hits if i not in below_energy]
print(only_h)
for i in below_both:
    display_markdown(make_double(i), raw=True)
for i in only_e:
    display_markdown(make_double(i), raw=True)
for i in only_h:
    display_markdown(make_double(i), raw=True)
# ## MC particles
#
# Now we look at the MC particles, and see how the rares corrispond to anomaly scores.
from particle import Particle

mc_data_file = "/home/dayhallh/Data/p22_th90_ph90_en10-100_MCParticles.npz"
data = np.load(mc_data_file)
indices = data["idx"]
pdg = data["PDG"]
pdg.shape
set_pdg = sorted(set(pdg.flatten()))
set_abs = sorted(set(np.abs(pdg.flatten())))
print(
    f"The number of PDG id's in the data is {len(set_pdg)}, and if we ignore the difference between particle and antiparticle {len(set_abs)}"
)
count_abs = {p: np.sum([p in row for row in np.abs(pdg)]) for p in set_abs}
sorted_abs = sorted(set_abs, key=lambda p: count_abs[p])
sorted_abs = [
    [p, Particle.from_pdgid(p).name, count_abs[p]] for p in sorted_abs if p != 0
]
print(sorted_abs)

hyper_rare = [a[0] for a in sorted_abs if a[2] < 10]
hyper_rare_label = "/".join([a[1] for a in sorted_abs if a[2] < 10])
print(f"The hyper rare particles are {hyper_rare_label}")
rare = [a[0] for a in sorted_abs if a[0] not in hyper_rare and a[2] < 500]
print(f"The rare particles are {' '.join([a[1] for a in sorted_abs if a[0] in rare])}")
hyper_rare_scores = []
rare_scores = []
n_scores = []
p_scores = []
for i, ps in zip(indices, pdg):
    if np.any([p in ps for p in hyper_rare]):
        hyper_rare_scores.append(scores[i, 1])
    if np.any([p in ps for p in rare]):
        rare_scores.append(scores[i, 1])
    if 2112 in ps:
        n_scores.append(scores[i, 1])
    if 2212 in ps:
        p_scores.append(scores[i, 1])


# Lets see how the anomaly scores of these events are distrbuted
from matplotlib import pyplot as plt

n, bins, patches = plt.hist(scores[:, 1], alpha=0.5, bins=50, label="All events")
plt.hist(rare_scores, bins=bins, label="Events with rare MC", histtype="step")
plt.hist(
    hyper_rare_scores, bins=bins, label="Events with hyper rare MC", histtype="step"
)
plt.hist(n_scores, bins=bins, label="Events with a neutron in", histtype="step")
plt.hist(p_scores, bins=bins, label="Events with a proton in", histtype="step")
plt.legend()
plt.semilogy()
plt.xlabel("Reconstruction difficulty")
plt.ylabel("N showers")
[i in indices for i in high_anomaly]
len(scores)

high_anomaly
