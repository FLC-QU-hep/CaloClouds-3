import os
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from pointcloud.utils.stats_accumulator import StatsAccumulator
from pointcloud.utils.plotting import heatmap


plt.ioff()

# acc = StatsAccumulator.load(
#    "../point-cloud-diffusion-logs/wish/dataset_accumulators"
#    "/10-90GeV_x36_grid_regular_524k_float32/from_10.h5"
# )
# acc = StatsAccumulator.load("../point-cloud-diffusion-logs/"
#                             "wish/dataset_accumulators/wish_v4.h5")
# acc = StatsAccumulator.load(
#    "../point-cloud-diffusion-logs/wish/dataset_accumulators"
#    "/sim-photon-showers_10-90GeV_Zpos4_validation_merge.h5"
# )
acc = StatsAccumulator.load(
    "../point-cloud-diffusion-logs/wish/dataset_accumulators"
    # "../local_data"
    "/p22_th90_ph90_en10-100_accumulator.h5"
    #    #"/sim-photon-showers_10-90GeV_Zpos4_validation_102_all_steps.h5"
)
hists_2D = "evt_mean_E", "evt_mean_E_sq", "pnt_mean_E_sq"

incident_lims = 10, 90
# there is a buffer length 1 each side of the incident_bin_boundaries
first_bin = next(
    i + 1 for i, b in enumerate(acc.incident_bin_boundaries) if b >= incident_lims[0]
)
last_bin = -next(
    i + 2
    for i, b in enumerate(acc.incident_bin_boundaries[::-1])
    if b <= incident_lims[-1]
)
last_bin = len(acc.incident_bin_boundaries) + 1 + last_bin

xs = acc.incident_bin_boundaries[first_bin], acc.incident_bin_boundaries[last_bin]
zs = acc.layer_bottom[0], acc.layer_bottom[-1] + acc.cell_thickness_global


def save_ensure_dir_exists(file_name):
    dir_name = os.path.dirname(file_name)
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    plt.savefig(file_name + "png")
    plt.savefig(file_name + "pdf")


def expand_name(name):
    return (
        name.replace("_", " ")
        .replace("evt", "event")
        .replace("E", "Energy (per evt)")
        .replace("sq", "squared")
    )


n_events = acc.total_events[first_bin:last_bin, np.newaxis]

print("Doing 2D")
for name in hists_2D:
    plt.close()
    arr = getattr(acc, name + "_hist")[first_bin:last_bin] / n_events

    heatmap(
        arr, xs, zs, "Incident Energy", "Z (layers)", expand_name(name), cmap="viridis"
    )
    plt.tight_layout()
    save_ensure_dir_exists(
        f"../point-cloud-diffusion-images/stats_heatmap_wishv5/{name}."
    )

plt.close()

hists_4D = "counts", "energy", "evt_mean_counts"

xs = acc.Xmin - acc.lateral_bin_size * 0.5, acc.Xmax + acc.lateral_bin_size * 0.5
ys = acc.Ymin - acc.lateral_bin_size * 0.5, acc.Ymax + acc.lateral_bin_size * 0.5

n_events = acc.total_events[:, np.newaxis, np.newaxis, np.newaxis]
mask = n_events <= 0
n_events[mask] = 1.0

print("Doing 4D")
for name in hists_4D:
    print(name)
    hist = getattr(acc, name + "_hist") / n_events
    vmax_global = hist.max()
    for incident_bin in range(first_bin, last_bin):
        print(f"{(incident_bin - first_bin)/(last_bin-first_bin):.1%}", end="\r")
        hist_inci = hist[incident_bin]
        vmax_inci = hist_inci.max()
        incident_energy_range = (
            acc.incident_bin_boundaries[incident_bin],
            acc.incident_bin_boundaries[incident_bin + 1],
        )
        incident_energy_txt = (
            f"Incidient energy {incident_energy_range[0]}-{incident_energy_range[1]}"
        )
        for layer in range(acc.n_layers):
            plt.close()
            arr = hist[incident_bin, layer]
            heatmap(
                arr,
                xs,
                ys,
                "X",
                "Y",
                expand_name(name),
                cmap="viridis",
                vmin=0.0,
                vmax=vmax_global,
            )
            plt.tight_layout()
            save_ensure_dir_exists(
                f"../point-cloud-diffusion-images/stats_heatmap_wishv5/{name}_layers"
                + f"/global/E{incident_energy_range[0]:02}_layer{layer:02}_global."
            )

            plt.close()
            heatmap(
                arr,
                xs,
                ys,
                "X",
                "Y",
                expand_name(name),
                cmap="viridis",
                vmin=0.0,
                vmax=vmax_inci,
            )
            plt.tight_layout()
            save_ensure_dir_exists(
                f"../point-cloud-diffusion-images/stats_heatmap_wishv5/{name}_layers/"
                + f"perInci/E{incident_energy_range[0]:02}_layer{layer:02}_perInci."
            )

            plt.close()
            heatmap(arr, xs, ys, "X", "Y", expand_name(name), cmap="viridis")
            plt.tight_layout()
            save_ensure_dir_exists(
                f"../point-cloud-diffusion-images/stats_heatmap_wishv5/{name}_layers/"
                + f"local/E{incident_energy_range[0]:02}_layer{layer:02}_local."
            )


print("Doing point energy")
name = "point_energy"
hist = acc.energy_hist
devisor = acc.counts_hist
mask = devisor > 0
hist[mask] /= devisor[mask]
vmax_global = hist.max()
for incident_bin in range(first_bin, last_bin):
    print(f"{(incident_bin - first_bin)/(last_bin-first_bin):.1%}", end="\r")
    hist_inci = hist[incident_bin]
    vmax_inci = hist_inci.max()
    incident_energy_range = (
        acc.incident_bin_boundaries[incident_bin],
        acc.incident_bin_boundaries[incident_bin + 1],
    )
    incident_energy_txt = (
        f"Incidient energy {incident_energy_range[0]}-{incident_energy_range[1]}"
    )
    for layer in range(acc.n_layers):
        plt.close()
        arr = hist[incident_bin, layer]
        heatmap(
            arr,
            xs,
            ys,
            "X",
            "Y",
            expand_name(name),
            cmap="viridis",
            vmin=0.0,
            vmax=vmax_global,
        )
        plt.tight_layout()
        save_ensure_dir_exists(
            f"../point-cloud-diffusion-images/stats_heatmap_wishv5/{name}_layers/"
            + f"global/E{incident_energy_range[0]:02}_layer{layer:02}_global."
        )

        plt.close()
        heatmap(
            arr,
            xs,
            ys,
            "X",
            "Y",
            expand_name(name),
            cmap="viridis",
            vmin=0.0,
            vmax=vmax_inci,
        )
        plt.tight_layout()
        save_ensure_dir_exists(
            f"../point-cloud-diffusion-images/stats_heatmap_wishv5/{name}_layers/"
            + f"perInci/E{incident_energy_range[0]:02}_layer{layer:02}_perInci."
        )

        plt.close()
        heatmap(arr, xs, ys, "X", "Y", expand_name(name), cmap="viridis")
        plt.tight_layout()
        save_ensure_dir_exists(
            f"../point-cloud-diffusion-images/stats_heatmap_wishv5/{name}_layers/"
            + f"local/E{incident_energy_range[0]:02}_layer{layer:02}_local."
        )
