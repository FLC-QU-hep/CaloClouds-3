"""
Example of creating metadata for a new dataset.
You need to change the values to match your dataset. 
Comment out anything you aren't sure about.
"""

import h5py
import numpy as np
import os
from pointcloud.config_varients import sepPhotonData as my_conf

conf = my_conf.Configs()
metadata_folder = os.path.join("pointcloud/metadata", conf.metadata_folder)
if not os.path.exists(metadata_folder):
    os.mkdir(metadata_folder)

# saved loose
np.save(f"{metadata_folder}/gun_xyz_pos_global.npy", [0.0, 18047.0, -50])
np.save(f"{metadata_folder}/gun_xyz_pos_hdf5.npy", [0.0, 18047.0, -50])
# np.save(f"{metadata_folder}/gun_xyz_pos_hdf5.npy", [0.0, 0.0, 0.0])
layer_bottoms = np.load(
    f"{metadata_folder}/../"
    "10-90GeV_highGran_fixedAng_05.2024/layer_bottom_pos_global.npy"
)
np.save(f"{metadata_folder}/layer_bottom_pos_global.npy", layer_bottoms)
is_descretised = False
if is_descretised:
    np.save(f"{metadata_folder}/layer_bottom_pos_hdf5.npy", np.linspace(0, 29, 30))
else:
    np.save(f"{metadata_folder}/layer_bottom_pos_hdf5.npy", layer_bottoms)

# saved in dicts
cube_bounds = {
    # for CC3
    "Xmin_global": -250,
    "Xmax_global": 250,
    "Zmin_global": -250,
    "Zmax_global": 250,
    "Ymin_global": 0,
}
np.save(f"{metadata_folder}/box_edges.npy", cube_bounds, allow_pickle=True)
cell_dims = np.load(
    f"{metadata_folder}/../"
    "10-90GeV_highGran_fixedAng_05.2024/cell_dimensions_global.npy",
    allow_pickle=True,
).item()
if is_descretised:
    cell_dims["cell_thickness_hdf5"] = 0.5
else:
    cell_dims["cell_thickness_hdf5"] = cell_dims["cell_thickness_global"]
np.save(f"{metadata_folder}/cell_dimensions_global.npy", cell_dims, allow_pickle=True)

# we don't know the hdf5 dimensions really. See if we need to add them later

np.save(
    f"{metadata_folder}/downsample_settings.npy",
    {
        # "all_steps": False,
        "all_steps": True,
        # "dm": 5,
        "dm": -1,  # no downsample
        "sort": False,
        # "aligne": True,
        "aligne": False,
        # "local_xyz_orientaion": True,
        "local_xyz_orientaion": False,
    },
    allow_pickle=True,
)

with h5py.File(conf.dataset_path.format("0"), "r") as f:
    points = f["events"]
    es = points[:, :, 3]
    # Note that this is gotta be in shower-local coords
    xs_shower_coords = points[:, :, 2]
    ys_shower_coords = points[:, :, 0]
    zs_shower_coords = points[:, :, 1]

xs_cog = np.sum(xs_shower_coords * es, axis=1) / np.sum(es, axis=1)
ys_cog = np.sum(ys_shower_coords * es, axis=1) / np.sum(es, axis=1)
zs_cog = np.sum(zs_shower_coords * es, axis=1) / np.sum(es, axis=1)

mask = es > 0
points_per_event = np.sum(mask, axis=1)
sum_xs = np.sum(xs_shower_coords * mask, axis=1)
mean_xs = sum_xs / points_per_event
std_xs = np.mean(
    np.sqrt(
        np.sum((xs_shower_coords - mean_xs[:, None]) ** 2 * mask, axis=1)
        / points_per_event
    )
)
mean_xs = np.mean(mean_xs)
sum_ys = np.sum(ys_shower_coords * mask, axis=1)
mean_ys = sum_ys / points_per_event
std_ys = np.mean(
    np.sqrt(
        np.sum((ys_shower_coords - mean_ys[:, None]) ** 2 * mask, axis=1)
        / points_per_event
    )
)
mean_ys = np.mean(mean_ys)
sum_zs = np.sum(zs_shower_coords * mask, axis=1)
mean_zs = sum_zs / points_per_event
std_zs = np.mean(
    np.sqrt(
        np.sum((zs_shower_coords - mean_zs[:, None]) ** 2 * mask, axis=1)
        / points_per_event
    )
)
mean_zs = np.mean(mean_zs)

es[es <= 0 ] = 1
log_es = np.log(es)
sum_log_es = np.sum(log_es * mask, axis=1)
mean_log_es = sum_log_es / points_per_event
std_log_es = np.mean(
    np.sqrt(
        np.sum((log_es - mean_log_es[:, None]) ** 2 * mask, axis=1) / points_per_event
    )
)
mean_log_es = np.mean(log_es)

mean_cog = np.mean([xs_cog, ys_cog, zs_cog], axis=1)
std_cog = np.std([xs_cog, ys_cog, zs_cog], axis=1)


np.save(
    f"{metadata_folder}/rescales.npy",
    # CC3 values
    {
        "incident_rescale": 127,
        "vis_eng_rescale": 3.4,
        "n_pts_rescale": 7864,
        "mean_cog": mean_cog,
        "std_cog": std_cog,
        "mean_xs": mean_xs,
        "std_xs": std_xs,
        "mean_ys": mean_ys,
        "std_ys": std_ys,
        "mean_zs": mean_zs,
        "std_zs": std_zs,
        "mean_log_es": mean_log_es,
        "std_log_es": std_log_es,
    },
    allow_pickle=True,
)

np.save(f"{metadata_folder}/orientation.npy", "hdf5:xyz==local:yzx")
np.save(f"{metadata_folder}/orientation_global.npy", "hdf5:xyz==global:xyz")
# np.save(f"{metadata_folder}/orientation.npy", "hdf5:xyz==local:xyz")
# np.save(f"{metadata_folder}/orientation_global.npy", "hdf5:xyz==global:yzx")
