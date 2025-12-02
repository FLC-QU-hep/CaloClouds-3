"""
Example of creating metadata for a new dataset.
You need to change the values to match your dataset. 
Comment out anything you aren't sure about.
"""
import numpy as np

metadata_folder = "pointcloud/metadata/sim-E1261AT600AP180-180_file_W"

# saved loose
np.save(f"{metadata_folder}/gun_xyz_pos_global.npy", [0.0, 18047.0, -50])
np.save(f"{metadata_folder}/gun_xyz_pos_hdf5.npy", [0.0, 0.0, 0.0])
layer_bottoms = np.load(
    f"{metadata_folder}/../"
    "10-90GeV_highGran_fixedAng_05.2024/layer_bottom_pos_global.npy"
)
np.save(f"{metadata_folder}/layer_bottom_pos_global.npy", layer_bottoms)
np.save(f"{metadata_folder}/layer_bottom_pos_hdf5.npy", np.linspace(0, 29, 30))

# saved in dicts
cube_bounds = {
    "Ymin_global": 1811,
    "Xmin_global": -250,
    "Xmax_global": 250,
    "Zmin_global": -250,
    "Zmax_global": 250,
}
np.save(f"{metadata_folder}/box_edges.npy", cube_bounds, allow_pickle=True)
cell_dims = np.load(
    f"{metadata_folder}/../"
    "10-90GeV_highGran_fixedAng_05.2024/cell_dimensions_global.npy",
    allow_pickle=True,
)
np.save(f"{metadata_folder}/cell_dimensions_global.npy", cell_dims, allow_pickle=True)

# we don't know the hdf5 dimensions really. See if we need to add them later

np.save(
    f"{metadata_folder}/downsample_settings.npy",
    {
        "all_steps": False,
        "dm": 5,
        "sort": False,
        "aligne": True,
        "local_xyz_orientaion": True,
    },
    allow_pickle=True,
)

np.save(
    f"{metadata_folder}/rescales.npy",
    {"incident_rescale": 127, "vis_eng_rescale": 3.4, "n_pts_rescale": 7864},
    allow_pickle=True,
)

np.save(f"{metadata_folder}/orientation.npy", "hdf5:xyz==local:xyz")
np.save(f"{metadata_folder}/orientation_global.npy", "hdf5:xyz==global:yzx")
