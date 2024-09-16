import sys
import os
import numpy as np

name_change_dict = {
    "gun_xyz_pos": "gun_xyz_pos_global",
    "gun_xz_pos_raw": "gun_xyz_pos_hdf5",
    "layer_bottom_pos": "layer_bottom_pos_global",
    "layer_bottom_pos_raw": "layer_bottom_pos_hdf5",
    "Ymin": "Ymin_global",
    "Xmin": "Xmin_global",
    "Xmax": "Xmax_global",
    "Zmin": "Zmin_global",
    "Zmax": "Zmax_global",
    "half_cell_size": "half_cell_size_global",
    "half_cell_size_raw": "half_cell_size_hdf5",
    "cell_thickness": "cell_thickness_global",
    "cell_thickness_raw": "cell_thickness_hdf5",
}


def update(metadata_folder):
    # read the content
    shallow_content = {}
    deep_content = {}
    for file in os.listdir(metadata_folder):
        if not file.endswith(".npy"):
            continue
        content = np.load(os.path.join(metadata_folder, file), allow_pickle=True)
        if content.dtype == "O":
            deep_content[file] = content.item()
        else:
            shallow_content[file] = content

    # name updates
    new_shallow_content = {}
    new_deep_content = {}
    for key, value in shallow_content.items():
        base_name = key[:-4]
        new_key = name_change_dict.get(base_name, base_name) + ".npy"
        new_shallow_content[new_key] = value

    for key, file_dict in deep_content.items():
        new_file_dict = {}
        for base_name, value in file_dict.items():
            new_base_name = name_change_dict.get(base_name, base_name)
            new_file_dict[new_base_name] = value
        if key == "cell_dimensions.npy":
            key = "cell_dimensions_global.npy"
        if key == "cell_dimensions_raw.npy":
            key = "cell_dimensions_hdf5.npy"
        new_deep_content[key] = new_file_dict

    if "downsample_settings.npy" in deep_content:
        ori = deep_content["downsample_settings.npy"]["local_xyz_orientation"]
        if isinstance(ori, str):
            ori = False
        new_deep_content["downsample_settings.npy"]["local_xyz_orientation"] = ori
        if ori:
            new_shallow_content["orientation_global.npy"] = "hdf5:xyz==global:zxy"
            new_shallow_content["orientation.npy"] = "hdf5:xyz==local:xyz"
        else:
            new_shallow_content["orientation_global.npy"] = "hdf5:xyz==global:xyz"
            new_shallow_content["orientation.npy"] = "hdf5:xyz==local:zxy"
    else:
        shower_axis = 2

    old_pos = new_shallow_content["gun_xyz_pos_global.npy"]
    if len(old_pos) == 2:
        new_pos = [old_pos[0], old_pos[1]]
        new_pos.insert(shower_axis, 0.0)
        new_shallow_content["gun_xyz_pos_global.npy"] = new_pos

    # now save the new content
    for key, value in new_shallow_content.items():
        np.save(os.path.join(metadata_folder, key), value)
    for key, file_dict in new_deep_content.items():
        np.save(os.path.join(metadata_folder, key), file_dict, allow_pickle=True)

    for key in shallow_content:
        if key not in new_shallow_content:
            os.remove(os.path.join(metadata_folder, key))

    for key in deep_content:
        if key not in new_deep_content:
            os.remove(os.path.join(metadata_folder, key))

if __name__ == "__main__":
    for path in sys.argv[1:]:
        update(path)
