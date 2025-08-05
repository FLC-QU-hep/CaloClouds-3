""" Module to mock functions in read_write.py for testing purposes. """

import numpy as np
from pointcloud.utils import metadata


def mock_get_n_events(path, n_files):
    return [3]


def mock_read_raw_regaxes(config, idxs):
    meta = metadata.Metadata(config)
    z_in_layers = meta.layer_bottom_pos_hdf5 + meta.cell_thickness_hdf5 / 2

    incident_energies = np.array([1.0, 2.0, 3.0])
    coordinates = np.zeros((len(incident_energies), 10, 4))

    coordinates[0, :3, 2] = z_in_layers[[0, 1, 1]]
    coordinates[0, :3][:, [0, 1, 3]] = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 0.0, 3.0]]
    )

    coordinates[1, :4, 2] = z_in_layers[[0, 1, 1, 2]]
    coordinates[1, :4][:, [0, 1, 3]] = np.array(
        [[0.0, 0.0, 2.0], [0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
    )

    return incident_energies, coordinates
