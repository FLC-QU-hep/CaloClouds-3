import numpy as np

from pointcloud.utils import stats_accumulator


def make(add_varients=False, start_acc=None):
    if start_acc is None:
        acc = stats_accumulator.StatsAccumulator()
    else:
        acc = start_acc
    if add_varients:
        energies = [0.5, 1, 1.5]
    else:
        energies = [1]

    incident_mid_points = (
        acc.incident_bin_boundaries[:-1] + acc.incident_energy_bin_size / 2
    )
    n_incidents = len(incident_mid_points)
    n_energies = len(energies)

    for e, energy in enumerate(energies):

        def get_points_for_layer(layer_n):
            layer_z = acc.layer_bottom[layer_n] + acc.cell_thickness / 2
            points_in_layer = np.array(
                [
                    [0.0, 0.0, layer_z, energy],
                    [0.0, 0.0, layer_z, energy],
                    [1.0, 0.0, layer_z, energy],
                    [0.0, 1.0, layer_z, energy],
                ]
            )
            return points_in_layer

        all_points = np.concatenate(
            [get_points_for_layer(i) for i in range(acc.n_layers)]
        )
        all_points = np.array([all_points for i in range(n_incidents)])
        # need multiple events to test standard devation calculations
        for i in range(4):
            indices = (
                np.arange(n_incidents)
                + i * n_incidents
                + e * n_energies * 5 * n_incidents
            )
            acc.add(indices, incident_mid_points, all_points)
    return acc


def add_random(acc, n_events=100):
    events = np.random.rand(n_events, 100, 4)
    incident_energies = 10 + np.random.rand(n_events) * 80
    try:
        first_index = np.max(acc.accumulated_indices) + 1
    except ValueError:
        first_index = 0
    indices = list(range(first_index, first_index + n_events))
    acc.add(indices, incident_energies, events)
