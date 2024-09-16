"""
Read and write dat aon disk, creates a common interface.
"""

import h5py
import numpy as np
from functools import lru_cache
import glob

from ..utils.metadata import Metadata


@lru_cache(maxsize=1)
def get_possible_files(dataset_path):
    """
    Given a dataset path, pottentially with unfilled format entries,
    find all the files that are accessible and match the pattern.

    Parameters
    ----------
    dataset_path: str
        Path to the dataset
        The path can contain format placeholders, e.g. "data_{}.h5"

    Returns
    -------
    matching_files: list of str
        List of all the files that match the pattern
        Sorted in deterministic order.

    Raises
    ------
    FileNotFoundError
        If no files are found with the pattern
    """
    replacements = ["*" for _ in range(dataset_path.count("{"))]
    subbed_name = dataset_path.format(*replacements)
    matching_files = sorted(glob.glob(subbed_name))
    if not matching_files:
        raise FileNotFoundError(f"No files found with pattern {subbed_name}")
    return matching_files


@lru_cache(maxsize=1)
def get_files(dataset_path, n_files):
    """
    Deterministically get the requested number of files from the dataset.

    Parameters
    ----------
    dataset_path: str
        Path to the dataset
        The path can contain format placeholders, e.g. "data_{}.h5"
    n_files: int
        Requested number of files to get
        If 0, all files are returned.

    Returns
    -------
    files: list of str
        List of the files to be read
        If n_files is 0, all files are returned
        Sorted into deterministic order.
    """
    files = get_possible_files(dataset_path)
    if n_files == 0:
        return files
    return files[:n_files]


def regularise_event_axes(events, is_transposed=None, known_regular=False):
    """
    Regularize the axis format for the input showers.
    The regular format is (event_n, point_n, x_y_z_e).
    This performs two checks;
      - If there are aditional axis of length 1 before the event axis, they are removed.
      - Attempt to ensure the features axes (x_y_z_e) is the last axes.
        If only one axes is length 4, it is assumed to be the features axes.
        If the last two axis are length 4, the second last axes is considered
        the features axes if is_transposed is True, otherwise the
        last axes is considered the features axes.

    Parameters
    ----------
    events : array like with at least 3 axes
        The input showers tensor.
    is_transposed : bool, or None, optional
        If True, the second last axes is always considered the features axes.
        If False, the last axes is always considered the features axes.
        If None, the length of the axes are checked to determine the features axes.
        Default is None.

    Returns
    -------
    events : array like with 3 axes
        The regularized showers.

    Raises
    ------
    ValueError
        If the input showers do not have at least 3 axes.
        If there are axes not length 1 before the event axis.
        If no features axes are found.
        Basically, any invalid axis configuration.

    RuntimeError
       There isn't enough information to determine the features axes.
    """
    if len(events.shape) < 3:
        raise ValueError("Input showers must have at least 3 axes.")
    if len(events.shape) > 3:
        prior_axes = events.shape[:-3]
        if not all([a == 1 for a in prior_axes]):
            raise ValueError(
                "There are axes not length 1 before the event axis; " f"{events.shape}"
            )
        while len(events.shape) > 3:
            # don't want to straight up squeeze in case there is only one event
            events = events[0]
    if is_transposed == True:
        assert (
            events.shape[1] == 4
        ), "The second last axes must be length 4 in transposed events"
        events = events.transpose(0, 2, 1)
    elif is_transposed == False:
        assert (
            events.shape[2] == 4
        ), "The last axes must be length 4 in non-transposed events"
    else:
        if events.shape[2] != 4:
            if not events.shape[1] == 4:
                raise ValueError("Neither of the last two axes are length 4")
            events = events.transpose(0, 2, 1)
        elif events.shape[1] == events.shape[2]:
            raise RuntimeError(
                "There are two axes of length 4 in the shower"
                " and we do not know if it is transposed"
            )
    return events


@lru_cache(maxsize=1)
def get_n_events(dataset_path, n_files=0):
    """
    Get the number of events in the dataset

    Parameters
    ----------
    dataset_path: str
        Path to the dataset
    n_files: int
        Number of files in the dataset.
        If 0, the dataset is assumed to be a single file.
        If > 0, the dataset is assumed to be split into multiple files,
        and the dataset path should contain a format string with a single
        integer placeholder.

    Returns
    -------
    n_events: int or list of ints
        Number of events in the dataset, or in each file if n_files > 0

    """
    n_events = []
    for file_name in get_files(dataset_path, n_files):
        with h5py.File(file_name, "r") as on_disk:
            events_array_shape = on_disk["events"].shape
            if np.sum(events_array_shape):
                n_events.append(on_disk["events"].shape[-3])
    if len(n_events) < 2:
        n_events = np.sum(n_events)
    return n_events


def events_to_local(events, orientation):
    """
    Rotate axes order so that the shower progresses along the z-axis.
    Modifes in place.
    To get local coordinates from global coordinates,
    the axis along which the shower is developing is swapped with the z-axis.
    To revert it, the swap is performed again.

    Parameters
    ----------
    events : array
        The input showers tensor.
    global_shower_orientation : char
        "x", "y", or "z", the direction of the "z" axis of the local
        shower coordinate system in the detector coordinate system.

    """
    local_orientation = _validate_orientations(orientation)
    column_target = ["xyz".index(c) for c in local_orientation]
    events[..., column_target] = events[..., [0, 1, 2]]


def _validate_orientations(orientation, orientation_global=None):
    """
    Check the format of orientation strings, return the local and global orientations
    relative to the disk.

    Parameters
    ----------
    orientation : str
        The relationship between orientation on disk and local coordinates.
    orientation_global : str, optional
        The relationship between orientation on disk and global coordinates.
        If None, doesn't need checking

    Returns
    -------
    short_orientation : str
        The relationship between orientation on disk and local coordinates.
    short_orientation_global : str (Optional)
        The relationship between orientation on disk and global coordinates.

    """
    expected_start = "hdf5:xyz==local:"
    if not orientation.startswith(expected_start):
        raise NotImplementedError(
            f"Do not know how to interpret orientation {orientation}"
        )
    short_orientation = orientation[len(expected_start) :]
    if orientation_global is not None:
        expected_start = "hdf5:xyz==global:"
        if not orientation_global.startswith(expected_start):
            raise NotImplementedError(
                f"Do not know how to interpret orientation {orientation_global}"
            )
        short_orientation_global = orientation_global[len(expected_start) :]
        return short_orientation, short_orientation_global
    return short_orientation


def local_to_global(events, orientation, orientation_global):
    """
    Rotate events given in a local coordinate system to the global coordinate system.
    Two orientations strings are supplied becuase the metadata saves the relationship
    bettween the disk coordinates and the local coordinates, and the disk
    coordinates and the global coordinates.

    Does not act in place.

    Parameters
    ----------
    events : array [..., 4]
        The input points, in local coordinates, with last axis in xyze format.
    orientation : str
        The relationship between orientation on disk and local coordinates.
    orientation_global : str
        The relationship between orientation on disk and global coordinates.

    Returns
    -------
    events : array [..., 4]
        The input points, in global coordinates, with last axis in xyze format.

    """
    if not len(events):
        return events
    local_to_disk, disk_to_global = _validate_orientations(
        orientation, orientation_global
    )
    to_global = [local_to_disk.index(c) for c in disk_to_global]
    new_events = events.copy()
    new_events[..., to_global] = events[..., [0, 1, 2]]
    return new_events


def global_to_local(events, orientation, orientation_global):
    """
    Rotate events given in a global coordinate system to the local coordinate system.
    Two orientations strings are supplied becuase the metadata saves the relationship
    bettween the disk coordinates and the local coordinates, and the disk
    coordinates and the global coordinates.

    Does not act in place.

    Parameters
    ----------
    events : array [..., 4]
        The input points, in global coordinates, with last axis in xyze format.
    orientation : str
        The relationship between orientation on disk and local coordinates.
    orientation_global : str
        The relationship between orientation on disk and global coordinates.

    Returns
    -------
    events : array [..., 4]
        The input points, in local coordinates, with last axis in xyze format.

    """
    if not len(events):
        return events
    local_to_disk, disk_to_global = _validate_orientations(
        orientation, orientation_global
    )
    to_local = [disk_to_global.index(c) for c in local_to_disk]
    new_events = events.copy()
    new_events[..., to_local] = events[..., [0, 1, 2]]
    return new_events


def read_raw_regaxes(config, pick_events=None, total_size=None):
    """
    Draw raw events form the dataset, reshape, and move to local axis orientation,
    but don't alter values.

    Parameters
    ----------
    config: configs.Configs
        The configuration object
    pick_events: list of ints or slice
        Indices of the events to pick
        Optional, default is None, then a linearly spaced selection is made
    total_size: int
        Total number of events to pick
        Ignored if pick_events is not None
        If set to -1, read all events.
        Optional, default is 100

    Returns
    -------
    energy: array
        Incident energies of the events
        The first index is the event number
    events: array
        The events themselves
        The First index is the event number,
        the second is the point number,
        the third is the coordinate number in xyze format

    """
    if not hasattr(config, "n_dataset_files"):
        config.n_dataset_files = 0
    n_events = get_n_events(config.dataset_path, config.n_dataset_files)
    n_total_events = np.sum(n_events)
    if total_size is None:
        total_size = min(100, n_total_events)
    if pick_events is None:
        if total_size == -1:
            pick_events = np.arange(n_total_events)
        else:
            print("Selecting evenly spaced events")
            pick_events = np.linspace(0, n_total_events - 1, max(total_size, 1)).astype(
                int
            )
    elif isinstance(pick_events, slice):
        pick_events = np.arange(n_total_events)[pick_events]
    else:
        pick_events = np.array(pick_events, dtype=int)
        assert (
            np.max(pick_events, initial=0) < n_total_events
        ), "Event index out of range in pick_events"

    file_names = get_files(config.dataset_path, config.n_dataset_files)
    file_indices = []
    if len(file_names) == 1:
        file_indices.append(pick_events)
    else:
        file_start = 0
        for i, file_name in enumerate(file_names):
            file_end = file_start + n_events[i]
            file_indices.append(
                pick_events[(pick_events >= file_start) & (pick_events < file_end)]
                - file_start
            )
            file_start = file_end

    energy = []
    events = []
    for name, indices in zip(file_names, file_indices):
        with h5py.File(name, "r") as dataset:
            events_here = dataset["events"]
            energy_here = dataset["energy"]

            # treat empty files
            if len(events_here) == 0:
                events.append(np.zeros((0, 0, 4)))
                energy.append(np.zeros(0))
                continue

            # account for the variations in shower axes layout
            events_here = events_here[..., indices, :, :]
            # and make the axes order regular
            events_here = regularise_event_axes(events_here)
            events.append(events_here)

            # energy has more randomness again
            # can be (n_events,) or (1, n_events) or even (1, n_events, 1)
            axes_index = [0 for _ in energy_here.shape]
            axes_index[np.argmax(energy_here.shape)] = indices
            energy_here = energy_here[tuple(axes_index)]
            energy.append(energy_here)

    energy = np.concatenate(energy)
    events = np.vstack(events)

    metadata = Metadata(config)
    events_to_local(events, metadata.orientation)

    return energy, events


def check_regaxes(incedent_energy, events):
    """
    Check if the input data is in the regular shape.

    Parameters
    ----------
    incedent_energy: array (n_events,)
        Incident energies of the events
        The first index is the event number
    events: array (n_events, n_points, 4)
        The events themselves
        The First index is the event number,
        the second is the point number,
        the third is the coordinate number in xyze format

    Raises
    ------
    ValueError
        If the input data is not in the regular shape
    """
    if len(incedent_energy.shape) != 1:
        raise ValueError("Incident energy must be 1D")
    if len(events.shape) != 3:
        raise ValueError("Events must be 3D")
    n_events = incedent_energy.shape[0]
    if n_events != events.shape[0]:
        raise ValueError("Number of events in energy and events do not match")
    if events.shape[2] != 4:
        raise ValueError("Events must have 4 coordinates")


def write_raw_regaxes(destination, incedent_energy, events):
    """
    Write data to the disk, require the data be in a regular shape.
    Assumes the axis order is the correct one for the local_xyz_orientation
    as specified in the metadata.

    Parameters
    ----------
    destination: str
        Path to the destination file
    incedent_energy: array (n_events,)
        Incident energies of the events
        The first index is the event number
    events: array (n_events, n_points, 4)
        The events themselves
        The First index is the event number,
        the second is the point number,
        the third is the coordinate number in xyze format

    Raises
    ------
    ValueError
        If the input data is not in the regular shape
    """
    check_regaxes(incedent_energy, events)
    with h5py.File(destination, "w") as dataset:
        dataset.create_dataset("energy", data=incedent_energy)
        dataset.create_dataset("events", data=events)
