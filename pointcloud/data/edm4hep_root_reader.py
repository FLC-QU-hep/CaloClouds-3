import uproot
import pandas as pd
from particle import Particle
import numpy as np
from functools import lru_cache
import awkward as ak


def get_events_filenames(files):
    """
    Get the events branches and the file names,
    either from a list of file names or from a
    list of open uproot files.

    Parameters
    ----------
    files : iterable of str or uproot.reading.ReadOnlyDirectory
        The files to read from.

    Returns
    -------
    file_names : list of str
        Full paths for the files.
    events : list of uproot.behaviors.TBranch.TBranch
        The events branches.

    """
    if not files:
        return tuple(), []
    if isinstance(files[0], str):
        file_names = tuple(files)
        events = _get_events(file_names)
    else:
        file_names = tuple(file.file.file_path for file in files)
        events = [file["events"] for file in files]
    return file_names, events


@lru_cache(maxsize=1)
def _get_events(file_names):
    """
    Get the events branches from a list of file names.

    Parameters
    ----------
    file_names : tuple of str
        The files to read from. Must be a tuple for hashing.

    Returns
    -------
    events : list of uproot.behaviors.TBranch.TBranch
        The events branches.

    """
    events = [uproot.open(file_name)["events"] for file_name in file_names]
    return events


class MultiRootReader:
    def __init__(self, files, file_length_branch="MCParticles/MCParticles.PDG"):
        self.file_names, self.events = get_events_filenames(tuple(files))
        self.file_lengths = [
            len(event[file_length_branch].array()) for event in self.events
        ]
        self._cumulative_lengths = np.cumsum(self.file_lengths)

    @lru_cache(maxsize=10)
    def decumulated_indices(self, event_n):
        file_idx = np.searchsorted(self._cumulative_lengths, event_n, side="right")
        if file_idx == 0:
            idx_in_file = event_n
        else:
            idx_in_file = event_n - self._cumulative_lengths[file_idx - 1]
        return file_idx, idx_in_file

    def get(self, branch_name, event_n):
        file_idx, idx_in_file = self.decumulated_indices(event_n)
        events = self.events[file_idx]
        return events[branch_name].array()[idx_in_file]

    def __len__(self):
        return self._cumulative_lengths[-1]

    def keys(self):
        return self.events[0].keys()

    def _get_MC_points(self, event_n):
        data = {}
        key_map = {
            "PDG": "PDG",
            "Time": "time",
            "X": "vertex.x",
            "Y": "vertex.y",
            "Z": "vertex.z",
        }
        for key in key_map:
            root_key = f"MCParticles/MCParticles.{key_map[key]}"
            data[key] = self.get(root_key, event_n).tolist()
        mass = self.get("MCParticles/MCParticles.mass", event_n)
        momentum2 = (
            self.get("MCParticles/MCParticles.momentum.x", event_n) ** 2
            + self.get("MCParticles/MCParticles.momentum.y", event_n) ** 2
            + self.get("MCParticles/MCParticles.momentum.z", event_n) ** 2
        )
        data["Energy"] = ((momentum2 + mass**2) ** 0.5).tolist()
        data["Part"] = ["MCParticles"] * len(data["Energy"])
        return data


class Points:
    keys = ["Part", "PDG", "Name", "Energy", "Time", "X", "Y", "Z"]

    def __init__(self, files):
        self._reader = MultiRootReader(files)
        self._contrib_names = [
            key.split("/")[0]
            for key in self._reader.keys()
            if key.endswith("Contributions.PDG") and "ScHit" not in key
        ]
        self.parts = ["MCParticles"] + self._contrib_names
        self.part_short_names = {
            name: name.replace("Contributions", "")
            .replace("Odd", "")
            .replace("Even", "")
            for name in self.parts
        }
        ecal_symbols = ["square", "diamond"]
        hcal_symbols = ["cross", "x"]
        self.symbol_map = {"MCParticles": "circle"}
        set_short_names = set(self.part_short_names.values())
        set_short_names.remove("MCParticles")
        for name in set_short_names:
            low_name = name.lower()
            if "hcal" in low_name:
                symbols = hcal_symbols
            elif "ecal" in low_name:
                symbols = ecal_symbols
            else:
                symbols = ["circle-open", "diamond-open"]

            self.symbol_map[name] = symbols[int("barrel" in low_name)]

    @lru_cache(maxsize=100)
    def _get_contrib(self, contrib_name, event_n):
        data = {}
        key_map = {
            "PDG": "PDG",
            "Energy": "energy",
            "Time": "time",
            "X": "stepPosition.x",
            "Y": "stepPosition.y",
            "Z": "stepPosition.z",
        }
        for key in key_map:
            root_key = f"{contrib_name}/{contrib_name}.{key_map[key]}"
            data[key] = self._reader.get(root_key, event_n).tolist()
        data["Part"] = [self.part_short_names[contrib_name]] * len(data["Energy"])
        return data

    @lru_cache(maxsize=10)
    def __getitem__(self, event_n):
        data = self._reader._get_MC_points(event_n)
        for part in self._contrib_names:
            contrib_data = self._get_contrib(part, event_n)
            for key in contrib_data:
                data[key] += contrib_data[key]
        data["Name"] = []
        for pdgid in data["PDG"]:
            try:
                data["Name"].append(Particle.from_pdgid(pdgid).name)
            except Exception:
                data["Name"].append(str(pdgid))
        data["Energy"] = np.array(data["Energy"]) * 1000
        data["EnergySize"] = np.clip(np.log(data["Energy"] + 1), 0.0, 3)
        return pd.DataFrame(data)

    def __len__(self):
        return len(self._reader)

    def get_ecal_center(self, event_n):
        relevent_contribs = [
            contrib for contrib in self._contrib_names if "ECal" in contrib
        ]
        data = {"Energy": [], "X": [], "Y": [], "Z": []}
        for part in relevent_contribs:
            contrib_data = self._get_contrib(part, event_n)
            for key in data:
                data[key] += contrib_data[key]
        energy = np.array(data["Energy"])
        sum_energy = np.sum(energy)
        if sum_energy <= 0:
            return 0, 0, 0
        x_mean = (np.array(data["X"]) * energy).sum() / sum_energy
        y_mean = (np.array(data["Y"]) * energy).sum() / sum_energy
        z_mean = (np.array(data["Z"]) * energy).sum() / sum_energy
        return x_mean, y_mean, z_mean


class Relationships:
    def __init__(self, files):
        self._reader = MultiRootReader(files)

    def _get_reference_columns(self, column_base):
        column_begin = column_base + "_begin"
        column_end = column_base + "_end"
        relationship_type = column_base.split(".")[-1]
        particle_type = column_base.split("/")[0]
        branch_name = f"_{particle_type}_{relationship_type}"
        column_index = f"{branch_name}/{branch_name}.index"
        return column_begin, column_end, column_index

    def _get_refered_indices(self, event_n, column_begin, column_end, column_index):
        file_idx, idx_in_file = self._reader.decumulated_indices(event_n)
        events = self._reader.events[file_idx]
        beginings = events[column_begin].array()[idx_in_file]
        ends = events[column_end].array()[idx_in_file]
        lengths = ends - beginings
        flat_indices = events[column_index].array()[idx_in_file]
        return lengths, flat_indices

    def _connected_values(self, event_n, data, reference_base):
        column_begin, column_end, column_index = self._get_reference_columns(
            reference_base
        )
        lengths, flat_indices = self._get_refered_indices(
            event_n, column_begin, column_end, column_index
        )
        refered_values = {}
        for key in data:
            flat_values = np.array(data[key])[flat_indices]
            refered_values[key] = ak.unflatten(flat_values, lengths)
        return refered_values

    def _connected_idxs(self, column_base, event_n):
        column_begin, column_end, column_index = self._get_reference_columns(
            column_base
        )
        lengths, flat_indices = self._get_refered_indices(
            event_n, column_begin, column_end, column_index
        )
        return ak.unflatten(flat_indices, lengths)

    def parent_MCParticle_idxs(self, event_n, part_name=None):
        return self._connected_idxs("MCParticles/MCParticles.parents", event_n)

    def child_MCParticle_idxs(self, event_n):
        return self._connected_idxs("MCParticles/MCParticles.daughters", event_n)

    def MCParticles(self, event_n):
        mc_points = self._reader._get_MC_points(event_n)
        return mc_points

    def __len__(self):
        return len(self._reader)
