import numpy as np

from pointcloud.utils.metadata import Metadata
from pointcloud.utils import detector_map
from pointcloud.data.read_write import read_raw_regaxes, get_n_events
from pointcloud.evaluation.bin_standard_metrics import BinnedData
from pointcloud.evaluation.bin_standard_metrics import get_path as _base_get_path
from pointcloud.utils.gen_utils import gen_cond_showers_batch
from pointcloud.data.conditioning import (
    read_raw_regaxes_withcond,
    get_cond_features_names,
)


def get_path(config, name):
    base = _base_get_path(config, name, detector_projection=True)
    path = base.replace("_detectorProj", "_scaleFactor")
    return path


class DetectorScaleFactors:
    """
    Create an object that can be used to caculate scale
    factors for energy and number of points.
    Can be filled out in the same way as BinnedData
    """

    arguments = [
        "g4_xyz_limits",
        "model_xyz_limits",
        "g4_energy_scale",
        "model_energy_scale",
        "MAP",
        "g4_layer_bottom_pos",
        "model_layer_bottom_pos",
        "half_cell_size_global",
        "cell_thickness",
        "g4_gun_xyz_pos",
        "model_gun_xyz_pos",
        "model_config",
        "model",
        "shower_flow",
    ]
    other_savables = [
        "cond",
        "g4_detector_n",
        "g4_detector_e",
        "real_coeff",
        "fake_coeff",
        "final_n_coeff",
        "final_e_coeff",
    ]

    def __init__(
        self,
        g4_xyz_limits,
        model_xyz_limits,
        g4_energy_scale,
        model_energy_scale,
        MAP,
        g4_layer_bottom_pos,
        model_layer_bottom_pos,
        half_cell_size_global,
        cell_thickness,
        g4_gun_xyz_pos,
        model_gun_xyz_pos,
        model_config,
        model,
        shower_flow,
    ):
        """Construct a BinnedData object.

        Parameters
        ----------
        energy_scale : float
            How much to divide the observed energy by before binning.
        MAP : list
            list of dictionaries, each containing the grid of cells for a layer
            in global coordinates
            As returned by detector_map.create_map
            Should be already offset, such that the gun fires at the origin.
        layer_bottom_pos : np.array
            The y positions of the bottom of each layer, in the coordinates
            the data is given in.
        half_cell_size_global : float
            Half the size of the cells in the detector,
            perpendicular to the radial direction
        cell_thickness: float
            The thickness in the y direction of each cell, in the coordinates
            the data is given in.
        gun_xyz_pos : np.array
            The x, y and z positions of the gun, in the coordinates the data is
            given in. The data is shifted so that the gun is at the origin.
        """
        self.g4_binner = BinnedData(
            "G4",
            g4_xyz_limits,
            g4_energy_scale,
            g4_layer_bottom_pos,
            cell_thickness,
            g4_gun_xyz_pos,
        )
        self.model_binner = BinnedData(
            "Model",
            model_xyz_limits,
            model_energy_scale,
            model_layer_bottom_pos,
            cell_thickness,
            model_gun_xyz_pos,
        )
        self.MAP = MAP
        self.half_cell_size_global = half_cell_size_global
        self.cell_thickness = cell_thickness

        self.model_config = model_config
        self.model = model
        self.shower_flow = shower_flow

        self.cond = None
        self.g4_detector_n = None
        self.g4_detector_e = None

    def add_events(self, cond, g4_events):
        """
        Add events to the histograms that were created in the constructor.

        Parameters
        ----------
        events : np.array (n_showers, n_points, 4)
            The events to add to the histograms.
        """
        if self.cond is None:
            self.cond = cond
        else:
            self.cond = {
                k: np.concatenate((self.cond[k], cond[k]), axis=0)
                for k in self.cond.keys()
            }

        g4_n, g4_e = self.get_projected_values(g4_events, self.g4_binner)
        if self.g4_detector_n is None:
            self.g4_detector_n = g4_n
            self.g4_detector_e = g4_e
        else:
            self.g4_detector_n = np.concatenate((self.g4_detector_n, g4_n), axis=0)
            self.g4_detector_e = np.concatenate((self.g4_detector_e, g4_e), axis=0)

    def get_projected_values(self, events, binner):
        # binner.rescaled_events(events)
        gun_shifted = events - binner.gun_shift
        events_as_cells = detector_map.get_projections(
            gun_shifted,
            self.MAP,
            layer_bottom_pos=binner.layer_bottom_pos,
            half_cell_size_global=self.half_cell_size_global,
            cell_thickness_global=binner.cell_thickness,
            return_cell_point_cloud=False,
            include_artifacts=False,
        )
        detector_map.mip_cut(events_as_cells)
        n = np.fromiter(
            (np.sum(np.sum(l > 0) for l in event) for event in events_as_cells),
            dtype=int,
        )
        e = np.fromiter(
            (np.sum(np.sum(l) for l in event) for event in events_as_cells), dtype=float
        )
        return n, e

    def model_events(self, cond_idxs, n_coeff, e_coeff, fake_n_coeff=None):

        self.model_config.shower_flow_n_scaling = True
        self.model_config.shower_flow_coef_real = n_coeff
        self.model_config.shower_flow_coef_fake = fake_n_coeff

        cond = {}
        for part in ["showerflow", "diffusion"]:
            part_cond = np.hstack(
                [
                    self.cond["points" if name == "n_points" else name][cond_idxs]
                    for name in get_cond_features_names(self.model_config, part)
                ]
            )
            cond[part] = part_cond

        model_events = gen_cond_showers_batch(
            self.model,
            self.shower_flow,
            cond,
            bs=len(cond_idxs),
            config=self.model_config,
        )
        if e_coeff is not None:
            model_events[:, :, 3] = np.poly1d(e_coeff)(model_events[:, :, 3])
        model_events[model_events[:, :, 3] <= 0] = 0

        n, e = self.get_projected_values(model_events, self.model_binner)
        return n, e

    def divergance(self, n_events, n_coeff, e_coeff=None, fake_n_coeff=None):
        #self.cond_idxs = np.random.choice(len(self.cond["energy"]), n_events)
        self.cond_idxs = np.linspace(0, len(self.cond["energy"])-1, n_events, dtype=int)

        g4_n = self.g4_detector_n[self.cond_idxs]
        g4_e = self.g4_detector_e[self.cond_idxs]

        self.model_n, self.model_e = self.model_events(
            self.cond_idxs, n_coeff, e_coeff, fake_n_coeff
        )
        self.model_e /= self.model_binner.energy_scale

        return np.mean((g4_n - self.model_n) ** 2), np.mean((g4_e - self.model_e) ** 2)

    def save_dict(self):
        save_items = {}
        for name in self.arguments + self.other_savables:
            if hasattr(self, name):
                save_items[name] = getattr(self, name)
                continue
            g4_trim = name.replace("g4_", "")
            if hasattr(self.g4_binner, g4_trim):
                save_items[name] = getattr(self.g4_binner, g4_trim)
                continue
            model_trim = name.replace("model_", "")
            if hasattr(self.model_binner, model_trim):
                save_items[name] = getattr(self.model_binner, model_trim)
                continue
        return save_items

    def save(self, path):
        save_items = self.save_dict()
        np.savez(path, **save_items)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        init_arguments = {}
        to_set = {}
        for key in data.keys():
            value = data[key]
            try:
                value = value.item()
            except ValueError:
                pass
            if key in cls.arguments:
                init_arguments[key] = value
            else:
                to_set[key] = value
        new_dsf = cls(**init_arguments)
        for key in to_set:
            setattr(new_dsf, key, to_set[key])
        return new_dsf


def sample_g4(config, scale, n_events):
    """
    Draw samples from the g4 data and add them to the binned data.

    Parameters
    ----------
    config : Configs
        The configuration used to find the dataset, etc.
    scale : DetectorScaleFactors
        The scale factors that we are looking to calculate.
    n_events : int
        The number of events to sample.
    """
    batch_len = 1000
    batch_starts = np.arange(0, n_events, batch_len)
    n_batches = np.ceil(n_events / batch_len)
    old_cond_name = config.cond_features_names[:]
    old_cond_dim = config.cond_features
    all_cond_name = ["energy", "points", "p_norm_local"]
    config.cond_features_names = all_cond_name
    config.cond_features = 5
    for b, start in enumerate(batch_starts):
        print(f"{b/n_batches:.1%}", end="\r", flush=True)
        cond, events = read_raw_regaxes_withcond(
            config, pick_events=slice(start, start + batch_len)
        )
        split_cond = {
            "energy": cond["diffusion"][:, [0]],
            "points": cond["diffusion"][:, [1]],
            "p_norm_local": cond["diffusion"][:, 2:],
        }
        scale.add_events(split_cond, events)
    print()
    print("Done")


def construct_dsf(
    config,
    model_name,
    model_config,
    model,
    shower_flow,
    max_g4_events=10_000,
):
    meta = Metadata(config)
    MAP, _ = detector_map.create_map(config=config)
    shifted_MAP = MAP[:]
    for layer in shifted_MAP:
        layer["xedges"] -= 30
        #layer["xedges"] -= 50
        #layer["zedges"] -= 50
    floors, ceilings = detector_map.floors_ceilings(
        meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5, 0
    )

    g4_name = "Geant 4"
    n_g4_events = np.sum(get_n_events(config.dataset_path, config.n_dataset_files))
    if max_g4_events:
        n_g4_events = min(n_g4_events, max_g4_events)

    # Get the g4 data
    print(f"Need to process {g4_name}")

    raw_floors, raw_ceilings = detector_map.floors_ceilings(
        meta.layer_bottom_pos_hdf5, meta.cell_thickness_hdf5, 0
    )
    g4_xyz_limits = [
        [meta.Zmin_global, meta.Zmax_global],
        [meta.Xmin_global, meta.Xmax_global],
        [raw_floors[0], raw_ceilings[-1]],
    ]
    g4_gun_pos = np.array([-50, 0, 0])
    g4_layer_bottom_pos = meta.layer_bottom_pos_hdf5
    g4_energy_scale = 1.0
    cc_floors, cc_ceilings = detector_map.floors_ceilings(
        meta.layer_bottom_pos_global, meta.cell_thickness_global, 0
    )
    model_xyz_limits = [
        [meta.Zmin_global, meta.Zmax_global],
        [meta.Xmin_global, meta.Xmax_global],
        [cc_floors[0], cc_ceilings[-1]],
    ]
    model_layer_bottom_pos = meta.layer_bottom_pos_global
    model_rescale_energy = 1e3
    if "3" in model_name:
        model_gun_pos = np.array([0, 0, 0])
    else:
        model_gun_pos = np.array([-50, 0, 0])
    dsf = DetectorScaleFactors(
        g4_xyz_limits,
        model_xyz_limits,
        g4_energy_scale,
        model_rescale_energy,
        shifted_MAP,
        g4_layer_bottom_pos,
        model_layer_bottom_pos,
        meta.half_cell_size_global,
        meta.cell_thickness_global,
        g4_gun_pos,
        model_gun_pos,
        model_config,
        model,
        shower_flow,
    )
    sample_g4(config, dsf, n_g4_events)
    return dsf
