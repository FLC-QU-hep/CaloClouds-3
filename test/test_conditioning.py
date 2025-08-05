""" Module to test the data/conditioning.py module. """

import numpy as np
import numpy.testing as npt
import torch
import os
import tempfile
import shutil
import h5py
import warnings

from pointcloud.data import conditioning
from pointcloud.utils.metadata import Metadata

from helpers import config_creator
from helpers.mock_metadata import TemporaryMetadata


def test_get_cond_features_names():
    config = config_creator.make()
    if hasattr(config, "cond_features"):
        del config.cond_features
    if hasattr(config, "shower_flow_cond_features"):
        del config.shower_flow_cond_features
    with warnings.catch_warnings(record=True) as w:
        found = conditioning.get_cond_features_names(config, "showerflow")
        assert len(w) == 1
        assert "Did not find cond_features" in str(w[0].message)
        assert found == ["energy", "n_points"]
    with warnings.catch_warnings(record=True) as w:
        found = conditioning.get_cond_features_names(config, "diffusion")
        assert len(w) == 1
        assert "Did not find cond_features" in str(w[0].message)
        assert found == ["energy", "n_points"]
    config.shower_flow_cond_features = ["jelly"]
    with warnings.catch_warnings(record=True) as w:
        found = conditioning.get_cond_features_names(config, "showerflow")
        assert len(w) == 0
        assert found == ["jelly"]
    with warnings.catch_warnings(record=True) as w:
        found = conditioning.get_cond_features_names(config, "diffusion")
        assert len(w) == 1
        assert "Did not find cond_features" in str(w[0].message)
        assert found == ["jelly"]
    config.cond_features = ["beans"]
    found = conditioning.get_cond_features_names(config, "showerflow")
    assert found == ["jelly"]
    with warnings.catch_warnings(record=True) as w:
        found = conditioning.get_cond_features_names(config, "diffusion")
        assert len(w) == 0
        assert found == ["beans"]
    with warnings.catch_warnings(record=True) as w:
        config.cond_features = 1
        found = conditioning.get_cond_features_names(config, "diffusion")
        assert len(w) == 0
        assert found == ["energy"]
        config.cond_features = 2
        found = conditioning.get_cond_features_names(config, "diffusion")
        assert len(w) == 0
        assert found == ["energy", "n_points"]
        config.cond_features = 4
        found = conditioning.get_cond_features_names(config, "diffusion")
        assert len(w) == 0
        assert found == ["energy", "p_norm_local"]
        config.cond_features = 5
        found = conditioning.get_cond_features_names(config, "diffusion")
        assert len(w) == 0
        assert found == ["energy", "n_points", "p_norm_local"]
    try:
        config.cond_features = 3
        found = conditioning.get_cond_features_names(config, "diffusion")
    except ValueError as e:
        pass
    else:
        assert False, "Should have raised a ValueError"


def test_get_cond_feats():
    config = config_creator.make()
    # play with naming of n_points
    config.cond_features = ["energy", "n_points"]
    config.shower_flow_cond_features = ["energy"]
    batch = {
        "energy": torch.tensor([1.0, 2.0, 3.0])[None, :, None],
        "points": torch.tensor([10, 20, 30])[None, :, None],
    }
    found = conditioning.get_cond_feats(config, batch, "showerflow")
    npt.assert_allclose(found, [[1.0], [2.0], [3.0]])
    found = conditioning.get_cond_feats(config, batch, "diffusion")
    npt.assert_allclose(found, [[1.0, 10], [2.0, 20], [3.0, 30]])
    # reverse
    batch = {
        "energy": torch.tensor([1.0, 2.0, 3.0])[None, :, None],
        "n_points": torch.tensor([10, 20, 30])[None, :, None],
    }
    config.cond_features = ["energy", "points"]
    found = conditioning.get_cond_feats(config, batch, "diffusion")
    npt.assert_allclose(found, [[1.0, 10], [2.0, 20], [3.0, 30]])


def test_normalise_cond_feats():
    config = config_creator.make()
    config.cond_features = ["energy", "n_points", "p_norm_local"]
    config.shower_flow_cond_features = ["energy", "points", "p_norm_local"]
    # no changes
    config.norm_cond = False
    original_cond = torch.arange(0, 15).reshape(3, 5).float()
    cond = original_cond.clone()
    found = conditioning.normalise_cond_feats(config, cond, "diffusion")
    assert cond is found
    npt.assert_allclose(found, original_cond)
    # normalise diffusion
    config.norm_cond = True
    found = conditioning.normalise_cond_feats(config, cond, "diffusion")
    npt.assert_allclose((original_cond[:, 0] / 100) * 2 - 1, found[:, 0])
    npt.assert_allclose((original_cond[:, 1] / config.max_points) * 2 - 1, found[:, 1])
    npt.assert_allclose(original_cond[:, 2:], found[:, 2:])
    npt.assert_allclose(cond, original_cond)
    assert found is not cond
    # normalise showerflow
    meta = Metadata(config)
    found = conditioning.normalise_cond_feats(config, cond, "showerflow")
    npt.assert_allclose((original_cond[:, 0] / meta.incident_rescale), found[:, 0])
    npt.assert_allclose((original_cond[:, 1] / config.max_points) * 2 - 1, found[:, 1])
    npt.assert_allclose(original_cond[:, 2:], found[:, 2:])
    npt.assert_allclose(cond, original_cond)
    assert found is not cond


def test_get_cond_dim():
    config = config_creator.make()
    config.cond_features = ["energy", "n_points", "p_norm_local"]
    config.shower_flow_cond_features = ["energy", "points"]
    found = conditioning.get_cond_dim(config, "diffusion")
    assert found == 5
    found = conditioning.get_cond_dim(config, "showerflow")
    assert found == 2
    config.cond_features = []
    found = conditioning.get_cond_dim(config, "diffusion")
    assert found == 0


def test_has_n_points(tmpdir):
    file_name = os.path.join(tmpdir, "test.h5")
    with h5py.File(file_name, "w") as f:
        f.create_dataset("n_points", data=np.array([1, 2, 3]))
    assert conditioning.has_n_points(file_name)
    with h5py.File(file_name, "a") as f:
        del f["n_points"]
    # due to the way caching works, this should still return True
    assert conditioning.has_n_points(file_name)
    file_name = os.path.join(tmpdir, "test2.h5")
    with h5py.File(file_name, "w") as f:
        f.create_dataset("n_sheep", data=np.array([1, 2, 3]))
    assert not conditioning.has_n_points(file_name)
    file_name = os.path.join(tmpdir, "test3.h5")
    with h5py.File(file_name, "w") as f:
        f.create_dataset("points", data=np.array([1, 2, 3]))
    assert conditioning.has_n_points(file_name)


def test_padding_position():
    events = torch.tensor(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]
    )
    found = conditioning.padding_position(events, True)
    assert found == "unknown"
    found = conditioning.padding_position(events, False)
    assert found == "unknown"
    # now add some zeros
    events[0, 1, 1] = 0  # should make no difference
    found = conditioning.padding_position(events, True)
    assert found == "unknown"
    found = conditioning.padding_position(events, False)
    assert found == "unknown"
    events[0, 3, 0] = 0  # padding for roll_axis
    found = conditioning.padding_position(events, True)
    assert found == "front"
    found = conditioning.padding_position(events, False)
    assert found == "unknown"
    events[0, -1, 3] = 0  # padding for not roll_axis
    found = conditioning.padding_position(events, True)
    assert found == "unknown"
    found = conditioning.padding_position(events, False)
    assert found == "back"
    # an empty event should return unknown
    events = torch.tensor([]).reshape(1, 0, 4)
    found = conditioning.padding_position(events, True)
    assert found == "unknown"


def test_calculate_n_points_from_events():
    events = np.ones((3, 3, 4))
    found = conditioning.calculate_n_points_from_events(events, "front")
    npt.assert_allclose(found, np.array([3, 3, 3])[:, None])
    found = conditioning.calculate_n_points_from_events(events, "back")
    npt.assert_allclose(found, np.array([3, 3, 3])[:, None])
    events[:, :, :3] = 0
    found = conditioning.calculate_n_points_from_events(events, "front")
    npt.assert_allclose(found, np.array([3, 3, 3])[:, None])
    found = conditioning.calculate_n_points_from_events(events, "back")
    npt.assert_allclose(found, np.array([3, 3, 3])[:, None])
    events[0, 0, 3] = 0
    events[1, -1, 3] = 0
    events[2, 1, 3] = 0
    found = conditioning.calculate_n_points_from_events(events, "front")
    npt.assert_allclose(found, np.array([2, 3, 3])[:, None])
    found = conditioning.calculate_n_points_from_events(events, "back")
    npt.assert_allclose(found, np.array([3, 2, 3])[:, None])
    # if the padding isn't given, but is unambiguous, it should be found
    events[0, -1, 3] = 0
    found = conditioning.calculate_n_points_from_events(events)
    npt.assert_allclose(found, np.array([2, 2, 3])[:, None])


class TestRead:
    """Test the reader in the conditioning module."""

    energy0 = np.array([[1.0]])
    direction0 = np.array([[10.0, 20.0, 30.0]])
    reg_energy0 = np.array([1.0])
    # one event, 4 points
    # designed to make it impossible to tell that the
    # axis order is events, xyze, point
    events0 = np.array(
        [
            [
                [
                    [1.0, 1.01, 1.02, 1.03],
                    [1.10, 1.11, 1.12, 1.13],
                    [1.20, 1.21, 1.22, 1.23],
                    [1.30, 1.31, 1.32, 1.33],
                ]
            ]
        ]
    )
    reg_events0 = np.array(
        [
            [
                [1.0, 1.1, 1.2, 1.3],
                [1.01, 1.11, 1.21, 1.31],
                [1.02, 1.12, 1.22, 1.32],
                [1.03, 1.13, 1.23, 1.33],
            ]
        ]
    )
    energy1 = np.array([[2.0, 3.0, 4.0]])
    direction1 = np.array(
        [[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]]
    )
    reg_energy1 = np.array([2.0, 3.0, 4.0])
    # three events, 1 point, 2 points, 1 point
    events1 = np.array(
        [
            [
                [
                    [2.0, 0.0],
                    [2.1, 0.0],
                    [2.2, 0.0],
                    [2.3, 0.0],
                ],
                [
                    [3.0, 3.01],
                    [3.10, 3.11],
                    [3.20, 3.21],
                    [3.30, 3.31],
                ],
                [
                    [4.0, 0.0],
                    [4.1, 0.0],
                    [4.2, 0.0],
                    [4.3, 0.0],
                ],
            ]
        ]
    )
    reg_events1 = np.array(
        [
            [[2.0, 2.1, 2.2, 2.3], [0.0, 0.0, 0.0, 0.0]],
            [[3.0, 3.1, 3.2, 3.3], [3.01, 3.11, 3.21, 3.31]],
            [[4.0, 4.1, 4.2, 4.3], [0.0, 0.0, 0.0, 0.0]],
        ]
    )

    folders = {}
    paths = {}
    tmpdir = tempfile.mkdtemp()
    temp_meta = TemporaryMetadata("WdatasetW")

    # as this deals in the global filesystem, we may as well have a
    # classmethod to create directories
    @classmethod
    def mkdir(cls, name):
        path = os.path.join(cls.tmpdir, name)
        os.mkdir(path)
        return path

    @classmethod
    def setup_class(cls):
        """
        The readers require data on disk to read, so we need to create
        some test data.
        """
        cls.temp_meta.__enter__()
        # the first will contain one empty dataset
        cls.folders["empty"] = cls.mkdir("empty_dataset")
        path_format = os.path.join(cls.folders["empty"], "empty_dataset.h5")
        with h5py.File(path_format, "w") as f:
            f.create_dataset("energy", data=np.empty(0))
            f.create_dataset("p_norm_local", data=np.empty(0))
            f.create_dataset("events", data=np.empty(0))
        cls.paths["empty"] = path_format
        # the second will contain one dataset with one event
        # in event, point, xyze format
        cls.folders["one_event"] = cls.mkdir("one_event_folder")
        cls.paths["one_event"] = os.path.join(
            cls.folders["one_event"], "one_event_dataset.h5"
        )
        energy = np.array([1.0])
        p_norm_local = np.array([[10.0, 10.0, 10.0]])
        events = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        with h5py.File(cls.paths["one_event"], "w") as f:
            f.create_dataset("energy", data=energy)
            f.create_dataset("p_norm_local", data=p_norm_local)
            f.create_dataset("events", data=events)
        # the third will contain four events, split between 2 files
        # in 1, event, xyze, point format
        cls.folders["four_events"] = cls.mkdir("four_events_folder")
        path_format = os.path.join(
            cls.folders["four_events"], "four_events_dataset_{}.h5"
        )
        cls.paths["four_events"] = path_format
        with h5py.File(path_format.format(0), "w") as f:
            f.create_dataset("energy", data=cls.energy0)
            f.create_dataset("p_norm_local", data=cls.direction0)
            f.create_dataset("events", data=cls.events0)
        with h5py.File(path_format.format(1), "w") as f:
            f.create_dataset("energy", data=cls.energy1)
            f.create_dataset("p_norm_local", data=cls.direction1)
            f.create_dataset("events", data=cls.events1)
        # the fourth will be just the last part of the third
        cls.folders["last_segment"] = cls.folders["four_events"]
        cls.paths["last_segment"] = path_format.format(1)

        # the fifth will contain a dataset with 9 events across 3 files
        # in 1, event, xyze, point format
        cls.folders["nine_events"] = cls.mkdir("nine_events_folder")
        cls.paths["nine_events"] = os.path.join(
            cls.folders["nine_events"], "nine_events_dataset_{}.h5"
        )
        for i in range(3):
            with h5py.File(cls.paths["nine_events"].format(i), "w") as f:
                f.create_dataset("energy", data=cls.energy1)
                f.create_dataset("p_norm_local", data=cls.direction1)
                f.create_dataset("events", data=cls.events1)

    @classmethod
    def teardown_class(cls):
        """
        Remove the test data from disk.
        """
        cls.temp_meta.__exit__()
        # remove the test data
        shutil.rmtree(cls.tmpdir)

    def test_read_raw_regaxes_withcond(self):
        config = config_creator.make()
        config.dataset_path = self.paths["empty"]
        config.n_dataset_files = 0
        config.cond_features = ["energy", "points"]
        config.shower_flow_cond_features = ["energy"]
        config.event_padding_position = "back"
        found_cond, found_events = conditioning.read_raw_regaxes_withcond(config)
        assert len(found_cond["showerflow"]) == 0
        assert len(found_cond["diffusion"]) == 0
        assert len(found_events) == 0
        config.dataset_path = self.paths["one_event"]
        for i in range(2):
            config.n_dataset_files = i
            found_cond, found_events = conditioning.read_raw_regaxes_withcond(config)
            npt.assert_allclose(found_cond["showerflow"], [1.0])
            npt.assert_allclose(found_cond["diffusion"], [[1.0, 2.0]])
            npt.assert_allclose(
                found_events, [[[3.0, 1.0, 2.0, 4.0], [7.0, 5.0, 6.0, 8.0]]]
            )

        # this should work without a defined padding position
        del config.event_padding_position
        config.dataset_path = self.paths["last_segment"]
        found_cond, found_events = conditioning.read_raw_regaxes_withcond(config)
        npt.assert_allclose(found_cond["showerflow"], self.reg_energy1)
        npt.assert_allclose(found_cond["diffusion"][:, 0], self.reg_energy1)
        npt.assert_allclose(found_cond["diffusion"][:, 1], [1, 2, 1])
        local_ax_1 = self.reg_events1[..., [2, 0, 1, 3]]
        npt.assert_allclose(found_events, local_ax_1)

        # finally, the nine events should be fine
        config.dataset_path = self.paths["nine_events"]
        config.n_dataset_files = 1
        found_cond, found_events = conditioning.read_raw_regaxes_withcond(config)
        npt.assert_allclose(found_cond["showerflow"], self.reg_energy1)
        npt.assert_allclose(found_cond["diffusion"][:, 0], self.reg_energy1)
        npt.assert_allclose(found_cond["diffusion"][:, 1], [1, 2, 1])
        npt.assert_allclose(found_events, local_ax_1)
        config.n_dataset_files = 3
        found_cond, found_events = conditioning.read_raw_regaxes_withcond(config)
        npt.assert_allclose(
            found_cond["showerflow"], [2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0]
        )
        npt.assert_allclose(
            found_cond["diffusion"][:, 0], [2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0]
        )
        npt.assert_allclose(found_cond["diffusion"][:, 1], [1, 2, 1, 1, 2, 1, 1, 2, 1])
        npt.assert_allclose(found_events, np.tile(local_ax_1, (3, 1, 1)))

    def test_read_raw_regaxes_weithcond_direction(self):
        # Again, but with the p_norm_local vector too
        # as we don't ask for points, we shouldn't need to know the padding
        config = config_creator.make()
        config.dataset_path = self.paths["empty"]
        config.n_dataset_files = 0
        config.shower_flow_cond_features = ["energy"]
        config.cond_features = ["energy", "p_norm_local"]
        found_cond, found_events = conditioning.read_raw_regaxes_withcond(config)
        assert len(found_cond["showerflow"]) == 0
        assert len(found_events) == 0
        config.dataset_path = self.paths["one_event"]
        for i in range(2):
            config.n_dataset_files = i
            found_cond, found_events = conditioning.read_raw_regaxes_withcond(config)
            npt.assert_allclose(found_cond["showerflow"], [1.0])
            npt.assert_allclose(found_cond["diffusion"], [[1.0, 10.0, 10.0, 10.0]])
            npt.assert_allclose(
                found_events, [[[3.0, 1.0, 2.0, 4.0], [7.0, 5.0, 6.0, 8.0]]]
            )

        # this should work
        config.dataset_path = self.paths["last_segment"]
        found_cond, found_events = conditioning.read_raw_regaxes_withcond(config)
        cond = np.hstack([self.reg_energy1[:, None], self.direction1[0]])
        npt.assert_allclose(found_cond["diffusion"], cond)
        npt.assert_allclose(found_cond["showerflow"], self.reg_energy1)
        local_ax_1 = self.reg_events1[..., [2, 0, 1, 3]]
        npt.assert_allclose(found_events, local_ax_1)

        # finally, the nine events should be fine
        config.dataset_path = self.paths["nine_events"]
        config.n_dataset_files = 1
        found_cond, found_events = conditioning.read_raw_regaxes_withcond(config)
        npt.assert_allclose(found_cond["diffusion"], cond)
        npt.assert_allclose(found_cond["showerflow"], self.reg_energy1)
        npt.assert_allclose(found_events, local_ax_1)
        config.n_dataset_files = 3
        found_cond, found_events = conditioning.read_raw_regaxes_withcond(config)
        cond = np.tile(cond, (3, 1))
        npt.assert_allclose(found_cond["diffusion"], cond)
        npt.assert_allclose(found_cond["showerflow"], cond[:, 0])
        npt.assert_allclose(found_events, np.tile(local_ax_1, (3, 1, 1)))


def test_cond_dim_at_path():
    # just run this on the models that are saved as examples
    # might not hit all the forks
    model_path_1 = "test/example_cm_model.pt"
    found = conditioning.cond_dim_at_path(model_path_1, "diffusion")
    assert found == 2
