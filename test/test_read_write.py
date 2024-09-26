""" Module to test the data/read_write.py module. """

import numpy as np
import numpy.testing as npt
import os
import tempfile
import shutil
import h5py

from pointcloud.data import read_write

from helpers import config_creator
from helpers.mock_metadata import TemporaryMetadata


class TestReaders:
    """Test the readers in the read_write module."""

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
            f.create_dataset("direction", data=np.empty(0))
            f.create_dataset("events", data=np.empty(0))
        cls.paths["empty"] = path_format
        # the second will contain one dataset with one event
        # in event, point, xyze format
        cls.folders["one_event"] = cls.mkdir("one_event_folder")
        cls.paths["one_event"] = os.path.join(
            cls.folders["one_event"], "one_event_dataset.h5"
        )
        energy = np.array([1.0])
        direction = np.array([[10.0, 10.0, 10.0]])
        events = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        with h5py.File(cls.paths["one_event"], "w") as f:
            f.create_dataset("energy", data=energy)
            f.create_dataset("direction", data=direction)
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
            f.create_dataset("direction", data=cls.direction0)
            f.create_dataset("events", data=cls.events0)
        with h5py.File(path_format.format(1), "w") as f:
            f.create_dataset("energy", data=cls.energy1)
            f.create_dataset("direction", data=cls.direction1)
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
                f.create_dataset("direction", data=cls.direction1)
                f.create_dataset("events", data=cls.events1)

    @classmethod
    def teardown_class(cls):
        """
        Remove the test data from disk.
        """
        cls.temp_meta.__exit__()
        # remove the test data
        shutil.rmtree(cls.tmpdir)

    def test_get_possible_files(self):
        possibles = read_write.get_possible_files(self.paths["empty"])
        assert possibles == [self.paths["empty"]]
        possibles = read_write.get_possible_files(self.paths["one_event"])
        assert possibles == [self.paths["one_event"]]
        possibles = read_write.get_possible_files(self.paths["four_events"])
        assert possibles == [
            self.paths["four_events"].format(0),
            self.paths["four_events"].format(1),
        ]
        possibles = read_write.get_possible_files(self.paths["last_segment"])
        assert possibles == [self.paths["last_segment"]]
        possibles = read_write.get_possible_files(self.paths["nine_events"])
        assert possibles == [self.paths["nine_events"].format(i) for i in range(3)]

    def test_get_files(self):
        found = read_write.get_files(self.paths["empty"], 10)
        assert found == [self.paths["empty"]]
        found = read_write.get_files(self.paths["one_event"], 0)
        assert found == [self.paths["one_event"]]
        found = read_write.get_files(self.paths["four_events"], 1)
        assert found == [self.paths["four_events"].format(0)]
        found = read_write.get_files(self.paths["four_events"], 2)
        assert found == [
            self.paths["four_events"].format(0),
            self.paths["four_events"].format(1),
        ]
        found = read_write.get_files(self.paths["four_events"], 10)
        assert found == [
            self.paths["four_events"].format(0),
            self.paths["four_events"].format(1),
        ]
        found = read_write.get_files(self.paths["last_segment"], 10)
        assert found == [self.paths["last_segment"]]
        found = read_write.get_files(self.paths["nine_events"], 10)
        assert found == [self.paths["nine_events"].format(i) for i in range(3)]

    def test_get_n_events(self):
        found = read_write.get_n_events(self.paths["empty"])
        assert found == 0
        found = read_write.get_n_events(self.paths["one_event"])
        assert found == 1
        # even though this dataset has ambiguous axis order,
        # the number of events should be clear
        found = read_write.get_n_events(self.paths["four_events"])
        npt.assert_allclose(found, [1, 3])
        found = read_write.get_n_events(self.paths["four_events"], 1)
        assert found == 1
        found = read_write.get_n_events(self.paths["four_events"], 2)
        npt.assert_allclose(found, [1, 3])
        found = read_write.get_n_events(self.paths["last_segment"])
        assert found == 3
        found = read_write.get_n_events(self.paths["nine_events"])
        npt.assert_allclose(found, [3, 3, 3])

    def test_read_raw_regaxes(self):
        configs = config_creator.make()
        configs.dataset_path = self.paths["empty"]
        configs.n_dataset_files = 0
        found_energy, found_events = read_write.read_raw_regaxes(configs)
        assert len(found_energy) == 0
        assert len(found_events) == 0
        configs.dataset_path = self.paths["one_event"]
        for i in range(2):
            configs.n_dataset_files = i
            found_energy, found_events = read_write.read_raw_regaxes(configs)
            npt.assert_allclose(found_energy, [1.0])
            npt.assert_allclose(
                found_events, [[[3.0, 1.0, 2.0, 4.0], [7.0, 5.0, 6.0, 8.0]]]
            )
        # becuase there are 4 points int he longest event,
        # this should throw RuntimeError
        configs.dataset_path = self.paths["four_events"]
        configs.n_dataset_files = 2
        npt.assert_raises(RuntimeError, read_write.read_raw_regaxes, configs)
        npt.assert_raises(RuntimeError, read_write.read_raw_regaxes, configs, 1)

        # this should work
        configs.dataset_path = self.paths["last_segment"]
        found_energy, found_events = read_write.read_raw_regaxes(configs)
        npt.assert_allclose(found_energy, self.reg_energy1)
        local_ax_1 = self.reg_events1[..., [2, 0, 1, 3]]
        npt.assert_allclose(found_events, local_ax_1)

        # finally, the nine events should be fine
        configs.dataset_path = self.paths["nine_events"]
        configs.n_dataset_files = 1
        found_energy, found_events = read_write.read_raw_regaxes(configs)
        npt.assert_allclose(found_energy, self.reg_energy1)
        npt.assert_allclose(found_events, local_ax_1)
        configs.n_dataset_files = 3
        found_energy, found_events = read_write.read_raw_regaxes(configs)
        npt.assert_allclose(found_energy, [2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0])
        npt.assert_allclose(found_events, np.tile(local_ax_1, (3, 1, 1)))

    def test_read_raw_regaxes_direction(self):
        # Again, but with the direction vector too
        configs = config_creator.make()
        configs.dataset_path = self.paths["empty"]
        configs.n_dataset_files = 0
        per_event_cols = ["energy", "direction"]
        found_cond, found_events = read_write.read_raw_regaxes(
            configs, per_event_cols=per_event_cols
        )
        assert len(found_cond) == 0
        assert len(found_events) == 0
        configs.dataset_path = self.paths["one_event"]
        for i in range(2):
            configs.n_dataset_files = i
            found_cond, found_events = read_write.read_raw_regaxes(
                configs, per_event_cols=per_event_cols
            )
            npt.assert_allclose(found_cond, [[1.0, 10.0, 10.0, 10.0]])
            npt.assert_allclose(
                found_events, [[[3.0, 1.0, 2.0, 4.0], [7.0, 5.0, 6.0, 8.0]]]
            )

        # this should work
        configs.dataset_path = self.paths["last_segment"]
        found_cond, found_events = read_write.read_raw_regaxes(
            configs, per_event_cols=per_event_cols
        )
        cond = np.hstack([self.reg_energy1[:, None], self.direction1[0]])
        npt.assert_allclose(found_cond, cond)
        local_ax_1 = self.reg_events1[..., [2, 0, 1, 3]]
        npt.assert_allclose(found_events, local_ax_1)

        # finally, the nine events should be fine
        configs.dataset_path = self.paths["nine_events"]
        configs.n_dataset_files = 1
        found_cond, found_events = read_write.read_raw_regaxes(
            configs, per_event_cols=per_event_cols
        )
        npt.assert_allclose(found_cond, cond)
        npt.assert_allclose(found_events, local_ax_1)
        configs.n_dataset_files = 3
        found_cond, found_events = read_write.read_raw_regaxes(
            configs, per_event_cols=per_event_cols
        )
        cond = np.tile(cond, (3, 1))
        npt.assert_allclose(found_cond, cond)
        npt.assert_allclose(found_events, np.tile(local_ax_1, (3, 1, 1)))


def test_regularise_event_axes():
    found = read_write.regularise_event_axes(TestReaders.events1)
    assert found.shape == (3, 2, 4)
    npt.assert_allclose(found, TestReaders.reg_events1)
    # but the events0 should be impossible without knowing if
    # the axis are transposed
    npt.assert_raises(
        RuntimeError, read_write.regularise_event_axes, TestReaders.events0
    )
    found = read_write.regularise_event_axes(TestReaders.events0, is_transposed=True)
    assert found.shape == (1, 4, 4)
    np.allclose(found, TestReaders.reg_events0)


def test_check_regaxes():
    read_write.check_regaxes(TestReaders.reg_energy0, TestReaders.reg_events0)
    read_write.check_regaxes(TestReaders.reg_energy1, TestReaders.reg_events1)
    npt.assert_raises(
        ValueError, read_write.check_regaxes, TestReaders.energy1, TestReaders.events1
    )


@TemporaryMetadata("test_W", target_folder="sim-E1261AT600AP180-180_file_W")
def test_write_raw_regaxes(tmpdir):
    path_0 = os.path.join(tmpdir, "test_0.h5")
    read_write.write_raw_regaxes(
        path_0, TestReaders.reg_energy0, TestReaders.reg_events0
    )
    configs = config_creator.make()
    configs.dataset_path = path_0
    # when the number of points is also 4 it is ambigus to read
    npt.assert_raises(RuntimeError, read_write.read_raw_regaxes, configs)
    path_1 = os.path.join(tmpdir, "test_1.h5")
    configs.dataset_path = path_1
    read_write.write_raw_regaxes(
        path_1, TestReaders.reg_energy1, TestReaders.reg_events1
    )
    found_energy, found_events = read_write.read_raw_regaxes(configs)
    npt.assert_allclose(found_energy, TestReaders.reg_energy1)
    npt.assert_allclose(found_events, TestReaders.reg_events1)
    # should raise an error if the axes are not regular
    npt.assert_raises(
        ValueError,
        read_write.write_raw_regaxes,
        path_1,
        TestReaders.energy1,
        TestReaders.events1,
    )


def test_events_to_local():
    events = np.array(
        [
            [
                [1.0, 1.01, 1.02, 1.03],
                [1.10, 1.11, 1.12, 1.13],
                [1.20, 1.21, 1.22, 1.23],
                [1.30, 1.31, 1.32, 1.33],
            ],
            [
                [2.0, 2.01, 2.02, 2.03],
                [2.10, 2.11, 2.12, 2.13],
                [2.20, 2.21, 2.22, 2.23],
                [2.30, 2.31, 2.32, 2.33],
            ],
        ]
    )
    before = events.copy()
    no_change = "hdf5:xyz==local:xyz"
    read_write.events_to_local(events, no_change)
    npt.assert_allclose(before, events)
    to_rotate = "hdf5:xyz==local:yzx"
    read_write.events_to_local(events, to_rotate)
    expected = before[:, :, [2, 0, 1, 3]]
    npt.assert_allclose(events, expected)
    invalid = "variable"
    npt.assert_raises(NotImplementedError, read_write.events_to_local, events, invalid)


def test_local_global_exchanges():
    events = np.array(
        [
            [
                [1.0, 1.01, 1.02, 1.03],
                [1.10, 1.11, 1.12, 1.13],
                [1.20, 1.21, 1.22, 1.23],
                [1.30, 1.31, 1.32, 1.33],
            ],
            [
                [2.0, 2.01, 2.02, 2.03],
                [2.10, 2.11, 2.12, 2.13],
                [2.20, 2.21, 2.22, 2.23],
                [2.30, 2.31, 2.32, 2.33],
            ],
        ]
    )
    # start with local_to_global
    no_change = "hdf5:xyz==local:xyz"
    no_change_global = "hdf5:xyz==global:xyz"
    # check empty events don't make it choke
    empty_events = np.empty((0, 4, 4))
    found = read_write.local_to_global(empty_events, no_change, no_change_global)
    assert empty_events.shape == found.shape
    # check some simple cases
    found = read_write.local_to_global(events, no_change, no_change_global)
    npt.assert_allclose(events, found)
    to_rotate = "hdf5:xyz==local:yzx"
    found = read_write.local_to_global(events, to_rotate, no_change_global)
    expected = events[:, :, [1, 2, 0, 3]]
    npt.assert_allclose(found, expected)
    npt.assert_raises(
        NotImplementedError, read_write.local_to_global, events, no_change, no_change
    )
    # now global_to_local
    found = read_write.global_to_local(empty_events, no_change, no_change_global)
    assert empty_events.shape == found.shape
    found = read_write.global_to_local(events, no_change, no_change_global)
    npt.assert_allclose(events, found)
    found = read_write.global_to_local(events, to_rotate, no_change_global)
    expected = events[:, :, [2, 0, 1, 3]]
    npt.assert_allclose(found, expected)
