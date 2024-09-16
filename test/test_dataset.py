""" Module to test the data/dataset.py module. """

import os
import tempfile
import shutil
import h5py
import numpy as np
import numpy.testing as npt

from pointcloud.data import dataset, read_write

from helpers import config_creator


def make_test_batch(key, dataset_class, trim=False):
    if key == "0":
        normed_copy = TestData.events0.copy()
        energy = TestData.energy0
        points = TestData.points0
    elif key == "1":
        normed_copy = TestData.events1.copy()[:, :2]
        energy = TestData.energy1
        points = TestData.points1
    elif key == "01":
        normed_copy = np.concatenate([TestData.events0, TestData.events1], axis=0)
        energy = np.concatenate([TestData.energy0, TestData.energy1])
        points = np.concatenate([TestData.points0, TestData.points1])
    else:
        raise ValueError("Invalid key")
    read_write.events_to_local(normed_copy, dataset_class.metadata.orientation)
    dataset_class.normalize_xyze(normed_copy)
    if trim:
        normed_copy = normed_copy[:, :trim]
        points = np.minimum(points, trim)
    if len(energy.shape) == 1:
        energy = energy[:, np.newaxis]
    if len(points.shape) == 1:
        points = points[:, np.newaxis]
    test_batch = {
        "energy": energy,
        "event": normed_copy,
        "points": points,
    }
    return test_batch


def compare_batches(batch1, batch2):
    """
    Compare two batches of data, ignoring the order of the events.
    """
    assert sorted(batch1.keys()) == sorted(batch2.keys()), "Keys do not match"
    batch1_energy_order = np.argsort(batch1["energy"].flatten())
    batch2_energy_order = np.argsort(batch2["energy"].flatten())
    assert len(batch1_energy_order) == len(
        batch2_energy_order
    ), "Different number of events"
    for key in batch1.keys():
        npt.assert_array_equal(
            batch1[key][batch1_energy_order],
            batch2[key][batch2_energy_order],
            err_msg=f"Key {key} does not match",
        )


class TestData:
    energy0 = np.array([1.0])
    points0 = np.array([4])
    # one event, 4 points
    events0 = np.array(
        [
            [
                [1.0, 1.1, 1.2, 1.3],
                [1.01, 1.11, 1.21, 1.31],
                [1.02, 1.12, 1.22, 1.32],
                [1.03, 1.13, 1.23, 1.33],
            ]
        ]
    )

    # three events, 1 point, 2 points, 1 point
    # keep the padding at 4 points, otherwise it's really hard to process them
    # as the same dataset
    energy1 = np.array([2.0, 3.0, 4.0])
    points1 = np.array([1, 2, 1])
    events_data1 = [
        [[2.0, 2.1, 2.2, 2.3]],
        [[3.0, 3.1, 3.2, 3.3], [3.01, 3.11, 3.21, 3.31]],
        [[4.0, 4.1, 4.2, 4.3]],
    ]
    events1 = np.zeros((3, 4, 4))
    for i, n_points in enumerate(points1):
        events1[i, :n_points] = events_data1[i]

    @classmethod
    def write_0(cls, filename):
        with h5py.File(filename, "w") as f:
            f.create_dataset("energy", data=cls.energy0)
            f.create_dataset("n_points", data=cls.points0)
            f.create_dataset("events", data=cls.events0)

    @classmethod
    def write_1(cls, filename):
        with h5py.File(filename, "w") as f:
            # don't write n_points, make the dataset calculate it
            f.create_dataset("energy", data=cls.energy1)
            f.create_dataset("events", data=cls.events1)


class AbsDatasetTest:
    paths = {}

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
        The datasets require data on disk to read, so we need to create
        some test data.
        """
        cls.tmpdir = tempfile.mkdtemp()
        # three folders
        # one with just datasets 0
        # one with just datasets 1
        # one with datasets 0 and 1

        folder0 = cls.mkdir("just0")
        cls.paths["data0"] = os.path.join(folder0, "data_0.h5")
        TestData.write_0(cls.paths["data0"])

        folder1 = cls.mkdir("just1")
        cls.paths["data1"] = os.path.join(folder1, "data_1.h5")
        TestData.write_1(cls.paths["data1"])

        folder01 = cls.mkdir("both")
        cls.paths["data01"] = os.path.join(folder01, "data_{}.h5")
        TestData.write_0(cls.paths["data01"].format(0))
        TestData.write_1(cls.paths["data01"].format(1))

    @classmethod
    def teardown_class(cls):
        """
        Remove the test data from disk.
        """
        shutil.rmtree(cls.tmpdir)


class TestPointCloudDataset(AbsDatasetTest):
    ds_class = dataset.PointCloudDataset
    ds_class.metadata.layer_bottom_pos_hdf5 = np.linspace(0, 10, 30)
    ds_class.metadata.cell_thickness_hdf5 = 1 / 3

    def test_init(self):
        # For data0
        ds0 = self.ds_class(
            self.paths["data0"], bs=10, max_ds_seq_len=4, quantized_pos=False
        )
        assert len(ds0.open_files) == 1
        npt.assert_array_equal(ds0.index_list, [(4, 0, 0)])
        assert not ds0.front_padded
        assert ds0.bs == 10
        assert not ds0.quantized_pos
        assert not ds0._roll_axis
        assert len(ds0) == 1

        # try imposing a max_ds_seq_len
        ds0 = self.ds_class(
            self.paths["data0"], bs=10, max_ds_seq_len=2, quantized_pos=False
        )
        npt.assert_array_equal(ds0.index_list, [(2, 0, 0)])

        # For data1
        ds1 = self.ds_class(
            self.paths["data1"], bs=10, max_ds_seq_len=4, quantized_pos=False
        )
        assert len(ds1.open_files) == 1
        # remober it is sorted
        npt.assert_array_equal(ds1.index_list, [(1, 0, 0), (1, 0, 2), (2, 0, 1)])
        assert not ds1.front_padded
        assert ds1.bs == 10
        assert not ds1.quantized_pos
        assert not ds1._roll_axis
        assert len(ds1) == 3

        # for data01
        ds01 = self.ds_class(
            self.paths["data01"],
            bs=10,
            max_ds_seq_len=4,
            quantized_pos=False,
            n_files=2,
        )
        assert len(ds01.open_files) == 2
        npt.assert_array_equal(
            ds01.index_list, [(1, 1, 0), (1, 1, 2), (2, 1, 1), (4, 0, 0)]
        )
        assert not ds01.front_padded
        assert ds01.bs == 10
        assert not ds01.quantized_pos
        assert not ds01._roll_axis
        assert len(ds01) == 4

    def test_getitem(self):
        # For data0
        ds0 = self.ds_class(
            self.paths["data0"], bs=10, max_ds_seq_len=4, quantized_pos=True
        )
        batch = ds0[0]
        test_batch = make_test_batch("0", self.ds_class)
        compare_batches(batch, test_batch)

        # try imposing a max_ds_seq_len
        ds0 = self.ds_class(
            self.paths["data0"], bs=10, max_ds_seq_len=2, quantized_pos=True
        )
        batch = ds0[0]
        test_batch = make_test_batch("0", self.ds_class, trim=2)
        compare_batches(batch, test_batch)

        # For data1
        ds1 = self.ds_class(
            self.paths["data1"], bs=10, max_ds_seq_len=4, quantized_pos=True
        )
        batch = ds1[0]
        test_batch = make_test_batch("1", self.ds_class)
        compare_batches(batch, test_batch)

        # for data01
        ds01 = self.ds_class(
            self.paths["data01"],
            bs=10,
            max_ds_seq_len=4,
            quantized_pos=True,
            n_files=2,
        )
        batch = ds01[0]
        test_batch = make_test_batch("01", self.ds_class)
        compare_batches(batch, test_batch)

        # and limit the batch size
        ds01 = self.ds_class(
            self.paths["data01"],
            bs=2,
            max_ds_seq_len=4,
            quantized_pos=True,
            n_files=2,
        )
        batch = ds01[0]
        test_batch = {
            "energy": test_batch["energy"][[1, 3]],
            "event": test_batch["event"][[1, 3], :1],
            "points": test_batch["points"][[1, 3]],
        }
        compare_batches(batch, test_batch)

    def test_get_n_points(self):
        found = dataset.PointCloudDataset.get_n_points(TestData.events0)
        npt.assert_array_equal(found, TestData.points0)
        found = dataset.PointCloudDataset.get_n_points(TestData.events1)
        npt.assert_array_equal(found, TestData.points1)

    def test_normalize_xyze(self):
        xmin = dataset.PointCloudDataset.metadata.Xmin_global
        xmax = dataset.PointCloudDataset.metadata.Xmax_global
        ymin = dataset.PointCloudDataset.metadata.Zmin_global
        ymax = dataset.PointCloudDataset.metadata.Zmax_global
        zmin = dataset.PointCloudDataset.metadata.layer_bottom_pos_hdf5[0]
        zmax = (
            dataset.PointCloudDataset.metadata.layer_bottom_pos_hdf5[-1]
            + dataset.PointCloudDataset.metadata.cell_thickness_hdf5
        )
        # make data
        fake_event = np.vstack(
            (
                np.linspace(xmin, xmax, 10),
                np.linspace(ymin, ymax, 10),
                np.linspace(zmin, zmax, 10),
                np.arange(10) + 1,
            )
        ).T
        fake_events = np.tile(fake_event, (2, 1, 1))

        # for a single event
        dataset.PointCloudDataset.normalize_xyze(fake_event)
        npt.assert_allclose(
            fake_event[:, 0], np.linspace(-1, 1, 10), atol=1e-3, rtol=1e-3
        )
        npt.assert_allclose(
            fake_event[:, 1], np.linspace(-1, 1, 10), atol=1e-3, rtol=1e-3
        )
        npt.assert_allclose(
            fake_event[:, 2], np.linspace(-1, 1, 10), atol=1e-3, rtol=1e-3
        )
        npt.assert_allclose(
            fake_event[:, 3] / dataset.PointCloudDataset.energy_scale, np.arange(10) + 1
        )

        # for a batch of events
        dataset.PointCloudDataset.normalize_xyze(fake_events)
        last_event = fake_events[-1]
        npt.assert_allclose(
            last_event[:, 0], np.linspace(-1, 1, 10), atol=1e-3, rtol=1e-3
        )
        npt.assert_allclose(
            last_event[:, 1], np.linspace(-1, 1, 10), atol=1e-3, rtol=1e-3
        )
        npt.assert_allclose(
            last_event[:, 2], np.linspace(-1, 1, 10), atol=1e-3, rtol=1e-3
        )
        npt.assert_allclose(
            last_event[:, 3] / dataset.PointCloudDataset.energy_scale, np.arange(10) + 1
        )


class TestPointCloudDatasetUnordered(AbsDatasetTest):
    ds_class = dataset.PointCloudDatasetUnordered
    ds_class.metadata.layer_bottom_pos_hdf5 = np.linspace(0, 10, 30)
    ds_class.metadata.cell_thickness_hdf5 = 1 / 3

    def test_getitem(self):
        # in this dataset, the events are delivered at random
        # For data0, only one event, there is no difference
        ds0 = self.ds_class(
            self.paths["data0"], bs=10, max_ds_seq_len=4, quantized_pos=True
        )
        batch = ds0[0]
        test_batch = make_test_batch("0", self.ds_class)
        compare_batches(batch, test_batch)

        ds1 = self.ds_class(
            self.paths["data1"], bs=10, max_ds_seq_len=4, quantized_pos=True
        )
        batch = ds1[0]
        test_batch = make_test_batch("1", self.ds_class)
        compare_batches(batch, test_batch)

        # for data01
        ds01 = self.ds_class(
            self.paths["data01"],
            bs=10,
            max_ds_seq_len=4,
            quantized_pos=True,
            n_files=2,
        )
        batch = ds01[0]
        test_batch = make_test_batch("01", self.ds_class)
        compare_batches(batch, test_batch)


def test_dataset_class_from_config():
    # set a test config
    cfg = config_creator.make()

    # try for x36_grid datasets
    cfg.dataset = "x36_grid"
    dataset_class = dataset.dataset_class_from_config(cfg)
    ds = dataset_class(cfg.dataset_path)
    assert isinstance(ds, dataset.PointCloudDataset)
    batch = ds[0]
    assert len(batch["energy"].shape) == 2
    n_events = batch["energy"].shape[0]
    assert batch["event"].shape[0] == n_events
    assert batch["event"].shape[2] == 4

    # try for getting_high
    cfg.dataset = "getting_high"
    dataset_class = dataset.dataset_class_from_config(cfg)
    ds = dataset_class(cfg.dataset_path)
    assert isinstance(ds, dataset.PointCloudDatasetGH)
    batch = ds[0]
    assert len(batch["energy"].shape) == 2
    n_events = batch["energy"].shape[0]
    assert batch["event"].shape[0] == n_events
    assert batch["event"].shape[2] == 4
