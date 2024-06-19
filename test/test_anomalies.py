""" Module to test the anomalies module. """
import os
import numpy as np
import numpy.testing as npt
from unittest.mock import patch
from torch_geometric.data import Data, DataLoader
import torch
import tempfile
import shutil

from pointcloud.anomalies import prep_data, autoencoder, train, detect
from pointcloud.config_varients import default
from pointcloud.utils import metadata
from pointcloud.data import trees

from helpers.mock_read_write import mock_read_raw_regaxes, mock_get_n_events
from helpers import sample_trees

# ## prep_data.py ###


@patch("pointcloud.data.trees.get_n_events", new=mock_get_n_events)
@patch("pointcloud.data.trees.read_raw_regaxes", new=mock_read_raw_regaxes)
def test_format_data():
    configs = default.Configs()
    data_as_trees = trees.DataAsTrees(configs, max_trees_in_memory=2)

    features, edges = prep_data.format_data(data_as_trees, [0, 1])
    assert len(features) == 2
    assert len(edges) == 2

    tree0 = data_as_trees.get([0])[0]
    max_skips = tree0.max_skips
    number_of_children = max_skips + 1
    number_of_features = 5 + number_of_children
    n_points = 3 + 1  # for the root
    assert features[0].shape == (n_points, number_of_features)
    assert edges[0].shape == (2, n_points - 1)

    root_xz = metadata.Metadata(data_as_trees.configs).gun_xz_pos_raw

    energies = features[0][:, 3]
    energy_order = np.argsort(energies)
    npt.assert_allclose(energies[energy_order], [1.0, 1.0, 2.0, 3.0])

    xs = features[0][energy_order, 0]
    npt.assert_allclose(xs, np.array([root_xz[0], 0.0, 0.0, 1.0]))

    zs = features[0][energy_order, 2]
    npt.assert_allclose(zs, np.array([root_xz[1], 0.0, 1.0, 0.0]))

    incident_energy = features[0][energy_order, 4]
    npt.assert_allclose(incident_energy, np.array([1.0, 1.0, 1.0, 1.0]))

    empty_tree = sample_trees.empty()
    empty_tree.incident = 1.0

    def alternate_get(sample_idxs):
        n_idxs = len(sample_idxs)
        return [empty_tree] * n_idxs

    data_as_trees.get = alternate_get
    features, edges = prep_data.format_data(data_as_trees, [0, 1])
    assert len(features) == 2
    assert len(edges) == 2
    assert features[0].shape == (1, 6 + empty_tree.max_skips)
    assert edges[0].shape == (2, 0)


@patch("pointcloud.data.trees.get_n_events", new=mock_get_n_events)
@patch("pointcloud.data.trees.read_raw_regaxes", new=mock_read_raw_regaxes)
def test_direct_data():
    configs = default.Configs()
    configs.device = "cpu"
    data = prep_data.direct_data(configs, "dummy")
    for item in data:
        # check it's a torch_geometric.data.Data object
        assert isinstance(item, Data)


@patch("pointcloud.data.trees.get_n_events", new=mock_get_n_events)
@patch("pointcloud.data.trees.read_raw_regaxes", new=mock_read_raw_regaxes)
def test_format_save(tmpdir):
    configs = default.Configs()
    configs.device = "cpu"
    save_dir = tmpdir.mkdir("test_format_save")
    configs.formatted_tree_base = os.path.join(save_dir, "formatted_trees")

    prep_data.format_save(configs)
    features_path = configs.formatted_tree_base + "_features.npz"
    edges_path = configs.formatted_tree_base + "_edges.npz"
    assert os.path.exists(features_path)
    assert os.path.exists(edges_path)

    root_xz = metadata.Metadata(configs).gun_xz_pos_raw

    features = np.load(features_path)["arr_0"]
    edges = np.load(edges_path)["arr_0"]

    energies = features[:, 3]
    energy_order = np.argsort(energies)
    npt.assert_allclose(energies[energy_order], [1.0, 1.0, 2.0, 3.0])

    xs = features[energy_order, 0]
    npt.assert_allclose(xs, np.array([root_xz[0], 0.0, 0.0, 1.0]))

    zs = features[energy_order, 2]
    npt.assert_allclose(zs, np.array([root_xz[1], 0.0, 1.0, 0.0]))

    incident_energy = features[energy_order, 4]
    npt.assert_allclose(incident_energy, np.array([1.0, 1.0, 1.0, 1.0]))

    assert edges.shape == (2, 3)


### autoencoder.py ###


def test_GraphAutoencoder(tmpdir):
    """Test the GraphAutoencoder class."""
    # Create a GraphAutoencoder object.
    ae = autoencoder.GraphAutoencoder(6, 8)
    features = torch.tensor(np.random.rand(4, 6), dtype=torch.float32)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data = Data(x=features, edge_index=edges)
    # forward pass test
    output = ae.forward(data)
    assert output.shape == (4, 6)

    # save and load test
    initial_state_dict = ae.state_dict()
    save_path = str(tmpdir.join("ae.pth"))
    ae.save(save_path)

    ae2 = autoencoder.GraphAutoencoder.load(save_path)
    loaded_state_dict = ae2.state_dict()
    for key in initial_state_dict:
        npt.assert_allclose(
            initial_state_dict[key], loaded_state_dict[key], err_msg=f"key: {key}"
        )


### train.py ###


def test_checkpoint_location(tmpdir):
    folder = str(tmpdir.mkdir("test_checkpoint_locations"))
    file_path = train.checkpoint_location(0.1, folder)
    directory = os.path.dirname(file_path)
    assert os.path.exists(directory)

    folder = os.path.join(folder, "subfolder")
    file_path = train.checkpoint_location(0.1, folder)
    directory = os.path.dirname(file_path)
    assert os.path.exists(directory)


def test_get_critrion():
    criterion = train.get_criterion()
    sample = torch.tensor([1.0, 0.0], dtype=torch.float32)
    found = criterion(sample, sample)
    npt.assert_allclose(found, 0.0)
    sample2 = torch.tensor([1.0, 1.0], dtype=torch.float32)
    found2 = criterion(sample, sample2)
    assert found2 > 0.0


# Make an enviroment that creates a sample dataset
class SampleDataset:
    tmpdir = tempfile.mkdtemp()
    configs = default.Configs()
    configs_empty = default.Configs()
    n_features = 8
    sample_model = os.path.join(tmpdir, "sample_model.pth")

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
        The dataset requires data on disk to read, so we need to create
        some test data.
        """
        cls.configs.device = "cpu"
        cls.configs.anomaly_checkpoint = cls.mkdir("checkpoints")
        cls.configs.anomaly_hidden_dim = 8
        # create some fake data
        features = []
        edges = []
        features.append(np.random.rand(4, cls.n_features))
        edges.append(np.random.randint(0, 4, (2, 3)))

        features.append(np.random.rand(1, cls.n_features))
        edges.append(np.random.randint(0, 0, (2, 0)))

        features.append(np.random.rand(2, cls.n_features))
        edges.append(np.random.randint(0, 2, (2, 1)))

        dataset_dir = cls.mkdir("dataset")
        data_base = os.path.join(dataset_dir, "tree_")
        cls.configs.formatted_tree_base = data_base
        features_path = data_base + "_features.npz"
        np.savez(features_path, *features)
        edges_path = data_base + "_edges.npz"
        np.savez(edges_path, *edges)

        empty_dir = cls.mkdir("empty")
        empty_base = os.path.join(empty_dir, "tree_")
        cls.configs_empty.formatted_tree_base = empty_base
        features_path = empty_base + "_features.npz"
        np.savez(features_path)
        edges_path = empty_base + "_edges.npz"
        np.savez(edges_path)

        # make a sample model
        hidden_dim = cls.configs.anomaly_hidden_dim
        model = autoencoder.GraphAutoencoder(cls.n_features, hidden_dim)
        model.save(cls.sample_model)

    @classmethod
    def teardown_class(cls):
        """
        Remove the test data from disk.
        """
        # remove the test data
        shutil.rmtree(cls.tmpdir)



class TestTrain(SampleDataset):
    tmpdir = tempfile.mkdtemp()
    configs = default.Configs()
    configs_empty = default.Configs()
    n_features = 8
    sample_model = os.path.join(tmpdir, "sample_model.pth")


    def test_TreeDataset(self):
        dataset = train.TreeDataset(self.configs)
        assert len(dataset) == 3
        item0 = dataset[0]
        assert isinstance(item0, Data)
        assert item0.x.shape == (4, 8)
        assert item0.edge_index.shape == (2, 3)

        item1 = dataset[1]
        assert isinstance(item1, Data)
        assert item1.x.shape == (1, 8)
        assert item1.edge_index.shape == (2, 0)

        item2 = dataset[2]
        assert isinstance(item2, Data)
        assert item2.x.shape == (2, 8)
        assert item2.edge_index.shape == (2, 1)

        dataset_empty = train.TreeDataset(self.configs_empty)
        assert len(dataset_empty) == 0

    def test_epoch(self):
        model = autoencoder.GraphAutoencoder(8, 8)
        dataloader = DataLoader(train.TreeDataset(self.configs), batch_size=2)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = train.get_criterion()
        loss = train.epoch(model, dataloader, optimiser, criterion)
        assert not np.isnan(loss)

    def test_train(self):
        train.train(self.configs, self.sample_model, 1)


### detect.py ###


def test_save_path_for_model():
    model_path = "fiction/model.pth"
    save_path = detect.save_path_for_model(model_path)
    assert save_path.startswith(model_path.split(".")[0])
    assert save_path.endswith("_scores.npy")


user_input = []


def mock_input(prompt):
    return user_input.pop()


# replace the user input with a mocked value
@patch("builtins.input", new=mock_input)
def test_get_rating():
    global user_input
    user_input.append("0")
    rating = detect.get_rating()
    assert rating == 0

    user_input.append("1")
    user_input.append("-2")
    rating = detect.get_rating()
    assert rating == 1

    user_input.append("2")
    user_input.append("30")
    user_input.append("food")
    rating = detect.get_rating()
    assert rating == 2


class TestDetect(SampleDataset):
    tmpdir = tempfile.mkdtemp()
    configs = default.Configs()
    n_features = 8
    sample_model = os.path.join(tmpdir, "sample_model.pth")
    
    def test_score_data(self):
        dataset = train.TreeDataset(self.configs) 
        scores = detect.score_data(self.sample_model, dataset, 0, 3)
        assert scores.shape == (3, 2)
        npt.assert_allclose(scores[:, 0], [0, 1, 2])
        assert np.all(scores[:, 1] > 0)

    def test_detect(self):
        expected_path = detect.detect(self.sample_model, self.configs, 2)
        assert os.path.exists(expected_path)
        scores = np.load(expected_path)
        assert scores.shape == (3, 2)
        npt.assert_allclose(scores[:, 0], [0, 1, 2])
        assert np.all(scores[:, 1] > 0)
