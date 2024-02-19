""" Module to test the metadata module. """
import sys
from pathlib import Path
path_root1 = Path(__file__).parents[1]
sys.path.append(str(path_root1))

from unittest.mock import patch
import os
import numpy as np

from utils import metadata
from configs import Configs

def test_get_metadata_folder():
    data_dir = metadata.get_metadata_folder()
    assert isinstance(data_dir, str)
    # try changing the config
    config = Configs()
    config.dataset_path = "varied_20_sample.h5py"
    data_dir = metadata.get_metadata_folder(config)
    assert isinstance(data_dir, str)
    assert "varied_20_sample" in data_dir


def test_Metadata(tmpdir):
    # save some things for the metadata to read
    array_path = os.path.join(str(tmpdir), "dog.npy")
    np.save(array_path, np.array([1, 2, 3]))
    dict_path = os.path.join(str(tmpdir), "cat.npy")
    np.save(dict_path, {"meow": 1, "purr": 2})

    # also mock the muon map subfolder
    os.mkdir(os.path.join(str(tmpdir), "muon_map"))
    muon_map_path_X = os.path.join(str(tmpdir), "muon_map/X.npy")
    np.save(muon_map_path_X, np.array([1, 2, 3]))
    muon_map_path_Y = os.path.join(str(tmpdir), "muon_map/Y.npy")
    np.save(muon_map_path_Y, np.array([3, 2, 3]))
    muon_map_path_Z = os.path.join(str(tmpdir), "muon_map/Z.npy")
    np.save(muon_map_path_Z, np.array([1, 4, 3]))
    muon_map_path_E = os.path.join(str(tmpdir), "muon_map/E.npy")
    np.save(muon_map_path_E, np.array([1, 6, 3]))
    
    with patch('utils.metadata.get_metadata_folder', return_value=str(tmpdir)):
        m = metadata.Metadata()
        assert hasattr(m, "dog")
        assert np.array_equal(m.dog, np.array([1, 2, 3]))
        assert not hasattr(m, "cat")
        assert hasattr(m, "meow")
        assert m.meow == 1
        assert hasattr(m, "purr")
        assert m.purr == 2

        # muon map loads only when requested
        assert not hasattr(m, "muon_map_X")
        m.load_muon_map()
        assert hasattr(m, "muon_map_X")
        assert hasattr(m, "muon_map_Y")
        assert hasattr(m, "muon_map_Z")
        assert hasattr(m, "muon_map_E")
        assert np.array_equal(m.muon_map_X, np.array([1, 2, 3]))
        assert np.array_equal(m.muon_map_Y, np.array([3, 2, 3]))
        assert np.array_equal(m.muon_map_Z, np.array([1, 4, 3]))
        assert np.array_equal(m.muon_map_E, np.array([1, 6, 3]))


