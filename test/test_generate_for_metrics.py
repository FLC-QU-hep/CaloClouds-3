"""
Module to test the functions in evaluation.generate
"""
from unittest.mock import patch
import argparse
import pickle
import numpy as np
from pointcloud.evaluation import generate_for_metrics
from test_generate import write_fake_flow_model

from helpers import config_creator


@patch(
    "argparse.ArgumentParser.parse_args",
    return_value=argparse.Namespace(caloclouds="test"),
)
def test_get_cli_args(mock_args):
    args = generate_for_metrics.get_cli_args()
    assert args.caloclouds == "test"


def test_make_params_dict():
    params = generate_for_metrics.make_params_dict("cm")
    assert isinstance(params, dict)
    assert params["total_events"] == 500_000
    assert params["n_events"] == 50_000
    assert params["min_energy"] == 10
    assert params["max_energy"] == 90
    assert (
        params["pickle_path"]
        == "/beegfs/desy/user/buhmae/6_PointCloudDiffusion/output/metrics/"
    )
    assert isinstance(params["n_scaling"], bool)
    assert params["n_scaling"]
    assert params["prefix"] == ""
    assert params["key_exceptions"] == [
        "e_radial",
        "e_layers",
        "e_layers_distibution",
        "occ_layers",
        "e_radial_lists",
    ]
    assert params["caloclouds"] == "cm"
    assert params["seed"] == 1234567
    assert params["batch_size"] == 16


def test_get_g4_data():
    test_data_path = "test/mini_data_sample.hdf5"
    all_events, all_energy = generate_for_metrics.get_g4_data(test_data_path)
    #assert isinstance(all_events, generate_for_metrics.lazy_loading.DatasetView)
    assert isinstance(all_events, np.ndarray)
    #assert isinstance(all_energy, h5py.Dataset)
    assert isinstance(all_energy, np.ndarray)
    n_events_in_sample = 2
    max_hits_per_event = 6000
    assert all_events.shape == (n_events_in_sample, max_hits_per_event, 4)
    assert all_energy.shape == (n_events_in_sample, 1)


def test_yield_g4_showers():
    config = config_creator.make()
    param_dict = generate_for_metrics.make_params_dict("cm")
    test_data_path = "test/mini_data_sample.hdf5"
    g4_data = generate_for_metrics.get_g4_data(test_data_path)
    max_hits_per_event = 6000
    for showers, cond_E in generate_for_metrics.yield_g4_showers(
        config, param_dict, g4_data
    ):
        assert showers.shape[1:] == (max_hits_per_event, 4)
        assert cond_E.shape[1] == 1
        assert cond_E.shape[0] == showers.shape[0]


def test_shower_generator_factory(tmpdir):
    # Need to test the g4 version
    config = config_creator.make()
    param_dict = generate_for_metrics.make_params_dict("g4")
    param_dict["g4_data_path"] = "test/mini_data_sample.hdf5"
    shower_generator = generate_for_metrics.shower_generator_factory(config, param_dict)
    max_hits_per_event = 6000
    for showers, cond_E in shower_generator():
        assert showers.shape[1:] == (max_hits_per_event, 4)
        assert cond_E.shape[1] == 1
        assert cond_E.shape[0] == showers.shape[0]

    # Also need to test the generator version,
    # will use only cm, as the other models are not available
    # in the repo as test examples
    param_dict = generate_for_metrics.make_params_dict("cm")
    param_dict["n_events"] = 10
    param_dict["batch_size"] = 2
    param_dict["diffusion_model_path"] = "test/example_cm_model.pt"
    test_model_path = str(tmpdir) + "/example_flow_model.pt"
    write_fake_flow_model(config, test_model_path)
    param_dict["flow_model_path"] = test_model_path
    shower_generator = generate_for_metrics.shower_generator_factory(config, param_dict)
    for showers, cond_E in shower_generator():
        assert showers.shape[1:] == (config.max_points, 4)
        assert cond_E.shape[1] == 1
        assert cond_E.shape[0] == showers.shape[0]
        assert cond_E.shape[0] == param_dict["n_events"]
        break


# TODO test plotting.get_features.... it's fairly important
def test_add_chunk():
    config = config_creator.make()
    param_dict = generate_for_metrics.make_params_dict("g4")
    param_dict["n_events"] = 50
    # make a fake shower generator
    max_hits_per_event = 6000

    def shower_generator_func():
        fake_events = np.random.rand(param_dict["n_events"], max_hits_per_event, 4)
        fake_energy = np.random.rand(param_dict["n_events"], 1)
        while True:
            yield fake_events, fake_energy

    my_generator = shower_generator_func()
    merged_dict = {}
    generate_for_metrics.add_chunk(config, param_dict, my_generator, merged_dict)
    # stuff we should save
    assert "e_sum" in merged_dict
    assert "hits" in merged_dict
    assert "occ" in merged_dict
    assert "e_layers_std" in merged_dict
    assert "hits_noThreshold" in merged_dict
    assert "binned_layer_e" in merged_dict
    assert "binned_radial_e" in merged_dict

    # stuff we should not save
    assert "e_radial" not in merged_dict
    assert "e_layers" not in merged_dict
    assert "e_layers_distibution" not in merged_dict
    assert "occ_layers" not in merged_dict
    assert "e_radial_lists" not in merged_dict

    # if we call it twice, the data should get twice as long
    inital_length = len(merged_dict["e_sum"])
    generate_for_metrics.add_chunk(config, param_dict, my_generator, merged_dict)
    assert len(merged_dict["e_sum"]) == inital_length * 2


def test_write_merge_dict(tmpdir):
    param_dict = generate_for_metrics.make_params_dict("cm")
    param_dict["pickle_path"] = str(tmpdir)

    dummy_merged_dict = {"test": np.random.rand(10, 10), "test2": 3}
    dummy_merged_dict["min_energy"] = 10
    dummy_merged_dict["max_energy"] = 90
    dummy_merged_dict["total_events"] = 50
    dummy_merged_dict["caloclouds"] = "cm"

    file_path = generate_for_metrics.write_merge_dict(param_dict, dummy_merged_dict)
    with open(file_path, "rb") as pickle_file:
        pickled_dict = pickle.load(pickle_file)
        assert pickled_dict["test"].shape == (10, 10)
        assert pickled_dict["test2"] == 3
