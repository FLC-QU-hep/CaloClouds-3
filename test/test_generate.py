"""
Module to test the functions in evaluation.generate
"""
import os
import torch
from pointcloud.evaluation import generate

from helpers import config_creator
from helpers.sample_models import write_fake_flow_model, write_fake_wish_model


def test_make_params_dict():
    params_dict = generate.make_params_dict()
    assert isinstance(params_dict, dict)
    assert "caloclouds_list" in params_dict
    assert "seed_list" in params_dict
    assert "min_energy_list" in params_dict
    assert "max_energy_list" in params_dict
    assert "n_events" in params_dict
    assert "batch_size" in params_dict
    assert "n_scaling" in params_dict
    assert "out_path" in params_dict
    assert "prefix" in params_dict


def test_load_flow_model_caloclouds(tmpdir):
    config = config_creator.make("caloclouds_3", my_tmpdir=tmpdir)
    config.shower_flow_num_blocks = 10
    config.shower_flow_cond_features = ["energy"]
    test_model_path = str(tmpdir) + "/example_flow_model.pt"
    write_fake_flow_model(config, test_model_path)
    _, distribution, transforms = generate.load_flow_model(
        config, model_path=test_model_path
    )
    assert isinstance(transforms, list)
    # ignoring the flow, as we don't use it in generate
    batch_size = 10
    cond_E_batch = torch.FloatTensor(batch_size, 1).uniform_(10, 20).to(config.device)
    samples = (
        distribution.condition(cond_E_batch / 100)
        .sample(
            torch.Size(
                [
                    batch_size,
                ]
            )
        )
        .cpu()
        .numpy()
    )
    # breakdown of the expected dimensions of the sample
    dim_num_clusters = 1
    dim_total_energy = 1
    dim_cog_x = 1
    dim_cog_y = 1
    dim_cog_z = 1
    dim_layer_energy = 30
    dim_layer_hits = 30
    expected_dim = (
        dim_num_clusters
        + dim_total_energy
        + dim_cog_x
        + dim_cog_y
        + dim_cog_z
        + dim_layer_energy
        + dim_layer_hits
    )
    assert samples.shape == (batch_size, expected_dim)
    assert samples.dtype == "float32"


def test_load_flow_model_wish(tmpdir):
    config = config_creator.make("wish", my_tmpdir=tmpdir)
    config.shower_flow_num_blocks = 10
    flow, distribution = generate.load_flow_model(config, model_path="dummy")
    assert flow is None
    assert distribution is None


def test_load_diffusion_model_calocloud():
    config = config_creator.make("caloclouds_3")
    config.shower_flow_num_blocks = 10
    config.shower_flow_cond_features = ["energy"]
    # only testing the cm model, becuse the file is small and
    # can be kept in the test dir of the repo
    model_name = "cm"
    test_model_path = "test/example_cm_model.pt"
    model, coef_real, coef_fake, n_splines = generate.load_diffusion_model(
        config, model_name, model_path=test_model_path
    )
    assert isinstance(model, torch.nn.Module)
    assert coef_real.shape == (4,)
    assert coef_fake.shape == (4,)
    assert n_splines is None
    # check the model can be sampled from
    batch_size = 5
    max_hits = 100
    total_hits = torch.ones((batch_size, 1), device=config.device)
    total_energy = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
    conditioning = torch.cat([total_hits, total_energy], -1)

    samples = model.sample(conditioning, max_hits, config)
    assert samples.shape == (batch_size, max_hits, 4)


def test_load_diffusion_model_wish(tmpdir):
    config = config_creator.make("wish", my_tmpdir=tmpdir)
    test_model_path = os.path.join(str(tmpdir), "wish_model.pt")
    write_fake_wish_model(config, test_model_path)
    model_name = "wish"

    model, coef_real, coef_fake, n_splines = generate.load_diffusion_model(
        config, model_name, model_path=test_model_path
    )
    assert isinstance(model, torch.nn.Module)
    assert coef_real is None
    assert coef_fake is None
    assert n_splines is None
    # check the model can be sampled from
    batch_size = 5
    max_hits = 100
    total_hits = torch.ones((batch_size, 1), device=config.device)
    total_energy = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
    conditioning = torch.cat([total_hits, total_energy], -1)

    samples = model.sample(conditioning, max_hits)
    assert samples.shape == (batch_size, max_hits, 4)


def test_generate_showers(tmpdir):
    config = config_creator.make("caloclouds_3", my_tmpdir=tmpdir)
    config.shower_flow_num_blocks = 10
    config.shower_flow_cond_features = ["energy"]
    params_dict = generate.make_params_dict()
    # make it short for testing
    params_dict["n_events"] = 10
    params_dict["batch_size"] = 2

    # fake the flow model
    test_model_path = str(tmpdir) + "/example_flow_model.pt"
    write_fake_flow_model(config, test_model_path)
    flow_model = generate.load_flow_model(config, model_path=test_model_path)

    diff_model = generate.load_diffusion_model(
        config, "cm", model_path="test/example_cm_model.pt"
    )
    showers, cond_E = generate.generate_showers(
        config, 10, 20, params_dict, flow_model, diff_model
    )
    assert showers.shape == (params_dict["n_events"], config.max_points, 4)
    assert cond_E.shape == (params_dict["n_events"], 1)


def test_write_showers(tmpdir):
    config = config_creator.make("caloclouds_3", my_tmpdir=tmpdir)
    config.shower_flow_num_blocks = 10
    config.shower_flow_cond_features = ["energy"]
    params_dict = generate.make_params_dict()
    # make it short for testing
    params_dict["n_events"] = 10
    params_dict["batch_size"] = 2
    params_dict["out_path"] = str(tmpdir)
    params_dict["prefix"] = "test"
    params_dict["caloclouds_list"] = ["cm"]

    # fake the flow model
    test_flow_model_path = str(tmpdir) + "/example_flow_model.pt"
    write_fake_flow_model(config, test_flow_model_path)

    test_diff_model_path = "test/example_cm_model.pt"

    generate.write_showers(config, params_dict, test_flow_model_path, test_diff_model_path)
    # probably should check the output.... TODO
