"""
Module to test the functions in evaluation.generate
"""
import os
import torch
from pointcloud.models import load

from helpers import config_creator
from helpers.sample_models import write_fake_flow_model


def test_load_flow_model_caloclouds(tmpdir):
    config = config_creator.make("caloclouds_3", my_tmpdir=tmpdir)
    config.shower_flow_num_blocks = 10
    config.cond_features = 1
    config.cond_features_names = ["energy"]
    config.shower_flow_cond_features = ["energy"]
    test_model_path = str(tmpdir) + "/example_flow_model.pt"
    write_fake_flow_model(config, test_model_path)
    _, distribution, transforms = load.load_flow_model(
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


def test_load_diffusion_model_calocloud():
    config = config_creator.make("caloclouds_3")
    config.shower_flow_num_blocks = 10
    config.cond_features = 2
    config.cond_features_names = ["energy", "points"]
    config.shower_flow_cond_features = ["energy"]
    # only testing the cm model, becuse the file is small and
    # can be kept in the test dir of the repo
    model_name = "cm"
    test_model_path = "test/example_cm_model.pt"
    model, coef_real, coef_fake, n_splines = load.load_diffusion_model(
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


