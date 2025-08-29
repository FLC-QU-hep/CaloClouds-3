"""
Unit test for the script create_standard_metrics.py
"""
import os

from pointcloud.evaluation.bin_standard_metrics import get_path

from helpers.sample_models import write_fake_flow_model
from helpers import config_creator

from pointcloud.evaluation.bin_standard_metrics import (
    get_caloclouds_models,
)

from scripts.evaluation.create_standard_metrics import main


def fake_models(config, tmpdir):
    # test model
    test_cm_model_path = "test/example_cm_model.pt"
    # fake the flow model
    test_model_path = str(tmpdir) + "/example_flow_model.pt"
    write_fake_flow_model(config, test_model_path)
    models = get_caloclouds_models(test_cm_model_path, test_model_path, config=config)
    model, flow_dist, config = models["CaloClouds"][0]
    return models


def test_main(tmpdir):
    config = config_creator.make("caloclouds_3")
    config.cond_features = 2
    config.cond_features_names = ["energy", "points"]
    config.shower_flow_cond_features = ["energy"]
    config.log_dir = tmpdir

    models = fake_models(config, tmpdir)

    main(
        config=config,
        redo_g4_data=False,
        redo_g4_acc_data=False,
        redo_model_data=False,
        max_g4_events=10,
        max_model_events=0,
        models=models,
    )

    # check things actually got saved
    expected_path = get_path(config, "fake model")
    assert os.path.exists(expected_path)
    expected_path = get_path(config, "Geant 4")
    assert os.path.exists(expected_path)
    expected_path = get_path(config, "Geant 4 Accumulator")
    assert os.path.exists(expected_path)
