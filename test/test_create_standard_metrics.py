"""
Unit test for the script create_standard_metrics.py
"""
import os
import numpy as np

from pointcloud.evaluation.bin_standard_metrics import get_path

from helpers.sample_models import write_fake_flow_model
from helpers import config_creator, example_paths

from pointcloud.evaluation.bin_standard_metrics import (
    get_caloclouds_models,
)

from scripts.evaluation.create_standard_metrics import main, get_configs


def fake_models(config, tmpdir):
    # test model
    test_cm_model_path = example_paths.example_cm_model
    # fake the flow model
    test_model_path = str(tmpdir) + "/example_flow_model.pt"
    write_fake_flow_model(config, test_model_path)
    models = get_caloclouds_models(test_cm_model_path, test_model_path, config=config)
    model, flow_dist, config = models["CaloClouds3"]
    return models


def test_main(tmpdir):
    config = config_creator.make("caloclouds_3")
    config.logdir = tmpdir

    g4_gun = np.array([40, 50, 0])
    cc3_model_gun = cc2_model_gun = g4_gun
    cc2_model_gun[2] -= 0.1
    cc2_model_gun[0] += 0.2
    guns = [g4_gun, cc2_model_gun, cc3_model_gun]

    models = fake_models(config, tmpdir)

    main(
        config,
        guns,
        False,
        True,
        True,
        redo_g4_data=False,
        redo_model_data=False,
        max_g4_events=10,
        max_model_events=0,
        models=models,
    )

    # check things actually got saved
    expected_path = get_path(config, "CaloClouds3")
    assert os.path.exists(expected_path)
    expected_path = get_path(config, "Geant 4")
    assert os.path.exists(expected_path)
