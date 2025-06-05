"""
Unit test for the script create_standard_metrics.py
"""
import os

from pointcloud.evaluation.bin_standard_metrics import get_path

from helpers.sample_models import make_fake_wish_model
from helpers.sample_accumulator import make as make_fake_accumulator
from helpers import config_creator

from scripts.create_standard_metrics import main


def fake_models(config):
    config.fit_attempts = 2
    config.poly_degree = 2
    model = make_fake_wish_model(config)
    models = {"fake model": (model, None, config)}
    return models


def fake_accumulator(logdir):
    accumulator = make_fake_accumulator()
    path = os.path.join(logdir, "fake_accumulator.h5")
    accumulator.save(path)
    return path


def test_main(tmpdir):
    config = config_creator.make("wish", my_tmpdir=tmpdir)

    models = fake_models(config)
    accumulator_path = fake_accumulator(config.logdir)

    main(
        config=config,
        redo_g4_data=False,
        redo_g4_acc_data=False,
        redo_model_data=False,
        max_g4_events=10,
        max_model_events=0,
        models=models,
        accumulator_path=accumulator_path,
    )

    # check things actually got saved
    expected_path = get_path(config, "fake model")
    assert os.path.exists(expected_path)
    expected_path = get_path(config, "Geant 4")
    assert os.path.exists(expected_path)
    expected_path = get_path(config, "Geant 4 Accumulator")
    assert os.path.exists(expected_path)
