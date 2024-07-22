"""
Unit test for the script create_standard_metrics.py
"""
import os

from pointcloud.config_varients import wish

from pointcloud.evaluation.bin_standard_metrics import get_path

from helpers.sample_models import make_fake_wish_model
from helpers.sample_accumulator import make as make_fake_accumulator

from scripts.create_standard_metrics import main


def fake_models(configs):
    configs.fit_attempts = 2
    configs.poly_degree = 2
    model = make_fake_wish_model(configs)
    models = {"fake model": (model, None, configs)}
    return models


def fake_accumulator(logdir):
    accumulator = make_fake_accumulator()
    path = os.path.join(logdir, "fake_accumulator.h5")
    accumulator.save(path)
    return path


def test_main(tmpdir):
    configs = wish.Configs()
    configs.log_comet = False
    test_dir = os.path.dirname(os.path.realpath(__file__))
    configs.dataset_path = os.path.join(test_dir, "mini_data_sample.hdf5")
    configs.max_iters = 2
    configs.logdir = tmpdir.mkdir("logs")
    configs.device = "cpu"

    models = fake_models(configs)
    accumulator_path = fake_accumulator(configs.logdir)

    main(
        configs=configs,
        redo_g4_data=False,
        redo_g4_acc_data=False,
        redo_model_data=False,
        max_g4_events=10,
        max_model_events=0,
        models=models,
        accumulator_path=accumulator_path,
    )

    # check things actually got saved
    expected_path = get_path(configs, "fake model")
    assert os.path.exists(expected_path)
    expected_path = get_path(configs, "Geant 4")
    assert os.path.exists(expected_path)
    expected_path = get_path(configs, "Geant 4 Accumulator")
    assert os.path.exists(expected_path)
