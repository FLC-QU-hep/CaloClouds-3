"""
Some common setting for testing configs, regardless of the underlying config version.
"""
import os
import shutil
from pointcloud.config_varients import default, wish, caloclouds_3, configs_calotransf

dict_of_configs = {
    "default": default,
    "wish": wish,
    "caloclouds_3": caloclouds_3,
    "configs_calotransf": configs_calotransf,
}


def make(module_name="default", start_from=None, my_tmpdir=None):
    if start_from is not None:
        config = start_from
    else:
        config = dict_of_configs[module_name].Configs()
    # no logging for tests, as we would need a comet key
    config.log_comet = False
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    test_dir = os.path.abspath(test_dir)
    config.dataset_path = os.path.join(test_dir, "mini_data_sample.hdf5")
    config.max_iters = 2
    config.device = "cpu"
    config.fit_attempts = 2
    if my_tmpdir is not None:
        my_tmpdir = str(my_tmpdir)
        config.storage_base = my_tmpdir
        config.logdir = os.path.join(my_tmpdir, "logs")
        os.makedirs(config.logdir, exist_ok=True)
        config.logdir_uda = os.path.join(my_tmpdir, "logs_uda")
        os.makedirs(config.logdir_uda, exist_ok=True)
        config.model_path = "mini_ckpt_sample.pt"
        # actually put the model in the logdir
        shutil.copy(os.path.join(test_dir, config.model_path), config.logdir)
    return config
