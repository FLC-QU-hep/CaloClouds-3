# get the folder above on the path
import os
import glob
import pytest
from pointcloud.config_varients import default, wish
from scripts.main import main


def customise_configs(configs, my_tmpdir):
    configs.log_comet = False
    test_dir = os.path.dirname(os.path.realpath(__file__))
    configs.dataset_path = os.path.join(test_dir, "mini_data_sample.hdf5")
    configs.max_iters = 2
    configs.logdir = my_tmpdir.mkdir("logs")
    configs.device = "cpu"
    configs.fit_attempts = 2


# The user warning is about the number of workers,
# but this number works well on our setup
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_main_default(tmpdir):
    # set a test config
    configs = default.Configs()
    customise_configs(configs, tmpdir)
    main(configs)
    # check the model ckpt was created
    assert glob.glob(f"{configs.logdir}/{configs.name}*/ckpt_*.pt")


def test_main_wish(tmpdir):
    # set a test config
    configs = wish.Configs()
    customise_configs(configs, tmpdir)
    main(configs)
    # check the model ckpt was created
    assert glob.glob(f"{configs.logdir}/{configs.name}*/ckpt_*.pt")
