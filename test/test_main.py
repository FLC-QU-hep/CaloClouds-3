# get the folder above on the path
import glob
import pytest
from scripts.main import main
from helpers import config_creator


def test_calotranf(tmpdir):
    cfg = config_creator.make("configs_calotransf", my_tmpdir=tmpdir)
    # no logging for tests, as we would need a comet key
    # run from scratch
    # cfg.model_path = ""
    main(cfg)
    # assert glob.glob(f"{cfg.logdir_uda}/{cfg.name}*/ckpt_*.pt")
    # the calotransf needs to check that a recent checkpoint can be loaded.
    # cfg.model_path = "pointcloud/calotransfer/pretrained/ckpt_0.000000_2000000.pt"
    # shutil.copy(os.path.join(test_dir, cfg.model_path), cfg.logdir_uda)
    # print(os.listdir(cfg.logdir_uda))
    # main(cfg)
    # check the model ckpt was created
    # assert glob.glob(f"{cfg.logdir}/{cfg.name}*/ckpt_*.pt")



# The user warning is about the number of workers,
# but this number works well on our setup
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_main_default(tmpdir):
    # set a test config
    configs = config_creator.make("default", my_tmpdir=tmpdir)
    main(configs)
    # check the model ckpt was created
    assert glob.glob(f"{configs.logdir}/{configs.name}*/ckpt_*.pt")


def test_main_wish(tmpdir):
    # set a test config
    configs = config_creator.make("wish", my_tmpdir=tmpdir)
    main(configs)
    # check the model ckpt was created
    assert glob.glob(f"{configs.logdir}/{configs.name}*/ckpt_*.pt")
