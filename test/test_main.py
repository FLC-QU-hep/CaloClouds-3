# get the folder above on the path
import shutil
import os
import glob
import pytest
from scripts.main import main
from helpers import config_creator


def test_calotranf(tmpdir):
    cfg = config_creator.make("configs_calotransf", my_tmpdir=tmpdir)
    # no logging for tests, as we would need a comet key
    # run from scratch
    cfg.model_path = ""
    main(cfg)
    model_glob = glob.glob(f"{cfg.logdir}/{cfg.name}*/ckpt_*.pt")
    assert model_glob
    # the calotransf needs to check that a recent checkpoint can be loaded.
    model_written = model_glob[-1]
    shutil.copy(model_written, cfg.logdir_uda)
    cfg.model_path = os.path.basename(model_written)
    main(cfg)
    # check a new model has been made
    second_glob = glob.glob(f"{cfg.logdir}/{cfg.name}*/ckpt_*.pt")
    assert len(second_glob) > len(model_glob)


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
    try:
        main(configs)
    except RuntimeError:
        # the sample data is too small to sensibly condition the model
        # it choses unphysical parameters
        pass
    # we can get round this by using a pretrained model
    test_dir = os.path.dirname(os.path.realpath(__file__))
    example_wish = os.path.join(test_dir, "example_wish_model.pt")
    configs.checkpoint_path = example_wish
    main(configs)
    # check the model ckpt was created
    assert glob.glob(f"{configs.logdir}/{configs.name}*/ckpt_*.pt")
