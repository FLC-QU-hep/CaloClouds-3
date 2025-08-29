# get the folder above on the path
import shutil
import os
import glob
import pytest
from scripts.training.main import main
from helpers import config_creator


# The user warning is about the number of workers,
# but this number works well on our setup
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_main_default(tmpdir):
    # set a test config
    config = config_creator.make("default", my_tmpdir=tmpdir)
    main(config)
    # check the model ckpt was created
    assert glob.glob(f"{config.logdir}/{config.name}*/ckpt_*.pt")

