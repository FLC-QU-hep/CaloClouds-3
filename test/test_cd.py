import pytest
from scripts.training.cd import main

from helpers import config_creator


# The user warning is about the number of workers,
# but this number works well on our setup
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_main(tmpdir):
    # set a test config
    cfg = config_creator.make("caloclouds_3", my_tmpdir=tmpdir)
    main(cfg)
