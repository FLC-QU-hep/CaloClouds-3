"""
Tests for the misc module.

We don't test the flow model as it's likely to be phased out in the future.
"""
import os
import torch
import numpy as np
import random

from pointcloud.utils import misc


def test_BlackHole():
    bh = misc.BlackHole()
    bh.mine = 5
    bh("oops")
    thing = bh.mine
    assert bh == thing


def test_CheckpointManager(tmpdir):
    checkpoint_dir = str(tmpdir / "checkpoints")
    cm = misc.CheckpointManager(checkpoint_dir)
    assert cm.save_dir == checkpoint_dir
    # it should create the directory if it doesn't exist
    assert os.path.isdir(checkpoint_dir)
    assert len(cm.ckpts) == 0

    idx = cm.get_worst_ckpt_idx()
    assert idx is None
    idx = cm.get_best_ckpt_idx()
    assert idx is None
    idx = cm.get_latest_ckpt_idx()
    assert idx is None

    demo_1 = torch.nn.Linear(10, 10)
    cm.save(demo_1, (0.5, 1), 5.0, step=0)
    assert len(cm.ckpts) == 1
    assert cm.get_worst_ckpt_idx() == 0
    assert cm.get_best_ckpt_idx() == 0
    assert cm.get_latest_ckpt_idx() == 0

    demo_2 = torch.nn.Linear(10, 10)
    cm.save(demo_2, (0.5, 3), 10.0, step=1)
    best = cm.get_best_ckpt_idx()
    worst = cm.get_worst_ckpt_idx()
    latest = cm.get_latest_ckpt_idx()
    assert latest == worst
    assert best != worst

    best = cm.load_best()
    assert best["args"][0] == 0.5
    assert best["args"][1] == 1

    for key in best["state_dict"]:
        assert torch.equal(best["state_dict"][key], demo_1.state_dict()[key])
    assert best["others"] is None

    reloaded_cm = misc.CheckpointManager(checkpoint_dir)
    assert len(reloaded_cm.ckpts) == 2
    reloaded_best = reloaded_cm.load_best()
    assert reloaded_best["args"][0] == best["args"][0]
    assert reloaded_best["args"][1] == best["args"][1]


def test_seed_all():
    seed = 55
    misc.seed_all(seed)
    assert torch.initial_seed() == seed
    np_random_state = np.random.get_state()
    np.random.seed(seed)
    expected_state = np.random.get_state()
    assert np.all(np_random_state[1] == expected_state[1])
    random_state = random.getstate()
    random.seed(seed)
    expected_state = random.getstate()
    assert random_state[1] == expected_state[1]


def test_get_new_log_dir(tmpdir):
    prefix = "good_prefix"
    log_dir = misc.get_new_log_dir(str(tmpdir), prefix=prefix)
    assert os.path.isdir(log_dir)
    assert os.path.basename(log_dir).startswith(prefix)


def test_mean_flat():
    n_batches = 10
    x = torch.rand(n_batches, 10, 3, 4)
    mean = misc.mean_flat(x)
    assert mean.shape == (n_batches,)
    for i in range(n_batches):
        assert torch.allclose(mean[i], x[i].mean())
