"""
Test the gen_utils (generator utilities) module.
"""
import numpy as np
import torch
from numpy import testing as npt
from pointcloud.utils import gen_utils
from pointcloud.data import conditioning

from pointcloud.models.load import get_model_class, Diffusion, load_diffusion_model
from pointcloud.models.shower_flow import compile_HybridTanH_model

from helpers import config_creator, example_paths


def test_get_cog():
    # with just one particle, the cog is always at the particle's position
    found = gen_utils.get_cog(np.zeros(1), np.zeros(1), np.zeros(1), np.ones(1))
    npt.assert_allclose(found, np.zeros(3))
    found = gen_utils.get_cog(np.ones(1), np.ones(1), np.ones(1), np.ones(1))
    npt.assert_allclose(found, np.ones(3))
    found = gen_utils.get_cog(np.ones(1), -np.ones(1), -np.ones(1), np.ones(1))
    npt.assert_allclose(found, [1, -1, -1])

    # with two particles, the cog is at the weighted center
    found = gen_utils.get_cog(np.zeros(2), np.zeros(2), np.zeros(2), np.ones(2))
    npt.assert_allclose(found, np.zeros(3))
    found = gen_utils.get_cog(np.arange(2), -np.arange(2), np.arange(2), np.ones(2))
    npt.assert_allclose(found, [0.5, -0.5, 0.5])
    found = gen_utils.get_cog(
        np.arange(2), -np.arange(2), np.arange(2), np.arange(1, 3)
    )
    npt.assert_allclose(found, [2 / 3, -2 / 3, 2 / 3])


# TODO test get_scale_factor, but I don't really know what is expected of it.


def test_cond_batcher():
    # don't choke on empty input
    for batch in gen_utils.cond_batcher(1, torch.ones(0)):
        pass
    # we no longer sort
    data = torch.tensor([[3, 7], [1, 2], [5, 8], [4, 6]])
    expected = np.array([[[3, 7], [1, 2]], [[5, 8], [4, 6]]])
    for i, batch in enumerate(gen_utils.cond_batcher(2, data)):
        npt.assert_allclose(batch["diffusion"], expected[i])


class TestGenMethods:
    config = []
    models = []
    model_names = {}

    @classmethod
    def setup_class(cls):
        """
        Make a list of models to test.
        """
        # make a caloclouds/showerflow model
        config = config_creator.make("caloclouds_3")
        cond_dim = conditioning.get_cond_dim(config, "showerflow")

        # fake the flow model
        flow, distribution, transforms = compile_HybridTanH_model(
            num_blocks=config.shower_flow_num_blocks,
            num_inputs=60,
            num_cond_inputs=cond_dim,
            device=config.device,
        )
        diff_model, coef_real, coef_fake, n_splines = load_diffusion_model(
            config, "cm", model_path=example_paths.example_cm_model
        )


        cls.config.append(config)
        cls.models.append((diff_model, distribution))
        cls.model_names["diffusion"] = 0

        # try with 2 cond features. Not in the list of models, because requires different input
        config = config_creator.make("caloclouds_2")
        config.cond_features_names = ["energy", "points"]

        cond_dim = conditioning.get_cond_dim(config, "showerflow")
        flow, distribution, transforms = compile_HybridTanH_model(
            num_blocks=config.shower_flow_num_blocks,
            num_inputs=65,
            num_cond_inputs=cond_dim,
            device=config.device,
        )
        diff_model = get_model_class(config)(config)

        cls.two_cond_flow = (diff_model, distribution)

    @classmethod
    def teardown_class(cls):
        cls.config.clear()
        cls.models.clear()

    def test_model_classes(self):
        assert isinstance(self.models[self.model_names["diffusion"]][0], Diffusion)
        assert isinstance(self.two_cond_flow[0], Diffusion)

    def test_get_shower(self):
        num_points = 100
        bs = 1
        energy = 50
        conditioning = [energy, 0, 0, 1]
        cond_N = 50
        model = self.models[self.model_names["diffusion"]][0]
        for bs in [0, 1, 2]:
            found = gen_utils.get_shower(model, num_points, conditioning, bs=bs)
            assert found.shape == (bs, num_points, 4)
            found = gen_utils.get_shower(
                self.two_cond_flow[0], num_points, energy, cond_N, bs=bs
            )
            assert found.shape == (bs, num_points, 4)
            # there is actually nothing at this stage to prevet the model
            # from producing unphysical values. It's a raw diffusion model.

    def test_gen_showers_batch(self):
        num = 6
        bs = 2
        e_min = 40
        e_max = 100
        # shoud work for all model/config pairs
        for config, models in zip(self.config, self.models):
            found, found_cond = gen_utils.gen_showers_batch(
                *models, e_min, e_max, config, num=num, bs=bs
            )
            assert found.shape == (num, config.max_points, 4)
            assert found_cond.shape == (num, 4)
            assert (found[:, :, 3] >= 0).all()

    def test_gen_cond_showers_batch(self):
        cond_E = (torch.arange(10).reshape(-1, 1) * 10).float()
        cond_dir = torch.tensor([[0, 0, 1]] * 10).float()
        cond = torch.cat([cond_E, cond_dir], dim=1)
        # cond as a double
        for config, models in zip(self.config, self.models):
            found = gen_utils.gen_cond_showers_batch(*models, cond, config=config)
            assert found.shape == (10, config.max_points, 4)
            assert np.all(found[:, :, 3] >= 0)

    def test_gen_v1_inner_batch(self):
        model, shower_flow = self.models[self.model_names["diffusion"]]
        config = self.config[self.model_names["diffusion"]]
        destination_array = np.zeros((10, config.max_points, 4))
        first_index = 2
        cond = {"diffusion": (torch.tensor([[10, 0, 0, 1], [20, 0, 0.5, 0.5]]).float())}
        gen_utils.gen_v1_inner_batch(
            cond, destination_array, first_index, model, shower_flow, config
        )
        assert np.all(destination_array[:2] == 0)
        assert np.all(destination_array[4:] == 0)
        assert np.all(destination_array[2:4, :, 3] >= 0)
