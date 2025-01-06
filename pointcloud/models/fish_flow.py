"""
I think I'm doing this wrong.
Fish should be just one big distribution....
"""
import torch
from . import shower_flow
from . import fish
from pyro.distributions.conditional import ConditionalDistribution
from torch.distributions import Distribution, Gumbel, Weibull, LogNormal, Normal
from ..utils.metadata import Metadata
from ..utils.maths import (
    gumbel,
    gumbel_params,
    weibull_params,
    logNorm_params,
    torch_polyval,
    gaussian,
)


def FishBasis(ConditionalDistribution):
    """
    Condition me to get a basis in the style of fish.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def condition(self, context):
        conditioned = SmoothFishBasis(self.model, context, self.device)
        return conditioned


def GranularFishBasis(Distribution):
    """
    Fish distribution at specific incident energy.

    Working from https://github.com/pytorch/pytorch/blob/main/torch/distributions/laplace.py
    and https://github.com/pytorch/pytorch/blob/main/torch/distributions/transformed_distribution.py
    We will predict;
    - Center of gravity in x, y, z
    - Number of clusters on each layer
    - Mean cluster energy on each layer

    """

    def __init__(self, model, incident_energy):
        self.model = model
        self.layer_depths = model.layer_depths
        self.incident_energy = incident_energy
        self.batch_shape = torch.Size([63])

        self.idx_cog_x = 0
        self._cog_x = Normal(0.0, 1.0)
        self.idx_cog_y = 1
        self._cog_y = Gumbel(0.0, 1.0)
        self.idx_cog_z = 2
        shift_gumbel_mu, shift_gumbel_beta = fish.calc_shifts(
            self.model.parametrisation, self.incident_energy
        )
        self._cog_z = Normal(shift_gumbel_mu, shift_gumbel_beta)

        # then we cannot get the remainders upfront,
        # as they depend on the shift
        self.idx_layer_clusters = slice(3, 33)
        self.idx_layer_energy = slice(33, 63)

    def _dist_n_hits(self, relative_locations):
        mean_n_hits, std_n_hits = fish.calc_dist_nHits(
            self.model.parametrisation, self.incident_energy, relative_locations
        )
        weibull_scale, weibull_concentration = weibull_params(mean_n_hits, std_n_hits)
        return Weibull(weibull_scale, weibull_concentration)

    def _dist_mean_energy(self, relative_locations):
        mean_energy, std_energy = fish.calc_dist_energy(
            self.model.parametrisation, self.incident_energy, relative_locations
        )
        logNorm_scale, logNorm_std = logNorm_params(mean_energy, std_energy)
        return LogNormal(logNorm_scale, logNorm_std)

    @property
    def mean(self):
        # TODO could be done, but not cheaply without transformed distributions
        raise NotImplementedError

    @property
    def variance(self):
        # TODO could be done, but not cheaply without transformed distributions
        raise NotImplementedError

    @property
    def stddev(self):
        # TODO could be done, but not cheaply without transformed distributions
        raise NotImplementedError

    def _sample(self, sample_shape=torch.Size(), sample_func="rsample"):
        cog_x = getattr(self._cog_x, sample_func)(sample_shape)
        cog_y = getattr(self._cog_y, sample_func)(sample_shape)
        cog_z = getattr(self._cog_z, sample_func)(sample_shape)
        # the number of clusters is a bit more complicated
        # take the cog_y we sampled as the shift
        relative_locations = self.layer_depths[None, :] - cog_y[..., None]
        n_hits_dist = self._dist_n_hits(relative_locations)
        n_hits = getattr(n_hits_dist, sample_func)(sample_shape)
        # the mean energy is about the same
        energy_dist = self._dist_mean_energy(relative_locations)
        energy = getattr(energy_dist, sample_func)(sample_shape)
        concatenated = torch.cat([cog_x, cog_y, cog_z, n_hits, energy], dim=-1)
        return concatenated

    def rsample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, "rsample")

    def sample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, "sample")

    def _value_func(self, value, func_name):
        # center of gravity is simple
        cog_x = getattr(self._cog_x, func_name)(value[..., self.idx_cog_x])
        cog_y = getattr(self._cog_y, func_name)(value[..., self.idx_cog_y])
        cog_z = getattr(self._cog_z, func_name)(value[..., self.idx_cog_z])
        # the number of clusters is a bit more complicated
        # take the cog_y we are given, not the cog_y the of the points
        relative_locations = (
            self.layer_depths[None, :] - value[..., self.idx_cog_y, None]
        )
        n_hits_dist = self._dist_n_hits(relative_locations)
        n_hits = getattr(n_hits_dist, func_name)(value[..., self.idx_layer_clusters])
        # the mean energy is about the same
        energy_dist = self._dist_mean_energy(relative_locations)
        energy = getattr(energy_dist, func_name)(value[..., self.idx_layer_energy])
        concatenated = torch.cat([cog_x, cog_y, cog_z, n_hits, energy], dim=-1)
        return concatenated

    def pdf(self, value):
        return self._value_func(value, "pdf")

    def log_prob(self, value):
        return self._value_func(value, "log_prob")

    def cdf(self, value):
        return self._value_func(value, "cdf")

    def icdf(self, value):
        return self._value_func(value, "icdf")

    def entropy(self):
        raise NotImplementedError


def SmoothFishBases(Distibution):
    def __init__(self, model, layer_structure):
        self.layer_structure = layer_structure
        self.model = model
        self.stats = {}

    def parameters(self):
        return list(self.stats.values())

    def set_conditioning(self, conditioning):
        incident_energy = conditioning[..., -1].flatten()
        self.stats = self.model.calc_stats(incident_energy, False, False)

    def sample(self):
        pass


def get_fish_basis(num_inputs, device, fish_model, **kwargs):
    """
    There are 63 dimensions.
    In order of appearance, the dimensions encode;
    - center of gravity (energy) in x
    - center of gravity (energy) in y
    - center of gravity (energy) in z
    - number of clusters on layer 1
    ...
    - number of clusters on layer 30
    - total energy on layer 1
    ...
    - total energy on layer 30
    """
    if num_inputs != 63:
        raise ValueError("The fish basis must have 63 inputs.")
    if isinstance(fish_model, str):
        fish_model = fish.Fish.load(fish_model)
    # get a conditional distribution for the number of clusters
    return FishBasis(fish_model, device)


def compile_fish_v1(num_blocks, num_cond_inputs, device):
    num_inputs = 63
    factory = shower_flow.HybridTanH_factory(num_inputs, num_cond_inputs, device)

    transform_pattern = [
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "spline_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
    ]

    model, flow_dist = factory.create(
        num_blocks, transform_pattern, base_dist_gen=get_fish_basis, count_bins=8
    )
    return model, flow_dist


class Adaptor(Distribution):
    """
    Adapt the 63 element output of the fish basis to
    a 65 element output comparable with the showerflow model.
    Including renormalising.
    """

    def __init__(self, configs, base_dist=None):
        self.configs = configs
        self.metadata = Metadata(configs)
        self.base_dist = base_dist
        if base_dist is None:
            # treat it as data
            self._std_cog_y = self.metadata.std_cog[1]
            self._mean_cog_y = self.metadata.mean_cog[1]
        else:
            self._std_cog_y = base_dist._cog_y.stddev
            self._mean_cog_y = base_dist._cog_y.mean

    def from_basis(self, value):
        """
        Create the 65 element output from the 63 element input.
        """
        outputs = []
        # the total clusters and the total eneryg ned creating from scratch
        outputs.append(value[..., 3:33].sum(dim=-1)[..., None])
        outputs.append(value[..., 33:63].sum(dim=-1)[..., None])
        # COG in the output has mean 0 and std 1
        # so COG x and z actually have the same normalisation
        outputs.append(value[..., 0][..., None])
        # COG y needs rescaling
        outputs.append(
            ((value[..., 1] - self._mean_cog_y) / self._std_cog_y)[..., None]
        )
        outputs.append(value[..., 2][..., None])
        # then the other outputs need rescaling by their max
        max_n_hits = value[..., 3:33].max(dim=-1)
        outputs.append(value[..., 3:33] / max_n_hits)
        max_energy = value[..., 33:63].max(dim=-1)
        outputs.append(value[..., 33:63] / max_energy)
        # finally we concatenate along the last dimension
        return torch.cat(outputs, dim=-1)

    def to_basis(self, value):
        """
        Create the 63 element input from the 65 element output.
        """
        inputs = []
        # COG x and z can be taken as is
        # COG y needs rescaling
        inputs.append(value[..., 2][..., None])
        inputs.append((value[..., 3] * self._std_cog_y + self._mean_cog_y)[..., None])
        inputs.append(value[..., 4][..., None])
        # the other outputs need rescaling so that they sum to the first and second elements
        current_sum = value[..., 5:35].sum(dim=-1)
        required_rescale = value[..., 0] / current_sum
        inputs.append(value[..., 5:35] * required_rescale[..., None])
        current_sum = value[..., 35:65].sum(dim=-1)
        required_rescale = value[..., 1] / current_sum
        inputs.append(value[..., 35:65] * required_rescale[..., None])
        # finally we concatenate along the last dimension
        return torch.cat(inputs, dim=-1)

    def _sample(self, sample_shape=torch.Size(), sample_func="rsample"):
        value = self.base_dist._sample(sample_shape, sample_func)
        return self.from_basis(value)

    def rsample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, "rsample")

    def sample(self, sample_shape=torch.Size()):
        return self._sample(sample_shape, "sample")

    def _value_func(self, value, func_name):
        basis_value = self.to_basis(value)
        return self.base_dist._value_func(basis_value, func_name)

    def pdf(self, value):
        return self._value_func(value, "pdf")

    def log_prob(self, value):
        return self._value_func(value, "log_prob")

    def cdf(self, value):
        return self._value_func(value, "cdf")

    def icdf(self, value):
        return self._value_func(value, "icdf")
