import math
from numbers import Number

import torch
from torch.distributions import constraints, Categorical
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

__all__ = ["PartGFlashRadial", "GFlashRadial", "RadialPlane"]


class PartGFlashRadial(Distribution):
    """
    Sample from one term of the GFlash radial distribution.
    p(x) = 2 * x * R^2 / (x^2 + R^2)^2
    """

    arg_constraints = {"R": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, R, validate_args=None):
        self.R = R
        self.R2 = R**2
        if isinstance(R, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.R.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.R * torch.pi / 2

    @property
    def variance(self):
        raise NotImplementedError("Does not converge")

    def rsample(self, sample_shape=torch.Size()):
        # need to avoid inputting 1 to the icdf as it has a singularity there
        finfo = torch.finfo(self.R.dtype)
        if torch._C._get_tracing_state():
            # [JIT WORKAROUND] lack of support for .uniform_()
            u = torch.rand(sample_shape, dtype=self.R.dtype, device=self.R.device)
            produced = self.icdf(u.clamp(max=1 - finfo.tiny))
        # this doesn't work if torch.no_grad() is used
        # gives the wrong shape....
        # u = self.R.new(sample_shape).uniform_(0, 1 - finfo.eps)
        u = torch.FloatTensor(*sample_shape).uniform_(0, 1 - finfo.eps)
        produced = self.icdf(u)
        return produced

    def prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 2 * value * self.R2 / ((value**2 + self.R2) ** 2)

    def log_prob(self, value):
        return torch.log(self.prob(value))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return value**2 / (value**2 + self.R2)

    def icdf(self, value):
        numerator = self.R * torch.sqrt(value)
        denominator = torch.sqrt(1 - value)
        return numerator / denominator

    def entropy(self):
        raise NotImplementedError


class GFlashRadial(Distribution):
    """
    Sample from the GFlash radial distribution.
    p(x) =       p * 2 * x * Rc^2 / (x^2 + Rc^2)^2 +
           (1 - p) * 2 * x * Rt^2 / (x^2 + Rt^2)^2
    """

    arg_constraints = {
        "Rc": constraints.positive,
        "Rt_extend": constraints.positive,
        "p": constraints.unit_interval,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, Rc, Rt_extend, p, validate_args=None):
        self.Rc, self.Rt_extend, self.p = broadcast_all(Rc, Rt_extend, p)
        self.Rt = self.Rc + self.Rt_extend
        self.part_c = PartGFlashRadial(self.Rc)
        self.part_t = PartGFlashRadial(self.Rt)
        self.choice = Categorical(probs=torch.tensor([self.p, 1 - self.p]))
        if isinstance(self.Rc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.Rc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.p * self.part_c.mean + (1 - self.p) * self.part_t.mean

    @property
    def mode(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError("Does not converge")

    def rsample(self, sample_shape=torch.Size()):
        # if we are asked for no points, treat as special case
        if sample_shape[0]:
            choice = self.choice.sample(sample_shape)
        else:
            choice = torch.Tensor([], device=self.Rc.device)
        c_sample = self.part_c.sample(sample_shape)
        t_sample = self.part_t.sample(sample_shape)
        out = torch.where(
            choice == 0,
            c_sample,
            t_sample,
        )
        return out

    def prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.p * self.part_c.prob(value) + (1 - self.p) * self.part_t.prob(value)

    def log_prob(self, value):
        return torch.log(self.prob(value))

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


class RadialPlane(Distribution):
    """
    Probability on a plane given a radial distribution.
    """

    has_rsample = True
    _event_shape = torch.Size([2])
    _batch_shape = torch.Size()

    def __init__(self, radial, validate_args=None):
        """
        Constructor

        Parameters
        ----------
        radial : torch.distributions.Distribution
            The radial distribution, should be normalised to 1.
        """
        self.radial = radial
        self.normalisation = 1 / (2 * math.pi)

    @property
    def mean(self):
        # by symmetry, the mean is at the origin
        return torch.zeros(2)

    @property
    def mode(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size()):
        # first draw the angular part
        phi = 2 * math.pi * torch.rand(*sample_shape)
        # then draw the radial part
        r = self.radial.rsample(sample_shape)
        # convert to cartesian
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        out = torch.stack([x, y], dim=-1)
        return out

    def prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # the radial part is independent of the angular part
        # to get the probability density,
        # we need to multiply the radial part by the jacobian
        # of the transformation from polar to cartesian coordinates
        scalar_radius = value.norm(dim=-1)
        radial_prob = self.radial.prob(scalar_radius)
        jacoben = scalar_radius ** (-1)
        jacoben[scalar_radius <= 0] = 1
        return self.normalisation * radial_prob * jacoben

    def log_prob(self, value):
        return torch.log(self.prob(value))

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


def to_torch1d(x):
    """
    Convert x to a torch tensor with at least 1 dimension.

    Parameters
    ----------
    x : array_like or Number
        The input to convert.

    Returns
    -------
    torch.Tensor
        The converted tensor.

    """
    try:
        return torch.atleast_1d(x)
    except TypeError:
        x = torch.tensor(x)
        return torch.atleast_1d(x)


def make_tailed_exponential(core, to_tail, p_core):
    """
    Make a distribution whos pdf has the format;

    p(x) = p * (exp(-x/core)/core) + (1-p) * (exp(-x/tail)/tail)
        where
    tail = core + to_tail

    Parameters
    ----------
    core : array_like or float
        The core of the exponential distribution.
    to_tail : array_like or float
        The distance from the core to the tail of
        the exponential distribution.
    p_core : array_like or float
        The probability that a value is drawn from the
        core, rather than the tail.

    Returns
    -------
    torch.distributions.Distribution
        The distribution.

    """
    core = to_torch1d(core)
    to_tail = to_torch1d(to_tail)
    p_core = to_torch1d(p_core)
    rates = torch.vstack([1 / core, 1 / (core + to_tail)]).T
    probs = torch.vstack([p_core, 1 - p_core]).T
    exponential = torch.distributions.Exponential(rates)
    mix = torch.distributions.Categorical(probs)
    combined = torch.distributions.MixtureSameFamily(mix, exponential)
    return combined
