import math
from typing import Any, Optional, Union

import numpy as np
from scipy import stats
from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from HARK.distributions.base import Distribution
from HARK.distributions.discrete import DiscreteDistribution

# CONTINUOUS DISTRIBUTIONS


class ContinuousFrozenDistribution(rv_continuous_frozen, Distribution):
    """
    Parameterized continuous distribution from scipy.stats with seed management.
    """

    def __init__(
        self, dist: rv_continuous, *args: Any, seed: int = 0, **kwds: Any
    ) -> None:
        """
        Parameterized continuous distribution from scipy.stats with seed management.

        Parameters
        ----------
        dist : rv_continuous
            Continuous distribution from scipy.stats.
        seed : int, optional
            Seed for random number generator, by default 0
        """
        rv_continuous_frozen.__init__(self, dist, *args, **kwds)
        Distribution.__init__(self, seed=seed)


class Normal(ContinuousFrozenDistribution):
    """
    A Normal distribution.

    Parameters
    ----------
    mu : float or [float]
        One or more means.  Number of elements T in mu determines number
        of rows of output.
    sigma : float or [float]
        One or more standard deviations. Number of elements T in sigma
        determines number of rows of output.
    seed : int
        Seed for random number generator.
    """

    def __init__(self, mu=0.0, sigma=1.0, seed=0):
        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)

        if self.mu.size != self.sigma.size:
            raise AttributeError(
                "mu and sigma must be of same size, are %s, %s"
                % ((self.mu.size), (self.sigma.size))
            )

        super().__init__(stats.norm, loc=mu, scale=sigma, seed=seed)

    def discretize(self, N, method="hermite", endpoints=False):
        """
        For normal distributions, the Gauss-Hermite quadrature rule is
        used as the default method for discretization.
        """

        return super().discretize(N, method, endpoints)

    def _approx_hermite(self, N, endpoints=False):
        """
        Returns a discrete approximation of this distribution
        using the Gauss-Hermite quadrature rule.

        TODO: add endpoints option

        Parameters
        ----------
        N : int
            Number of discrete points to approximate the distribution.

        Returns
        -------
        DiscreteDistribution
            Discrete approximation of this distribution.
        """

        x, w = np.polynomial.hermite.hermgauss(N)
        # normalize w
        pmv = w * np.pi**-0.5
        # correct x
        atoms = math.sqrt(2.0) * self.sigma * x + self.mu

        limit = {"dist": self, "method": "hermite", "N": N, "endpoints": endpoints}

        return DiscreteDistribution(
            pmv,
            atoms,
            seed=self._rng.integers(0, 2**31 - 1, dtype="int32"),
            limit=limit,
        )

    def _approx_equiprobable(self, N, endpoints=False):
        """
        Returns a discrete equiprobable approximation of this distribution.

        TODO: add endpoints option

        Parameters
        ----------
        N : int
            Number of discrete points to approximate the distribution.

        Returns
        -------
        DiscreteDistribution
            Discrete approximation of this distribution.
        """

        CDF = np.linspace(0, 1, N + 1)
        lims = stats.norm.ppf(CDF)
        pdf = stats.norm.pdf(lims)

        # Find conditional means using Mills's ratio
        pmv = np.diff(CDF)
        atoms = self.mu - np.diff(pdf) / pmv * self.sigma

        limit = {"dist": self, "method": "equiprobable", "N": N, "endpoints": endpoints}

        return DiscreteDistribution(
            pmv,
            atoms,
            seed=self._rng.integers(0, 2**31 - 1, dtype="int32"),
            limit=limit,
        )


class Lognormal(ContinuousFrozenDistribution):
    """
    A Lognormal distribution

    Parameters
    ----------
    mu : float or [float]
        One or more means of underlying normal distribution.
        Number of elements T in mu determines number of rows of output.
    sigma : float or [float]
        One or more standard deviations of underlying normal distribution.
        Number of elements T in sigma determines number of rows of output.
    seed : int
        Seed for random number generator.
    """

    def __new__(
        cls,
        mu: Union[float, np.ndarray] = 0.0,
        sigma: Union[float, np.ndarray] = 1.0,
        seed: Optional[int] = 0,
        mean=None,
        std=None,
    ):
        """
        Create a new Lognormal distribution. If sigma is zero, return a
        DiscreteDistribution with a single atom at exp(mu).

        Parameters
        ----------
        mu : Union[float, np.ndarray], optional
            Mean of underlying normal distribution, by default 0.0
        sigma : Union[float, np.ndarray], optional
            Standard deviation of underlying normal distribution, by default 1.0
        seed : Optional[int], optional
            Seed for random number generator, by default None
        mean: optional
            For alternative mean/std parameterization, the mean of the lognormal distribution
        std: optional
            For alternative mean/std parameterization, the standard deviation of the lognormal distribution

        Returns
        -------
        Lognormal or DiscreteDistribution
            Lognormal distribution or DiscreteDistribution with a single atom.
        """

        if sigma == 0:
            # If sigma is zero, return a DiscreteDistribution with a single atom
            if mean is not None:
                return DiscreteDistribution([1.0], mean, seed=seed)
            return DiscreteDistribution([1.0], [np.exp(mu)], seed=seed)

        return super().__new__(cls)

    def __init__(
        self,
        mu: Union[float, np.ndarray] = 0.0,
        sigma: Union[float, np.ndarray] = 1.0,
        seed: Optional[int] = 0,
        mean=None,
        std=None,
    ):
        if mean is not None and std is not None:
            mean_squared = mean**2
            variance = std**2
            mu = np.log(mean / (np.sqrt(1.0 + variance / mean_squared)))
            sigma = np.sqrt(np.log(1.0 + variance / mean_squared))

        if mean is not None and std is None:
            mu = np.log(mean) - sigma**2 / 2

        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)

        if self.mu.size != self.sigma.size:
            raise AttributeError(
                "mu and sigma must be of same size, are %s, %s"
                % ((self.mu.size), (self.sigma.size))
            )

        # Set up the RNG
        super().__init__(
            stats.lognorm, s=self.sigma, scale=np.exp(self.mu), loc=0, seed=seed
        )

    def _approx_equiprobable(
        self, N, endpoints=False, tail_N=0, tail_bound=None, tail_order=np.e
    ):
        """
        Construct a discrete approximation to a lognormal distribution with underlying
        normal distribution N(mu,sigma).  Makes an equiprobable distribution by
        default, but user can optionally request augmented tails with exponentially
        sized point masses.  This can improve solution accuracy in some models.

        TODO: add endpoints option

        Parameters
        ----------
        N: int
            Number of discrete points in the "main part" of the approximation.
        tail_N: int
            Number of points in each "tail part" of the approximation; 0 = no tail.
        tail_bound: [float]
            CDF boundaries of the tails vs main portion; tail_bound[0] is the lower
            tail bound, tail_bound[1] is the upper tail bound.  Inoperative when
            tail_N = 0.  Can make "one tailed" approximations with 0.0 or 1.0.
        tail_order: float
            Factor by which consecutive point masses in a "tail part" differ in
            probability.  Should be >= 1 for sensible spacing.

        Returns
        -------
        d : DiscreteDistribution
            Probability associated with each point in array of discrete
            points for discrete probability mass function.
        """

        tail_bound = tail_bound if tail_bound is not None else [0.02, 0.98]
        # Find the CDF boundaries of each segment
        if self.sigma > 0.0:
            if tail_N > 0:
                lo_cut = tail_bound[0]
                hi_cut = tail_bound[1]
            else:
                lo_cut = 0.0
                hi_cut = 1.0
            inner_size = hi_cut - lo_cut
            inner_CDF_vals = [
                lo_cut + x * N ** (-1.0) * inner_size for x in range(1, N)
            ]
            if inner_size < 1.0:
                scale = 1.0 / tail_order
                mag = (1.0 - scale**tail_N) / (1.0 - scale)
            lower_CDF_vals = [0.0]
            if lo_cut > 0.0:
                for x in range(tail_N - 1, -1, -1):
                    lower_CDF_vals.append(lower_CDF_vals[-1] + lo_cut * scale**x / mag)
            upper_CDF_vals = [hi_cut]
            if hi_cut < 1.0:
                for x in range(tail_N):
                    upper_CDF_vals.append(
                        upper_CDF_vals[-1] + (1.0 - hi_cut) * scale**x / mag
                    )
            CDF_vals = lower_CDF_vals + inner_CDF_vals + upper_CDF_vals
            temp_cutoffs = list(
                stats.lognorm.ppf(
                    CDF_vals[1:-1], s=self.sigma, loc=0, scale=np.exp(self.mu)
                )
            )
            cutoffs = [0] + temp_cutoffs + [np.inf]
            CDF_vals = np.array(CDF_vals)

            K = CDF_vals.size - 1  # number of points in approximation
            pmv = CDF_vals[1 : (K + 1)] - CDF_vals[0:K]
            atoms = np.zeros(K)
            for i in range(K):
                zBot = cutoffs[i]
                zTop = cutoffs[i + 1]
                # Manual check to avoid the RuntimeWarning generated by "divide by zero"
                # with np.log(zBot).
                if zBot == 0:
                    tempBot = np.inf
                else:
                    tempBot = (self.mu + self.sigma**2 - np.log(zBot)) / (
                        np.sqrt(2) * self.sigma
                    )
                tempTop = (self.mu + self.sigma**2 - np.log(zTop)) / (
                    np.sqrt(2) * self.sigma
                )
                if tempBot <= 4:
                    atoms[i] = (
                        -0.5
                        * np.exp(self.mu + (self.sigma**2) * 0.5)
                        * (math.erf(tempTop) - math.erf(tempBot))
                        / pmv[i]
                    )
                else:
                    atoms[i] = (
                        -0.5
                        * np.exp(self.mu + (self.sigma**2) * 0.5)
                        * (math.erfc(tempBot) - math.erfc(tempTop))
                        / pmv[i]
                    )

        else:
            pmv = np.ones(N) / N
            atoms = np.exp(self.mu) * np.ones(N)

        limit = {
            "dist": self,
            "method": "equiprobable",
            "N": N,
            "endpoints": endpoints,
            "tail_N": tail_N,
            "tail_bound": tail_bound,
            "tail_order": tail_order,
        }

        return DiscreteDistribution(
            pmv,
            atoms,
            seed=self._rng.integers(0, 2**31 - 1, dtype="int32"),
            limit=limit,
        )

    def _approx_hermite(self, N, endpoints=False):
        """
        Returns a discrete approximation of this distribution
        using the Gauss-Hermite quadrature rule.

        TODO: add endpoints option

        Parameters
        ----------
        N : int
            Number of discrete points to approximate the distribution.

        Returns
        -------
        DiscreteDistribution
            Discrete approximation of this distribution.
        """

        x, w = np.polynomial.hermite.hermgauss(N)
        # normalize w
        pmv = w * np.pi**-0.5
        # correct x
        atoms = np.exp(np.sqrt(2.0) * self.sigma * x + self.mu)

        limit = {"dist": self, "method": "hermite", "N": N, "endpoints": endpoints}

        return DiscreteDistribution(
            pmv,
            atoms,
            seed=self._rng.integers(0, 2**31 - 1, dtype="int32"),
            limit=limit,
        )

    @classmethod
    def from_mean_std(cls, mean, std, seed=0):
        """
        Construct a LogNormal distribution from its
        mean and standard deviation.

        This is unlike the normal constructor for the distribution,
        which takes the mu and sigma for the normal distribution
        that is the logarithm of the Log Normal distribution.

        Parameters
        ----------
        mean : float or [float]
            One or more means.  Number of elements T in mu determines number
            of rows of output.
        std : float or [float]
            One or more standard deviations. Number of elements T in sigma
            determines number of rows of output.
        seed : int
            Seed for random number generator.

        Returns
        ---------
        LogNormal

        """
        mean_squared = mean**2
        variance = std**2
        mu = np.log(mean / (np.sqrt(1.0 + variance / mean_squared)))
        sigma = np.sqrt(np.log(1.0 + variance / mean_squared))

        return cls(mu=mu, sigma=sigma, seed=seed)


class MeanOneLogNormal(Lognormal):
    """
    A Lognormal distribution with mean 1.
    """

    def __init__(self, sigma=1.0, seed=0):
        mu = -0.5 * sigma**2
        super().__init__(mu=mu, sigma=sigma, seed=seed)


class Uniform(ContinuousFrozenDistribution):
    """
    A Uniform distribution.

    Parameters
    ----------
    bot : float or [float]
        One or more bottom values.
        Number of elements T in mu determines number
        of rows of output.
    top : float or [float]
        One or more top values.
        Number of elements T in top determines number of
        rows of output.
    seed : int
        Seed for random number generator.
    """

    def __init__(self, bot=0.0, top=1.0, seed=0):
        self.bot = np.asarray(bot)
        self.top = np.asarray(top)

        super().__init__(
            stats.uniform, loc=self.bot, scale=self.top - self.bot, seed=seed
        )

    def _approx_equiprobable(self, N, endpoints=False):
        """
        Makes a discrete approximation to this uniform distribution.

        Parameters
        ----------
        N : int
            The number of points in the discrete approximation.
        endpoints : bool
            Whether to include the endpoints in the approximation.

        Returns
        -------
        d : DiscreteDistribution
            Probability associated with each point in array of discrete
            points for discrete probability mass function.
        """
        pmv = np.ones(N) / float(N)

        center = (self.top + self.bot) / 2.0
        width = (self.top - self.bot) / 2.0
        atoms = center + width * np.linspace(-(N - 1.0) / 2.0, (N - 1.0) / 2.0, N) / (
            N / 2.0
        )

        if endpoints:  # insert endpoints with infinitesimally small mass
            atoms = np.concatenate(([self.bot], atoms, [self.top]))
            pmv = np.concatenate(([0.0], pmv, [0.0]))

        limit = {"dist": self, "method": "equiprobable", "N": N, "endpoints": endpoints}

        return DiscreteDistribution(
            pmv,
            atoms,
            seed=self._rng.integers(0, 2**31 - 1, dtype="int32"),
            limit=limit,
        )


class Weibull(ContinuousFrozenDistribution):
    """
    A Weibull distribution.

    Parameters
    ----------
    scale : float or [float]
        One or more scales.  Number of elements T in scale
        determines number of
        rows of output.
    shape : float or [float]
        One or more shape parameters. Number of elements T in scale
        determines number of rows of output.
    seed : int
        Seed for random number generator.
    """

    def __init__(self, scale=1.0, shape=1.0, seed=0):
        self.scale = np.asarray(scale)
        self.shape = np.asarray(shape)
        # Set up the RNG
        super().__init__(stats.weibull_min, c=shape, scale=scale, seed=seed)
