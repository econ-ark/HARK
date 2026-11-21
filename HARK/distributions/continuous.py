import functools
import warnings
from math import factorial
from typing import Any, Optional, Union

import numpy as np
from scipy.special import erfc, ndtr, ndtri
from scipy.stats import (
    rv_continuous,
    norm,
    lognorm,
    uniform,
    weibull_min,
)
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from HARK.distributions.base import Distribution
from HARK.distributions.discrete import DiscreteDistribution


def _hermite_discretize(distribution, N, endpoints, atoms_transform, lower_endpoint):
    """Gauss-Hermite quadrature discretization shared by Normal/LogNormal.

    ``atoms_transform`` maps the linear ``sqrt(2)*sigma*x + mu`` grid to the
    distribution's support (identity for Normal, ``np.exp`` for LogNormal).
    ``lower_endpoint`` is the value prepended when ``endpoints=True``.
    """
    x, w = np.polynomial.hermite.hermgauss(N)
    pmv = w * np.pi**-0.5
    atoms = atoms_transform(np.sqrt(2.0) * distribution.sigma * x + distribution.mu)
    if endpoints:
        atoms = np.r_[lower_endpoint, atoms, np.inf]
        pmv = np.r_[0.0, pmv, 0.0]
    return DiscreteDistribution(
        pmv,
        atoms,
        seed=distribution.random_seed(),
        limit={
            "dist": distribution,
            "method": "hermite",
            "N": N,
            "endpoints": endpoints,
        },
    )


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

    def __new__(
        cls,
        mu: Union[float, np.ndarray] = 0.0,
        sigma: Union[float, np.ndarray] = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Create a new Normal distribution. If sigma is zero, return a
        DiscreteDistribution with a single atom at mu.
        Parameters
        ----------
        mu : Union[float, np.ndarray], optional
            Mean of normal distribution, by default 0.0
        sigma : Union[float, np.ndarray], optional
            Standard deviation of normal distribution, by default 1.0
        seed : Optional[int], optional
            Seed for random number generator, by default None
        Returns
        -------
        Normal or DiscreteDistribution
            Normal distribution or DiscreteDistribution with a single atom.
        """

        if sigma == 0:
            # If sigma is zero, return a DiscreteDistribution with a single atom
            return DiscreteDistribution([1.0], mu, seed=seed)

        return super().__new__(cls)

    def __init__(self, mu=0.0, sigma=1.0, seed=None):
        self.mu = np.asarray(mu)
        self.sigma = np.asarray(sigma)

        if self.mu.size != self.sigma.size:
            raise AttributeError(
                f"'mu' and 'sigma' must be of the same size. Instead 'mu' is of "
                f"size {self.mu.size} and 'sigma' is of size {self.sigma.size}."
            )

        super().__init__(norm, loc=mu, scale=sigma, seed=seed)
        self.infimum = -np.inf * np.ones(self.mu.size)
        self.supremum = np.inf * np.ones(self.mu.size)

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

        Parameters
        ----------
        N : int
            Number of discrete points to approximate the distribution.

        Returns
        -------
        DiscreteDistribution
            Discrete approximation of this distribution.
        """
        return _hermite_discretize(self, N, endpoints, lambda v: v, -np.inf)

    def _approx_equiprobable(self, N, endpoints=False):
        """
        Returns a discrete equiprobable approximation of this distribution.

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
        lims = norm.ppf(CDF)
        pdf = norm.pdf(lims)

        # Find conditional means using Mills's ratio
        pmv = np.diff(CDF)
        atoms = self.mu - np.diff(pdf) / pmv * self.sigma

        if endpoints:
            atoms = np.r_[-np.inf, atoms, np.inf]
            pmv = np.r_[0.0, pmv, 0.0]

        limit = {"dist": self, "method": "equiprobable", "N": N, "endpoints": endpoints}

        return DiscreteDistribution(
            pmv,
            atoms,
            seed=self._rng.integers(0, 2**31 - 1, dtype="int32"),
            limit=limit,
        )


class LogNormal(ContinuousFrozenDistribution):
    """
    A LogNormal distribution

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
        seed: Optional[int] = None,
        mean=None,
        std=None,
    ):
        """
        Create a new LogNormal distribution. If sigma is zero, return a
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
        LogNormal or DiscreteDistribution
            LogNormal distribution or DiscreteDistribution with a single atom.
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
        seed: Optional[int] = None,
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
        super().__init__(lognorm, s=self.sigma, scale=np.exp(self.mu), loc=0, seed=seed)
        if np.all(self.sigma == 0.0):
            # Degenerate at exp(mu), as in retirement's permanent shock
            self.infimum = np.atleast_1d(np.exp(self.mu)).astype(float)
            self.supremum = self.infimum.copy()
        else:
            self.infimum = np.array([0.0])
            self.supremum = np.array([np.inf])

    def _approx_equiprobable(
        self, N, endpoints=False, tail_N=0, tail_bound=None, tail_order=np.e
    ):
        """
        Construct a discrete approximation to a lognormal distribution with underlying
        normal distribution N(mu,sigma).  Makes an equiprobable distribution by
        default, but user can optionally request augmented tails with exponentially
        sized point masses.  This can improve solution accuracy in some models.

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
        tail_bound = tail_bound or [0.02, 0.98]

        # Handle the trivial case first
        if self.sigma == 0.0:
            pmv = np.ones(N) / N
            atoms = np.exp(self.mu) * np.ones(N)

        else:
            # Find the CDF boundaries of each segment
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
            CDF_vals = np.array(lower_CDF_vals + inner_CDF_vals + upper_CDF_vals)
            CDF_vals[-1] = 1.0
            CDF_vals[0] = 0.0  # somehow these need fixing sometimes

            # Calculate probability masses for each node
            pmv = CDF_vals[1:] - CDF_vals[:-1]
            pmv /= np.sum(pmv)

            # Translate the CDF values to z-scores (stdevs from mean), then get q-scores
            z_cuts = norm.ppf(CDF_vals)
            q_cuts = (z_cuts - self.sigma) / np.sqrt(2)

            # Evaluate the (complementary) error function at the q values
            erf_q = erfc(q_cuts)
            erf_q_neg = erfc(-q_cuts)

            # Evaluate the base for the conditional expectations
            vals_base = erf_q[:-1] - erf_q[1:]
            these = q_cuts[:-1] < -2.0  # flag low q values and use the *other* version
            vals_base[these] = erf_q_neg[1:][these] - erf_q_neg[:-1][these]

            # Make and apply the normalization factor and probability weights
            norm_fac = 0.5 * np.exp(self.mu + 0.5 * self.sigma**2) / pmv
            atoms = vals_base * norm_fac

        if endpoints:
            atoms = np.r_[0.0, atoms, np.inf]
            pmv = np.r_[0.0, pmv, 0.0]

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
            seed=self.random_seed(),
            limit=limit,
        )

    def _approx_hermite(self, N, endpoints=False):
        """
        Returns a discrete approximation of this distribution
        using the Gauss-Hermite quadrature rule.

        Parameters
        ----------
        N : int
            Number of discrete points to approximate the distribution.

        Returns
        -------
        DiscreteDistribution
            Discrete approximation of this distribution.
        """

        return _hermite_discretize(self, N, endpoints, np.exp, 0.0)

    @classmethod
    def from_mean_std(cls, mean, std, seed=None):
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


class LogNormalMeanOne(LogNormal):
    """
    A LogNormal distribution with mean 1.
    """

    def __init__(self, sigma=1.0, seed=0):
        mu = -0.5 * sigma**2
        super().__init__(mu=mu, sigma=sigma, seed=seed)


def _normal_mass(lo, hi):
    """
    P(lo < Z < hi) for a standard normal Z, taken on the side of zero that
    keeps its digits in the tails.
    """
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return np.where(lo > 0.0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))


def _truncated_normal_ppf(q, lo, hi):
    """
    Quantile q of a standard normal truncated to [lo, hi].
    """
    mass = _normal_mass(lo, hi)
    with np.errstate(invalid="ignore"):
        upper = -ndtri(ndtr(-lo) - q * mass)
        lower = ndtri(ndtr(lo) + q * mass)
    return np.where(np.asarray(lo) > 0.0, upper, lower)


def _bound(bound, default):
    """A truncation bound in standard deviations; None is no bound on that side."""
    return default if bound is None else float(bound)


def truncated_normal_variance(bot=-3.0, top=3.0):
    """
    The variance of a standard normal truncated to [bot, top], in standard
    deviations; None is no bound on that side. A lognormal truncated in the log
    at these bounds has Var(ln X) = sigma**2 times this factor.

    Parameters
    ----------
    bot : float or None, optional
        Lower bound, by default -3.0.
    top : float or None, optional
        Upper bound, by default 3.0.

    Returns
    -------
    float
    """
    lo, hi = _bound(bot, -np.inf), _bound(top, np.inf)
    mass = _normal_mass(lo, hi)
    pdf_lo, pdf_hi = norm.pdf(lo), norm.pdf(hi)
    z_pdf_lo = lo * pdf_lo if np.isfinite(lo) else 0.0
    z_pdf_hi = hi * pdf_hi if np.isfinite(hi) else 0.0
    return float(1.0 + (z_pdf_lo - z_pdf_hi) / mass - ((pdf_lo - pdf_hi) / mass) ** 2)


class _truncated_lognorm_gen(rv_continuous):
    """
    The lognormal exp(s Z), with Z a standard normal truncated to [lo, hi] in its
    standard deviations. scipy's scale is exp(mu).
    """

    def _argcheck(self, s, lo, hi):
        return (s > 0) & (lo < hi)

    def _get_support(self, s, lo, hi):
        return np.exp(s * lo), np.exp(s * hi)

    def _pdf(self, x, s, lo, hi):
        with np.errstate(divide="ignore"):
            z = np.log(x) / s
        inside = (z >= lo) & (z <= hi)
        dens = norm.pdf(z) / (s * x * _normal_mass(lo, hi))
        return np.where(inside, dens, 0.0)

    def _cdf(self, x, s, lo, hi):
        with np.errstate(divide="ignore"):
            z = np.clip(np.log(x) / s, lo, hi)
        return _normal_mass(lo, z) / _normal_mass(lo, hi)

    def _ppf(self, q, s, lo, hi):
        return np.exp(s * _truncated_normal_ppf(q, lo, hi))

    def _munp(self, n, s, lo, hi):
        return (
            np.exp(0.5 * (n * s) ** 2)
            * _normal_mass(lo - n * s, hi - n * s)
            / _normal_mass(lo, hi)
        )


_truncated_lognorm = _truncated_lognorm_gen(
    a=0.0, name="truncated_lognorm", shapes="s, lo, hi"
)


class LogNormalTruncated(ContinuousFrozenDistribution):
    """
    A lognormal distribution truncated in the log: X = exp(mu + sigma Z), with Z a
    standard normal conditioned on bot <= Z <= top. The bounds are in standard
    deviations of that normal, and None is no bound on that side, so with both
    None this is LogNormal(mu, sigma).

    Parameters
    ----------
    mu : float, optional
        Mean of the underlying normal before truncation, by default 0.0.
    sigma : float, optional
        Standard deviation of the underlying normal before truncation, by
        default 1.0. With sigma = 0 the distribution is the point exp(mu).
    bot : float or None, optional
        Lower truncation bound, by default -3.0.
    top : float or None, optional
        Upper truncation bound, by default 3.0.
    seed : int, optional
        Seed for random number generator, by default None.
    """

    def __new__(cls, mu=0.0, sigma=1.0, bot=-3.0, top=3.0, seed=None):
        if sigma == 0:
            # A point at exp(mu), whose bounds are the point
            return DiscreteDistribution([1.0], [np.exp(mu)], seed=seed)
        return super().__new__(cls)

    def __init__(self, mu=0.0, sigma=1.0, bot=-3.0, top=3.0, seed=None):
        if np.ndim(mu) or np.ndim(sigma):
            raise ValueError(
                "mu and sigma must be scalars; use IndexDistribution for time variation."
            )
        if sigma < 0:
            raise ValueError("sigma must be nonnegative, but it is " + str(sigma))
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.bot = _bound(bot, -np.inf)
        self.top = _bound(top, np.inf)
        if not self.bot < self.top:
            raise ValueError("The truncation bounds must satisfy bot < top.")
        super().__init__(
            _truncated_lognorm,
            s=self.sigma,
            lo=self.bot,
            hi=self.top,
            scale=np.exp(self.mu),
            loc=0,
            seed=seed,
        )
        self.infimum = np.array([np.exp(self.mu + self.sigma * self.bot)])
        self.supremum = np.array([np.exp(self.mu + self.sigma * self.top)])

    @property
    def mass(self):
        """The probability mass of the underlying normal that the truncation keeps."""
        return float(_normal_mass(self.bot, self.top))

    def z(self, x):
        """The standard normal coordinate (ln x - mu) / sigma of x."""
        with np.errstate(divide="ignore"):
            return (np.log(x) - self.mu) / self.sigma

    def moment(self, order):
        """
        The raw moment E[X**order], for any real order, in closed form.
        """
        k = np.asarray(order, dtype=float)
        ks = k * self.sigma
        return (
            np.exp(k * self.mu + 0.5 * ks**2)
            * _normal_mass(self.bot - ks, self.top - ks)
            / self.mass
        )

    def partial_moment(self, order, lo, hi):
        """
        The partial moment E[X**order 1{lo < X < hi}], for any real order and
        arrays of interval ends.
        """
        k = np.asarray(order, dtype=float)
        ks = k * self.sigma
        z_lo = np.maximum(
            self.z(np.maximum(np.asarray(lo, dtype=float), 0.0)), self.bot
        )
        z_hi = np.minimum(self.z(np.asarray(hi, dtype=float)), self.top)
        mass = np.maximum(_normal_mass(z_lo - ks, z_hi - ks), 0.0)
        return np.where(
            z_hi > z_lo, np.exp(k * self.mu + 0.5 * ks**2) * mass / self.mass, 0.0
        )

    def _approx_equiprobable(self, N, endpoints=False):
        """
        A discrete approximation with N atoms of probability 1/N, each the
        conditional mean of its bin of the truncated distribution, so the mean is
        exact.

        Parameters
        ----------
        N : int
            Number of atoms.
        endpoints : bool, optional
            Whether to add zero-probability atoms at the infimum and supremum,
            by default False.

        Returns
        -------
        DiscreteDistribution
        """
        cuts = _truncated_normal_ppf(np.arange(N + 1) / N, self.bot, self.top)
        cuts[0], cuts[-1] = self.bot, self.top
        pmv = np.ones(N) / N
        bin_mass = _normal_mass(cuts[:-1] - self.sigma, cuts[1:] - self.sigma)
        atoms = np.exp(self.mu + 0.5 * self.sigma**2) * bin_mass / (self.mass / N)
        if endpoints:
            atoms = np.r_[self.infimum, atoms, self.supremum]
            pmv = np.r_[0.0, pmv, 0.0]
        limit = {"dist": self, "method": "equiprobable", "N": N, "endpoints": endpoints}
        return DiscreteDistribution(pmv, atoms, seed=self.random_seed(), limit=limit)


class LogNormalTruncatedMeanOne(LogNormalTruncated):
    """
    A truncated lognormal with mean one after the truncation: mu is set so that
    E[X] = 1 for the truncated distribution. The bounds are in standard
    deviations of the underlying normal, by default three either side.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the underlying normal before truncation, by
        default 1.0. With sigma = 0 the distribution is the point 1.
    bot : float or None, optional
        Lower truncation bound, by default -3.0.
    top : float or None, optional
        Upper truncation bound, by default 3.0.
    seed : int, optional
        Seed for random number generator, by default 0.
    """

    def __new__(cls, sigma=1.0, bot=-3.0, top=3.0, seed=0):
        if sigma == 0:
            return DiscreteDistribution([1.0], [1.0], seed=seed)
        return super().__new__(cls)

    def __init__(self, sigma=1.0, bot=-3.0, top=3.0, seed=0):
        lo, hi = _bound(bot, -np.inf), _bound(top, np.inf)
        mu = -0.5 * sigma**2 - np.log(
            _normal_mass(lo - sigma, hi - sigma) / _normal_mass(lo, hi)
        )
        super().__init__(mu=mu, sigma=sigma, bot=bot, top=top, seed=seed)

    @classmethod
    def from_sd_log(cls, sd_log, bot=-3.0, top=3.0, seed=0):
        """
        The mean-one truncated lognormal whose log has standard deviation
        sd_log after the truncation. The underlying normal's sigma is
        sd_log / sqrt(truncated_normal_variance(bot, top)), so an estimated
        dispersion keeps its value under the truncation.

        Parameters
        ----------
        sd_log : float
            Standard deviation of ln X after truncation.
        bot : float or None, optional
            Lower truncation bound, by default -3.0.
        top : float or None, optional
            Upper truncation bound, by default 3.0.
        seed : int, optional
            Seed for random number generator, by default 0.

        Returns
        -------
        LogNormalTruncatedMeanOne or DiscreteDistribution
        """
        sigma = sd_log / np.sqrt(truncated_normal_variance(bot, top))
        return cls(sigma=sigma, bot=bot, top=top, seed=seed)


def _by_shape(fn, x, *shapes):
    """
    Apply fn(x, *shapes) once per distinct set of shape parameters, as scipy
    broadcasts them.
    """
    x, *shapes = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, *shapes))
    )
    out = np.empty(x.shape)
    keys = np.stack([shape.ravel() for shape in shapes], axis=1)
    for key in np.unique(keys, axis=0):
        sel = np.all(keys == key, axis=1).reshape(x.shape)
        out[sel] = fn(x[sel], *key)
    return out


# Close to the bound of a jumpless truncated lognormal the CDF is summed as a
# series in the distance from the bound, where its closed form, a difference of
# normal probabilities, would lose its digits: within this many standard
# deviations of the bound, and less for a bound far from the center.
_JUMPLESS_SERIES_WITHIN = 0.25
_JUMPLESS_SERIES_TERMS = 14


class _JumplessNormal:
    """
    The standard normal coordinate Z of a jumpless truncated lognormal whose
    bound is at bot < 0: on z >= bot, the standard normal density less its
    reflection in the bound,

        (phi(z) - phi(2 bot - z)) / (Phi(-bot) - Phi(bot)).

    At the bound the reflection equals the density, so the density of Z is zero
    there and rises linearly from it; a distance d above the bound the
    reflection is exp(2 bot d) of the density, so the upper tail is the
    normal's own.
    """

    def __init__(self, bot):
        self.bot = float(bot)
        b = self.b = -self.bot
        self.mass = float(_normal_mass(self.bot, b))
        self.phi = float(norm.pdf(b))
        self.near = min(_JUMPLESS_SERIES_WITHIN, 0.75 / b)
        # The CDF a distance d above the bound is 2 phi(b) / mass times the
        # integral of exp(-s**2 / 2) sinh(b s) over s from 0 to d, which is
        # d**2 times a power series in d**2 with these coefficients
        n = _JUMPLESS_SERIES_TERMS
        coef = np.zeros(n)
        for i in range(n):  # sinh: b**(2i + 1) s**(2i + 1) / (2i + 1)!
            for j in range(n - i):  # exp: (-1/2)**j s**(2j) / j!
                coef[i + j] += (
                    b ** (2 * i + 1) / factorial(2 * i + 1) * (-0.5) ** j / factorial(j)
                )
        self._series = coef / (2.0 * np.arange(n) + 2.0)

    def pdf(self, z):
        z = np.asarray(z, dtype=float)
        d = z - self.bot
        # close to the bound, the difference in a form that keeps its digits
        near = np.clip(d, 0.0, 1.0)
        close = 2.0 * self.phi * np.exp(-0.5 * near * near) * np.sinh(self.b * near)
        far = norm.pdf(z) - norm.pdf(z + 2.0 * self.b)
        return np.where(d <= 0.0, 0.0, np.where(d < 1.0, close, far)) / self.mass

    def cdf(self, z):
        z = np.asarray(z, dtype=float)
        d = np.maximum(z - self.bot, 0.0)
        d2 = np.minimum(d, self.near) ** 2
        close = 2.0 * self.phi * d2 * np.polynomial.polynomial.polyval(d2, self._series)
        far = ndtr(z) - 2.0 * ndtr(self.bot) + ndtr(-z - 2.0 * self.b)
        return np.minimum(np.where(d < self.near, close, far) / self.mass, 1.0)

    def sf(self, z):
        """1 - cdf(z), with the upper tail's digits kept."""
        z = np.asarray(z, dtype=float)
        out = (ndtr(-z) - ndtr(-z - 2.0 * self.b)) / self.mass
        return np.where(z <= self.bot, 1.0, out)

    def ppf(self, q):
        """
        The quantile function, by Newton's method on the logarithm of the
        probability on the nearer side: the law's density is log-concave, so
        that logarithm is concave and the steps close on the quantile.
        """
        q = np.asarray(q, dtype=float)
        out = np.full(q.shape, np.nan)
        out[q == 0.0] = self.bot
        out[q == 1.0] = np.inf
        inner = (q > 0.0) & (q < 1.0)
        if not np.any(inner):
            return out
        p = q[inner]
        low = p <= 0.5
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Starts, as distances above the bound. Close to it the probability
            # is phi(b) b d**2 / mass; away from it the reflection has died off
            d = np.where(
                low,
                np.minimum(
                    np.sqrt(p * self.mass / (self.phi * self.b)),
                    ndtri(p * self.mass + 2.0 * ndtr(self.bot)) + self.b,
                ),
                self.b - ndtri((1.0 - p) * self.mass),
            )
            d = np.maximum(d, np.finfo(float).tiny)
            target = np.where(low, np.log(p), np.log1p(-p))
            sign = np.where(low, 1.0, -1.0)
            for _ in range(200):
                z = self.bot + d
                prob = np.where(low, self.cdf(z), self.sf(z))
                step = sign * (np.log(prob) - target) * prob / self.pdf(z)
                # a probability or a density that has underflowed leaves the start
                step = np.where(np.isfinite(step), step, 0.0)
                new = np.where(d - step > 0.0, d - step, 0.1 * d)
                done = np.abs(new - d) <= 1e-15 * np.maximum(d, 1.0)
                d = new
                if np.all(done):
                    break
        out[inner] = self.bot + d
        return out

    def expectation_exp(self, c, z_lo, z_hi):
        """E[exp(c Z) 1{z_lo < Z < z_hi}], for arrays of interval ends."""
        lo = np.maximum(np.asarray(z_lo, dtype=float), self.bot)
        hi = np.asarray(z_hi, dtype=float)
        shift = 2.0 * self.b
        with np.errstate(invalid="ignore"):
            body = np.exp(0.5 * c * c) * (
                _normal_mass(lo - c, hi - c)
                - np.exp(-shift * c) * _normal_mass(lo + shift - c, hi + shift - c)
            )
        return np.where(hi > lo, body, 0.0) / self.mass

    def mean(self):
        """E[Z]."""
        return 2.0 * self.b * float(ndtr(self.bot)) / self.mass

    def variance(self):
        """Var(Z)."""
        b = self.b
        second = (
            2.0 * b * self.phi + ndtr(b) - (1.0 + 4.0 * b * b) * ndtr(self.bot)
        ) / self.mass
        return float(second - self.mean() ** 2)


@functools.lru_cache(maxsize=None)
def _jumpless_normal(bot):
    return _JumplessNormal(bot)


class _jumpless_lognorm_gen(rv_continuous):
    """
    The jumpless truncated lognormal exp(s Z), with Z a jumpless normal bounded
    below at lo < 0, in its standard deviations. scipy's scale is exp(mu).
    """

    def _argcheck(self, s, lo):
        return (s > 0) & (lo < 0)

    def _get_support(self, s, lo):
        return np.exp(s * lo), np.inf

    def _pdf(self, x, s, lo):
        def pdf(x, s, lo):
            with np.errstate(divide="ignore"):
                z = np.log(x) / s
            return _jumpless_normal(lo).pdf(z) / (s * x)

        return _by_shape(pdf, x, s, lo)

    def _cdf(self, x, s, lo):
        def cdf(x, s, lo):
            with np.errstate(divide="ignore"):
                return _jumpless_normal(lo).cdf(np.log(x) / s)

        return _by_shape(cdf, x, s, lo)

    def _sf(self, x, s, lo):
        def sf(x, s, lo):
            with np.errstate(divide="ignore"):
                return _jumpless_normal(lo).sf(np.log(x) / s)

        return _by_shape(sf, x, s, lo)

    def _ppf(self, q, s, lo):
        return _by_shape(
            lambda q, s, lo: np.exp(s * _jumpless_normal(lo).ppf(q)), q, s, lo
        )

    def _munp(self, n, s, lo):
        return _by_shape(
            lambda n, s, lo: _jumpless_normal(lo).expectation_exp(
                n * s, -np.inf, np.inf
            ),
            n,
            s,
            lo,
        )


_jumpless_lognorm = _jumpless_lognorm_gen(
    a=0.0, name="jumpless_lognorm", shapes="s, lo"
)


class LogNormalTruncatedJumpless(ContinuousFrozenDistribution):
    """
    A lognormal truncated below whose density has no jump at the truncation:
    X = exp(mu + sigma Z), where Z, on z >= bot, has the standard normal
    density less its reflection in the bound,

        (phi(z) - phi(2 bot - z)) / (Phi(-bot) - Phi(bot)).

    A lognormal that is simply truncated (LogNormalTruncated) has a density
    that jumps from zero to a positive value at its bound. Here the reflection
    equals the density at the bound, so the density of X is zero at the bound,
    which is its infimum, and rises linearly from it; and the reflection dies
    off above the bound (half a standard deviation above a bound of -3 it is 5%
    of the density), so that everywhere else the law is the lognormal's own,
    with the lognormal's upper tail and no upper bound. What the reflection
    takes out is the lognormal's tail below the bound, folded over the bound:
    the law leaves out twice that tail's probability, 0.27% at a bound of -3.

    The bound is in standard deviations of the underlying normal from its mean,
    and lies below it: bot = -3.0 puts it at exp(mu - 3 sigma). A positive bot
    is read as that many standard deviations below the mean. Neither mu nor
    sigma is a moment of this law: its mean is moment(1), a little above
    exp(mu + sigma**2 / 2), and the standard deviation of ln X is a little below
    sigma (LogNormalTruncatedJumplessMeanOne sets both).

    Parameters
    ----------
    mu : float, optional
        Mean of the underlying normal, by default 0.0.
    sigma : float, optional
        Standard deviation of the underlying normal, by default 1.0. With
        sigma = 0 the distribution is the point exp(mu).
    bot : float, optional
        The truncation, in standard deviations of the underlying normal below
        its mean, by default -3.0 (3.0 means the same).
    seed : int, optional
        Seed for random number generator, by default None.
    """

    def __new__(cls, mu=0.0, sigma=1.0, bot=-3.0, seed=None):
        if sigma == 0:
            # A point at exp(mu), whose bounds are the point
            return DiscreteDistribution([1.0], [np.exp(mu)], seed=seed)
        return super().__new__(cls)

    def __init__(self, mu=0.0, sigma=1.0, bot=-3.0, seed=None):
        if np.ndim(mu) or np.ndim(sigma):
            raise ValueError(
                "mu and sigma must be scalars; use IndexDistribution for time variation."
            )
        if sigma < 0:
            raise ValueError("sigma must be nonnegative, but it is " + str(sigma))
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.bot = _jumpless_bound(bot)
        self.top = np.inf
        self._law = _jumpless_normal(self.bot)
        super().__init__(
            _jumpless_lognorm,
            s=self.sigma,
            lo=self.bot,
            scale=np.exp(self.mu),
            loc=0,
            seed=seed,
        )
        self.infimum = np.array([np.exp(self.mu + self.sigma * self.bot)])
        self.supremum = np.array([np.inf])

    @property
    def mass(self):
        """
        The probability mass of the underlying normal that the law keeps: all
        but twice the tail below the bound.
        """
        return self._law.mass

    # The density, CDF, survival function and quantile function come straight
    # from the law of Z, as scipy's would through _jumpless_lognorm, but without
    # its handling of shape parameters that may vary by element, which this
    # law's, being scalars, never do.

    def _in_z(self, x):
        """
        The coordinate z of x, and where x is above the infimum, which z
        recovers only to rounding.
        """
        x = np.asarray(x, dtype=float)
        inside = x > self.infimum[0]
        return self.z(np.where(inside, x, 1.0)), inside, x

    def pdf(self, x):
        """The density of X at x."""
        z, inside, x = self._in_z(x)
        out = self._law.pdf(z) / (self.sigma * np.where(inside, x, 1.0))
        return np.where(inside, out, 0.0)[()]

    def cdf(self, x):
        """P(X <= x)."""
        z, inside, _ = self._in_z(x)
        return np.where(inside, self._law.cdf(z), 0.0)[()]

    def sf(self, x):
        """P(X > x)."""
        z, inside, _ = self._in_z(x)
        return np.where(inside, self._law.sf(z), 1.0)[()]

    def ppf(self, q):
        """The quantile function: the x with P(X <= x) = q, nan outside [0, 1]."""
        q = np.asarray(q, dtype=float)
        with np.errstate(over="ignore"):
            out = np.exp(self.mu + self.sigma * self._law.ppf(np.clip(q, 0.0, 1.0)))
        return np.where((q >= 0.0) & (q <= 1.0), out, np.nan)[()]

    def z(self, x):
        """The standard normal coordinate (ln x - mu) / sigma of x."""
        with np.errstate(divide="ignore"):
            return (np.log(x) - self.mu) / self.sigma

    def moment(self, order):
        """The raw moment E[X**order], for any real order, in closed form."""
        k = float(order)
        return np.exp(k * self.mu) * self._law.expectation_exp(
            k * self.sigma, -np.inf, np.inf
        )

    def _approx_equiprobable(self, N, endpoints=False):
        """
        A discrete approximation with N atoms of probability 1/N, each the
        conditional mean of its bin, so the mean is exact.

        Parameters
        ----------
        N : int
            Number of atoms.
        endpoints : bool, optional
            Whether to add zero-probability atoms at the infimum and supremum,
            by default False.

        Returns
        -------
        DiscreteDistribution
        """
        cuts = self._law.ppf(np.arange(N + 1) / N)
        cuts[0], cuts[-1] = self.bot, np.inf
        pmv = np.ones(N) / N
        atoms = (
            np.exp(self.mu)
            * self._law.expectation_exp(self.sigma, cuts[:-1], cuts[1:])
            * N
        )
        if endpoints:
            atoms = np.r_[self.infimum, atoms, self.supremum]
            pmv = np.r_[0.0, pmv, 0.0]
        limit = {"dist": self, "method": "equiprobable", "N": N, "endpoints": endpoints}
        return DiscreteDistribution(pmv, atoms, seed=self.random_seed(), limit=limit)


def _jumpless_bound(bot):
    """
    The bound of a jumpless truncated lognormal, in standard deviations of the
    underlying normal from its mean: below the mean, so a positive number is
    read as the distance below.
    """
    if bot is None or not np.isfinite(bot) or bot == 0:
        raise ValueError(
            "A jumpless truncated lognormal needs a finite bound below the mean of "
            "the underlying normal, in its standard deviations, such as -3.0; "
            "without a bound the law is LogNormal."
        )
    return -abs(float(bot))


class LogNormalTruncatedJumplessMeanOne(LogNormalTruncatedJumpless):
    """
    A jumpless truncated lognormal with mean one: mu is set so that E[X] = 1.
    The bound is in standard deviations of the underlying normal below its
    mean, by default three.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the underlying normal, by default 1.0. With
        sigma = 0 the distribution is the point 1.
    bot : float, optional
        The truncation, in standard deviations of the underlying normal below
        its mean, by default -3.0 (3.0 means the same).
    seed : int, optional
        Seed for random number generator, by default 0.
    """

    def __new__(cls, sigma=1.0, bot=-3.0, seed=0):
        if sigma == 0:
            return DiscreteDistribution([1.0], [1.0], seed=seed)
        return super().__new__(cls)

    def __init__(self, sigma=1.0, bot=-3.0, seed=0):
        law = _jumpless_normal(_jumpless_bound(bot))
        mu = -np.log(float(law.expectation_exp(sigma, -np.inf, np.inf)))
        super().__init__(mu=mu, sigma=sigma, bot=bot, seed=seed)

    @classmethod
    def from_sd_log(cls, sd_log, bot=-3.0, seed=0):
        """
        The mean-one jumpless truncated lognormal whose log has standard
        deviation sd_log. The truncation and the reflection narrow the log
        slightly, so the underlying normal's sigma is set a little above
        sd_log, so that an estimated dispersion keeps its value.

        Parameters
        ----------
        sd_log : float
            Standard deviation of ln X.
        bot : float, optional
            The truncation, in standard deviations of the underlying normal
            below its mean, by default -3.0 (3.0 means the same).
        seed : int, optional
            Seed for random number generator, by default 0.

        Returns
        -------
        LogNormalTruncatedJumplessMeanOne or DiscreteDistribution
        """
        sigma = sd_log / np.sqrt(_jumpless_normal(_jumpless_bound(bot)).variance())
        return cls(sigma=sigma, bot=bot, seed=seed)


# The names these classes had before they were renamed. Each is the class under
# its new name, with a DeprecationWarning where it is looked up (__getattr__
# below, and its counterpart in HARK.distributions); none is scheduled for removal.
_RENAMED = {
    "Lognormal": "LogNormal",
    "MeanOneLogNormal": "LogNormalMeanOne",
    "TruncatedLognormal": "LogNormalTruncated",
    "TruncatedLogNormal": "LogNormalTruncated",
    "MeanOneTruncatedLogNormal": "LogNormalTruncatedMeanOne",
}


def _renamed_message(old):
    """The warning for a lookup of the old name old, which names the new one."""
    return (
        f"{old} is now called {_RENAMED[old]}. The name {old} still works and is "
        "not scheduled for removal."
    )


def __getattr__(name):
    # PEP 562: an old name, looked up in this module (by an import, an attribute
    # access, or pickle loading an object stored under it), is the class itself
    if name in _RENAMED:
        warnings.warn(
            _renamed_message(name),
            DeprecationWarning,
            stacklevel=2,
            # an import from a package looks the name up twice, once inside the
            # import machinery: both warnings name the line of the import
            skip_file_prefixes=("<frozen importlib",),
        )
        return globals()[_RENAMED[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

    def __init__(self, bot=0.0, top=1.0, seed=None):
        self.bot = np.asarray(bot)
        self.top = np.asarray(top)

        super().__init__(uniform, loc=self.bot, scale=self.top - self.bot, seed=seed)
        self.infimum = np.array(self.bot, dtype=float, ndmin=1)
        self.supremum = np.array(self.top, dtype=float, ndmin=1)

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

        limit = {
            "dist": self,
            "method": "equiprobable",
            "N": N,
            "endpoints": endpoints,
        }

        return DiscreteDistribution(
            pmv,
            atoms,
            seed=self.random_seed(),
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

    def __init__(self, scale=1.0, shape=1.0, seed=None):
        self.scale = np.asarray(scale)
        self.shape = np.asarray(shape)

        # Set up the RNG
        super().__init__(weibull_min, c=shape, scale=scale, seed=seed)
        self.infimum = np.array([0.0])
        self.supremum = np.array([np.inf])
