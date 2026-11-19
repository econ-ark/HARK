import functools
from typing import Any, Optional, Union

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfc, hyp1f1, log_ndtr, ndtr, ndtri
from scipy.stats import (
    rv_continuous,
    norm,
    lognorm,
    uniform,
    weibull_min,
)
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from HARK.distributions.base import Distribution, _renamed_class
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


def _log_coordinate(mu, sigma):
    """The standard normal coordinate (ln x - mu) / sigma, -inf where x <= 0."""

    def coord(x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                x > 0.0, (np.log(np.where(x > 0.0, x, 1.0)) - mu) / sigma, -np.inf
            )

    return coord


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

    def normal_coordinate(self):
        """
        The coordinate in which this law is the standard normal, as
        expected_continuous() integrates in it: z = (x - mu) / sigma.

        Returns
        -------
        bot, top : float
            The bounds of z, here -inf and inf.
        value : callable
            The law's value at z.
        coord : callable
            The coordinate z of a value.
        """
        mu, sigma = float(np.squeeze(self.mu)), float(np.squeeze(self.sigma))
        return -np.inf, np.inf, lambda z: mu + sigma * z, lambda x: (x - mu) / sigma

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

    def normal_coordinate(self):
        """
        The coordinate in which this law is the standard normal, as
        expected_continuous() integrates in it: z = (ln x - mu) / sigma.

        Returns
        -------
        bot, top : float
            The bounds of z, here -inf and inf.
        value : callable
            The law's value at z.
        coord : callable
            The coordinate z of a value, -inf at values not above zero.
        """
        mu, sigma = float(np.squeeze(self.mu)), float(np.squeeze(self.sigma))
        return (
            -np.inf,
            np.inf,
            lambda z: np.exp(mu + sigma * z),
            _log_coordinate(mu, sigma),
        )

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

    def normal_coordinate(self):
        """
        The coordinate in which this law is the standard normal truncated to
        [bot, top], as expected_continuous() integrates in it: the underlying
        normal's z = (ln x - mu) / sigma.

        Returns
        -------
        bot, top : float
            The bounds of z, the truncation's.
        value : callable
            The law's value at z.
        coord : callable
            The coordinate z of a value, -inf at values not above zero.
        """
        mu, sigma = self.mu, self.sigma
        return (
            self.bot,
            self.top,
            lambda z: np.exp(mu + sigma * z),
            _log_coordinate(mu, sigma),
        )

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


# Each tail of a tapered lognormal is tapered over this many standard deviations
# of the log, ending at its bound.
_TAPER_SD = 1.0


def _taper_moment(eta, k, s=1.0):
    """
    E[exp(k T) 1{T > 1 - s}] for T ~ Beta(1, eta + 1), whose density is
    (eta + 1) (1 - t)**eta on [0, 1], with s in [0, 1] the distance left to the
    end at t = 1. Substituting u = 1 - t and using Kummer's transformation, it is
    exp(k) s**(eta + 1) M(eta + 1, eta + 2, -k s), with M the confluent
    hypergeometric function; at s = 1 it is M(1, eta + 2, k).
    """
    s = np.asarray(s, dtype=float)
    return np.exp(k) * s ** (eta + 1.0) * hyp1f1(eta + 1.0, eta + 2.0, -k * s)


@functools.lru_cache(maxsize=None)
def _fit_taper(a, k):
    """
    The power eta of the tail that replaces a standard normal's tail beyond a.
    With t the distance past a, measured outward, the tail's density on
    0 <= t <= 1 is (eta + 1) P (1 - t)**eta, with P = P(Z > a): it carries the
    normal tail's probability and is zero at t = 1. eta is set so that the
    tail's integral of exp(k t) is the normal tail's, P exp(k**2/2 - k a)
    P(Z > a - k) / P(Z > a); with k = sigma on the upper tail and -sigma on the
    lower, that is the tail's share of the lognormal's mean.

    The integral falls with eta when k > 0 and rises with it when k < 0, from
    its value for a uniform tail at eta = 0 toward P as eta grows, so eta is the
    one root of a one-dimensional equation.

    Returns
    -------
    eta, or None if no eta > 0 fits: the normal tail's mean lies farther out
    than a uniform tail's, so no taper over one standard deviation carries it.
    """
    log_target = 0.5 * k * k - k * a + log_ndtr(k - a) - log_ndtr(-a)

    def gap(eta):
        return np.log(_taper_moment(eta, k)) - log_target

    lo, hi = 1e-12, 1.0
    if gap(lo) * np.sign(k) <= 0.0:
        return None
    while gap(hi) * np.sign(k) > 0.0:
        hi *= 2.0
        if hi > 1e6:
            return None
    return float(brentq(gap, lo, hi, xtol=1e-15, rtol=4.0 * np.finfo(float).eps))


class _TaperedTail:
    """
    One tapered tail of a tapered standard normal: it starts at cut and runs
    outward, in the direction outward (+1 for the upper tail, -1 for the lower),
    for _TAPER_SD standard deviations to its bound. At distance t past the cut
    its density is height (1 - t)**eta, with height = (eta + 1) prob and prob the
    normal's probability beyond the cut.
    """

    def __init__(self, cut, outward, sigma):
        self.cut = float(cut)
        self.outward = outward
        self.a = outward * self.cut
        self.prob = float(ndtr(-self.a))
        eta = _fit_taper(self.a, outward * float(sigma))
        if eta is None:
            raise ValueError(
                "No tapered tail fits beyond "
                + str(self.cut)
                + " standard deviations: the lognormal's tail there has its mean "
                "too far out for a taper over one standard deviation to carry. "
                "Put the bounds farther out."
            )
        self.eta = eta
        self.height = (eta + 1.0) * self.prob
        # The density of X along the tail is proportional to
        # (1 - t)**eta exp(-outward sigma t), which falls all the way to the
        # bound only if eta + outward sigma > 0
        if not eta + outward * float(sigma) > 0.0:
            raise ValueError(
                "The density of X beyond "
                + str(self.cut)
                + " standard deviations would rise before falling to its bound. "
                "Put the bounds farther out: at three standard deviations from "
                "the median both tails fall to their bounds for sigma below 0.8, "
                "at 3.5 for sigma below 1.27, and at four for sigma below 1.74."
            )

    def density(self, d):
        """The density of Z at distance d past the cut, for d in [0, 1]."""
        d = np.clip(np.asarray(d, dtype=float), 0.0, 1.0)
        with np.errstate(divide="ignore"):
            return self.height * np.exp(self.eta * np.log1p(-d))

    def integral(self, d, k=0.0):
        """
        The integral of density(t) exp(k t) over t from d to 1, for an array of d
        in [0, 1]: the tail's probability beyond d when k = 0.
        """
        d = np.clip(np.asarray(d, dtype=float), 0.0, 1.0)
        return self.prob * _taper_moment(self.eta, k, 1.0 - d)

    def distance(self, prob):
        """The distance d past the cut beyond which the tail holds prob."""
        share = np.clip(np.asarray(prob, dtype=float) / self.prob, 0.0, 1.0)
        with np.errstate(divide="ignore"):
            return -np.expm1(np.log(share) / (self.eta + 1.0))

    def moments(self):
        """The integrals of density(t) t and density(t) t**2 over the tail."""
        b = self.eta + 2.0
        return self.prob / b, 2.0 * self.prob / (b * (b + 1.0))


class _TaperedNormal:
    """
    The standard normal coordinate Z of a tapered lognormal: a standard normal
    on [bot + 1, top - 1] and a tapered tail from there to each finite bound.
    """

    def __init__(self, sigma, bot, top):
        self.bot, self.top = float(bot), float(top)
        self.lower = (
            _TaperedTail(bot + _TAPER_SD, -1, sigma) if np.isfinite(bot) else None
        )
        self.upper = (
            _TaperedTail(top - _TAPER_SD, 1, sigma) if np.isfinite(top) else None
        )
        self.c_lo = self.lower.cut if self.lower else -np.inf
        self.c_hi = self.upper.cut if self.upper else np.inf

    def pdf(self, z):
        z = np.asarray(z, dtype=float)
        out = np.where((z >= self.c_lo) & (z <= self.c_hi), norm.pdf(z), 0.0)
        if self.lower:
            past = (z < self.c_lo) & (z >= self.bot)
            out = np.where(past, self.lower.density(self.c_lo - z), out)
        if self.upper:
            past = (z > self.c_hi) & (z <= self.top)
            out = np.where(past, self.upper.density(z - self.c_hi), out)
        return out

    def cdf(self, z):
        z = np.asarray(z, dtype=float)
        out = ndtr(z)
        if self.lower:
            out = np.where(z < self.c_lo, self.lower.integral(self.c_lo - z), out)
        if self.upper:
            out = np.where(z > self.c_hi, 1.0 - self.upper.integral(z - self.c_hi), out)
        return np.where(z <= self.bot, 0.0, np.where(z >= self.top, 1.0, out))

    def sf(self, z):
        """1 - cdf(z), with the upper tail's digits kept."""
        z = np.asarray(z, dtype=float)
        out = ndtr(-z)
        if self.lower:
            out = np.where(z < self.c_lo, 1.0 - self.lower.integral(self.c_lo - z), out)
        if self.upper:
            out = np.where(z > self.c_hi, self.upper.integral(z - self.c_hi), out)
        return np.where(z <= self.bot, 1.0, np.where(z >= self.top, 0.0, out))

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = ndtri(q)
        if self.lower:
            low = q < self.lower.prob
            out = np.where(low, self.c_lo - self.lower.distance(q), out)
        if self.upper:
            high = q > 1.0 - self.upper.prob
            out = np.where(high, self.c_hi + self.upper.distance(1.0 - q), out)
        out = np.where(q <= 0.0, self.bot, out)
        return np.where(q >= 1.0, self.top, out)

    def expectation_exp(self, c, z_lo, z_hi):
        """E[exp(c Z) 1{z_lo < Z < z_hi}], for arrays of interval ends."""
        z_lo, z_hi = np.asarray(z_lo, dtype=float), np.asarray(z_hi, dtype=float)
        lo, hi = np.maximum(z_lo, self.c_lo), np.minimum(z_hi, self.c_hi)
        body = np.where(
            hi > lo, np.exp(0.5 * c * c) * _normal_mass(lo - c, hi - c), 0.0
        )
        if self.upper:
            d1 = np.clip(z_lo - self.c_hi, 0.0, 1.0)
            d2 = np.clip(z_hi - self.c_hi, 0.0, 1.0)
            body = body + np.exp(c * self.c_hi) * (
                self.upper.integral(d1, c) - self.upper.integral(d2, c)
            )
        if self.lower:
            d1 = np.clip(self.c_lo - z_hi, 0.0, 1.0)
            d2 = np.clip(self.c_lo - z_lo, 0.0, 1.0)
            body = body + np.exp(c * self.c_lo) * (
                self.lower.integral(d1, -c) - self.lower.integral(d2, -c)
            )
        return body

    def variance(self):
        """Var(Z)."""
        lo, hi = self.c_lo, self.c_hi
        pdf_lo, pdf_hi = norm.pdf(lo), norm.pdf(hi)
        mean = pdf_lo - pdf_hi
        second = (
            _normal_mass(lo, hi)
            + (lo * pdf_lo if np.isfinite(lo) else 0.0)
            - (hi * pdf_hi if np.isfinite(hi) else 0.0)
        )
        for tail in (self.lower, self.upper):
            if tail:
                # Z = cut + outward t on the tail
                m1, m2 = tail.moments()
                mean += tail.prob * tail.cut + tail.outward * m1
                second += (
                    tail.prob * tail.cut**2 + 2.0 * tail.cut * tail.outward * m1 + m2
                )
        return float(second - mean**2)


@functools.lru_cache(maxsize=None)
def _tapered_normal(sigma, bot, top):
    return _TaperedNormal(sigma, bot, top)


def _by_shape(fn, x, s, lo, hi):
    """Apply fn(x, s, lo, hi) once per distinct (s, lo, hi), as scipy broadcasts them."""
    x, s, lo, hi = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, s, lo, hi))
    )
    out = np.empty(x.shape)
    keys = np.stack([s.ravel(), lo.ravel(), hi.ravel()], axis=1)
    for key in np.unique(keys, axis=0):
        sel = np.all(keys == key, axis=1).reshape(x.shape)
        out[sel] = fn(x[sel], *key)
    return out


class _tapered_lognorm_gen(rv_continuous):
    """
    The tapered lognormal exp(s Z), with Z a tapered standard normal on [lo, hi]
    in its standard deviations. scipy's scale is exp(mu).
    """

    def _argcheck(self, s, lo, hi):
        return (s > 0) & (hi - lo >= 2.0 * _TAPER_SD)

    def _get_support(self, s, lo, hi):
        return np.exp(s * lo), np.exp(s * hi)

    def _pdf(self, x, s, lo, hi):
        def pdf(x, s, lo, hi):
            with np.errstate(divide="ignore"):
                z = np.log(x) / s
            return _tapered_normal(s, lo, hi).pdf(z) / (s * x)

        return _by_shape(pdf, x, s, lo, hi)

    def _cdf(self, x, s, lo, hi):
        def cdf(x, s, lo, hi):
            with np.errstate(divide="ignore"):
                return _tapered_normal(s, lo, hi).cdf(np.log(x) / s)

        return _by_shape(cdf, x, s, lo, hi)

    def _ppf(self, q, s, lo, hi):
        return _by_shape(
            lambda q, s, lo, hi: np.exp(s * _tapered_normal(s, lo, hi).ppf(q)),
            q,
            s,
            lo,
            hi,
        )

    def _munp(self, n, s, lo, hi):
        return _by_shape(
            lambda n, s, lo, hi: _tapered_normal(s, lo, hi).expectation_exp(
                n * s, -np.inf, np.inf
            ),
            n,
            s,
            lo,
            hi,
        )


_tapered_lognorm = _tapered_lognorm_gen(
    a=0.0, name="tapered_lognorm", shapes="s, lo, hi"
)


class LogNormalTapered(ContinuousFrozenDistribution):
    """
    A lognormal whose tails are tapered to zero at its bounds: X = exp(mu + sigma Z),
    where Z has the standard normal density between bot + 1 and top - 1 and, over
    the last standard deviation before each bound, a power taper:

        h (1 - t)**eta,

    with t in [0, 1] the distance past the start of the tapered tail, measured
    outward. Each tail's power eta is the one for which the tail carries exactly
    the share of E[X] of the lognormal's tail that it replaces, and
    h = (eta + 1) P, with P that tail's probability, so that it carries the
    probability too; within the tail, t has the distribution Beta(1, eta + 1).
    So the density of X is LogNormal(mu, sigma)'s own between the tapered tails,
    it steps to h / (sigma x) at the start of each tapered tail, it is zero at the
    bounds, which are its infimum and supremum, and E[X] is
    exp(mu + sigma**2 / 2), as for LogNormal(mu, sigma).

    The bounds are in standard deviations of the underlying normal, and None is
    no bound on that side, where the lognormal's own tail is kept. With both None
    this is LogNormal(mu, sigma). The density of X must fall all the way to each
    bound, and bounds for which it would rise first, or for which no taper with
    eta > 0 carries the tail's mean, are refused: at three standard deviations
    from the median both tails fall to their bounds for sigma below 0.8, at 3.5
    for sigma below 1.27, and at four for sigma below 1.74.

    Parameters
    ----------
    mu : float, optional
        Mean of the underlying normal, by default 0.0.
    sigma : float, optional
        Standard deviation of the underlying normal, by default 1.0. With
        sigma = 0 the distribution is the point exp(mu).
    bot : float or None, optional
        Lower bound, by default -3.0.
    top : float or None, optional
        Upper bound, by default 3.0.
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
        if not self.top - self.bot >= 2.0 * _TAPER_SD:
            raise ValueError(
                "The bounds must be at least two standard deviations apart, so "
                "that each tapered tail has its standard deviation."
            )
        self._law = _tapered_normal(self.sigma, self.bot, self.top)
        super().__init__(
            _tapered_lognorm,
            s=self.sigma,
            lo=self.bot,
            hi=self.top,
            scale=np.exp(self.mu),
            loc=0,
            seed=seed,
        )
        self.infimum = np.array([np.exp(self.mu + self.sigma * self.bot)])
        self.supremum = np.array([np.exp(self.mu + self.sigma * self.top)])

    # The density, CDF, survival function and quantile function come straight
    # from the tapered normal, as scipy's would through _tapered_lognorm, but
    # without its handling of shape parameters that may vary by element, which
    # this law's, being scalars, never do.

    def _in_z(self, x):
        """
        The coordinate z of x, and where x is below, inside and above the open
        support, so that the bounds, which z recovers only to rounding, are
        outside it.
        """
        x = np.asarray(x, dtype=float)
        below, above = x <= self.infimum[0], x >= self.supremum[0]
        inside = ~(below | above)
        return self.z(np.where(inside, x, 1.0)), below, inside, above, x

    def pdf(self, x):
        """The density of X at x."""
        z, _, inside, _, x = self._in_z(x)
        out = self._law.pdf(z) / (self.sigma * np.where(inside, x, 1.0))
        return np.where(inside, out, 0.0)[()]

    def cdf(self, x):
        """P(X <= x)."""
        z, _, inside, above, _ = self._in_z(x)
        return np.where(inside, self._law.cdf(z), np.where(above, 1.0, 0.0))[()]

    def sf(self, x):
        """P(X > x)."""
        z, below, inside, _, _ = self._in_z(x)
        return np.where(inside, self._law.sf(z), np.where(below, 1.0, 0.0))[()]

    def ppf(self, q):
        """The quantile function: the x with P(X <= x) = q, nan outside [0, 1]."""
        q = np.asarray(q, dtype=float)
        out = np.exp(self.mu + self.sigma * self._law.ppf(np.clip(q, 0.0, 1.0)))
        return np.where((q >= 0.0) & (q <= 1.0), out, np.nan)[()]

    def z(self, x):
        """The standard normal coordinate (ln x - mu) / sigma of x."""
        with np.errstate(divide="ignore"):
            return (np.log(x) - self.mu) / self.sigma

    def breakpoints(self):
        """
        The values at which the density steps: where each tapered tail starts,
        one standard deviation inside its bound.

        Returns
        -------
        np.ndarray
            The points, in increasing order.
        """
        starts = (self._law.c_lo, self._law.c_hi)
        return np.array(
            [np.exp(self.mu + self.sigma * c) for c in starts if np.isfinite(c)]
        )

    def moment(self, order):
        """The raw moment E[X**order], for any real order."""
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
        cuts[0], cuts[-1] = self.bot, self.top
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


class LogNormalTaperedMeanOne(LogNormalTapered):
    """
    A tapered lognormal with mean one: mu = -sigma**2 / 2, since the tapering
    keeps the lognormal's mean. The bounds are in standard deviations of the
    underlying normal, by default three either side, so the tapered tails start
    two standard deviations out.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the underlying normal, by default 1.0. With
        sigma = 0 the distribution is the point 1.
    bot : float or None, optional
        Lower bound, by default -3.0.
    top : float or None, optional
        Upper bound, by default 3.0.
    seed : int, optional
        Seed for random number generator, by default 0.
    """

    def __new__(cls, sigma=1.0, bot=-3.0, top=3.0, seed=0):
        if sigma == 0:
            return DiscreteDistribution([1.0], [1.0], seed=seed)
        return super().__new__(cls)

    def __init__(self, sigma=1.0, bot=-3.0, top=3.0, seed=0):
        super().__init__(mu=-0.5 * sigma**2, sigma=sigma, bot=bot, top=top, seed=seed)

    @classmethod
    def from_sd_log(cls, sd_log, bot=-3.0, top=3.0, seed=0):
        """
        The mean-one tapered lognormal whose log has standard deviation sd_log.
        The tapering narrows the log slightly, so the underlying normal's sigma
        is set a little above sd_log, so that an estimated dispersion keeps its
        value.

        Parameters
        ----------
        sd_log : float
            Standard deviation of ln X.
        bot : float or None, optional
            Lower bound, by default -3.0.
        top : float or None, optional
            Upper bound, by default 3.0.
        seed : int, optional
            Seed for random number generator, by default 0.

        Returns
        -------
        LogNormalTaperedMeanOne or DiscreteDistribution
        """
        if sd_log == 0:
            return cls(sigma=0.0, bot=bot, top=top, seed=seed)
        lo, hi = _bound(bot, -np.inf), _bound(top, np.inf)
        sigma = float(sd_log)
        for _ in range(50):
            new = sd_log / np.sqrt(_tapered_normal(sigma, lo, hi).variance())
            if abs(new - sigma) <= 1e-15 * sigma:
                break
            sigma = new
        return cls(sigma=sigma, bot=bot, top=top, seed=seed)


def _pooled_points(sigma, bot, top):
    """
    The pooled points of a lognormal exp(sigma Z) with tails beyond bot and top
    pooled at their means, divided by its mean exp(sigma**2 / 2): the
    conditional means of the tails, in logs so that far tails keep their digits.
    An unpooled side (an infinite bound) gives nan.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = (
            np.exp(log_ndtr(bot - sigma) - log_ndtr(bot))
            if np.isfinite(bot)
            else np.nan
        )
        hi = (
            np.exp(log_ndtr(sigma - top) - log_ndtr(-top))
            if np.isfinite(top)
            else np.nan
        )
    return float(lo), float(hi)


def _pooled_log_variance(sigma, bot, top):
    """
    Var(ln X) / sigma**2 for a lognormal exp(mu + sigma Z) whose tails beyond bot
    and top, in standard deviations of Z, are pooled at their means.
    """
    mass = float(_normal_mass(bot, top))
    pdf_lo, pdf_hi = norm.pdf(bot), norm.pdf(top)
    z_pdf_lo = bot * pdf_lo if np.isfinite(bot) else 0.0
    z_pdf_hi = top * pdf_hi if np.isfinite(top) else 0.0
    m1, m2 = pdf_lo - pdf_hi, mass + z_pdf_lo - z_pdf_hi
    x_lo, x_hi = _pooled_points(sigma, bot, top)
    for prb, x in ((ndtr(bot), x_lo), (ndtr(-top), x_hi)):
        if prb > 0.0:
            # The point's coordinate (ln x - mu) / sigma
            w = 0.5 * sigma + np.log(x) / sigma
            m1, m2 = m1 + prb * w, m2 + prb * w * w
    return float(m2 - m1 * m1)


class LogNormalPooledTails(Distribution):
    """
    A lognormal whose tails are pooled at their means. X = exp(mu + sigma Z) with
    Z a standard normal between the cutoffs lo = exp(mu + sigma bot) and
    hi = exp(mu + sigma top), and each tail beyond a cutoff is replaced by a
    single point, which has the tail's probability and sits at the tail's
    conditional mean:

        P(X = x_lo) = Phi(bot),      x_lo = E[X] Phi(bot - sigma) / Phi(bot),
        P(X = x_hi) = 1 - Phi(top),  x_hi = E[X] Phi(sigma - top) / Phi(-top),

    with E[X] = exp(mu + sigma**2 / 2), the lognormal's mean, which pooling keeps.
    So the law is LogNormal(mu, sigma)'s own between lo and hi, its points x_lo
    < lo and x_hi > hi are its infimum and supremum, and it is a mean-preserving
    contraction of LogNormal(mu, sigma): E[f(X)] is no larger for any convex f.
    Its lowest value has positive probability, infimum_prb.

    This is not censoring in the statistical sense, which puts each tail's
    probability at its cutoff and so changes the mean.

    The cutoffs are in standard deviations of the underlying normal, and None is
    no pooling on that side, where the lognormal's own tail is kept. With both
    None this is LogNormal(mu, sigma).

    Parameters
    ----------
    mu : float, optional
        Mean of the underlying normal, by default 0.0.
    sigma : float, optional
        Standard deviation of the underlying normal, by default 1.0. With
        sigma = 0 the distribution is the point exp(mu).
    bot : float or None, optional
        Lower cutoff, by default -2.0.
    top : float or None, optional
        Upper cutoff, by default 2.0.
    seed : int, optional
        Seed for random number generator, by default None.

    Attributes
    ----------
    infimum, supremum : np.array
        The pooled points x_lo and x_hi; 0 and inf on a side with no pooling.
    infimum_prb, supremum_prb : float
        The points' probabilities, Phi(bot) and 1 - Phi(top); zero on a side
        with no pooling.
    body : LogNormalTruncated
        The law between the cutoffs, given that X is there.
    body_prb : float
        The probability that X is between the cutoffs.
    """

    def __new__(cls, mu=0.0, sigma=1.0, bot=-2.0, top=2.0, seed=None):
        if sigma == 0:
            # A point at exp(mu), whose bounds are the point
            return DiscreteDistribution([1.0], [np.exp(mu)], seed=seed)
        return super().__new__(cls)

    def __init__(self, mu=0.0, sigma=1.0, bot=-2.0, top=2.0, seed=None):
        if np.ndim(mu) or np.ndim(sigma):
            raise ValueError(
                "mu and sigma must be scalars; use IndexDistribution for time variation."
            )
        if sigma < 0:
            raise ValueError("sigma must be nonnegative, but it is " + str(sigma))
        super().__init__(seed=seed)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.bot = _bound(bot, -np.inf)
        self.top = _bound(top, np.inf)
        if not self.bot < self.top:
            raise ValueError("The cutoffs must satisfy bot < top.")
        self.body = LogNormalTruncated(
            mu=self.mu, sigma=self.sigma, bot=self.bot, top=self.top, seed=0
        )
        self.body_prb = self.body.mass
        self.infimum_prb = float(ndtr(self.bot))
        self.supremum_prb = float(ndtr(-self.top))
        mean = np.exp(self.mu + 0.5 * self.sigma**2)
        x_lo, x_hi = _pooled_points(self.sigma, self.bot, self.top)
        lo = mean * x_lo if np.isfinite(self.bot) else 0.0
        hi = mean * x_hi if np.isfinite(self.top) else np.inf
        self.infimum = np.array([lo])
        self.supremum = np.array([hi])

    def z(self, x):
        """The standard normal coordinate (ln x - mu) / sigma of x."""
        with np.errstate(divide="ignore"):
            return (np.log(x) - self.mu) / self.sigma

    def breakpoints(self):
        """
        The cutoffs, where the quantile function jumps from a pooled point to
        the lognormal's own part, across the gap between them that holds no
        probability.

        Returns
        -------
        np.ndarray
            The points, in increasing order.
        """
        cuts = (self.bot, self.top)
        return np.array(
            [np.exp(self.mu + self.sigma * c) for c in cuts if np.isfinite(c)]
        )

    def _points(self):
        """The pooled points and their probabilities, on the sides that have one."""
        return [
            (prb, bound[0])
            for prb, bound in (
                (self.infimum_prb, self.infimum),
                (self.supremum_prb, self.supremum),
            )
            if prb > 0.0
        ]

    def pdf(self, x):
        """
        The density of the law's continuous part, LogNormal(mu, sigma)'s own
        between the cutoffs and zero elsewhere. The points' probabilities are
        infimum_prb and supremum_prb.
        """
        return self.body_prb * self.body.pdf(x)

    def cdf(self, x):
        """The cumulative distribution function, with a step at each point."""
        x = np.asarray(x, dtype=float)
        F = self.infimum_prb + self.body_prb * self.body.cdf(x)
        F = np.where(x < self.infimum[0], 0.0, F)
        return np.where(x >= self.supremum[0], 1.0, F)

    def ppf(self, q):
        """
        The quantile function, the least x with cdf(x) >= q: the lower point for
        q up to infimum_prb and the upper point for q above 1 - supremum_prb.
        """
        q = np.asarray(q, dtype=float)
        inside = np.clip((q - self.infimum_prb) / self.body_prb, 0.0, 1.0)
        x = self.body.ppf(inside)
        if self.infimum_prb > 0.0:
            x = np.where(q <= self.infimum_prb, self.infimum[0], x)
        if self.supremum_prb > 0.0:
            x = np.where(q > 1.0 - self.supremum_prb, self.supremum[0], x)
        return x

    def moment(self, order):
        """The raw moment E[X**order], for any real order, in closed form."""
        k = float(order)
        return float(
            self.body_prb * self.body.moment(k)
            + sum(prb * point**k for prb, point in self._points())
        )

    def mean(self):
        """The mean, exp(mu + sigma**2 / 2)."""
        return self.moment(1)

    def var(self):
        """The variance."""
        return self.moment(2) - self.mean() ** 2

    def std(self):
        """The standard deviation."""
        return np.sqrt(self.var())

    def expect(self, func=None):
        """
        The expectation of func under the law: by quadrature against the normal
        density between the cutoffs, plus the points. With no func, the mean.
        """
        if func is None:
            return self.mean()

        def integrand(z):
            # Zero where the normal density is, so that func is not evaluated out
            # where exp(mu + sigma z) would overflow
            w = norm.pdf(z)
            return 0.0 if w == 0.0 else func(np.exp(self.mu + self.sigma * z)) * w

        inside, _ = quad(
            integrand,
            self.bot,
            self.top,
            limit=200,
            epsabs=1e-14,
            epsrel=1e-12,
        )
        return float(inside + sum(prb * func(point) for prb, point in self._points()))

    def rvs(self, size=None, random_state=None):
        """Random draws, through the quantile function."""
        rng = self._rng if random_state is None else random_state
        return self.ppf(rng.uniform(size=size))

    def _approx_equiprobable(self, N, endpoints=False):
        """
        A discrete approximation that keeps the pooled points, with their
        probabilities, as atoms of their own, and puts N atoms of equal
        probability between the cutoffs, each the conditional mean of its bin,
        so the mean is exact and the lowest atom is the infimum whatever N is.

        Parameters
        ----------
        N : int
            Number of atoms between the cutoffs.
        endpoints : bool, optional
            Whether to add zero-probability atoms at the infimum and supremum
            on a side with no pooling, by default False. A pooled point is
            already its side's bound.

        Returns
        -------
        DiscreteDistribution
        """
        body = self.body._approx_equiprobable(N)
        atoms = np.ravel(body.atoms)
        pmv = self.body_prb * np.asarray(body.pmv)
        if self.infimum_prb > 0.0 or endpoints:
            atoms = np.r_[self.infimum, atoms]
            pmv = np.r_[self.infimum_prb, pmv]
        if self.supremum_prb > 0.0 or endpoints:
            atoms = np.r_[atoms, self.supremum]
            pmv = np.r_[pmv, self.supremum_prb]
        limit = {"dist": self, "method": "equiprobable", "N": N, "endpoints": endpoints}
        return DiscreteDistribution(pmv, atoms, seed=self.random_seed(), limit=limit)


class LogNormalPooledTailsMeanOne(LogNormalPooledTails):
    """
    A lognormal with tails pooled at their means, with mean one:
    mu = -sigma**2 / 2, since pooling keeps the lognormal's mean. The cutoffs are
    in standard deviations of the underlying normal, by default two either
    side.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the underlying normal, by default 1.0. With
        sigma = 0 the distribution is the point 1.
    bot : float or None, optional
        Lower cutoff, by default -2.0.
    top : float or None, optional
        Upper cutoff, by default 2.0.
    seed : int, optional
        Seed for random number generator, by default 0.
    """

    def __new__(cls, sigma=1.0, bot=-2.0, top=2.0, seed=0):
        if sigma == 0:
            return DiscreteDistribution([1.0], [1.0], seed=seed)
        return super().__new__(cls)

    def __init__(self, sigma=1.0, bot=-2.0, top=2.0, seed=0):
        super().__init__(mu=-0.5 * sigma**2, sigma=sigma, bot=bot, top=top, seed=seed)

    @classmethod
    def from_sd_log(cls, sd_log, bot=-2.0, top=2.0, seed=0):
        """
        The mean-one law with pooled tails whose log has standard deviation
        sd_log. Pooling changes the log's dispersion slightly, so the underlying
        normal's sigma is set so that an estimated dispersion keeps its value.

        Parameters
        ----------
        sd_log : float
            Standard deviation of ln X.
        bot : float or None, optional
            Lower cutoff, by default -2.0.
        top : float or None, optional
            Upper cutoff, by default 2.0.
        seed : int, optional
            Seed for random number generator, by default 0.

        Returns
        -------
        LogNormalPooledTailsMeanOne or DiscreteDistribution
        """
        if sd_log == 0:
            return cls(sigma=0.0, bot=bot, top=top, seed=seed)
        lo, hi = _bound(bot, -np.inf), _bound(top, np.inf)
        sigma = float(sd_log)
        for _ in range(50):
            new = sd_log / np.sqrt(_pooled_log_variance(sigma, lo, hi))
            if abs(new - sigma) <= 1e-15 * sigma:
                break
            sigma = new
        return cls(sigma=sigma, bot=bot, top=top, seed=seed)


# The names these classes had before: they still work, with a DeprecationWarning
Lognormal = _renamed_class(LogNormal, "Lognormal")
MeanOneLogNormal = _renamed_class(LogNormalMeanOne, "MeanOneLogNormal")
TruncatedLognormal = _renamed_class(LogNormalTruncated, "TruncatedLognormal")
TruncatedLogNormal = _renamed_class(LogNormalTruncated, "TruncatedLogNormal")
MeanOneTruncatedLogNormal = _renamed_class(
    LogNormalTruncatedMeanOne, "MeanOneTruncatedLogNormal"
)


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
