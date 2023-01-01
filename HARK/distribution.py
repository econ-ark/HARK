import math
from itertools import product
from typing import Optional, Union
from warnings import warn

import numpy as np
import xarray as xr
from numpy import random
from scipy import stats
from scipy.stats._distn_infrastructure import rv_continuous_frozen


class Distribution:
    """
    Base class for all probability distributions
    with seed and random number generator.

    For discussion on random number generation and random seeds, see
    https://docs.scipy.org/doc/scipy/tutorial/stats.html#random-number-generation

    Parameters
    ----------
    seed : Optional[int]
        Seed for random number generator.
    """

    def __init__(self, seed: Optional[int] = 0):
        """
        Generic distribution class with seed management.

        Parameters
        ----------
        seed : Optional[int], optional
            Seed for random number generator, by default None
            generates random seed based on entropy.

        Raises
        ------
        ValueError
            Seed must be an integer type.
        """
        if seed is None:
            # generate random seed
            _seed = random.SeedSequence().entropy
        elif isinstance(seed, (int, np.integer)):
            _seed = seed
        else:
            raise ValueError("seed must be an integer")

        self._seed = _seed
        self._rng = random.default_rng(self._seed)

    @property
    def seed(self) -> int:
        """
        Seed for random number generator.

        Returns
        -------
        int
            Seed.
        """
        return self._seed  # type: ignore

    @seed.setter
    def seed(self, seed):
        """
        Set seed for random number generator.

        Parameters
        ----------
        seed : int
            Seed for random number generator.
        """

        if isinstance(seed, (int, np.integer)):
            self._seed = seed
            self._rng = random.default_rng(seed)
        else:
            raise ValueError("seed must be an integer")

    def reset(self):
        """
        Reset the random number generator of this distribution.
        Resetting the seed will result in the same sequence of
        random numbers being generated.

        Parameters
        ----------
        """
        self._rng = random.default_rng(self.seed)

    def random_seed(self):
        """
        Generate a new random seed for this distribution.
        """
        self.seed = random.SeedSequence().entropy

    def draw(self, N):
        """
        Generate arrays of draws from this distribution.
        If input N is a number, output is a length N array of draws from the
        distribution. If N is a list, output is a length T list whose
        t-th entry is a length N array of draws from the distribution[t].

        Parameters
        ----------
        N : int
            Number of draws in each row.

        Returns:
        ------------
        draws : np.array or [np.array]
            T-length list of arrays of random variable draws each of size N, or
            a single array of size N (if sigma is a scalar).
        """

        mean = self.mean() if callable(self.mean) else self.mean
        n = (N, mean.size) if mean.size != 1 else N
        return self.rvs(size=n, random_state=self._rng)


### CONTINUOUS DISTRIBUTIONS


class ContinuousFrozenDistribution(rv_continuous_frozen, Distribution):
    """
    Parametrized continuous distribution from scipy.stats with seed management.
    """

    def __init__(self, dist, *args, seed=0, **kwds):
        """
        Parametrized continuous distribution from scipy.stats with seed management.

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

    def approx(self, N):
        """
        Returns a discrete approximation of this distribution.


        """
        x, w = np.polynomial.hermite.hermgauss(N)
        # normalize w
        pmv = w * np.pi**-0.5
        # correct x
        atoms = math.sqrt(2.0) * self.sigma * x + self.mu
        return DiscreteDistribution(
            pmv, atoms, seed=self._rng.integers(0, 2**31 - 1, dtype="int32")
        )

    def approx_equiprobable(self, N):
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
        lims = stats.norm.ppf(CDF)
        pdf = stats.norm.pdf(lims)

        # Find conditional means using Mills's ratio
        pmv = np.diff(CDF)
        atoms = self.mu - np.diff(pdf) / pmv * self.sigma

        return DiscreteDistribution(
            pmv, atoms, seed=self._rng.integers(0, 2**31 - 1, dtype="int32")
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

        Returns
        -------
        _type_
            _description_
        """

        if sigma == 0:
            # If sigma is zero, return a DiscreteDistribution with a single atom
            return DiscreteDistribution([1.0], [np.exp(mu)], seed=seed)

        return super(Lognormal, cls).__new__(cls)

    def __init__(
        self,
        mu: Union[float, np.ndarray] = 0.0,
        sigma: Union[float, np.ndarray] = 1.0,
        seed: Optional[int] = 0,
    ):
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

    def approx(self, N, tail_N=0, tail_bound=None, tail_order=np.e):
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
                    lower_CDF_vals.append(
                        lower_CDF_vals[-1] + lo_cut * scale**x / mag
                    )
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
        return DiscreteDistribution(
            pmv, atoms, seed=self._rng.integers(0, 2**31 - 1, dtype="int32")
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

    def approx(self, N, endpoint=False):
        """
        Makes a discrete approximation to this uniform distribution.

        Parameters
        ----------
        N : int
            The number of points in the discrete approximation.
        endpoint : bool
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

        if endpoint:  # insert endpoints with infinitesimally small mass
            atoms = np.concatenate(([self.bot], atoms, [self.top]))
            pmv = np.concatenate(([0.0], pmv, [0.0]))

        return DiscreteDistribution(
            pmv, atoms, seed=self._rng.integers(0, 2**31 - 1, dtype="int32")
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


### MULTIVARIATE DISTRIBUTIONS


class MVNormal(Distribution):
    """
    A Multivariate Normal distribution.

    Parameters
    ----------
    mu : numpy array
        Mean vector.
    Sigma : 2-d numpy array. Each dimension must have length equal to that of
            mu.
        Variance-covariance matrix.
    seed : int
        Seed for random number generator.
    """

    mu = None
    Sigma = None

    def __init__(self, mu=np.array([1, 1]), Sigma=np.array([[1, 0], [0, 1]]), seed=0):
        self.mu = mu
        self.Sigma = Sigma
        self.M = len(self.mu)
        super().__init__(seed)

    def draw(self, N):
        """
        Generate an array of multivariate normal draws.

        Parameters
        ----------
        N : int
            Number of multivariate draws.

        Returns
        -------
        draws : np.array
            Array of dimensions N x M containing the random draws, where M is
            the dimension of the multivariate normal and N is the number of
            draws. Each row represents a draw.
        """
        draws = self.RNG.multivariate_normal(self.mu, self.Sigma, N)

        return draws

    def approx(self, N, equiprobable=False):
        """
        Returns a discrete approximation of this distribution.

        The discretization will have N**M points, where M is the dimension of
        the multivariate normal.

        It uses the fact that:
            - Being positive definite, Sigma can be factorized as Sigma = QVQ',
              with V diagonal. So, letting A=Q*sqrt(V), Sigma = A*A'.
            - If Z is an N-dimensional multivariate standard normal, then
              A*Z ~ N(0,A*A' = Sigma).

        The idea therefore is to construct an equiprobable grid for a standard
        normal and multiply it by matrix A.
        """

        # Start by computing matrix A.
        v, Q = np.linalg.eig(self.Sigma)
        sqrtV = np.diag(np.sqrt(v))
        A = np.matmul(Q, sqrtV)

        # Now find a discretization for a univariate standard normal.
        if equiprobable:
            z_approx = Normal().approx_equiprobable(N)
        else:
            z_approx = Normal().approx(N)

        # Now create the multivariate grid and pmv
        Z = np.array(list(product(*[z_approx.atoms.flatten()] * self.M)))
        pmv = np.prod(np.array(list(product(*[z_approx.pmv] * self.M))), axis=1)

        # Apply mean and standard deviation to the Z grid
        atoms = self.mu[None, ...] + np.matmul(Z, A.T)

        # Construct and return discrete distribution
        return DiscreteDistribution(
            pmv, atoms.T, seed=self.RNG.integers(0, 2**31 - 1, dtype="int32")
        )


### DISCRETE DISTRIBUTIONS


class Bernoulli(Distribution):
    """
    A Bernoulli distribution.

    Parameters
    ----------
    p : float or [float]
        Probability or probabilities of the event occurring (True).

    seed : int
        Seed for random number generator.
    """

    p = None

    def __init__(self, p=0.5, seed=0):
        self.p = np.array(p)
        # Set up the RNG
        super().__init__(seed)

    def draw(self, N):
        """
        Generates arrays of booleans drawn from a simple Bernoulli distribution.
        The input p can be a float or a list-like of floats; its length T determines
        the number of entries in the output.  The t-th entry of the output is an
        array of N booleans which are True with probability p[t] and False otherwise.

        Arguments
        ---------
        N : int
            Number of draws in each row.

        Returns
        -------
        draws : np.array or [np.array]
            T-length list of arrays of Bernoulli draws each of size N, or a single
        array of size N (if sigma is a scalar).
        """
        draws = []
        for j in range(self.p.size):
            draws.append(self.RNG.uniform(size=N) < self.p.item(j))
        return draws[0] if len(draws) == 1 else draws


class DiscreteDistribution(Distribution):
    """
    A representation of a discrete probability distribution.

    Parameters
    ----------
    pmv : np.array
        An array of floats representing a probability mass function.
    atoms : np.array
        Discrete point values for each probability mass.
        For multivariate distributions, the last dimension of atoms must index
        "atom" or the random realization. For instance, if atoms.shape == (2,6,4),
        the random variable has 4 possible realizations and each of them has shape (2,6).
    seed : int
        Seed for random number generator.
    """

    pmv = None
    atoms = None

    def __init__(self, pmv, atoms, seed=0):

        self.pmv = pmv

        if len(atoms.shape) < 2:
            self.atoms = atoms[None, ...]
        else:
            self.atoms = atoms

        # Set up the RNG
        super().__init__(seed)

        # Check that pmv and atoms have compatible dimensions.
        same_dims = len(pmv) == atoms.shape[-1]
        if not same_dims:
            raise ValueError(
                "Provided pmv and atoms arrays have incompatible dimensions. "
                + "The length of the pmv must be equal to that of atoms's last dimension."
            )

    def dim(self):
        """
        Last dimension of self.atoms indexes "atom."
        """
        return self.atoms.shape[:-1]

    def draw_events(self, n):
        """
        Draws N 'events' from the distribution PMF.
        These events are indices into atoms.
        """
        # Generate a cumulative distribution
        base_draws = self.RNG.uniform(size=n)
        cum_dist = np.cumsum(self.pmv)

        # Convert the basic uniform draws into discrete draws
        indices = cum_dist.searchsorted(base_draws)

        return indices

    def draw(self, N, atoms=None, exact_match=False):
        """
        Simulates N draws from a discrete distribution with probabilities P and outcomes atoms.

        Parameters
        ----------
        N : int
            Number of draws to simulate.
        atoms : None, int, or np.array
            If None, then use this distribution's atoms for point values.
            If an int, then the index of atoms for the point values.
            If an np.array, use the array for the point values.
        exact_match : boolean
            Whether the draws should "exactly" match the discrete distribution (as
            closely as possible given finite draws).  When True, returned draws are
            a random permutation of the N-length list that best fits the discrete
            distribution.  When False (default), each draw is independent from the
            others and the result could deviate from the input.

        Returns
        -------
        draws : np.array
            An array of draws from the discrete distribution; each element is a value in atoms.
        """
        if atoms is None:
            atoms = self.atoms
        elif isinstance(atoms, int):
            atoms = self.atoms[atoms]

        if exact_match:
            events = np.arange(self.pmv.size)  # just a list of integers
            cutoffs = np.round(np.cumsum(self.pmv) * N).astype(
                int
            )  # cutoff points between discrete outcomes
            top = 0

            # Make a list of event indices that closely matches the discrete distribution
            event_list = []
            for j in range(events.size):
                bot = top
                top = cutoffs[j]
                event_list += (top - bot) * [events[j]]

            # Randomly permute the event indices
            indices = self.RNG.permutation(event_list)

        # Draw event indices randomly from the discrete distribution
        else:
            indices = self.draw_events(N)

        # Create and fill in the output array of draws based on the output of event indices
        draws = atoms[..., indices]

        # TODO: some models expect univariate draws to just be a 1d vector. Fix those models.
        if len(draws.shape) == 2 and draws.shape[0] == 1:
            draws = draws.flatten()

        return draws

    def expected(self, func=None, *args):
        """
        Expected value of a function, given an array of configurations of its
        inputs along with a DiscreteDistribution object that specifies the
        probability of each configuration.

        Parameters
        ----------
        func : function
            The function to be evaluated.
            This function should take the full array of distribution values
            and return either arrays of arbitrary shape or scalars.
            It may also take other arguments *args.
            This function differs from the standalone `calc_expectation`
            method in that it uses numpy's vectorization and broadcasting
            rules to avoid costly iteration.
            Note: If you need to use a function that acts on single outcomes
            of the distribution, consier `distribution.calc_expectation`.
        *args :
            Other inputs for func, representing the non-stochastic arguments.
            The the expectation is computed at f(dstn, *args).

        Returns
        -------
        f_exp : np.array or scalar
            The expectation of the function at the queried values.
            Scalar if only one value.
        """

        if func is None:
            # if no function is provided, it's much faster to go straight
            # to dot product instead of calling the dummy function.
            f_query = self.atoms
        else:
            # if a function is provided, we need to add one more dimension,
            # the atom dimension, to any inputs that are n-dim arrays.
            # This allows numpy to easily broadcast the function's output.
            # For more information on broadcasting, see:
            # https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
            args = [
                arg[..., np.newaxis] if isinstance(arg, np.ndarray) else arg
                for arg in args
            ]

            f_query = func(self.atoms, *args)

        f_exp = np.dot(f_query, self.pmv)

        return f_exp

    def dist_of_func(self, func=lambda x: x, *args):
        """
        Finds the distribution of a random variable Y that is a function
        of discrete random variable atoms, Y=f(atoms).

        Parameters
        ----------
        func : function
            The function to be evaluated.
            This function should take the full array of distribution values.
            It may also take other arguments *args.
        *args :
            Additional non-stochastic arguments for func,
            The function is computed as f(dstn, *args).

        Returns
        -------
        f_dstn : DiscreteDistribution
            The distribution of func(dstn).
        """
        # we need to add one more dimension,
        # the atom dimension, to any inputs that are n-dim arrays.
        # This allows numpy to easily broadcast the function's output.
        args = [
            arg[..., np.newaxis] if isinstance(arg, np.ndarray) else arg for arg in args
        ]
        f_query = func(self.atoms, *args)

        f_dstn = DiscreteDistribution(list(self.pmv), f_query, seed=self.seed)

        return f_dstn


class DiscreteDistributionLabeled(DiscreteDistribution):
    """
    A representation of a discrete probability distribution
    stored in an underlying `xarray.Dataset`.

    Parameters
    ----------
    pmv : np.array
        An array of values representing a probability mass function.
    data : np.array
        Discrete point values for each probability mass.
        For multivariate distributions, the last dimension of atoms must index
        "atom" or the random realization. For instance, if atoms.shape == (2,6,4),
        the random variable has 4 possible realizations and each of them has shape (2,6).
    seed : int
        Seed for random number generator.
    name : str
        Name of the distribution.
    attrs : dict
        Attributes for the distribution.
    var_names : list of str
        Names of the variables in the distribution.
    var_attrs : list of dict
        Attributes of the variables in the distribution.

    """

    def __init__(
        self,
        pmv,
        data,
        seed=0,
        name="DiscreteDistributionLabeled",
        attrs=None,
        var_names=None,
        var_attrs=None,
    ):

        # vector-value distributions
        if data.ndim < 2:
            data = data[np.newaxis, ...]
        if data.ndim > 2:
            raise NotImplementedError(
                "Only vector-valued distributions are supported for now."
            )

        if attrs is None:
            attrs = {}

        attrs["name"] = name
        attrs["seed"] = seed
        attrs["RNG"] = np.random.default_rng(seed)

        n_var = data.shape[0]

        # give dummy names to variables if none are provided
        if var_names is None:
            var_names = ["var_" + str(i) for i in range(n_var)]

        assert (
            len(var_names) == n_var
        ), "Number of variable names does not match number of variables."

        # give dummy attributes to variables if none are provided
        if var_attrs is None:
            var_attrs = [None] * n_var

        # a DiscreteDistributionLabeled is an xr.Dataset where the only
        # dimension is "atom", which indexes the random realizations.
        self.dataset = xr.Dataset(
            {
                var_names[i]: xr.DataArray(
                    data[i],
                    dims=("atom"),
                    attrs=var_attrs[i],
                )
                for i in range(n_var)
            },
            attrs=attrs,
        )

        # the probability mass values are stored in
        # a DataArray with dimension "atom"
        self.pmf = xr.DataArray(pmv, dims=("atom"))

    @classmethod
    def from_unlabeled(
        cls,
        dist,
        name="DiscreteDistributionLabeled",
        attrs=None,
        var_names=None,
        var_attrs=None,
    ):

        ldd = cls(
            dist.pmv,
            dist.atoms,
            seed=dist.seed,
            name=name,
            attrs=attrs,
            var_names=var_names,
            var_attrs=var_attrs,
        )

        return ldd

    @classmethod
    def from_dataset(cls, x_obj, pmf):

        ldd = cls.__new__(cls)

        if isinstance(x_obj, xr.Dataset):
            ldd.dataset = x_obj
        elif isinstance(x_obj, xr.DataArray):
            ldd.dataset = xr.Dataset({x_obj.name: x_obj})
        elif isinstance(x_obj, dict):
            ldd.dataset = xr.Dataset(x_obj)

        ldd.pmf = pmf

        return ldd

    @property
    def _weighted(self):
        """
        Returns a DatasetWeighted object for the distribution.
        """
        return self.dataset.weighted(self.pmf)

    @property
    def variables(self):
        """
        A dict-like container of DataArrays corresponding to
        the variables of the distribution.
        """
        return self.dataset.data_vars

    @property
    def atoms(self):
        """
        Returns the distribution's data as a numpy.ndarray.
        """
        data_vars = self.variables
        return np.vstack([data_vars[key].values for key in data_vars.keys()])

    @property
    def pmv(self):
        """
        Returns the distribution's probability mass function.
        """
        return self.pmf.values

    @property
    def seed(self):
        """
        Returns the distribution's seed.
        """
        return self.dataset.seed

    @property
    def RNG(self):
        """
        Returns the distribution's random number generator.
        """
        return self.dataset.RNG

    @property
    def name(self):
        """
        The distribution's name.
        """
        return self.dataset.name

    @property
    def attrs(self):
        """
        The distribution's attributes.
        """
        return self.dataset.attrs

    def dist_of_func(self, func=lambda x: x, *args, **kwargs):
        """
        Finds the distribution of a random variable Y that is a function
        of discrete random variable atoms, Y=f(atoms).

        Parameters
        ----------
        func : function
            The function to be evaluated.
            This function should take the full array of distribution values.
            It may also take other arguments *args.
        *args :
            Additional non-stochastic arguments for func,
            The function is computed as f(dstn, *args).
        **kwargs :
            Additional keyword arguments for func. Must be xarray compatible
            in order to work with xarray broadcasting.

        Returns
        -------
        f_dstn : DiscreteDistribution or DiscreteDistributionLabeled
            The distribution of func(dstn).
        """

        def func_wrapper(x, *args):
            """
            Wrapper function for `func` that handles labeled indexing.
            """

            idx = self.variables.keys()
            wrapped = dict(zip(idx, x))

            return func(wrapped, *args)

        if len(kwargs):
            f_query = func(self.dataset, **kwargs)
            ldd = DiscreteDistributionLabeled.from_dataset(f_query, self.pmv)

            return ldd
        else:
            return super().dist_of_func(func_wrapper, *args)

    def expected(self, func=None, *args, **kwargs):
        """
        Expectation of a function, given an array of configurations of its inputs
        along with a DiscreteDistributionLabeled object that specifies the probability
        of each configuration.

        Parameters
        ----------
        func : function
            The function to be evaluated.
            This function should take the full array of distribution values
            and return either arrays of arbitrary shape or scalars.
            It may also take other arguments *args.
            This function differs from the standalone `calc_expectation`
            method in that it uses numpy's vectorization and broadcasting
            rules to avoid costly iteration.
            Note: If you need to use a function that acts on single outcomes
            of the distribution, consier `distribution.calc_expectation`.
        *args :
            Other inputs for func, representing the non-stochastic arguments.
            The the expectation is computed at f(dstn, *args).
        labels : bool
            If True, the function should use labeled indexing instead of integer
            indexing using the distribution's underlying rv coordinates. For example,
            if `dims = ('rv', 'x')` and `coords = {'rv': ['a', 'b'], }`, then
            the function can be `lambda x: x["a"] + x["b"]`.

        Returns
        -------
        f_exp : np.array or scalar
            The expectation of the function at the queried values.
            Scalar if only one value.
        """

        def func_wrapper(x, *args):
            """
            Wrapper function for `func` that handles labeled indexing.
            """

            idx = self.variables.keys()
            wrapped = dict(zip(idx, x))

            return func(wrapped, *args)

        if len(kwargs):
            f_query = func(self.dataset, *args, **kwargs)
            ldd = DiscreteDistributionLabeled.from_dataset(f_query, self.pmf)

            return ldd._weighted.mean("atom")
        else:
            if func is None:
                return super().expected()
            else:
                return super().expected(func_wrapper, *args)


class IndexDistribution(Distribution):
    """
    This class provides a way to define a distribution that
    is conditional on an index.

    The current implementation combines a defined distribution
    class (such as Bernoulli, LogNormal, etc.) with information
    about the conditions on the parameters of the distribution.

    For example, an IndexDistribution can be defined as
    a Bernoulli distribution whose parameter p is a function of
    a different inpute parameter.

    Parameters
    ----------

    engine : Distribution class
        A Distribution subclass.

    conditional: dict
        Information about the conditional variation
        on the input parameters of the engine distribution.
        Keys should match the arguments to the engine class
        constructor.

    seed : int
        Seed for random number generator.
    """

    conditional = None
    engine = None

    def __init__(self, engine, conditional, RNG=None, seed=0):

        if RNG is None:
            # Set up the RNG
            super().__init__(seed)
        else:
            # If an RNG is received, use it in whatever state it is in.
            self.RNG = RNG
            # The seed will still be set, even if it is not used for the RNG,
            # for whenever self.reset() is called.
            # Note that self.reset() will stop using the RNG that was passed
            # and create a new one.
            self.seed = seed

        self.conditional = conditional
        self.engine = engine

        self.dstns = []

        # Test one item to determine case handling
        item0 = list(self.conditional.values())[0]

        if type(item0) is list:
            # Create and store all the conditional distributions
            for y in range(len(item0)):
                cond = {key: val[y] for (key, val) in self.conditional.items()}
                self.dstns.append(
                    self.engine(seed=self.RNG.integers(0, 2**31 - 1), **cond)
                )

        elif type(item0) is float:

            self.dstns = [
                self.engine(seed=self.RNG.integers(0, 2**31 - 1), **conditional)
            ]

        else:
            raise (
                Exception(
                    f"IndexDistribution: Unhandled case for __getitem__ access. item0: {item0}; conditional: {self.conditional}"
                )
            )

    def __getitem__(self, y):

        return self.dstns[y]

    def approx(self, N, **kwds):
        """
        Approximation of the distribution.

        Parameters
        ----------
        N : init
            Number of discrete points to approximate
            continuous distribution into.

        kwds: dict
            Other keyword arguments passed to engine
            distribution approx() method.

        Returns:
        ------------
        dists : [DiscreteDistribution]
            A list of DiscreteDistributions that are the
            approximation of engine distribution under each condition.

            TODO: It would be better if there were a conditional discrete
            distribution representation. But that integrates with the
            solution code. This implementation will return the list of
            distributions representations expected by the solution code.
        """

        # test one item to determine case handling
        item0 = list(self.conditional.values())[0]

        if type(item0) is float:
            # degenerate case. Treat the parameterization as constant.
            return self.dstns[0].approx(N, **kwds)

        if type(item0) is list:
            return TimeVaryingDiscreteDistribution(
                [self[i].approx(N, **kwds) for i, _ in enumerate(item0)]
            )

    def draw(self, condition):
        """
        Generate arrays of draws.
        The input is an array containing the conditions.
        The output is an array of the same length (axis 1 dimension)
        as the conditions containing random draws of the conditional
        distribution.

        Parameters
        ----------
        condition : np.array
            The input conditions to the distribution.

        Returns:
        ------------
        draws : np.array
        """
        # for now, assume that all the conditionals
        # are of the same type.
        # this matches the HARK 'time-varying' model architecture.

        # test one item to determine case handling
        item0 = list(self.conditional.values())[0]

        if type(item0) is float:
            # degenerate case. Treat the parameterization as constant.
            N = condition.size

            return self.engine(
                seed=self.RNG.integers(0, 2**31 - 1), **self.conditional
            ).draw(N)

        if type(item0) is list:
            # conditions are indices into list
            # somewhat convoluted sampling strategy retained
            # for test backwards compatibility

            draws = np.zeros(condition.size)

            for c in np.unique(condition):
                these = c == condition
                N = np.sum(these)

                cond = {key: val[c] for (key, val) in self.conditional.items()}
                draws[these] = self[c].draw(N)

            return draws


class TimeVaryingDiscreteDistribution(Distribution):
    """
    This class provides a way to define a discrete distribution that
    is conditional on an index.

    Wraps a list of discrete distributions.

    Parameters
    ----------

    distributions : [DiscreteDistribution]
        A list of discrete distributions

    seed : int
        Seed for random number generator.
    """

    distributions = []

    def __init__(self, distributions, seed=0):
        # Set up the RNG
        super().__init__(seed)

        self.distributions = distributions

    def __getitem__(self, y):
        return self.distributions[y]

    def draw(self, condition):
        """
        Generate arrays of draws.
        The input is an array containing the conditions.
        The output is an array of the same length (axis 1 dimension)
        as the conditions containing random draws of the conditional
        distribution.

        Parameters
        ----------
        condition : np.array
            The input conditions to the distribution.

        Returns:
        ------------
        draws : np.array
        """
        # for now, assume that all the conditionals
        # are of the same type.
        # this matches the HARK 'time-varying' model architecture.

        # conditions are indices into list
        # somewhat convoluted sampling strategy retained
        # for test backwards compatibility

        draws = np.zeros(condition.size)

        for c in np.unique(condition):
            these = c == condition
            N = np.sum(these)

            draws[these] = self.distributions[c].draw(N)

        return draws


def approx_lognormal_gauss_hermite(N, mu=0.0, sigma=1.0, seed=0):
    d = Normal(mu, sigma).approx(N)
    return DiscreteDistribution(d.pmv, np.exp(d.atoms), seed=seed)


def calc_normal_style_pars_from_lognormal_pars(avg_lognormal, std_lognormal):
    varLognormal = std_lognormal**2
    varNormal = math.log(1 + varLognormal / avg_lognormal**2)
    avgNormal = math.log(avg_lognormal) - varNormal * 0.5
    std_normal = math.sqrt(varNormal)
    return avgNormal, std_normal


def calc_lognormal_style_pars_from_normal_pars(mu_normal, std_normal):
    varNormal = std_normal**2
    avg_lognormal = math.exp(mu_normal + varNormal * 0.5)
    varLognormal = (math.exp(varNormal) - 1) * avg_lognormal**2
    std_lognormal = math.sqrt(varLognormal)
    return avg_lognormal, std_lognormal


def approx_beta(N, a=1.0, b=1.0):
    """
    Calculate a discrete approximation to the beta distribution.  May be quite
    slow, as it uses a rudimentary numeric integration method to generate the
    discrete approximation.

    Parameters
    ----------
    N : int
        Size of discrete space vector to be returned.
    a : float
        First shape parameter (sometimes called alpha).
    b : float
        Second shape parameter (sometimes called beta).

    Returns
    -------
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    """
    P = 1000
    vals = np.reshape(stats.beta.ppf(np.linspace(0.0, 1.0, N * P), a, b), (N, P))
    atoms = np.mean(vals, axis=1)
    pmv = np.ones(N) / float(N)
    return DiscreteDistribution(pmv, atoms)


def make_markov_approx_to_normal(x_grid, mu, sigma, K=351, bound=3.5):
    """
    Creates an approximation to a normal distribution with mean mu and standard
    deviation sigma, returning a stochastic vector called p_vec, corresponding
    to values in x_grid.  If a RV is distributed x~N(mu,sigma), then the expectation
    of a continuous function f() is E[f(x)] = numpy.dot(p_vec,f(x_grid)).

    Parameters
    ----------
    x_grid: numpy.array
        A sorted 1D array of floats representing discrete values that a normally
        distributed RV could take on.
    mu: float
        Mean of the normal distribution to be approximated.
    sigma: float
        Standard deviation of the normal distribution to be approximated.
    K: int
        Number of points in the normal distribution to sample.
    bound: float
        Truncation bound of the normal distribution, as +/- bound*sigma.

    Returns
    -------
    p_vec: numpy.array
        A stochastic vector with probability weights for each x in x_grid.
    """
    x_n = x_grid.size  # Number of points in the outcome grid
    lower_bound = -bound  # Lower bound of normal draws to consider, in SD
    upper_bound = bound  # Upper bound of normal draws to consider, in SD
    raw_sample = np.linspace(
        lower_bound, upper_bound, K
    )  # Evenly spaced draws between bounds
    f_weights = stats.norm.pdf(raw_sample)  # Relative probability of each draw
    sample = mu + sigma * raw_sample  # Adjusted bounds, given mean and stdev
    w_vec = np.zeros(x_n)  # A vector of outcome weights

    # Find the relative position of each of the draws
    sample_pos = np.searchsorted(x_grid, sample)
    sample_pos[sample_pos < 1] = 1
    sample_pos[sample_pos > x_n - 1] = x_n - 1

    # Make arrays of the x_grid point directly above and below each draw
    bot = x_grid[sample_pos - 1]
    top = x_grid[sample_pos]
    alpha = (sample - bot) / (top - bot)

    # Keep the weights (alpha) in bounds
    alpha_clipped = np.clip(alpha, 0.0, 1.0)

    # Loop through each x_grid point and add up the probability that each nearby
    # draw contributes to it (accounting for distance)
    for j in range(1, x_n):
        c = sample_pos == j
        w_vec[j - 1] = w_vec[j - 1] + np.dot(f_weights[c], 1.0 - alpha_clipped[c])
        w_vec[j] = w_vec[j] + np.dot(f_weights[c], alpha_clipped[c])

    # Reweight the probabilities so they sum to 1
    W = np.sum(w_vec)
    p_vec = w_vec / W

    # Check for obvious errors, and return p_vec
    assert (
        (np.all(p_vec >= 0.0))
        and (np.all(p_vec <= 1.0))
        and (np.isclose(np.sum(p_vec), 1.0))
    )
    return p_vec


def make_markov_approx_to_normal_by_monte_carlo(x_grid, mu, sigma, N_draws=10000):
    """
    Creates an approximation to a normal distribution with mean mu and standard
    deviation sigma, by Monte Carlo.
    Returns a stochastic vector called p_vec, corresponding
    to values in x_grid.  If a RV is distributed x~N(mu,sigma), then the expectation
    of a continuous function f() is E[f(x)] = numpy.dot(p_vec,f(x_grid)).

    Parameters
    ----------
    x_grid: numpy.array
        A sorted 1D array of floats representing discrete values that a normally
        distributed RV could take on.
    mu: float
        Mean of the normal distribution to be approximated.
    sigma: float
        Standard deviation of the normal distribution to be approximated.
    N_draws: int
        Number of draws to use in Monte Carlo.

    Returns
    -------
    p_vec: numpy.array
        A stochastic vector with probability weights for each x in x_grid.
    """

    # Take random draws from the desired normal distribution
    random_draws = np.random.normal(loc=mu, scale=sigma, size=N_draws)

    # Compute the distance between the draws and points in x_grid
    distance = np.abs(x_grid[:, np.newaxis] - random_draws[np.newaxis, :])

    # Find the indices of the points in x_grid that are closest to the draws
    distance_minimizing_index = np.argmin(distance, axis=0)

    # For each point in x_grid, the approximate probability of that point is the number
    # of Monte Carlo draws that are closest to that point
    p_vec = np.zeros_like(x_grid)
    for p_index, p in enumerate(p_vec):
        p_vec[p_index] = np.sum(distance_minimizing_index == p_index) / N_draws

    # Check for obvious errors, and return p_vec
    assert (
        (np.all(p_vec >= 0.0))
        and (np.all(p_vec <= 1.0))
        and (np.isclose(np.sum(p_vec)), 1.0)
    )
    return p_vec


def make_tauchen_ar1(N, sigma=1.0, ar_1=0.9, bound=3.0):
    """
    Function to return a discretized version of an AR1 process.
    See http://www.fperri.net/TEACHING/macrotheory08/numerical.pdf for details

    Parameters
    ----------
    N: int
        Size of discretized grid
    sigma: float
        Standard deviation of the error term
    ar_1: float
        AR1 coefficient
    bound: float
        The highest (lowest) grid point will be bound (-bound) multiplied by the unconditional
        standard deviation of the process

    Returns
    -------
    y: np.array
        Grid points on which the discretized process takes values
    trans_matrix: np.array
        Markov transition array for the discretized process
    """
    yN = bound * sigma / ((1 - ar_1**2) ** 0.5)
    y = np.linspace(-yN, yN, N)
    d = y[1] - y[0]
    trans_matrix = np.ones((N, N))
    for j in range(N):
        for k_1 in range(N - 2):
            k = k_1 + 1
            trans_matrix[j, k] = stats.norm.cdf(
                (y[k] + d / 2.0 - ar_1 * y[j]) / sigma
            ) - stats.norm.cdf((y[k] - d / 2.0 - ar_1 * y[j]) / sigma)
        trans_matrix[j, 0] = stats.norm.cdf((y[0] + d / 2.0 - ar_1 * y[j]) / sigma)
        trans_matrix[j, N - 1] = 1.0 - stats.norm.cdf(
            (y[N - 1] - d / 2.0 - ar_1 * y[j]) / sigma
        )

    return y, trans_matrix


# ================================================================================
# ==================== Functions for manipulating discrete distributions =========
# ================================================================================


def add_discrete_outcome_constant_mean(distribution, x, p, sort=False):
    """
    Adds a discrete outcome of x with probability p to an existing distribution,
    holding constant the relative probabilities of other outcomes and overall mean.

    Parameters
    ----------
    distribution : DiscreteDistribution
        A one-dimensional DiscreteDistribution.
    x : float
        The new value to be added to the distribution.
    p : float
        The probability of the discrete outcome x occuring.
    sort: bool
        Whether or not to sort atoms before returning it

    Returns
    -------
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    """

    if type(distribution) != TimeVaryingDiscreteDistribution:
        atoms = np.append(x, distribution.atoms * (1 - p * x) / (1 - p))
        pmv = np.append(p, distribution.pmv * (1 - p))

        if sort:
            indices = np.argsort(atoms)
            atoms = atoms[indices]
            pmv = pmv[indices]

        return DiscreteDistribution(pmv, atoms)
    elif type(distribution) == TimeVaryingDiscreteDistribution:
        # apply recursively on all the internal distributions
        return TimeVaryingDiscreteDistribution(
            [
                add_discrete_outcome_constant_mean(d, x, p)
                for d in distribution.distributions
            ],
            seed=distribution.seed,
        )


def add_discrete_outcome(distribution, x, p, sort=False):
    """
    Adds a discrete outcome of x with probability p to an existing distribution,
    holding constant the relative probabilities of other outcomes.

    Parameters
    ----------
    distribution : DiscreteDistribution
        One-dimensional distribution to which the outcome is to be added.
    x : float
        The new value to be added to the distribution.
    p : float
        The probability of the discrete outcome x occuring.

    Returns
    -------
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    """

    atoms = np.append(x, distribution.atoms)
    pmv = np.append(p, distribution.pmv * (1 - p))

    if sort:
        indices = np.argsort(atoms)
        atoms = atoms[indices]
        pmv = pmv[indices]

    return DiscreteDistribution(pmv, atoms)


def combine_indep_dstns(*distributions, seed=0):
    """
    Given n independent vector-valued discrete distributions, construct their joint discrete distribution.
    Can take multivariate discrete distributions as inputs.

    Parameters
    ----------
    distributions : DiscreteDistribution
        Arbitrary number of discrete distributionss to combine. Their realizations must be
        vector-valued (for each D in distributions, it must be the case that len(D.dim())==1).

    Returns
    -------
    A DiscreteDistribution representing the joint distribution of the given
    random variables.
    """
    # Get information on the distributions
    dist_lengths = ()
    dist_dims = ()
    dist_is_labeled = ()
    var_labels = ()
    for dist in distributions:

        if len(dist.dim()) > 1:
            raise NotImplementedError(
                "We currently only support combining vector-valued distributions."
            )

        dist_dims += (dist.dim(),)
        dist_lengths += (len(dist.pmv),)

        labeled = isinstance(dist, DiscreteDistributionLabeled)
        dist_is_labeled += (labeled,)
        if labeled:
            var_labels += tuple(dist.dataset.data_vars.keys())
        else:
            var_labels += tuple([""] * dist.dim()[0])

    number_of_distributions = len(distributions)

    all_labeled = all(dist_is_labeled)
    labels_are_unique = len(var_labels) == len(set(var_labels))

    # We need the combinations of indices of realizations in all
    # distributions
    inds = np.meshgrid(
        *[np.array(range(l), dtype=int) for l in dist_lengths], indexing="ij"
    )
    inds = [x.flatten() for x in inds]

    atoms_out = []
    P_temp = []
    for i, ind_vec in enumerate(inds):
        atoms_out += [distributions[i].atoms[..., ind_vec]]
        P_temp += [distributions[i].pmv[ind_vec]]

    atoms_out = np.concatenate(atoms_out, axis=0)
    P_temp = np.stack(P_temp, axis=0)
    P_out = np.prod(P_temp, axis=0)

    assert np.isclose(np.sum(P_out), 1), "Probabilities do not sum to 1!"

    if all_labeled and labels_are_unique:
        combined_dstn = DiscreteDistributionLabeled(
            pmv=P_out,
            data=atoms_out,
            var_names=var_labels,
            seed=seed,
        )
    else:
        if all_labeled and not labels_are_unique:
            warn(
                "There are duplicated labels in the provided distributions. Returning a non-labeled combination"
            )
        combined_dstn = DiscreteDistribution(P_out, atoms_out, seed=seed)

    return combined_dstn


def calc_expectation(dstn, func=lambda x: x, *args):
    """
    Expectation of a function, given an array of configurations of its inputs
    along with a DiscreteDistribution object that specifies the probability
    of each configuration.

    Parameters
    ----------
    dstn : DiscreteDistribution
        The distribution over which the function is to be evaluated.
    func : function
        The function to be evaluated.
        This function should take an array of shape dstn.dim() and return
        either arrays of arbitrary shape or scalars.
        It may also take other arguments *args.
    *args :
        Other inputs for func, representing the non-stochastic arguments.
        The the expectation is computed at f(dstn, *args).

    Returns
    -------
    f_exp : np.array or scalar
        The expectation of the function at the queried values.
        Scalar if only one value.
    """

    f_query = [func(dstn.atoms[..., i], *args) for i in range(len(dstn.pmv))]

    f_query = np.stack(f_query, axis=-1)

    # From the numpy.dot documentation:
    # If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
    # If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    # Thus, if func returns scalars, f_exp will be a scalar and if it returns arrays f_exp
    # will be an array of the same shape.
    f_exp = np.dot(f_query, dstn.pmv)

    return f_exp


def distr_of_function(dstn, func=lambda x: x, *args):
    """
    Finds the distribution of a random variable Y that is a function
    of discrete random variable atoms, Y=f(atoms).

    Parameters
    ----------
    dstn : DiscreteDistribution
        The distribution over which the function is to be evaluated.
    func : function
        The function to be evaluated.
        This function should take an array of shape dstn.dim().
        It may also take other arguments *args.
    *args :
        Additional non-stochastic arguments for func,
        The function is computed at f(dstn, *args).

    Returns
    -------
    f_dstn : DiscreteDistribution
        The distribution of func(dstn).
    """
    # Apply function to every event realization
    f_query = [func(dstn.atoms[..., i], *args) for i in range(len(dstn.pmv))]

    # Stack results along their last (new) axis
    f_query = np.stack(f_query, axis=-1)

    f_dstn = DiscreteDistribution(dstn.pmv, f_query)

    return f_dstn


class MarkovProcess(Distribution):
    """
    A representation of a discrete Markov process.

    Parameters
    ----------
    transition_matrix : np.array
        An array of floats representing a probability mass for
        each state transition.
    seed : int
        Seed for random number generator.

    """

    transition_matrix = None

    def __init__(self, transition_matrix, seed=0):
        """
        Initialize a discrete distribution.

        """
        self.transition_matrix = transition_matrix

        # Set up the RNG
        super().__init__(seed)

    def draw(self, state):
        """
        Draw new states fromt the transition matrix.

        Parameters
        ----------
        state : int or nd.array
            The state or states (1-D array) from which to draw new states.

        Returns
        -------
        new_state : int or nd.array
            New states.
        """

        def sample(s):
            return self.RNG.choice(
                self.transition_matrix.shape[1], p=self.transition_matrix[s, :]
            )

        array_sample = np.frompyfunc(sample, 1, 1)

        return array_sample(state)


def expected(func=None, dist=None, args=(), **kwargs):
    """
    Expectation of a function, given an array of configurations of its inputs
    along with a DiscreteDistribution(atomsRA) object that specifies the probability
    of each configuration.

    Parameters
    ----------
    func : function
        The function to be evaluated.
        This function should take the full array of distribution values
        and return either arrays of arbitrary shape or scalars.
        It may also take other arguments *args.
        This function differs from the standalone `calc_expectation`
        method in that it uses numpy's vectorization and broadcasting
        rules to avoid costly iteration.
        Note: If you need to use a function that acts on single outcomes
        of the distribution, consier `distribution.calc_expectation`.
    dist : DiscreteDistribution or DiscreteDistributionLabeled
        The distribution over which the function is to be evaluated.
    args : tuple
        Other inputs for func, representing the non-stochastic arguments.
        The the expectation is computed at f(dstn, *args).
    labels : bool
        If True, the function should use labeled indexing instead of integer
        indexing using the distribution's underlying rv coordinates. For example,
        if `dims = ('rv', 'x')` and `coords = {'rv': ['a', 'b'], }`, then
        the function can be `lambda x: x["a"] + x["b"]`.

    Returns
    -------
    f_exp : np.array or scalar
        The expectation of the function at the queried values.
        Scalar if only one value.
    """

    if not isinstance(args, tuple):
        args = (args,)

    if isinstance(dist, DiscreteDistributionLabeled):
        return dist.expected(func, *args, **kwargs)
    elif isinstance(dist, DiscreteDistribution):
        return dist.expected(func, *args)
