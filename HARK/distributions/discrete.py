from typing import Any, Callable, Dict, List, Optional, Union

from copy import deepcopy
import numpy as np
import xarray as xr
from scipy import stats
from scipy.stats import rv_discrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen

from HARK.distributions.base import Distribution


class DiscreteFrozenDistribution(rv_discrete_frozen, Distribution):
    """
    Parameterized discrete distribution from scipy.stats with seed management.
    """

    def __init__(
        self, dist: rv_discrete, *args: Any, seed: int = None, **kwds: Any
    ) -> None:
        """
        Parameterized discrete distribution from scipy.stats with seed management.

        Parameters
        ----------
        dist : rv_discrete
            Discrete distribution from scipy.stats.
        seed : int, optional
            Seed for random number generator, by default 0
        """

        rv_discrete_frozen.__init__(self, dist, *args, **kwds)
        Distribution.__init__(self, seed=seed)


class Bernoulli(DiscreteFrozenDistribution):
    """
    A Bernoulli distribution.

    Parameters
    ----------
    p : float or [float]
        Probability or probabilities of the event occurring (True).

    seed : int
        Seed for random number generator.
    """

    def __init__(self, p=0.5, seed=None):
        self.p = np.asarray(p)
        # Set up the RNG
        super().__init__(stats.bernoulli, p=self.p, seed=seed)

        self.pmv = np.array([1 - self.p, self.p])
        self.atoms = np.array(
            [[0, 1]]
        )  # Ensure atoms is properly shaped like other distributions
        self.limit = {
            "dist": self,
            "infimum": np.array([0.0]),
            "supremum": np.array([1.0]),
        }
        self.infimum = np.array([0.0])
        self.supremum = np.array([1.0])

    def dim(self):
        """
        Last dimension of self.atoms indexes "atom."
        """
        return self.atoms.shape[:-1]


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
    limit : dict
        Dictionary with information about the continuous distribution from which
        this distribution was generated. The reference distribution is in the entry
        called 'dist'.
    seed : int
        Seed for random number generator.
    """

    def __init__(
        self,
        pmv: np.ndarray,
        atoms: np.ndarray,
        seed: int = None,
        limit: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(seed=seed)

        self.pmv = np.asarray(pmv)
        self.atoms = np.atleast_2d(atoms)
        if limit is None:
            limit = {
                "infimum": np.min(self.atoms, axis=-1),
                "supremum": np.max(self.atoms, axis=-1),
            }
        self.limit = limit

        # Check that pmv and atoms have compatible dimensions.
        if not self.pmv.size == self.atoms.shape[-1]:
            raise ValueError(
                "Provided pmv and atoms arrays have incompatible dimensions. "
                + "The length of the pmv must be equal to that of atoms's last dimension."
            )

    def dim(self) -> int:
        """
        Last dimension of self.atoms indexes "atom."
        """
        return self.atoms.shape[:-1]

    def draw_events(self, N: int) -> np.ndarray:
        """
        Draws N 'events' from the distribution PMF.
        These events are indices into atoms.
        """
        # Generate a cumulative distribution
        base_draws = self._rng.uniform(size=N)
        cum_dist = np.cumsum(self.pmv)

        # Convert the basic uniform draws into discrete draws
        indices = cum_dist.searchsorted(base_draws)

        return indices

    def draw(
        self,
        N: int,
        atoms: Union[None, int, np.ndarray] = None,
        exact_match: bool = False,
    ) -> np.ndarray:
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
            indices = self._rng.permutation(event_list)

        # Draw event indices randomly from the discrete distribution
        else:
            indices = self.draw_events(N)

        # Create and fill in the output array of draws based on the output of event indices
        draws = atoms[..., indices]

        # TODO: some models expect univariate draws to just be a 1d vector. Fix those models.
        if len(draws.shape) == 2 and draws.shape[0] == 1:
            draws = draws.flatten()

        return draws

    def expected(
        self, func: Optional[Callable] = None, *args: np.ndarray
    ) -> np.ndarray:
        """
        Expected value of a function, given an array of configurations of its
        inputs along with a DiscreteDistribution object that specifies the
        probability of each configuration.

        If no function is provided, it's much faster to go straight to dot
        product instead of calling the dummy function.

        If a function is provided, we need to add one more dimension,
        the atom dimension, to any inputs that are n-dim arrays.
        This allows numpy to easily broadcast the function's output.
        For more information on broadcasting, see:
        https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules

        Parameters
        ----------
        func : function
            The function to be evaluated.
            This function should take the full array of distribution values
            and return either arrays of arbitrary shape or scalars.
            It may also take other arguments \\*args.
            This function differs from the standalone `calc_expectation`
            method in that it uses numpy's vectorization and broadcasting
            rules to avoid costly iteration.
            Note: If you need to use a function that acts on single outcomes
            of the distribution, consider `distribution.calc_expectation`.
        \\*args :
            Other inputs for func, representing the non-stochastic arguments.
            The the expectation is computed at ``f(dstn, *args)``.

        Returns
        -------
        f_exp : np.array or scalar
            The expectation of the function at the queried values.
            Scalar if only one value.
        """

        if func is None:
            f_query = self.atoms
        else:
            args = [
                np.expand_dims(arg, -1) if isinstance(arg, np.ndarray) else arg
                for arg in args
            ]

            f_query = func(self.atoms, *args)

        f_exp = np.dot(f_query, self.pmv)

        return f_exp

    def dist_of_func(
        self, func: Callable[..., float] = lambda x: x, *args: Any
    ) -> "DiscreteDistribution":
        """
        Finds the distribution of a random variable Y that is a function
        of discrete random variable atoms, Y=f(atoms).

        Parameters
        ----------
        func : function
            The function to be evaluated.
            This function should take the full array of distribution values.
            It may also take other arguments \\*args.
        \\*args :
            Additional non-stochastic arguments for func,
            The function is computed as ``f(dstn, *args)``.

        Returns
        -------
        f_dstn : DiscreteDistribution
            The distribution of func(dstn).
        """
        # we need to add one more dimension,
        # the atom dimension, to any inputs that are n-dim arrays.
        # This allows numpy to easily broadcast the function's output.
        args = [
            np.expand_dims(arg, -1) if isinstance(arg, np.ndarray) else arg
            for arg in args
        ]
        f_query = func(self.atoms, *args)

        f_dstn = DiscreteDistribution(list(self.pmv), f_query, seed=self.seed)

        return f_dstn

    def discretize(self, N: int, *args: Any, **kwargs: Any) -> "DiscreteDistribution":
        """
        `DiscreteDistribution` is already an approximation, so this method
        returns a copy of the distribution.

        TODO: print warning message?
        """
        return self

    def make_univariate(self, dim_to_keep, seed=0):
        """
        Make a univariate discrete distribution from this distribution, keeping
        only the specified dimension.

        Parameters
        ----------
        dim_to_keep : int
            Index of the distribution to be kept. Any other dimensions will be
            "collapsed" into the univariate atoms, combining probabilities.
        seed : int, optional
            Seed for random number generator of univariate distribution

        Returns
        -------
        univariate_dstn : DiscreteDistribution
            Univariate distribution with only the specified index.
        """
        # Do basic validity and triviality checks
        if (self.atoms.shape[0] == 1) and (dim_to_keep == 0):
            return deepcopy(self)  # Return copy of self if only one dimension
        if dim_to_keep >= self.atoms.shape[0]:
            raise ValueError("dim_to_keep exceeds dimensionality of distribution.")

        # Construct values and probabilities for univariate distribution
        atoms_temp = self.atoms[dim_to_keep]
        vals_to_keep = np.unique(atoms_temp)
        probs_to_keep = np.zeros_like(vals_to_keep)
        for i in range(vals_to_keep.size):
            val = vals_to_keep[i]
            these = atoms_temp == val
            probs_to_keep[i] = np.sum(self.pmv[these])

        # Make and return the univariate distribution
        univariate_dstn = DiscreteDistribution(
            pmv=probs_to_keep, atoms=vals_to_keep, seed=seed
        )
        return univariate_dstn


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
        pmv: np.ndarray,
        atoms: np.ndarray,
        seed: int = None,
        limit: Optional[Dict[str, Any]] = None,
        name: str = "DiscreteDistributionLabeled",
        attrs: Optional[Dict[str, Any]] = None,
        var_names: Optional[List[str]] = None,
        var_attrs: Optional[List[Optional[Dict[str, Any]]]] = None,
    ):
        super().__init__(pmv, atoms, seed=seed, limit=limit)

        # vector-value distributions

        if self.atoms.ndim > 2:
            raise NotImplementedError(
                "Only vector-valued distributions are supported for now."
            )

        attrs = {} if attrs is None else attrs
        if limit is None:
            limit = {
                "infimum": np.min(self.atoms, axis=-1),
                "supremum": np.max(self.atoms, axis=-1),
            }
            self.limit = limit
        attrs.update(limit)
        attrs["name"] = name

        n_var = self.atoms.shape[0]

        # give dummy names to variables if none are provided
        if var_names is None:
            var_names = ["var_" + str(i) for i in range(n_var)]

        assert len(var_names) == n_var, (
            "Number of variable names does not match number of variables."
        )

        # give dummy attributes to variables if none are provided
        if var_attrs is None:
            var_attrs = [None] * n_var

        # a DiscreteDistributionLabeled is an xr.Dataset where the only
        # dimension is "atom", which indexes the random realizations.
        self.dataset = xr.Dataset(
            {
                var_names[i]: xr.DataArray(
                    self.atoms[i],
                    dims=("atom"),
                    attrs=var_attrs[i],
                )
                for i in range(n_var)
            },
            attrs=attrs,
        )

        # the probability mass values are stored in
        # a DataArray with dimension "atom"
        self.probability = xr.DataArray(self.pmv, dims=("atom"))

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
            limit=dist.limit,
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

        ldd.probability = pmf

        return ldd

    @property
    def _weighted(self):
        """
        Returns a DatasetWeighted object for the distribution.
        """
        return self.dataset.weighted(self.probability)

    @property
    def variables(self):
        """
        A dict-like container of DataArrays corresponding to
        the variables of the distribution.
        """
        return self.dataset.data_vars

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

    def dist_of_func(
        self, func: Callable = lambda x: x, *args, **kwargs
    ) -> DiscreteDistribution:
        """
        Finds the distribution of a random variable Y that is a function
        of discrete random variable atoms, Y=f(atoms).

        Parameters
        ----------
        func : function
            The function to be evaluated.
            This function should take the full array of distribution values.
            It may also take other arguments \\*args.
        \\*args :
            Additional non-stochastic arguments for func,
            The function is computed as ``f(dstn, *args)``.
        \\*\\*kwargs :
            Additional keyword arguments for func. Must be xarray compatible
            in order to work with xarray broadcasting.

        Returns
        -------
        f_dstn : DiscreteDistribution or DiscreteDistributionLabeled
            The distribution of func(dstn).
        """

        def func_wrapper(x: np.ndarray, *args: Any) -> np.ndarray:
            """
            Wrapper function for `func` that handles labeled indexing.
            """

            idx = self.variables.keys()
            wrapped = dict(zip(idx, x))

            return func(wrapped, *args)

        if len(kwargs):
            f_query = func(self.dataset, **kwargs)
            ldd = DiscreteDistributionLabeled.from_dataset(f_query, self.probability)

            return ldd

        return super().dist_of_func(func_wrapper, *args)

    def expected(
        self, func: Optional[Callable] = None, *args: Any, **kwargs: Any
    ) -> Union[float, np.ndarray]:
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
            It may also take other arguments \\*args.
            This function differs from the standalone `calc_expectation`
            method in that it uses numpy's vectorization and broadcasting
            rules to avoid costly iteration.
            Note: If you need to use a function that acts on single outcomes
            of the distribution, consider `distribution.calc_expectation`.
        \\*args :
            Other inputs for func, representing the non-stochastic arguments.
            The the expectation is computed at ``f(dstn, *args)``.
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
            ldd = DiscreteDistributionLabeled.from_dataset(f_query, self.probability)

            return ldd._weighted.mean("atom")
        else:
            if func is None:
                return super().expected()
            else:
                return super().expected(func_wrapper, *args)
