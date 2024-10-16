from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import xarray as xr
from numpy import random


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

    def __init__(self, seed: Optional[int] = 0) -> None:
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
            _seed: int = random.SeedSequence().entropy
        elif isinstance(seed, (int, np.integer)):
            _seed = seed
        else:
            raise ValueError("seed must be an integer")

        self._seed: int = _seed
        self._rng: random.Generator = random.default_rng(self._seed)

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
    def seed(self, seed: int) -> None:
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

    def reset(self) -> None:
        """
        Reset the random number generator of this distribution.
        Resetting the seed will result in the same sequence of
        random numbers being generated.

        Parameters
        ----------
        """
        self._rng = random.default_rng(self.seed)

    def random_seed(self) -> None:
        """
        Generate a new random seed for this distribution.
        """
        self.seed = random.SeedSequence().entropy

    def draw(self, N: int) -> np.ndarray:
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
            T-length list of arrays of random variable draws each of size n, or
            a single array of size N (if sigma is a scalar).
        """

        mean = self.mean() if callable(self.mean) else self.mean
        size = (N, mean.size) if mean.size != 1 else N
        return self.rvs(size=size, random_state=self._rng)

    def discretize(
        self, N: int, method: str = "equiprobable", endpoints: bool = False, **kwds: Any
    ) -> "DiscreteDistribution":
        """
        Discretize the distribution into N points using the specified method.

        Parameters
        ----------
        N : int
            Number of points in the discretization.
        method : str, optional
            Method for discretization, by default "equiprobable"
        endpoints : bool, optional
            Whether to include endpoints in the discretization, by default False

        Returns
        -------
        DiscreteDistribution
            Discretized distribution.

        Raises
        ------
        NotImplementedError
            If method is not implemented for this distribution.
        """

        approx_method = "_approx_" + method

        if not hasattr(self, approx_method):
            raise NotImplementedError(
                "discretize() with method = {} not implemented for {} class".format(
                    method, self.__class__.__name__
                )
            )

        approx = getattr(self, approx_method)
        return approx(N, endpoints, **kwds)


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

    def __init__(
        self,
        pmv: np.ndarray,
        atoms: np.ndarray,
        seed: int = 0,
        limit: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(seed=seed)

        self.pmv = np.asarray(pmv)
        self.atoms = np.atleast_2d(atoms)
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
        seed: int = 0,
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
        limit = {} if limit is None else limit
        attrs.update(limit)
        attrs["name"] = name

        n_var = self.atoms.shape[0]

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
            return self._rng.choice(
                self.transition_matrix.shape[1], p=self.transition_matrix[s, :]
            )

        array_sample = np.frompyfunc(sample, 1, 1)

        return array_sample(state)


class IndexDistribution(Distribution):
    """
    This class provides a way to define a distribution that
    is conditional on an index.

    The current implementation combines a defined distribution
    class (such as Bernoulli, LogNormal, etc.) with information
    about the conditions on the parameters of the distribution.

    For example, an IndexDistribution can be defined as
    a Bernoulli distribution whose parameter p is a function of
    a different input parameter.

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
            self._rng = RNG
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
                    self.engine(seed=self._rng.integers(0, 2**31 - 1), **cond)
                )

        elif type(item0) is float:
            self.dstns = [
                self.engine(seed=self._rng.integers(0, 2**31 - 1), **conditional)
            ]

        else:
            raise (
                Exception(
                    f"IndexDistribution: Unhandled case for __getitem__ access. item0: {item0}; conditional: {self.conditional}"
                )
            )

    def __getitem__(self, y):
        return self.dstns[y]

    def discretize(self, N, **kwds):
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
            return self.dstns[0].discretize(N, **kwds)

        if type(item0) is list:
            return TimeVaryingDiscreteDistribution(
                [self[i].discretize(N, **kwds) for i, _ in enumerate(item0)]
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
                seed=self._rng.integers(0, 2**31 - 1), **self.conditional
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
