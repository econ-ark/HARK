from typing import Any, Optional

import numpy as np
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

        # Bounds of distribution support should be overwritten by subclasses
        self.infimum = np.array([])
        self.supremum = np.array([])

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
        discretized_dstn : DiscreteDistribution
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
        discretized_dstn = approx(N, endpoints, **kwds)
        discretized_dstn.limit["infimum"] = self.infimum.copy()
        discretized_dstn.limit["supremum"] = self.supremum.copy()
        return discretized_dstn


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
