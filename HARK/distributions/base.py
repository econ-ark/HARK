from typing import Any, Optional

import numpy as np
from numpy import random

MAX_INT32 = 2**31 - 1


def random_seed():
    """
    Generate a random seed for use in random number generation. This random seed
    is derived from the system clock and other variables, and is therefore
    different every time the code is run.
    For discussion on random number generation and random seeds, see
    https://docs.scipy.org/doc/scipy/tutorial/stats.html#random-number-generation
    Parameters
    ----------
    None
    Returns
    -------
    seed : int
        Random seed.
    """
    return random.SeedSequence().entropy


class Distribution:
    """
    Base class for all probability distributions
    with seed and random number generator.

    Parameters
    ----------
    seed : Optional[int]
        Seed for random number generator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
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
        self.seed = seed

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
        return self._seed

    @seed.setter
    def seed(self, seed: int) -> None:
        """
        Set seed for random number generator.

        Parameters
        ----------
        seed : int
            Seed for random number generator.
        """

        if seed is None:
            # random seed from entropy
            self._seed = random_seed()
        elif isinstance(seed, (int, np.integer)):
            self._seed = seed
        else:
            raise ValueError("seed must be an integer")

        # set random number generator with seed
        self._rng = random.default_rng(self._seed)

    def reset(self) -> None:
        """
        Reset the random number generator of this distribution.
        Resetting the seed will result in the same sequence of
        random numbers being generated.

        Parameters
        ----------
        """
        self._rng = random.default_rng(self.seed)

    def random_seed(self) -> int:
        """
        Generate a new random seed derived from the random seed in this distribution.
        """
        return self._rng.integers(0, MAX_INT32, dtype=np.int32)

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
        return self.rvs(size=N, random_state=self._rng).T

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

    It can also wrap a list of pre-discretized distributions (previously
    provided by TimeVaryingDiscreteDistribution) and provide the same API.

    Parameters
    ----------

    engine : Distribution class
        A Distribution subclass.

    conditional: dict
        Information about the conditional variation on the input parameters of the engine
        distribution. Keys should match the arguments to the engine class constructor.

    distributions: [DiscreteDistribution]
        Optional. A list of discrete distributions to wrap directly.

    seed : int
        Seed for random number generator.
    """

    conditional = None
    engine = None

    def __init__(
        self, engine=None, conditional=None, distributions=None, RNG=None, seed=None
    ):
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

        # Mode 1: wrapping a list of discrete distributions
        if distributions is not None:
            self.distributions = distributions
            self.engine = None
            self.conditional = None
            self.dstns = []
            return

        # Mode 2: engine + conditional parameters (original IndexDistribution)
        self.conditional = conditional if conditional is not None else {}
        self.engine = engine

        self.dstns = []

        # If no engine/conditional were provided, this is an invalid state.
        if self.engine is None and not self.conditional:
            raise ValueError(
                "MarkovProcess: No engine or conditional parameters provided; this should not happen in normal use."
            )

        # Test one item to determine case handling
        item0 = list(self.conditional.values())[0]

        if type(item0) is list:
            # Create and store all the conditional distributions
            for y in range(len(item0)):
                cond = {key: val[y] for (key, val) in self.conditional.items()}
                self.dstns.append(self.engine(seed=self.random_seed(), **cond))

        elif type(item0) is float:
            self.dstns = [self.engine(seed=self.random_seed(), **self.conditional)]

        else:
            raise (
                Exception(
                    f"IndexDistribution: Unhandled case for __getitem__ access. item0: {item0}; conditional: {self.conditional}"
                )
            )

    def __getitem__(self, y):
        # Prefer discrete list mode if present
        if hasattr(self, "distributions") and self.distributions:
            return self.distributions[y]
        return self.dstns[y]

    def reset(self):
        # Reset the main RNG and each member distribution
        super().reset()
        for d in self.dstns:
            d.reset()

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
        dists : [DiscreteDistribution] or IndexDistribution
            If parameterization is constant, returns a single DiscreteDistribution.
            If parameterization varies with index, returns an IndexDistribution in
            discrete-list mode, wrapping the corresponding discrete distributions.
        """

        # If already in discrete list mode, return self (already discretized)
        if hasattr(self, "distributions") and self.distributions:
            return self

        # test one item to determine case handling
        item0 = list(self.conditional.values())[0]

        if type(item0) is float:
            # degenerate case. Treat the parameterization as constant.
            return self.dstns[0].discretize(N, **kwds)

        if type(item0) is list:
            # Return an IndexDistribution wrapping a list of discrete distributions
            return IndexDistribution(
                distributions=[
                    self[i].discretize(N, **kwds) for i, _ in enumerate(item0)
                ],
                seed=self.seed,
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

        # If wrapping discrete distributions, draw from those
        if hasattr(self, "distributions") and self.distributions:
            draws = np.zeros(condition.size)
            for c in np.unique(condition):
                these = c == condition
                N = np.sum(these)
                draws[these] = self.distributions[c].draw(N)
            return draws

        # test one item to determine case handling
        item0 = list(self.conditional.values())[0]

        if type(item0) is float:
            # degenerate case. Treat the parameterization as constant.
            N = condition.size

            return self.engine(seed=self.random_seed(), **self.conditional).draw(N)

        if type(item0) is list:
            # conditions are indices into list
            # somewhat convoluted sampling strategy retained
            # for test backwards compatibility

            draws = np.zeros(condition.size)

            for c in np.unique(condition):
                these = c == condition
                N = np.sum(these)

                draws[these] = self[c].draw(N)

            return draws
