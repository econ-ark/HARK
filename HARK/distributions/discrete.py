from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import rv_discrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen

from HARK.distributions.base import Distribution


class DiscreteFrozenDistribution(rv_discrete_frozen, Distribution):
    """
    Parameterized discrete distribution from scipy.stats with seed management.
    """

    def __init__(
        self, dist: rv_discrete, *args: Any, seed: int = 0, **kwds: Any
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

    def __init__(self, p=0.5, seed=0):
        self.p = np.asarray(p)
        # Set up the RNG
        super().__init__(stats.bernoulli, p=self.p, seed=seed)

        self.pmv = [1 - self.p, self.p]
        self.atoms = [0, 1]
        self.limit = {"dist": self}
