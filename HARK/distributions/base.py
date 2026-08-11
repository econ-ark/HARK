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

    def draw(self, state, shuffle=False, sort_key=None, draws=None):
        """
        Draw new states from the transition matrix.

        Parameters
        ----------
        state : int or nd.array
            The state or states (1-D array) from which to draw new states.
        shuffle : bool
            When True, use deterministic target counts per source state
            (floor-plus-leftover algorithm) with random agent assignment.
            This eliminates sampling noise in state transition counts.
            Falls back to iid when N_j * min(probs) < 1 for a source state.
        sort_key : np.array or None
            When provided (same length as state), agents within each source
            state are sorted by this key and assigned to target states via
            systematic sampling rather than random permutation.  This makes
            the transitioning subgroup representative of the source population
            with respect to the sort variable (e.g. pLvl).  Only used when
            shuffle=True.
        draws : np.array or None
            When provided (same length as state, values in U[0,1]) AND
            sort_key is None, use rank-based stratified inverse-CDF
            assignment instead of random permutation.  Each agent's target
            is determined by the rank of their draw u_i within the source
            state's draw distribution: agents are sorted by u_i, then
            assigned in order to targets according to the quota counts
            K[j,k].  This preserves quota-exact target counts AND ensures
            the per-agent assignment is asymptotically equivalent to per-
            agent iid via Glivenko-Cantelli (rank/N_j -> u_i as N_j ->
            infinity).  Recommended for downstream estimators that use
            common random numbers across counterfactual scenarios and
            integrate over per-agent trajectories (e.g. CRN-coupled welfare
            integrals validated against iid).

        Returns
        -------
        new_state : int or nd.array
            New states.
        """
        if not shuffle:
            return self._draw_iid(state)
        return self._draw_shuffled(state, sort_key=sort_key, draws=draws)

    def _draw_iid(self, state):
        """Draw new states independently for each agent (original behavior)."""

        def sample(s):
            return self._rng.choice(
                self.transition_matrix.shape[1], p=self.transition_matrix[s, :]
            )

        array_sample = np.frompyfunc(sample, 1, 1)

        return array_sample(state)

    def _draw_shuffled(self, state, sort_key=None, draws=None):
        """Deterministic state counts with random or systematic agent assignment.

        For each source state j with N_j agents, compute target counts
        using the floor-plus-leftover algorithm (same as
        DiscreteDistribution.draw(shuffle=True)), then assign agents to
        target states.  Three assignment modes are supported:

        - ``sort_key`` provided: systematic sampling on the sorted order
          so that each target group is representative of the source
          population with respect to the sort variable.
        - ``draws`` provided (and ``sort_key`` is None): rank-based
          stratified inverse-CDF assignment.  Agents are sorted by their
          per-agent draw ``u_i``, then assigned in order to targets
          according to the quota counts ``K[j,k]``.  This is
          asymptotically equivalent to per-agent iid via Glivenko-
          Cantelli (rank/N_j -> u_i as N_j -> infinity), so a shuffled
          run with ``draws=u`` and an iid run sharing the same ``u``
          produce identical per-agent assignments in the large-N limit
          (with finite-N differences concentrated at O(sqrt(N_j))
          "borderline" agents whose u_i is near a target-CDF cutoff).
          Use this mode when the downstream estimator integrates over
          per-agent trajectories with shared draws across counterfactual
          scenarios (e.g., CRN-coupled welfare integrals).
        - Neither provided (default): random permutation.  Per-agent
          assignment is uncorrelated with any per-agent draw, so the
          shuffled run does NOT preserve per-agent identity with an iid
          run that shares the underlying random draws.  This is fine
          for aggregate estimators that depend only on marginal counts.

        Each source state uses an independent sub-RNG derived deterministically
        from the parent RNG's base seed via ``np.random.SeedSequence.spawn``.
        This guarantees that two calls with the same starting RNG state but
        different transition matrices produce identical random permutations
        for any source state whose transition row and agent set are the same
        - a correctness requirement for common random numbers in scenario
        comparisons (e.g., counterfactual experiments where the policy-period
        transition matrix differs from the baseline).  Without this isolation,
        the RNG state drift from earlier source states' leftover-slot
        consumption would contaminate later source states' permutations even
        when those rows are untouched by the policy change.

        Falls back to iid when a source state has too few agents for
        meaningful deterministic counts.
        """
        if draws is not None and sort_key is not None:
            raise ValueError(
                "draws and sort_key cannot both be provided; "
                "they specify mutually exclusive assignment modes."
            )
        state = np.asarray(state)
        new_state = np.empty_like(state, dtype=int)
        J = self.transition_matrix.shape[1]
        J_src = self.transition_matrix.shape[0]

        # Derive one independent sub-RNG per source state.  Advancing the
        # parent RNG by a single integers() call fixes the entropy for this
        # _draw_shuffled call; SeedSequence.spawn(J_src) then produces
        # J_src statistically-independent child seeds deterministically.
        base_entropy = int(self._rng.integers(0, 2**63 - 1))
        sub_seeds = np.random.SeedSequence(base_entropy).spawn(J_src)

        for j in range(J_src):
            agents_in_j = np.where(state == j)[0]
            N_j = len(agents_in_j)
            if N_j == 0:
                continue

            probs = self.transition_matrix[j]
            sub_rng = np.random.default_rng(sub_seeds[j])

            # Fall back to iid when population is too small for deterministic counts
            if N_j * np.min(probs[probs > 0]) < 1:
                for idx in agents_in_j:
                    new_state[idx] = sub_rng.choice(J, p=probs)
                continue

            # Floor-plus-leftover algorithm (matches DiscreteDistribution.draw)
            K_exact = N_j * probs
            K = np.floor(K_exact).astype(int)
            M = N_j - np.sum(K)  # unallocated slots

            if M > 0:
                eps = 1.0 / N_j
                Q = K_exact - eps * K  # residual probability mass
                # Local variable name avoids shadowing the `draws` parameter
                # used by the rank-based stratified mode below.
                leftover_draws = sub_rng.random(M)
                for m in range(M):
                    Q_adj = Q / np.sum(Q)
                    Q_sum = np.cumsum(Q_adj)
                    idx = np.searchsorted(Q_sum, leftover_draws[m])
                    K[idx] += 1
                    Q[idx] = 0.0

            if sort_key is not None:
                # Systematic sampling: sort agents by key, then assign
                # minority transitions evenly across the sorted order.
                sorted_agents = agents_in_j[np.argsort(sort_key[agents_in_j])]
                assigned = np.empty(N_j, dtype=int)
                remaining_mask = np.ones(N_j, dtype=bool)

                # Process target states smallest-first so minority
                # transitions get systematically spread across the
                # full range of the sort variable.
                for jp in np.argsort(K):
                    if K[jp] == 0:
                        continue
                    remaining_pos = np.where(remaining_mask)[0]
                    N_rem = len(remaining_pos)
                    if N_rem == K[jp]:
                        # Last group: assign all remaining
                        assigned[remaining_pos] = jp
                        remaining_mask[remaining_pos] = False
                    else:
                        # Systematic sample with random offset
                        spacing = N_rem / K[jp]
                        u = sub_rng.uniform(0, spacing)
                        sel = np.floor(u + np.arange(K[jp]) * spacing).astype(int)
                        sel = np.clip(sel, 0, N_rem - 1)
                        chosen = remaining_pos[sel]
                        assigned[chosen] = jp
                        remaining_mask[chosen] = False

                new_state[sorted_agents] = assigned
            elif draws is not None:
                # Rank-based stratified inverse-CDF assignment.  Sort
                # agents in source state j by their per-agent draw u_i,
                # then assign them in order to targets according to the
                # quota counts K[j,:].  Agent at rank r in source j is
                # sent to target k iff sum(K[:k]) <= r < sum(K[:k+1]).
                # As N_j -> inf, rank/N_j -> u_i (Glivenko-Cantelli), so
                # the assignment converges to per-agent iid
                # searchsorted(cumsum(P[j,:]), u_i).  Finite-N differences
                # are O(sqrt(N_j)) "borderline" agents near cutoffs.
                draws_j = draws[agents_in_j]
                sort_order = np.argsort(draws_j)
                sorted_agents = agents_in_j[sort_order]
                offset = 0
                for jp in range(J):
                    if K[jp] == 0:
                        continue
                    new_state[sorted_agents[offset : offset + K[jp]]] = jp
                    offset += int(K[jp])
            else:
                # Randomly assign agents to target states
                perm = sub_rng.permutation(agents_in_j)
                offset = 0
                for jp in range(J):
                    new_state[perm[offset : offset + K[jp]]] = jp
                    offset += K[jp]

        return new_state


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
