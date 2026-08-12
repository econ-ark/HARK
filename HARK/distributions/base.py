from typing import Any, Optional

import warnings

import numpy as np
from numpy import random

MAX_INT32 = 2**31 - 1


def allocate_remainder_slots(K_exact, K, M, rng):
    """Hand out the M slots that ``floor`` left unallocated, without bias.

    ``K = floor(K_exact)`` always leaves ``M = sum(K_exact) - sum(K)`` slots
    unassigned, and how those are handed out decides whether the quota-exact
    construction is actually unbiased.  Systematic sampling on the fractional
    remainders ``Q = K_exact - K`` makes ``P(atom j gets an extra slot)``
    exactly ``Q[j]``, hence ``E[count_j] == K_exact[j]``.  Drawing the M slots
    one at a time proportional to the remaining Q instead is successive
    sampling, whose inclusion probabilities are not proportional to Q, and
    that biases the result whenever ``M >= 2``.

    This lives in one place on purpose.  It was previously written out
    separately inside ``DiscreteDistribution.draw`` and
    ``MarkovProcess._draw_shuffled``; the first copy was fixed in #1808 and
    the second silently kept the bias, because the two copies are in
    different functions and nothing makes a divergence show up as a conflict.
    Every caller that spreads leftover slots over ``J >= 2`` atoms must call
    this rather than reimplement it.

    The qualifier is doing real work.  The bias only exists when ``M >= 2``,
    since with a single leftover slot successive and systematic sampling
    are the same draw.  A two-outcome split therefore cannot exhibit it:
    the two fractional parts sum to an integer, so ``M`` is 0 or 1 and
    never more.  ``PerfForesightConsumerType._sim_death_shuffled`` is that
    case, and it resolves its remainder with one Bernoulli draw instead of
    calling this function.  That is deliberate, not an oversight -- see the
    comment there.

    M uniforms are drawn although systematic sampling needs only the first,
    so that this consumes exactly as many random numbers as the original
    implementation did and every downstream draw stays put.

    Parameters
    ----------
    K_exact : np.ndarray
        Real-valued slot counts, ``N * P``.
    K : np.ndarray
        Integer slot counts, ``floor(K_exact)``.  Modified in place and
        also returned.
    M : int
        Number of unallocated slots, ``N - sum(K)``.
    rng : np.random.Generator
        Source of the M uniforms.

    Returns
    -------
    K : np.ndarray
        ``K`` with the M leftover slots added.
    """
    draws = rng.random(M)
    if M > 0:
        Q = K_exact - K  # "missing" slots, fractional; these sum to M
        edges = np.cumsum(Q)
        # The final edge is unbounded rather than sum(Q). Queries run up
        # to draws[0] + M - 1, which is below M in exact arithmetic but
        # rounds to exactly M once draws[0] is within an ulp of 1, while
        # cumsum drift can leave sum(Q) a few ulp below M. Either alone
        # puts the last query at or past the last edge, and searchsorted
        # would then return J and index out of bounds.
        edges[-1] = np.inf
        picks = np.searchsorted(edges, draws[0] + np.arange(M), side="right")
        np.add.at(K, picks, 1)
    return K


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
            Falls back to iid when N_j * min(probs[probs > 0]) < 1 for a
            source state.  The minimum is taken over the strictly positive
            entries: a row with a structural zero would otherwise have
            min(probs) == 0 and fall back forever, however well supported
            its reachable targets are.
        sort_key : np.array or None
            When provided (same length as state), agents within each source
            state are sorted by this key and assigned to target states via
            systematic sampling rather than random permutation.  This makes
            the transitioning subgroup representative of the source population
            with respect to the sort variable (e.g. pLvl).  Only meaningful
            when shuffle=True; passing it with shuffle=False warns, because
            silently ignoring it would let a caller believe a variance
            reduction is in effect when the draw is plain iid.
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

        Raises
        ------
        IndexError
            If any source state is outside the transition matrix's rows.
            Both paths reject the same inputs; see the note below on why the
            unshuffled path cannot be left to numpy.
        """
        # Validated here rather than in _draw_shuffled, so both paths agree.
        # The unshuffled path indexes transition_matrix[state] directly, and
        # numpy raises for a state past the last row but NOT for a negative
        # one: -1 resolves to the last row and the agent is silently
        # transitioned from a different Markov state, in range and plausible,
        # with nothing to distinguish it. Measured on
        # [[0.99, 0.01], [0.50, 0.50]]: agents marked -1 moved to state 1 at
        # frequency 0.50, the last row's rate, where their own row 0 gives
        # 0.01. Unlike the shuffled path's uninitialized memory, this can
        # never come back out of range and blow up downstream.
        #
        # -1 is not hypothetical: ConsAggIndMarkovModel._UNSET_MICRO is
        # exactly -1, and MarkovConsumerType.get_markov_states passes
        # shocks["Mrkv"] straight in with no check of its own. In the
        # combined = N * macro + micro encoding, the last row is the highest
        # macro and highest micro state, so a stray sentinel lands on the
        # most favourable cell in the chain.
        state_arr = np.asarray(state)
        J_src = self.transition_matrix.shape[0]
        out_of_range = (state_arr < 0) | (state_arr >= J_src)
        if np.any(out_of_range):
            bad = np.unique(state_arr[out_of_range])
            raise IndexError(
                f"source states {bad.tolist()} are outside the transition "
                f"matrix's {J_src} rows, for {int(np.sum(out_of_range))} of "
                f"{state_arr.size} agents. Negative states are rejected "
                "rather than wrapped: numpy would resolve -1 to the last "
                "row and transition those agents from the wrong state."
            )

        if not shuffle:
            ignored = [
                name
                for name, val in (("sort_key", sort_key), ("draws", draws))
                if val is not None
            ]
            if ignored:
                warnings.warn(
                    f"{' and '.join(ignored)} passed with shuffle=False, so "
                    f"the argument has no effect and the draw is plain iid. "
                    f"Pass shuffle=True to use the requested assignment mode.",
                    stacklevel=2,
                )
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
        # Sentinel rather than np.empty: the loop below only writes agents
        # whose source state is one of the matrix's rows, so an agent outside
        # that range is never assigned. With np.empty it keeps whatever the
        # freed buffer held, which in a running simulation is the previous
        # period's Mrkv array -- in-range, plausible, and wrong. The
        # unshuffled path raises IndexError on the same input, so silence
        # here is also an inconsistency between the two paths. -1 cannot
        # collide with a real assignment, which is always in range(J).
        _UNSET = -1
        new_state = np.full_like(state, _UNSET, dtype=int)
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

            # Fall back to iid when population is too small for deterministic
            # counts. The minimum is over the strictly positive entries: a
            # transition row with a structural zero has min(probs) == 0, so
            # an unfiltered minimum would make this test true for every N_j
            # and pin such rows to the iid path permanently, however well
            # supported their reachable targets are.
            if N_j * np.min(probs[probs > 0]) < 1:
                for idx in agents_in_j:
                    new_state[idx] = sub_rng.choice(J, p=probs)
                continue

            # Floor-plus-leftover algorithm (matches DiscreteDistribution.draw)
            K_exact = N_j * probs
            K = np.floor(K_exact).astype(int)
            M = N_j - np.sum(K)  # unallocated slots

            # Unbiased allocation of the leftover slots; shared with
            # DiscreteDistribution.draw so the two cannot drift apart.
            K = allocate_remainder_slots(K_exact, K, M, sub_rng)

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
                # Target j repeated K[j] times, concatenated, is exactly the
                # rank-to-target map described above: sum(K) == N_j, so the
                # r-th entry is the target for the agent at rank r.
                new_state[sorted_agents] = np.repeat(np.arange(J), K)
            else:
                # Randomly assign agents to target states
                new_state[sub_rng.permutation(agents_in_j)] = np.repeat(np.arange(J), K)

        unset = new_state == _UNSET
        if np.any(unset):
            bad = np.unique(state[unset])
            raise IndexError(
                f"source states {bad.tolist()} are outside the transition "
                f"matrix's {J_src} rows, so {int(unset.sum())} of "
                f"{state.size} agents were assigned no target state. "
                "draw(..., shuffle=False) raises IndexError on the same "
                "input; this is the shuffled path reporting it rather than "
                "returning the uninitialized buffer."
            )
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
