"""
General-purpose consumer type with combined aggregate and idiosyncratic
discrete Markov states - the "hierarchical Markov" pattern.

Both Krusell-Smith (1998) and the HAFiscal aggregate-demand model share
a common structure: agents face discrete *macro* (aggregate) states that
are common to the whole economy, plus discrete *micro* (idiosyncratic)
states whose transition probabilities depend on the current macro state.
The full state space is the Cartesian product of the two, encoded as a
single integer index:

    combined = num_micro_states * macro_state + micro_state

This module provides:

* ``make_hierarchical_mrkv_array`` - builds the full (M*N) x (M*N) Markov
  transition matrix from an M x M aggregate matrix and either M conditional
  N x N micro matrices (destination-conditioned) or, in the general format,
  M x M of them keyed by source and destination macro state.

* ``AggIndMrkvConsumerType`` - a ``MarkovConsumerType`` subclass that implements
  two-step Markov draws (macro from economy, micro per-agent) each period.

Design note
-----------
This class was created in response to the prompt:

    "Create a comprehensive, self-contained prompt document that will guide
    creation of a new general-purpose HARK class (AggIndMrkvConsumerType)
    and companion AggIndMarkovEconomy that unify the patterns currently
    implemented ad-hoc in HARK's KrusellSmithType and HAFiscal's
    AggFiscalType."

The prompt was executed by Claude Opus 4.6 (Anthropic, 2025).
"""

import numpy as np

from HARK.ConsumptionSaving.ConsMarkovModel import (
    MarkovConsumerType,
    resolve_balanced_sort_key,
)
from HARK.distributions.base import MarkovProcess

# Sentinel written into the micro-state buffer before the drawing loops run, so
# that an agent no loop wrote to is detectable instead of holding stale memory.
_UNSET_MICRO = -1


def _cond_mrkv_is_general(CondMrkvArrays):
    """Whether ``CondMrkvArrays`` uses the general ``[i][j]`` format.

    Simple format: one ``(N, N)`` matrix per destination macro state, so the
    first element is a 2-D ndarray.  General format: conditioned on source
    AND destination, so the first element is itself a sequence of matrices
    (or an ndarray whose ``ndim`` is not 2).

    One predicate on purpose.  Three places branch on this format, and the
    contract they are branching on is a shape convention with no runtime
    marker, so a divergence between two of them would not fail anywhere
    near the copy that got it wrong.
    """
    first = CondMrkvArrays[0]
    return isinstance(first, (list, tuple)) or (
        isinstance(first, np.ndarray) and first.ndim != 2
    )


def _cell_seed(base_entropy, macro_prev, macro_next, micro_prev):
    """Seed for one transition cell, addressed by content, not by call order.

    The seed a cell gets must depend only on which cell it is, never on how
    many other cells happened to be occupied first.  Drawing it from
    ``self.RNG`` inside the loop ties it to position: the number and order of
    non-empty cells depends on the realized macro path, so a scenario that
    perturbs a few agents shifts every later cell's seed and redraws micro
    states for agents it never touched.  Measured at 15.5% of agents in one
    such comparison.

    That defeats the guarantee :meth:`MarkovProcess.draw` documents and goes
    to some trouble to provide -- it spawns one sub-RNG per source state from
    a single parent draw, precisely so an untouched row keeps its permutation
    when a policy change alters a different row.  The isolation was being
    undone one level up.

    Hoisting the draw above the loop is not sufficient and is the tempting
    wrong fix: drawing one seed per *occupied* cell still indexes by position.
    The seed has to come from the cell's identity, which is what this does.
    ``MarkovProcess.draw`` gets the same property by spawning over every
    source state rather than every populated one.
    """
    # macro_prev is None in the simple format, which conditions only on the
    # destination.  0 stands for "not applicable" and the real indices shift
    # up by one so they cannot collide with it.
    mp = 0 if macro_prev is None else int(macro_prev) + 1
    entropy = [base_entropy, mp, int(macro_next) + 1, int(micro_prev) + 1]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def _zero_transition_msg(macro_prev, macro_next, micro_prev, n_agents):
    """Message for a macro transition that carries agents but no probability.

    Parameters
    ----------
    macro_prev : int or None
        Source macro state, or None for the simple (destination-conditioned)
        ``CondMrkvArrays`` format, which does not condition on the source.
    macro_next : int
        Destination macro state.
    micro_prev : int
        Source micro state whose conditional row sums to zero.
    n_agents : int
        Number of agents assigned this transition.
    """
    if macro_prev is None:
        cell = f"CondMrkvArrays[{macro_next}]"
        which = f"macro state {macro_next}"
    else:
        cell = f"CondMrkvArrays[{macro_prev}][{macro_next}]"
        which = f"macro transition ({macro_prev}, {macro_next})"
    return (
        f"{n_agents} agents were assigned {which}, but row {micro_prev} of "
        f"{cell} sums to zero, so there is no micro transition to draw.  The "
        "macro state reached a zero-probability transition: either "
        "MacroMrkvArray forbids it, or the macro states supplied to this agent "
        "type are inconsistent with the conditional arrays it was given."
    )


# =============================================================================
# Utility: build the full hierarchical Markov transition matrix
# =============================================================================


__all__ = [
    "make_hierarchical_mrkv_array",
    "extract_cond_mrkv_arrays",
    "AggIndMrkvConsumerType",
]


def make_hierarchical_mrkv_array(MacroMrkvArray, CondMrkvArrays):
    """
    Build a full (M*N) x (M*N) Markov transition matrix.

    Parameters
    ----------
    MacroMrkvArray : np.ndarray, shape (M, M)
        Aggregate Markov transition matrix.
    CondMrkvArrays : list of np.ndarray or list of list of np.ndarray
        Conditional micro transition matrices, auto-detected between two
        formats.  Simple: a flat list of M arrays, each (N, N), where
        ``CondMrkvArrays[j][mi, mj]`` is ``Pr(micro'=mj | micro=mi,
        macro'=j)`` (micro transitions depend only on the destination
        macro state).  General: a nested M x M list of (N, N) arrays,
        where ``CondMrkvArrays[i][j]`` conditions on both the source and
        destination macro state (Krusell-Smith style).  Detection: if
        ``CondMrkvArrays[0]`` is a 2-D ndarray, the simple format is
        used.

    Returns
    -------
    np.ndarray, shape (M*N, M*N)
        Full transition matrix with combined-state indexing
        ``combined = N * macro + micro``.
    """
    M = MacroMrkvArray.shape[0]
    general = _cond_mrkv_is_general(CondMrkvArrays)

    if general:
        N = CondMrkvArrays[0][0].shape[0]
    else:
        N = CondMrkvArrays[0].shape[0]

    full_size = M * N
    MrkvArray = np.zeros((full_size, full_size))

    for i in range(M):
        for j in range(M):
            p_macro = MacroMrkvArray[i, j]
            cond_micro = CondMrkvArrays[i][j] if general else CondMrkvArrays[j]
            MrkvArray[N * i : N * (i + 1), N * j : N * (j + 1)] = p_macro * cond_micro

    return MrkvArray


# =============================================================================
# AggIndMrkvConsumerType
# =============================================================================


def extract_cond_mrkv_arrays(MrkvIndArray, MacroMrkvArray, N):
    """
    Extract conditional micro transition arrays in the general ``[i][j]``
    format from a combined (M*N) x (M*N) transition matrix.

    Each (N x N) block ``MrkvIndArray[N*i:N*(i+1), N*j:N*(j+1)]`` equals
    ``MacroMrkvArray[i,j] * CondMrkvArrays[i][j]``.  This function recovers
    ``CondMrkvArrays[i][j]`` by dividing each block by the corresponding
    macro probability.

    Parameters
    ----------
    MrkvIndArray : np.ndarray, shape (M*N, M*N)
        Full combined Markov transition matrix.
    MacroMrkvArray : np.ndarray, shape (M, M)
        Aggregate Markov transition matrix.
    N : int
        Number of micro states.

    Returns
    -------
    list of list of np.ndarray
        ``result[i][j]`` is an (N, N) conditional micro transition matrix.
        Blocks whose macro probability is zero come back as zero matrices.

    Raises
    ------
    ValueError
        If ``MrkvIndArray`` is not (M*N) x (M*N), or if any block is not
        ``MacroMrkvArray[i,j]`` times a row-stochastic matrix.  The latter
        condition is necessary and sufficient for the extracted arrays to
        be valid transition matrices, so an unchecked input would return
        rows that silently fail to sum to one.
    """
    M = MacroMrkvArray.shape[0]
    expected_shape = (M * N, M * N)
    if MrkvIndArray.shape != expected_shape:
        raise ValueError(
            f"MrkvIndArray has shape {MrkvIndArray.shape}, but "
            f"M={M} macro states and N={N} micro states require "
            f"{expected_shape}."
        )

    CondMrkvArrays = []
    for i in range(M):
        row = []
        for j in range(M):
            block = MrkvIndArray[N * i : N * (i + 1), N * j : N * (j + 1)]
            p_macro = MacroMrkvArray[i, j]
            block_sums = block.sum(axis=1)
            if not np.allclose(block_sums, p_macro):
                raise ValueError(
                    f"Block ({i},{j}) of MrkvIndArray is not hierarchical: "
                    f"its rows sum to {block_sums}, but MacroMrkvArray[{i},{j}]"
                    f" = {p_macro}.  Each block must be the macro probability "
                    "times a row-stochastic micro transition matrix."
                )
            if p_macro > 0:
                row.append(block / p_macro)
            else:
                row.append(np.zeros((N, N)))
        CondMrkvArrays.append(row)
    return CondMrkvArrays


class AggIndMrkvConsumerType(MarkovConsumerType):
    """
    A MarkovConsumerType with built-in hierarchical macro+micro Markov
    decomposition.  Inherits all of MarkovConsumerType's functionality
    (income shocks, state-dependent parameters, solver, simulation) and adds
    a two-step Markov draw:

        1. ``get_macro_markov_states()`` - reads aggregate state
        2. ``get_micro_markov_states()`` - draws idiosyncratic states
        3. Combines: ``shocks["Mrkv"] = num_micro_states * MacroMrkv + MicroMrkv``

    When ``num_macro_states`` / ``num_micro_states`` are not set, the class
    falls back to standard MarkovConsumerType behavior (pure clone).

    Models that don't need MarkovConsumerType's income-shock / lifecycle
    infrastructure (e.g. Krusell-Smith) should pass ``construct=False`` and
    supply their own solver via ``default_["solver"]``.

    Subclasses override:

    - ``get_macro_markov_states`` - how to read macro state (economy sow, etc.)
    - ``get_micro_markov_states`` - how to draw micro states (searchsorted, etc.)
    """

    def __init__(self, num_macro_states=None, num_micro_states=None, **kwds):
        """
        Parameters
        ----------
        num_macro_states : int or None
            Number of aggregate (macro) Markov states M.  If None, the class
            behaves as a plain MarkovConsumerType.
        num_micro_states : int or None
            Number of idiosyncratic (micro) Markov states N.  If None, the
            class behaves as a plain MarkovConsumerType.
        **kwds
            All other keyword arguments are passed through to
            ``MarkovConsumerType.__init__``.
        """
        if num_macro_states is not None:
            kwds["num_macro_states"] = num_macro_states
        if num_micro_states is not None:
            kwds["num_micro_states"] = num_micro_states
        MarkovConsumerType.__init__(self, **kwds)
        if not hasattr(self, "num_macro_states"):
            self.num_macro_states = None
        if not hasattr(self, "num_micro_states"):
            self.num_micro_states = None

    @property
    def _hierarchical(self):
        """True when both ``num_macro_states`` and ``num_micro_states`` are set."""
        return self.num_micro_states is not None and self.num_macro_states is not None

    # ----- Simulation setup --------------------------------------------------

    def initialize_sim(self):
        MarkovConsumerType.initialize_sim(self)
        if self._hierarchical:
            self.MacroMrkvNow = self.macro_from_combined(self.shocks["Mrkv"])
            self.MicroMrkvNow = self.micro_from_combined(self.shocks["Mrkv"])

    # ----- Markov state drawing ----------------------------------------------

    def get_markov_states(self):
        """Two-step hierarchical draw when configured; otherwise parent draw."""
        if not self._hierarchical:
            MarkovConsumerType.get_markov_states(self)
            return

        self.get_macro_markov_states()
        self.get_micro_markov_states()

        N = self.num_micro_states

        if getattr(self, "global_markov", False):
            self.shocks["Mrkv"] = (N * self.MacroMrkvNow + self.MicroMrkvNow).astype(
                int
            )
        else:
            dont_change = self.t_age == 0
            if self.t_sim == 0:
                dont_change[:] = True
            MrkvPrev = self.shocks["Mrkv"].copy()
            self.shocks["Mrkv"] = (N * self.MacroMrkvNow + self.MicroMrkvNow).astype(
                int
            )
            self.shocks["Mrkv"][dont_change] = MrkvPrev[dont_change]
            self.MacroMrkvNow = self.macro_from_combined(self.shocks["Mrkv"])
            self.MicroMrkvNow = self.micro_from_combined(self.shocks["Mrkv"])

    def get_macro_markov_states(self):
        """Read the aggregate Markov state.  Override in subclasses.

        Reads ``self.shocks["MrkvAgg"]`` when the economy sows it, and
        otherwise recovers the macro state from the combined index.
        """
        if "MrkvAgg" in self.shocks:
            macro = int(self.shocks["MrkvAgg"])
            self.MacroMrkvNow = macro * np.ones(self.AgentCount, dtype=int)
        else:
            self.MacroMrkvNow = self.macro_from_combined(self.shocks["Mrkv"])

    def _micro_transition_cells(self, general, macro_prev, macro_next, micro_prev, N):
        """Yield ``(macro_prev, macro_next, micro_prev, cond, mask)`` per cell.

        ``macro_prev`` is None in the simple format, which conditions only on
        the destination; that is the value ``_zero_transition_msg`` expects
        for the cell label it cannot report.

        Iteration order does **not** affect the draws. Each cell's seed comes
        from :func:`_cell_seed`, which derives it from the cell's own
        ``(macro_prev, macro_next, micro_prev)`` identity, so a cell gets the
        same seed no matter what else is or is not occupied. An earlier
        version consumed one ``self.RNG`` draw per non-empty cell, which made
        this order load-bearing and defeated common random numbers; see
        ``get_micro_markov_states``.

        The order below still reproduces the two format-specific loops it
        replaced: lexicographic in ``(macro_prev, macro_next)`` for the
        general format (what ``np.unique(..., axis=0)`` returns), ascending in
        ``macro_next`` for the simple one, and ascending in ``micro_prev``
        within each.
        """
        if general:
            pairs = np.unique(np.column_stack([macro_prev, macro_next]), axis=0)
            for mp, mn in pairs:
                mp_i, mn_i = int(mp), int(mn)
                cond = self.CondMrkvArrays[mp_i][mn_i]
                cell_mask = (macro_prev == mp_i) & (macro_next == mn_i)
                for mi in range(N):
                    yield (
                        mp_i,
                        mn_i,
                        mi,
                        cond,
                        np.logical_and(cell_mask, micro_prev == mi),
                    )
        else:
            for mn in np.unique(macro_next):
                mn_i = int(mn)
                cond = self.CondMrkvArrays[mn_i]
                cell_mask = macro_next == mn_i
                for mi in range(N):
                    yield (
                        None,
                        mn_i,
                        mi,
                        cond,
                        np.logical_and(cell_mask, micro_prev == mi),
                    )

    def get_micro_markov_states(self):
        """Draw micro states from ``CondMrkvArrays``.

        When ``markov_shuffle`` is True, uses
        :class:`~HARK.distributions.base.MarkovProcess` with ``shuffle=True``
        per cell - analogous to ``get_markov_states`` on a flat Markov chain;
        with ``balanced_transitions``, systematic sampling by pLvl within
        each cell.  Default remains iid ``RNG.choice`` per cell.

        What counts as a cell depends on the ``CondMrkvArrays`` format.  The
        simple format conditions only on the destination macro state, giving
        (destination-macro, source-micro) cells; the general format also
        conditions on the source, giving the finer partition
        (source-macro, destination-macro, source-micro).

        Override entirely for custom logic (e.g. Krusell-Smith exact-match
        employment permutations).

        Raises
        ------
        ValueError
            If agents are assigned a macro transition whose conditional row
            sums to zero.  ``extract_cond_mrkv_arrays`` returns a zero matrix
            where the macro probability is zero, so this means the macro
            states supplied to the agent contradict the conditional arrays it
            was given; there is no distribution to draw from.
        """
        N = self.num_micro_states
        micro_prev = self.micro_from_combined(self.shocks["Mrkv"])
        # Sentinel fill rather than np.empty: any agent left unwritten by the
        # loops below is caught at the end instead of carrying whatever was in
        # the allocation into shocks["Mrkv"] and indexing the solution arrays.
        new_micro = np.full(self.AgentCount, _UNSET_MICRO, dtype=int)

        general = _cond_mrkv_is_general(self.CondMrkvArrays)

        if getattr(self, "markov_shuffle", False):
            # Resolved here rather than at the top of the method: the default
            # branch below never consults a sort key, so warning there would
            # describe a fallback that does not apply.
            pLvl_prev = resolve_balanced_sort_key(self)
            balanced = pLvl_prev is not None

            macro_next = np.asarray(self.MacroMrkvNow, dtype=int)
            macro_prev = self.macro_from_combined(self.shocks["Mrkv"])

            # One loop over both formats. The zero-transition guard and the
            # draw used to appear once per format, and their two calls to
            # _zero_transition_msg had already drifted apart in shape, which
            # is what a validation branch looks like just before one copy
            # stops matching the other.
            # One draw for the whole call, then a per-cell seed derived from
            # the cell's identity. See _cell_seed for why the draw cannot go
            # inside the loop, and why hoisting it alone would not be enough.
            base_entropy = int(self.RNG.integers(0, 2**63 - 1))

            cells = self._micro_transition_cells(
                general, macro_prev, macro_next, micro_prev, N
            )
            for mp_i, mn_i, mi, cond, mask in cells:
                n = int(mask.sum())
                if n == 0:
                    continue
                if cond[mi, :].sum() <= 0.0:
                    raise ValueError(_zero_transition_msg(mp_i, mn_i, mi, n))
                mp_proc = MarkovProcess(
                    cond, seed=_cell_seed(base_entropy, mp_i, mn_i, mi)
                )
                idx = np.flatnonzero(mask)
                sort_key = np.asarray(pLvl_prev)[idx] if balanced else None
                new_micro[idx] = mp_proc.draw(
                    np.full(n, mi, dtype=int),
                    shuffle=True,
                    sort_key=sort_key,
                )
        else:
            for macro in np.unique(self.MacroMrkvNow):
                macro_mask = self.MacroMrkvNow == macro
                cond = self.CondMrkvArrays[int(macro)]
                if general and not isinstance(cond, np.ndarray):
                    raise NotImplementedError(
                        "AggIndMrkvConsumerType default get_micro_markov_states with "
                        "general CondMrkvArrays[i][j] requires markov_shuffle=True "
                        "or a subclass override."
                    )
                for mi in range(N):
                    mask = np.logical_and(macro_mask, micro_prev == mi)
                    n = mask.sum()
                    if n == 0:
                        continue
                    probs = cond[mi, :]
                    probs = probs / probs.sum()
                    new_micro[mask] = self.RNG.choice(N, size=n, p=probs)

        unset = new_micro == _UNSET_MICRO
        if unset.any():
            raise RuntimeError(
                f"get_micro_markov_states left {int(unset.sum())} of "
                f"{self.AgentCount} agents without a micro state.  This is a "
                "bug in the drawing loops, not a modelling error; please report "
                "it with the CondMrkvArrays and macro state that produced it."
            )
        self.MicroMrkvNow = new_micro

    # ----- Convenience helpers -----------------------------------------------

    def macro_from_combined(self, mrkv):
        """Extract the macro state index from a combined Markov index.

        Parameters
        ----------
        mrkv : int or np.ndarray
            Combined Markov state index (= N * macro + micro).

        Returns
        -------
        int or np.ndarray
            Macro state index.
        """
        result = np.asarray(mrkv, dtype=int) // self.num_micro_states
        return int(result) if np.ndim(mrkv) == 0 else result

    def micro_from_combined(self, mrkv):
        """Extract the micro state index from a combined Markov index.

        Parameters
        ----------
        mrkv : int or np.ndarray
            Combined Markov state index (= N * macro + micro).

        Returns
        -------
        int or np.ndarray
            Micro state index.
        """
        result = np.asarray(mrkv, dtype=int) % self.num_micro_states
        return int(result) if np.ndim(mrkv) == 0 else result
