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
  transition matrix from an M x M aggregate matrix and M conditional N x N
  micro matrices.

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

import warnings

import numpy as np

from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.distributions.base import MarkovProcess

# Sentinel written into the micro-state buffer before the drawing loops run, so
# that an agent no loop wrote to is detectable instead of holding stale memory.
_UNSET_MICRO = -1


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
    first = CondMrkvArrays[0]
    general = isinstance(first, (list, tuple)) or (
        isinstance(first, np.ndarray) and first.ndim != 2
    )

    if general:
        N = CondMrkvArrays[0][0].shape[0]
    else:
        N = first.shape[0]

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

        Default lookup order: ``self.EconomyMrkvNow``, then
        ``self.shocks["MrkvAgg"]``, then derived from the combined state.
        """
        if hasattr(self, "EconomyMrkvNow") and self.EconomyMrkvNow is not None:
            self.MacroMrkvNow = int(self.EconomyMrkvNow) * np.ones(
                self.AgentCount, dtype=int
            )
        elif "MrkvAgg" in self.shocks:
            self.MacroMrkvNow = int(self.shocks["MrkvAgg"]) * np.ones(
                self.AgentCount, dtype=int
            )
        else:
            self.MacroMrkvNow = self.macro_from_combined(self.shocks["Mrkv"])

    def get_micro_markov_states(self):
        """Draw micro states from ``CondMrkvArrays``.

        When ``markov_shuffle`` is True, uses
        :class:`~HARK.distributions.base.MarkovProcess` with ``shuffle=True``
        per (macro, source-micro) cell - analogous to ``get_markov_states``
        on a flat Markov chain; with ``balanced_transitions``, systematic
        sampling by pLvl within each cell.  Default remains iid
        ``RNG.choice`` per cell.

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

        first = self.CondMrkvArrays[0]
        general = isinstance(first, (list, tuple)) or (
            isinstance(first, np.ndarray) and first.ndim != 2
        )

        if getattr(self, "markov_shuffle", False):
            # Checked here rather than at the top of the method: the default
            # branch below never consults a sort key, so warning there would
            # describe a fallback that does not apply.
            #
            # state_prev, not state_now: this method runs inside get_shocks,
            # which _sim_period_prologue calls *after* blanking every ndarray
            # in state_now with np.empty.  Sorting on state_now["pLvl"] there
            # sorts by uninitialized memory, which never raises and never
            # produces NaN.  The key is present either way, so testing
            # membership in state_now would not catch it.
            pLvl_prev = getattr(self, "state_prev", {}).get("pLvl")
            balanced = getattr(self, "balanced_transitions", False)
            if balanced and pLvl_prev is None:
                warnings.warn(
                    "balanced_transitions=True, but state_prev has no 'pLvl' "
                    "to sort on; micro transitions fall back to unbalanced "
                    "shuffling.  Set balanced_transitions=False to silence "
                    "this, or use an agent type that tracks pLvl.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                balanced = False

            macro_next = np.asarray(self.MacroMrkvNow, dtype=int)
            macro_prev = self.macro_from_combined(self.shocks["Mrkv"])

            if general:
                pairs = np.unique(np.column_stack([macro_prev, macro_next]), axis=0)
                for mp, mn in pairs:
                    mp_i, mn_i = int(mp), int(mn)
                    cond = self.CondMrkvArrays[mp_i][mn_i]
                    pair_mask = (macro_prev == mp_i) & (macro_next == mn_i)
                    for mi in range(N):
                        mask = np.logical_and(pair_mask, micro_prev == mi)
                        n = int(mask.sum())
                        if n == 0:
                            continue
                        if cond[mi, :].sum() <= 0.0:
                            raise ValueError(_zero_transition_msg(mp_i, mn_i, mi, n))
                        mp_proc = MarkovProcess(
                            cond, seed=int(self.RNG.integers(0, 2**31 - 1))
                        )
                        idx = np.flatnonzero(mask)
                        sort_key = None
                        if balanced:
                            sort_key = np.asarray(pLvl_prev)[idx]
                        new_micro[idx] = mp_proc.draw(
                            np.full(n, mi, dtype=int),
                            shuffle=True,
                            sort_key=sort_key,
                        )
            else:
                for mn in np.unique(macro_next):
                    mn_i = int(mn)
                    cond = self.CondMrkvArrays[mn_i]
                    macro_mask = macro_next == mn_i
                    for mi in range(N):
                        mask = np.logical_and(macro_mask, micro_prev == mi)
                        n = int(mask.sum())
                        if n == 0:
                            continue
                        if cond[mi, :].sum() <= 0.0:
                            raise ValueError(_zero_transition_msg(None, mn_i, mi, n))
                        mp_proc = MarkovProcess(
                            cond, seed=int(self.RNG.integers(0, 2**31 - 1))
                        )
                        idx = np.flatnonzero(mask)
                        sort_key = None
                        if balanced:
                            sort_key = np.asarray(pLvl_prev)[idx]
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
