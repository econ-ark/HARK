"""
General-purpose consumer type with combined aggregate and idiosyncratic
discrete Markov states — the "hierarchical Markov" pattern.

Both Krusell-Smith (1998) and the HAFiscal aggregate-demand model share
a common structure: agents face discrete *macro* (aggregate) states that
are common to the whole economy, plus discrete *micro* (idiosyncratic)
states whose transition probabilities depend on the current macro state.
The full state space is the Cartesian product of the two, encoded as a
single integer index:

    combined = num_micro_states * macro_state + micro_state

This module provides:

* ``make_hierarchical_mrkv_array`` — builds the full (M*N) x (M*N) Markov
  transition matrix from an M x M aggregate matrix and M conditional N x N
  micro matrices.

* ``AggIndMrkvConsumerType`` — a ``MarkovConsumerType`` subclass that implements
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

from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType


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

        1. ``get_macro_markov_states()`` — reads aggregate state
        2. ``get_micro_markov_states()`` — draws idiosyncratic states
        3. Combines: ``shocks["Mrkv"] = num_micro_states * MacroMrkv + MicroMrkv``

    When ``num_macro_states`` / ``num_micro_states`` are not set, the class
    falls back to standard MarkovConsumerType behavior (pure clone).

    Models that don't need MarkovConsumerType's income-shock / lifecycle
    infrastructure (e.g. Krusell-Smith) should pass ``construct=False`` and
    supply their own solver via ``default_["solver"]``.

    Subclasses override:

    - ``get_macro_markov_states`` — how to read macro state (economy sow, etc.)
    - ``get_micro_markov_states`` — how to draw micro states (searchsorted, etc.)
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

        Draws are iid ``RNG.choice`` per (macro, source-micro) cell.

        Override entirely for custom logic (e.g. Krusell-Smith exact-match
        employment permutations).
        """
        N = self.num_micro_states
        micro_prev = self.micro_from_combined(self.shocks["Mrkv"])
        new_micro = np.empty(self.AgentCount, dtype=int)

        first = self.CondMrkvArrays[0]
        general = isinstance(first, (list, tuple)) or (
            isinstance(first, np.ndarray) and first.ndim != 2
        )

        for macro in np.unique(self.MacroMrkvNow):
            macro_mask = self.MacroMrkvNow == macro
            cond = self.CondMrkvArrays[int(macro)]
            if general and not isinstance(cond, np.ndarray):
                raise NotImplementedError(
                    "AggIndMrkvConsumerType default get_micro_markov_states with "
                    "general CondMrkvArrays[i][j] requires "
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
