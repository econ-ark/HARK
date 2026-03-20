"""
Consumption-saving models with combined aggregate and idiosyncratic discrete
Markov states.  Provides AggIndMrkvConsumerType, a MarkovConsumerType subclass
with built-in hierarchical macro+micro Markov decomposition.

Used by both the Krusell-Smith (1998) model and the HAFiscal aggregate-demand
model.  KS-like models pass ``construct=False`` and supply their own solver;
HAFiscal-like models use the full MarkovConsumerType solver infrastructure.

The KrusellSmithTypeHM and KrusellSmithEconomyHM reference implementations
live in ConsAggShockModel.py alongside the originals they mirror.
"""

import numpy as np

from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType


# =============================================================================
# Generic hierarchical Markov utility functions
# =============================================================================


def make_hierarchical_mrkv_array(MacroMrkvArray, CondMrkvArrays):
    """
    Build a full (M*N) x (M*N) Markov transition matrix from macro and
    conditional micro transition matrices.

    Two formats for CondMrkvArrays are supported, auto-detected:

    **Simple** (HAFiscal-style): a flat list of M arrays, each N x N.
        ``CondMrkvArrays[j]`` = Pr(micro' | micro, macro'=j).
        Micro transitions depend only on the *destination* macro state.

    **General** (Krusell-Smith-style): a nested list M x M of N x N arrays.
        ``CondMrkvArrays[i][j]`` = Pr(micro' | micro, macro=i -> macro'=j).
        Micro transitions depend on *both* source and destination macro state.

    Detection: if ``CondMrkvArrays[0]`` is a 2-D ndarray, use simple format;
    if it is a list/sequence of arrays, use general format.

    The combined state index convention is:
        combined = N * macro + micro

    Parameters
    ----------
    MacroMrkvArray : np.ndarray, shape (M, M)
        Transition matrix for the aggregate Markov process.
    CondMrkvArrays : list of np.ndarray or list of list of np.ndarray
        Conditional micro transition matrices.  Simple format:
        ``CondMrkvArrays[j]`` is (N, N).  General format:
        ``CondMrkvArrays[i][j]`` is (N, N).

    Returns
    -------
    np.ndarray, shape (M*N, M*N)
        Full combined Markov transition matrix.
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
    """
    M = MacroMrkvArray.shape[0]
    CondMrkvArrays = []
    for i in range(M):
        row = []
        for j in range(M):
            block = MrkvIndArray[N * i : N * (i + 1), N * j : N * (j + 1)]
            p_macro = MacroMrkvArray[i, j]
            if p_macro > 0:
                row.append(block / p_macro)
            else:
                row.append(np.zeros((N, N)))
        CondMrkvArrays.append(row)
    return CondMrkvArrays


# =============================================================================
# Constructor wrapper (used by KS and other models that auto-build MrkvIndArray)
# =============================================================================


def construct_MrkvIndArray(MacroMrkvArray, CondMrkvArrays):
    """Thin wrapper around :func:`make_hierarchical_mrkv_array` for use in
    HARK constructors dicts."""
    return make_hierarchical_mrkv_array(MacroMrkvArray, CondMrkvArrays)


# =============================================================================
# AggIndMrkvConsumerType — unified hierarchical Markov consumer type
# =============================================================================


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
        """Stochastic draw of micro states from ``CondMrkvArrays``.

        Override for deterministic / searchsorted / permutation draws
        (e.g. Krusell-Smith exact-match employment transitions).
        """
        N = self.num_micro_states
        micro_prev = self.micro_from_combined(self.shocks["Mrkv"])
        new_micro = np.empty(self.AgentCount, dtype=int)

        for macro in np.unique(self.MacroMrkvNow):
            macro_mask = self.MacroMrkvNow == macro
            cond = self.CondMrkvArrays[int(macro)]
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
        return np.array(mrkv, dtype=int) // self.num_micro_states

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
        return np.array(mrkv, dtype=int) % self.num_micro_states


# Backward-compatible alias for code that imports the old name
AggIndMarkovConsumerType = AggIndMrkvConsumerType
