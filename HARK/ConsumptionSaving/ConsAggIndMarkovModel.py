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

* ``AggIndMarkovConsumerType`` — an ``AgentType`` subclass that implements
  two-step Markov draws (macro from economy, micro per-agent) each period.

Design note
-----------
This class was created in response to the prompt:

    "Create a comprehensive, self-contained prompt document that will guide
    creation of a new general-purpose HARK class (AggIndMarkovConsumerType)
    and companion AggIndMarkovEconomy that unify the patterns currently
    implemented ad-hoc in HARK's KrusellSmithType and HAFiscal's
    AggFiscalType."

The prompt was executed by Claude Opus 4.6 (Anthropic, 2025).
"""

import numpy as np

from HARK import AgentType


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
    CondMrkvArrays : list of np.ndarray, each shape (N, N)
        ``CondMrkvArrays[j][mi, mj]`` is ``Pr(micro'=mj | micro=mi, macro'=j)``.
        There must be M such matrices (one per destination macro state).

    Returns
    -------
    np.ndarray, shape (M*N, M*N)
        Full transition matrix with combined-state indexing
        ``combined = N * macro + micro``.
    """
    M = MacroMrkvArray.shape[0]
    N = CondMrkvArrays[0].shape[0]
    full_size = M * N
    MrkvArray = np.zeros((full_size, full_size))

    for i in range(M):
        for j in range(M):
            p_macro = MacroMrkvArray[i, j]
            cond_micro = CondMrkvArrays[j]
            MrkvArray[N * i : N * (i + 1), N * j : N * (j + 1)] = p_macro * cond_micro

    return MrkvArray


# =============================================================================
# AggIndMarkovConsumerType
# =============================================================================


class AggIndMarkovConsumerType(AgentType):
    """
    A consumer with two-level hierarchical discrete Markov states.

    * **M** aggregate (macro) states — common to all agents, received from
      an economy each period via the sow variable ``"Mrkv"``.
    * **N** idiosyncratic (micro) states — drawn per-agent each period,
      conditional on the new macro state.

    The combined state index is ``N * macro + micro``.

    This class does **not** bundle a solver; subclasses must set one
    via ``default_["solver"]``.

    Subclass hooks
    --------------
    * ``get_micro_markov_states()`` — override for exact-match or other
      custom micro-state draws (default: stochastic draw from
      ``CondMrkvArrays``).
    * ``get_states()``, ``get_controls()``, ``get_poststates()`` — the
      usual model-specific economics.

    Attributes set each period
    --------------------------
    * ``MacroMrkvNow`` (int): current macro state (scalar, from economy).
    * ``MicroMrkvNow`` (np.ndarray of int): per-agent micro states.
    * ``MrkvCombined`` (np.ndarray of int): per-agent combined-state indices.
    """

    shock_vars_ = ["Mrkv"]

    def __init__(self, num_macro_states, num_micro_states, **kwds):
        self.num_macro_states = num_macro_states
        self.num_micro_states = num_micro_states
        self.CondMrkvArrays = None  # set by economy or subclass
        AgentType.__init__(self, **kwds)

    # ----- Hierarchical Markov draw machinery --------------------------------

    def get_markov_states(self):
        """Two-step draw: macro (from economy), then micro (per-agent)."""
        self.get_macro_markov_states()
        self.get_micro_markov_states()
        N = self.num_micro_states
        self.MrkvCombined = N * self.MacroMrkvNow + self.MicroMrkvNow

    def get_macro_markov_states(self):
        """Read the scalar macro state sowed by the economy as ``"Mrkv"``."""
        self.MacroMrkvNow = int(self.shocks["Mrkv"])

    def get_micro_markov_states(self):
        """
        Draw idiosyncratic micro states conditional on the current macro state.

        Default implementation: stochastic draw from ``self.CondMrkvArrays``.
        Override for exact-match permutation logic (e.g. Krusell-Smith).
        """
        N = self.num_micro_states
        MacroNow = self.MacroMrkvNow
        micro_prev = self.MicroMrkvNow.copy()

        new_micro = np.empty_like(micro_prev)
        for mi in range(N):
            mask = micro_prev == mi
            n_agents = mask.sum()
            if n_agents == 0:
                continue
            probs = self.CondMrkvArrays[MacroNow][mi, :]
            probs = probs / probs.sum()
            draws = self.RNG.choice(N, size=n_agents, p=probs)
            new_micro[mask] = draws
        self.MicroMrkvNow = new_micro

    # ----- Convenience -------------------------------------------------------

    def macro_from_combined(self, mrkv):
        """Extract the macro state from a combined-state index."""
        return mrkv // self.num_micro_states

    def micro_from_combined(self, mrkv):
        """Extract the micro state from a combined-state index."""
        return mrkv % self.num_micro_states
