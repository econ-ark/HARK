"""Tests for AgentType and solver helpers extracted during the dedup refactor.

Kept out of test_core.py: that file's final line is the merge point for
several in-flight branches, so anything appended there collides with all of
them at once.
"""

import unittest

import numpy as np

from HARK.core import AgentType, _resolve_solve_one_period


class testIsIdioState(unittest.TestCase):
    """Both branches of AgentType._is_idio_state.

    Every existing caller uses an agent whose state_now entries are all
    per-agent ndarrays, so the False branch -- the entire reason the helper
    exists, keeping newborns from overwriting Market-set aggregates -- is
    otherwise never reached.
    """

    def setUp(self):
        self.agent = AgentType()
        self.agent.AgentCount = 3

    def test_per_agent_array_is_idiosyncratic(self):
        self.agent.state_now = {"aNrm": np.zeros(3)}
        self.assertTrue(self.agent._is_idio_state("aNrm"))

    def test_scalar_aggregate_is_not_idiosyncratic(self):
        self.agent.state_now = {"AggAsset": 1.5}
        self.assertFalse(self.agent._is_idio_state("AggAsset"))

    def test_wrong_length_array_is_not_idiosyncratic(self):
        self.agent.state_now = {"AggHist": np.zeros(7)}
        self.assertFalse(self.agent._is_idio_state("AggHist"))

    def test_non_array_sequence_is_not_idiosyncratic(self):
        """A list of the right length is still not a per-agent ndarray."""
        self.agent.state_now = {"weird": [0.0, 0.0, 0.0]}
        self.assertFalse(self.agent._is_idio_state("weird"))


class testResolveSolveOnePeriod(unittest.TestCase):
    """Both branches of _resolve_solve_one_period.

    The per-period-sequence branch is user-facing API but is not exercised
    anywhere in HARK itself, so a regression there would be silent.
    """

    def test_single_callable_reused_for_every_period(self):
        agent = AgentType()
        agent.solve_one_period = lambda vary_1: None

        for k in (0, 1, 5):
            solver, args = _resolve_solve_one_period(agent, k)
            self.assertIs(solver, agent.solve_one_period)
            self.assertEqual(list(args), ["vary_1"])

    def test_sequence_of_solvers_selects_by_period(self):
        agent = AgentType()
        first = lambda alpha: None
        second = lambda beta: None
        agent.solve_one_period = [first, second]

        solver0, args0 = _resolve_solve_one_period(agent, 0)
        solver1, args1 = _resolve_solve_one_period(agent, 1)

        self.assertIs(solver0, first)
        self.assertIs(solver1, second)
        self.assertEqual(list(args0), ["alpha"])
        self.assertEqual(list(args1), ["beta"])
