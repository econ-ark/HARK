"""Tests for the supported-cycle gate in IndShockConsumerType.check_conditions.

Kept in its own module rather than appended to test_IndShockConsumerType.py:
that file is the tail end of several in-flight branches, and every addition to
its final line collides with theirs.
"""

import unittest

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)


class testConditionCheckUnsupportedCycles(unittest.TestCase):
    """The supported-cycle gate in _setup_condition_check.

    check_conditions() only applies to infinite-horizon models with a cycle
    length of 1. Every other caller in the suite uses cycles=0 with T_cycle=1,
    so the early-return branch -- and the obligation to thread `verbose` back
    out through the returned tuple -- is otherwise unexercised.
    """

    TRIVIAL = "only supported for infinite horizon models"

    def test_lifecycle_model_skips_report(self):
        agent = IndShockConsumerType(**init_lifecycle)
        agent.check_conditions()

        self.assertIn(self.TRIVIAL, agent.bilt["conditions_report"])
        self.assertEqual(agent.conditions, {})
        self.assertFalse(agent.degenerate)

    def test_infinite_horizon_single_cycle_produces_a_report(self):
        """The complementary case: the gate must let normal models through."""
        agent = IndShockConsumerType()
        agent.cycles = 0
        agent.check_conditions()

        self.assertNotIn(self.TRIVIAL, agent.bilt["conditions_report"])
        self.assertTrue(agent.conditions)
