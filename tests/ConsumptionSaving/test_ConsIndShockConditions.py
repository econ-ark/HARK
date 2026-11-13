"""Tests for IndShockConsumerType.check_conditions: the supported-cycle gate
and the condition factors.

Kept in its own module rather than appended to test_IndShockConsumerType.py:
that file is the tail end of several in-flight branches, and every addition to
its final line collides with theirs.
"""

import unittest

import numpy as np

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


class testFiniteValueOfAutarkyFactor(unittest.TestCase):
    """The risk-modified FVAC factor discounts by survival.

    Death ends utility, so the value of autarky discounts each period by
    DiscFac * LivPrb, as the perfect foresight factor PFVAFac already does.
    """

    def make_agent(self, LivPrb):
        agent = IndShockConsumerType(cycles=0, LivPrb=[LivPrb])
        agent.check_conditions()
        return agent

    def test_VAFac_matches_its_definition(self):
        agent = self.make_agent(0.98)
        dstn = agent.PermShkDstn[0]
        EPermShk = np.dot(dstn.pmv, dstn.atoms[0] ** (1.0 - agent.CRRA))
        VAFac = (
            agent.DiscFac
            * agent.LivPrb[0]
            * agent.PermGroFac[0] ** (1.0 - agent.CRRA)
            * EPermShk
        )
        self.assertAlmostEqual(agent.bilt["VAFac"], VAFac, places=12)

    def test_VAFac_scales_with_survival(self):
        immortal = self.make_agent(1.0)
        mortal = self.make_agent(0.9)
        self.assertAlmostEqual(
            mortal.bilt["VAFac"], 0.9 * immortal.bilt["VAFac"], places=12
        )
