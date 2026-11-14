"""Tests for IndShockConsumerType.check_conditions: the supported-cycle gate, the
condition factors, and the probability of the worst income event under the laws
of the shocks.

Kept in its own module rather than appended to test_IndShockConsumerType.py:
that file is the tail end of several in-flight branches, and every addition to
its final line collides with theirs.
"""

import unittest

import numpy as np

from HARK import _log
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)


class testConditionCheckUnsupportedCycles(unittest.TestCase):
    """The supported-cycle gate in _setup_condition_check.

    The conditions concern the limit as the horizon extends, so check_conditions()
    tests and reports them only for an infinite-horizon model, and only with a
    cycle length of 1. Every other caller in the suite uses cycles=0 with
    T_cycle=1, so the early-return branches -- and the obligation to thread
    `verbose` back out through the returned tuple -- are otherwise unexercised.
    """

    TRIVIAL = "only supported for infinite horizon models"

    def test_finite_horizon_model_has_no_conditions(self):
        # Neither tested nor reported: no conditions, no report, nothing logged
        agent = IndShockConsumerType(**init_lifecycle)
        self.assertNotEqual(agent.cycles, 0)
        with self.assertNoLogs(_log, level="DEBUG"):
            agent.check_conditions()
        self.assertEqual(agent.bilt["conditions_report"], "")
        self.assertEqual(agent.conditions, {})
        self.assertFalse(agent.degenerate)
        # The same for a finite horizon with a single period repeated
        agent = IndShockConsumerType(cycles=3)
        with self.assertNoLogs(_log, level="DEBUG"):
            agent.check_conditions(verbose=3)
        self.assertEqual(agent.bilt["conditions_report"], "")
        self.assertEqual(agent.conditions, {})

    def test_infinite_horizon_with_a_longer_cycle_says_so(self):
        agent = IndShockConsumerType(**init_lifecycle)
        agent.cycles = 0
        agent.check_conditions()
        self.assertIn(self.TRIVIAL, agent.bilt["conditions_report"])
        self.assertEqual(agent.conditions, {})

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


class testWorstIncomeEventFromTheLaws(unittest.TestCase):
    """The probability of the worst income event, WorstPrb, comes from the laws
    of the shocks (eq:pWorst of BufferStockTheory's appendix "Positive Minimum
    Income"), not from the discretization's lowest atoms (WorstPrbAtoms)."""

    def conditions(self, **kwds):
        agent = IndShockConsumerType(cycles=0, verbose=0, **kwds)
        agent.check_conditions(verbose=0)
        return agent

    def test_positive_income_in_the_worst_event(self):
        # HARK's default: income 0.3 in unemployment and lognormal shocks, whose
        # infima carry no mass, so the worst event has probability zero and the
        # MPC at the bottom is one; the lowest atoms would say UnempPrb / 7
        for BoroCnstArt in (0.0, None):
            agent = self.conditions(BoroCnstArt=BoroCnstArt)
            self.assertEqual(agent.bilt["WorstPrb"], 0.0)
            self.assertAlmostEqual(agent.bilt["WorstPrbAtoms"], 0.05 / 7, places=12)
            self.assertEqual(agent.bilt["WRPFac"], 0.0)
            self.assertEqual(agent.bilt["MPCmax"], 1.0)
            self.assertTrue(agent.conditions["WRIC"])

    def test_zero_income_in_the_worst_event(self):
        # With zero income in unemployment the worst event is unemployment,
        # whatever the permanent shock: both readings give UnempPrb
        agent = self.conditions(IncUnemp=0.0, BoroCnstArt=None)
        self.assertAlmostEqual(agent.bilt["WorstPrb"], 0.05, places=12)
        self.assertAlmostEqual(agent.bilt["WorstPrbAtoms"], 0.05, places=12)
        self.assertAlmostEqual(
            agent.bilt["MPCmax"],
            1.0 - 0.05**0.5 * agent.bilt["RPFac"],
            places=12,
        )

    def test_shocks_with_a_single_value(self):
        # With both shocks degenerate, the worst event is unemployment at income
        # 0.3, with probability UnempPrb; the laws' infima are the points
        agent = self.conditions(PermShkStd=[0.0], TranShkStd=[0.0], BoroCnstArt=None)
        np.testing.assert_allclose(agent.IncShkDstn[0].limit["infimum"], [1.0, 0.3])
        self.assertAlmostEqual(agent.bilt["WorstPrb"], 0.05, places=12)
        self.assertAlmostEqual(
            agent.bilt["MPCmax"],
            1.0 - 0.05**0.5 * agent.bilt["RPFac"],
            places=12,
        )

    def test_the_report_names_the_laws(self):
        agent = self.conditions()
        self.assertIn("under the laws of the shocks", agent.bilt["conditions_report"])
        self.assertIn("℘^(1/ρ) Þ/R < 1", agent.bilt["conditions_report"])
