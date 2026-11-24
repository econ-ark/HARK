"""
Tests for NewbornEmployed: a newborn that draws a transitory shock in its first
period (NewbornTransShk True) is born employed by default, its draw coming from
the income distribution with the unemployment event left out. False lets it
draw the unemployment event; a newborn that draws no transitory shock gets 1
either way.
"""

import unittest
from copy import deepcopy

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsMarkovModel import (
    MarkovConsumerType,
    init_indshk_markov,
)
from HARK.distributions import employed_atoms, restrict_to_atoms
from HARK.dual_measure import DualMeasureMixin

# A high probability of unemployment, so that a newborn drawing from the whole
# law draws the event many times among the agents
UNEMP = {"UnempPrb": 0.3, "IncUnemp": 0.2}


def ind_shock_agent(**kwargs):
    agent = IndShockConsumerType(
        cycles=0, AgentCount=1000, T_sim=1, seed=5, **UNEMP, **kwargs
    )
    agent.solve()
    return agent


def first_period(agent):
    """Simulate the birth period of every agent, and return its shocks."""
    agent.initialize_sim()
    agent.simulate(1)
    return agent.shocks["PermShk"], agent.shocks["TranShk"]


class testEmployedAtoms(unittest.TestCase):
    def test_the_unemployment_event_is_left_out(self):
        dstn = IndShockConsumerType(cycles=0, **UNEMP).IncShkDstn[0]
        keep = employed_atoms(dstn)
        np.testing.assert_array_equal(keep, np.asarray(dstn.atoms)[1] != 0.2)
        part = restrict_to_atoms(dstn, keep)
        self.assertAlmostEqual(part.pmv.sum(), 1.0, places=14)
        self.assertNotIn(0.2, np.asarray(part.atoms)[1])
        # The employed draws keep their relative probabilities
        np.testing.assert_allclose(part.pmv, dstn.pmv[keep] / (1.0 - 0.3), rtol=1e-12)
        self.assertIs(part._rng, dstn._rng)

    def test_no_unemployment_event_recorded(self):
        dstn = IndShockConsumerType(cycles=0, UnempPrb=0.0).IncShkDstn[0]
        self.assertIsNone(employed_atoms(dstn))


class testNewbornEmployedIndShock(unittest.TestCase):
    def test_born_employed_by_default(self):
        agent = ind_shock_agent(NewbornTransShk=True)
        self.assertTrue(agent.NewbornEmployed)
        perm, tran = first_period(agent)
        self.assertNotIn(0.2, tran)
        self.assertGreater(np.unique(tran).size, 1)

    def test_off_lets_newborns_draw_the_event(self):
        agent = ind_shock_agent(NewbornTransShk=True, NewbornEmployed=False)
        perm, tran = first_period(agent)
        self.assertIn(0.2, tran)

    def test_no_transitory_shock_is_one(self):
        agent = ind_shock_agent(NewbornTransShk=False)
        perm, tran = first_period(agent)
        np.testing.assert_array_equal(tran, 1.0)

    def test_survivors_still_draw_the_event(self):
        agent = ind_shock_agent(NewbornTransShk=True, LivPrb=[1.0])
        agent.T_sim = 2
        agent.initialize_sim()
        agent.simulate(2)
        self.assertIn(0.2, agent.shocks["TranShk"])


class testNewbornEmployedMarkov(unittest.TestCase):
    def test_born_employed_by_default(self):
        params = deepcopy(init_indshk_markov)
        params["MrkvArray"] = [np.array([[0.9, 0.1], [0.2, 0.8]])]
        params["constructors"] = dict(params["constructors"])
        params["constructors"]["MrkvArray"] = None
        params.update(AgentCount=1000, T_sim=1, seed=11, NewbornTransShk=True)
        params["UnempPrb"] = np.array([0.3, 0.3])
        params["IncUnemp"] = np.array([0.2, 0.2])
        agent = MarkovConsumerType(**params)
        agent.cycles = 0
        agent.solve()
        perm, tran = first_period(agent)
        self.assertNotIn(0.2, tran)
        agent.NewbornEmployed = False
        perm, tran = first_period(agent)
        self.assertIn(0.2, tran)


class DualIndShock(DualMeasureMixin, IndShockConsumerType):
    pass


class testNewbornEmployedDualMeasure(unittest.TestCase):
    def test_q_measure_newborns_are_born_employed(self):
        agent = DualIndShock(
            cycles=0, AgentCount=1000, T_sim=1, seed=13, NewbornTransShk=True, **UNEMP
        )
        agent.solve()
        agent.setup_Q_measure()
        agent.initialize_sim()
        agent.simulate(1)
        self.assertNotIn(0.2, agent.shocks["TranShk"])
        self.assertNotIn(0.2, agent.shocks_Q["TranShk"])


if __name__ == "__main__":
    unittest.main()
