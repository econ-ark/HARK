"""
Tests for NewbornPermShk, the counterpart of NewbornTransShk for the permanent
shock: when it is False, a newborn's first-period permanent shock (which carries
expected growth) is 1, so it enters with pLvl drawn from pLvlInitDstn exactly.
The default, True, is the existing behavior.
"""

import unittest
from copy import deepcopy

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsMarkovModel import (
    MarkovConsumerType,
    init_indshk_markov,
)
from HARK.dual_measure import DualMeasureMixin

# Newborns with no assets and permanent income 1, as in a model whose newborns
# enter at P = 1 and a = 0. exp(-1000) is exactly zero in floating point.
NEWBORN = {
    "kLogInitMean": -1000.0,
    "kLogInitStd": 0.0,
    "pLogInitMean": 0.0,
    "pLogInitStd": 0.0,
    "PermGroFac": [1.02],
}


def ind_shock_agent(**kwargs):
    agent = IndShockConsumerType(
        cycles=0, AgentCount=500, T_sim=1, seed=7, **NEWBORN, **kwargs
    )
    agent.track_vars = ["pLvl", "mNrm"]
    agent.solve()
    return agent


def first_period(agent):
    """Simulate the birth period of every agent, and return its shocks."""
    agent.initialize_sim()
    agent.simulate(1)
    return agent.shocks["PermShk"], agent.shocks["TranShk"]


class testNewbornPermShkIndShock(unittest.TestCase):
    def test_default_draws_permanent_shock_and_growth(self):
        agent = ind_shock_agent(NewbornTransShk=True)
        self.assertTrue(agent.NewbornPermShk)
        perm, tran = first_period(agent)
        # The default is unchanged: newborns draw the shock, growth included
        self.assertGreater(np.unique(perm).size, 1)
        np.testing.assert_allclose(agent.history["pLvl"][0], perm)

    def test_off_gives_newborns_pLvl_of_exactly_one(self):
        agent = ind_shock_agent(NewbornTransShk=True, NewbornPermShk=False)
        perm, tran = first_period(agent)
        np.testing.assert_array_equal(perm, 1.0)
        np.testing.assert_array_equal(agent.history["pLvl"][0], 1.0)
        # With no assets, market resources are the transitory draw alone
        np.testing.assert_array_equal(agent.history["mNrm"][0], tran)
        self.assertGreater(np.unique(tran).size, 1)

    def test_both_off(self):
        agent = ind_shock_agent(NewbornTransShk=False, NewbornPermShk=False)
        first_period(agent)
        np.testing.assert_array_equal(agent.history["pLvl"][0], 1.0)
        np.testing.assert_array_equal(agent.history["mNrm"][0], 1.0)

    def test_survivors_still_draw(self):
        # Only the birth period is affected
        agent = ind_shock_agent(NewbornTransShk=True, NewbornPermShk=False)
        agent.T_sim = 2
        agent.LivPrb = [1.0]
        agent.initialize_sim()
        agent.simulate(2)
        self.assertGreater(np.unique(agent.shocks["PermShk"]).size, 1)

    def test_model_file_simulator_refuses(self):
        agent = ind_shock_agent(NewbornPermShk=False)
        with self.assertRaises(NotImplementedError):
            agent.initialize_sym()


class testNewbornPermShkMarkov(unittest.TestCase):
    def test_off_gives_newborns_pLvl_of_exactly_one(self):
        params = deepcopy(init_indshk_markov)
        params["MrkvArray"] = [np.array([[0.9, 0.1], [0.2, 0.8]])]
        params["constructors"] = dict(params["constructors"])
        params["constructors"]["MrkvArray"] = None
        params.update(NEWBORN)
        params["PermGroFac"] = [np.array([1.01, 1.02])]
        params.update(
            AgentCount=500, T_sim=1, seed=11, NewbornTransShk=True, NewbornPermShk=False
        )
        agent = MarkovConsumerType(**params)
        agent.cycles = 0
        agent.IncShkDstn = [[agent.IncShkDstn[0][0]] * 2]
        agent.track_vars = ["pLvl", "mNrm"]
        agent.solve()
        perm, tran = first_period(agent)
        np.testing.assert_array_equal(perm, 1.0)
        np.testing.assert_array_equal(agent.history["pLvl"][0], 1.0)
        np.testing.assert_array_equal(agent.history["mNrm"][0], tran)


class DualIndShock(DualMeasureMixin, IndShockConsumerType):
    pass


class testNewbornPermShkDualMeasure(unittest.TestCase):
    def test_q_measure_newborns_follow_the_flag(self):
        agent = DualIndShock(
            cycles=0,
            AgentCount=500,
            T_sim=1,
            seed=13,
            NewbornTransShk=True,
            NewbornPermShk=False,
            **NEWBORN,
        )
        agent.track_vars = ["pLvl"]
        agent.solve()
        agent.setup_Q_measure()
        agent.initialize_sim()
        agent.simulate(1)
        np.testing.assert_array_equal(agent.shocks["PermShk"], 1.0)
        np.testing.assert_array_equal(agent.shocks_Q["PermShk"], 1.0)
