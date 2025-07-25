import unittest
from copy import copy

import numpy as np

from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    IndShockExplicitPermIncConsumerType,
    PersistentShockConsumerType,
)
from tests import HARK_PRECISION

GenIncDictionary = {
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability
    "AgentCount": 10000,  # Number of agents of this type (only matters for simulation)
    "kNrmInitMean": 0.0,  # Mean of log initial assets (only matters for simulation)
    "kNrmInitStd": 1.0,  # Standard deviation of log initial assets (only for simulation)
    "pLogInitMean": 0.0,  # Mean of log initial permanent income (only matters for simulation)
    "pLogInitStd": 0.4,  # Standard deviation of log initial permanent income (only matters for simulation)
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor (only matters for simulation)
    "T_age": None,  # Age after which simulated agents are automatically killed
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 30,  # Maximum end-of-period "assets above minimum" value
    # Some other value of "assets above minimum" to add to the grid
    "aXtraExtra": np.array([0.005, 0.01]),
    "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    # Parameters describing the income process
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
    "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "tax_rate": 0.0,  # Flat income tax rate
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "CubicBool": False,  # Use cubic spline interpolation when True, linear interpolation when False
    "vFuncBool": True,  # Whether to calculate the value function during solution
    # More parameters specific to "Explicit Permanent income" shock model
    "PermGroFac": [
        1.0
    ],  # Permanent income growth factor - long run permanent income growth doesn't work yet
}


class testIndShockExplicitPermIncConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockExplicitPermIncConsumerType(cycles=1, **GenIncDictionary)
        self.agent.solve()

    def test_solution(self):
        pLvlGrid = self.agent.pLvlGrid[0]
        self.assertAlmostEqual(
            self.agent.pLvlGrid[0][0], 0.28063, places=HARK_PRECISION
        )

        self.assertAlmostEqual(self.agent.solution[0].mLvlMin(pLvlGrid[0]), 0.0)

        self.assertAlmostEqual(
            self.agent.solution[0].cFunc(10, pLvlGrid[5]).tolist(),
            5.408106,
            places=HARK_PRECISION,
        )


class testPersistentShockConsumerType(unittest.TestCase):
    def setUp(self):
        # "persistent idiosyncratic shocks" model
        PrstIncCorr = 0.98  # Serial correlation coefficient for persistent income
        persistent_shocks = copy(GenIncDictionary)
        persistent_shocks["PrstIncCorr"] = PrstIncCorr

        # "persistent idisyncratic shocks" consumer
        self.agent = PersistentShockConsumerType(cycles=1, **persistent_shocks)
        self.agent.solve()

    def test_solution(self):
        pLvlGrid = self.agent.pLvlGrid[0]

        self.assertAlmostEqual(
            self.agent.solution[0].cFunc(10, pLvlGrid[1]).tolist(),
            5.27723,
            places=HARK_PRECISION,
        )

    def test_value(self):
        pLvlGrid = self.agent.pLvlGrid[0]

        self.assertTrue(self.agent.vFuncBool)
        self.assertAlmostEqual(
            self.agent.solution[0].vFunc(10, pLvlGrid[3]),
            -0.36683,
            places=HARK_PRECISION,
        )

    def test_simulation(self):
        self.agent.T_sim = 25

        self.agent.track_vars = ["aLvl", "mLvl", "cLvl", "pLvl"]
        self.agent.initialize_sim()
        self.agent.simulate()

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(np.mean(self.agent.history["mLvl"]), 1.20439, places=HARK_PRECISION)
