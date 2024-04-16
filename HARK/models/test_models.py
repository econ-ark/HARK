from HARK.distribution import Lognormal
import HARK.models.fisher as fm
import HARK.models.perfect_foresight as pfm
import HARK.models.perfect_foresight_normalized as pfnm
from HARK.simulation.monte_carlo import AgentTypeMonteCarloSimulator

from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType

import unittest

PFexample = PerfForesightConsumerType()
PFexample.cycles = 0

SimulationParams = {
    "AgentCount": 3,  # Number of agents of this type
    "T_sim": 120,  # Number of periods to simulate
    "aNrmInitMean": -6.0,  # Mean of log initial assets
    "aNrmInitStd": 0,  # 1.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed,
    "LivPrb": [0.98],
}

PFexample.assign_parameters(**SimulationParams)
PFexample.solve()

class test_pfm(unittest.TestCase):
    def setUp(self):
        self.mcs = AgentTypeMonteCarloSimulator(
            pfm.block,
            {"c": lambda m: PFexample.solution[0].cFunc(m)},
            # danger: normalized decision rule for unnormalized problem
            {  # initial states
                "a": Lognormal(-6, 0),
                #'live' : 1,
                "p": 1.0,
            },
            agent_count=3,
            T_sim=120,
        )

    def test_simulate(self):
        ## smoke test
        self.mcs.initialize_sim()
        self.mcs.simulate()

class test_pfnm(unittest.TestCase):
    def setUp(self):
        self.mcs = AgentTypeMonteCarloSimulator( ### Use fm, blockified
            pfnm.block,
            {"c_nrm": lambda m_nrm: PFexample.solution[0].cFunc(m_nrm)},
            {  # initial states
                "a_nrm": Lognormal(-6, 0),
                #'live' : 1,
                "p": 1.0,
            },
            agent_count=3,
            T_sim=120,
        )

    def test_simulate(self):
        ## smoke test
        self.mcs.initialize_sim()
        self.mcs.simulate()