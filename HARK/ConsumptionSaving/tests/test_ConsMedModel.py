from HARK.utilities import CRRAutility_inv
import numpy as np
from HARK.ConsumptionSaving.ConsMedModel import MedShockConsumerType
import unittest


class testMedShockConsumerType(unittest.TestCase):
    def setUp(self):

        self.agent = MedicalExample = MedShockConsumerType()

    def test_solution(self):

        self.agent.solve()

        self.agent.T_sim = 10
        self.agent.track_vars = ["mLvl", "cLvl", "Med"]
        self.agent.make_shock_history()
        self.agent.initialize_sim()
        self.agent.simulate()
