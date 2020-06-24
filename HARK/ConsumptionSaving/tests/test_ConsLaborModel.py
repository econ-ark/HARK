from HARK.ConsumptionSaving.ConsLaborModel import (
    LaborIntMargConsumerType,
    init_labor_lifecycle,
)
import unittest


class test_LaborIntMargConsumerType(unittest.TestCase):
    def setUp(self):
        self.model = LaborIntMargConsumerType()
        self.model_finte_lifecycle = LaborIntMargConsumerType(**init_labor_lifecycle)
        self.model_finte_lifecycle.cycles = 1

    def test_solution(self):
        self.model.solve()
        self.model_finte_lifecycle.solve()

        self.model.T_sim = 120
        self.model.track_vars = ["bNrmNow", "cNrmNow"]
        self.model.initializeSim()
        self.model.simulate()
