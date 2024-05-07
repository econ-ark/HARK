import unittest

from HARK.ConsumptionSaving.ConsLaborModel import (
    LaborIntMargConsumerType,
    init_labor_lifecycle,
)


class test_LaborIntMargConsumerType(unittest.TestCase):
    def setUp(self):
        self.model = LaborIntMargConsumerType()
        self.model_finite_lifecycle = LaborIntMargConsumerType(**init_labor_lifecycle)
        self.model_finite_lifecycle.cycles = 1

    def test_solution(self):
        self.model.solve()
        self.model_finite_lifecycle.solve()

        self.model.T_sim = 120
        self.model.track_vars = ["bNrm", "cNrm"]
        self.model.initialize_sim()
        self.model.simulate()
