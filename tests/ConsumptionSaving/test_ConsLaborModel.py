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

    def test_simulation(self):
        self.model.solve()
        self.model.T_sim = 120
        self.model.track_vars = ["bNrm", "cNrm"]
        self.model.initialize_sim()
        self.model.simulate()

    def test_plotting(self):
        self.model.solve()
        self.model.plot_cFunc(0)
        self.model.plot_LbrFunc(0)

    def test_invalid_parameters(self):
        BadType = LaborIntMargConsumerType(CRRA=0.1)
        self.assertRaises(ValueError, BadType.solve)

        BadType = LaborIntMargConsumerType(BoroCnstArt=0.0)
        self.assertRaises(ValueError, BadType.solve)

        BadType = LaborIntMargConsumerType(CubicBool=True)
        self.assertRaises(ValueError, BadType.solve)

        BadType = LaborIntMargConsumerType(vFuncBool=True)
        self.assertRaises(ValueError, BadType.solve)
