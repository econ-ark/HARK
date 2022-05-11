from HARK.ConsumptionSaving.ConsIndShockModel import KinkedRconsumerType
import numpy as np
import unittest


class testKinkedRConsumerType(unittest.TestCase):
    def test_liquidity_constraint(self):

        KinkyExample = KinkedRconsumerType()
        KinkyExample.cycles = 0

        # The consumer cannot borrow more than 0.4
        # times their permanent income
        KinkyExample.BoroCnstArt = -0.4

        # Solve the consumer's problem
        KinkyExample.solve()

        decimalPlacesTo = 2 # tolerance of this group of tests
        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(1).tolist(), 0.9623337593984276, decimalPlacesTo
        )

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(4).tolist(), 1.343766303734248, decimalPlacesTo
        )

        KinkyExample.BoroCnstArt = -0.2
        KinkyExample.solve()

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(1).tolist(), 0.9346895908550565, decimalPlacesTo
        )

        self.assertAlmostEqual(
            KinkyExample.solution[0].cFunc(4).tolist(), 1.3401428646781697, decimalPlacesTo
        )
