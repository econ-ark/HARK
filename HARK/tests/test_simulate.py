"""
This file implements unit tests for the simulate method.
"""

# Bring in modules we need
import unittest

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_lifecycle,
)


class testsForIndShk(unittest.TestCase):
    def setUp(self):
        self.model = IndShockConsumerType(**init_lifecycle)

    def test_no_solve(self):
        model = self.model
        # Correctly assign time variables
        model.T_sim = 4
        model.t_cycle = 0
        model.t_age = 18
        # But forget to solve, and go straight to simulate
        with self.assertRaises(Exception):
            model.simulate()
