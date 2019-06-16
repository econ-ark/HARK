"""
This file implements unit tests for the simulate method.
"""

# Bring in modules we need
import unittest

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
import HARK.ConsumptionSaving.ConsumerParameters as param

class testsForIndShk(unittest.TestCase):
    def setUp(self):
        pars = param.init_lifecycle
        self.model = IndShockConsumerType(**pars)
    def test_no_solve(self):
        model = self.model
        # Correctly assign time variables
        model.T_sim = 4
        model.t_cycle = 0
        model.t_age=18
        # But forget to solve, and go straight to simulate
        self.assertRaises(Exception, model.simulate())
