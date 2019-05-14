"""
This file tests whether ConsIndShockModel's are initialized correctly.
"""


# Bring in modules we need
import unittest
import numpy as np
import HARK.ConsumptionSaving.ConsumerParameters as Params
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.utilities import plotFuncsDer, plotFuncs


class testsForConsIndShockModelInitialization(unittest.TestCase):
    # We don't need a setUp method for the tests to run, but it's convenient
    # if we want to test various things on the same model in different test_*
    # methods.
    def setUp(self):

        # Make and solve an idiosyncratic shocks consumer with a finite lifecycle
        LifecycleExample = IndShockConsumerType(**Params.init_lifecycle)
        self.model = LifecycleExample

    def test_LifecycleIncomeProcess(self):
        self.assertEqual(len(self.model.IncomeDstn), self.model.T_cycle)
