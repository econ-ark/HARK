"""
This file tests whether HARK's models are initialized correctly.
"""


# Bring in modules we need
import unittest
import numpy as np
import HARK.ConsumptionSaving.ConsumerParameters as Params
from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import KinkedRconsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.utilities import plotFuncsDer, plotFuncs
from copy import copy

class testInitialization(unittest.TestCase):
    # We don't need a setUp method for the tests to run, but it's convenient
    # if we want to test various things on the same model in different test_*
    # methods.
    def test_PerfForesightConsumerType(self):
        try:
            model = PerfForesightConsumerType()
        except:
            self.fail("PerfForesightConsumerType failed to initialize with Params.init_perfect_foresight.")

    def test_IndShockConsumerType(self):
        try:
            model = IndShockConsumerType(**Params.init_lifecycle)
        except:
            self.fail("IndShockConsumerType failed to initialize with Params.init_lifecycle.")

    def test_KinkedRconsumerType(self):
        try:
            model = KinkedRconsumerType(**Params.init_kinked_R)
        except:
            self.fail("KinkedRconsumerType failed to initialize with Params.init_kinked_R.")

    def test_MarkovConsumerType(self):
        try:
            unemp_length = 5 # Averange length of unemployment spell
            urate_good = 0.05        # Unemployment rate when economy is in good state
            urate_bad = 0.12         # Unemployment rate when economy is in bad state
            bust_prob = 0.01         # Probability of economy switching from good to bad
            recession_length = 20    # Averange length of bad state
            p_reemploy =1.0/unemp_length
            p_unemploy_good = p_reemploy*urate_good/(1-urate_good)
            p_unemploy_bad = p_reemploy*urate_bad/(1-urate_bad)
            boom_prob = 1.0/recession_length
            MrkvArray = np.array([[(1-p_unemploy_good)*(1-bust_prob),p_unemploy_good*(1-bust_prob),
                                   (1-p_unemploy_good)*bust_prob,p_unemploy_good*bust_prob],
                                  [p_reemploy*(1-bust_prob),(1-p_reemploy)*(1-bust_prob),
                                   p_reemploy*bust_prob,(1-p_reemploy)*bust_prob],
                                  [(1-p_unemploy_bad)*boom_prob,p_unemploy_bad*boom_prob,
                                   (1-p_unemploy_bad)*(1-boom_prob),p_unemploy_bad*(1-boom_prob)],
                                  [p_reemploy*boom_prob,(1-p_reemploy)*boom_prob,
                                   p_reemploy*(1-boom_prob),(1-p_reemploy)*(1-boom_prob)]])

            # Make a consumer with serially correlated unemployment, subject to boom and bust cycles
            init_serial_unemployment = copy(Params.init_idiosyncratic_shocks)
            init_serial_unemployment['MrkvArray'] = [MrkvArray]
            init_serial_unemployment['UnempPrb'] = 0 # to make income distribution when employed
            init_serial_unemployment['global_markov'] = False
            SerialUnemploymentExample = MarkovConsumerType(**init_serial_unemployment)
        except:
            self.fail("MarkovConsumerType failed to initialize with boom/bust unemployment.")
