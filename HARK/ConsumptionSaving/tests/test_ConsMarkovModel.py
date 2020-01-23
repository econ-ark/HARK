import numpy as np
from HARK import MarkovConsumerType
import HARK.ConsumptionSaving.ConsumerParameters as Params
from copy import copy
import unittest

class test_ConsMarkovSolver(unittest.TestCase):
    def setUp(self):

        # Define the Markov transition matrix for serially correlated unemployment
        unemp_length = 5  # Averange length of unemployment spell
        urate_good = 0.05  # Unemployment rate when economy is in good state
        urate_bad = 0.12  # Unemployment rate when economy is in bad state
        bust_prob = 0.01  # Probability of economy switching from good to bad
        recession_length = 20  # Averange length of bad state
        p_reemploy = 1.0 / unemp_length
        p_unemploy_good = p_reemploy * urate_good / (1 - urate_good)
        p_unemploy_bad = p_reemploy * urate_bad / (1 - urate_bad)
        boom_prob = 1.0 / recession_length
        self.MrkvArray = np.array(
            [
                [
                    (1 - p_unemploy_good) * (1 - bust_prob),
                    p_unemploy_good * (1 - bust_prob),
                    (1 - p_unemploy_good) * bust_prob,
                    p_unemploy_good * bust_prob,
                ],
                [
                    p_reemploy * (1 - bust_prob),
                    (1 - p_reemploy) * (1 - bust_prob),
                    p_reemploy * bust_prob,
                    (1 - p_reemploy) * bust_prob,
                ],
                [
                    (1 - p_unemploy_bad) * boom_prob,
                    p_unemploy_bad * boom_prob,
                    (1 - p_unemploy_bad) * (1 - boom_prob),
                    p_unemploy_bad * (1 - boom_prob),
                ],
                [
                    p_reemploy * boom_prob,
                    (1 - p_reemploy) * boom_prob,
                    p_reemploy * (1 - boom_prob),
                    (1 - p_reemploy) * (1 - boom_prob),
                ],
            ]
        )

        init_serial_unemployment = copy(Params.init_idiosyncratic_shocks)
        init_serial_unemployment["MrkvArray"] = [self.MrkvArray]       
        self.model = MarkovConsumerType(**init_serial_unemployment) 
    
    def test_checkMarkovInputs(self):
        # check Rfree
        self.assertRaises(ValueError, self.model.checkMarkovInputs)
        # fix Rfree
        self.model.Rfree = np.array(4 * [self.model.Rfree])
        # check MrkvArray, first mess up the setup
        self.model.MrkvArray = np.random.rand(3, 3)
        self.assertRaises(ValueError, self.model.checkMarkovInputs)
        # then fix it back
        self.model.MrkvArray = self.MrkvArray
        # check LivPrb
        self.assertRaises(ValueError, self.model.checkMarkovInputs)
        # fix LivPrb
        self.model.LivPrb = [np.array(4 * self.model.LivPrb)]
        # check PermGroFac
        self.assertRaises(ValueError, self.model.checkMarkovInputs)
        # fix PermGroFac
        self.model.PermGroFac = [np.array(4 * self.model.PermGroFac)]
