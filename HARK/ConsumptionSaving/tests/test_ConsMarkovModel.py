import numpy as np
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.distribution import DiscreteDistribution
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
        MrkvArray = np.array(
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

        init_serial_unemployment = copy(init_idiosyncratic_shocks)
        init_serial_unemployment["MrkvArray"] = [MrkvArray]       
        init_serial_unemployment["UnempPrb"] = 0  # to make income distribution when employed
        init_serial_unemployment["global_markov"] = False
        self.model = MarkovConsumerType(**init_serial_unemployment)
        self.model.cycles = 0
        self.model.vFuncBool = False  # for easy toggling here

        # Replace the default (lognormal) income distribution with a custom one
        employed_income_dist = DiscreteDistribution(np.ones(1), [np.ones(1), np.ones(1)])  # Definitely get income
        unemployed_income_dist = DiscreteDistribution(np.ones(1), [np.ones(1), np.zeros(1)]) # Definitely don't
        self.model.IncomeDstn = [
            [
                employed_income_dist,
                unemployed_income_dist,
                employed_income_dist,
                unemployed_income_dist,
            ]
        ]

    def test_checkMarkovInputs(self):
        # check Rfree
        self.assertRaises(ValueError, self.model.checkMarkovInputs)
        # fix Rfree
        self.model.Rfree = np.array(4 * [self.model.Rfree])
        # check MrkvArray, first mess up the setup
        self.MrkvArray = self.model.MrkvArray
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

    def test_solve(self):
        self.model.Rfree = np.array(4 * [self.model.Rfree])
        self.model.LivPrb = [np.array(4 * self.model.LivPrb)]
        self.model.PermGroFac = [np.array(4 * self.model.PermGroFac)]
        self.model.solve()

    def test_simulation(self):
        self.model.Rfree = np.array(4 * [self.model.Rfree])
        self.model.LivPrb = [np.array(4 * self.model.LivPrb)]
        self.model.PermGroFac = [np.array(4 * self.model.PermGroFac)]
        self.model.solve()
        self.model.T_sim = 120
        self.model.MrkvPrbsInit = [0.25, 0.25, 0.25, 0.25]
        self.model.track_vars = ["mNrmNow", "cNrmNow"]
        self.model.makeShockHistory()  # This is optional
        self.model.initializeSim()
        self.model.simulate()
