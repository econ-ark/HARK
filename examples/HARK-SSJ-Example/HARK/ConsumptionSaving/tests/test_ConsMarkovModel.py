import unittest
from copy import copy

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.distribution import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    MeanOneLogNormal,
    combine_indep_dstns,
)


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
        init_serial_unemployment[
            "UnempPrb"
        ] = 0.0  # to make income distribution when employed
        init_serial_unemployment["global_markov"] = False
        self.model = MarkovConsumerType(**init_serial_unemployment)
        self.model.cycles = 0
        self.model.vFuncBool = False  # for easy toggling here

        # Replace the default (lognormal) income distribution with a custom one
        employed_income_dist = DiscreteDistributionLabeled(
            pmv=np.ones(1),
            atoms=np.array([[1.0], [1.0]]),
            var_names=["PermShk", "TranShk"],
        )  # Definitely get income
        unemployed_income_dist = DiscreteDistributionLabeled(
            pmv=np.ones(1),
            atoms=np.array([[1.0], [0.0]]),
            var_names=["PermShk", "TranShk"],
        )  # Definitely don't
        self.model.IncShkDstn = [
            [
                employed_income_dist,
                unemployed_income_dist,
                employed_income_dist,
                unemployed_income_dist,
            ]
        ]

    def test_check_markov_inputs(self):
        # check Rfree
        self.assertRaises(ValueError, self.model.check_markov_inputs)
        # fix Rfree
        self.model.Rfree = np.array(4 * [self.model.Rfree])
        # check MrkvArray, first mess up the setup
        self.MrkvArray = self.model.MrkvArray
        self.model.MrkvArray = np.random.rand(3, 3)
        self.assertRaises(ValueError, self.model.check_markov_inputs)
        # then fix it back
        self.model.MrkvArray = self.MrkvArray
        # check LivPrb
        self.assertRaises(ValueError, self.model.check_markov_inputs)
        # fix LivPrb
        self.model.LivPrb = [np.array(4 * self.model.LivPrb)]
        # check PermGroFac
        self.assertRaises(ValueError, self.model.check_markov_inputs)
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
        self.model.track_vars = ["mNrm", "cNrm"]
        self.model.make_shock_history()  # This is optional
        self.model.initialize_sim()
        self.model.simulate()


Markov_Dict = {
    # Parameters shared with the perfect foresight model
    # Coefficient of relative risk aversion
    "CRRA": 2,
    "Rfree": np.array([1.05**0.25] * 2),  # Interest factor on assets
    "DiscFac": 0.985,  # Intertemporal discount factor
    "LivPrb": [np.array([0.99375] * 2)],  # Survival probability
    # Permanent income growth factor
    "PermGroFac": [np.array([1.00] * 2)],
    # Parameters that specify the income distribution over the lifecycle
    # Standard deviation of log permanent shocks to income
    "PermShkStd": [0.06],
    # Number of points in discrete approximation to permanent income shocks
    "PermShkCount": 7,
    # Standard deviation of log transitory shocks to income
    "TranShkStd": [0.3],
    # Number of points in discrete approximation to transitory income shocks
    "TranShkCount": 7,
    "UnempPrb": 0.00,  # .08                        # Probability of unemployment while working
    # Unemployment benefits replacement rate
    "IncUnemp": 0.0,
    # Probability of "unemployment" while retired
    "UnempPrbRet": 0.0005,
    # "Unemployment" benefits when retired
    "IncUnempRet": 0.0,
    # Period of retirement (0 --> no retirement)
    "T_retire": 0.0,
    # Flat income tax rate (legacy parameter, will be removed in future)
    "tax_rate": 0.0,
    # Parameters for constructing the "assets above minimum" grid
    # Minimum end-of-period "assets above minimum" value
    "aXtraMin": 0.001,
    # Maximum end-of-period "assets above minimum" value
    "aXtraMax": 60,
    # Number of points in the base grid of "assets above minimum"
    "aXtraCount": 60,
    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraNestFac": 4,
    # Additional values to add to aXtraGrid
    "aXtraExtra": [None],
    # A few other parameters
    # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "BoroCnstArt": 0.0,
    # Whether to calculate the value function during solution
    "vFuncBool": True,
    # Preference shocks currently only compatible with linear cFunc
    "CubicBool": False,
    # Number of periods in the cycle for this agent type
    "T_cycle": 1,
    # Parameters only used in simulation
    "AgentCount": 100000,  # Number of agents of this type
    "T_sim": 200,  # Number of periods to simulate
    # Mean of log initial assets
    "aNrmInitMean": np.log(0.8) - (0.5**2) / 2,
    # Standard deviation of log initial assets
    "aNrmInitStd": 0.5,
    # Mean of log initial permanent income
    "pLvlInitMean": 0.0,
    # Standard deviation of log initial permanent income
    "pLvlInitStd": 0.0,
    # Aggregate permanent income growth factor
    "PermGroFacAgg": 1.0,
    # Age after which simulated agents are automatically killed
    "T_age": None,
    # markov array
    "MrkvArray": [np.array([[0.984, 0.856], [0.0152, 0.14328]]).T],
}


class test_make_EndOfPrdvFuncCond(unittest.TestCase):
    def main_test(self):
        Markov_vFuncBool_example = MarkovConsumerType(**Markov_Dict)

        TranShkDstn_e = MeanOneLogNormal(
            Markov_vFuncBool_example.TranShkStd[0], 123
        ).discretize(Markov_vFuncBool_example.TranShkCount, method="equiprobable")
        TranShkDstn_u = DiscreteDistribution(np.ones(1), np.ones(1) * 0.2)
        PermShkDstn = MeanOneLogNormal(
            Markov_vFuncBool_example.PermShkStd[0], 123
        ).discretize(Markov_vFuncBool_example.PermShkCount, method="equiprobable")

        # employed Income shock distribution
        employed_IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn_e)

        # unemployed Income shock distribution
        unemployed_IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn_u)

        # Specify list of IncShkDstns for each state
        Markov_vFuncBool_example.IncShkDstn = [
            [employed_IncShkDstn, unemployed_IncShkDstn]
        ]

        # solve the consumer's problem
        Markov_vFuncBool_example.solve()

        self.assertAlmostEqual(
            Markov_vFuncBool_example.solution[0].vFunc[1](0.4), -4.12794
        )
