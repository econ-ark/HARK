import unittest
from copy import deepcopy

import numpy as np

from HARK.ConsumptionSaving.ConsMarkovModel import (
    MarkovConsumerType,
    init_indshk_markov,
    make_ratchet_markov,
)
from HARK.distributions import (
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

        init_serial_unemployment = deepcopy(init_indshk_markov)
        init_serial_unemployment["MrkvArray"] = [MrkvArray]
        init_serial_unemployment["UnempPrb"] = np.zeros(2)
        # Income process is overwritten below to make income distribution when employed
        init_serial_unemployment["global_markov"] = False
        init_serial_unemployment["Rfree"] = [np.array([1.03, 1.03, 1.03, 1.03])]
        init_serial_unemployment["LivPrb"] = [np.array([0.98, 0.98, 0.98, 0.98])]
        init_serial_unemployment["PermGroFac"] = [np.array([1.01, 1.01, 1.01, 1.01])]
        init_serial_unemployment["constructors"]["MrkvArray"] = None

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
        self.model.Rfree = [1.03]
        self.assertRaises(ValueError, self.model.check_markov_inputs)
        # fix Rfree
        self.model.Rfree = [np.array(4 * self.model.Rfree)]
        # check MrkvArray, first mess up the setup
        self.MrkvArray = self.model.MrkvArray
        self.model.MrkvArray = np.random.rand(3, 3)
        self.assertRaises(ValueError, self.model.check_markov_inputs)
        # then fix it back
        self.model.MrkvArray = self.MrkvArray
        # check LivPrb
        self.model.LivPrb = [0.98]
        self.assertRaises(ValueError, self.model.check_markov_inputs)
        # fix LivPrb
        self.model.LivPrb = [np.array(4 * self.model.LivPrb)]
        # check PermGroFac
        self.model.PermGroFac = [1.01]
        self.assertRaises(ValueError, self.model.check_markov_inputs)
        # fix PermGroFac
        self.model.PermGroFac = [np.array(4 * self.model.PermGroFac)]

    def test_solve(self):
        self.model.solve()

    def test_simulation(self):
        self.model.solve()
        self.model.T_sim = 120
        self.model.T_age = 100
        self.model.MrkvPrbsInit = [0.25, 0.25, 0.25, 0.25]
        self.model.track_vars = ["mNrm", "cNrm"]
        self.model.make_shock_history()  # This is optional
        self.model.initialize_sim()
        self.model.simulate()

    def test_global_markov(self):
        self.model.solve()
        self.model.assign_parameters(
            T_sim=120, global_markov=True, MrkvPrbsInit=[0.25, 0.25, 0.25, 0.25]
        )
        self.model.track_vars = ["mNrm", "cNrm"]
        self.model.initialize_sim()
        self.model.simulate()


Markov_Dict = {
    # Parameters shared with the perfect foresight model
    # Coefficient of relative risk aversion
    "CRRA": 2,
    "Rfree": [np.array([1.05**0.25] * 2)],  # Interest factor on assets
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


class testMarkovValueFunc(unittest.TestCase):
    def test_vFunc(self):
        agent = MarkovConsumerType(cycles=0, vFuncBool=True)
        agent.solve()
        self.assertAlmostEqual(agent.solution[0].vFunc[0](5.0), -30.78459, places=4)
        self.assertAlmostEqual(agent.solution[0].vFunc[1](5.0), -30.37644, places=4)


class testRatchet(unittest.TestCase):
    def test_ratchet_markov(self):
        some_probs = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.3, 0.2, 0.1])]
        MrkvArray = make_ratchet_markov(2, some_probs)
        self.assertAlmostEqual(MrkvArray[0][0, 1], 0.1)
        self.assertAlmostEqual(MrkvArray[0][3, 3], 0.6)
        self.assertAlmostEqual(MrkvArray[1][0, 1], 0.4)
        self.assertAlmostEqual(MrkvArray[1][3, 3], 0.9)

    def test_errors(self):
        some_probs = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.3, 0.2, 0.1])]
        self.assertRaises(ValueError, make_ratchet_markov, 3, some_probs)

        weird_probs = [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.3, 0.3])]
        self.assertRaises(ValueError, make_ratchet_markov, 2, weird_probs)


class testMarkovTransitionMatrix(unittest.TestCase):
    """Tests for the transition-matrix methods on MarkovConsumerType."""

    def _make_2state_agent(self):
        """Create a simple symmetric 2-state Markov agent for testing."""
        params = deepcopy(init_indshk_markov)
        params["Mrkv_p11"] = [0.9]
        params["Mrkv_p22"] = [0.9]
        params["Rfree"] = [np.array([1.03, 1.03])]
        params["LivPrb"] = [np.array([0.98, 0.98])]
        params["PermGroFac"] = [np.array([1.0, 1.0])]
        params["cycles"] = 0
        agent = MarkovConsumerType(**params)
        return agent

    def test_column_sums_2state(self):
        """TM column sums must equal 1.0 for a 2-state model."""
        agent = self._make_2state_agent()
        agent.solve()
        agent.neutral_measure = True
        agent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")
        agent.define_distribution_grid()
        agent.calc_transition_matrix()
        col_sums = agent.tran_matrix.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_column_sums_4state(self):
        """TM column sums must equal 1.0 for a 4-state model."""
        unemp_length = 5
        urate_good = 0.05
        urate_bad = 0.12
        bust_prob = 0.01
        recession_length = 20
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

        params = deepcopy(init_indshk_markov)
        params["MrkvArray"] = [MrkvArray]
        params["constructors"] = deepcopy(params["constructors"])
        params["constructors"]["MrkvArray"] = None
        params["Rfree"] = [np.array([1.03] * 4)]
        params["LivPrb"] = [np.array([0.98] * 4)]
        params["PermGroFac"] = [np.array([1.0] * 4)]
        params["PermShkStd"] = np.array([[0.1] * 4])
        params["TranShkStd"] = np.array([[0.1] * 4])
        params["UnempPrb"] = np.array([0.05] * 4)
        params["IncUnemp"] = np.array([0.3] * 4)
        params["MrkvPrbsInit"] = np.array([0.25, 0.25, 0.25, 0.25])
        params["cycles"] = 0

        agent = MarkovConsumerType(**params)
        agent.solve()
        agent.neutral_measure = True
        agent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")
        agent.define_distribution_grid()
        agent.calc_transition_matrix()
        col_sums = agent.tran_matrix.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_ergodic_markov_fractions(self):
        """Ergodic Markov state fractions from TM should match analytical stationary dist."""
        agent = self._make_2state_agent()
        agent.solve()
        agent.neutral_measure = True
        agent.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")
        agent.define_distribution_grid()
        agent.calc_transition_matrix()
        agent.calc_ergodic_dist()

        M = len(agent.dist_mGrid)
        pi_0 = np.sum(agent.vec_erg_dstn[:M])
        pi_1 = np.sum(agent.vec_erg_dstn[M:])

        # Symmetric p11=p22=0.9 => stationary distribution is (0.5, 0.5)
        np.testing.assert_allclose(pi_0, 0.5, atol=0.01)
        np.testing.assert_allclose(pi_1, 0.5, atol=0.01)

    def test_j1_matches_nk(self):
        """J=1 Markov TM builder with NK's own policy should match NK TM exactly.

        The Markov and IndShock solvers produce slightly different cFuncs, so we
        feed the NK's policy into gen_tran_matrix_1D_markov and verify that the
        *TM construction* logic is identical.
        """
        from HARK.ConsumptionSaving.ConsNewKeynesianModel import (
            NewKeynesianConsumerType,
        )
        from HARK.utilities import gen_tran_matrix_1D_markov, jump_to_grid_1D

        nk = NewKeynesianConsumerType(cycles=0)
        nk.solve()
        nk.neutral_measure = True
        nk.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")
        nk.define_distribution_grid()
        nk.calc_transition_matrix()

        dist_mGrid = nk.dist_mGrid
        M = len(dist_mGrid)
        aPol = dist_mGrid - nk.solution[0].cFunc(dist_mGrid)
        aPol_2d = aPol.reshape(1, M)

        shk_prbs = nk.IncShkDstn[0].pmv
        perm_shks = nk.IncShkDstn[0].atoms[0]
        tran_shks = nk.IncShkDstn[0].atoms[1]

        newborn_1d = jump_to_grid_1D(np.ones_like(tran_shks), shk_prbs, dist_mGrid)

        markov_tm = gen_tran_matrix_1D_markov(
            dist_mGrid,
            aPol_2d,
            np.array([[1.0]]),
            np.array([nk.Rfree[0]]),
            np.array([nk.PermGroFac[0]]),
            np.array([nk.LivPrb[0]]),
            shk_prbs,
            perm_shks,
            tran_shks,
            newborn_1d,
        )

        np.testing.assert_allclose(markov_tm, nk.tran_matrix, atol=1e-12)

    def test_compute_pe_steady_state(self):
        """compute_pe_steady_state should return finite positive A_ss and C_ss."""
        agent = self._make_2state_agent()
        A_ss, C_ss = agent.compute_pe_steady_state()
        self.assertTrue(np.isfinite(A_ss))
        self.assertTrue(np.isfinite(C_ss))
        self.assertGreater(A_ss, 0.0)
        self.assertGreater(C_ss, 0.0)
