import pickle
import unittest
from copy import copy, deepcopy

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    PerfForesightConsumerType,
    init_idiosyncratic_shocks,
    init_lifecycle,
)
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.distributions.base import MarkovProcess
from HARK.utilities import plot_funcs, plot_funcs_der
from tests import HARK_PRECISION


class testIndShockConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockConsumerType(AgentCount=2, T_sim=10)

        self.agent.solve()

    def test_get_shocks(self):
        self.agent.initialize_sim()
        self.agent.sim_birth(np.array([True, False]))
        self.agent.sim_one_period()
        self.agent.sim_birth(np.array([False, True]))

        self.agent.get_shocks()

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.agent.shocks["PermShk"][0], 1.04274, place = HARK_PRECISION)
        # self.assertAlmostEqual(self.agent.shocks["PermShk"][1], 0.92781, place = HARK_PRECISION)
        # self.assertAlmostEqual(self.agent.shocks["TranShk"][0], 0.88176, place = HARK_PRECISION)

    def test_ConsIndShockSolverBasic(self):
        LifecycleExample = IndShockConsumerType(**init_lifecycle)
        LifecycleExample.cycles = 1
        LifecycleExample.solve()

        # test the solution_terminal
        self.assertAlmostEqual(LifecycleExample.solution[-1].cFunc(2).tolist(), 2)

        self.assertAlmostEqual(
            LifecycleExample.solution[9].cFunc(1), 0.79430, places=HARK_PRECISION
        )
        self.assertAlmostEqual(
            LifecycleExample.solution[8].cFunc(1), 0.79392, places=HARK_PRECISION
        )
        self.assertAlmostEqual(
            LifecycleExample.solution[7].cFunc(1), 0.79253, places=HARK_PRECISION
        )

        self.assertAlmostEqual(
            LifecycleExample.solution[0].cFunc(1).tolist(),
            0.75062,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            LifecycleExample.solution[1].cFunc(1).tolist(),
            0.75863,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            LifecycleExample.solution[2].cFunc(1).tolist(),
            0.76812,
            places=HARK_PRECISION,
        )

        self.assertRaises(ValueError, LifecycleExample.calc_stable_points)

    def test_invalid_calc_stable_points(self):
        TestType = IndShockConsumerType(cycles=0)
        self.assertRaises(ValueError, TestType.calc_stable_points)
        TestType.check_conditions()
        self.assertRaises(ValueError, TestType.calc_stable_points)

    def test_simulated_values(self):
        self.agent.initialize_sim()
        self.agent.simulate()

        # MPCnow depends on assets, which are stochastic
        # self.assertAlmostEqual(self.agent.MPCnow[1], 0.57115, place = HARK_PRECISION)

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(self.agent.state_now["aLvl"][1], 0.18438, place = HARK_PRECISION)

    def test_income_dist_random_seeds(self):
        a1 = IndShockConsumerType(seed=1000)
        a2 = IndShockConsumerType(seed=200)

        self.assertFalse(a1.PermShkDstn.seed == a2.PermShkDstn.seed)

    def test_check_conditions(self):
        TestType = IndShockConsumerType(cycles=0, quiet=False, verbose=False)
        TestType.check_conditions()

        # make DiscFac way too big
        TestType = IndShockConsumerType(cycles=0, DiscFac=1.06)
        TestType.check_conditions()

        # make PermGroFac big
        TestType = IndShockConsumerType(cycles=0, DiscFac=0.96, PermGroFac=[1.1])
        TestType.check_conditions()

        # make Rfree too big
        TestType = IndShockConsumerType(cycles=0, Rfree=[1.1])
        TestType.check_conditions()

        # Make unemployment very likely
        TestType = IndShockConsumerType(
            cycles=0, Rfree=[0.93], IncUnemp=0.0, UnempPrb=0.99
        )
        TestType.check_conditions()

        # Use log utility
        TestType = IndShockConsumerType(cycles=0, CRRA=1.0)
        TestType.check_conditions()

    def test_invalid_beta(self):
        TestType = IndShockConsumerType(DiscFac=-0.1, cycles=0)
        self.assertRaises(ValueError, TestType.solve)

    def test_replicate_sim(self):
        TestType = IndShockConsumerType(cycles=0, seed=12022025, T_sim=100)
        TestType.solve()
        TestType.initialize_sim()
        TestType.simulate()
        A0 = np.mean(TestType.state_now["aLvl"])

        # Make sure a simulation result is replicated when re-run
        TestType.initialize_sim()
        TestType.simulate()
        A1 = np.mean(TestType.state_now["aLvl"])
        self.assertAlmostEqual(A0, A1)


class testBufferStock(unittest.TestCase):
    """Tests of the results of the BufferStock REMARK."""

    def setUp(self):
        # Make a dictionary containing all parameters needed to solve the model
        self.base_params = copy(init_idiosyncratic_shocks)

        # Set the parameters for the baseline results in the paper
        # using the variable values defined in the cell above
        self.base_params["PermGroFac"] = [1.03]
        self.base_params["Rfree"] = [1.04]
        self.base_params["DiscFac"] = 0.96
        self.base_params["CRRA"] = 2.00
        self.base_params["UnempPrb"] = 0.005
        self.base_params["IncUnemp"] = 0.0
        self.base_params["PermShkStd"] = [0.1]
        self.base_params["TranShkStd"] = [0.1]
        self.base_params["LivPrb"] = [1.0]
        self.base_params["CubicBool"] = True
        self.base_params["T_cycle"] = 1
        self.base_params["BoroCnstArt"] = None

    def test_baseEx(self):
        baseEx = IndShockConsumerType(**self.base_params)
        baseEx.cycles = 100  # Make this type have a finite horizon (Set T = 100)

        baseEx.solve()
        baseEx.unpack("cFunc")

        m = np.linspace(0, 9.5, 1000)

        c_m = baseEx.cFunc[0](m)
        c_t1 = baseEx.cFunc[-2](m)
        c_t5 = baseEx.cFunc[-6](m)
        c_t10 = baseEx.cFunc[-11](m)

        self.assertAlmostEqual(c_m[500], 1.40081, places=HARK_PRECISION)
        self.assertAlmostEqual(c_t1[500], 2.92274, places=HARK_PRECISION)
        self.assertAlmostEqual(c_t5[500], 1.73506, places=HARK_PRECISION)
        self.assertAlmostEqual(c_t10[500], 1.49914, places=HARK_PRECISION)
        self.assertAlmostEqual(c_t10[600], 1.61015, places=HARK_PRECISION)
        self.assertAlmostEqual(c_t10[700], 1.71965, places=HARK_PRECISION)

    def test_GICRawFails(self):
        GICRaw_fail_dictionary = dict(self.base_params)
        GICRaw_fail_dictionary["Rfree"] = [1.08]
        GICRaw_fail_dictionary["PermGroFac"] = [1.00]
        GICRaw_fail_dictionary["cycles"] = (
            0  # cycles=0 makes this an infinite horizon consumer
        )

        GICRawFailExample = IndShockConsumerType(**GICRaw_fail_dictionary)

        GICRawFailExample.solve()
        GICRawFailExample.unpack("cFunc")
        m = np.linspace(0, 5, 1000)
        c_m = GICRawFailExample.cFunc[0](m)

        self.assertAlmostEqual(c_m[500], 0.77726, places=HARK_PRECISION)
        self.assertAlmostEqual(c_m[700], 0.83926, places=HARK_PRECISION)

        self.assertFalse(GICRawFailExample.conditions["GICRaw"])

    def test_infinite_horizon(self):
        baseEx_inf = IndShockConsumerType(**self.base_params)
        baseEx_inf.assign_parameters(cycles=0)
        baseEx_inf.solve(verbose=True)
        baseEx_inf.unpack("cFunc")

        m1 = np.linspace(
            1, baseEx_inf.solution[0].mNrmStE, 50
        )  # m1 defines the plot range on the left of target m value (e.g. m <= target m)
        c_m1 = baseEx_inf.cFunc[0](m1)

        self.assertAlmostEqual(c_m1[0], 0.85279, places=HARK_PRECISION)
        self.assertAlmostEqual(c_m1[-1], 1.00363, places=HARK_PRECISION)

        x1 = np.linspace(0, 25, 1000)
        cfunc_m = baseEx_inf.cFunc[0](x1)

        self.assertAlmostEqual(cfunc_m[500], 1.89021, places=HARK_PRECISION)
        self.assertAlmostEqual(cfunc_m[700], 2.15915, places=HARK_PRECISION)

        m = np.linspace(0.001, 8, 1000)

        # Use the HARK method derivative to get the derivative of cFunc, and the values are just the MPC
        MPC = baseEx_inf.cFunc[0].derivative(m)

        self.assertAlmostEqual(MPC[500], 0.08415, places=HARK_PRECISION)
        self.assertAlmostEqual(MPC[700], 0.07173, places=HARK_PRECISION)


IdiosyncDict = {
    # Parameters shared with the perfect foresight model
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability
    "PermGroFac": [1.01],  # Permanent income growth factor
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd": [0.1],  # Standard deviation of log permanent shocks to income
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.2],  # Standard deviation of log transitory shocks to income
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate
    "UnempPrbRet": 0.0005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "tax_rate": 0.0,  # Flat income tax rate (legacy parameter, will be removed in future)
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraCount": 48,  # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra": None,  # Additional values to add to aXtraGrid
    # A few other parameters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 10000,  # Number of agents of this type
    "T_sim": 120,  # Number of periods to simulate
    "kLogInitMean": -6.0,  # Mean of log initial assets
    "kLogInitStd": 1.0,  # Standard deviation of log initial assets
    "pLogInitMean": 0.0,  # Mean of log initial permanent income
    "pLogInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
}


class testIndShockConsumerTypeExample(unittest.TestCase):
    def setUp(self):
        IndShockExample = IndShockConsumerType(**IdiosyncDict)
        IndShockExample.assign_parameters(
            cycles=0
        )  # Make this type have an infinite horizon
        self.IndShockExample = IndShockExample

    def test_infinite_horizon(self):
        IndShockExample = self.IndShockExample
        IndShockExample.solve()

        self.assertAlmostEqual(
            IndShockExample.solution[0].mNrmStE, 1.54882, places=HARK_PRECISION
        )
        # self.assertAlmostEqual(
        #    IndShockExample.solution[0].cFunc.functions[0].x_list[0],
        #    -0.25018,
        #    places=HARK_PRECISION,
        # )
        # This test is commented out because it was trivialized by revisions to the "worst income shock" code.
        # The bottom x value of the unconstrained consumption function will definitely be zero, so this is pointless.

        IndShockExample.track_vars = ["aNrm", "mNrm", "cNrm", "pLvl", "who_dies"]
        IndShockExample.initialize_sim()
        IndShockExample.simulate()

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(        #    IndShockExample.history["mNrm"][0][0], 1.01702, place = HARK_PRECISION        # )

    def test_euler_error_function(self):
        IndShockExample = self.IndShockExample
        IndShockExample.solve()
        IndShockExample.make_euler_error_func()
        self.assertAlmostEqual(
            IndShockExample.eulerErrorFunc(5.0), -5.9e-5, places=HARK_PRECISION
        )

    def test_plotting(self):
        MyType = self.IndShockExample
        MyType.solve()
        MyType.unpack("cFunc")
        plot_funcs(MyType.cFunc, 0.0, 10.0, legend_kwds={"labels": ["cFunc"]})
        plot_funcs(MyType.cFunc[0], 0.0, 10.0)
        plot_funcs_der(MyType.cFunc, 0.0, 10.0, legend_kwds={"labels": ["MPC"]})
        plot_funcs_der(MyType.cFunc[0], 0.0, 10.0)


LifecycleDict = {  # Click arrow to expand this fairly large parameter dictionary
    # Parameters shared with the perfect foresight model
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": 10 * [1.03],  # Interest factor on assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    "PermGroFac": [1.01, 1.01, 1.01, 1.02, 1.02, 1.02, 0.7, 1.0, 1.0, 1.0],
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0, 0, 0],
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0, 0, 0],
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate
    "UnempPrbRet": 0.0005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "T_retire": 7,  # Period of retirement (0 --> no retirement)
    "tax_rate": 0.0,  # Flat income tax rate (legacy parameter, will be removed in future)
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraCount": 48,  # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra": None,  # Additional values to add to aXtraGrid
    # A few other parameters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 10,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 10000,  # Number of agents of this type
    "T_sim": 120,  # Number of periods to simulate
    "kLogInitMean": -6.0,  # Mean of log initial assets
    "kLogInitStd": 1.0,  # Standard deviation of log initial assets
    "pLogInitMean": 0.0,  # Mean of log initial permanent income
    "pLogInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": 11,  # Age after which simulated agents are automatically killed
}


class testIndShockConsumerTypeLifecycle(unittest.TestCase):
    def test_lifecyle(self):
        LifecycleExample = IndShockConsumerType(**LifecycleDict)
        LifecycleExample.cycles = 1
        LifecycleExample.solve(verbose=True)

        self.assertEqual(len(LifecycleExample.solution), 11)

        mMin = np.min(
            [
                LifecycleExample.solution[t].mNrmMin
                for t in range(LifecycleExample.T_cycle)
            ]
        )

        self.assertAlmostEqual(
            LifecycleExample.solution[5].cFunc(3).tolist(),
            2.12998,
            places=HARK_PRECISION,
        )


class testIndShockConsumerTypeLifecycleRfree(unittest.TestCase):
    def test_lifecyleRfree(self):
        Rfree = list(np.linspace(1.02, 1.04, 10))
        LifeCycleRfreeDict = LifecycleDict.copy()
        LifeCycleRfreeDict["Rfree"] = Rfree

        LifecycleRfreeExample = IndShockConsumerType(**LifeCycleRfreeDict)
        LifecycleRfreeExample.cycles = 1
        LifecycleRfreeExample.solve()

        self.assertEqual(len(LifecycleRfreeExample.solution), 11)

        mMin = np.min(
            [
                LifecycleRfreeExample.solution[t].mNrmMin
                for t in range(LifecycleRfreeExample.T_cycle)
            ]
        )


CyclicalDict = {
    # Parameters shared with the perfect foresight model
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": 4 * [1.03],  # Interest factor on assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": 4 * [0.98],  # Survival probability
    "PermGroFac": [1.1, 1.082251, 2.8, 0.3],
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd": [0.1, 0.1, 0.1, 0.1],
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.2, 0.2, 0.2, 0.2],
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate
    "UnempPrbRet": 0.0005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "tax_rate": 0.0,  # Flat income tax rate (legacy parameter, will be removed in future)
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraCount": 48,  # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra": None,  # Additional values to add to aXtraGrid
    # A few other parameters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 4,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 10000,  # Number of agents of this type
    "T_sim": 120,  # Number of periods to simulate
    "kLogInitMean": -6.0,  # Mean of log initial assets
    "kLogInitStd": 1.0,  # Standard deviation of log initial assets
    "pLogInitMean": 0.0,  # Mean of log initial permanent income
    "pLogInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
}


class testIndShockConsumerTypeCyclical(unittest.TestCase):
    def test_cyclical(self):
        CyclicalExample = IndShockConsumerType(**CyclicalDict)
        CyclicalExample.cycles = 0  # Make this consumer type have an infinite horizon
        CyclicalExample.solve()

        self.assertAlmostEqual(
            CyclicalExample.solution[3].cFunc(3).tolist(),
            1.59584,
            places=HARK_PRECISION,
        )

        CyclicalExample.initialize_sim()
        CyclicalExample.simulate()

        self.assertAlmostEqual(
            CyclicalExample.state_now["aLvl"][1], 0.55127, places=HARK_PRECISION
        )

        self.assertRaises(ValueError, CyclicalExample.calc_stable_points)


# %% Tests of 'stable points'


# Create the base infinite horizon parametrization from the "Buffer Stock
# Theory" paper.
bst_params = copy(init_idiosyncratic_shocks)
bst_params["PermGroFac"] = [1.03]  # Permanent income growth factor
bst_params["Rfree"] = [1.04]  # Interest factor on assets
bst_params["DiscFac"] = 0.96  # Time Preference Factor
bst_params["CRRA"] = 2.00  # Coefficient of relative risk aversion
# Probability of unemployment (e.g. Probability of Zero Income in the paper)
bst_params["UnempPrb"] = 0.005
bst_params["IncUnemp"] = 0.0  # Induces natural borrowing constraint
bst_params["PermShkStd"] = [0.1]  # Standard deviation of log permanent income shocks
bst_params["TranShkStd"] = [0.1]  # Standard deviation of log transitory income shocks
bst_params["LivPrb"] = [1.0]  # 100 percent probability of living to next period
bst_params["CubicBool"] = True  # Use cubic spline interpolation
bst_params["T_cycle"] = 1  # No 'seasonal' cycles
bst_params["BoroCnstArt"] = None  # No artificial borrowing constraint


class testStablePoints(unittest.TestCase):
    def test_IndShock_stable_points(self):
        # Test for the target and individual steady state of the infinite
        # horizon solution using the parametrization in the "Buffer Stock
        # Theory" paper.

        # Create and solve the agent
        baseAgent_Inf = IndShockConsumerType(verbose=0, **bst_params)
        baseAgent_Inf.assign_parameters(cycles=0)
        baseAgent_Inf.solve()

        # Extract stable points
        mNrmStE = baseAgent_Inf.solution[0].mNrmStE
        mNrmTrg = baseAgent_Inf.solution[0].mNrmTrg

        # Check against pre-computed values
        self.assertAlmostEqual(mNrmStE, 1.37731, places=HARK_PRECISION)
        self.assertAlmostEqual(mNrmTrg, 1.39102, places=HARK_PRECISION)


JACDict = {
    # Parameters shared with the perfect foresight model
    "CRRA": 2,  # Coefficient of relative risk aversion
    "Rfree": [1.05**0.25],  # Interest factor on assets
    "DiscFac": 0.972,  # Intertemporal discount factor
    "LivPrb": [0.99375],  # Survival probability
    "PermGroFac": [1.00],  # Permanent income growth factor
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd": [
        (0.01 * 4 / 11) ** 0.5
    ],  # Standard deviation of log permanent shocks to income
    "PermShkCount": 5,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.2],  # Standard deviation of log transitory shocks to income
    "TranShkCount": 5,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "IncUnemp": 0.1,  # Unemployment benefits replacement rate
    "UnempPrbRet": 0.0005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "tax_rate": 0.2,  # Flat income tax rate (legacy parameter, will be removed in future)
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 15,  # Maximum end-of-period "assets above minimum" value
    "aXtraCount": 48,  # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra": None,  # Additional values to add to aXtraGrid
    # A few other parameters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 5000,  # Number of agents of this type
    "T_sim": 100,  # Number of periods to simulate
    "kLogInitMean": np.log(2) - (0.5**2) / 2,  # Mean of log initial assets
    "kLogInitStd": 0.5,  # Standard deviation of log initial assets
    "pLogInitMean": 0,  # Mean of log initial permanent income
    "pLogInitStd": 0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
}


class testPerfMITShk(unittest.TestCase):
    def jacobian(self):
        class Test_agent(IndShockConsumerType):
            def __init__(self, cycles=0, **kwds):
                IndShockConsumerType.__init__(self, cycles=0, **kwds)

            def get_Rport(self):
                """
                Returns an array of size self.AgentCount with self.Rfree in every entry.
                Parameters
                ----------
                None
                Returns
                -------
                RfreeNow : np.array
                     Array of size self.AgentCount with risk free interest rate for each agent.
                """

                if type(self.Rfree) == list:
                    RfreeNow = self.Rfree[self.t_sim] * np.ones(self.AgentCount)
                else:
                    RfreeNow = ss.Rfree * np.ones(self.AgentCount)

                return RfreeNow

        ss = Test_agent(**JACDict)
        ss.cycles = 0
        ss.T_sim = 1200
        ss.solve()
        ss.initialize_sim()
        ss.simulate()

        class Test_agent2(Test_agent):
            def transition(self):
                pLvlPrev = self.state_prev["pLvl"]
                aNrmPrev = self.state_prev["aNrm"]
                RfreeNow = self.get_Rport()

                # Calculate new states: normalized market resources and permanent income level
                pLvlNow = (
                    pLvlPrev * self.shocks["PermShk"]
                )  # Updated permanent income level

                # "Effective" interest factor on normalized assets
                ReffNow = RfreeNow / self.shocks["PermShk"]
                bNrmNow = ReffNow * aNrmPrev  # Bank balances before labor income
                mNrmNow = (
                    bNrmNow + self.shocks["TranShk"]
                )  # Market resources after income

                if self.t_sim == 0:
                    mNrmNow = ss.state_now["mNrm"]
                    pLvlNow = ss.state_now["pLvl"]

                return pLvlNow, bNrmNow, mNrmNow, None

        listA_g = []
        params = deepcopy(JACDict)
        params["T_cycle"] = 200
        params["LivPrb"] = params["T_cycle"] * [ss.LivPrb[0]]
        params["PermGroFac"] = params["T_cycle"] * [1]
        params["PermShkStd"] = params["T_cycle"] * [(0.01 * 4 / 11) ** 0.5]
        params["TranShkStd"] = params["T_cycle"] * [0.2]
        params["Rfree"] = params["T_cycle"] * [ss.Rfree]

        ss_dx = Test_agent2(**params)
        ss_dx.pseudo_terminal = False
        ss_dx.PerfMITShk = True
        ss_dx.track_vars = ["aNrm", "mNrm", "cNrm", "pLvl", "aLvl"]
        ss_dx.cFunc_terminal_ = deepcopy(ss.solution[0].cFunc)
        ss_dx.T_sim = params["T_cycle"]
        ss_dx.cycles = 1
        ss_dx.IncShkDstn = params["T_cycle"] * ss_dx.IncShkDstn

        ss_dx.solve()
        ss_dx.initialize_sim()
        ss_dx.simulate()

        for j in range(ss_dx.T_sim):
            Ag = np.mean(ss_dx.history["aLvl"][j, :])
            listA_g.append(Ag)

        A_dx0 = np.array(listA_g)

        ##############################################################################

        example = Test_agent2(**params)
        example.pseudo_terminal = False
        example.cFunc_terminal_ = deepcopy(ss.solution[0].cFunc)
        example.T_sim = params["T_cycle"]
        example.cycles = 1
        example.PerfMITShk = True
        example.track_vars = ["aNrm", "mNrm", "cNrm", "pLvl", "aLvl"]
        example.IncShkDstn = params["T_cycle"] * example.IncShkDstn

        AHist = []
        listA = []
        dx = 0.001
        i = 50

        example.Rfree = (
            i * [ss.Rfree] + [ss.Rfree + dx] + (params["T_cycle"] - i - 1) * [ss.Rfree]
        )

        example.solve()
        example.initialize_sim()
        example.simulate()

        for j in range(example.T_sim):
            a = np.mean(example.history["aLvl"][j, :])
            listA.append(a)

        AHist.append(np.array(listA))
        JACA = (AHist[0] - A_dx0) / (dx)

        self.assertAlmostEqual(JACA[175], 6.44193e-06)


dict_harmenberg = {
    # Parameters shared with the perfect foresight model
    "CRRA": 2,  # Coefficient of relative risk aversion
    "Rfree": [1.04**0.25],  # Interest factor on assets
    "DiscFac": 0.9735,  # Intertemporal discount factor
    "LivPrb": [0.99375],  # Survival probability
    "PermGroFac": [1.00],  # Permanent income growth factor
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd": [
        0.06
    ],  # [(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount": 5,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.3],  # Standard deviation of log transitory shocks to income
    "TranShkCount": 5,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.07,  # Probability of unemployment while working
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate
    "UnempPrbRet": 0.0005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "tax_rate": 0.18,  # Flat income tax rate (legacy parameter, will be removed in future)
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraCount": 48,  # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra": None,  # Additional values to add to aXtraGrid
    # A few other parameters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 500,  # Number of agents of this type
    "T_sim": 100,  # Number of periods to simulate
    "kLogInitMean": np.log(1.3) - (0.5**2) / 2,  # Mean of log initial assets
    "kLogInitStd": 0.5,  # Standard deviation of log initial assets
    "pLogInitMean": 0.0,  # Mean of log initial permanent income
    "pLogInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
    # Parameters for Transition Matrix Simulation
    "mMin": 0.001,
    "mMax": 20,
    "mCount": 48,
    "mFac": 3,
}


class test_Harmenbergs_method(unittest.TestCase):
    def test_Harmenberg_mtd(self):
        example = IndShockConsumerType(**dict_harmenberg, verbose=0)
        example.cycles = 0
        example.track_vars = ["aNrm", "mNrm", "cNrm", "pLvl", "aLvl"]
        example.T_sim = 20000

        example.solve()

        example.neutral_measure = True
        example.update_income_process()

        example.initialize_sim()
        example.simulate()

        Asset_list = []
        Consumption_list = []
        M_list = []

        for i in range(example.T_sim):
            Assetagg = np.mean(example.history["aNrm"][i])
            Asset_list.append(Assetagg)
            ConsAgg = np.mean(example.history["cNrm"][i])
            Consumption_list.append(ConsAgg)
            Magg = np.mean(example.history["mNrm"][i])
            M_list.append(Magg)

        #########################################################

        example2 = IndShockConsumerType(**dict_harmenberg, verbose=0)
        example2.cycles = 0
        example2.track_vars = ["aNrm", "mNrm", "cNrm", "pLvl", "aLvl"]
        example2.T_sim = 20000

        example2.solve()
        example2.initialize_sim()
        example2.simulate()

        Asset_list2 = []
        Consumption_list2 = []
        M_list2 = []

        for i in range(example2.T_sim):
            Assetagg = np.mean(example2.history["aLvl"][i])
            Asset_list2.append(Assetagg)
            ConsAgg = np.mean(example2.history["cNrm"][i] * example2.history["pLvl"][i])
            Consumption_list2.append(ConsAgg)
            Magg = np.mean(example2.history["mNrm"][i] * example2.history["pLvl"][i])
            M_list2.append(Magg)

        c_std2 = np.std(Consumption_list2)
        c_std1 = np.std(Consumption_list)
        c_std_ratio = c_std2 / c_std1

        # simulation tests -- seed/generator specific
        # But these are based on aggregate population statistics.
        # WARNING: May fail stochastically, or based on specific RNG types.
        # self.assertAlmostEqual(c_std2, 0.0376882, places = 2)
        # self.assertAlmostEqual(c_std1, 0.0044117, places = 2)
        # self.assertAlmostEqual(c_std_ratio, 8.5426941, places = 2)


# %% Shock pre-computing tests


class testReadShock(unittest.TestCase):
    """
    Tests the functionality for pre computing shocks and using them in simulations
    """

    def setUp(self):
        # Make a dictionary containing all parameters needed to solve the model
        self.base_params = copy(init_idiosyncratic_shocks)

        agent_count = 10
        t_sim = 200
        # Make agents die relatively often
        LivPrb = [0.9]
        # No interest or growth to facilitate computations
        Rfree = 1.0
        PermGroFac = 1.0

        self.base_params.update(
            {
                "AgentCount": agent_count,
                "T_sim": t_sim,
                "LivPrb": LivPrb,
                "PermGroFac": [PermGroFac],
                "Rfree": [Rfree],
            }
        )

    def test_NewbornStatesAndShocks(self):
        # Make agent, shock and initial condition histories
        agent = IndShockConsumerType(**self.base_params)
        agent.track_vars = ["bNrm", "t_age"]
        agent.make_shock_history()

        # Find indices of agents and time periods that correspond to deaths
        # this will be non-nan indices of newborn_init_history for states
        # that are used in initializing the agent. aNrm is one of them.
        idx = np.logical_not(np.isnan(agent.newborn_init_history["aNrm"]))

        # Change the values
        a_init_newborns = 20
        agent.newborn_init_history["aNrm"][idx] = a_init_newborns
        # Also change the shocks of newborns
        pshk_newborns = 0.5
        agent.shock_history["PermShk"][idx] = pshk_newborns
        agent.shock_history["TranShk"][idx] = 0.0

        # Solve and simulate the agent
        agent.solve()
        agent.initialize_sim()
        agent.simulate()

        # Given our manipulation of initial wealth and permanent shocks,
        # agents of age == 1 should have starting resources a_init_newborns/pshk_newborns
        # (no interest, no deterministic growth and no transitory shock)
        age = agent.history["t_age"]
        self.assertTrue(
            np.all(agent.history["bNrm"][age == 1] == a_init_newborns / pshk_newborns)
        )


class testLCMortalityReadShocks(unittest.TestCase):
    """
    Tests that mortality is working adequately when shocks are read
    """

    def setUp(self):
        # Make a dictionary containing all parameters needed to solve the model
        self.base_params = copy(init_lifecycle)

        agent_count = 10
        t_sim = 2000

        self.base_params.update(
            {
                "AgentCount": agent_count,
                "T_sim": t_sim,
            }
        )

    def test_compare_t_age_t_cycle(self):
        # Make agent, shock and initial condition histories
        agent = IndShockConsumerType(**self.base_params)
        agent.track_vars = ["t_age", "t_cycle"]
        agent.make_shock_history()

        # Solve and simulate the agent
        agent.solve()
        agent.initialize_sim()
        agent.simulate()

        hist = copy(agent.history)
        for key, array in hist.items():
            hist[key] = array.flatten(order="F")

        # Check that t_age is always t_cycle
        # Except possibly in cases where the agent reach t_age = T_age. In this case,
        # t_cycle is set to 0 at the end of the period, and the agent dies,
        # But t_age is reset only at the start of next period and thus we can have
        # t_age = T_age and t_cycle = 0
        self.assertTrue(
            np.all(
                np.logical_or(
                    hist["t_age"] == hist["t_cycle"],
                    np.logical_and(
                        hist["t_cycle"] == 0, hist["t_age"] == agent.T_cycle
                    ),
                )
            )
        )

    def test_compare_t_age_t_cycle_premature_death(self):
        # Re-do the previous test in an instance where we prematurely
        # kill agents
        par = copy(self.base_params)
        par["T_age"] = par["T_age"] - 8
        # Make agent, shock and initial condition histories
        agent = IndShockConsumerType(**par)
        agent.track_vars = ["t_age", "t_cycle"]
        agent.make_shock_history()

        # Solve and simulate the agent
        agent.solve()
        agent.initialize_sim()
        agent.simulate()

        hist = copy(agent.history)
        for key, array in hist.items():
            hist[key] = array.flatten(order="F")

        # Check that t_age is always t_cycle
        # (the exception from before should not happen
        # because we are killing agents before T_cycle)
        self.assertTrue(np.all(hist["t_age"] == hist["t_cycle"]))


class testInitShuffle(unittest.TestCase):
    """Tests for init_shuffle parameter on IndShockConsumerType.

    init_shuffle=True makes IndShockConsumerType.sim_birth draw initial
    kNrm and pLvl from the discretized init distributions using
    exact-marginal matching (floor-plus-leftover), instead of iid
    sampling.  This addresses the cross-sectional noise in the initial
    wealth/permanent-income distribution, a residual noise source that
    per-period shuffle flags (income_shuffle, markov_shuffle) cannot
    address, because sim_birth runs once per agent at initialize_sim
    before any period loop.
    """

    # HARK's default kLogInitStd = pLogInitStd = 0.0, which makes the
    # init distributions degenerate (single atom).  For the tests below
    # we override to non-degenerate values so there's a meaningful
    # distribution to shuffle.
    _nondegen_init = {
        "pLogInitMean": 0.0,
        "pLogInitStd": 0.3,
        "kLogInitMean": -2.0,
        "kLogInitStd": 0.5,
    }

    def test_init_shuffle_runs(self):
        """init_shuffle=True should solve and simulate without error."""
        agent = IndShockConsumerType(
            AgentCount=1500,  # = 15 * 100 (clean replicate for default 15-atom init dstns)
            T_sim=20,
            init_shuffle=True,
            **self._nondegen_init,
        )
        agent.solve()
        agent.initialize_sim()
        agent.simulate()
        # Basic shape check: nothing broke
        self.assertEqual(agent.state_now["pLvl"].shape, (1500,))

    def test_init_shuffle_exact_frequencies(self):
        """Empirical pLvl frequencies should match the discretized dstn
        within +/-1 per atom (the floor-plus-leftover algorithm's worst case).

        Note: ``Lognormal.discretize(N, method='equiprobable')`` produces
        pmv = np.full(N, 1/N), and ``1/N`` is not exactly representable in
        float64 for N=15, so the per-atom count can deviate by 1 from the
        ideal ``N_draws/N_atoms`` due to leftover-slot allocation absorbing
        the floating-point residual.  What we *can* verify is that
        (a) every atom appears between ``floor(N/J)`` and ``ceil(N/J)+1``
        times, (b) the total count equals ``N`` exactly, and (c) the
        sample mean equals the analytical mean to machine precision,
        which is much stronger than iid sampling can achieve.
        """
        N = 1500  # = 15 * 100 (default pLvlInitCount = 15)
        agent = IndShockConsumerType(
            AgentCount=N,
            T_sim=1,
            init_shuffle=True,
            **self._nondegen_init,
        )
        agent.solve()
        agent.initialize_sim()

        # pLvl: check per-atom count is within +/-1 of the ideal
        n_pLvl_atoms = agent.pLvlInitDstn.atoms.shape[-1]
        ideal_per_atom = N / n_pLvl_atoms  # 100.0 exactly
        pLvl_atom_vals = np.sort(np.unique(agent.pLvlInitDstn.atoms.flatten()))
        pLvl_obs = agent.state_now["pLvl"] / agent.state_now["PlvlAgg"]

        total_counted = 0
        for val in pLvl_atom_vals:
            count = int(np.sum(np.isclose(pLvl_obs, val, rtol=1e-10)))
            total_counted += count
            # Each count must be within +/-1 of ideal (floor-plus-leftover bound)
            self.assertTrue(
                abs(count - ideal_per_atom) <= 1,
                f"pLvl atom {val}: count={count}, expected ~= {ideal_per_atom} +/-1",
            )
        self.assertEqual(total_counted, N, "All N draws must be accounted for")

        # Sample mean should equal analytical mean to machine precision.
        # This is the strongest guarantee shuffle gives for equiprobable
        # lognormal discretisations: the aggregate is exact even when
        # individual atom counts deviate by +/-1 due to float rounding.
        expected_mean = float(
            np.sum(agent.pLvlInitDstn.pmv * agent.pLvlInitDstn.atoms.flatten())
        )
        # At N divisible by 15 with exactly 100 per atom, the sample mean
        # would equal the analytical mean to floating-point precision.
        # With +/-1 slack from floating-point pmv rounding, the sample
        # mean is off by at most (max_atom - min_atom)/N ~= 0.002 at N=1500.
        # That's still O(1/N), way tighter than O(1/sqrt(N)) from iid.
        self.assertAlmostEqual(float(np.mean(pLvl_obs)), expected_mean, delta=0.01)

        # Same check for kNrm
        n_kNrm_atoms = agent.kNrmInitDstn.atoms.shape[-1]
        ideal_per_atom_k = N / n_kNrm_atoms
        kNrm_atom_vals = np.sort(np.unique(agent.kNrmInitDstn.atoms.flatten()))
        kNrm_obs = agent.state_now["aNrm"]
        total_counted_k = 0
        for val in kNrm_atom_vals:
            count = int(np.sum(np.isclose(kNrm_obs, val, rtol=1e-10)))
            total_counted_k += count
            self.assertTrue(abs(count - ideal_per_atom_k) <= 1)
        self.assertEqual(total_counted_k, N)

    def test_init_shuffle_reduces_seed_variance(self):
        """Shuffle must reduce seed-to-seed variance of cross-sectional
        mean(pLvl) at t=0 by at least an order of magnitude compared
        to iid sampling.

        Note: the shuffle variance is not *exactly* zero when atom
        probabilities can't be exactly represented in float64.  For
        ``Lognormal.discretize(15, method='equiprobable')``, the pmv
        is ``[1/15, ..., 1/15]`` in float64, and the floor-plus-leftover
        algorithm has to allocate ~1 leftover slot per call, which
        lands in a different atom for different seeds.  That produces
        a tiny residual of order ``(atom_range)/N``, vastly smaller
        than the iid O(1/sqrt(N)) but nonzero.
        """
        N = 1500
        n_seeds = 8
        means_shuffle = []
        means_iid = []
        for seed in range(n_seeds):
            agent_sh = IndShockConsumerType(
                AgentCount=N,
                T_sim=1,
                init_shuffle=True,
                seed=seed,
                **self._nondegen_init,
            )
            agent_sh.solve()
            agent_sh.initialize_sim()
            means_shuffle.append(float(np.mean(agent_sh.state_now["pLvl"])))

            agent_iid = IndShockConsumerType(
                AgentCount=N,
                T_sim=1,
                init_shuffle=False,
                seed=seed,
                **self._nondegen_init,
            )
            agent_iid.solve()
            agent_iid.initialize_sim()
            means_iid.append(float(np.mean(agent_iid.state_now["pLvl"])))

        sd_shuffle = float(np.std(means_shuffle))
        sd_iid = float(np.std(means_iid))

        # iid sampling must have meaningful seed variance (baseline)
        self.assertGreater(sd_iid, 1e-4, "iid must show measurable seed variance")

        # Shuffle variance should be at least 10x smaller than iid
        # (in practice it's usually 50x+ smaller; 10x is a safe bound)
        self.assertLess(
            sd_shuffle,
            sd_iid / 10.0,
            f"shuffle SD {sd_shuffle:.6g} should be << iid SD {sd_iid:.6g}",
        )

        # All shuffle means should cluster tightly around the analytical mean
        expected_mean = float(
            np.sum(agent_sh.pLvlInitDstn.pmv * agent_sh.pLvlInitDstn.atoms.flatten())
        )
        for m in means_shuffle:
            # Much tighter than 1/sqrt(N) ~= 0.008 for iid at this N
            self.assertAlmostEqual(m, expected_mean, delta=0.005)

    def test_init_shuffle_default_false(self):
        """Default init_shuffle should be False."""
        agent = IndShockConsumerType(AgentCount=100)
        self.assertFalse(getattr(agent, "init_shuffle", False))


class testInitShuffleStreamInvariance(unittest.TestCase):
    """Default-path behavior golden captured on main at a25d3ae0: with
    init_shuffle at its default, simulations are bit-identical."""

    def test_default_sim_unchanged(self):
        agent = IndShockConsumerType(AgentCount=200, T_sim=8, seed=555)
        agent.track_vars = ["cNrm"]
        agent.solve()
        agent.initialize_sim()
        agent.simulate()
        np.testing.assert_allclose(
            [float(x) for x in agent.history["cNrm"][3, :4]],
            [
                1.1070787532288362,
                0.9087055494949798,
                1.1694416325917305,
                0.9579870570215201,
            ],
            rtol=1e-10,
        )


class testDeathShuffle(unittest.TestCase):
    """Tests for the death_shuffle parameter on IndShockConsumerType.

    These exercise sim_death() rather than _sim_death_shuffled() directly, so
    that they also pin the dispatch: if sim_death stopped consulting
    death_shuffle, the death counts below would go back to being binomial and
    every determinism assertion here would fail.
    """

    def test_deaths_constant_over_simulation(self):
        """Over a full simulation, each period kills exactly AgentCount*DiePrb."""
        agent = IndShockConsumerType(
            AgentCount=5000,
            T_sim=25,
            seed=1234,
            death_shuffle=True,
            T_age=None,  # no old-age deaths, so mortality is the only killer
        )
        agent.track_vars = ["who_dies"]
        agent.solve()
        agent.initialize_sim()
        agent.simulate()

        DiePrb = 1.0 - np.asarray(agent.LivPrb)[-1]
        expected_deaths = round(agent.AgentCount * DiePrb)
        history = np.asarray(agent.history["who_dies"], dtype=float)
        # The final row is never written by simulate(), so drop unfilled rows.
        recorded = history[~np.isnan(history).all(axis=1)]
        self.assertGreater(recorded.shape[0], 1)
        counts = recorded.sum(axis=1)
        self.assertEqual(set(counts.tolist()), {float(expected_deaths)})

    def test_exact_count_when_Np_integral(self):
        """Repeated sim_death draws kill the same number of agents every time.

        The count is exactly floor(N*DiePrb) only when N*DiePrb is an integer;
        otherwise the fractional part is resolved by a coin flip and the count
        alternates between floor and floor+1 (see the next test).  Keep this
        calibration integral, or assert on the expectation instead.
        """
        agent = IndShockConsumerType(
            AgentCount=5000, T_sim=2, seed=2, death_shuffle=True, T_age=None
        )
        agent.solve()
        agent.initialize_sim()

        DiePrb = 1.0 - np.asarray(agent.LivPrb)[-1]
        N_times_p = agent.AgentCount * DiePrb
        # Guard the assumption this test's exact-count assertion rests on.
        self.assertAlmostEqual(N_times_p, round(N_times_p), places=9)

        counts = {int(agent.sim_death().sum()) for _ in range(30)}
        self.assertEqual(counts, {round(N_times_p)})

    def test_unbiased_when_Np_fractional(self):
        """With a fractional remainder the count straddles floor and floor+1.

        The contract is an unbiased expected number of deaths and a marginal
        death probability of DiePrb for every agent, not a fixed count.
        """
        agent = IndShockConsumerType(
            AgentCount=5001, T_sim=2, seed=20260811, death_shuffle=True, T_age=None
        )
        agent.solve()
        agent.initialize_sim()

        DiePrb = 1.0 - np.asarray(agent.LivPrb)[-1]
        N_times_p = agent.AgentCount * DiePrb
        self.assertGreater(N_times_p - np.floor(N_times_p), 0.0)

        reps = 1000
        deaths_by_agent = np.zeros(agent.AgentCount)
        counts = np.empty(reps)
        for i in range(reps):
            who_dies = agent.sim_death()
            deaths_by_agent += who_dies
            counts[i] = who_dies.sum()

        # Only two counts are reachable, and they bracket N*DiePrb.
        self.assertEqual(
            set(counts.tolist()),
            {float(np.floor(N_times_p)), float(np.floor(N_times_p) + 1)},
        )
        # Expected number of deaths is preserved (5 standard errors).
        se_count = np.sqrt(DiePrb * (1.0 - DiePrb) / reps)
        self.assertLess(abs(counts.mean() - N_times_p), 5.0 * se_count)

        # Every agent faces the same death probability: no bias by index, which
        # is what a remainder handed out in index order would produce.
        rates = deaths_by_agent[:5000].reshape(10, 500).mean(axis=1) / reps
        se_rate = np.sqrt(DiePrb * (1.0 - DiePrb) / (reps * 500))
        self.assertLess(np.abs(rates - DiePrb).max(), 5.0 * se_rate)

    def test_death_shuffle_default_false(self):
        """death_shuffle defaults to False on every type that consults it."""
        for agent in (PerfForesightConsumerType(), IndShockConsumerType()):
            self.assertFalse(agent.death_shuffle)


class testDeathShuffleStreamInvariance(unittest.TestCase):
    """Default-path behavior golden captured on main at a25d3ae0: with
    death_shuffle at its default, simulations are bit-identical."""

    def test_default_sim_unchanged(self):
        agent = IndShockConsumerType(AgentCount=200, T_sim=8, seed=555)
        agent.track_vars = ["cNrm"]
        agent.solve()
        agent.initialize_sim()
        agent.simulate()
        np.testing.assert_allclose(
            [float(x) for x in agent.history["cNrm"][3, :4]],
            [
                1.1070787532288362,
                0.9087055494949798,
                1.1694416325917305,
                0.9579870570215201,
            ],
            rtol=1e-10,
        )


class testCubicSolutionSerialization(unittest.TestCase):
    """
    A solved agent whose consumption function is a cubic spline must survive
    deepcopy, and its solution must survive pickle: with CubicBool the cFunc
    wraps a scipy spline, and scipy 1.18.0 stores unpicklable module objects
    on spline instances (scipy issue #25489), so CubicHermiteInterp rebuilds
    the spline on deserialization instead of serializing it.
    """

    def setUp(self):
        self.agent = IndShockConsumerType(CubicBool=True, vFuncBool=True)
        self.agent.solve()
        self.m = np.linspace(0.5, 20.0, 50)

    def check_solution(self, solution):
        np.testing.assert_array_equal(
            self.agent.solution[0].cFunc(self.m), solution.cFunc(self.m)
        )
        np.testing.assert_array_equal(
            self.agent.solution[0].vFunc(self.m), solution.vFunc(self.m)
        )

    def test_deepcopy_solved_agent(self):
        clone = deepcopy(self.agent)
        self.check_solution(clone.solution[0])

    def test_pickle_solution(self):
        restored = pickle.loads(pickle.dumps(self.agent.solution[0]))
        self.check_solution(restored)


class testIncomeShuffleIndShock(unittest.TestCase):
    """Tests for the income_shuffle parameter on IndShockConsumerType."""

    def test_default_shuffle_false(self):
        """Backward compat: default income_shuffle=False works unchanged."""
        agent = IndShockConsumerType(AgentCount=100, T_sim=10)
        agent.solve()
        agent.initialize_sim()
        agent.simulate()
        # Just verify it runs and produces valid shocks
        self.assertEqual(agent.shocks["PermShk"].shape, (100,))
        self.assertTrue(np.all(agent.shocks["PermShk"] > 0))
        self.assertTrue(np.all(agent.shocks["TranShk"] >= 0))

    def test_shuffle_true_runs(self):
        """income_shuffle=True solve+simulate completes without error."""
        agent = IndShockConsumerType(AgentCount=1000, T_sim=20, income_shuffle=True)
        agent.solve()
        agent.initialize_sim()
        agent.simulate()
        self.assertEqual(agent.shocks["PermShk"].shape, (1000,))
        self.assertTrue(np.all(agent.shocks["PermShk"] > 0))

    def test_shuffle_empirical_frequencies(self):
        """With shuffle=True, empirical PermShk frequencies should match pmv closely."""
        agent = IndShockConsumerType(AgentCount=5000, T_sim=5, income_shuffle=True)
        agent.solve()
        agent.initialize_sim()
        agent.simulate()

        # Check that the empirical distribution of PermShk values matches
        # the theoretical pmv of the joint income shock distribution.
        # Multiple joint atoms can share the same PermShk value, so we
        # aggregate probabilities by unique PermShk atom.
        dstn = agent.IncShkDstn[0]
        perm_atoms = dstn.atoms[0]
        pmv = dstn.pmv

        unique_perm = np.unique(perm_atoms)
        perm_shks = agent.shocks["PermShk"] / agent.PermGroFac[0]
        for atom in unique_perm:
            expected_freq = np.sum(pmv[np.isclose(perm_atoms, atom, rtol=1e-10)])
            empirical_freq = np.mean(np.isclose(perm_shks, atom, rtol=1e-10))
            # The tolerance has to sit below iid sampling noise or the test
            # passes whether or not income_shuffle is honored.  Measured at
            # this AgentCount over 40 seeds: worst per-atom deviation is
            # 0.01306 for iid draws and 0.00214 with shuffling.
            np.testing.assert_allclose(
                empirical_freq,
                expected_freq,
                atol=0.005,
                err_msg=f"Frequency mismatch for PermShk atom {atom}",
            )


class testIncomeShuffleStreamInvariance(unittest.TestCase):
    """Default-path RNG-stream golden captured on main at a25d3ae0: with
    income_shuffle at its default, the exact shock draws must not change."""

    def test_default_shock_stream_unchanged(self):
        agent = IndShockConsumerType(AgentCount=200, T_sim=8, seed=555)
        agent.solve()
        agent.initialize_sim()
        agent.simulate()
        np.testing.assert_allclose(
            [float(x) for x in agent.shocks["PermShk"][:5]],
            [
                1.0887560662509859,
                0.9278094171517418,
                0.8589344616271869,
                1.0427376294215152,
                1.0427376294215152,
            ],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            [float(x) for x in agent.shocks["TranShk"][:5]],
            [
                1.209379023455466,
                1.0317263121066038,
                1.209379023455466,
                0.9524671973887084,
                1.0317263121066038,
            ],
            rtol=1e-10,
        )


class testIncomeShuffleMarkov(unittest.TestCase):
    """Tests for the income_shuffle parameter on MarkovConsumerType."""

    @staticmethod
    def make_markov_agent(**kwargs):
        params = {
            "MrkvArray": [np.array([[0.9, 0.1], [0.1, 0.9]])],
            "AgentCount": 500,
            "T_sim": 10,
        }
        params.update(kwargs)
        agent = MarkovConsumerType(**params)
        agent.cycles = 0
        agent.solve()
        return agent

    def test_markov_shuffle_runs(self):
        """MarkovConsumerType with income_shuffle=True completes simulation."""
        agent = self.make_markov_agent(income_shuffle=True)
        agent.initialize_sim()
        agent.simulate()
        self.assertEqual(agent.shocks["PermShk"].shape, (500,))
        self.assertTrue(np.all(agent.shocks["PermShk"] > 0))

    def test_markov_shuffle_beats_iid_within_state(self):
        """income_shuffle must actually change the Markov draws.

        Within each discrete state the realized shock counts should track
        that state's pmv far more closely than iid draws do.  A smoke test
        that only checks shapes passes with the flag ignored, so this
        compares the two paths against each other rather than against a
        fixed tolerance.
        """
        worst = {}
        for flag in (False, True):
            agent = self.make_markov_agent(income_shuffle=flag, T_sim=1)
            dev = 0.0
            for seed in range(20):
                agent.initialize_sim()
                for j, dstn in enumerate(agent.IncShkDstn[0]):
                    dstn._rng = np.random.default_rng(1000 * j + seed)
                agent.simulate()
                mrkv = agent.shocks["Mrkv"]
                for j, dstn in enumerate(agent.IncShkDstn[0]):
                    these = mrkv == j
                    if np.sum(these) < 200:
                        continue
                    shks = agent.shocks["PermShk"][these] / agent.PermGroFac[0][j]
                    for atom in np.unique(dstn.atoms[0]):
                        expected = np.sum(
                            dstn.pmv[np.isclose(dstn.atoms[0], atom, rtol=1e-10)]
                        )
                        empirical = np.mean(np.isclose(shks, atom, rtol=1e-10))
                        dev = max(dev, abs(empirical - expected))
            worst[flag] = dev
        self.assertLess(
            worst[True],
            worst[False] / 2.0,
            f"income_shuffle worst per-atom deviation {worst[True]:.5f} should be "
            f"well under the iid {worst[False]:.5f}; the flag looks ignored",
        )


class testMarkovTransitionShuffle(unittest.TestCase):
    """Tests for the markov_shuffle parameter on MarkovConsumerType."""

    def test_markov_shuffle_state_counts(self):
        """With markov_shuffle=True, state counts should match deterministic targets."""

        TM = np.array([[0.95, 0.05], [0.5, 0.5]])
        mp = MarkovProcess(TM, seed=42)

        # Start with 9500 in state 0, 500 in state 1
        state = np.array([0] * 9500 + [1] * 500)
        new_state = mp.draw(state, shuffle=True)

        # Expected: 9500*0.95=9025 stay in 0, 9500*0.05=475 go to 1
        #           500*0.5=250 go to 0, 500*0.5=250 stay in 1
        count_0_to_0 = np.sum((state == 0) & (new_state == 0))
        count_0_to_1 = np.sum((state == 0) & (new_state == 1))
        count_1_to_0 = np.sum((state == 1) & (new_state == 0))
        count_1_to_1 = np.sum((state == 1) & (new_state == 1))

        # With shuffle, counts should be within +/-1 of deterministic target
        self.assertAlmostEqual(count_0_to_0, 9025, delta=1)
        self.assertAlmostEqual(count_0_to_1, 475, delta=1)
        self.assertAlmostEqual(count_1_to_0, 250, delta=1)
        self.assertAlmostEqual(count_1_to_1, 250, delta=1)

    def test_markov_shuffle_consistent_over_time(self):
        """markov_shuffle=True produces correct counts over multiple periods."""

        TM = np.array([[0.95, 0.05], [0.5, 0.5]])
        mp = MarkovProcess(TM, seed=123)
        state = np.zeros(10000, dtype=int)

        for _ in range(100):
            state = mp.draw(state, shuffle=True)
            total = len(state)
            n0 = np.sum(state == 0)
            n1 = np.sum(state == 1)
            self.assertEqual(n0 + n1, total)

        # After 100 steps, should be near steady state: pi_0 = 0.5/0.55 is about 0.909
        ss_0 = 0.5 / (0.05 + 0.5)
        empirical_0 = np.sum(state == 0) / len(state)
        np.testing.assert_allclose(empirical_0, ss_0, atol=0.02)

    def test_markov_consumer_shuffle(self):
        """MarkovConsumerType with markov_shuffle=True completes simulation."""

        agent = MarkovConsumerType(
            MrkvArray=[np.array([[0.9, 0.1], [0.1, 0.9]])],
            AgentCount=1000,
            T_sim=20,
            markov_shuffle=True,
        )
        agent.cycles = 0
        agent.solve()
        agent.initialize_sim()
        agent.simulate()
        self.assertEqual(agent.shocks["Mrkv"].shape, (1000,))


class testMarkovShuffleEndToEnd(unittest.TestCase):
    """markov_shuffle and balanced_transitions, exercised through
    ``get_markov_states`` rather than by calling ``MarkovProcess`` directly.

    Both parameters could previously be deleted at their dispatch site with
    the whole suite still green, because the only test that touched them
    asserted on ``shocks["Mrkv"].shape``.

    ``MrkvArray`` is assigned after construction on purpose.  It is a
    constructed parameter on ``MarkovConsumerType``, so passing
    ``MrkvArray=`` to ``__init__`` is silently overwritten by the
    constructor and the agent runs on the default matrix instead.  The
    calibration below is chosen so that ``N_j * P[j,k]`` is not an integer,
    which is what makes the leftover-slot path run at all.
    """

    TM = np.array([[0.93, 0.07], [0.40, 0.60]])

    def make_agent(self, **flags):
        agent = MarkovConsumerType(
            AgentCount=997,
            T_sim=15,
            seed=0,
            LivPrb=[np.array([1.0, 1.0])],
            Rfree=[np.array([1.03, 1.03])],
            PermGroFac=[np.array([1.01, 1.01])],
            track_vars=["Mrkv", "pLvl"],
            **flags,
        )
        agent.MrkvArray = [self.TM]
        agent.cycles = 0
        agent.solve()
        agent.initialize_sim()
        agent.simulate()
        return (
            np.asarray(agent.history["Mrkv"], dtype=int),
            np.asarray(agent.history["pLvl"], dtype=float),
        )

    def test_markov_shuffle_makes_simulated_counts_quota_exact(self):
        """Realized transition counts must sit within one agent of the quota.

        Nobody dies here (LivPrb=1), so period t's counts are a clean
        transition out of period t-1's states.  Under iid the counts scatter
        by roughly sqrt(N_j p (1-p)), which is about 9 agents for the rarer
        target at this population; measured worst deviation without the flag
        is 17.9 agents against 0.9 with it.
        """
        mrkv, _ = self.make_agent(markov_shuffle=True)

        worst = 0.0
        checked = 0
        for t in range(1, mrkv.shape[0]):
            src, tgt = mrkv[t - 1], mrkv[t]
            for j in range(2):
                in_j = src == j
                N_j = int(in_j.sum())
                if N_j == 0:
                    continue
                for k in range(2):
                    count = int((in_j & (tgt == k)).sum())
                    worst = max(worst, abs(count - N_j * self.TM[j, k]))
                    checked += 1
        self.assertGreater(checked, 20)
        self.assertLessEqual(
            worst,
            1.0,
            f"worst |count - N_j*P[j,k]| was {worst:.2f}; with markov_shuffle "
            f"on, every transition count must be within one agent of its "
            f"quota, and iid draws are not.",
        )

    def test_balanced_transitions_makes_movers_representative(self):
        """The agents who change state must look like the ones who do not.

        With balanced_transitions on, agents are sorted by permanent income
        and the movers are systematically sampled across that order, so the
        movers' mean pLvl tracks the source population's.  Under the plain
        random permutation the movers are just a random subset and the gap
        is roughly sd/sqrt(n_movers).

        This is also the only regression test on the sort key itself.  The
        key must come from ``state_prev``: ``_sim_period_prologue`` blanks
        ``state_now`` with ``np.empty`` before ``get_shocks`` runs, so
        reading pLvl from there sorts on uninitialized memory, which
        scrambles the order and returns the gap to its unsorted size.

        Measured over eight seeds: mean standardized gap at most 0.029 with
        the flag and at least 0.071 without it, so 0.045 separates them with
        no overlap.
        """
        gaps = {}
        for label, flags in (
            ("shuffle_only", dict(markov_shuffle=True)),
            ("balanced", dict(markov_shuffle=True, balanced_transitions=True)),
        ):
            mrkv, plvl = self.make_agent(**flags)
            seen = []
            for t in range(1, mrkv.shape[0]):
                src, tgt, key = mrkv[t - 1], mrkv[t], plvl[t - 1]
                for j in range(2):
                    in_j = src == j
                    movers = in_j & (tgt != j)
                    if movers.sum() >= 5:
                        seen.append(
                            abs(key[movers].mean() - key[in_j].mean()) / key[in_j].std()
                        )
            self.assertGreater(len(seen), 10)
            gaps[label] = float(np.mean(seen))

        self.assertLess(
            gaps["balanced"],
            0.045,
            f"balanced_transitions should make the movers representative of "
            f"their source population in pLvl; mean standardized gap was "
            f"{gaps['balanced']:.4f} against {gaps['shuffle_only']:.4f} for "
            f"the plain shuffle.",
        )
        self.assertLess(gaps["balanced"], 0.5 * gaps["shuffle_only"])
