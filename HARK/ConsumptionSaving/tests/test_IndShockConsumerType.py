import unittest
from copy import copy, deepcopy

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    ConsIndShockSolverBasic,
    init_lifecycle,
    init_idiosyncratic_shocks,
)


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
        #self.assertEqual(self.agent.shocks["PermShk"][0], 1.0427376294215103)
        #self.assertAlmostEqual(self.agent.shocks["PermShk"][1], 0.9278094171517413)
        #self.assertAlmostEqual(self.agent.shocks["TranShk"][0], 0.881761797501595)

    def test_ConsIndShockSolverBasic(self):
        LifecycleExample = IndShockConsumerType(**init_lifecycle)
        LifecycleExample.cycles = 1
        LifecycleExample.solve()

        # test the solution_terminal
        self.assertAlmostEqual(LifecycleExample.solution[-1].cFunc(2).tolist(), 2)

        self.assertAlmostEqual(LifecycleExample.solution[9].cFunc(1), 0.79429538)
        self.assertAlmostEqual(LifecycleExample.solution[8].cFunc(1), 0.79391692)
        self.assertAlmostEqual(LifecycleExample.solution[7].cFunc(1), 0.79253095)

        self.assertAlmostEqual(
            LifecycleExample.solution[0].cFunc(1).tolist(), 0.7506184692092213
        )
        self.assertAlmostEqual(
            LifecycleExample.solution[1].cFunc(1).tolist(), 0.7586358637239385
        )
        self.assertAlmostEqual(
            LifecycleExample.solution[2].cFunc(1).tolist(), 0.7681247572911291
        )

        solver = ConsIndShockSolverBasic(
            LifecycleExample.solution[1],
            LifecycleExample.IncShkDstn[0],
            LifecycleExample.LivPrb[0],
            LifecycleExample.DiscFac,
            LifecycleExample.CRRA,
            LifecycleExample.Rfree,
            LifecycleExample.PermGroFac[0],
            LifecycleExample.BoroCnstArt,
            LifecycleExample.aXtraGrid,
            LifecycleExample.vFuncBool,
            LifecycleExample.CubicBool,
        )

        solver.prepare_to_solve()

        self.assertAlmostEqual(solver.DiscFacEff, 0.9586233599999999)
        self.assertAlmostEqual(solver.PermShkMinNext, 0.6554858756904397)
        self.assertAlmostEqual(solver.cFuncNowCnst(4).tolist(), 4.0)
        self.assertAlmostEqual(
            solver.prepare_to_calc_EndOfPrdvP()[0], -0.19792871012285213
        )
        self.assertAlmostEqual(
            solver.prepare_to_calc_EndOfPrdvP()[-1], 19.801071289877118
        )

        EndOfPrdvP = solver.calc_EndOfPrdvP()

        self.assertAlmostEqual(EndOfPrdvP[0], 6657.839372100613)
        self.assertAlmostEqual(EndOfPrdvP[-1], 0.2606075215645896)

        solution = solver.make_basic_solution(
            EndOfPrdvP, solver.aNrmNow, solver.make_linear_cFunc
        )
        solver.add_MPC_and_human_wealth(solution)

        self.assertAlmostEqual(solution.cFunc(4).tolist(), 1.0028005137373956)

    def test_simulated_values(self):
        self.agent.initialize_sim()
        self.agent.simulate()

        # MPCnow depends on assets, which are stochastic
        #self.assertAlmostEqual(self.agent.MPCnow[1], 0.5711503906043797)
        
        # simulation test -- seed/generator specific
        #self.assertAlmostEqual(self.agent.state_now["aLvl"][1], 0.18438326264597635)


class testBufferStock(unittest.TestCase):
    """Tests of the results of the BufferStock REMARK."""

    def setUp(self):
        # Make a dictionary containing all parameters needed to solve the model
        self.base_params = copy(init_idiosyncratic_shocks)

        # Set the parameters for the baseline results in the paper
        # using the variable values defined in the cell above
        self.base_params["PermGroFac"] = [1.03]
        self.base_params["Rfree"] = 1.04
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

        self.assertAlmostEqual(c_m[500], 1.4008090582203356)
        self.assertAlmostEqual(c_t1[500], 2.9227437159255216)
        self.assertAlmostEqual(c_t5[500], 1.7350607327187664)
        self.assertAlmostEqual(c_t10[500], 1.4991390649979213)
        self.assertAlmostEqual(c_t10[600], 1.6101476268581576)
        self.assertAlmostEqual(c_t10[700], 1.7196531041366991)

    def test_GICRawFails(self):
        GICRaw_fail_dictionary = dict(self.base_params)
        GICRaw_fail_dictionary["Rfree"] = 1.08
        GICRaw_fail_dictionary["PermGroFac"] = [1.00]
        GICRaw_fail_dictionary[
            "cycles"
        ] = 0  # cycles=0 makes this an infinite horizon consumer

        GICRawFailExample = IndShockConsumerType(**GICRaw_fail_dictionary)

        GICRawFailExample.solve()
        GICRawFailExample.unpack("cFunc")
        m = np.linspace(0, 5, 1000)
        c_m = GICRawFailExample.cFunc[0](m)

        self.assertAlmostEqual(c_m[500], 0.7772637042393458)
        self.assertAlmostEqual(c_m[700], 0.8392649061916746)

        self.assertFalse(GICRawFailExample.conditions["GICRaw"])

    def test_infinite_horizon(self):
        baseEx_inf = IndShockConsumerType(**self.base_params)
        baseEx_inf.assign_parameters(cycles=0)
        baseEx_inf.solve()
        baseEx_inf.unpack("cFunc")

        m1 = np.linspace(
            1, baseEx_inf.solution[0].mNrmStE, 50
        )  # m1 defines the plot range on the left of target m value (e.g. m <= target m)
        c_m1 = baseEx_inf.cFunc[0](m1)

        self.assertAlmostEqual(c_m1[0], 0.8527887545025995)
        self.assertAlmostEqual(c_m1[-1], 1.0036279936408656)

        x1 = np.linspace(0, 25, 1000)
        cfunc_m = baseEx_inf.cFunc[0](x1)

        self.assertAlmostEqual(cfunc_m[500], 1.8902146173138235)
        self.assertAlmostEqual(cfunc_m[700], 2.1591451850267176)

        m = np.linspace(0.001, 8, 1000)

        # Use the HARK method derivative to get the derivative of cFunc, and the values are just the MPC
        MPC = baseEx_inf.cFunc[0].derivative(m)

        self.assertAlmostEqual(MPC[500], 0.08415000641504392)
        self.assertAlmostEqual(MPC[700], 0.07173144137912524)


IdiosyncDict = {
    # Parameters shared with the perfect foresight model
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": 1.03,  # Interest factor on assets
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
    "aXtraExtra": [None],  # Additional values to add to aXtraGrid
    # A few other paramaters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 10000,  # Number of agents of this type
    "T_sim": 120,  # Number of periods to simulate
    "aNrmInitMean": -6.0,  # Mean of log initial assets
    "aNrmInitStd": 1.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
}


class testIndShockConsumerTypeExample(unittest.TestCase):
    def test_infinite_horizon(self):
        IndShockExample = IndShockConsumerType(**IdiosyncDict)
        IndShockExample.assign_parameters(
            cycles=0
        )  # Make this type have an infinite horizon
        IndShockExample.solve()

        self.assertAlmostEqual(IndShockExample.solution[0].mNrmStE, 1.5488165705077026)
        self.assertAlmostEqual(
            IndShockExample.solution[0].cFunc.functions[0].x_list[0], -0.25017509
        )

        IndShockExample.track_vars = ["aNrm", "mNrm", "cNrm", "pLvl"]
        IndShockExample.initialize_sim()
        IndShockExample.simulate()

        # simulation test -- seed/generator specific
        #self.assertAlmostEqual(
        #    IndShockExample.history["mNrm"][0][0], 1.0170176090252379
        #)


LifecycleDict = {  # Click arrow to expand this fairly large parameter dictionary
    # Parameters shared with the perfect foresight model
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": 1.03,  # Interest factor on assets
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
    "aXtraExtra": [None],  # Additional values to add to aXtraGrid
    # A few other paramaters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 10,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 10000,  # Number of agents of this type
    "T_sim": 120,  # Number of periods to simulate
    "aNrmInitMean": -6.0,  # Mean of log initial assets
    "aNrmInitStd": 1.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": 11,  # Age after which simulated agents are automatically killed
}


class testIndShockConsumerTypeLifecycle(unittest.TestCase):
    def test_lifecyle(self):
        LifecycleExample = IndShockConsumerType(**LifecycleDict)
        LifecycleExample.cycles = 1
        LifecycleExample.solve()

        self.assertEqual(len(LifecycleExample.solution), 11)

        mMin = np.min(
            [
                LifecycleExample.solution[t].mNrmMin
                for t in range(LifecycleExample.T_cycle)
            ]
        )

        self.assertAlmostEqual(
            LifecycleExample.solution[5].cFunc(3).tolist(), 2.129983771775666
        )


class testIndShockConsumerTypeLifecycleRfree(unittest.TestCase):
    def test_lifecyleRfree(self):
        Rfree = list(np.linspace(1.02, 1.04, 10))
        LifeCycleRfreeDict = LifecycleDict.copy()
        LifeCycleRfreeDict["RFree"] = Rfree

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
    "Rfree": 1.03,  # Interest factor on assets
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
    "aXtraExtra": [None],  # Additional values to add to aXtraGrid
    # A few other paramaters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 4,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 10000,  # Number of agents of this type
    "T_sim": 120,  # Number of periods to simulate
    "aNrmInitMean": -6.0,  # Mean of log initial assets
    "aNrmInitStd": 1.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
}


class testIndShockConsumerTypeCyclical(unittest.TestCase):
    def test_cyclical(self):
        CyclicalExample = IndShockConsumerType(**CyclicalDict)
        CyclicalExample.cycles = 0  # Make this consumer type have an infinite horizon
        CyclicalExample.solve()

        self.assertAlmostEqual(
            CyclicalExample.solution[3].cFunc(3).tolist(), 1.5958390056965004
        )

        CyclicalExample.initialize_sim()
        CyclicalExample.simulate()

        self.assertAlmostEqual(CyclicalExample.state_now["aLvl"][1], 0.41839957)


# %% Tests of 'stable points'


# Create the base infinite horizon parametrization from the "Buffer Stock
# Theory" paper.
bst_params = copy(init_idiosyncratic_shocks)
bst_params["PermGroFac"] = [1.03]  # Permanent income growth factor
bst_params["Rfree"] = 1.04  # Interest factor on assets
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
        decimalPlacesTo = 10
        self.assertAlmostEqual(mNrmStE, 1.37731133865, decimalPlacesTo)
        self.assertAlmostEqual(mNrmTrg, 1.39101653806, decimalPlacesTo)


JACDict = {
    # Parameters shared with the perfect foresight model
    "CRRA": 2,  # Coefficient of relative risk aversion
    "Rfree": 1.05**0.25,  # Interest factor on assets
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
    "aXtraExtra": [None],  # Additional values to add to aXtraGrid
    # A few other parameters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 5000,  # Number of agents of this type
    "T_sim": 100,  # Number of periods to simulate
    "aNrmInitMean": np.log(2) - (0.5**2) / 2,  # Mean of log initial assets
    "aNrmInitStd": 0.5,  # Standard deviation of log initial assets
    "pLvlInitMean": 0,  # Mean of log initial permanent income
    "pLvlInitStd": 0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
}


class testPerfMITShk(unittest.TestCase):
    def jacobian(self):
        class Test_agent(IndShockConsumerType):
            def __init__(self, cycles=0, **kwds):

                IndShockConsumerType.__init__(self, cycles=0, **kwds)

            def get_Rfree(self):
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
                RfreeNow = self.get_Rfree()

                # Calculate new states: normalized market resources and permanent income level
                pLvlNow = (
                    pLvlPrev * self.shocks["PermShk"]
                )  # Updated permanent income level
                # Updated aggregate permanent productivity level
                PlvlAggNow = self.state_prev["PlvlAgg"] * self.PermShkAggNow
                # "Effective" interest factor on normalized assets
                ReffNow = RfreeNow / self.shocks["PermShk"]
                bNrmNow = ReffNow * aNrmPrev  # Bank balances before labor income
                mNrmNow = (
                    bNrmNow + self.shocks["TranShk"]
                )  # Market resources after income

                if self.t_sim == 0:
                    mNrmNow = ss.state_now["mNrm"]
                    pLvlNow = ss.state_now["pLvl"]

                return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None

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
        ss_dx.del_from_time_inv("Rfree")
        ss_dx.add_to_time_vary("Rfree")

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
        example.del_from_time_inv("Rfree")
        example.add_to_time_vary("Rfree")
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

        self.assertAlmostEqual(JACA[175], 6.441930322509393e-06)


dict_harmenberg = {
    # Parameters shared with the perfect foresight model
    "CRRA": 2,  # Coefficient of relative risk aversion
    "Rfree": 1.04**0.25,  # Interest factor on assets
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
    "aXtraExtra": [None],  # Additional values to add to aXtraGrid
    # A few other parameters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 500,  # Number of agents of this type
    "T_sim": 100,  # Number of periods to simulate
    "aNrmInitMean": np.log(1.3) - (0.5**2) / 2,  # Mean of log initial assets
    "aNrmInitStd": 0.5,  # Standard deviation of log initial assets
    "pLvlInitMean": 0,  # Mean of log initial permanent income
    "pLvlInitStd": 0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
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
        #self.assertAlmostEqual(c_std2, 0.0376882, places = 2)
        #self.assertAlmostEqual(c_std1, 0.0044117, places = 2)
        #self.assertAlmostEqual(c_std_ratio, 8.5426941, places = 2)


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
                "Rfree": Rfree,
                "track_vars": ["bNrm", "t_age"],
            }
        )

    def test_NewbornStatesAndShocks(self):
        # Make agent, shock and initial condition histories
        agent = IndShockConsumerType(**self.base_params)
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
                "track_vars": ["t_age", "t_cycle"],
            }
        )

    def test_compare_t_age_t_cycle(self):

        # Make agent, shock and initial condition histories
        agent = IndShockConsumerType(**self.base_params)
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

#%% Test Transition Matrix Methods

class test_Transition_Matrix_Methods(unittest.TestCase):
    def test_calc_tran_matrix(self):
        
        example1 = IndShockConsumerType(**dict_harmenberg)
        example1.cycles= 0
        example1.solve()
        
        example1.define_distribution_grid()
        p = example1.dist_pGrid # Grid of permanent income levels
        
        example1.calc_transition_matrix() 
        c = example1.cPol_Grid # Normalized Consumption Policy Grid
        asset = example1.aPol_Grid # Normalized Asset Policy Grid
        
        example1.calc_ergodic_dist()
        vecDstn = example1.vec_erg_dstn # Distribution of market resources and permanent income as a vector (m*p)x1 vector where 
        
        
        #Compute Aggregate Consumption and Aggregate Assets
        gridc = np.zeros( (len(c),len(p)) )
        grida = np.zeros( (len(asset),len(p)) )
        
        for j in range(len(p)):
            gridc[:,j] = p[j]*c # unnormalized Consumption policy grid
            grida[:,j] = p[j]*asset # unnormalized Asset policy grid
            
        AggC = np.dot(gridc.flatten(), vecDstn) #Aggregate Consumption
        AggA = np.dot(grida.flatten(), vecDstn) #Aggregate Assets
        
        
              
        self.assertAlmostEqual(AggA[0],  1.1951311747835132, places =4) 
        self.assertAlmostEqual(AggC[0], 1.0041701073134557, places = 4)

#%% Test Heterogenous Agent Jacobian Methods


class test_Jacobian_methods(unittest.TestCase):
    def test_calc_jacobian(self):
        
        Agent = IndShockConsumerType(**dict_harmenberg)
        Agent.compute_steady_state()
        
        CJAC_Perm,AJAC_Perm = Agent.calc_jacobian('PermShkStd',50)

              
        self.assertAlmostEqual(CJAC_Perm.T[30][29],  -0.061180723891496314) 
        self.assertAlmostEqual(CJAC_Perm.T[30][30],  0.05308804499781772) 
        self.assertAlmostEqual(CJAC_Perm.T[30][31],  0.04675702789391778) 
