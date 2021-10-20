import unittest
from copy import copy
import numpy as np
from HARK.ConsumptionSaving.ConsIndShockModel import (
    init_lifecycle,
    init_idiosyncratic_shocks,
)
from HARK.ConsumptionSaving.ConsIndShockModelFast import IndShockConsumerTypeFast
from HARK.ConsumptionSaving.tests.test_IndShockConsumerType import (
    CyclicalDict,
    LifecycleDict,
    IdiosyncDict,
)


class testIndShockConsumerTypeFast(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockConsumerTypeFast(AgentCount=2, T_sim=10)

        self.agent.solve()

    def test_get_shocks(self):
        self.agent.initialize_sim()
        self.agent.sim_birth(np.array([True, False]))
        self.agent.sim_one_period()
        self.agent.sim_birth(np.array([False, True]))

        self.agent.get_shocks()

        self.assertEqual(self.agent.shocks['PermShk'][0], 1.0427376294215103)
        self.assertAlmostEqual(self.agent.shocks['PermShk'][1], 0.9278094171517413)
        self.assertAlmostEqual(self.agent.shocks['TranShk'][0], 0.881761797501595)

    def test_ConsIndShockSolverBasic(self):
        LifecycleExample = IndShockConsumerTypeFast(**init_lifecycle)
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

    def test_simulated_values(self):
        self.agent.initialize_sim()
        self.agent.simulate()
        self.assertAlmostEqual(self.agent.MPCnow[1], 0.5711503906043797)

        self.assertAlmostEqual(self.agent.state_now['aLvl'][1], 0.18438326264597635)


class testBufferStock(unittest.TestCase):
    """ Tests of the results of the BufferStock REMARK.
    """

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
        baseEx = IndShockConsumerTypeFast(**self.base_params)
        baseEx.cycles = 100  # Make this type have a finite horizon (Set T = 100)

        baseEx.solve()
        baseEx.unpack_cFunc()

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
        GICRaw_fail_dictionary['cycles'] = 0 # cycles=0 makes this an infinite horizon consumer

        GICRawFailExample = IndShockConsumerTypeFast(
            **GICRaw_fail_dictionary
        )

        GICRawFailExample.solve()
        GICRawFailExample.unpack_cFunc()
        m = np.linspace(0, 5, 1000)
        c_m = GICRawFailExample.cFunc[0](m)

        self.assertAlmostEqual(c_m[500], 0.7772637042393458)
        self.assertAlmostEqual(c_m[700], 0.8392649061916746)

        self.assertFalse(GICRawFailExample.conditions["GICRaw"])

    def test_infinite_horizon(self):
        baseEx_inf = IndShockConsumerTypeFast(**self.base_params)
        baseEx_inf.assign_parameters(cycles = 0)
        baseEx_inf.solve()
        baseEx_inf.unpack_cFunc()

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


class testIndShockConsumerTypeFastExample(unittest.TestCase):
    def test_infinite_horizon(self):
        IndShockExample = IndShockConsumerTypeFast(**IdiosyncDict)
        IndShockExample.assign_parameters(cycles = 0)  # Make this type have an infinite horizon
        IndShockExample.solve()

        self.assertAlmostEqual(IndShockExample.solution[0].mNrmStE, 1.5488165705077026)
        self.assertAlmostEqual(
            IndShockExample.solution[0].cFunc.functions[0].x_list[0], -0.25017509
        )

        IndShockExample.track_vars = ['aNrm', "mNrm", 'cNrm', 'pLvl']
        IndShockExample.initialize_sim()
        IndShockExample.simulate()

        self.assertAlmostEqual(
            IndShockExample.history["mNrm"][0][0], 1.0170176090252379
        )


class testIndShockConsumerTypeFastLifecycle(unittest.TestCase):
    def test_lifecyle(self):
        LifecycleExample = IndShockConsumerTypeFast(**LifecycleDict)
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


class testIndShockConsumerTypeFastCyclical(unittest.TestCase):
    def test_cyclical(self):
        CyclicalExample = IndShockConsumerTypeFast(**CyclicalDict)
        CyclicalExample.cycles = 0  # Make this consumer type have an infinite horizon
        CyclicalExample.solve()

        self.assertAlmostEqual(
            CyclicalExample.solution[3].cFunc(3).tolist(), 1.5958390056965004
        )
