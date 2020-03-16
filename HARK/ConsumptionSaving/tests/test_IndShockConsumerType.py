from HARK.ConsumptionSaving.ConsIndShockModel import \
    IndShockConsumerType, ConsIndShockSolverBasic
import HARK.ConsumptionSaving.ConsumerParameters as Params
import numpy as np
import unittest

class testIndShockConsumerType(unittest.TestCase):

    def setUp(self):
        self.agent = IndShockConsumerType(
            AgentCount = 2,
            T_sim = 10
        )

        self.agent.solve()

    def test_getShocks(self):
        self.agent.initializeSim()
        self.agent.simBirth(np.array([True,False]))
        self.agent.simOnePeriod()
        self.agent.simBirth(np.array([False,True]))

        self.agent.getShocks()

        self.assertEqual(self.agent.PermShkNow[0],
                         1.0050166461586711)
        self.assertEqual(self.agent.PermShkNow[1],
                         1.0050166461586711)
        self.assertEqual(self.agent.TranShkNow[0],
                         1.1176912196531754)

    def test_ConsIndShockSolverBasic(self):
        LifecycleExample = IndShockConsumerType(
            **Params.init_lifecycle)
        LifecycleExample.cycles = 1
        LifecycleExample.solve()

        # test the solution_terminal
        self.assertAlmostEqual(
            LifecycleExample.solution[10].cFunc(2).tolist(),
            2)

        solver = ConsIndShockSolverBasic(
            LifecycleExample.solution[-2],
            LifecycleExample.IncomeDstn[0],
            LifecycleExample.LivPrb[0],
            LifecycleExample.DiscFac,
            LifecycleExample.CRRA,
            LifecycleExample.Rfree,
            LifecycleExample.PermGroFac[0],
            LifecycleExample.BoroCnstArt,
            LifecycleExample.aXtraGrid,
            LifecycleExample.vFuncBool,
            LifecycleExample.CubicBool)

        solver.prepareToSolve()

        self.assertAlmostEqual(solver.DiscFacEff,
                               0.9503999999999999)
        self.assertAlmostEqual(solver.PermShkMinNext,
                               0.850430160026919)
        self.assertAlmostEqual(solver.cFuncNowCnst(4).tolist(),
                               4.0)
        self.assertAlmostEqual(solver.prepareToCalcEndOfPrdvP()[0],
                               -0.2491750859108316)
        self.assertAlmostEqual(solver.prepareToCalcEndOfPrdvP()[-1],
                               19.74982491408914)

        EndOfPrdvP = solver.calcEndOfPrdvP()

        self.assertAlmostEqual(EndOfPrdvP[0],
                               6622.251864311334)
        self.assertAlmostEqual(EndOfPrdvP[-1],
                               0.026301061207747087)

        solution = solver.makeBasicSolution(EndOfPrdvP,
                                            solver.aNrmNow,
                                            solver.makeLinearcFunc)
        solver.addMPCandHumanWealth(solution)

        self.assertAlmostEqual(solution.cFunc(4).tolist(),
                               1.7391265696400773)

    def test_simulated_values(self):
        self.agent.initializeSim()
        self.agent.simulate()

        print(self.agent.aLvlNow)

        self.assertAlmostEqual(self.agent.MPCnow[1],
                               0.5535801655448935)

        self.assertAlmostEqual(self.agent.aLvlNow[1],
                               0.18832361)


class testBufferStock(unittest.TestCase):
    """ Tests of the results of the BufferStock REMARK.
    """
    
    def setUp(self):
        # Make a dictionary containing all parameters needed to solve the model
        self.base_params = Params.init_idiosyncratic_shocks

        # Set the parameters for the baseline results in the paper
        # using the variable values defined in the cell above
        self.base_params['PermGroFac'] = [1.03]
        self.base_params['Rfree']      = 1.04
        self.base_params['DiscFac']    = 0.96
        self.base_params['CRRA']       = 2.00
        self.base_params['UnempPrb']   = 0.005
        self.base_params['IncUnemp']   = 0.0
        self.base_params['PermShkStd'] = [0.1]
        self.base_params['TranShkStd'] = [0.1]
        self.base_params['LivPrb']       = [1.0]
        self.base_params['CubicBool']    = True
        self.base_params['T_cycle']      = 1
        self.base_params['BoroCnstArt']  = None

    def test_baseEx(self):
        baseEx = IndShockConsumerType(**self.base_params)
        baseEx.cycles = 100   # Make this type have a finite horizon (Set T = 100)

        baseEx.solve()
        baseEx.unpackcFunc()

        m = np.linspace(0,9.5,1000)

        c_m  = baseEx.cFunc[0](m)
        c_t1 = baseEx.cFunc[-2](m)
        c_t5 = baseEx.cFunc[-6](m)
        c_t10 = baseEx.cFunc[-11](m)

        self.assertAlmostEqual(c_m[500], 1.4008090582203356)
        self.assertAlmostEqual(c_t1[500], 2.9227437159255216)
        self.assertAlmostEqual(c_t5[500], 1.7350607327187664)
        self.assertAlmostEqual(c_t10[500], 1.4991390649979213)
        self.assertAlmostEqual(c_t10[600], 1.6101476268581576)
        self.assertAlmostEqual(c_t10[700], 1.7196531041366991)

    def test_GICFails(self):
        GIC_fail_dictionary = dict(self.base_params)
        GIC_fail_dictionary['Rfree']      = 1.08
        GIC_fail_dictionary['PermGroFac'] = [1.00]

        GICFailExample = IndShockConsumerType(
            cycles=0, # cycles=0 makes this an infinite horizon consumer
            **GIC_fail_dictionary)

        GICFailExample.solve()
        GICFailExample.unpackcFunc()
        m = np.linspace(0,5,1000)
        c_m = GICFailExample.cFunc[0](m)

        self.assertAlmostEqual(c_m[500], 0.7772637042393458)
        self.assertAlmostEqual(c_m[700], 0.8392649061916746)

        self.assertFalse(GICFailExample.GICPF)

    def test_infinite_horizon(self):
        baseEx_inf = IndShockConsumerType(cycles=0,
                                          **self.base_params)

        baseEx_inf.solve()
        baseEx_inf.unpackcFunc()

        m1 = np.linspace(1,baseEx_inf.solution[0].mNrmSS,50) # m1 defines the plot range on the left of target m value (e.g. m <= target m)
        c_m1 = baseEx_inf.cFunc[0](m1)

        self.assertAlmostEqual(c_m1[0], 0.8527887545025995)
        self.assertAlmostEqual(c_m1[-1], 1.0036279936408656)

        x1 = np.linspace(0,25,1000)
        cfunc_m = baseEx_inf.cFunc[0](x1)

        self.assertAlmostEqual(cfunc_m[500], 1.8902146173138235)
        self.assertAlmostEqual(cfunc_m[700], 2.1591451850267176)

        m = np.linspace(0.001,8,1000)

        # Use the HARK method derivative to get the derivative of cFunc, and the values are just the MPC
        MPC = baseEx_inf.cFunc[0].derivative(m)

        self.assertAlmostEqual(MPC[500], 0.08415000641504392)
        self.assertAlmostEqual(MPC[700], 0.07173144137912524)


IdiosyncDict={
    # Parameters shared with the perfect foresight model
    "CRRA": 2.0,                           # Coefficient of relative risk aversion
    "Rfree": 1.03,                         # Interest factor on assets
    "DiscFac": 0.96,                       # Intertemporal discount factor
    "LivPrb" : [0.98],                     # Survival probability
    "PermGroFac" :[1.01],                  # Permanent income growth factor
        
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : [0.1],                  # Standard deviation of log permanent shocks to income
    "PermShkCount" : 7,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [0.2],                  # Standard deviation of log transitory shocks to income
    "TranShkCount" : 7,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" : 0.3,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.0,                      # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid

    # A few other paramaters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution   
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 1,                         # Number of periods in the cycle for this agent type        

    # Parameters only used in simulation
    "AgentCount" : 10000,                  # Number of agents of this type
    "T_sim" : 120,                         # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
}

class testIndShockConsumerTypeExample(unittest.TestCase):

    def test_infinite_horizon(self):
        IndShockExample = IndShockConsumerType(**IdiosyncDict)
        IndShockExample.cycles = 0 # Make this type have an infinite horizon
        IndShockExample.solve()

        self.assertAlmostEqual(IndShockExample.solution[0].mNrmSS,
                               1.5488165705077026)
        self.assertAlmostEqual(IndShockExample.solution[0].cFunc.functions[0].x_list[0],
                               -0.25017509)

        IndShockExample.track_vars = ['aNrmNow','mNrmNow','cNrmNow','pLvlNow']
        IndShockExample.initializeSim()
        IndShockExample.simulate()

        self.assertAlmostEqual(IndShockExample.mNrmNow_hist[0][0],
                               1.0170176090252379)

LifecycleDict={ # Click arrow to expand this fairly large parameter dictionary
    # Parameters shared with the perfect foresight model
    "CRRA": 2.0,                           # Coefficient of relative risk aversion
    "Rfree": 1.03,                         # Interest factor on assets
    "DiscFac": 0.96,                       # Intertemporal discount factor
    "LivPrb" : [0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],
    "PermGroFac" : [1.01,1.01,1.01,1.02,1.02,1.02,0.7,1.0,1.0,1.0],
    
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : [0.1,0.2,0.1,0.2,0.1,0.2,0.1,0,0,0],
    "PermShkCount" : 7,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [0.3,0.2,0.1,0.3,0.2,0.1,0.3,0,0,0],
    "TranShkCount" : 7,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" : 0.3,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 7,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.0,                      # Flat income tax rate (legacy parameter, will be removed in future)
    
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid
    
    # A few other paramaters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution   
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 10,                        # Number of periods in the cycle for this agent type        
    
    # Parameters only used in simulation
    "AgentCount" : 10000,                  # Number of agents of this type
    "T_sim" : 120,                         # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : 11,                          # Age after which simulated agents are automatically killed     
}

class testIndShockConsumerTypeLifecycle(unittest.TestCase):

    def test_lifecyle(self):
        LifecycleExample = IndShockConsumerType(**LifecycleDict)
        LifecycleExample.cycles = 1
        LifecycleExample.solve()

        self.assertEquals(len(LifecycleExample.solution), 11)

        mMin = np.min([LifecycleExample.solution[t].mNrmMin for t in
                       range(LifecycleExample.T_cycle)])

        self.assertAlmostEqual(LifecycleExample.solution[5].cFunc(3).tolist(),
                               2.129983771775666)

CyclicalDict = { 
    # Parameters shared with the perfect foresight model
    "CRRA": 2.0,                           # Coefficient of relative risk aversion
    "Rfree": 1.03,                         # Interest factor on assets
    "DiscFac": 0.96,                       # Intertemporal discount factor
    "LivPrb" : 4*[0.98],                   # Survival probability
    "PermGroFac" : [1.082251, 2.8, 0.3, 1.1],
    
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd" : [0.1,0.1,0.1,0.1],
    "PermShkCount" : 7,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [0.2,0.2,0.2,0.2],
    "TranShkCount" : 7,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05,                     # Probability of unemployment while working
    "IncUnemp" : 0.3,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : 0.0,                      # Flat income tax rate (legacy parameter, will be removed in future)
    
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 20,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid
    
    # A few other paramaters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution   
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 4,                         # Number of periods in the cycle for this agent type        
    
    # Parameters only used in simulation
    "AgentCount" : 10000,                  # Number of agents of this type
    "T_sim" : 120,                         # Number of periods to simulate
    "aNrmInitMean" : -6.0,                 # Mean of log initial assets
    "aNrmInitStd"  : 1.0,                  # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed     
}


class testIndShockConsumerTypeLifecycle(unittest.TestCase):

    def test_lifecyle(self):
        CyclicalExample = IndShockConsumerType(**CyclicalDict)
        CyclicalExample.cycles = 0 # Make this consumer type have an infinite horizon
        CyclicalExample.solve()

        self.assertAlmostEqual(CyclicalExample.solution[3].cFunc(3).tolist(),
                               1.5958390056965004)
