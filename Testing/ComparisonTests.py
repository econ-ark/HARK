# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 13:29:44 2016

@author: lowd
"""
<<<<<<< HEAD

import sys 
sys.path.insert(0,'../')
sys.path.insert(0,'../ConsumptionSavingModel')
sys.path.insert(0,'../TractableBufferStock')

from ConsumptionSavingModel import ConsumerType, solveConsumptionSavingMarkov, consumptionSavingSolverENDG, solvePerfForesight
from TractableBufferStock import TractableConsumerType
=======
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSavingModel'))
sys.path.insert(0, os.path.abspath('./'))

from ConsIndShockModel import solvePerfForesight, IndShockConsumerType
from ConsMarkovModel import MarkovConsumerType
from TractableBufferStockModel import TractableConsumerType
>>>>>>> eeb37f24755d0c683c9d9efbe5e7447425c98b86
from copy import deepcopy

import unittest
import numpy as np

class Compare_PerfectForesight_and_Infinite(unittest.TestCase):
    
    def setUp(self):

        # Set up and solve infinite type
<<<<<<< HEAD
        import SetupConsumerParameters as Params
        
        InfiniteType = ConsumerType(**Params.init_consumer_objects)
        InfiniteType.solveOnePeriod = consumptionSavingSolverENDG
        InfiniteType.assignParameters(LivPrb = [1.],
                                      DiscFac = [0.955],
=======
        import ConsumerParameters as Params
        
        InfiniteType = IndShockConsumerType(**Params.init_idiosyncratic_shocks)
        InfiniteType.assignParameters(LivPrb = [1.],
                                      DiscFac = 0.955,
>>>>>>> eeb37f24755d0c683c9d9efbe5e7447425c98b86
                                      PermGroFac = [1.],
                                      PermShkStd  = [0.],
                                      TempShkStd  = [0.],
                                      T_total = 1, T_retire = 0, BoroCnstArt = None, UnempPrb = 0.,
                                      cycles = 0) # This is what makes the type infinite horizon

        InfiniteType.updateIncomeProcess()        
        InfiniteType.solve()
        InfiniteType.timeFwd()
<<<<<<< HEAD
        InfiniteType.unpack_cFunc()
=======
        InfiniteType.unpackcFunc()
>>>>>>> eeb37f24755d0c683c9d9efbe5e7447425c98b86


        # Make and solve a perfect foresight consumer type
        PerfectForesightType = deepcopy(InfiniteType)    
        PerfectForesightType.solveOnePeriod = solvePerfForesight
        
        PerfectForesightType.solve()
<<<<<<< HEAD
        PerfectForesightType.unpack_cFunc()
=======
        PerfectForesightType.unpackcFunc()
>>>>>>> eeb37f24755d0c683c9d9efbe5e7447425c98b86
        PerfectForesightType.timeFwd()

        self.InfiniteType = InfiniteType
        self.PerfectForesightType = PerfectForesightType


    def test_consumption(self):
        # Now compare the consumption functions and make sure they are "close"

        diffFunc       = lambda m : self.PerfectForesightType.solution[0].cFunc(m) - self.InfiniteType.cFunc[0](m)
        points         = np.arange(0.5,10.,.01)
        difference     = diffFunc(points)
        max_difference = np.max(np.abs(difference))

        self.assertLess(max_difference,0.01)



class Compare_TBS_and_Markov(unittest.TestCase):
    
    def setUp(self):
        # Set up and solve TBS
        base_primitives = {'UnempPrb' : .015,
                           'DiscFac' : 0.9,
                           'Rfree' : 1.1,
                           'PermGroFac' : 1.05,
                           'CRRA' : .95}
                           

        TBSType = TractableConsumerType(**base_primitives)
        TBSType.solve()
 
        # Set up and solve Markov
<<<<<<< HEAD
=======
        MrkvArray = np.array([[1.0-base_primitives['UnempPrb'],base_primitives['UnempPrb']],[0.0,1.0]])
>>>>>>> eeb37f24755d0c683c9d9efbe5e7447425c98b86
        Markov_primitives = {"CRRA":base_primitives['CRRA'],
                            "Rfree":np.array(2*[base_primitives['Rfree']]),
                            "PermGroFac":[np.array(2*[base_primitives['PermGroFac']/(1.0-base_primitives['UnempPrb'])])],
                            "BoroCnstArt":None,
                            "PermShkStd":[0.0],
                            "PermShkCount":1,
                            "TranShkStd":[0.0],
                            "TranShkCount":1,
                            "T_total":1,
                            "UnempPrb":0.0,
                            "UnempPrbRet":0.0,
                            "T_retire":0,
                            "IncUnemp":0.0,
                            "IncUnempRet":0.0,
                            "aXtraMin":0.001,
                            "aXtraMax":TBSType.mUpperBnd,
                            "aXtraCount":48,
                            "aXtraExtra":[None],
                            "exp_nest":3,
                            "LivPrb":[1.0],
<<<<<<< HEAD
                            "DiscFac":[base_primitives['DiscFac']],
=======
                            "DiscFac":base_primitives['DiscFac'],
>>>>>>> eeb37f24755d0c683c9d9efbe5e7447425c98b86
                            'Nagents':1,
                            'psi_seed':0,
                            'xi_seed':0,
                            'unemp_seed':0,
                            'tax_rate':0.0,
                            'vFuncBool':False,
<<<<<<< HEAD
                            'CubicBool':True
                            }
        
        
        MarkovType = ConsumerType(**Markov_primitives)                           
        MrkvArray                            = np.array([[1.0-base_primitives['UnempPrb'],
                                                          base_primitives['UnempPrb']],[0.0,1.0]])
        employed_income_dist                 = [np.ones(1),np.ones(1),np.ones(1)]
        unemployed_income_dist               = [np.ones(1),np.ones(1),np.zeros(1)]
        MarkovType.solution_terminal.cFunc   = 2*[MarkovType.solution_terminal.cFunc]
        MarkovType.solution_terminal.vFunc   = 2*[MarkovType.solution_terminal.vFunc]
        MarkovType.solution_terminal.vPfunc  = 2*[MarkovType.solution_terminal.vPfunc]
        MarkovType.solution_terminal.vPPfunc = 2*[MarkovType.solution_terminal.vPPfunc]
        MarkovType.solution_terminal.mNrmMin = 2*[MarkovType.solution_terminal.mNrmMin]
        MarkovType.solution_terminal.MPCmax  = np.array(2*[MarkovType.solution_terminal.MPCmax])
        MarkovType.IncomeDstn = [[employed_income_dist,unemployed_income_dist]]
        MarkovType.MrkvArray = MrkvArray
        MarkovType.time_inv.append('MrkvArray')
        MarkovType.solveOnePeriod = solveConsumptionSavingMarkov
        MarkovType.cycles = 0
        MarkovType.solve()
        MarkovType.unpack_cFunc()
=======
                            'CubicBool':True,
                            'MrkvArray':MrkvArray
                            }
                
        MarkovType = MarkovConsumerType(**Markov_primitives)                           
        MarkovType.cycles = 0
        employed_income_dist                 = [np.ones(1),np.ones(1),np.ones(1)]
        unemployed_income_dist               = [np.ones(1),np.ones(1),np.zeros(1)]
        MarkovType.IncomeDstn = [[employed_income_dist,unemployed_income_dist]]
        
        MarkovType.solve()
        MarkovType.unpackcFunc()
>>>>>>> eeb37f24755d0c683c9d9efbe5e7447425c98b86
 
        self.TBSType    = TBSType
        self.MarkovType = MarkovType

    def test_consumption(self):
        # Now compare the consumption functions and make sure they are "close"

        diffFunc       = lambda m : self.TBSType.solution[0].cFunc(m) - self.MarkovType.cFunc[0][0](m)
        points         = np.arange(0.1,10.,.01)
        difference     = diffFunc(points)
        max_difference = np.max(np.abs(difference))

        self.assertLess(max_difference,0.01)
        
if __name__ == '__main__':
    # Run all the tests    
    unittest.main()
