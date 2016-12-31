"""
This file implements unit tests for several of the ConsumptionSaving models in HARK.

These tests compare the output of different models in specific cases in which those models
should yield the same output.  The code will pass these tests if and only if the output is close
"enough".
"""



# First, tell Python what directories we will be using
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))
sys.path.insert(0, os.path.abspath('./'))

# Bring in modules we need
import unittest
from copy import deepcopy
import numpy as np


# Bring in the HARK models we want to test
from ConsIndShockModel import solvePerfForesight, IndShockConsumerType
from ConsMarkovModel import MarkovConsumerType
from TractableBufferStockModel import TractableConsumerType


class Compare_PerfectForesight_and_Infinite(unittest.TestCase):
    """
    Class to compare output of the perfect foresight and infinite horizon models.
    
    When income uncertainty is removed from the infinite horizon model, it reduces in theory to
    the perfect foresight model.  This class implements tests to make sure it reduces in practice
    to the perfect foresight model as well.
    """
    
    def setUp(self):
        """
        Prepare to compare the models by initializing and solving them
        """
        # Set up and solve infinite type
        import ConsumerParameters as Params
        
        InfiniteType = IndShockConsumerType(**Params.init_idiosyncratic_shocks)
        InfiniteType.assignParameters(LivPrb      = [1.],
                                      DiscFac     = 0.955,
                                      PermGroFac  = [1.],
                                      PermShkStd  = [0.],
                                      TempShkStd  = [0.],
                                      T_total = 1, T_retire = 0, BoroCnstArt = None, UnempPrb = 0.,
                                      cycles = 0) # This is what makes the type infinite horizon

        InfiniteType.updateIncomeProcess()        
        InfiniteType.solve()
        InfiniteType.timeFwd()
        InfiniteType.unpackcFunc()


        # Make and solve a perfect foresight consumer type with the same parameters
        PerfectForesightType = deepcopy(InfiniteType)    
        PerfectForesightType.solveOnePeriod = solvePerfForesight
        
        PerfectForesightType.solve()
        PerfectForesightType.unpackcFunc()
        PerfectForesightType.timeFwd()

        self.InfiniteType         = InfiniteType
        self.PerfectForesightType = PerfectForesightType


    def test_consumption(self):
        """"
        Now compare the consumption functions and make sure they are "close"
        """
        diffFunc       = lambda m : self.PerfectForesightType.solution[0].cFunc(m) - \
                                    self.InfiniteType.cFunc[0](m)
        points         = np.arange(0.5,10.,.01)
        difference     = diffFunc(points)
        max_difference = np.max(np.abs(difference))

        self.assertLess(max_difference,0.01)



class Compare_TBS_and_Markov(unittest.TestCase):
    """
    Class to compare output of the Tractable Buffer Stock and Markov models.
    
    The only uncertainty in the TBS model is over when the agent will enter an absorbing state 
    with 0 income.  With the right transition arrays and income processes, this is just a special
    case of the Markov model.  So with the right inputs, we should be able to solve the two
    different models and get the same outputs.
    """
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
        MrkvArray = np.array([[1.0-base_primitives['UnempPrb'],base_primitives['UnempPrb']],
                              [0.0,1.0]])
        Markov_primitives = {"CRRA":base_primitives['CRRA'],
                            "Rfree":np.array(2*[base_primitives['Rfree']]),
                            "PermGroFac":[np.array(2*[base_primitives['PermGroFac']/
                                          (1.0-base_primitives['UnempPrb'])])],
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
                            "aXtraNestFac":3,
                            "LivPrb":[1.0],
                            "DiscFac":base_primitives['DiscFac'],
                            'Nagents':1,
                            'psi_seed':0,
                            'xi_seed':0,
                            'unemp_seed':0,
                            'tax_rate':0.0,
                            'vFuncBool':False,
                            'CubicBool':True,
                            'MrkvArray':MrkvArray
                            }
                
        MarkovType             = MarkovConsumerType(**Markov_primitives)                           
        MarkovType.cycles      = 0
        employed_income_dist   = [np.ones(1),np.ones(1),np.ones(1)]
        unemployed_income_dist = [np.ones(1),np.ones(1),np.zeros(1)]
        MarkovType.IncomeDstn  = [[employed_income_dist,unemployed_income_dist]]
        
        MarkovType.solve()
        MarkovType.unpackcFunc()
 
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
