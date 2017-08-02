'''
This module contains models for solving representative agent macroeconomic models.
This stands in contrast to all other model modules in HARK, which (unsurprisingly)
take a heterogeneous agents approach.  In these models, all attributes are either
time invariant or exist on a short cycle.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
from HARKinterpolation import LinearInterp
from ConsIndShockModel import IndShockConsumerType, ConsumerSolution, MargValueFunc

def solveConsRepAgent(solution_next,DiscFac,CRRA,IncomeDstn,CapShare,DeprFac,PermGroFac,aXtraGrid,SocPlannerBool):
    '''
    Solve one period of the simple representative agent consumption-saving model.
    
    Parameters
    ----------
    solution_next : ConsumerSolution
        Solution to the next period's problem (i.e. previous iteration). 
    DiscFac : float
        Intertemporal discount factor for future utility.        
    CRRA : float
        Coefficient of relative risk aversion.
    IncomeDstn : [np.array]
        A list containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.
    CapShare : float
        Capital's share of income in Cobb-Douglas production function.
    DeprFac : float
        Depreciation rate of capital.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.  In this model, the minimum acceptable
        level is always zero.
    SocPlannerBool : bool
        Indicator for whether to solve as a representative agent or the social
        planner.  The social planner (True) recognizes that saving more at the
        end of the period will change the factor prices in the next period, but
        the representative agent (False) does not.
        
    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's problem (new iteration).
    '''
    # Unpack next period's solution and the income distribution
    vPfuncNext      = solution_next.vPfunc
    ShkPrbsNext     = IncomeDstn[0]
    PermShkValsNext = IncomeDstn[1]
    TranShKValsNext = IncomeDstn[2]
    
    # Make tiled versions of end-of-period assets, shocks, and probabilities
    aNrmNow     = aXtraGrid
    aNrmCount   = aNrmNow.size
    ShkCount    = ShkPrbsNext.size
    aNrm_tiled  = np.tile(np.reshape(aNrmNow,(aNrmCount,1)),(1,ShkCount))

    # Tile arrays of the income shocks and put them into useful shapes
    PermShkVals_tiled = np.tile(np.reshape(PermShkValsNext,(1,ShkCount)),(aNrmCount,1))
    TranShkVals_tiled = np.tile(np.reshape(TranShKValsNext,(1,ShkCount)),(aNrmCount,1))
    ShkPrbs_tiled     = np.tile(np.reshape(ShkPrbsNext,(1,ShkCount)),(aNrmCount,1))
    
    # Calculate next period's capital-to-permanent-labor ratio under each combination
    # of end-of-period assets and shock realization
    kNrmNext = aNrm_tiled/(PermGroFac*PermShkVals_tiled)
    
    # Calculate end-of-period marginal value from perspective of social planner or rep agent
    if SocPlannerBool: # For social planner:
        print('Social planner solution not yet implemented!')
    else: # For representative agent:
        # Calculate next period's market resources
        RfreeNext = 1. + CapShare*kNrmNext**(CapShare-1.)*TranShkVals_tiled**(1.-CapShare) - (1. - DeprFac)
        wRteNext  = (1.-CapShare)*kNrmNext**CapShare**TranShkVals_tiled**(-CapShare)
        mNrmNext  = RfreeNext*kNrmNext + wRteNext*TranShkVals_tiled
        
        # Calculate end-of-period marginal value of assets for the RA
        vPnext = vPfuncNext(mNrmNext)
        EndOfPrdvP = DiscFac*np.sum(RfreeNext*(PermGroFac*PermShkVals_tiled)**(-CRRA)*vPnext*ShkPrbs_tiled,axis=1)
        
    # Invert the first order condition to get consumption, then find endogenous gridpoints
    cNrmNow = EndOfPrdvP**(-1./CRRA)
    mNrmNow = aNrmNow + cNrmNow
    
    # Construct the consumption function and the marginal value function
    cFuncNow  = LinearInterp(np.insert(mNrmNow,0,0.0),np.insert(cNrmNow,0,0.0))
    vPfuncNow = MargValueFunc(cFuncNow,CRRA)
    
    # Construct and return the solution for this period
    solution_now = ConsumerSolution(cFunc=cFuncNow,vPfunc=vPfuncNow)
    return solution_now



class RepAgentConsumerType(IndShockConsumerType):
    '''
    A class for representing representative agents with inelastic labor supply.
    '''
    time_inv_ = IndShockConsumerType.time_inv_ + ['CapShare','DeprFac','SocPlannerBool']
    
    def __init__(self,time_flow=True,**kwds):
        '''
        Make a new instance of a representative agent.
        
        Parameters
        ----------
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.
        
        Returns
        -------
        None
        '''       
        IndShockConsumerType.__init__(self,cycles=0,time_flow=time_flow,**kwds)
        self.AgentCount = 1 # Hardcoded, because this is rep agent
        self.solveOnePeriod = solveConsRepAgent
        
    def getStates(self):
        '''
        Calculates updated values of normalized market resources and permanent income level.
        Uses pLvlNow, aNrmNow, PermShkNow, TranShkNow.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        pLvlPrev = self.pLvlNow
        aNrmPrev = self.aNrmNow
        
        # Calculate new states: normalized market resources and permanent income level
        self.pLvlNow = pLvlPrev*self.PermShkNow # Updated permanent income level
        self.kNrmNow = aNrmPrev/self.PermShkNow
        self.Rfree = 1. + self.CapShare*self.kNrmNow**(self.CapShare-1.)*self.TranShkNow**(1.-self.CapShare) - (1. - self.DeprFac)
        self.wRte  = (1.-self.CapShare)*self.kNrmNow**self.CapShare**self.TranShkNow**(-self.CapShare)
        self.mNrmNow = self.Rfree*self.kNrmNow + self.wRte*self.TranShkNow
        
        
        
        
if __name__ == '__main__':
    from copy import deepcopy
    from time import clock
    from HARKutilities import plotFuncs
    import ConsumerParameters as Params
    import matplotlib.pyplot as plt
    
    # Make a quick example dictionary
    RAparams = deepcopy(Params.init_idiosyncratic_shocks)
    RAparams['DeprFac'] = 0.95
    RAparams['CapShare'] = 0.36
    RAparams['UnempPrb'] = 0.0
    RAparams['LivPrb'] = [1.0]
    RAparams['SocPlannerBool'] = False
    
    # Make and solve a rep agent model
    RAexample = RepAgentConsumerType(**RAparams)
    t_start = clock()
    RAexample.solve()
    t_end = clock()
    print('Solving a representative agent problem took ' + str(t_end-t_start) + ' seconds.')
    plotFuncs(RAexample.solution[0].cFunc,0,10)
    
    # Simulate the representative agent model
    RAexample.T_sim = 2000
    RAexample.track_vars = ['cNrmNow','mNrmNow','Rfree','wRte']
    RAexample.initializeSim()
    t_start = clock()
    RAexample.simulate()
    t_end = clock()
    print('Simulating a representative agent for ' + str(RAexample.T_sim) + ' periods took ' + str(t_end-t_start) + ' seconds.')
    