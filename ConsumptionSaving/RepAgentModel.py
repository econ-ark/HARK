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
from HARKsimulation import drawUniform, drawDiscrete
from ConsIndShockModel import IndShockConsumerType, ConsumerSolution, MargValueFunc

def solveConsRepAgent(solution_next,DiscFac,CRRA,IncomeDstn,CapShare,DeprFac,PermGroFac,aXtraGrid):
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
    
    # Calculate next period's market resources
    KtoLnext  = kNrmNext/TranShkVals_tiled
    RfreeNext = 1. - DeprFac + CapShare*KtoLnext**(CapShare-1.)
    wRteNext  = (1.-CapShare)*KtoLnext**CapShare
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



def solveConsRepAgentMarkov(solution_next,MrkvArray,DiscFac,CRRA,IncomeDstn,CapShare,DeprFac,PermGroFac,aXtraGrid):
    '''
    Solve one period of the simple representative agent consumption-saving model.
    This version supports a discrete Markov process.
    
    Parameters
    ----------
    solution_next : ConsumerSolution
        Solution to the next period's problem (i.e. previous iteration). 
    MrkvArray : np.array
        Markov transition array between this period and next period.
    DiscFac : float
        Intertemporal discount factor for future utility.        
    CRRA : float
        Coefficient of relative risk aversion.
    IncomeDstn : [[np.array]]
        A list of lists containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.
    CapShare : float
        Capital's share of income in Cobb-Douglas production function.
    DeprFac : float
        Depreciation rate of capital.
    PermGroFac : [float]
        Expected permanent income growth factor for each state we could be in
        next period.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.  In this model, the minimum acceptable
        level is always zero.
        
    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's problem (new iteration).
    '''
    # Define basic objects
    StateCount  = MrkvArray.shape[0]
    aNrmNow     = aXtraGrid
    aNrmCount   = aNrmNow.size
    EndOfPrdvP_cond = np.zeros((StateCount,aNrmCount)) + np.nan
    
    # Loop over *next period* states, calculating conditional EndOfPrdvP
    for j in range(StateCount):
        # Define next-period-state conditional objects
        vPfuncNext  = solution_next.vPfunc[j]
        ShkPrbsNext     = IncomeDstn[j][0]
        PermShkValsNext = IncomeDstn[j][1]
        TranShKValsNext = IncomeDstn[j][2]
        
        # Make tiled versions of end-of-period assets, shocks, and probabilities
        ShkCount    = ShkPrbsNext.size
        aNrm_tiled  = np.tile(np.reshape(aNrmNow,(aNrmCount,1)),(1,ShkCount))
    
        # Tile arrays of the income shocks and put them into useful shapes
        PermShkVals_tiled = np.tile(np.reshape(PermShkValsNext,(1,ShkCount)),(aNrmCount,1))
        TranShkVals_tiled = np.tile(np.reshape(TranShKValsNext,(1,ShkCount)),(aNrmCount,1))
        ShkPrbs_tiled     = np.tile(np.reshape(ShkPrbsNext,(1,ShkCount)),(aNrmCount,1))
        
        # Calculate next period's capital-to-permanent-labor ratio under each combination
        # of end-of-period assets and shock realization
        kNrmNext = aNrm_tiled/(PermGroFac[j]*PermShkVals_tiled)
        
        # Calculate next period's market resources
        KtoLnext  = kNrmNext/TranShkVals_tiled
        RfreeNext = 1. - DeprFac + CapShare*KtoLnext**(CapShare-1.)
        wRteNext  = (1.-CapShare)*KtoLnext**CapShare
        mNrmNext  = RfreeNext*kNrmNext + wRteNext*TranShkVals_tiled
        
        # Calculate end-of-period marginal value of assets for the RA
        vPnext = vPfuncNext(mNrmNext)
        EndOfPrdvP_cond[j,:] = DiscFac*np.sum(RfreeNext*(PermGroFac[j]*PermShkVals_tiled)**(-CRRA)*vPnext*ShkPrbs_tiled,axis=1)

    # Apply the Markov transition matrix to get unconditional end-of-period marginal value
    EndOfPrdvP = np.dot(MrkvArray,EndOfPrdvP_cond)
    
    # Construct the consumption function and marginal value function for each discrete state
    cFuncNow_list = []
    vPfuncNow_list = []
    for i in range(StateCount):
        # Invert the first order condition to get consumption, then find endogenous gridpoints
        cNrmNow = EndOfPrdvP[i,:]**(-1./CRRA)
        mNrmNow = aNrmNow + cNrmNow
    
        # Construct the consumption function and the marginal value function
        cFuncNow_list.append(LinearInterp(np.insert(mNrmNow,0,0.0),np.insert(cNrmNow,0,0.0)))
        vPfuncNow_list.append(MargValueFunc(cFuncNow_list[-1],CRRA))
    
    # Construct and return the solution for this period
    solution_now = ConsumerSolution(cFunc=cFuncNow_list,vPfunc=vPfuncNow_list)
    return solution_now



class RepAgentConsumerType(IndShockConsumerType):
    '''
    A class for representing representative agents with inelastic labor supply.
    '''
    time_inv_ = IndShockConsumerType.time_inv_ + ['CapShare','DeprFac']
    
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
        self.delFromTimeInv('Rfree','BoroCnstArt','vFuncBool','CubicBool')
        
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
        self.yNrmNow = self.kNrmNow**self.CapShare*self.TranShkNow**(1.-self.CapShare)
        self.Rfree = 1. + self.CapShare*self.kNrmNow**(self.CapShare-1.)*self.TranShkNow**(1.-self.CapShare) - self.DeprFac
        self.wRte  = (1.-self.CapShare)*self.kNrmNow**self.CapShare*self.TranShkNow**(-self.CapShare)
        self.mNrmNow = self.Rfree*self.kNrmNow + self.wRte*self.TranShkNow
        

class RepAgentMarkovConsumerType(RepAgentConsumerType):
    '''
    A class for representing representative agents with inelastic labor supply
    and a discrete MarkovState
    '''
    time_inv_ = RepAgentConsumerType.time_inv_ + ['MrkvArray']

    def __init__(self,time_flow=True,**kwds):
        '''
        Make a new instance of a representative agent with Markov state.
        
        Parameters
        ----------
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.
        
        Returns
        -------
        None
        '''       
        RepAgentConsumerType.__init__(self,time_flow=time_flow,**kwds)
        self.solveOnePeriod = solveConsRepAgentMarkov
        
    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        RepAgentConsumerType.updateSolutionTerminal(self)
        
        # Make replicated terminal period solution
        StateCount = self.MrkvArray.shape[0]
        self.solution_terminal.cFunc   = StateCount*[self.cFunc_terminal_]
        self.solution_terminal.vPfunc  = StateCount*[self.solution_terminal.vPfunc]
        self.solution_terminal.mNrmMin = StateCount*[self.solution_terminal.mNrmMin]
    
    
    def getShocks(self):
        '''
        Draws a new Markov state and income shocks for the representative agent.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        cutoffs = np.cumsum(self.MrkvArray[self.MrkvNow,:])
        MrkvDraw = drawUniform(N=1,seed=self.RNG.randint(0,2**31-1))
        self.MrkvNow = np.searchsorted(cutoffs,MrkvDraw)
        
        t = self.t_cycle[0]
        i = self.MrkvNow[0]
        IncomeDstnNow    = self.IncomeDstn[t-1][i] # set current income distribution
        PermGroFacNow    = self.PermGroFac[t-1][i] # and permanent growth factor
        Indices          = np.arange(IncomeDstnNow[0].size) # just a list of integers
        # Get random draws of income shocks from the discrete distribution
        EventDraw        = drawDiscrete(N=1,X=Indices,P=IncomeDstnNow[0],exact_match=False,seed=self.RNG.randint(0,2**31-1))
        PermShkNow = IncomeDstnNow[1][EventDraw]*PermGroFacNow # permanent "shock" includes expected growth
        TranShkNow = IncomeDstnNow[2][EventDraw]
        self.PermShkNow = np.array(PermShkNow)
        self.TranShkNow = np.array(TranShkNow)
        
        
    def getControls(self):
        '''
        Calculates consumption for the representative agent using the consumption functions.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        t = self.t_cycle[0]
        i = self.MrkvNow[0]
        self.cNrmNow = self.solution[t].cFunc[i](self.mNrmNow)

        
###############################################################################        
if __name__ == '__main__':
    from copy import deepcopy
    from time import clock
    from HARKutilities import plotFuncs
    import ConsumerParameters as Params
    import matplotlib.pyplot as plt
    
    # Make a quick example dictionary
    RA_params = deepcopy(Params.init_idiosyncratic_shocks)
    RA_params['DeprFac'] = 0.05
    RA_params['CapShare'] = 0.36
    RA_params['UnempPrb'] = 0.0
    RA_params['LivPrb'] = [1.0]
    
    # Make and solve a rep agent model
    RAexample = RepAgentConsumerType(**RA_params)
    t_start = clock()
    RAexample.solve()
    t_end = clock()
    print('Solving a representative agent problem took ' + str(t_end-t_start) + ' seconds.')
    plotFuncs(RAexample.solution[0].cFunc,0,20)
    
    # Simulate the representative agent model
    RAexample.T_sim = 2000
    RAexample.track_vars = ['cNrmNow','mNrmNow','Rfree','wRte']
    RAexample.initializeSim()
    t_start = clock()
    RAexample.simulate()
    t_end = clock()
    print('Simulating a representative agent for ' + str(RAexample.T_sim) + ' periods took ' + str(t_end-t_start) + ' seconds.')
    
    # Make and solve a Markov representative agent
    RA_markov_params = deepcopy(RA_params)
    RA_markov_params['PermGroFac'] = [[0.97,1.03]]
    RA_markov_params['MrkvArray'] = np.array([[0.99,0.01],[0.01,0.99]])
    RA_markov_params['MrkvNow'] = 0
    RAmarkovExample = RepAgentMarkovConsumerType(**RA_markov_params)
    RAmarkovExample.IncomeDstn[0] = 2*[RAmarkovExample.IncomeDstn[0]]
    t_start = clock()
    RAmarkovExample.solve()
    t_end = clock()
    print('Solving a two state representative agent problem took ' + str(t_end-t_start) + ' seconds.')
    plotFuncs(RAmarkovExample.solution[0].cFunc,0,10)
    
    # Simulate the two state representative agent model
    RAmarkovExample.T_sim = 2000
    RAmarkovExample.track_vars = ['cNrmNow','mNrmNow','Rfree','wRte','MrkvNow']
    RAmarkovExample.initializeSim()
    t_start = clock()
    RAmarkovExample.simulate()
    t_end = clock()
    print('Simulating a two state representative agent for ' + str(RAexample.T_sim) + ' periods took ' + str(t_end-t_start) + ' seconds.')
    