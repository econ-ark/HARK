'''
Implements Sticky Expectations into the ConsAggShockModel module.
Agents update their beliefs about the aggregate economy with a fixed probability
'''
import sys
import os 
sys.path.insert(0,'../')

import numpy as np
from ConsAggShockModel import AggShockConsumerType, CobbDouglasEconomy, solveConsAggShock
from HARKsimulation import drawBernoulli
from HARKcore import AgentType
from ImpulseResponse import AggCons_impulseResponseAgg, impulseResponseAgg
from copy import copy, deepcopy
import matplotlib.pyplot as plt
     ###############################################################################
        
class AggShockStickyExpectationsConsumerType(AggShockConsumerType):
    '''
    A class to represent consumers who face idiosyncratic (transitory and per-
    manent) shocks to their income and live in an economy that has aggregate
    (transitory and permanent) shocks to labor productivity.  As the capital-
    to-labor ratio varies in the economy, so does the wage rate and interest
    rate.  "Aggregate shock consumers" have beliefs about how the capital ratio
    evolves over time and take aggregate shocks into account when making their
    decision about how much to consume.   
    Difference between this and AggShockConsumerType is that agents don't 
    necessarily update their belief about the aggregate economy
    '''
    def __init__(self,time_flow=True,**kwds):
        '''
        Make a new instance of AggShockConsumerType, an extension of
        IndShockConsumerType.  Sets appropriate solver and input lists.
        '''
        AgentType.__init__(self,solution_terminal=deepcopy(AggShockConsumerType.solution_terminal_),
                           time_flow=time_flow,pseudo_terminal=False,**kwds)
        
        self.time_vary = deepcopy(AggShockConsumerType.time_vary_)
        self.time_inv = deepcopy(AggShockConsumerType.time_inv_)
        self.delFromTimeInv('Rfree','BoroCnstArt','vFuncBool','CubicBool')
        self.solveOnePeriod = solveConsAggShock
        self.p_init = np.ones(self.Nagents)
        self.update()
        
    def getEconomyData(self,Economy):
        '''
        Imports economy-determined objects into self from a Market.
        Instances of AggShockConsumerType "live" in some macroeconomy that has
        attributes relevant to their microeconomic model, like the relationship
        between the capital-to-labor ratio and the interest and wage rates; this
        method imports those attributes from an "economy" object and makes them
        attributes of the ConsumerType.
        
        Parameters
        ----------
        Economy : Market
            The "macroeconomy" in which this instance "lives".  Might be of the
            subclass CobbDouglasEconomy, which has methods to generate the
            relevant attributes.
            
        Returns
        -------
        None
        '''
        AggShockConsumerType.getEconomyData(self, Economy)
        self.KtoLBeliefNow_init = Economy.KtoLnow_init*np.ones(self.Nagents)
        
    def initializeSim(self,a_init=None,p_init=None,t_init=0,sim_prds=None):
        '''
        Readies this type for simulation by clearing its history, initializing
        state variables, and setting time indices to their correct position.
        
        Parameters
        ----------
        a_init : np.array
            Array of initial end-of-period assets at the beginning of the sim-
            ulation.  Should be of size self.Nagents.  If omitted, will default
            to values in self.a_init (which are all 0 by default).
        p_init : np.array
            Array of initial permanent income levels at the beginning of the sim-
            ulation.  Should be of size self.Nagents.  If omitted, will default
            to values in self.p_init (which are all 1 by default).
        t_init : int
            Period of life in which to begin the simulation.  Defaults to 0.
        sim_prds : int
            Number of periods to simulate.  Defaults to the length of the trans-
            itory income shock history.
        
        Returns
        -------
        none
        '''
        AggShockConsumerType.initializeSim(self,a_init,p_init,t_init,sim_prds)
        blank_history = np.zeros_like(self.pHist) + np.nan
        # After doing the same initialization as AggShockConsumerType, add in beliefs
        self.pBeliefHist = copy(blank_history)
        self.bBeliefHist = copy(blank_history)
        self.mBeliefHist = copy(blank_history)
        self.aBeliefHist = copy(blank_history)
        
        self.aBeliefNow = self.a_init
        self.pBeliefNow = self.p_init
        self.KtoLBeliefNow = self.KtoLBeliefNow_init
        self.updateBeliefNow = np.ones_like(self.p_init) #beliefs must be up to date at the start of simulation
 
       
    def simOnePrd(self):
        '''
        Simulate a single period of a consumption-saving model with permanent
        and transitory income shocks at both the idiosyncratic and aggregate level.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''        
        # Unpack objects from self for convenience
        aPrev          = self.aNow
        pPrev          = self.pNow
        TranShkNow     = self.TranShkNow
        PermShkNow     = self.PermShkNow
        RfreeNow       = self.RfreeNow
        cFuncNow       = self.cFuncNow
        KtoLnow        = self.KtoLnow*np.ones_like(aPrev)
        kNextFunc = self.kNextFunc
        
        update_belief = self.updateBeliefNow
        aBeliefPrev = self.aBeliefNow
        pBeliefPrev = self.pBeliefNow
        TranShkBeliefNow = self.TranShkBeliefNow
        PermShkBeliefNow = self.PermShkBeliefNow
        KtoLBeliefPrev = self.KtoLBeliefNow
        
        aBeliefNow = np.zeros_like(aBeliefPrev) + np.nan
        pBeliefNow = np.zeros_like(pBeliefPrev) + np.nan
        KtoLBeliefNow = np.zeros_like(KtoLBeliefPrev) + np.nan
        ReffBeliefNow = np.zeros_like(KtoLBeliefPrev) + np.nan
        mBeliefNow = np.zeros_like(pBeliefPrev) + np.nan
        bBeliefNow = np.zeros_like(pBeliefPrev) + np.nan

        # Simulate the period
        pNow    = pPrev*PermShkNow      # Updated permanent income level
        ReffNow = RfreeNow/PermShkNow   # "effective" interest factor on normalized assets
        bNow    = ReffNow*aPrev         # Bank balances before labor income
        mNow    = bNow + TranShkNow     # Market resources after income
        
        for i in range(self.Nagents):
            if update_belief[i]:
                KtoLBeliefNow[i] = KtoLnow[i]
                pBeliefNow[i] = pNow[i]
                ReffBeliefNow[i] = ReffNow[i]
                bBeliefNow[i] = bNow[i]
                mBeliefNow[i] = mNow[i]
            else:
                KtoLBeliefNow[i] = kNextFunc(KtoLBeliefPrev[i])
                pBeliefNow[i] = pBeliefPrev[i]*PermShkBeliefNow[i]
                ReffBeliefNow[i] = RfreeNow/PermShkBeliefNow[i] 
                bBeliefNow[i]    = ReffBeliefNow[i]*aBeliefPrev[i]         
                mBeliefNow[i]    = bBeliefNow[i] + TranShkBeliefNow[i]    
                
        cNow    = cFuncNow(mBeliefNow,KtoLBeliefNow) # Consumption (normalized by permanent income)
        MPCnow  = cFuncNow.derivativeX(mBeliefNow,KtoLBeliefNow) # Marginal propensity to consume

        aNow    = mNow - cNow*pBeliefNow/pNow          # Assets after all actions are accomplished
        # Hack so that assets don't go below their minimum value
        aNow = np.max([aNow, self.aXtraMin*np.ones_like(aNow)],0)
        aBeliefNow = mBeliefNow - cNow
        
        # Store the new state and control variables
        self.pNow   = pNow
        self.bNow   = bNow
        self.mNow   = mNow
        self.cNow   = cNow
        self.MPCnow = MPCnow
        self.aNow   = aNow
        
        self.pBeliefNow   = pBeliefNow
        self.bBeliefNow   = bBeliefNow
        self.mBeliefNow   = mBeliefNow
        self.aBeliefNow   = aBeliefNow
        
    def advanceIncShks(self):
        '''
        Advance the permanent and transitory income shocks to the next period of
        the shock history objects.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        self.PermShkNow = self.PermShkHist[self.Shk_idx]
        self.TranShkNow = self.TranShkHist[self.Shk_idx]
        self.updateBeliefNow = self.updateBeliefHist[self.Shk_idx]
        self.Shk_idx += 1
        if self.Shk_idx >= self.PermShkHist.shape[0]:
            self.Shk_idx = 0 # Reset to zero if we've run out of shocks
        
            
    def marketAction(self):
        '''
        In the aggregate shocks model, the "market action" is to simulate one
        period of receiving income and choosing how much to consume.
        
        Parameters
        ----------
        none
            
        Returns
        -------
        none
        '''
        # Simulate the period
        self.advanceIncShks()
        self.advancecFunc()
        self.simMortality()
        
        # update beliefs with some probability
        # NOTE - I still need to change wRteNow to do this properly...
        self.TranShkBeliefNow = self.TranShkNow*(1-self.updateBeliefNow)  + self.TranShkNow*self.wRteNow*self.updateBeliefNow
        self.PermShkBeliefNow = self.PermShkNow*(1-self.updateBeliefNow) + self.PermShkNow*self.PermShkAggNow*self.updateBeliefNow
        
        self.TranShkNow = self.TranShkNow*self.wRteNow
        self.PermShkNow = self.PermShkNow*self.PermShkAggNow

        self.simOnePrd()
        
        # Record the results of the period
        self.pHist[self.t_agg_sim,:] = self.pNow
        self.bHist[self.t_agg_sim,:] = self.bNow
        self.mHist[self.t_agg_sim,:] = self.mNow
        self.cHist[self.t_agg_sim,:] = self.cNow
        self.MPChist[self.t_agg_sim,:] = self.MPCnow
        self.aHist[self.t_agg_sim,:] = self.aNow

        self.pBeliefHist[self.t_agg_sim,:] = self.pBeliefNow
        self.bBeliefHist[self.t_agg_sim,:] = self.bBeliefNow
        self.mBeliefHist[self.t_agg_sim,:] = self.mBeliefNow
        self.aBeliefHist[self.t_agg_sim,:] = self.aBeliefNow

        self.t_agg_sim += 1
        
    def makeIncShkHist(self):
        '''
        Makes histories of simulated income shocks for this consumer type by
        drawing from the discrete income distributions, storing them as attributes
        of self for use by simulation methods.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        # After running the routine for AggShockConsumerType, just add in updateBeliefHist
        AggShockConsumerType.makeIncShkHist(self)
        orig_time = self.time_flow
        self.timeFwd()
        self.resetRNG()
        
        # Initialize the shock histories
        updateBeliefHist = np.zeros((self.sim_periods,self.Nagents)) + np.nan
        updateBeliefHist[0,:] = 1.0
        t_idx = 0
        
        # Loop through each simulated period
        for t in range(1,self.sim_periods):
            updateBeliefHist[t,:] = drawBernoulli(self.Nagents,self.updateBeliefProb,seed=self.RNG.randint(0,2**31-1))
            # Advance the time index, looping if we've run out of income distributions
            t_idx += 1
            if t_idx >= len(self.IncomeDstn):
                t_idx = 0
        
        # Store the results as attributes of self and restore time to its original flow
        self.updateBeliefHist = updateBeliefHist
        if not orig_time:
            self.timeRev()
            
def AggConsSticky_impulseResponseAgg(Market, PermShk, TranShk, TimeBefore = 100, TimeAfter = 20):
    '''
    Routine to calculate the aggregate consumption impulse response for a Market
    
    Parameters
    ----------
    Market : Market
        An instance of the Market to be shocked
    PermShk : [np.array]
        An array containing the permanent aggregate shocks
    TranShk : [np.array]
        An array containing the transitory aggregate shocks 
    TimeBefore : float
        Number of time periods to run the simulation before applying the shocks
            TimeBefore : float
        Number of time periods to run the simulation after applying the shocks
    Returns
    -------
    impulse_response : [np.array]
        An array containing the aggregate consumption response
    '''
    (ShockedMarket, NoShockMarket) = impulseResponseAgg(Market, PermShk, TranShk, TimeBefore, TimeAfter)
    
    cAggNoShkHist = np.zeros(NoShockMarket.act_T)
    for this_type in NoShockMarket.agents:
        pBeliefNoShkHist = this_type.pBeliefHist
        cNoShkHist = this_type.cHist
        cAggNoShkHist += np.sum(pBeliefNoShkHist*cNoShkHist,1)
    
    cAggShkHist = np.zeros(ShockedMarket.act_T)
    for this_type in ShockedMarket.agents:
        pBeliefShkHist = this_type.pBeliefHist
        cShkHist = this_type.cHist
        cAggShkHist += np.sum(pBeliefShkHist*cShkHist,1)
    
    cAggImpulseResponse = (cAggShkHist - cAggNoShkHist)/cAggNoShkHist
    
    # Remove TimeBefore periods
    cAggImpulseResponse = cAggImpulseResponse[TimeBefore:]

    return cAggImpulseResponse
    
###############################################################################

if __name__ == '__main__':
    import ConsumerParameters as Params
    from time import clock
    from HARKutilities import plotFuncs
    mystr = lambda number : "{:.4f}".format(number)

    
    # Make an aggregate shocks sticky expectations consumer
    StickyExample = AggShockStickyExpectationsConsumerType(**Params.init_sticky_shocks)
    NotStickyExample = AggShockConsumerType(**Params.init_sticky_shocks)
    StickyExample.cycles = 0
    NotStickyExample.cycles = 0
    StickyExample.sim_periods = 3000
    NotStickyExample.sim_periods = 3000
    StickyExample.makeIncShkHist()  # Simulate a history of idiosyncratic shocks
    NotStickyExample.makeIncShkHist()  # Simulate a history of idiosyncratic shocks
    
    # Make a Cobb-Douglas economy for the agents
    StickyEconomyExample = CobbDouglasEconomy(agents = [StickyExample],act_T=StickyExample.sim_periods,**Params.init_cobb_douglas)
    NotStickyEconomyExample = CobbDouglasEconomy(agents = [NotStickyExample],act_T=NotStickyExample.sim_periods,**Params.init_cobb_douglas)
    StickyEconomyExample.makeAggShkHist() # Simulate a history of aggregate shocks
    NotStickyEconomyExample.makeAggShkHist() # Simulate a history of aggregate shocks
    
    # Have the consumers inherit relevant objects from the economy
    StickyExample.getEconomyData(StickyEconomyExample)
    NotStickyExample.getEconomyData(NotStickyEconomyExample)
    
    # Solve the microeconomic model for the aggregate shocks example type (and display results)
    t_start = clock()
    StickyExample.solve()
    NotStickyExample.solve()
    t_end = clock()
    print('Solving an aggregate shocks consumer took ' + mystr(t_end-t_start) + ' seconds.')
    
#    # Solve the "macroeconomic" model by searching for a "fixed point dynamic rule"
#    t_start = clock()
#    StickyEconomyExample.solve()
#    NotStickyEconomyExample.solve()
#    t_end = clock()
#    print('Solving the "macroeconomic" aggregate shocks model took ' + str(t_end - t_start) + ' seconds.')
  
    PermShk = 1.1
    TranShk = 1
    StickyImpulseResponse = AggConsSticky_impulseResponseAgg(StickyEconomyExample,PermShk,TranShk,100,25)
    NotStickyImpulseResponse = AggCons_impulseResponseAgg(NotStickyEconomyExample,PermShk,TranShk,100,25)
    plt.plot(StickyImpulseResponse)
    plt.plot(NotStickyImpulseResponse)
    
