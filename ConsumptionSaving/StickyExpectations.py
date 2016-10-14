'''

Implements Sticky Expectations into the ConsAggShockModel module.

Agents update their beliefs about the aggregate economy with a fixed probability

'''

import sys 

sys.path.insert(0,'../')













import numpy as np

from ConsAggShockModel import AggShockConsumerType, CobbDouglasEconomy, solveConsAggShock

from HARKsimulation import drawBernoulli, drawDiscrete

from HARKcore import AgentType

from HARKutilities import combineIndepDstns

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

        self.a_init = Economy.KtoYSS*np.ones(self.Nagents)  # Initialize assets to steady state

        self.kGrid  = Economy.kSS*self.kGridBase            # Capital ratio grid adjusted around SS ratio

        self.kNextFunc = Economy.kNextFunc                  # Next period's capital ratio as function of current ratio

        self.Rfunc = Economy.Rfunc                          # Interest factor as function of capital ratio

        self.wFunc = Economy.wFunc                          # (Normalized) wage rate as function of capital ratio

        IncomeDstnWithAggShks = combineIndepDstns(self.PermShkDstn,self.TranShkDstn,Economy.PermShkAggDstn,Economy.TranShkAggDstn)

        self.IncomeDstn = [IncomeDstnWithAggShks]           # Discrete income distribution with aggregate and idiosyncratic shocks

        self.DiePrb = 1.0 - self.LivPrb[0]                  # Only relevant for simulating with mortality

        self.addToTimeInv('kGrid','kNextFunc','Rfunc', 'wFunc')

        self.KtoLBeliefNow_init = Economy.KtoLnow_init*np.ones(self.Nagents)

        

    def initializeSim(self,a_init=None,p_init=None,t_init=0,sim_prds=None, update_belief_prob=None):

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

        # Fill in default values

        if a_init is None:

            a_init = self.a_init

        if p_init is None:

            p_init = self.p_init

        if sim_prds is None:

            sim_prds = len(self.TranShkHist)

            

        # Initialize indices

        self.resetRNG()

        self.Shk_idx   = t_init

        self.cFunc_idx = t_init

        

        # Initialize the history arrays

        self.aNow     = a_init

        self.pNow     = p_init

                

        if hasattr(self,'Rboro'):

            self.RboroNow = self.Rboro

            self.RsaveNow = self.Rsave

        elif hasattr(self,'Rfree'):

            self.RfreeNow = self.Rfree

        blank_history = np.zeros((sim_prds,self.Nagents)) + np.nan

        self.pHist    = copy(blank_history)

        self.bHist    = copy(blank_history)

        self.mHist    = copy(blank_history)

        self.cHist    = copy(blank_history)

        self.MPChist  = copy(blank_history)

        self.aHist    = copy(blank_history)

        

        self.pBeliefHist = copy(blank_history)

        self.bBeliefHist = copy(blank_history)

        self.mBeliefHist = copy(blank_history)

        self.aBeliefHist = copy(blank_history)

        

        self.aBeliefNow = a_init

        self.pBeliefNow = p_init

        self.KtoLBeliefNow = self.KtoLBeliefNow_init

        self.updateBeliefNow = np.ones_like(p_init) #beliefs must be up to date at the start of simulation

 

       

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

        orig_time = self.time_flow

        self.timeFwd()

        self.resetRNG()

        

        # Initialize the shock histories

        PermShkHist = np.zeros((self.sim_periods,self.Nagents)) + np.nan

        TranShkHist = np.zeros((self.sim_periods,self.Nagents)) + np.nan

        updateBeliefHist = np.zeros((self.sim_periods,self.Nagents)) + np.nan

        PermShkHist[0,:] = 1.0

        TranShkHist[0,:] = 1.0

        updateBeliefHist[0,:] = 1.0

        t_idx = 0

        

        # Loop through each simulated period

        for t in range(1,self.sim_periods):

            IncomeDstnNow    = self.IncomeDstn[t_idx] # set current income distribution

            PermGroFacNow    = self.PermGroFac[t_idx] # and permanent growth factor

            Indices          = np.arange(IncomeDstnNow[0].size) # just a list of integers

            # Get random draws of income shocks from the discrete distribution

            EventDraws       = drawDiscrete(N=self.Nagents,X=Indices,P=IncomeDstnNow[0],exact_match=True,seed=self.RNG.randint(0,2**31-1))

            PermShkHist[t,:] = IncomeDstnNow[1][EventDraws]*PermGroFacNow # permanent "shock" includes expected growth

            TranShkHist[t,:] = IncomeDstnNow[2][EventDraws]

            updateBeliefHist[t,:] = drawBernoulli(self.Nagents,self.updateBeliefProb,seed=self.RNG.randint(0,2**31-1))

            # Advance the time index, looping if we've run out of income distributions

            t_idx += 1

            if t_idx >= len(self.IncomeDstn):

                t_idx = 0

        

        # Store the results as attributes of self and restore time to its original flow

        self.PermShkHist = PermShkHist

        self.TranShkHist = TranShkHist

        self.updateBeliefHist = updateBeliefHist

        if not orig_time:

            self.timeRev()

        

    def calcAggConsumptionHist(self):

        '''

        Calculates aggregate consumption based on the simulated history of individual's

        consumption

        

        Parameters

        ----------

        None

            

        Returns

        -------

        None

        '''

        

        pBeliefHist = self.pBeliefHist

        cHist = self.cHist

        

        AggConsumption = np.sum(pBeliefHist*cHist,1)

        self.AggConsumption = AggConsumption

        

###############################################################################

     

if __name__ == '__main__':

    import ConsumerParameters as Params

    from time import clock

    from HARKutilities import plotFuncs

    mystr = lambda number : "{:.4f}".format(number)

    

    do_simulation           = True



    # Make an aggregate shocks consumer

    AggShockExample = AggShockStickyExpectationsConsumerType(**Params.init_sticky_shocks)

    AggShockExample.cycles = 0

    AggShockExample.sim_periods = 500

    AggShockExample.makeIncShkHist()  # Simulate a history of idiosyncratic shocks

    

    # Make a Cobb-Douglas economy for the agents

    EconomyExample = CobbDouglasEconomy(agents = [AggShockExample],act_T=AggShockExample.sim_periods,**Params.init_cobb_douglas)

    EconomyExample.makeAggShkHist() # Simulate a history of aggregate shocks

    

    # Have the consumers inherit relevant objects from the economy

    AggShockExample.getEconomyData(EconomyExample)

    

    # Solve the microeconomic model for the aggregate shocks example type (and display results)

    t_start = clock()

    AggShockExample.solve()

    t_end = clock()

    print('Solving an aggregate shocks consumer took ' + mystr(t_end-t_start) + ' seconds.')

    print('Consumption function at each capital-to-labor ratio gridpoint:')

    m_grid = np.linspace(0,10,200)

    AggShockExample.unpackcFunc()

    for k in AggShockExample.kGrid.tolist():

        c_at_this_k = AggShockExample.cFunc[0](m_grid,k*np.ones_like(m_grid))

        plt.plot(m_grid,c_at_this_k)

    plt.show()

    

    # Solve the "macroeconomic" model by searching for a "fixed point dynamic rule"

    t_start = clock()

    EconomyExample.solve()

    t_end = clock()

    print('Solving the "macroeconomic" aggregate shocks model took ' + str(t_end - t_start) + ' seconds.')

    print('Next capital-to-labor ratio as function of current ratio:')

    plotFuncs(EconomyExample.kNextFunc,0,2*EconomyExample.kSS)

    print('Consumption function at each capital-to-labor ratio gridpoint (in general equilibrium):')

    AggShockExample.unpackcFunc()

    m_grid = np.linspace(0,10,200)

    for k in AggShockExample.kGrid.tolist():

        c_at_this_k = AggShockExample.cFunc[0](m_grid,k*np.ones_like(m_grid))

        plt.plot(m_grid,c_at_this_k)

    plt.show()

    

    

    # Simulate model and plot aggregate consumption

    if do_simulation:

        AggShockExample.sim_periods = 500

        AggShockExample.makeIncShkHist()

        EconomyExample.makeZeroAggShkHist() 

        AggShockExample.initializeSim()

        EconomyExample.makeHistory()

        AggShockExample.calcAggConsumptionHist()

        AggConsZeroHist = AggShockExample.AggConsumption

        

        EconomyExample.PermShkAggHist[200]=1.1

        AggShockExample.initializeSim()

        EconomyExample.makeHistory()

        AggShockExample.calcAggConsumptionHist()

        AggConsShockHist = AggShockExample.AggConsumption

        

        ImpulseResponse_Sticky = AggConsShockHist-AggConsZeroHist



        

        #Set update probability to 1

        AggShockExample.updateBeliefHist = np.ones_like(AggShockExample.updateBeliefHist)

        EconomyExample.makeZeroAggShkHist()         

        AggShockExample.initializeSim()

        EconomyExample.makeHistory()

        AggShockExample.calcAggConsumptionHist()

        AggConsZeroHist_NotSticky = AggShockExample.AggConsumption

        

        EconomyExample.PermShkAggHist[200]=1.1

        AggShockExample.initializeSim()

        EconomyExample.makeHistory()

        AggShockExample.calcAggConsumptionHist()

        AggConsShockHist_NotSticky = AggShockExample.AggConsumption

        

        ImpulseResponse_NotSticky = AggConsShockHist_NotSticky-AggConsZeroHist_NotSticky

        

        plt.plot(ImpulseResponse_Sticky[190:240])

        plt.plot(ImpulseResponse_NotSticky[190:240])

        plt.show()

        

        

        

        