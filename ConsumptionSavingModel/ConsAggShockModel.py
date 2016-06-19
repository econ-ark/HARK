'''
Consumption-saving models with aggregate productivity shocks as well as idiosyn-
cratic income shocks.  Currently only contains one microeconomic model with a
basic solver.  Also includes a subclass of Market called CobbDouglas economy,
used for solving "macroeconomic" models with aggregate shocks.
'''
import sys 
sys.path.insert(0,'../')

import numpy as np
import scipy.stats as stats
from HARKinterpolation import LinearInterp, LinearInterpOnInterp1D
from HARKutilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv,\
                          CRRAutility_invP, CRRAutility_inv, combineIndepDstns,\
                          approxMeanOneLognormal
from HARKsimulation import drawDiscrete, drawBernoulli
from ConsIndShockModel import ConsumerSolution, IndShockConsumerType
from HARKcore import HARKobject, Market, AgentType
from copy import deepcopy

utility      = CRRAutility
utilityP     = CRRAutilityP
utilityPP    = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv  = CRRAutility_inv

class MargValueFunc2D():
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of v'(m,k) = u'(c(m,k)) holds (with CRRA utility).
    '''
    def __init__(self,cFunc,CRRA):
        '''
        Constructor for a new marginal value function object.
        
        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources: uP_inv(vPfunc(m,k)).  Called cFunc because when standard
            envelope condition applies, uP_inv(vPfunc(m,k)) = cFunc(m,k).
        CRRA : float
            Coefficient of relative risk aversion.
            
        Returns
        -------
        new instance of MargValueFunc
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        
    def __call__(self,m,k):
        return utilityP(self.cFunc(m,k),gam=self.CRRA)
        
###############################################################################
        
class AggShockConsumerType(IndShockConsumerType):
    '''
    A class to represent consumers who face idiosyncratic (transitory and per-
    manent) shocks to their income and live in an economy that has aggregate
    (transitory and permanent) shocks to labor productivity.  As the capital-
    to-labor ratio varies in the economy, so does the wage rate and interest
    rate.  "Aggregate shock consumers" have beliefs about how the capital ratio
    evolves over time and take aggregate shocks into account when making their
    decision about how much to consume.    
    '''
    def __init__(self,time_flow=True,**kwds):
        '''
        Make a new instance of AggShockConsumerType, an extension of the basic
        ConsumerType.  Sets appropriate solver and input lists.
        '''
        AgentType.__init__(self,solution_terminal=deepcopy(IndShockConsumerType.solution_terminal_),
                           time_flow=time_flow,pseudo_terminal=False,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(IndShockConsumerType.time_vary_)
        self.time_inv = deepcopy(IndShockConsumerType.time_inv_)
        self.time_inv.remove('Rfree')
        self.time_inv.remove('BoroCnstArt')
        self.time_inv.remove('vFuncBool')
        self.time_inv.remove('CubicBool')
        self.solveOnePeriod = solveConsAggShock
        self.p_init = np.ones(self.Nagents)
        self.update()
        
    def reset(self):
        '''
        Initialize this type for a new simulated history of K/L ratio.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        self.initializeSim()
        self.t_agg_sim = 0
        
    def updateSolutionTerminal(self):
        '''
        Updates the terminal period solution for an aggregate shock consumer.
        Only fills in the consumption function and marginal value function.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        vPfunc_terminal = lambda m,k : m**(-self.CRRA)
        cFunc_terminal  = lambda m,k : m
        self.solution_terminal = ConsumerSolution(cFunc=cFunc_terminal,vPfunc=vPfunc_terminal)
        
    def getEconomyData(self,Economy):
        '''
        Imports economy-determined objects into the ConsumerType from a Market.
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
        if not ('kGrid' in self.time_inv):
            self.time_inv.append('kGrid')
        if not ('kNextFunc' in self.time_inv):
            self.time_inv.append('kNextFunc')
        if not ('Rfunc' in self.time_inv):
            self.time_inv.append('Rfunc')
        if not ('wFunc' in self.time_inv):
            self.time_inv.append('wFunc')
        
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
        
        # Simulate the period
        pNow    = pPrev*PermShkNow      # Updated permanent income level
        ReffNow = RfreeNow/PermShkNow   # "effective" interest factor on normalized assets
        bNow    = ReffNow*aPrev         # Bank balances before labor income
        mNow    = bNow + TranShkNow     # Market resources after income
        cNow    = cFuncNow(mNow,KtoLnow) # Consumption (normalized by permanent income)
        MPCnow  = cFuncNow.derivativeX(mNow,KtoLnow) # Marginal propensity to consume
        aNow    = mNow - cNow           # Assets after all actions are accomplished
        
        # Store the new state and control variables
        self.pNow   = pNow
        self.bNow   = bNow
        self.mNow   = mNow
        self.cNow   = cNow
        self.MPCnow = MPCnow
        self.aNow   = aNow
        
    def simMortality(self):
        '''
        Simulates the mortality process, killing off some percentage of agents
        and replacing them with newborn agents.
        
        Parameters
        ----------
        none
            
        Returns
        -------
        none
        '''
        if hasattr(self,'DiePrb'):
            if self.DiePrb > 0:
                who_dies = drawBernoulli(self.DiePrb,self.Nagents,self.RNG.randint(low=1, high=2**31-1))
                wealth_all = self.aNow*self.pNow
                who_lives = np.logical_not(who_dies)
                wealth_of_dead = np.sum(wealth_all[who_dies])
                wealth_of_live = np.sum(wealth_all[who_lives])
                R_actuarial = 1.0 + wealth_of_dead/wealth_of_live
                self.aNow[who_dies] = 0.0
                self.pNow[who_dies] = 1.0
                self.aNow = self.aNow*R_actuarial
            
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
        self.t_agg_sim += 1
        
    def calcBoundingValues(self):
        '''
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.
        
        NOT YET IMPLEMENTED FOR THIS CLASS
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        raise NotImplementedError()
        
    def makeEulerErrorFunc(self,mMax=100,approx_inc_dstn=True):
        '''
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncomeDstn
        or to use a (temporary) very dense approximation.
        
        NOT YET IMPLEMENTED FOR THIS CLASS
        
        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncomeDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncomeDstn; when False, makes and uses a very dense approximation.
        
        Returns
        -------
        None
        '''
        raise NotImplementedError()
        
###############################################################################


def solveConsAggShock(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,PermGroFac,aXtraGrid,kGrid,kNextFunc,Rfunc,wFunc):
    '''
    Solve one period of a consumption-saving problem with idiosyncratic and 
    aggregate shocks (transitory and permanent).  This is a basic solver that
    can't handle borrowing (assumes liquidity constraint) or cubic splines, nor
    can it calculate a value function.
    
    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to the succeeding one period problem.
    IncomeDstn : [np.array]
        A list containing five arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, idisyncratic permanent shocks, idiosyncratic transitory
        shocks, aggregate permanent shocks, aggregate transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.    
    DiscFac : float
        Intertemporal discount factor for future utility.        
    CRRA : float
        Coefficient of relative risk aversion.
    PermGroGac : float
        Expected permanent income growth factor at the end of this period.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    kGrid : np.array
        A grid of capital-to-labor ratios in the economy.
    kNextFunc : function
        Next period's capital-to-labor ratio as a function of this period's ratio.
    Rfunc : function
        The net interest factor on assets as a function of capital ratio k.
    wFunc : function
        The wage rate for labor as a function of capital-to-labor ratio k.
                    
    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (linear interpolation over linear interpola-
        tions) and marginal value function vPfunc.
    '''
    # Unpack next period's solution
    vPfuncNext = solution_next.vPfunc
    
    # Unpack the income shocks
    ShkPrbsNext  = IncomeDstn[0]
    PermShkValsNext = IncomeDstn[1]
    TranShkValsNext = IncomeDstn[2]
    PermShkAggValsNext = IncomeDstn[3]
    TranShkAggValsNext = IncomeDstn[4]
    ShkCount = ShkPrbsNext.size
        
    # Make the grid of end-of-period asset values, and a tiled version
    aNrmNow = np.insert(aXtraGrid,0,0.0)
    aNrmNow_tiled   = np.tile(aNrmNow,(ShkCount,1))
    aCount = aNrmNow.size
    
    # Make tiled versions of the income shocks
    ShkPrbsNext_tiled = (np.tile(ShkPrbsNext,(aCount,1))).transpose()
    PermShkValsNext_tiled = (np.tile(PermShkValsNext,(aCount,1))).transpose()
    TranShkValsNext_tiled = (np.tile(TranShkValsNext,(aCount,1))).transpose()
    PermShkAggValsNext_tiled = (np.tile(PermShkAggValsNext,(aCount,1))).transpose()
    TranShkAggValsNext_tiled = (np.tile(TranShkAggValsNext,(aCount,1))).transpose()
    
    # Loop through the values in kGrid and calculate a linear consumption function for each
    cFuncByK_list = []
    for j in range(kGrid.size):
        kNow = kGrid[j]
        kNext = kNextFunc(kNow)
        
        # Calculate returns to capital and labor in the next period        
        kNextEff_array = kNext/TranShkAggValsNext_tiled
        Reff_array = Rfunc(kNextEff_array)/LivPrb # Effective interest rate
        wEff_array = wFunc(kNextEff_array)*TranShkAggValsNext_tiled # Effective wage rate (accounts for labor supply)
        
        # Calculate market resources next period (and a constant array of capital-to-labor ratio)
        PermShkTotal_array = PermGroFac*PermShkValsNext_tiled*PermShkAggValsNext_tiled # total / combined permanent shock
        mNrmNext_array = Reff_array*aNrmNow_tiled/PermShkTotal_array + TranShkValsNext_tiled*wEff_array
        kNext_array = kNext*np.ones_like(mNrmNext_array)
        
        # Find marginal value next period at every income shock realization and every asset gridpoint
        vPnext_array = Reff_array*PermShkTotal_array**(-CRRA)*vPfuncNext(mNrmNext_array,kNext_array)
        
        # Calculate expectated marginal value at the end of the period at every asset gridpoint
        EndOfPrdvP = DiscFac*LivPrb*PermGroFac**(-CRRA)*np.sum(vPnext_array*ShkPrbsNext_tiled,axis=0)
        
        # Calculate optimal consumption from each asset gridpoint, and construct a linear interpolation
        cNrmNow = EndOfPrdvP**(-1.0/CRRA)
        mNrmNow = aNrmNow + cNrmNow
        c_for_interpolation = np.insert(cNrmNow,0,0.0) # Add liquidity constrained portion
        m_for_interpolation = np.insert(mNrmNow,0,0.0)
        cFuncNow_j = LinearInterp(m_for_interpolation,c_for_interpolation)
        
        # Add the k-specific consumption function to the list
        cFuncByK_list.append(cFuncNow_j)
    
    # Construct the overall consumption function by combining the k-specific functions
    cFuncNow = LinearInterpOnInterp1D(cFuncByK_list,kGrid)
    
    # Construct the marginal value function using the envelope condition
    vPfuncNow = MargValueFunc2D(cFuncNow,CRRA)
    
    # Pack up and return the solution
    solution_now = ConsumerSolution(cFunc=cFuncNow,vPfunc=vPfuncNow)
    return solution_now
    
###############################################################################
    
class CobbDouglasEconomy(Market):            
    '''
    A class to represent an economy with a Cobb-Douglas aggregate production
    function over labor and capital, extending HARKcore.Market.  The "aggregate
    market process" for this market combines all individuals' asset holdings
    into aggregate capital, yielding the interest factor on assets and the wage
    rate for the upcoming period.
    
    Note: In the current implementation assumes a constant labor supply, but
    this will be generalized in the future.
    '''
    def __init__(self,agents=[],tolerance=0.0001,act_T=1000,**kwds):
        '''
        Make a new instance of CobbDouglasEconomy by filling in attributes
        specific to this kind of market.
        
        Parameters
        ----------
        agents : [ConsumerType]
            List of types of consumers that live in this economy.
        tolerance: float
            Minimum acceptable distance between "dynamic rules" to consider the
            solution process converged.  Distance depends on intercept and slope
            of the log-linear "next capital ratio" function.
        act_T : int
            Number of periods to simulate when making a history of of the market.
            
        Returns
        -------
        None
        '''
        Market.__init__(self,agents=agents,
                            sow_vars=['KtoLnow','RfreeNow','wRteNow','PermShkAggNow','TranShkAggNow'],
                            reap_vars=['pNow','aNow'],
                            track_vars=['KtoLnow'],
                            dyn_vars=['kNextFunc'],
                            tolerance=tolerance,
                            act_T=act_T)
        self.assignParameters(**kwds)
        self.update()
    
    
    def millRule(self,pNow,aNow):
        '''
        Function to calculate the capital to labor ratio, interest factor, and
        wage rate based on each agent's current state.  Just calls calcRandW().
        
        See documentation for calcRandW for more information.
        '''
        return self.calcRandW(pNow,aNow)
        
    def calcDynamics(self,KtoLnow):
        '''
        Calculates a new dynamic rule for the economy: next period's capital
        ratio as a function of this period's.  Just calls calcCapitalEvoRule().
        
        See documentation for calcCapitalEvoRule for more information.
        '''
        return self.calcCapitalEvoRule(KtoLnow)
        
    def update(self):
        '''
        Use primitive parameters (and perfect foresight calibrations) to make
        interest factor and wage rate functions (of capital to labor ratio),
        as well as discrete approximations to the aggregate shock distributions.
        
        Parameters
        ----------
        none
            
        Returns
        -------
        none
        '''
        self.kSS   = ((self.CRRA/self.DiscFac - (1.0-self.DeprFac))/self.CapShare)**(1.0/(self.CapShare-1.0))
        self.KtoYSS = self.kSS**(1.0-self.CapShare)
        self.wRteSS = (1.0-self.CapShare)*self.kSS**(self.CapShare)
        self.convertKtoY = lambda KtoY : KtoY**(1.0/(1.0 - self.CapShare)) # converts K/Y to K/L
        self.Rfunc = lambda k : (1.0 + self.CapShare*k**(self.CapShare-1.0) - self.DeprFac)
        self.wFunc = lambda k : ((1.0-self.CapShare)*k**(self.CapShare))/self.wRteSS
        self.KtoLnow_init = self.kSS
        self.RfreeNow_init = self.Rfunc(self.kSS)
        self.wRteNow_init = self.wFunc(self.kSS)
        self.PermShkAggNow_init = 1.0
        self.TranShkAggNow_init = 1.0
        self.TranShkAggDstn = approxMeanOneLognormal(sigma=self.TranShkAggStd,N=self.TranShkAggCount)
        self.PermShkAggDstn = approxMeanOneLognormal(sigma=self.PermShkAggStd,N=self.PermShkAggCount)
        self.AggShkDstn = combineIndepDstns(self.PermShkAggDstn,self.TranShkAggDstn)
        self.kNextFunc = CapitalEvoRule(self.intercept_prev,self.slope_prev)
        
    def reset(self):
        '''
        Reset the economy to prepare for a new simulation.  Sets the time index
        of aggregate shocks to zero and runs Market.reset().
        
        Parameters
        ----------
        none
            
        Returns
        -------
        none
        '''
        self.Shk_idx = 0
        Market.reset(self)
        
    def makeAggShkHist(self):
        '''
        Make simulated histories of aggregate transitory and permanent shocks.
        Histories are of length self.act_T, for use in the general equilibrium
        simulation.
        
        Parameters
        ----------
        none
            
        Returns
        -------
        none
        '''
        sim_periods = self.act_T
        Events      = np.arange(self.AggShkDstn[0].size) # just a list of integers
        EventDraws  = drawDiscrete(self.AggShkDstn[0],Events,sim_periods,seed=0)
        PermShkAggHist = self.AggShkDstn[1][EventDraws]
        TranShkAggHist = self.AggShkDstn[2][EventDraws]
        
        # Store the histories       
        self.PermShkAggHist = PermShkAggHist
        self.TranShkAggHist = TranShkAggHist
        
    def calcRandW(self,pNow,aNow):
        '''
        Calculates the interest factor and wage rate this period using each agent's
        capital stock to get the aggregate capital ratio.
        
        Parameters
        ----------
        pNow : [np.array]
            Agents' current permanent income levels.  Elements of the list corr-
            espond to types in the economy, entries within arrays to agents of
            that type.
        aNow : [np.array]
            Agents' current end-of-period assets (normalized).  Elements of the
            list correspond to types in the economy, entries within arrays to
            agents of that type.
            
        Returns
        -------
        AggVarsNow : CSTWaggVars
            An object containing the aggregate variables for the upcoming period:
            capital-to-labor ratio, interest factor, (normalized) wage rate,
            aggregate permanent and transitory shocks.
        '''
        # Calculate aggregate capital this period
        type_count = len(aNow)
        aAll = np.zeros((type_count,aNow[0].size))
        pAll = np.zeros((type_count,pNow[0].size))
        for j in range(type_count):
            aAll[j,:] = aNow[j]
            pAll[j,:] = pNow[j]
        KtoYnow = np.mean(aAll*pAll) # This version uses end-of-period assets and
        # permanent income to calculate aggregate capital, unlike the Mathematica
        # version, which first applies the idiosyncratic permanent income shocks
        # and then aggregates.  Obviously this is mathematically equivalent.
        
        # Get this period's aggregate shocks
        PermShkAggNow = self.PermShkAggHist[self.Shk_idx]
        TranShkAggNow = self.TranShkAggHist[self.Shk_idx]
        self.Shk_idx += 1
        
        # Calculate the interest factor and wage rate this period
        KtoLnow  = self.convertKtoY(KtoYnow)
        RfreeNow = self.Rfunc(KtoLnow/TranShkAggNow)
        wRteNow  = self.wFunc(KtoLnow/TranShkAggNow)*TranShkAggNow # "effective" wage accounts for labor supply
        
        # Package the results into an object and return it
        AggVarsNow = CobbDouglasAggVars(KtoLnow,RfreeNow,wRteNow,PermShkAggNow,TranShkAggNow)
        return AggVarsNow
        
    def calcCapitalEvoRule(self,KtoLnow):
        '''
        Calculate a new capital evolution rule as an AR1 process based on the history
        of the capital-to-labor ratio from a simulation.
        
        Parameters
        ----------
        KtoLnow : [float]
            List of the history of the simulated  capital-to-labor ratio for an economy.
            
        Returns
        -------
        CSTWdynamics : CSTWdynamicRule
            Object containing a new capital evolution rule, calculated from the
            history of the capital-to-labor ratio.
        '''
        verbose = False
        discard_periods = 200 # Throw out the first T periods to allow the simulation to approach the SS
        update_weight = 0.5   # Proportional weight to put on new function vs old function parameters
        total_periods = len(KtoLnow)
        
        # Auto-regress the log capital-to-labor ratio, one period lag only
        logKtoL_t   = np.log(KtoLnow[discard_periods:(total_periods-1)])
        logKtoL_tp1 = np.log(KtoLnow[(discard_periods+1):total_periods])
        slope, intercept, r_value, p_value, std_err = stats.linregress(logKtoL_t,logKtoL_tp1)
        
        # Make a new capital evolution rule by combining the new regression parameters
        # with the previous guess
        intercept = update_weight*intercept + (1.0-update_weight)*self.intercept_prev
        slope = update_weight*slope + (1.0-update_weight)*self.slope_prev
        kNextFunc = CapitalEvoRule(intercept,slope) # Make a new 
        
        # Save the new values as "previous" values for the next iteration    
        self.intercept_prev = intercept
        self.slope_prev = slope
    
        # Plot the history of the capital ratio for this run and print the new parameters
        if verbose:
            print('intercept=' + str(intercept) + ', slope=' + str(slope) + ', r-sq=' + str(r_value**2))
            plt.plot(KtoLnow[discard_periods:])
            plt.show()
        
        return CapDynamicRule(kNextFunc)
        
                
class CobbDouglasAggVars():
    '''
    A simple class for holding the relevant aggregate variables that should be
    passed from the market to each type.  Includes the capital-to-labor ratio,
    the interest factor, the wage rate, and the aggregate permanent and tran-
    sitory shocks.
    '''
    def __init__(self,KtoLnow,RfreeNow,wRteNow,PermShkAggNow,TranShkAggNow):
        '''
        Make a new instance of CSTWaggVars.
        
        Parameters
        ----------
        KtoLnow : float
            Capital-to-labor ratio in the economy this period.
        RfreeNow : float
            Interest factor on assets in the economy this period.
        wRteNow : float
            Wage rate for labor in the economy this period (normalized by the
            steady state wage rate).
        PermShkAggNow : float
            Permanent shock to aggregate labor productivity this period.
        TranShkAggNow : float
            Transitory shock to aggregate labor productivity this period.
            
        Returns
        -------
        new instance of CSTWaggVars
        '''
        self.KtoLnow       = KtoLnow
        self.RfreeNow      = RfreeNow
        self.wRteNow       = wRteNow
        self.PermShkAggNow = PermShkAggNow
        self.TranShkAggNow = TranShkAggNow
        
class CapitalEvoRule(HARKobject):
    '''
    A class to represent capital evolution rules.  Agents believe that the log
    capital ratio next period is a linear function of the log capital ratio
    this period.
    '''
    def __init__(self,intercept,slope):
        '''
        Make a new instance of CapitalEvoRule.
        
        Parameters
        ----------
        intercept : float
            Intercept of the log-linear capital evolution rule.
        slope : float
            Slope of the log-linear capital evolution rule.
            
        Returns
        -------
        new instance of CapitalEvoRule
        '''
        self.intercept         = intercept
        self.slope             = slope
        self.distance_criteria = ['slope','intercept']
        
    def __call__(self,kNow):
        '''
        Evaluates (expected) capital-to-labor ratio next period as a function
        of the capital-to-labor ratio this period.
        
        Parameters
        ----------
        kNow : float
            Capital-to-labor ratio this period.
            
        Returns
        -------
        kNext : (Expected) capital-to-labor ratio next period.
        '''
        kNext = np.exp(self.intercept + self.slope*np.log(kNow))
        return kNext

    
class CapDynamicRule(HARKobject):
    '''
    Just a container class for passing the capital evolution rule to agents.
    '''
    def __init__(self,kNextFunc):
        '''
        Make a new instance of CSTWdynamicRule.
        
        Parameters
        ----------
        kNextFunc : CapitalEvoRule
            Next period's capital-to-labor ratio as a function of this period's.
            
        Returns
        -------
        new instance of CSTWdynamicRule
        '''
        self.kNextFunc = kNextFunc
        self.distance_criteria = ['kNextFunc']
        
        
###############################################################################
        
if __name__ == '__main__':
    import ConsumerParameters as Params
    from time import clock
    import matplotlib.pyplot as plt
    from HARKutilities import plotFuncs
    mystr = lambda number : "{:.4f}".format(number)
    
    # Make an aggregate shocks consumer
    AggShockExample = AggShockConsumerType(**Params.init_agg_shocks)
    AggShockExample.cycles = 0
    AggShockExample.sim_periods = 3000
    AggShockExample.makeIncShkHist()  # Simulate a history of idiosyncratic shocks
    
    # Make a Cobb-Douglas economy for the agents
    EconomyExample = CobbDouglasEconomy(agents = [AggShockExample],act_T=3000,**Params.init_cobb_douglas)
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
    AggShockExample.unpack_cFunc()
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
    AggShockExample.unpack_cFunc()
    m_grid = np.linspace(0,10,200)
    for k in AggShockExample.kGrid.tolist():
        c_at_this_k = AggShockExample.cFunc[0](m_grid,k*np.ones_like(m_grid))
        plt.plot(m_grid,c_at_this_k)
    plt.show()
    