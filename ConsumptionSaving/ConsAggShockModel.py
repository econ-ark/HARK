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
from HARKinterpolation import LinearInterp, LinearInterpOnInterp1D, ConstantFunction,\
                              VariableLowerBoundFunc2D, BilinearInterp, LowerEnvelope2D, UpperEnvelope
from HARKutilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv,\
                          CRRAutility_invP, CRRAutility_inv, combineIndepDstns,\
                          approxMeanOneLognormal
from HARKsimulation import drawDiscrete
from ConsIndShockModel import ConsumerSolution, IndShockConsumerType
from HARKcore import HARKobject, Market, AgentType
from copy import deepcopy
import matplotlib.pyplot as plt

utility      = CRRAutility
utilityP     = CRRAutilityP
utilityPP    = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv  = CRRAutility_inv

class MargValueFunc2D(HARKobject):
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of v'(m,M) = u'(c(m,M)) holds (with CRRA utility).
    '''
    distance_criteria = ['cFunc','CRRA']
    
    def __init__(self,cFunc,CRRA):
        '''
        Constructor for a new marginal value function object.
        
        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on individual market
            resources and aggregate market resources-to-labor ratio: uP_inv(vPfunc(m,M)).
            Called cFunc because when standard envelope condition applies,
            uP_inv(vPfunc(m,M)) = cFunc(m,M).
        CRRA : float
            Coefficient of relative risk aversion.
            
        Returns
        -------
        new instance of MargValueFunc
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        
    def __call__(self,m,M):
        return utilityP(self.cFunc(m,M),gam=self.CRRA)
        
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
        Make a new instance of AggShockConsumerType, an extension of
        IndShockConsumerType.  Sets appropriate solver and input lists.
        '''
        AgentType.__init__(self,solution_terminal=deepcopy(IndShockConsumerType.solution_terminal_),
                           time_flow=time_flow,pseudo_terminal=False,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(IndShockConsumerType.time_vary_)
        self.time_inv = deepcopy(IndShockConsumerType.time_inv_)
        self.delFromTimeInv('Rfree','vFuncBool','CubicBool')
        self.poststate_vars = IndShockConsumerType.poststate_vars_
        self.solveOnePeriod = solveConsAggShock
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
        self.aLvlNow = self.kInit*np.ones(self.AgentCount) # Start simulation near SS
        self.aNrmNow = self.aLvlNow/self.pLvlNow
        
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
        vPfunc_terminal = lambda m,M : m**(-self.CRRA)
        cFunc_terminal  = lambda m,M : m
        mNrmMin_terminal = ConstantFunction(0)
        self.solution_terminal = ConsumerSolution(cFunc=cFunc_terminal,vPfunc=vPfunc_terminal,mNrmMin=mNrmMin_terminal)
        
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
        self.kInit = Economy.kSS                            # Initialize simulation assets to steady state
        self.aNrmInitMean = np.log(0.00000001)              # Initialize newborn assets to nearly zero
        self.MGrid  = Economy.kSS*self.kGridBase            # Market resources grid adjusted around SS capital ratio
        self.AFunc = Economy.AFunc                          # Next period's aggregate savings function
        self.Rfunc = Economy.Rfunc                          # Interest factor as function of capital ratio
        self.wFunc = Economy.wFunc                          # (Normalized) wage rate as function of capital ratio
        self.DeprFac = Economy.DeprFac                      # Rate of capital depreciation
        IncomeDstnWithAggShks = combineIndepDstns(self.PermShkDstn,self.TranShkDstn,Economy.PermShkAggDstn,Economy.TranShkAggDstn)
        self.IncomeDstn = [IncomeDstnWithAggShks]           # Discrete income distribution with aggregate and idiosyncratic shocks
        self.addToTimeInv('MGrid','AFunc','Rfunc', 'wFunc','DeprFac')
        
    def simBirth(self,which_agents):
        '''
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).
        
        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".
        
        Returns
        -------
        None
        '''
        IndShockConsumerType.simBirth(self,which_agents)
        if hasattr(self,'aLvlNow'):
            self.aLvlNow[which_agents] = self.aNrmNow[which_agents]*self.pLvlNow[which_agents]
        else:
            self.aLvlNow = self.aNrmNow*self.pLvlNow
        
    def simDeath(self):
        '''
        Randomly determine which consumers die, and distribute their wealth among the survivors.
        This method only works if there is only one period in the cycle.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        who_dies : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        '''
        # Divide agents into wealth groups, kill one random agent per wealth group
#        order = np.argsort(self.aLvlNow)
#        how_many_die = int(self.AgentCount*(1.0-self.LivPrb[0]))
#        group_size = self.AgentCount/how_many_die # This should be an integer
#        base_idx = self.RNG.randint(0,group_size,size=how_many_die)
#        kill_by_rank = np.arange(how_many_die,dtype=int)*group_size + base_idx
#        who_dies = np.zeros(self.AgentCount,dtype=bool)
#        who_dies[order[kill_by_rank]] = True
        how_many_die = int(round(self.AgentCount*(1.0-self.LivPrb[0])))
        base_bool = np.zeros(self.AgentCount,dtype=bool)
        base_bool[0:how_many_die] = True
        who_dies = self.RNG.permutation(base_bool)
        if self.T_age is not None:
            who_dies[self.t_age >= self.T_age] = True
        
        # Divide up the wealth of those who die, giving it to those who survive
        who_lives = np.logical_not(who_dies)
        wealth_living = np.sum(self.aLvlNow[who_lives])
        wealth_dead = np.sum(self.aLvlNow[who_dies])
        Ractuarial = 1.0 + wealth_dead/wealth_living
        self.aNrmNow[who_lives] = self.aNrmNow[who_lives]*Ractuarial
        self.aLvlNow[who_lives] = self.aLvlNow[who_lives]*Ractuarial
        return who_dies
        
    def getRfree(self):
        '''
        Returns an array of size self.AgentCount with self.RfreeNow in every entry.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        '''
        RfreeNow = self.RfreeNow*np.ones(self.AgentCount)
        return RfreeNow
        
    def getShocks(self):
        '''
        Finds the effective permanent and transitory shocks this period by combining the aggregate
        and idiosyncratic shocks of each type.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        IndShockConsumerType.getShocks(self) # Update idiosyncratic shocks
        self.TranShkNow = self.TranShkNow*self.TranShkAggNow*self.wRteNow
        self.PermShkNow = self.PermShkNow*self.PermShkAggNow
        
    def getControls(self):
        '''
        Calculates consumption for each consumer of this type using the consumption functions.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        MaggNow = self.MaggNow*np.ones(self.AgentCount)
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these] = self.solution[t].cFunc(self.mNrmNow[these],MaggNow[these])
            MPCnow[these]  = self.solution[t].cFunc.derivativeX(self.mNrmNow[these],MaggNow[these]) # Marginal propensity to consume
        self.cNrmNow = cNrmNow
        self.MPCnow = MPCnow
        return None
                
    def marketAction(self):
        '''
        In the aggregate shocks model, the "market action" is to simulate one
        period of receiving income and choosing how much to consume.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        self.simulate(1)
        
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


def solveConsAggShock(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,PermGroFac,aXtraGrid,BoroCnstArt,MGrid,AFunc,Rfunc,wFunc, DeprFac):
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
    BoroCnstArt : float
        Artificial borrowing constraint; minimum allowable end-of-period asset-to-
        permanent-income ratio.  Unlike other models, this *can't* be None.
    MNrmGrid : np.array
        A grid of aggregate market resourses to permanent income in the economy.
    AFunc : function
        Aggregate savings as a function of aggregate market resources.
    Rfunc : function
        The net interest factor on assets as a function of capital ratio k.
    wFunc : function
        The wage rate for labor as a function of capital-to-labor ratio k.
    DeprFac : float
        Capital Depreciation Rate
                    
    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (linear interpolation over linear interpola-
        tions) and marginal value function vPfunc.
    '''
    # Unpack next period's solution
    vPfuncNext = solution_next.vPfunc
    mNrmMinNext = solution_next.mNrmMin
    
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
        
    # Loop through the values in MGrid and calculate a linear consumption function for each
    cFuncBaseByM_list = []
    BoroCnstNat_array = np.zeros(MGrid.size)
    mNrmMinNext_array = mNrmMinNext(AFunc(MGrid)*(1-DeprFac))
    for j in range(MGrid.size):
        MNow = MGrid[j]
        AggA = AFunc(MNow)
        
        # Calculate returns to capital and labor in the next period        
        kNextEff_array = AggA*(1-DeprFac)/(PermGroFac*PermShkAggValsNext_tiled*TranShkAggValsNext_tiled)
        Reff_array = Rfunc(kNextEff_array)/LivPrb # Effective interest rate
        wEff_array = wFunc(kNextEff_array)*TranShkAggValsNext_tiled # Effective wage rate (accounts for labor supply)
        PermShkTotal_array = PermGroFac*PermShkValsNext_tiled*PermShkAggValsNext_tiled # total / combined permanent shock
        MNext_array = AggA*(1-DeprFac)/(PermGroFac*PermShkAggValsNext_tiled)*Reff_array + wEff_array
        
        # Find the natural borrowing constraint for this capital-to-labor ratio
        aNrmMin_candidates = PermGroFac*PermShkValsNext*PermShkAggValsNext/Reff_array[:,0]*(mNrmMinNext_array[j] - wEff_array[:,0]*TranShkValsNext)
        aNrmMin = np.max(aNrmMin_candidates)
        BoroCnstNat = aNrmMin
        BoroCnstNat_array[j] = BoroCnstNat
        
        # Calculate market resources next period (and a constant array of capital-to-labor ratio)       
        mNrmNext_array = Reff_array*(aNrmNow_tiled + aNrmMin)/PermShkTotal_array + TranShkValsNext_tiled*wEff_array
                
        # Find marginal value next period at every income shock realization and every aggregate market resource gridpoint
        vPnext_array = Reff_array*PermShkTotal_array**(-CRRA)*vPfuncNext(mNrmNext_array,MNext_array)
        
        # Calculate expectated marginal value at the end of the period at every asset gridpoint
        EndOfPrdvP = DiscFac*LivPrb*PermGroFac**(-CRRA)*np.sum(vPnext_array*ShkPrbsNext_tiled,axis=0)
        
        # Calculate optimal consumption from each asset gridpoint, and construct a linear interpolation
        cNrmNow = EndOfPrdvP**(-1.0/CRRA)
        mNrmNow = (aNrmNow + aNrmMin) + cNrmNow
        c_for_interpolation = np.insert(cNrmNow,0,0.0) # Add liquidity constrained portion
        m_for_interpolation = np.insert(mNrmNow-BoroCnstNat,0,0.0)
        cFuncBase_j = LinearInterp(m_for_interpolation,c_for_interpolation)
        
        # Add the k-specific consumption function to the list
        cFuncBaseByM_list.append(cFuncBase_j)
    
    # Construct the overall unconstrained consumption function by combining the k-specific functions
    BoroCnstNat = LinearInterp(np.insert(MGrid,0,0.0),np.insert(BoroCnstNat_array,0,0.0))
    cFuncBase = LinearInterpOnInterp1D(cFuncBaseByM_list,MGrid)
    cFuncUnc  = VariableLowerBoundFunc2D(cFuncBase,BoroCnstNat)
    
    # Make the constrained consumption function and combine it with the unconstrained component
    cFuncCnst = BilinearInterp(np.array([[0.0,0.0],[1.0,1.0]]),
                                           np.array([BoroCnstArt,BoroCnstArt+1.0]),np.array([0.0,1.0]))
    cFuncNow = LowerEnvelope2D(cFuncUnc,cFuncCnst)
    
    # Make the minimum m function as the greater of the natural and artificial constraints
    mNrmMinNow = UpperEnvelope(BoroCnstNat,ConstantFunction(BoroCnstArt))
    
    # Construct the marginal value function using the envelope condition
    vPfuncNow = MargValueFunc2D(cFuncNow,CRRA)
    
    # Pack up and return the solution
    solution_now = ConsumerSolution(cFunc=cFuncNow,vPfunc=vPfuncNow,mNrmMin=mNrmMinNow)
    return solution_now
    
###############################################################################
    
class CobbDouglasEconomy(Market):            
    '''
    A class to represent an economy with a Cobb-Douglas aggregate production
    function over labor and capital, extending HARKcore.Market.  The "aggregate
    market process" for this market combines all individuals' asset holdings
    into aggregate capital, yielding the interest factor on assets and the wage
    rate for the upcoming period.
    
    Note: The current implementation assumes a constant labor supply, but
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
                            sow_vars=['MaggNow','AggANow','RfreeNow','wRteNow','PermShkAggNow','TranShkAggNow'],
                            reap_vars=['aLvlNow','pLvlNow'],
                            track_vars=['MaggNow','AggANow'],
                            dyn_vars=['AFunc'],
                            tolerance=tolerance,
                            act_T=act_T)
        self.assignParameters(**kwds)
        self.max_loops = 20
        self.update()
    
    
    def millRule(self,aLvlNow,pLvlNow):
        '''
        Function to calculate the capital to labor ratio, interest factor, and
        wage rate based on each agent's current state.  Just calls calcRandW().
        
        See documentation for calcRandW for more information.
        '''
        return self.calcRandW(aLvlNow,pLvlNow)
        
    def calcDynamics(self,MaggNow,AggANow):
        '''
        Calculates a new dynamic rule for the economy: end of period savings as
        a function of aggregate market resources.  Just calls calcAFunc().
        
        See documentation for calcCapitalEvoRule for more information.
        '''
        return self.calcAFunc(MaggNow,AggANow)
        
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
        self.wFunc = lambda k : ((1.0-self.CapShare)*k**(self.CapShare))
        self.KtoLnow_init = self.kSS
        self.MaggNow_init = self.kSS
        self.AggANow_init = self.kSS
        self.RfreeNow_init = self.Rfunc(self.kSS)
        self.wRteNow_init = self.wFunc(self.kSS)
        self.PermShkAggNow_init = 1.0
        self.TranShkAggNow_init = 1.0
        self.TranShkAggDstn = approxMeanOneLognormal(sigma=self.TranShkAggStd,N=self.TranShkAggCount)
        self.PermShkAggDstn = approxMeanOneLognormal(sigma=self.PermShkAggStd,N=self.PermShkAggCount)
        self.AggShkDstn = combineIndepDstns(self.PermShkAggDstn,self.TranShkAggDstn)
        self.AFunc = CapitalEvoRule(self.intercept_prev,self.slope_prev)
        
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
        EventDraws  = drawDiscrete(N=sim_periods,P=self.AggShkDstn[0],X=Events,seed=0)
        PermShkAggHist = self.AggShkDstn[1][EventDraws]
        TranShkAggHist = self.AggShkDstn[2][EventDraws]
        
        # Store the histories       
        self.PermShkAggHist = PermShkAggHist
        self.TranShkAggHist = TranShkAggHist
        
    def calcRandW(self,aLvlNow,pLvlNow):
        '''
        Calculates the interest factor and wage rate this period using each agent's
        capital stock to get the aggregate capital ratio.
        
        Parameters
        ----------
        aLvlNow : [np.array]
            Agents' current end-of-period assets.  Elements of the list correspond
            to types in the economy, entries within arrays to agents of that type.
            
        Returns
        -------
        AggVarsNow : CobbDouglasAggVars
            An object containing the aggregate variables for the upcoming period:
            capital-to-labor ratio, interest factor, (normalized) wage rate,
            aggregate permanent and transitory shocks.
        '''
        # Calculate aggregate savings
        AggANow = np.mean(np.array(aLvlNow))/np.mean(pLvlNow)
        # Calculate aggregate capital this period
        AggregateK = (1.0 - self.DeprFac)*np.mean(np.array(aLvlNow)) # This version uses end-of-period assets and
        # permanent income to calculate aggregate capital, unlike the Mathematica
        # version, which first applies the idiosyncratic permanent income shocks
        # and then aggregates.  Obviously this is mathematically equivalent.
        
        # Get this period's aggregate shocks
        PermShkAggNow = self.PermShkAggHist[self.Shk_idx]
        TranShkAggNow = self.TranShkAggHist[self.Shk_idx]
        self.Shk_idx += 1
        
        AggregateL = np.mean(pLvlNow)*PermShkAggNow    #STRICTLY WE NEED THE PERMGROFAC HERE TOO
        
        # Calculate the interest factor and wage rate this period
        KtoLnow = AggregateK/AggregateL
        self.KtoYnow = KtoLnow**(1.0-self.CapShare)
        RfreeNow = self.Rfunc(KtoLnow/TranShkAggNow)
        wRteNow  = self.wFunc(KtoLnow/TranShkAggNow)
        MaggNow =KtoLnow*RfreeNow + wRteNow*TranShkAggNow
        
        self.KtoLnow = KtoLnow   # Need to store this as it is not a sow variable
        
        # Package the results into an object and return it
        AggVarsNow = CobbDouglasAggVars(MaggNow, AggANow,KtoLnow,RfreeNow,wRteNow,PermShkAggNow,TranShkAggNow)
        return AggVarsNow
        
    def calcAFunc(self,MaggNow,AggANow):
        '''
        Calculate a new aggregate savings rule based on the history
        of the aggregate savings and aggregate market resources from a simulation.
        
        Parameters
        ----------
        MaggNow : [float]
            List of the history of the simulated  aggregate market resources for an economy.
        AggANow : [float]
            List of the history of the simulated  aggregate savings for an economy.
            
        Returns
        -------
        (unnamed) : CapDynamicRule
            Object containing a new savings rule
        '''
        verbose = True
        discard_periods = 200 # Throw out the first T periods to allow the simulation to approach the SS
        update_weight = 0.5   # Proportional weight to put on new function vs old function parameters
        total_periods = len(MaggNow)
        
        # Regress the log savings against log market resources
        logAggA   = np.log(AggANow[discard_periods:total_periods])
        logMagg = np.log(MaggNow[discard_periods-1:total_periods-1])
        slope, intercept, r_value, p_value, std_err = stats.linregress(logMagg,logAggA)
        
        # Make a new aggregate savings rule by combining the new regression parameters
        # with the previous guess
        intercept = update_weight*intercept + (1.0-update_weight)*self.intercept_prev
        slope = update_weight*slope + (1.0-update_weight)*self.slope_prev
        AFunc = CapitalEvoRule(intercept,slope) # Make a new next-period capital function
        
        # Save the new values as "previous" values for the next iteration    
        self.intercept_prev = intercept
        self.slope_prev = slope
    
        # Plot the history of the capital ratio for this run and print the new parameters
        if verbose:
            print('intercept=' + str(intercept) + ', slope=' + str(slope) + ', r-sq=' + str(r_value**2))
            #plot_start = discard_periods
            #plt.plot(KtoLnow[plot_start:])
            #plt.show()
        
        return CapDynamicRule(AFunc)
        
        
class SmallOpenEconomy(Market):
    '''
    A class for representing a small open economy, where the wage rate and interest rate are
    exogenously determined by some "global" rate.  However, the economy is still subject to
    aggregate productivity shocks.
    '''
    def __init__(self,agents=[],tolerance=0.0001,act_T=1000,**kwds):
        '''
        Make a new instance of SmallOpenEconomy by filling in attributes specific to this kind of market.
        
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
                            reap_vars=[],
                            track_vars=['KtoLnow'],
                            dyn_vars=[],
                            tolerance=tolerance,
                            act_T=act_T)
        self.assignParameters(**kwds)
        self.update()
        
    def update(self):
        '''
        Use primitive parameters to set basic objects.  This is an extremely stripped-down version
        of update for CobbDouglasEconomy.
        
        Parameters
        ----------
        none
            
        Returns
        -------
        none
        '''
        self.kSS = 1.0
        self.KtoLnow_init = self.kSS
        self.Rfunc = ConstantFunction(self.Rfree)
        self.wFunc = ConstantFunction(self.wRte)
        self.RfreeNow_init = self.Rfunc(self.kSS)
        self.wRteNow_init = self.wFunc(self.kSS)
        self.PermShkAggNow_init = 1.0
        self.TranShkAggNow_init = 1.0
        self.TranShkAggDstn = approxMeanOneLognormal(sigma=self.TranShkAggStd,N=self.TranShkAggCount)
        self.PermShkAggDstn = approxMeanOneLognormal(sigma=self.PermShkAggStd,N=self.PermShkAggCount)
        self.AggShkDstn = combineIndepDstns(self.PermShkAggDstn,self.TranShkAggDstn)
        self.kNextFunc = ConstantFunction(1.0)
        
    def millRule(self):
        '''
        No aggregation occurs for a small open economy, because the wage and interest rates are
        exogenously determined.  However, aggregate shocks may occur.
        
        See documentation for getAggShocks() for more information.
        '''
        return self.getAggShocks()
        
    def calcDynamics(self,KtoLnow):
        '''
        Calculates a new dynamic rule for the economy, which is just an empty object.
        There is no "dynamic rule" for a small open economy, because K/L does not generate w and R.
        '''
        return HARKobject()
        
    def reset(self):
        '''
        Reset the economy to prepare for a new simulation.  Sets the time index of aggregate shocks
        to zero and runs Market.reset().  This replicates the reset method for CobbDouglasEconomy;
        future version should create parent class of that class and this one.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        self.Shk_idx = 0
        Market.reset(self)
        
    def makeAggShkHist(self):
        '''
        Make simulated histories of aggregate transitory and permanent shocks. Histories are of
        length self.act_T, for use in the general equilibrium simulation.  This replicates the same
        method for CobbDouglasEconomy; future version should create parent class.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        sim_periods = self.act_T
        Events      = np.arange(self.AggShkDstn[0].size) # just a list of integers
        EventDraws  = drawDiscrete(N=sim_periods,P=self.AggShkDstn[0],X=Events,seed=0)
        PermShkAggHist = self.AggShkDstn[1][EventDraws]
        TranShkAggHist = self.AggShkDstn[2][EventDraws]
        
        # Store the histories       
        self.PermShkAggHist = PermShkAggHist
        self.TranShkAggHist = TranShkAggHist
        
    def getAggShocks(self):
        '''
        Returns aggregate state variables and shocks for this period.  The capital-to-labor ratio
        is irrelevant and thus treated as constant, and the wage and interest rates are also
        constant.  However, aggregate shocks are assigned from a prespecified history.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        AggVarsNow : CobbDouglasAggVars
            Aggregate state and shock variables for this period.
        '''
        # Get this period's aggregate shocks
        PermShkAggNow = self.PermShkAggHist[self.Shk_idx]
        TranShkAggNow = self.TranShkAggHist[self.Shk_idx]
        self.Shk_idx += 1
        
        RfreeNow = np.nan
        wRteNow  = np.nan
        # Aggregates are also irrelavent
        AggANow = np.nan
        MaggNow = np.nan
        
        # Package the results into an object and return it
        AggVarsNow = CobbDouglasAggVars(MaggNow, AggANow,KtoLnow,RfreeNow,wRteNow,PermShkAggNow,TranShkAggNow)
        return AggVarsNow
                
class CobbDouglasAggVars():
    '''
    A simple class for holding the relevant aggregate variables that should be
    passed from the market to each type.  Includes the capital-to-labor ratio,
    the interest factor, the wage rate, and the aggregate permanent and tran-
    sitory shocks.
    '''
    def __init__(self,MaggNow,AggANow,KtoLnow,RfreeNow,wRteNow,PermShkAggNow,TranShkAggNow):
        '''
        Make a new instance of CobbDouglasAggVars.
        
        Parameters
        ----------
        MaggNow : float
            Aggregate market resources for this period normalized by mean permanent income
        AggANow : float
            Aggregate savings for this period normalized by mean permanent income
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
        None
        '''
        self.MaggNow       = MaggNow
        self.AggANow       = AggANow
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
        
    def __call__(self,MNow):
        '''
        Evaluates aggregate savings as a function
        of the aggregate market resources this period.
        
        Parameters
        ----------
        MNow : float
            Aggregate market resources this period.
            
        Returns
        -------
        AggA : Aggregate savings this period.
        '''
        AggA = np.exp(self.intercept + self.slope*np.log(MNow))
        return AggA

    
class CapDynamicRule(HARKobject):
    '''
    Just a container class for passing the capital evolution rule to agents.
    '''
    def __init__(self,AFunc):
        '''
        Make a new instance of CapDynamicRule.
        
        Parameters
        ----------
        AFunc : CapitalEvoRule
            Aggregate savings as a function of aggregate market resources.
            
        Returns
        -------
        None
        '''
        self.AFunc = AFunc
        self.distance_criteria = ['AFunc']
        
        
###############################################################################
        
if __name__ == '__main__':
    import ConsumerParameters as Params
    from time import clock
    from HARKutilities import plotFuncs
    mystr = lambda number : "{:.4f}".format(number)
    
    # Make an aggregate shocks consumer type
    AggShockExample = AggShockConsumerType(**Params.init_agg_shocks)
    AggShockExample.cycles = 0
    
    # Make a Cobb-Douglas economy for the agents
    EconomyExample = CobbDouglasEconomy(agents = [AggShockExample],**Params.init_cobb_douglas)
    EconomyExample.makeAggShkHist() # Simulate a history of aggregate shocks
    
    # Have the consumers inherit relevant objects from the economy
    AggShockExample.getEconomyData(EconomyExample)
    
    # Solve the microeconomic model for the aggregate shocks example type (and display results)
    t_start = clock()
    AggShockExample.solve()
    t_end = clock()
    print('Solving an aggregate shocks consumer took ' + mystr(t_end-t_start) + ' seconds.')
    print('Consumption function at each market resources-to-labor ratio gridpoint:')
    m_grid = np.linspace(0,10,200)
    AggShockExample.unpackcFunc()
    for M in AggShockExample.MGrid.tolist():
        mMin = AggShockExample.solution[0].mNrmMin(M)
        c_at_this_M = AggShockExample.cFunc[0](m_grid+mMin,M*np.ones_like(m_grid))
        plt.plot(m_grid+mMin,c_at_this_M)
    plt.show()
    
    # Solve the "macroeconomic" model by searching for a "fixed point dynamic rule"
    t_start = clock()
    EconomyExample.solve()
    t_end = clock()
    print('Solving the "macroeconomic" aggregate shocks model took ' + str(t_end - t_start) + ' seconds.')
    print('Aggregate savings as a function of aggregate market resources:')
    plotFuncs(EconomyExample.AFunc,0,2*EconomyExample.kSS)
    print('Consumption function at each aggregate market resources gridpoint (in general equilibrium):')
    AggShockExample.unpackcFunc()
    m_grid = np.linspace(0,10,200)
    AggShockExample.unpackcFunc()
    for M in AggShockExample.MGrid.tolist():
        mMin = AggShockExample.solution[0].mNrmMin(M)
        c_at_this_M = AggShockExample.cFunc[0](m_grid+mMin,M*np.ones_like(m_grid))
        plt.plot(m_grid+mMin,c_at_this_M)
    plt.show()
    