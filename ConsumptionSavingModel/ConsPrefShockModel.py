'''
Extensions to ConsIndShockModel concerning models with preference shocks.
It currently only has one model, in which utility is subject to an iid lognormal
multiplicative shock each period; it assumes that there are different interest
rates on borrowing and saving.
'''
import sys 
sys.path.insert(0,'../')

import numpy as np
from HARKutilities import approxMeanOneLognormal
from ConsIndShockModel import KinkedRconsumerType, ConsumerSolution, ConsKinkedRsolver, \
                                   ValueFunc, MargValueFunc
from HARKinterpolation import LinearInterpOnInterp1D, LinearInterp, CubicInterp, LowerEnvelope

class PrefShockConsumerType(KinkedRconsumerType):
    '''
    A class for representing consumers who experience multiplicative shocks to
    utility each period, specified as iid lognormal.
    '''
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data, and construct objects
        to be used during solution (income distribution, assets grid, etc).
        See ConsumerParameters.init_consumer_objects for a dictionary of
        the keywords that should be passed to the constructor.
        
        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.
        
        Returns
        -------
        None
        '''      
        KinkedRconsumerType.__init__(self,**kwds)
        self.solveOnePeriod = solveConsPrefShock # Choose correct solver
    
    def update(self):
        '''
        Updates the assets grid, income process, terminal period solution, and
        preference shock process.  A very slight extension of IndShockConsumerType.update()
        for the preference shock model.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        KinkedRconsumerType.update(self)  # Update assets grid, income process, terminal solution
        self.updatePrefShockProcess()     # Update the discrete preference shock process
        
    def updatePrefShockProcess(self):
        '''
        Make a discrete preference shock structure for each period in the cycle
        for this agent type, storing them as attributes of self for use in the
        solution (and other methods).
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        time_orig = self.time_flow
        self.timeFwd()
        
        PrefShkDstn = [] # discrete distributions of preference shocks
        for t in range(len(self.PrefShkStd)):
            PrefShkStd = self.PrefShkStd[t]
            PrefShkDstn.append(approxMeanOneLognormal(N=self.PrefShkCount,
                                                      sigma=PrefShkStd,tail_N=self.PrefShk_tail_N))
            
        # Store the preference shocks in self (time-varying) and restore time flow
        self.PrefShkDstn = PrefShkDstn
        self.addToTimeVary('PrefShkDstn')
        if not time_orig:
            self.timeRev()
            
    def makePrefShkHist(self):
        '''
        Makes histories of simulated preference shocks for this consumer type by
        drawing from the shock distribution's true lognormal form.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        orig_time = self.time_flow
        self.timeFwd()
        self.resetRNG()
        
        # Initialize the preference shock history
        PrefShkHist      = np.zeros((self.sim_periods,self.Nagents)) + np.nan
        PrefShkHist[0,:] = 1.0
        t_idx            = 0
        
        # Make discrete distributions of preference shocks to permute
        base_dstns = []
        for t_idx in range(len(self.PrefShkStd)):
            temp_dstn = approxMeanOneLognormal(N=self.Nagents,sigma=self.PrefShkStd[t_idx])
            base_dstns.append(temp_dstn[1]) # only take values, not probs
        
        # Fill in the preference shock history
        for t in range(1,self.sim_periods):
            dstn_now         = base_dstns[t_idx]
            PrefShkHist[t,:] = self.RNG.permutation(dstn_now)
            t_idx += 1
            if t_idx >= len(self.PrefShkStd):
                t_idx = 0
                
        self.PrefShkHist = PrefShkHist
        if not orig_time:
            self.timeRev()
            
    def advanceIncShks(self):
        '''
        Advance the permanent and transitory income shocks to the next period of
        the shock history objects, after first advancing the preference shocks.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        self.PrefShkNow = self.PrefShkHist[self.Shk_idx,:]
        KinkedRconsumerType.advanceIncShks(self)
            
    def simOnePrd(self):
        '''
        Simulate a single period of a consumption-saving model with permanent
        and transitory income shocks plus multiplicative utility shocks.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        # Unpack objects from self for convenience
        aPrev          = self.aNow
        pPrev          = self.pNow
        TranShkNow     = self.TranShkNow
        PermShkNow     = self.PermShkNow
        PrefShkNow     = self.PrefShkNow
        if hasattr(self,'RboroNow'):
            RboroNow   = self.RboroNow
            RsaveNow   = self.RsaveNow
            RfreeNow   = RboroNow*np.ones_like(aPrev)
            RfreeNow[aPrev > 0] = RsaveNow
        else:
            RfreeNow   = self.RfreeNow
        cFuncNow       = self.cFuncNow
        
        # Simulate the period
        pNow    = pPrev*PermShkNow      # Updated permanent income level
        ReffNow = RfreeNow/PermShkNow   # "effective" interest factor on normalized assets
        bNow    = ReffNow*aPrev         # Bank balances before labor income
        mNow    = bNow + TranShkNow     # Market resources after income
        cNow    = cFuncNow(mNow,PrefShkNow) # Consumption (normalized)
        MPCnow  = cFuncNow.derivativeX(mNow,PrefShkNow) # Marginal propensity to consume
        aNow    = mNow - cNow           # Assets after all actions are accomplished
        
        # Store the new state and control variables
        self.pNow   = pNow
        self.bNow   = bNow
        self.mNow   = mNow
        self.cNow   = cNow
        self.MPCnow = MPCnow
        self.aNow   = aNow
 
       
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


class ConsPrefShockSolver(ConsKinkedRsolver):
    '''
    A class for solving the one period consumption-saving problem with risky
    income (permanent and transitory shocks), a different interest factor on
    borrowing and saving, and multiplicative shocks to utility each period.
    '''
    def __init__(self,solution_next,IncomeDstn,PrefShkDstn,LivPrb,DiscFac,CRRA,
                      Rboro,Rsave,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver for problems with risky income, a different
        interest rate on borrowing and saving, and multiplicative shocks to utility.
        
        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to the succeeding one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        PrefShkDstn : [np.array]
            Discrete distribution of the multiplicative utility shifter.  Order:
            probabilities, preference shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.    
        DiscFac : float
            Intertemporal discount factor for future utility.        
        CRRA : float
            Coefficient of relative risk aversion.
        Rboro: float
            Interest factor on assets between this period and the succeeding
            period when assets are negative.
        Rsave: float
            Interest factor on assets between this period and the succeeding
            period when assets are positive.
        PermGroGac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.
        aXtraGrid: np.array
            Array of "extra" end-of-period asset values-- assets above the
            absolute minimum acceptable level.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear inter-
            polation.
            
        Returns
        -------
        None
        '''
        ConsKinkedRsolver.__init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,
                      Rboro,Rsave,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.PrefShkPrbs = PrefShkDstn[0]
        self.PrefShkVals = PrefShkDstn[1]
    
    def getPointsForInterpolation(self,EndOfPrdvP,aNrmNow):
        '''
        Find endogenous interpolation points for each asset point and each
        discrete preference shock.
        
        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrmNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
            
        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        '''
        c_base       = self.uPinv(EndOfPrdvP)
        PrefShkCount = self.PrefShkVals.size
        PrefShk_temp = np.tile(np.reshape(self.PrefShkVals**(1.0/self.CRRA),(PrefShkCount,1)),
                               (1,c_base.size))
        self.cNrmNow = np.tile(c_base,(PrefShkCount,1))*PrefShk_temp
        self.mNrmNow = self.cNrmNow + np.tile(aNrmNow,(PrefShkCount,1))
        
        # Add the bottom point to the c and m arrays
        m_for_interpolation = np.concatenate((self.BoroCnstNat*np.ones((PrefShkCount,1)),
                                              self.mNrmNow),axis=1)
        c_for_interpolation = np.concatenate((np.zeros((PrefShkCount,1)),self.cNrmNow),axis=1)
        return c_for_interpolation,m_for_interpolation
    
    def usePointsForInterpolation(self,cNrm,mNrm,interpolator):
        '''
        Make a basic solution object with a consumption function and marginal
        value function (unconditional on the preference shock).
        
        Parameters
        ----------
        cNrm : np.array
            Consumption points for interpolation.
        mNrm : np.array
            Corresponding market resource points for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.
            
        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        '''
        # Make the preference-shock specific consumption functions
        PrefShkCount = self.PrefShkVals.size
        cFunc_list   = []
        for j in range(PrefShkCount):
            MPCmin_j         = self.MPCminNow*self.PrefShkVals[j]**(1.0/self.CRRA)
            cFunc_this_shock = LowerEnvelope(LinearInterp(mNrm[j,:],cNrm[j,:],
                                             intercept_limit=self.hNrmNow*MPCmin_j,
                                             slope_limit=MPCmin_j),self.cFuncNowCnst)
            cFunc_list.append(cFunc_this_shock)
            
        # Combine the list of consumption functions into a single interpolation
        cFuncNow = LinearInterpOnInterp1D(cFunc_list,self.PrefShkVals)
            
        # Make the ex ante marginal value function (before the preference shock)
        m_grid = self.aXtraGrid + self.mNrmMinNow
        vP_vec = np.zeros_like(m_grid)
        for j in range(PrefShkCount): # numeric integration over the preference shock
            vP_vec += self.uP(cFunc_list[j](m_grid))*self.PrefShkPrbs[j]*self.PrefShkVals[j]
        vPnvrs_vec = self.uPinv(vP_vec)
        vPfuncNow  = MargValueFunc(LinearInterp(m_grid,vPnvrs_vec),self.CRRA)
    
        # Store the results in a solution object and return it
        solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow)
        return solution_now
        
    def makevFunc(self,solution):
        '''
        Make the beginning-of-period value function (unconditional on the shock).
        
        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, which must include the
            consumption function.
            
        Returns
        -------
        vFuncNow : ValueFunc
            A representation of the value function for this period, defined over
            normalized market resources m: v = vFuncNow(m).
        '''
        # Compute expected value and marginal value on a grid of market resources,
        # accounting for all of the discrete preference shocks
        PrefShkCount = self.PrefShkVals.size
        mNrm_temp   = self.mNrmMinNow + self.aXtraGrid
        vNrmNow     = np.zeros_like(mNrm_temp)
        vPnow       = np.zeros_like(mNrm_temp)
        for j in range(PrefShkCount):
            this_shock  = self.PrefShkVals[j]
            this_prob   = self.PrefShkPrbs[j]
            cNrmNow     = solution.cFunc(mNrm_temp,this_shock*np.ones_like(mNrm_temp))
            aNrmNow     = mNrm_temp - cNrmNow
            vNrmNow    += this_prob*(this_shock*self.u(cNrmNow) + self.EndOfPrdvFunc(aNrmNow))
            vPnow      += this_prob*this_shock*self.uP(cNrmNow)
        
        # Construct the beginning-of-period value function
        vNvrs        = self.uinv(vNrmNow) # value transformed through inverse utility
        vNvrsP       = vPnow*self.uinvP(vNrmNow)
        mNrm_temp    = np.insert(mNrm_temp,0,self.mNrmMinNow)
        vNvrs        = np.insert(vNvrs,0,0.0)
        vNvrsP       = np.insert(vNvrsP,0,self.MPCmaxEff**(-self.CRRA/(1.0-self.CRRA)))
        MPCminNvrs   = self.MPCminNow**(-self.CRRA/(1.0-self.CRRA))
        vNvrsFuncNow = CubicInterp(mNrm_temp,vNvrs,vNvrsP,MPCminNvrs*self.hNrmNow,MPCminNvrs)
        vFuncNow     = ValueFunc(vNvrsFuncNow,self.CRRA)
        return vFuncNow
        
        
def solveConsPrefShock(solution_next,IncomeDstn,PrefShkDstn,
                       LivPrb,DiscFac,CRRA,Rboro,Rsave,PermGroFac,BoroCnstArt,
                       aXtraGrid,vFuncBool,CubicBool):
    '''
    Solves a single period of a consumption-saving model with preference shocks
    to marginal utility.  Problem is solved using the method of endogenous gridpoints.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to the succeeding one period problem.
    IncomeDstn : [np.array]
        A list containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.
    PrefShkDstn : [np.array]
        Discrete distribution of the multiplicative utility shifter.  Order:
        probabilities, preference shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.    
    DiscFac : float
        Intertemporal discount factor for future utility.        
    CRRA : float
        Coefficient of relative risk aversion.
    Rboro: float
        Interest factor on assets between this period and the succeeding
        period when assets are negative.
    Rsave: float
        Interest factor on assets between this period and the succeeding
        period when assets are positive.
    PermGroGac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.

    Returns
    -------
    solution: ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (using linear splines), a marginal value
        function vPfunc, a minimum acceptable level of normalized market re-
        sources mNrmMin, normalized human wealth hNrm, and bounding MPCs MPCmin
        and MPCmax.  It might also have a value function vFunc.  The consumption
        function is defined over normalized market resources and the preference
        shock, c = cFunc(m,PrefShk), but the (marginal) value function is defined
        unconditionally on the shock, just before it is revealed.
    '''
    solver = ConsPrefShockSolver(solution_next,IncomeDstn,PrefShkDstn,LivPrb,
                             DiscFac,CRRA,Rboro,Rsave,PermGroFac,BoroCnstArt,aXtraGrid,
                             vFuncBool,CubicBool)
    solver.prepareToSolve()                                      
    solution = solver.solve()
    return solution

###############################################################################
    
if __name__ == '__main__':
    import ConsumerParameters as Params
    import matplotlib.pyplot as plt
    from HARKutilities import plotFuncs
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)
    
    do_simulation = True
    
    # Make and solve a preference shock consumer
    PrefShockExample = PrefShockConsumerType(**Params.init_preference_shocks)
    PrefShockExample.cycles = 0 # Infinite horizon    
    
    t_start = clock()
    PrefShockExample.solve()
    t_end = clock()
    print('Solving a preference shock consumer took ' + str(t_end-t_start) + ' seconds.')
    
    # Plot the consumption function at each discrete shock
    m = np.linspace(PrefShockExample.solution[0].mNrmMin,5,200)
    print('Consumption functions at each discrete shock:')
    for j in range(PrefShockExample.PrefShkDstn[0][1].size):
        PrefShk = PrefShockExample.PrefShkDstn[0][1][j]
        c = PrefShockExample.solution[0].cFunc(m,PrefShk*np.ones_like(m))
        plt.plot(m,c)
    plt.show()
    
    print('Consumption function (and MPC) when shock=1:')
    c = PrefShockExample.solution[0].cFunc(m,np.ones_like(m))
    k = PrefShockExample.solution[0].cFunc.derivativeX(m,np.ones_like(m))
    plt.plot(m,c)
    plt.plot(m,k)
    plt.show()
    
    if PrefShockExample.vFuncBool:
            print('Value function (unconditional on shock):')
            plotFuncs(PrefShockExample.solution[0].vFunc,PrefShockExample.solution[0].mNrmMin+0.5,5)
    
    # Test the simulator for the pref shock class
    if do_simulation:
        PrefShockExample.sim_periods = 120
        PrefShockExample.makeIncShkHist()
        PrefShockExample.makePrefShkHist()
        PrefShockExample.initializeSim()
        PrefShockExample.simConsHistory()
