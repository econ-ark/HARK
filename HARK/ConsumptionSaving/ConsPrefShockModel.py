'''
Extensions to ConsIndShockModel concerning models with preference shocks.
It currently only two models:

1) An extension of ConsIndShock, but with an iid lognormal multiplicative shock each period.
2) A combination of (1) and ConsKinkedR, demonstrating how to construct a new model
   by inheriting from multiple classes.
'''
from __future__ import division, print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
import numpy as np
from HARK import makeOnePeriodOOSolver
from HARK.distribution import MeanOneLogNormal
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType, ConsumerSolution, ConsIndShockSolver, \
                                   ValueFunc, MargValueFunc, KinkedRconsumerType, ConsKinkedRsolver, \
                                   init_idiosyncratic_shocks, init_kinked_R
from HARK.interpolation import LinearInterpOnInterp1D, LinearInterp, CubicInterp, LowerEnvelope


# Make a dictionary to specify a preference shock consumer
init_preference_shocks = dict(init_idiosyncratic_shocks,
                              **{
    'PrefShkCount' : 12,    # Number of points in discrete approximation to preference shock dist
    'PrefShk_tail_N' : 4,   # Number of "tail points" on each end of pref shock dist
    'PrefShkStd' : [0.30],  # Standard deviation of utility shocks
    'aXtraCount' : 48,
    'CubicBool' : False     # pref shocks currently only compatible with linear cFunc
})
    
# Make a dictionary to specify a "kinky preference" consumer
init_kinky_pref = dict(init_kinked_R,
                              **{
    'PrefShkCount' : 12,    # Number of points in discrete approximation to preference shock dist
    'PrefShk_tail_N' : 4,   # Number of "tail points" on each end of pref shock dist
    'PrefShkStd' : [0.30],  # Standard deviation of utility shocks
    'aXtraCount' : 48,
    'CubicBool' : False     # pref shocks currently only compatible with linear cFunc
})
init_kinky_pref['BoroCnstArt'] = None

__all__ = ['PrefShockConsumerType', 'KinkyPrefConsumerType', 'ConsPrefShockSolver', 'ConsKinkyPrefSolver']

class PrefShockConsumerType(IndShockConsumerType):
    '''
    A class for representing consumers who experience multiplicative shocks to
    utility each period, specified as iid lognormal.
    '''
    shock_vars_ = IndShockConsumerType.shock_vars_ + ['PrefShkNow']

    def __init__(self,
                 cycles=1,
                 **kwds):
        '''
        Instantiate a new ConsumerType with given data, and construct objects
        to be used during solution (income distribution, assets grid, etc).
        See ConsumerParameters.init_pref_shock for a dictionary of
        the keywords that should be passed to the constructor.

        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.

        Returns
        -------
        None
        '''
        params = init_preference_shocks.copy()
        params.update(kwds)

        IndShockConsumerType.__init__(self,
                                      cycles=cycles,
                                      **params)
        self.solveOnePeriod = makeOnePeriodOOSolver(ConsPrefShockSolver)
    def preSolve(self):
        self.updateSolutionTerminal()

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
        IndShockConsumerType.update(self)  # Update assets grid, income process, terminal solution
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
        PrefShkDstn = [] # discrete distributions of preference shocks
        for t in range(len(self.PrefShkStd)):
            PrefShkStd = self.PrefShkStd[t]
            PrefShkDstn.append(
                MeanOneLogNormal(
                    sigma=PrefShkStd
                ).approx(N=self.PrefShkCount,
                         tail_N=self.PrefShk_tail_N))

        # Store the preference shocks in self (time-varying) and restore time flow
        self.PrefShkDstn = PrefShkDstn
        self.addToTimeVary('PrefShkDstn')

    def getShocks(self):
        '''
        Gets permanent and transitory income shocks for this period as well as preference shocks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        IndShockConsumerType.getShocks(self) # Get permanent and transitory income shocks
        PrefShkNow = np.zeros(self.AgentCount) # Initialize shock array
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                PrefShkNow[these] = self.RNG.permutation(
                    MeanOneLogNormal(
                        sigma=self.PrefShkStd[t]
                    ).approx(N).X)
        self.PrefShkNow = PrefShkNow

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
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these] = self.solution[t].cFunc(self.mNrmNow[these],self.PrefShkNow[these])
        self.cNrmNow = cNrmNow
        return None


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

    
class KinkyPrefConsumerType(PrefShockConsumerType,KinkedRconsumerType):
    '''
    A class for representing consumers who experience multiplicative shocks to
    utility each period, specified as iid lognormal and different interest rates
    on borrowing vs saving.
    '''
    def __init__(self,cycles=1,**kwds):
        '''
        Instantiate a new ConsumerType with given data, and construct objects
        to be used during solution (income distribution, assets grid, etc).
        See init_kinky_pref for a dictionary of the keywords
        that should be passed to the constructor.

        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.

        Returns
        -------
        None
        '''
        params = init_kinky_pref.copy()
        params.update(kwds)
        kwds = params
        IndShockConsumerType.__init__(self,**kwds)
        self.solveOnePeriod = makeOnePeriodOOSolver(ConsKinkyPrefSolver)
        self.addToTimeInv('Rboro','Rsave')
        self.delFromTimeInv('Rfree')
        
    def preSolve(self):
        self.updateSolutionTerminal()

    def getRfree(self): # Specify which getRfree to use
        return KinkedRconsumerType.getRfree(self)

###############################################################################

class ConsPrefShockSolver(ConsIndShockSolver):
    '''
    A class for solving the one period consumption-saving problem with risky
    income (permanent and transitory shocks) and multiplicative shocks to utility
    each period.
    '''
    def __init__(self,solution_next,IncomeDstn,PrefShkDstn,LivPrb,DiscFac,CRRA,
                      Rfree,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
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
        Rfree : float
            Risk free interest factor on end-of-period assets.
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
        ConsIndShockSolver.__init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,
                      Rfree,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.PrefShkPrbs = PrefShkDstn.pmf
        self.PrefShkVals = PrefShkDstn.X

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
                       LivPrb,DiscFac,CRRA,Rfree,PermGroFac,BoroCnstArt,
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
    Rfree : float
        Risk free interest factor on end-of-period assets.
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
    solver = (solution_next,IncomeDstn,PrefShkDstn,LivPrb,
                             DiscFac,CRRA,Rfree,PermGroFac,BoroCnstArt,aXtraGrid,
                             vFuncBool,CubicBool)
    solver.prepareToSolve()
    solution = solver.solve()
    return solution

###############################################################################

class ConsKinkyPrefSolver(ConsPrefShockSolver,ConsKinkedRsolver):
    '''
    A class for solving the one period consumption-saving problem with risky
    income (permanent and transitory shocks), multiplicative shocks to utility
    each period, and a different interest rate on saving vs borrowing.
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
        self.PrefShkPrbs = PrefShkDstn.pmf
        self.PrefShkVals = PrefShkDstn.X
