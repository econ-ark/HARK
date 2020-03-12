'''
Classes to solve canonical consumption-savings models with idiosyncratic shocks
to income.  All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks are fully transitory or fully permanent.

It currently solves three types of models:
   1) A very basic "perfect foresight" consumption-savings model with no uncertainty.
   2) A consumption-savings model with risk over transitory and permanent income shocks.
   3) The model described in (2), with an interest rate for debt that differs
      from the interest rate for savings.

See NARK for information on variable naming conventions.
See HARK documentation for mathematical descriptions of the models being solved.
'''
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
from copy import copy, deepcopy
import numpy as np
from scipy.optimize import newton
from HARK import AgentType, Solution, NullFunc, HARKobject
from HARK.utilities import warnings  # Because of "patch" to warnings modules
from HARK.interpolation import CubicInterp, LowerEnvelope, LinearInterp
from HARK.simulation import drawDiscrete, drawLognormal, drawUniform
from HARK.utilities import approxMeanOneLognormal, addDiscreteOutcomeConstantMean,\
                           combineIndepDstns, makeGridExpMult, CRRAutility, CRRAutilityP, \
                           CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv, \
                           CRRAutilityP_invP


__all__ = ['ConsumerSolution', 'ValueFunc', 'MargValueFunc', 'MargMargValueFunc',
'ConsPerfForesightSolver', 'ConsIndShockSetup', 'ConsIndShockSolverBasic',
'ConsIndShockSolver', 'ConsKinkedRsolver', 'PerfForesightConsumerType',
'IndShockConsumerType', 'KinkedRconsumerType',
           'init_perfect_foresight','init_idiosyncratic_shocks','init_kinked_R',
           'init_lifecycle','init_cyclical']

utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

# =====================================================================
# === Classes that help solve consumption-saving models ===
# =====================================================================

class ConsumerSolution(Solution):
    '''
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function and marginal
    value function.

    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.
    '''
    distance_criteria = ['vPfunc']

    def __init__(self, cFunc=None, vFunc=None,
                       vPfunc=None, vPPfunc=None,
                       mNrmMin=None, hNrm=None, MPCmin=None, MPCmax=None):
        '''
        The constructor for a new ConsumerSolution object.

        Parameters
        ----------
        cFunc : function
            The consumption function for this period, defined over market
            resources: c = cFunc(m).
        vFunc : function
            The beginning-of-period value function for this period, defined over
            market resources: v = vFunc(m).
        vPfunc : function
            The beginning-of-period marginal value function for this period,
            defined over market resources: vP = vPfunc(m).
        vPPfunc : function
            The beginning-of-period marginal marginal value function for this
            period, defined over market resources: vPP = vPPfunc(m).
        mNrmMin : float
            The minimum allowable market resources for this period; the consump-
            tion function (etc) are undefined for m < mNrmMin.
        hNrm : float
            Human wealth after receiving income this period: PDV of all future
            income, ignoring mortality.
        MPCmin : float
            Infimum of the marginal propensity to consume this period.
            MPC --> MPCmin as m --> infinity.
        MPCmax : float
            Supremum of the marginal propensity to consume this period.
            MPC --> MPCmax as m --> mNrmMin.

        Returns
        -------
        None
        '''
        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        # vPFunc = NullFunc() if vPfunc is None else vPfunc
        self.vPPfunc = vPPfunc if vPPfunc is not None else NullFunc()
        self.mNrmMin      = mNrmMin
        self.hNrm         = hNrm
        self.MPCmin       = MPCmin
        self.MPCmax       = MPCmax

    def appendSolution(self,new_solution):
        '''
        Appends one solution to another to create a ConsumerSolution whose
        attributes are lists.  Used in ConsMarkovModel, where we append solutions
        *conditional* on a particular value of a Markov state to each other in
        order to get the entire solution.

        Parameters
        ----------
        new_solution : ConsumerSolution
            The solution to a consumption-saving problem; each attribute is a
            list representing state-conditional values or functions.

        Returns
        -------
        None
        '''
        if type(self.cFunc)!=list:
            # Then we assume that self is an empty initialized solution instance.
            # Begin by checking this is so.
            assert NullFunc().distance(self.cFunc) == 0, 'appendSolution called incorrectly!'

            # We will need the attributes of the solution instance to be lists.  Do that here.
            self.cFunc       = [new_solution.cFunc]
            self.vFunc       = [new_solution.vFunc]
            self.vPfunc      = [new_solution.vPfunc]
            self.vPPfunc     = [new_solution.vPPfunc]
            self.mNrmMin     = [new_solution.mNrmMin]
        else:
            self.cFunc.append(new_solution.cFunc)
            self.vFunc.append(new_solution.vFunc)
            self.vPfunc.append(new_solution.vPfunc)
            self.vPPfunc.append(new_solution.vPPfunc)
            self.mNrmMin.append(new_solution.mNrmMin)


class ValueFunc(HARKobject):
    '''
    A class for representing a value function.  The underlying interpolation is
    in the space of (m,u_inv(v)); this class "re-curves" to the value function.
    '''
    distance_criteria = ['func','CRRA']

    def __init__(self,vFuncNvrs,CRRA):
        '''
        Constructor for a new value function object.

        Parameters
        ----------
        vFuncNvrs : function
            A real function representing the value function composed with the
            inverse utility function, defined on market resources: u_inv(vFunc(m))
        CRRA : float
            Coefficient of relative risk aversion.

        Returns
        -------
        None
        '''
        self.func = deepcopy(vFuncNvrs)
        self.CRRA = CRRA

    def __call__(self,m):
        '''
        Evaluate the value function at given levels of market resources m.

        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose value is to
            be found.

        Returns
        -------
        v : float or np.array
            Lifetime value of beginning this period with market resources m; has
            same size as input m.
        '''
        return utility(self.func(m),gam=self.CRRA)


class MargValueFunc(HARKobject):
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
    distance_criteria = ['cFunc','CRRA']

    def __init__(self,cFunc,CRRA):
        '''
        Constructor for a new marginal value function object.

        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources: uP_inv(vPfunc(m)).  Called cFunc because when standard
            envelope condition applies, uP_inv(vPfunc(m)) = cFunc(m).
        CRRA : float
            Coefficient of relative risk aversion.

        Returns
        -------
        None
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self,m):
        '''
        Evaluate the marginal value function at given levels of market resources m.

        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            value is to be found.

        Returns
        -------
        vP : float or np.array
            Marginal lifetime value of beginning this period with market
            resources m; has same size as input m.
        '''
        return utilityP(self.cFunc(m),gam=self.CRRA)

    def derivative(self,m):
        '''
        Evaluate the derivative of the marginal value function at given levels
        of market resources m; this is the marginal marginal value function.

        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            marginal value is to be found.

        Returns
        -------
        vPP : float or np.array
            Marginal marginal lifetime value of beginning this period with market
            resources m; has same size as input m.
        '''
        c, MPC = self.cFunc.eval_with_derivative(m)
        return MPC*utilityPP(c,gam=self.CRRA)


class MargMargValueFunc(HARKobject):
    '''
    A class for representing a marginal marginal value function in models where
    the standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
    distance_criteria = ['cFunc','CRRA']

    def __init__(self,cFunc,CRRA):
        '''
        Constructor for a new marginal marginal value function object.

        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources: uP_inv(vPfunc(m)).  Called cFunc because when standard
            envelope condition applies, uP_inv(vPfunc(m)) = cFunc(m).
        CRRA : float
            Coefficient of relative risk aversion.

        Returns
        -------
        None
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self,m):
        '''
        Evaluate the marginal marginal value function at given levels of market
        resources m.

        Parameters
        ----------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            marginal value is to be found.

        Returns
        -------
        vPP : float or np.array
            Marginal marginal lifetime value of beginning this period with market
            resources m; has same size as input m.
        '''
        c, MPC = self.cFunc.eval_with_derivative(m)
        return MPC*utilityPP(c,gam=self.CRRA)




# =====================================================================
# === Classes and functions that solve consumption-saving models ===
# =====================================================================

class ConsPerfForesightSolver(object):
    '''
    A class for solving a one period perfect foresight consumption-saving problem.
    An instance of this class is created by the function solvePerfForesight in each period.
    '''
    def __init__(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac,BoroCnstArt,MaxKinks):
        '''
        Constructor for a new ConsPerfForesightSolver.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one-period problem.
        DiscFac : float
            Intertemporal discount factor for future utility.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the next period.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt : float or None
            Artificial borrowing constraint, as a multiple of permanent income.
            Can be None, indicating no artificial constraint.
        MaxKinks : int
            Maximum number of kink points to allow in the consumption function;
            additional points will be thrown out.  Only relevant in infinite
            horizon model with artificial borrowing constraint.

        Returns:
        ----------
        None
        '''
        # We ask that HARK users define single-letter variables they use in a dictionary
        # attribute called notation. Do that first.
        self.notation = {'a': 'assets after all actions',
                         'm': 'market resources at decision time',
                         'c': 'consumption'}
        self.assignParameters(solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac,BoroCnstArt,MaxKinks)

    def assignParameters(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac,BoroCnstArt,MaxKinks):
        '''
        Saves necessary parameters as attributes of self for use by other methods.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        DiscFac : float
            Intertemporal discount factor for future utility.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt : float or None
            Artificial borrowing constraint, as a multiple of permanent income.
            Can be None, indicating no artificial constraint.
        MaxKinks : int
            Maximum number of kink points to allow in the consumption function;
            additional points will be thrown out.

        Returns
        -------
        None
        '''
        self.solution_next  = solution_next
        self.DiscFac        = DiscFac
        self.LivPrb         = LivPrb
        self.CRRA           = CRRA
        self.Rfree          = Rfree
        self.PermGroFac     = PermGroFac
        self.MaxKinks       = MaxKinks
        self.BoroCnstArt    = BoroCnstArt

    def defUtilityFuncs(self):
        '''
        Defines CRRA utility function for this period (and its derivatives),
        saving them as attributes of self for other methods to use.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.u   = lambda c : utility(c,gam=self.CRRA)  # utility function
        self.uP  = lambda c : utilityP(c,gam=self.CRRA) # marginal utility function
        self.uPP = lambda c : utilityPP(c,gam=self.CRRA)# marginal marginal utility function

    def defValueFuncs(self):
        '''
        Defines the value and marginal value functions for this period.
        Uses the fact that for a perfect foresight CRRA utility problem,
        if the MPC in period t is :math:`\kappa_{t}`, and relative risk 
        aversion :math:`\rho`, then the inverse value vFuncNvrs has a 
        constant slope of :math:`\kappa_{t}^{-\rho/(1-\rho)}` and 
        vFuncNvrs has value of zero at the lower bound of market resources 
        mNrmMin.  See PerfForesightConsumerType.ipynb documentation notebook
        for a brief explanation and the links below for a fuller treatment.
            
        https://econ.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA/#vFuncAnalytical
        https://econ.jhu.edu/people/ccarroll/SolvingMicroDSOPs/#vFuncPF

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # See the PerfForesightConsumerType.ipynb documentation notebook for the derivations
        vFuncNvrsSlope = self.MPCmin**(-self.CRRA/(1.0-self.CRRA)) 
        vFuncNvrs      = LinearInterp(np.array([self.mNrmMinNow, self.mNrmMinNow+1.0]),np.array([0.0, vFuncNvrsSlope]))
        self.vFunc     = ValueFunc(vFuncNvrs,self.CRRA)
        self.vPfunc    = MargValueFunc(self.cFunc,self.CRRA)

    def makePFcFunc(self):
        '''
        Makes the (linear) consumption function for this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Use a local value of BoroCnstArt to prevent comparing None and float below.
        if self.BoroCnstArt is None:
            BoroCnstArt = -np.inf
        else:
            BoroCnstArt = self.BoroCnstArt
        
        # Calculate human wealth this period
        self.hNrmNow = (self.PermGroFac/self.Rfree)*(self.solution_next.hNrm + 1.0)
        
        # Calculate the lower bound of the marginal propensity to consume
        PatFac       = ((self.Rfree*self.DiscFacEff)**(1.0/self.CRRA))/self.Rfree
        self.MPCmin  = 1.0/(1.0 + PatFac/self.solution_next.MPCmin)
        
        # Extract the discrete kink points in next period's consumption function;
        # don't take the last one, as it only defines the extrapolation and is not a kink.
        mNrmNext = self.solution_next.cFunc.x_list[:-1]
        cNrmNext = self.solution_next.cFunc.y_list[:-1]
        
        # Calculate the end-of-period asset values that would reach those kink points
        # next period, then invert the first order condition to get consumption. Then
        # find the endogenous gridpoint (kink point) today that corresponds to each kink
        aNrmNow = (self.PermGroFac/self.Rfree)*(mNrmNext-1.0)
        cNrmNow = (self.DiscFacEff*self.Rfree)**(-1./self.CRRA)*(self.PermGroFac*cNrmNext)
        mNrmNow = aNrmNow + cNrmNow
        
        # Add an additional point to the list of gridpoints for the extrapolation,
        # using the new value of the lower bound of the MPC.
        mNrmNow = np.append(mNrmNow, mNrmNow[-1] + 1.0)
        cNrmNow = np.append(cNrmNow, cNrmNow[-1] + self.MPCmin)
        
        # If the artificial borrowing constraint binds, combine the constrained and
        # unconstrained consumption functions.
        if BoroCnstArt > mNrmNow[0]:
            # Find the highest index where constraint binds
            cNrmCnst = mNrmNow - BoroCnstArt
            CnstBinds = cNrmCnst < cNrmNow
            idx = np.where(CnstBinds)[0][-1]
            
            if idx < (mNrmNow.size-1):
                # If it is not the *very last* index, find the the critical level
                # of mNrm where the artificial borrowing contraint begins to bind.
                d0 = cNrmNow[idx] - cNrmCnst[idx]
                d1 = cNrmCnst[idx+1] - cNrmNow[idx+1]
                m0 = mNrmNow[idx]
                m1 = mNrmNow[idx+1]
                alpha = d0/(d0 + d1)
                mCrit = m0 + alpha*(m1 - m0)
                
                # Adjust the grids of mNrm and cNrm to account for the borrowing constraint.
                cCrit = mCrit - BoroCnstArt
                mNrmNow = np.concatenate(([BoroCnstArt, mCrit], mNrmNow[(idx+1):]))
                cNrmNow = np.concatenate(([0., cCrit], cNrmNow[(idx+1):]))
                
            else:
                # If it *is* the very last index, then there are only three points
                # that characterize the consumption function: the artificial borrowing
                # constraint, the constraint kink, and the extrapolation point.
                mXtra = cNrmNow[-1] - cNrmCnst[-1]/(1.0 - self.MPCmin)
                mCrit = mNrmNow[-1] + mXtra
                cCrit = mCrit - BoroCnstArt
                mNrmNow = np.array([BoroCnstArt, mCrit, mCrit + 1.0])
                cNrmNow = np.array([0., cCrit, cCrit + self.MPCmin])
                
        # If the mNrm and cNrm grids have become too large, throw out the last
        # kink point, being sure to adjust the extrapolation.
        if mNrmNow.size > self.MaxKinks:
            mNrmNow = np.concatenate((mNrmNow[:-2], [mNrmNow[-3] + 1.0]))
            cNrmNow = np.concatenate((cNrmNow[:-2], [cNrmNow[-3] + self.MPCmin]))
        
        # Construct the consumption function as a linear interpolation.
        self.cFunc = LinearInterp(mNrmNow, cNrmNow)
        
        # Calculate the upper bound of the MPC as the slope of the bottom segment.
        self.MPCmax = (cNrmNow[1] - cNrmNow[0]) / (mNrmNow[1] - mNrmNow[0])
        
        # Add two attributes to enable calculation of steady state market resources.
        self.ExIncNext = 1.0 # Perfect foresight income of 1
        self.mNrmMinNow = mNrmNow[0] # Relabeling for compatibility with addSSmNrm
        
        
    def addSSmNrm(self,solution):
        '''
        Finds steady state (normalized) market resources and adds it to the
        solution.  This is the level of market resources such that the expectation
        of market resources in the next period is unchanged.  This value doesn't
        necessarily exist.

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was passed, but now with the attribute mNrmSS.
        '''
        # Make a linear function of all combinations of c and m that yield mNext = mNow
        mZeroChangeFunc = lambda m : (1.0-self.PermGroFac/self.Rfree)*m + (self.PermGroFac/self.Rfree)*self.ExIncNext

        # Find the steady state level of market resources
        searchSSfunc = lambda m : solution.cFunc(m) - mZeroChangeFunc(m) # A zero of this is SS market resources
        m_init_guess = self.mNrmMinNow + self.ExIncNext # Minimum market resources plus next income is okay starting guess
        try:
            mNrmSS = newton(searchSSfunc,m_init_guess)
        except:
            mNrmSS = None

        # Add mNrmSS to the solution and return it
        solution.mNrmSS = mNrmSS
        return solution

    def solve(self):
        '''
        Solves the one period perfect foresight consumption-saving problem.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period's problem.
        '''
        self.defUtilityFuncs()
        self.DiscFacEff = self.DiscFac*self.LivPrb
        self.makePFcFunc()
        self.defValueFuncs()
        solution = ConsumerSolution(cFunc=self.cFunc, vFunc=self.vFunc, vPfunc=self.vPfunc,
                                    mNrmMin=self.mNrmMinNow, hNrm=self.hNrmNow,
                                    MPCmin=self.MPCmin, MPCmax=self.MPCmax)
        solution = self.addSSmNrm(solution)
        return solution


def solvePerfForesight(solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac,BoroCnstArt,MaxKinks):
    '''
    Solves a single period consumption-saving problem for a consumer with perfect foresight.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    DiscFac : float
        Intertemporal discount factor for future utility.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt : float or None
        Artificial borrowing constraint, as a multiple of permanent income.
        Can be None, indicating no artificial constraint.
    MaxKinks : int
        Maximum number of kink points to allow in the consumption function;
        additional points will be thrown out.  Only relevant in infinite horizon
        models with artificial borrowing constraint.

    Returns
    -------
    solution_now : ConsumerSolution
        The solution to this period's problem.
    '''
    
    solver = ConsPerfForesightSolver(solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac,BoroCnstArt,MaxKinks)
    solution_now = solver.solve()
    return solution_now


###############################################################################
###############################################################################
class ConsIndShockSetup(ConsPerfForesightSolver):
    '''
    A superclass for solvers of one period consumption-saving problems with
    constant relative risk aversion utility and permanent and transitory shocks
    to income.  Has methods to set up but not solve the one period problem.
    '''
    def __init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                      PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver-setup for problems with income subject to
        permanent and transitory shocks.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : float
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
        self.assignParameters(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.defUtilityFuncs()

    def assignParameters(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
        '''
        Assigns period parameters as attributes of self for use by other methods

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : float
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
        none
        '''
        ConsPerfForesightSolver.assignParameters(self,solution_next,DiscFac,LivPrb,
                                                CRRA,Rfree,PermGroFac,BoroCnstArt,None)
        self.aXtraGrid      = aXtraGrid
        self.IncomeDstn     = IncomeDstn
        self.vFuncBool      = vFuncBool
        self.CubicBool      = CubicBool


    def defUtilityFuncs(self):
        '''
        Defines CRRA utility function for this period (and its derivatives,
        and their inverses), saving them as attributes of self for other methods
        to use.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        ConsPerfForesightSolver.defUtilityFuncs(self)
        self.uPinv     = lambda u : utilityP_inv(u,gam=self.CRRA)
        self.uPinvP    = lambda u : utilityP_invP(u,gam=self.CRRA)
        self.uinvP     = lambda u : utility_invP(u,gam=self.CRRA)
        if self.vFuncBool:
            self.uinv  = lambda u : utility_inv(u,gam=self.CRRA)


    def setAndUpdateValues(self,solution_next,IncomeDstn,LivPrb,DiscFac):
        '''
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, next period's marginal value function
        (etc), the probability of getting the worst income shock next period,
        the patience factor, human wealth, and the bounding MPCs.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        '''
        self.DiscFacEff       = DiscFac*LivPrb # "effective" discount factor
        self.ShkPrbsNext      = IncomeDstn[0]
        self.PermShkValsNext  = IncomeDstn[1]
        self.TranShkValsNext  = IncomeDstn[2]
        self.PermShkMinNext   = np.min(self.PermShkValsNext)
        self.TranShkMinNext   = np.min(self.TranShkValsNext)
        self.vPfuncNext       = solution_next.vPfunc
        self.WorstIncPrb      = np.sum(self.ShkPrbsNext[
                                (self.PermShkValsNext*self.TranShkValsNext)==
                                (self.PermShkMinNext*self.TranShkMinNext)])

        if self.CubicBool:
            self.vPPfuncNext  = solution_next.vPPfunc

        if self.vFuncBool:
            self.vFuncNext    = solution_next.vFunc

        # Update the bounding MPCs and PDV of human wealth:
        self.PatFac       = ((self.Rfree*self.DiscFacEff)**(1.0/self.CRRA))/self.Rfree
        self.MPCminNow    = 1.0/(1.0 + self.PatFac/solution_next.MPCmin)
        self.ExIncNext    = np.dot(self.ShkPrbsNext,self.TranShkValsNext*self.PermShkValsNext)
        self.hNrmNow      = self.PermGroFac/self.Rfree*(self.ExIncNext + solution_next.hNrm)
        self.MPCmaxNow    = 1.0/(1.0 + (self.WorstIncPrb**(1.0/self.CRRA))*
                                        self.PatFac/solution_next.MPCmax)

        self.cFuncLimitIntercept = self.MPCminNow*self.hNrmNow
        self.cFuncLimitSlope = self.MPCminNow


    def defBoroCnst(self,BoroCnstArt):
        '''
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Uses the artificial and natural borrowing constraints.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.

        Returns
        -------
        none
        '''
        # Calculate the minimum allowable value of money resources in this period
        self.BoroCnstNat = (self.solution_next.mNrmMin - self.TranShkMinNext)*\
                           (self.PermGroFac*self.PermShkMinNext)/self.Rfree

        # Note: need to be sure to handle BoroCnstArt==None appropriately.
        # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
        # However in Py3, this raises a TypeError. Thus here we need to directly
        # address the situation in which BoroCnstArt == None:
        if BoroCnstArt is None:
            self.mNrmMinNow = self.BoroCnstNat
        else:
            self.mNrmMinNow = np.max([self.BoroCnstNat,BoroCnstArt])
        if self.BoroCnstNat < self.mNrmMinNow:
            self.MPCmaxEff = 1.0 # If actually constrained, MPC near limit is 1
        else:
            self.MPCmaxEff = self.MPCmaxNow

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(np.array([self.mNrmMinNow, self.mNrmMinNow+1]),
                                         np.array([0.0, 1.0]))


    def prepareToSolve(self):
        '''
        Perform preparatory work before calculating the unconstrained consumption
        function.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.setAndUpdateValues(self.solution_next,self.IncomeDstn,self.LivPrb,self.DiscFac)
        self.defBoroCnst(self.BoroCnstArt)


####################################################################################################
####################################################################################################

class ConsIndShockSolverBasic(ConsIndShockSetup):
    '''
    This class solves a single period of a standard consumption-saving problem,
    using linear interpolation and without the ability to calculate the value
    function.  ConsIndShockSolver inherits from this class and adds the ability
    to perform cubic interpolation and to calculate the value function.

    Note that this class does not have its own initializing method.  It initial-
    izes the same problem in the same way as ConsIndShockSetup, from which it
    inherits.
    '''
    def prepareToCalcEndOfPrdvP(self):
        '''
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.

        Parameters
        ----------
        none

        Returns
        -------
        aNrmNow : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.
        '''

        # We define aNrmNow all the way from BoroCnstNat up to max(self.aXtraGrid)
        # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
        # function as the lower envelope of the (by the artificial borrowing con-
        # straint) uconstrained consumption function, and the artificially con-
        # strained consumption function.
        aNrmNow     = np.asarray(self.aXtraGrid) + self.BoroCnstNat
        ShkCount    = self.TranShkValsNext.size
        aNrm_temp   = np.tile(aNrmNow,(ShkCount,1))

        # Tile arrays of the income shocks and put them into useful shapes
        aNrmCount         = aNrmNow.shape[0]
        PermShkVals_temp  = (np.tile(self.PermShkValsNext,(aNrmCount,1))).transpose()
        TranShkVals_temp  = (np.tile(self.TranShkValsNext,(aNrmCount,1))).transpose()
        ShkPrbs_temp      = (np.tile(self.ShkPrbsNext,(aNrmCount,1))).transpose()

        # Get cash on hand next period
        mNrmNext          = self.Rfree/(self.PermGroFac*PermShkVals_temp)*aNrm_temp + TranShkVals_temp # CDC 20191205: This should be divided by LivPrb[0] for Blanchard insurance

        # Store and report the results
        self.PermShkVals_temp  = PermShkVals_temp
        self.ShkPrbs_temp      = ShkPrbs_temp
        self.mNrmNext          = mNrmNext
        self.aNrmNow           = aNrmNow
        return aNrmNow


    def calcEndOfPrdvP(self):
        '''
        Calculate end-of-period marginal value of assets at each point in aNrmNow.
        Does so by taking a weighted sum of next period marginal values across
        income shocks (in a preconstructed grid self.mNrmNext).

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets
        '''

        EndOfPrdvP  = self.DiscFacEff*self.Rfree*self.PermGroFac**(-self.CRRA)*np.sum(
                      self.PermShkVals_temp**(-self.CRRA)*
                      self.vPfuncNext(self.mNrmNext)*self.ShkPrbs_temp,axis=0)
        return EndOfPrdvP


    def getPointsForInterpolation(self,EndOfPrdvP,aNrmNow):
        '''
        Finds interpolation points (c,m) for the consumption function.

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
        cNrmNow = self.uPinv(EndOfPrdvP)
        mNrmNow = cNrmNow + aNrmNow

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNrmNow,0,0.,axis=-1)
        m_for_interpolation = np.insert(mNrmNow,0,self.BoroCnstNat,axis=-1)

        # Store these for calcvFunc
        self.cNrmNow = cNrmNow
        self.mNrmNow = mNrmNow

        return c_for_interpolation,m_for_interpolation


    def usePointsForInterpolation(self,cNrm,mNrm,interpolator):
        '''
        Constructs a basic solution for this period, including the consumption
        function and marginal value function.

        Parameters
        ----------
        cNrm : np.array
            (Normalized) consumption points for interpolation.
        mNrm : np.array
            (Normalized) corresponding market resource points for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        '''
        # Construct the unconstrained consumption function
        cFuncNowUnc = interpolator(mNrm,cNrm)

        # Combine the constrained and unconstrained functions into the true consumption function
        cFuncNow = LowerEnvelope(cFuncNowUnc,self.cFuncNowCnst)

        # Make the marginal value function and the marginal marginal value function
        vPfuncNow = MargValueFunc(cFuncNow,self.CRRA)

        # Pack up the solution and return it
        solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow)
        return solution_now


    def makeBasicSolution(self,EndOfPrdvP,aNrm,interpolator):
        '''
        Given end of period assets and end of period marginal value, construct
        the basic solution for this period.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrm : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.

        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        '''
        cNrm,mNrm    = self.getPointsForInterpolation(EndOfPrdvP,aNrm)
        solution_now = self.usePointsForInterpolation(cNrm,mNrm,interpolator)
        return solution_now

    def addMPCandHumanWealth(self,solution):
        '''
        Take a solution and add human wealth and the bounding MPCs to it.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this period's consumption-saving problem.

        Returns:
        ----------
        solution : ConsumerSolution
            The solution to this period's consumption-saving problem, but now
            with human wealth and the bounding MPCs.
        '''
        solution.hNrm   = self.hNrmNow
        solution.MPCmin = self.MPCminNow
        solution.MPCmax = self.MPCmaxEff
        return solution

    def makeLinearcFunc(self,mNrm,cNrm):
        '''
        Makes a linear interpolation to represent the (unconstrained) consumption function.

        Parameters
        ----------
        mNrm : np.array
            Corresponding market resource points for interpolation.
        cNrm : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFuncUnc : LinearInterp
            The unconstrained consumption function for this period.
        '''
        cFuncUnc = LinearInterp(mNrm, cNrm, self.cFuncLimitIntercept, self.cFuncLimitSlope)
        return cFuncUnc

    def solve(self):
        '''
        Solves a one period consumption saving problem with risky income.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem.
        '''
        aNrm       = self.prepareToCalcEndOfPrdvP()
        EndOfPrdvP = self.calcEndOfPrdvP()
        solution   = self.makeBasicSolution(EndOfPrdvP,aNrm,self.makeLinearcFunc)
        solution   = self.addMPCandHumanWealth(solution)
        return solution


###############################################################################
###############################################################################

class ConsIndShockSolver(ConsIndShockSolverBasic):
    '''
    This class solves a single period of a standard consumption-saving problem.
    It inherits from ConsIndShockSolverBasic, adding the ability to perform cubic
    interpolation and to calculate the value function.
    '''

    def makeCubiccFunc(self,mNrm,cNrm):
        '''
        Makes a cubic spline interpolation of the unconstrained consumption
        function for this period.

        Parameters
        ----------
        mNrm : np.array
            Corresponding market resource points for interpolation.
        cNrm : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFuncUnc : CubicInterp
            The unconstrained consumption function for this period.
        '''
        EndOfPrdvPP = self.DiscFacEff*self.Rfree*self.Rfree*self.PermGroFac**(-self.CRRA-1.0)* \
                      np.sum(self.PermShkVals_temp**(-self.CRRA-1.0)*
                             self.vPPfuncNext(self.mNrmNext)*self.ShkPrbs_temp,axis=0)
        dcda        = EndOfPrdvPP/self.uPP(np.array(cNrm[1:]))
        MPC         = dcda/(dcda+1.)
        MPC         = np.insert(MPC,0,self.MPCmaxNow)

        cFuncNowUnc = CubicInterp(mNrm,cNrm,MPC,self.MPCminNow*self.hNrmNow,self.MPCminNow)
        return cFuncNowUnc


    def makeEndOfPrdvFunc(self,EndOfPrdvP):
        '''
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.aNrmNow.

        Returns
        -------
        none
        '''
        VLvlNext            = (self.PermShkVals_temp**(1.0-self.CRRA)*\
                               self.PermGroFac**(1.0-self.CRRA))*self.vFuncNext(self.mNrmNext)
        EndOfPrdv           = self.DiscFacEff*np.sum(VLvlNext*self.ShkPrbs_temp,axis=0)
        EndOfPrdvNvrs       = self.uinv(EndOfPrdv) # value transformed through inverse utility
        EndOfPrdvNvrsP      = EndOfPrdvP*self.uinvP(EndOfPrdv)
        EndOfPrdvNvrs       = np.insert(EndOfPrdvNvrs,0,0.0)
        EndOfPrdvNvrsP      = np.insert(EndOfPrdvNvrsP,0,EndOfPrdvNvrsP[0]) # This is a very good approximation, vNvrsPP = 0 at the asset minimum
        aNrm_temp           = np.insert(self.aNrmNow,0,self.BoroCnstNat)
        EndOfPrdvNvrsFunc   = CubicInterp(aNrm_temp,EndOfPrdvNvrs,EndOfPrdvNvrsP)
        self.EndOfPrdvFunc  = ValueFunc(EndOfPrdvNvrsFunc,self.CRRA)


    def addvFunc(self,solution,EndOfPrdvP):
        '''
        Creates the value function for this period and adds it to the solution.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, likely including the
            consumption function, marginal value function, etc.
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.aNrmNow.

        Returns
        -------
        solution : ConsumerSolution
            The single period solution passed as an input, but now with the
            value function (defined over market resources m) as an attribute.
        '''
        self.makeEndOfPrdvFunc(EndOfPrdvP)
        solution.vFunc = self.makevFunc(solution)
        return solution


    def makevFunc(self,solution):
        '''
        Creates the value function for this period, defined over market resources m.
        self must have the attribute EndOfPrdvFunc in order to execute.

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
        # Compute expected value and marginal value on a grid of market resources
        mNrm_temp   = self.mNrmMinNow + self.aXtraGrid
        cNrmNow     = solution.cFunc(mNrm_temp)
        aNrmNow     = mNrm_temp - cNrmNow
        vNrmNow     = self.u(cNrmNow) + self.EndOfPrdvFunc(aNrmNow)
        vPnow       = self.uP(cNrmNow)

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


    def addvPPfunc(self,solution):
        '''
        Adds the marginal marginal value function to an existing solution, so
        that the next solver can evaluate vPP and thus use cubic interpolation.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, which must include the
            consumption function.

        Returns
        -------
        solution : ConsumerSolution
            The same solution passed as input, but with the marginal marginal
            value function for this period added as the attribute vPPfunc.
        '''
        vPPfuncNow        = MargMargValueFunc(solution.cFunc,self.CRRA)
        solution.vPPfunc  = vPPfuncNow
        return solution


    def solve(self):
        '''
        Solves the single period consumption-saving problem using the method of
        endogenous gridpoints.  Solution includes a consumption function cFunc
        (using cubic or linear splines), a marginal value function vPfunc, a min-
        imum acceptable level of normalized market resources mNrmMin, normalized
        human wealth hNrm, and bounding MPCs MPCmin and MPCmax.  It might also
        have a value function vFunc and marginal marginal value function vPPfunc.

        Parameters
        ----------
        none

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem.
        '''
        # Make arrays of end-of-period assets and end-of-period marginal value
        aNrm         = self.prepareToCalcEndOfPrdvP()
        EndOfPrdvP   = self.calcEndOfPrdvP()

        # Construct a basic solution for this period
        if self.CubicBool:
            solution   = self.makeBasicSolution(EndOfPrdvP,aNrm,interpolator=self.makeCubiccFunc)
        else:
            solution   = self.makeBasicSolution(EndOfPrdvP,aNrm,interpolator=self.makeLinearcFunc)
        solution       = self.addMPCandHumanWealth(solution) # add a few things
        solution       = self.addSSmNrm(solution) # find steady state m

        # Add the value function if requested, as well as the marginal marginal
        # value function if cubic splines were used (to prepare for next period)
        if self.vFuncBool:
            solution = self.addvFunc(solution,EndOfPrdvP)
        if self.CubicBool:
            solution = self.addvPPfunc(solution)
        return solution


def solveConsIndShock(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                                BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
    '''
    Solves a single period consumption-saving problem with CRRA utility and risky
    income (subject to permanent and transitory shocks).  Can generate a value
    function if requested; consumption function can be linear or cubic splines.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn : [np.array]
        A list containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
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
        Indicator for whether the solver should use cubic or linear interpolation.

    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (using cubic or linear splines), a marginal
        value function vPfunc, a minimum acceptable level of normalized market
        resources mNrmMin, normalized human wealth hNrm, and bounding MPCs MPCmin
        and MPCmax.  It might also have a value function vFunc and marginal mar-
        ginal value function vPPfunc.
    '''
    # Use the basic solver if user doesn't want cubic splines or the value function
    if (not CubicBool) and (not vFuncBool):
        solver = ConsIndShockSolverBasic(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,
                                                  Rfree,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,
                                                  CubicBool)
    else: # Use the "advanced" solver if either is requested
        solver = ConsIndShockSolver(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                             PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
    solver.prepareToSolve()       # Do some preparatory work
    solution_now = solver.solve() # Solve the period
    return solution_now


####################################################################################################
####################################################################################################

class ConsKinkedRsolver(ConsIndShockSolver):
    '''
    A class to solve a single period consumption-saving problem where the interest
    rate on debt differs from the interest rate on savings.  Inherits from
    ConsIndShockSolver, with nearly identical inputs and outputs.  The key diff-
    erence is that Rfree is replaced by Rsave (a>0) and Rboro (a<0).  The solver
    can handle Rboro == Rsave, which makes it identical to ConsIndShocksolver, but
    it terminates immediately if Rboro < Rsave, as this has a different solution.
    '''
    def __init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,
                      Rboro,Rsave,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver for problems with risky income and a different
        interest rate on borrowing and saving.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
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
        PermGroFac : float
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
        assert Rboro>=Rsave, 'Interest factor on debt less than interest factor on savings!'

        # Initialize the solver.  Most of the steps are exactly the same as in
        # the non-kinked-R basic case, so start with that.
        ConsIndShockSolver.__init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rboro,
                                    PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)

        # Assign the interest rates as class attributes, to use them later.
        self.Rboro   = Rboro
        self.Rsave   = Rsave
                  
    def makeCubiccFunc(self,mNrm,cNrm):
        '''
        Makes a cubic spline interpolation that contains the kink of the unconstrained 
        consumption function for this period.

        Parameters
        ----------
        mNrm : np.array
            Corresponding market resource points for interpolation.
        cNrm : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFuncUnc : CubicInterp
            The unconstrained consumption function for this period.
        '''
        # Call the makeCubiccFunc from ConsIndShockSolver.
        cFuncNowUncKink = super().makeCubiccFunc(mNrm, cNrm)
        
        # Change the coeffients at the kinked points.
        cFuncNowUncKink.coeffs[self.i_kink + 1] = [cNrm[self.i_kink], mNrm[self.i_kink + 1] - mNrm[self.i_kink], 0, 0]

        return cFuncNowUncKink
    
    def prepareToCalcEndOfPrdvP(self):
        '''
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.  This differs from the baseline case because
        different savings choices yield different interest rates.

        Parameters
        ----------
        none

        Returns
        -------
        aNrmNow : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.
        '''
        KinkBool = self.Rboro > self.Rsave # Boolean indicating that there is actually a kink.
        # When Rboro == Rsave, this method acts just like it did in IndShock.
        # When Rboro < Rsave, the solver would have terminated when it was called.

        # Make a grid of end-of-period assets, including *two* copies of a=0
        if KinkBool:
            aNrmNow       = np.sort(np.hstack((np.asarray(self.aXtraGrid) + self.mNrmMinNow,
                                                   np.array([0.0,0.0]))))
        else:
            aNrmNow       = np.asarray(self.aXtraGrid) + self.mNrmMinNow
        aXtraCount        = aNrmNow.size

        # Make tiled versions of the assets grid and income shocks
        ShkCount          = self.TranShkValsNext.size
        aNrm_temp         = np.tile(aNrmNow,(ShkCount,1))
        PermShkVals_temp  = (np.tile(self.PermShkValsNext,(aXtraCount,1))).transpose()
        TranShkVals_temp  = (np.tile(self.TranShkValsNext,(aXtraCount,1))).transpose()
        ShkPrbs_temp      = (np.tile(self.ShkPrbsNext,(aXtraCount,1))).transpose()

        # Make a 1D array of the interest factor at each asset gridpoint
        Rfree_vec         = self.Rsave*np.ones(aXtraCount)
        if KinkBool:
            self.i_kink   = np.sum(aNrmNow<=0)-1 # Save the index of the kink point as an attribute
            Rfree_vec[0:self.i_kink] = self.Rboro
        self.Rfree        = Rfree_vec
        Rfree_temp        = np.tile(Rfree_vec,(ShkCount,1))

        # Make an array of market resources that we could have next period,
        # considering the grid of assets and the income shocks that could occur
        mNrmNext          = Rfree_temp/(self.PermGroFac*PermShkVals_temp)*aNrm_temp + TranShkVals_temp

        # Recalculate the minimum MPC and human wealth using the interest factor on saving.
        # This overwrites values from setAndUpdateValues, which were based on Rboro instead.
        if KinkBool:
            PatFacTop         = ((self.Rsave*self.DiscFacEff)**(1.0/self.CRRA))/self.Rsave
            self.MPCminNow    = 1.0/(1.0 + PatFacTop/self.solution_next.MPCmin)
            self.hNrmNow      = self.PermGroFac/self.Rsave*(np.dot(self.ShkPrbsNext,
                                self.TranShkValsNext*self.PermShkValsNext) + self.solution_next.hNrm)

        # Store some of the constructed arrays for later use and return the assets grid
        self.PermShkVals_temp = PermShkVals_temp
        self.ShkPrbs_temp     = ShkPrbs_temp
        self.mNrmNext         = mNrmNext
        self.aNrmNow          = aNrmNow
        return aNrmNow


def solveConsKinkedR(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rboro,Rsave,
                                   PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
    '''
    Solves a single period consumption-saving problem with CRRA utility and risky
    income (subject to permanent and transitory shocks), and different interest
    factors on borrowing and saving.  Restriction: Rboro >= Rsave.  Currently
    cannot construct a cubic spline consumption function, only linear. Can gen-
    erate a value function if requested.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn : [np.array]
        A list containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.
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
    PermGroFac : float
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
        Indicator for whether the solver should use cubic or linear interpolation.

    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (using cubic or linear splines), a marginal
        value function vPfunc, a minimum acceptable level of normalized market
        resources mNrmMin, normalized human wealth hNrm, and bounding MPCs MPCmin
        and MPCmax.  It might also have a value function vFunc.
    '''

    solver = ConsKinkedRsolver(solution_next,IncomeDstn,LivPrb,
                                            DiscFac,CRRA,Rboro,Rsave,PermGroFac,BoroCnstArt,
                                            aXtraGrid,vFuncBool,CubicBool)
    solver.prepareToSolve()
    solution = solver.solve()

    return solution

# ============================================================================
# == Classes for representing types of consumer agents (and things they do) ==
# ============================================================================

# Make a dictionary to specify a perfect foresight consumer type
init_perfect_foresight = {
    'CRRA': 2.0,          # Coefficient of relative risk aversion,
    'Rfree': 1.03,        # Interest factor on assets
    'DiscFac': 0.96,      # Intertemporal discount factor
    'LivPrb': [0.98],     # Survival probability
    'PermGroFac': [1.01], # Permanent income growth factor
    'BoroCnstArt': None,  # Artificial borrowing constraint
    'MaxKinks': 400,      # Maximum number of grid points to allow in cFunc (should be large)
    'AgentCount': 10000,  # Number of agents of this type (only matters for simulation)
    'aNrmInitMean' : 0.0, # Mean of log initial assets (only matters for simulation)
    'aNrmInitStd' : 1.0,  # Standard deviation of log initial assets (only for simulation)
    'pLvlInitMean' : 0.0, # Mean of log initial permanent income (only matters for simulation)
    'pLvlInitStd' : 0.0,  # Standard deviation of log initial permanent income (only matters for simulation)
    'PermGroFacAgg' : 1.0,# Aggregate permanent income growth factor (only matters for simulation)
    'T_age' : None,       # Age after which simulated agents are automatically killed
    'T_cycle' : 1         # Number of periods in the cycle for this agent type
}


class PerfForesightConsumerType(AgentType):
    '''
    A perfect foresight consumer type who has no uncertainty other than mortality.
    His problem is defined by a coefficient of relative risk aversion, intertemporal
    discount factor, interest factor, an artificial borrowing constraint (maybe)
    and time sequences of the permanent income growth rate and survival probability.
    '''
    # Define some universal values for all consumer types
    cFunc_terminal_      = LinearInterp([0.0, 1.0],[0.0,1.0]) # c=m in terminal period
    vFunc_terminal_      = LinearInterp([0.0, 1.0],[0.0,0.0]) # This is overwritten
    solution_terminal_   = ConsumerSolution(cFunc = cFunc_terminal_,
                                            vFunc = vFunc_terminal_, mNrmMin=0.0, hNrm=0.0,
                                            MPCmin=1.0, MPCmax=1.0)
    time_vary_ = ['LivPrb','PermGroFac']
    time_inv_  = ['CRRA','Rfree','DiscFac','MaxKinks','BoroCnstArt']
    poststate_vars_ = ['aNrmNow','pLvlNow']
    shock_vars_ = []

    def __init__(self,
                 cycles=1,
                 time_flow=True,
                 verbose=False,
                 quiet=False,
                 **kwds):
        '''
        Instantiate a new consumer type with given data.
        See init_perfect_foresight for a dictionary of
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

        params = init_perfect_foresight.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=deepcopy(self.solution_terminal_),
                           cycles=cycles,time_flow=time_flow,pseudo_terminal=False,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary      = deepcopy(self.time_vary_)
        self.time_inv       = deepcopy(self.time_inv_)
        self.poststate_vars = deepcopy(self.poststate_vars_)
        self.shock_vars     = deepcopy(self.shock_vars_)
        self.verbose        = verbose
        self.quiet          = quiet
        self.solveOnePeriod = solvePerfForesight # solver for perfect foresight model


    def preSolve(self):
        self.updateSolutionTerminal() # Solve the terminal period problem
        
        # Fill in BoroCnstArt and MaxKinks if they're not specified or are irrelevant.
        if not hasattr(self,'BoroCnstArt'): # If no borrowing constraint specified...
            self.BoroCnstArt = None       # ...assume the user wanted none
        if not hasattr(self,'MaxKinks'):
            if self.cycles > 0: # If it's not an infinite horizon model...
                self.MaxKinks = np.inf  # ...there's no need to set MaxKinks
            elif self.BoroCnstArt is None: # If there's no borrowing constraint...
                self.MaxKinks = np.inf # ...there's no need to set MaxKinks
            else:
                raise(AttributeError('PerfForesightConsumerType requires the attribute MaxKinks to be specified when BoroCnstArt is not None and cycles == 0.'))

            

    def checkRestrictions(self):
        """
        A method to check that various restrictions are met for the model class.
        """
        if self.DiscFac < 0:
            raise Exception('DiscFac is below zero with value: ' + str(self.DiscFac))

        return

    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.solution_terminal.vFunc   = ValueFunc(self.cFunc_terminal_,self.CRRA)
        self.solution_terminal.vPfunc  = MargValueFunc(self.cFunc_terminal_,self.CRRA)
        self.solution_terminal.vPPfunc = MargMargValueFunc(self.cFunc_terminal_,self.CRRA)

    def unpackcFunc(self):
        '''
        "Unpacks" the consumption functions into their own field for easier access.
        After the model has been solved, the consumption functions reside in the
        attribute cFunc of each element of ConsumerType.solution.  This method
        creates a (time varying) attribute cFunc that contains a list of consumption
        functions.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.cFunc = []
        for solution_t in self.solution:
            self.cFunc.append(solution_t.cFunc)
        self.addToTimeVary('cFunc')

    def initializeSim(self):
        self.PlvlAggNow = 1.0
        self.PermShkAggNow = self.PermGroFacAgg # This never changes during simulation
        AgentType.initializeSim(self)


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
        # Get and store states for newly born agents
        N = np.sum(which_agents) # Number of new consumers to make
        self.aNrmNow[which_agents] = drawLognormal(N,mu=self.aNrmInitMean,sigma=self.aNrmInitStd,seed=self.RNG.randint(0,2**31-1))
        pLvlInitMeanNow = self.pLvlInitMean + np.log(self.PlvlAggNow) # Account for newer cohorts having higher permanent income
        self.pLvlNow[which_agents] = drawLognormal(N,mu=pLvlInitMeanNow,sigma=self.pLvlInitStd,seed=self.RNG.randint(0,2**31-1))
        self.t_age[which_agents]   = 0 # How many periods since each agent was born
        self.t_cycle[which_agents] = 0 # Which period of the cycle each agent is currently in
        return None

    def simDeath(self):
        '''
        Determines which agents die this period and must be replaced.  Uses the sequence in LivPrb
        to determine survival probabilities for each agent.

        Parameters
        ----------
        None

        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        '''
        # Determine who dies
        DiePrb_by_t_cycle = 1.0 - np.asarray(self.LivPrb)
        DiePrb = DiePrb_by_t_cycle[self.t_cycle-1] # Time has already advanced, so look back one
        DeathShks = drawUniform(N=self.AgentCount,seed=self.RNG.randint(0,2**31-1))
        which_agents = DeathShks < DiePrb
        if self.T_age is not None: # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents,too_old)
        return which_agents

    def getShocks(self):
        '''
        Finds permanent and transitory income "shocks" for each agent this period.  As this is a
        perfect foresight model, there are no stochastic shocks: PermShkNow = PermGroFac for each
        agent (according to their t_cycle) and TranShkNow = 1.0 for all agents.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        PermGroFac = np.array(self.PermGroFac)
        self.PermShkNow = PermGroFac[self.t_cycle-1] # cycle time has already been advanced
        self.TranShkNow = np.ones(self.AgentCount)

    def getRfree(self):
        '''
        Returns an array of size self.AgentCount with self.Rfree in every entry.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        '''
        RfreeNow = self.Rfree*np.ones(self.AgentCount)
        return RfreeNow

    def getStates(self):
        '''
        Calculates updated values of normalized market resources and permanent income level for each
        agent.  Uses pLvlNow, aNrmNow, PermShkNow, TranShkNow.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        pLvlPrev = self.pLvlNow
        aNrmPrev = self.aNrmNow
        RfreeNow = self.getRfree()

        # Calculate new states: normalized market resources and permanent income level
        self.pLvlNow = pLvlPrev*self.PermShkNow # Updated permanent income level
        self.PlvlAggNow = self.PlvlAggNow*self.PermShkAggNow # Updated aggregate permanent productivity level
        ReffNow      = RfreeNow/self.PermShkNow # "Effective" interest factor on normalized assets
        self.bNrmNow = ReffNow*aNrmPrev         # Bank balances before labor income
        self.mNrmNow = self.bNrmNow + self.TranShkNow # Market resources after income
        return None

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
        MPCnow  = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(self.mNrmNow[these])
        self.cNrmNow = cNrmNow
        self.MPCnow = MPCnow
        return None

    def getPostStates(self):
        '''
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.aNrmNow = self.mNrmNow - self.cNrmNow
        self.aLvlNow = self.aNrmNow*self.pLvlNow   # Useful in some cases to precalculate asset level
        return None

    def checkAIC(self, thorn,verbose,public_call):
        '''
        Evaluate and report on the Absolute Impatience Condition
        '''
        AIF = thorn

        self.AIF = AIF
        if AIF<1:
            self.AIC = True
            if public_call or verbose:
                print('The value of the absolute impatience factor (AIF) for the supplied parameter values satisfies the Absolute Impatience Condition.', end = " ")
                self.violated = False
                if verbose:
                    print('   Because the AIF < 1, the absolute amount of consumption is expected to fall over time.')
            print()
        else:
            self.AIC = False
            print('The given type violates the Absolute Impatience Condition with the supplied parameter values; the AIF is %1.5f ' % (AIF), end=" ")
            self.violated = True
            if verbose:
                print('   Because the AIF > 1, the absolute amount of consumption is expected to grow over time')
            print()

    def checkGICPF(self,thorn,verbose,public_call):
        '''
        Evaluate and report on the Growth Impatience Condition
        '''
        GIFPF = thorn/self.PermGroFac[0]
        self.GIFPF = GIFPF

        if GIFPF<1:
            self.GICPF = True
            if public_call or verbose:
                print('The value of the Growth Impatience Factor for the supplied parameter values satisfies the Perfect Foresight Growth Impatience Condition.', end = " ")
                if verbose:
                    print(' Therefore, for a perfect foresight consumer, the ratio of individual wealth to permanent income will fall indefinitely.')
            print()
        else:
            self.GICPF = False
            self.violated = True
            print('The given parameter values violate the Perfect Foresight Growth Impatience Condition for this consumer type; the GIFPF is: %2.4f' % (GIFPF), end = " ")
            if verbose:
                print(' Therefore, for a perfect foresight consumer the ratio of individual wealth to permanent income is expected to grow toward infinity.')
            print()

    def checkRIC(self, thorn,verbose,public_call):
        '''
        Evaluate and report on the Return Impatience Condition
        '''

        RIF = thorn/self.Rfree
        self.RIF = RIF
        if RIF<1:
            self.RIC = True
            if public_call or verbose:
                print('The return impatience factor value for the supplied parameter values satisfies the Return Impatience Condition.', end = " ")
                if verbose:
                    print('Therefore, the limiting consumption function is not c(m)=0 for all m')
            print()
        else:
            self.RIC = False
            self.violated = True
            print('The given type violates the Return Impatience Condition with the supplied parameter values; the factor is %1.5f ' % (RIF), end = " ")
            if verbose:
                print('Therefore, the limiting consumption function is c(m)=0 for all m')
            print()

    def checkFHWC(self,verbose,public_call):
        '''
        Evaluate and report on the Finite Human Wealth Condition
        '''

        FHWF = self.PermGroFac[0]/self.Rfree
        self.FHWF = FHWF
        if FHWF<1:
            self.hNrm = 1.0/(1.0-self.PermGroFac[0]/self.Rfree)
            self.FHWC = True
            if public_call or verbose:
                print('The Finite Human wealth factor value for the supplied parameter values satisfies the Finite Human Wealth Condition.', end = " ")
                if verbose: 
                    print('Therefore, the limiting consumption function is not c(m)=Infinity')
                    print('and human wealth normalized by permanent income is %2.5f' % (self.hNrm))
                    self.cNrmPDV = 1.0/(1.0-self.Thorn/self.Rfree)
                    print('and the PDV of future consumption growth is %2.5f' % (self.cNrmPDV) )
            print()
        else:
            self.FHWC = False
            print('The given type violates the Finite Human Wealth Condition; the Finite Human wealth factor value %2.5f ' % (FHWF), end = " ")
            self.violated = True
            if verbose:
                print('Therefore, the limiting consumption function is c(m)=Infinity for all m')
            print()
        if verbose and self.violated and verbose_reference:
            print('[!] For more information on the conditions, see Table 3 in "Theoretical Foundations of Buffer Stock Saving" at http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/')
        return self.violated



    def checkConditions(self,verbose=False,verbose_reference=False,public_call=False):
        '''
        This method checks whether the instance's type satisfies the Absolute Impatience Condition (AIC), 
        the Return Impatience Condition (RIC), the Finite Human Wealth Condition (FHWC) and the perfect foresight 
        model's version of the Finite Value of the Growth Impatience Condition (GIC_PF) and 
        Autarky Condition (FVAC_PF). Depending on the configuration of parameter values, some 
        combination of these conditions must be satisfied in order for the problem to have 
        a nondegenerate solution. To check which conditions are required, in the verbose mode
        a reference to the relevant theoretical literature is made.

        Parameters
        ----------
        verbose : boolean
            Specifies different levels of verbosity of feedback. When False, it only reports whether the
            instance's type fails to satisfy a particular condition. When True, it reports all results, i.e.
            the factor values for all conditions.

        Returns
        -------
        None
        '''
        # This method only checks for the conditions for infinite horizon models
        # with a 1 period cycle. If these conditions are not met, we exit early.
        if self.cycles!=0 or self.T_cycle > 1:
            return

        self.violated = False

        Thorn = (self.Rfree*self.DiscFac*self.LivPrb[0])**(1/self.CRRA)
        self.Thorn = Thorn

        self.checkAIC(Thorn,verbose,public_call)
        self.checkGICPF(Thorn,verbose,public_call)
        self.checkRIC(Thorn,verbose,public_call)
        self.checkFHWC(verbose,public_call)



# Make a dictionary to specify an idiosyncratic income shocks consumer
init_idiosyncratic_shocks = dict(init_perfect_foresight,
                                 **{
    # assets above grid parameters
    'aXtraMin': 0.001,      # Minimum end-of-period "assets above minimum" value
    'aXtraMax': 20,         # Maximum end-of-period "assets above minimum" value
    'aXtraNestFac': 3,      # Exponential nesting factor when constructing "assets above minimum" grid
    'aXtraCount': 48,       # Number of points in the grid of "assets above minimum"
    'aXtraExtra': [None],   # Some other value of "assets above minimum" to add to the grid, not used
    # Income process variables
    'PermShkStd': [0.1],    # Standard deviation of log permanent income shocks
    'PermShkCount': 7,      # Number of points in discrete approximation to permanent income shocks
    'TranShkStd': [0.1],    # Standard deviation of log transitory income shocks
    'TranShkCount': 7,      # Number of points in discrete approximation to transitory income shocks
    'UnempPrb': 0.05,       # Probability of unemployment while working
    'UnempPrbRet': 0.005,   # Probability of "unemployment" while retired
    'IncUnemp': 0.3,        # Unemployment benefits replacement rate
    'IncUnempRet': 0.0,     # "Unemployment" benefits when retired
    'BoroCnstArt': 0.0,     # Artificial borrowing constraint; imposed minimum level of end-of period assets
    'tax_rate': 0.0,        # Flat income tax rate
    'T_retire': 0, # Period of retirement (0 --> no retirement)
    'vFuncBool': False,     # Whether to calculate the value function during solution
    'CubicBool': False,     # Use cubic spline interpolation when True, linear interpolation when False
})
class IndShockConsumerType(PerfForesightConsumerType):
    '''
    A consumer type with idiosyncratic shocks to permanent and transitory income.
    His problem is defined by a sequence of income distributions, survival prob-
    abilities, and permanent income growth rates, as well as time invariant values
    for risk aversion, discount factor, the interest rate, the grid of end-of-
    period assets, and an artificial borrowing constraint.
    '''
    time_inv_ = PerfForesightConsumerType.time_inv_ + ['BoroCnstArt','vFuncBool','CubicBool']
    time_inv_.remove('MaxKinks') # This is in the PerfForesight model but not ConsIndShock
    shock_vars_ = ['PermShkNow','TranShkNow']

    def __init__(self,
                 cycles=1,
                 time_flow=True,
                 verbose=False,
                 quiet=False,
                 **kwds):
        '''
        Instantiate a new ConsumerType with given data.
        See init_idiosyncratic_shocks for a dictionary of
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
        params = init_idiosyncratic_shocks.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        PerfForesightConsumerType.__init__(self,
                                           cycles=cycles,
                                           time_flow=time_flow,
                                           verbose=verbose,
                                           quiet=quiet,
                                           **params)

        # Add consumer-type specific objects, copying to create independent versions
        self.solveOnePeriod = solveConsIndShock # idiosyncratic shocks solver
        self.update() # Make assets grid, income process, terminal solution


    def updateIncomeProcess(self):
        '''
        Updates this agent's income process based on his own attributes.

        Parameters
        ----------
        none

        Returns:
        -----------
        none
        '''
        original_time = self.time_flow
        self.timeFwd()
        IncomeDstn, PermShkDstn, TranShkDstn = constructLognormalIncomeProcessUnemployment(self)
        self.IncomeDstn = IncomeDstn
        self.PermShkDstn = PermShkDstn
        self.TranShkDstn = TranShkDstn
        self.addToTimeVary('IncomeDstn','PermShkDstn','TranShkDstn')
        if not original_time:
            self.timeRev()

    def updateAssetsGrid(self):
        '''
        Updates this agent's end-of-period assets grid by constructing a multi-
        exponentially spaced grid of aXtra values.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        aXtraGrid = constructAssetsGrid(self)
        self.aXtraGrid = aXtraGrid
        self.addToTimeInv('aXtraGrid')

    def update(self):
        '''
        Update the income process, the assets grid, and the terminal solution.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.updateIncomeProcess()
        self.updateAssetsGrid()
        self.updateSolutionTerminal()

    def getShocks(self):
        '''
        Gets permanent and transitory income shocks for this period.  Samples from IncomeDstn for
        each period in the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        PermShkNow = np.zeros(self.AgentCount) # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        newborn = self.t_age == 0
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                IncomeDstnNow    = self.IncomeDstn[t-1] # set current income distribution
                PermGroFacNow    = self.PermGroFac[t-1] # and permanent growth factor
                Indices          = np.arange(IncomeDstnNow[0].size) # just a list of integers
                # Get random draws of income shocks from the discrete distribution
                EventDraws       = drawDiscrete(N,X=Indices,P=IncomeDstnNow[0],exact_match=False,seed=self.RNG.randint(0,2**31-1))
                PermShkNow[these] = IncomeDstnNow[1][EventDraws]*PermGroFacNow # permanent "shock" includes expected growth
                TranShkNow[these] = IncomeDstnNow[2][EventDraws]

        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            these = newborn
            IncomeDstnNow    = self.IncomeDstn[0] # set current income distribution
            PermGroFacNow    = self.PermGroFac[0] # and permanent growth factor
            Indices          = np.arange(IncomeDstnNow[0].size) # just a list of integers
            # Get random draws of income shocks from the discrete distribution
            EventDraws       = drawDiscrete(N,X=Indices,P=IncomeDstnNow[0],exact_match=False,seed=self.RNG.randint(0,2**31-1))
            PermShkNow[these] = IncomeDstnNow[1][EventDraws]*PermGroFacNow # permanent "shock" includes expected growth
            TranShkNow[these] = IncomeDstnNow[2][EventDraws]
#        PermShkNow[newborn] = 1.0
        TranShkNow[newborn] = 1.0

        # Store the shocks in self
        self.EmpNow = np.ones(self.AgentCount,dtype=bool)
        self.EmpNow[TranShkNow == self.IncUnemp] = False
        self.PermShkNow = PermShkNow
        self.TranShkNow = TranShkNow


    def calcBoundingValues(self):
        '''
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality (because your income matters to you only if you are still alive).
        The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Unpack the income distribution and get average and worst outcomes
        PermShkValsNext   = self.IncomeDstn[0][1]
        TranShkValsNext   = self.IncomeDstn[0][2]
        ShkPrbsNext       = self.IncomeDstn[0][0]
        ExIncNext         = np.dot(ShkPrbsNext,PermShkValsNext*TranShkValsNext)
        PermShkMinNext    = np.min(PermShkValsNext)
        TranShkMinNext    = np.min(TranShkValsNext)
        WorstIncNext      = PermShkMinNext*TranShkMinNext
        WorstIncPrb       = np.sum(ShkPrbsNext[(PermShkValsNext*TranShkValsNext)==WorstIncNext])

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        hNrm              = (ExIncNext*self.PermGroFac[0]/self.Rfree)/(1.0-self.PermGroFac[0]/self.Rfree)
        temp              = self.PermGroFac[0]*PermShkMinNext/self.Rfree
        BoroCnstNat       = -TranShkMinNext*temp/(1.0-temp)

        PatFac    = (self.DiscFac*self.LivPrb[0]*self.Rfree)**(1.0/self.CRRA)/self.Rfree
        if BoroCnstNat < self.BoroCnstArt:
            MPCmax    = 1.0 # if natural borrowing constraint is overridden by artificial one, MPCmax is 1
        else:
            MPCmax    = 1.0 - WorstIncPrb**(1.0/self.CRRA)*PatFac
        MPCmin = 1.0 - PatFac

        # Store the results as attributes of self
        self.hNrm   = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax

    def makeEulerErrorFunc(self,mMax=100,approx_inc_dstn=True):
        '''
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncomeDstn
        or to use a (temporary) very dense approximation.

        Only works on (one period) infinite horizon models at this time, will
        be generalized later.

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
        # Get the income distribution (or make a very dense one)
        if approx_inc_dstn:
            IncomeDstn = self.IncomeDstn[0]
        else:
            TranShkDstn = approxMeanOneLognormal(N=200,sigma=self.TranShkStd[0],
                                                 tail_N=50,tail_order=1.3, tail_bound=[0.05,0.95])
            TranShkDstn = addDiscreteOutcomeConstantMean(TranShkDstn,self.UnempPrb,self.IncUnemp)
            PermShkDstn = approxMeanOneLognormal(N=200,sigma=self.PermShkStd[0],
                                                 tail_N=50,tail_order=1.3, tail_bound=[0.05,0.95])
            IncomeDstn  = combineIndepDstns(PermShkDstn,TranShkDstn)

        # Make a grid of market resources
        mNowMin  = self.solution[0].mNrmMin + 10**(-15) # add tiny bit to get around 0/0 problem
        mNowMax  = mMax
        mNowGrid = np.linspace(mNowMin,mNowMax,1000)

        # Get the consumption function this period and the marginal value function
        # for next period.  Note that this part assumes a one period cycle.
        cFuncNow   = self.solution[0].cFunc
        vPfuncNext = self.solution[0].vPfunc

        # Calculate consumption this period at each gridpoint (and assets)
        cNowGrid = cFuncNow(mNowGrid)
        aNowGrid = mNowGrid - cNowGrid

        # Tile the grids for fast computation
        ShkCount          = IncomeDstn[0].size
        aCount            = aNowGrid.size
        aNowGrid_tiled    = np.tile(aNowGrid,(ShkCount,1))
        PermShkVals_tiled = (np.tile(IncomeDstn[1],(aCount,1))).transpose()
        TranShkVals_tiled = (np.tile(IncomeDstn[2],(aCount,1))).transpose()
        ShkPrbs_tiled     = (np.tile(IncomeDstn[0],(aCount,1))).transpose()

        # Calculate marginal value next period for each gridpoint and each shock
        mNextArray        = self.Rfree/(self.PermGroFac[0]*PermShkVals_tiled)*aNowGrid_tiled + TranShkVals_tiled
        vPnextArray       = vPfuncNext(mNextArray)

        # Calculate expected marginal value and implied optimal consumption
        ExvPnextGrid = self.DiscFac*self.Rfree*self.LivPrb[0]*self.PermGroFac[0]**(-self.CRRA)* \
                       np.sum(PermShkVals_tiled**(-self.CRRA)*vPnextArray*ShkPrbs_tiled,axis=0)
        cOptGrid     = ExvPnextGrid**(-1.0/self.CRRA) # This is the 'Endogenous Gridpoints' step

        # Calculate Euler error and store an interpolated function
        EulerErrorNrmGrid = (cNowGrid - cOptGrid)/cOptGrid
        eulerErrorFunc    = LinearInterp(mNowGrid,EulerErrorNrmGrid)
        self.eulerErrorFunc = eulerErrorFunc

    def preSolve(self):
#        AgentType.preSolve(self)
        # Update all income process variables to match any attributes that might
        # have been changed since `__init__` or `solve()` was last called.
#        self.updateIncomeProcess()
        self.updateSolutionTerminal()
        if not self.quiet:
            self.checkConditions(verbose=self.verbose,public_call=False)

    def checkGICInd(self,Thorn,verbose,public_call):
        '''
        Check Individual Growth Impatience Factor.
        '''
        if self.GIFInd<=1:
            self.GICInd = True
            if public_call or verbose:
                print('The value of the Individual Growth Impatience Factor for the supplied parameter values satisfies the Individual Growth Impatience Condition.', end = " ")
                if verbose:
                    print('Therefore, a target level of the individual market resources ratio m exists (see '+self.url+'/#onetarget for more).')
            print()
        else:
            self.GICInd = False
            self.violated = True
            print('The given parameter values violate the Individual Growth Impatience Condition; the GIFInd is: %2.4f' % (self.GIFInd), end = " ")
            if verbose:
                print('')
                print('Therefore, a target ratio of individual market resources to individual permanent income does not exist.  (see '+self.url+'/#onetarget for more).')
            print()

    def checkCIGAgg(self, Thorn,verbose,public_call):

        if self.GIFAgg<=1:
            self.GICAgg = True
            if public_call or verbose:
                print('The value of the Aggregate Growth Impatience Factor for the supplied parameter values satisfies the Aggregate Growth Impatience Condition.', end = " ")
                if verbose:
                    print('Therefore, it is possible that a target level of the ratio of aggregate market resources to aggregate permanent income exists.') # Need to provide reference 
            print()
        else:
            self.GICAgg = False
            self.violated = True
            print('The given parameter values violate the Aggregate Growth Impatience Condition; the GIFAgg is: %2.4f' % (self.GIFAgg), end = " ")
            if verbose:
                print('')
                print('Therefore, a target ratio of aggregate resources to aggregate permanent income does not exist.') # Need to provide reference
            print()

    def checkWRIC(self, verbose,public_call):
        '''
        Evaluate and report on the Weak Return Impatience Condition
        [url]/#WRIF modified to incorporate LivPrb
        '''
        WRIF=(self.UnempPrb**(1/self.CRRA))*(self.Rfree*self.DiscFac*self.LivPrb[0])**(1/self.CRRA)/self.Rfree
        self.WRIF = WRIF
        if WRIF<=1:
            self.WRIC = True
            if public_call or verbose:
                print('The Weak Return Impatience Factor value for the supplied parameter values satisfies the Weak Return Impatience Condition (see '+self.url+'/#WRIC for more).')
                print()
        else:
            self.WRIC = False
            self.violated = True
            print('The given type violates the Weak Return Impatience Condition with the supplied parameter values.  The WRIF is: %2.4f' % (WRIF), end = " ")
            if verbose:
                print('')
                print('Therefore, a nondegenerate solution is not available (see '+self.url+'/#WRIC for more.')
            print()

    def checkFVAC(self,verbose,public_call):
        '''
        Evaluate and report on the Finite Value of Autarky Condition
        Hyperlink to paper: [url]/#Autarky-Value
        '''
        EpShkuInv = np.dot(self.PermShkDstn[0][0],self.PermShkDstn[0][1]**(1-self.CRRA))
        if self.CRRA != 1.0:
            uInvEpShkuInv = EpShkuInv**(1/(1-self.CRRA)) # The term that gives a utility-consequence-adjusted utility growth
        else:
            uInvEpShkuInv = 1.0
        
        self.uInvEpShkuInv   = uInvEpShkuInv
        FVAF=self.LivPrb[0]*self.DiscFac*self.uInvEpShkuInv
        self.FVAF = FVAF
        if FVAF<=1:
            self.FVAC = True
            if public_call or verbose:
                print('The Finite Value of Autarky Factor (FVAV) for the supplied parameter values satisfies the Finite Value of Autarky Condition.')
                if self.WRIC:
                    print('Since both WRIC and FVAC are satisfied, the problem has a nondegenerate solution')
        else:
            self.FVAC = False
            print('The given type violates the Finite Value of Autarky Condition with the supplied parameter values. The FVAF is %2.4f' %(FVAF), end = " ")
            self.violated = True
            if public_call or verbose:
                print('Therefore, a nondegenerate solution is not available (see '+self.url+'/#Conditions-Under-Which-the-Problem-Defines-a-Contraction-Mapping')
            print()




    def checkConditions(self,verbose=False,public_call=True):
        '''
        This method checks whether the instance's type satisfies the Absolute Impatience Condition (AIC), Weak Return
        Impatience Condition (WRIC), Finite Human Wealth Condition (FHWC) and Finite Value of
        Autarky Condition (FVAC).  When combinations of these conditions are satisfied, the 
        solution to the problem exhibits different characteristics.  (For an exposition of the
        conditions, see http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/)

        Parameters
        ----------
        verbose : boolean
            Specifies different levels of verbosity of feedback. When False, it only reports whether the
            instance's type fails to satisfy a particular condition. When True, it reports all results, i.e.
            the factor values for all conditions.

        Returns
        -------
        None
        '''
        self.violated = False # PerfForesightConsumerType.checkConditions(self, verbose=False, verbose_reference=False)

        if self.cycles!=0 or self.T_cycle > 1:
            return

        # For theory, see hyperlink targets to expressions in 
        # url=http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory
        # For example, the hyperlink to the relevant section of the paper
        self.url='http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory'
        # would be referenced below as:
        # [url]/#Uncertainty-Modified-Conditions
        
        self.InvPermShkDstn=deepcopy(self.PermShkDstn)
        self.InvPermShkDstn[0][1]=1/self.PermShkDstn[0][1]
        EPermShkInv=np.dot(self.InvPermShkDstn[0][0],1/self.PermShkDstn[0][1]) # $\Ex_{t}[\psi^{-1}_{t+1}]$ (in first eqn in sec)

        InvEPermShkInv=(1/EPermShkInv)                          # $\underline{\psi}$ in the paper (\bar{\isp} in private version)
        PermGroFacAdj=self.PermGroFac[0]*InvEPermShkInv                # [url]/#PGroAdj
        # [url]/#Pat, adjusted to include mortality
#        Thorn = ((self.Rfree/self.LivPrb[0])*(self.DiscFac*self.LivPrb[0]))**(1/self.CRRA)
        Thorn = ((self.Rfree*self.DiscFac))**(1/self.CRRA)
        GIFPF  = Thorn/(self.PermGroFac[0]               )      # [url]/#GIF
        GIFInd = Thorn/(self.PermGroFac[0]*InvEPermShkInv)      # [url]/#GIFI
        GIFAgg = Thorn*self.LivPrb[0]/self.PermGroFac[0]        # Lower bound of aggregate wealth growth if all inheritances squandered

#        self.Rnorm           = self.Rfree*EPermShkInv/(self.PermGroFac[0]*self.LivPrb[0])
        self.GIFPF           = GIFPF
        self.GIFInd          = GIFInd
        self.GIFAgg          = GIFAgg
        self.Thorn           = Thorn
        self.PermGroFacAdj   = PermGroFacAdj
        self.EPermShkInv     = EPermShkInv
        self.InvEPermShkInv  = InvEPermShkInv
        self.DiscFacGIFPFMax = ((self.PermGroFac[0]              )**(self.CRRA))/(self.Rfree) # DiscFac at growth impatience knife edge
        self.DiscFacGIFIndMax = ((self.PermGroFac[0]*InvEPermShkInv)**(self.CRRA))/(self.Rfree) # DiscFac at growth impatience knife edge
        self.DiscFacGIFAggMax = ((self.PermGroFac[0]               )**(self.CRRA))/(self.Rfree*self.LivPrb[0]) # DiscFac at growth impatience knife edge

        self.checkGICPF(Thorn,verbose,public_call)
        self.checkGICInd(Thorn,verbose,public_call)
        self.checkCIGAgg(Thorn,verbose,public_call)
        self.checkWRIC(verbose,public_call)
        self.checkFVAC(verbose, public_call)

        if verbose and self.violated:
            print('\n[!] For more information on the conditions, see Tables 3 and 4 in "Theoretical Foundations of Buffer Stock Saving" at '+self.url+'/#Factors-Defined-And-Compared')
            print('')

        if verbose:
            print('GIFPF            = %2.6f ' % (GIFPF))
            print('GIFInd           = %2.6f ' % (GIFInd))
            print('GIFAgg           = %2.6f ' % (GIFAgg))
            print('Thorn = AIF      = %2.6f ' % (Thorn))
            print('PermGroFacAdj    = %2.6f ' % (PermGroFacAdj))
            print('uInvEpShkuInv    = %2.6f ' % (self.uInvEpShkuInv))
            print('FVAF             = %2.6f ' % (self.FVAF))
            print('WRIF             = %2.6f ' % (self.WRIF))
            print('DiscFacGIFIndMax = %2.6f ' % (self.DiscFacGIFIndMax))
            print('DiscFacGIFAggMax = %2.6f ' % (self.DiscFacGIFAggMax))

    def Ex_Mtp1_over_Ex_Ptp1(self,mRat,verbose=False):
        cRat        = self.solution[-1].cFunc(mRat)
        aRat        = mRat-cRat
        Ex_Ptp1     = PermGroFac[0]
        Ex_bLev_tp1 = aRat*self.Rfree
        Ex_Mtp1     = Ex_bLev_tp1
        return Ex_Mtp1/Ex_Ptp1
                    
    def Ex_mtp1(self,mRat,verbose=False):
        cRat        = self.solution[-1].cFunc(mRat)
        aRat        = mRat-cRat
        Ex_bRat_tp1 = aRat*self.Rfree*self.EPermShkInv/self.PermGroFac[0]
        Ex_Mtp1     = (Ex_bRat_tp1 + 1)*Ex_Ptp1 # mean TranShk and PermShk are 1
        return Ex_Mtp1/Ex_Ptp1
                    
    def calcTargets(self,verbose=False):
        '''
        If the problem is one that satisfies the conditions required for target ratios of different
        variables to permanent income to exist, and has been solved to within the self-defined
        tolerance, this method calculates the target values of market resources, consumption, 
        and assets.  

        Parameters
        ----------
        verbose : boolean
            Specifies different levels of verbosity of feedback. When False, it prints no results. 
            When True, it reports all target values, and passes the verbosity indicator to the 
            checkConditions method which responds accordingly.

        Returns
        -------
        None
        '''
        infinite_horizon = cycles_left == 0
        if not infinite_horizon:
            print('The calcTargets method works only for infinite horizon models.')
            return

        
        # To be written.
        # Defining:
        ## Rnorm    = Rfree/(PermGroFac[0]*PermShk)
        ## EPermShkInv    = E[PermShk**(-1)]
        ## InvEPermShkInv = 1/EPermShkInv
        ## ExRnorm  = E[Rfree/(PermGroFac[0]*PermShk)] = Rfree EPermShkInv / PermGroFac[0]
        ## InvExRnorm = 1/ExRnorm
        ## The "sustainable consumption" locus is given by
        # cSust = InvExRnorm + m*(1-InvExRnorm)

        # The target level of m, mTarg, will be the value such that
        # cSust[m] = cFunc[m]

# Make a dictionary to specify a "kinked R" idiosyncratic shock consumer
init_kinked_R = dict(init_idiosyncratic_shocks,
                     **{
    'Rboro' : 1.20, # Interest factor on assets when borrowing, a < 0
    'Rsave' : 1.02, # Interest factor on assets when saving, a > 0
    'BoroCnstArt' : None, # kinked R is a bit silly if borrowing not allowed
    'CubicBool' : True, # kinked R is now compatible with linear cFunc and cubic cFunc
    'aXtraCount' : 48,   # ...so need lots of extra gridpoints to make up for it
})
del init_kinked_R['Rfree'] # get rid of constant interest factor


class KinkedRconsumerType(IndShockConsumerType):
    '''
    A consumer type that faces idiosyncratic shocks to income and has a different
    interest factor on saving vs borrowing.  Extends IndShockConsumerType, with
    very small changes.  Solver for this class is currently only compatible with
    linear spline interpolation.
    '''
    time_inv_ = copy(IndShockConsumerType.time_inv_)
    time_inv_.remove('Rfree')
    time_inv_ += ['Rboro', 'Rsave']

    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data.
        See ConsumerParameters.init_kinked_R for a dictionary of
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
        params = init_kinked_R.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        PerfForesightConsumerType.__init__(self,cycles=cycles,time_flow=time_flow,**params)

        # Add consumer-type specific objects, copying to create independent versions
        self.solveOnePeriod = solveConsKinkedR # kinked R solver
        self.update() # Make assets grid, income process, terminal solution

    def preSolve(self):
#        AgentType.preSolve(self)
        self.updateSolutionTerminal()

    def calcBoundingValues(self):
        '''
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.  This version deals
        with the different interest rates on borrowing vs saving.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Unpack the income distribution and get average and worst outcomes
        PermShkValsNext   = self.IncomeDstn[0][1]
        TranShkValsNext   = self.IncomeDstn[0][2]
        ShkPrbsNext       = self.IncomeDstn[0][0]
        ExIncNext         = np.dot(ShkPrbsNext,PermShkValsNext*TranShkValsNext)
        PermShkMinNext    = np.min(PermShkValsNext)
        TranShkMinNext    = np.min(TranShkValsNext)
        WorstIncNext      = PermShkMinNext*TranShkMinNext
        WorstIncPrb       = np.sum(ShkPrbsNext[(PermShkValsNext*TranShkValsNext)==WorstIncNext])

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        hNrm              = (ExIncNext*self.PermGroFac[0]/self.Rsave)/(1.0-self.PermGroFac[0]/self.Rsave)
        temp              = self.PermGroFac[0]*PermShkMinNext/self.Rboro
        BoroCnstNat       = -TranShkMinNext*temp/(1.0-temp)

        PatFacTop = (self.DiscFac*self.LivPrb[0]*self.Rsave)**(1.0/self.CRRA)/self.Rsave
        PatFacBot = (self.DiscFac*self.LivPrb[0]*self.Rboro)**(1.0/self.CRRA)/self.Rboro
        if BoroCnstNat < self.BoroCnstArt:
            MPCmax    = 1.0 # if natural borrowing constraint is overridden by artificial one, MPCmax is 1
        else:
            MPCmax    = 1.0 - WorstIncPrb**(1.0/self.CRRA)*PatFacBot
        MPCmin = 1.0 - PatFacTop

        # Store the results as attributes of self
        self.hNrm   = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax

    def makeEulerErrorFunc(self,mMax=100,approx_inc_dstn=True):
        '''
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncomeDstn
        or to use a (temporary) very dense approximation.

        SHOULD BE INHERITED FROM ConsIndShockModel

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

    def getRfree(self):
        '''
        Returns an array of size self.AgentCount with self.Rboro or self.Rsave in each entry, based
        on whether self.aNrmNow >< 0.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        '''
        RfreeNow = self.Rboro*np.ones(self.AgentCount)
        RfreeNow[self.aNrmNow > 0] = self.Rsave
        return RfreeNow

    def checkConditions(self,verbose=False):
        '''
        This method checks whether the instance's type satisfies the Absolute Impatience Condition (AIC), 
        the Return Impatience Condition (RIC), the Growth Impatience Condition (GIC), the Weak Return 
        Impatience Condition (WRIC), the Finite Human Wealth Condition (FHWC) and the Finite Value of
        Autarky Condition (FVAC). To check which conditions are relevant to the model at hand, a 
        reference to the relevant theoretical literature is made.

        Parameters
        ----------
        verbose : boolean
            Specifies different levels of verbosity of feedback. When False, it only reports whether the
            instance's type fails to satisfy a particular condition. When True, it reports all results, i.e.
            the factor values for all conditions.

        Returns
        -------
        None
        '''
        raise NotImplementedError()

# ==================================================================================
# = Functions for generating discrete income processes and simulated income shocks =
# ==================================================================================

def constructLognormalIncomeProcessUnemployment(parameters):
    '''
    Generates a list of discrete approximations to the income process for each
    life period, from end of life to beginning of life.  Permanent shocks are mean
    one lognormally distributed with standard deviation PermShkStd[t] during the
    working life, and degenerate at 1 in the retirement period.  Transitory shocks
    are mean one lognormally distributed with a point mass at IncUnemp with
    probability UnempPrb while working; they are mean one with a point mass at
    IncUnempRet with probability UnempPrbRet.  Retirement occurs
    after t=T_retire periods of working.

    Note 1: All time in this function runs forward, from t=0 to t=T

    Note 2: All parameters are passed as attributes of the input parameters.

    Parameters (passed as attributes of the input parameters)
    ----------
    PermShkStd : [float]
        List of standard deviations in log permanent income uncertainty during
        the agent's life.
    PermShkCount : int
        The number of approximation points to be used in the discrete approxima-
        tion to the permanent income shock distribution.
    TranShkStd : [float]
        List of standard deviations in log transitory income uncertainty during
        the agent's life.
    TranShkCount : int
        The number of approximation points to be used in the discrete approxima-
        tion to the permanent income shock distribution.
    UnempPrb : float
        The probability of becoming unemployed during the working period.
    UnempPrbRet : float
        The probability of not receiving typical retirement income when retired.
    T_retire : int
        The index value for the final working period in the agent's life.
        If T_retire <= 0 then there is no retirement.
    IncUnemp : float
        Transitory income received when unemployed.
    IncUnempRet : float
        Transitory income received while "unemployed" when retired.
    T_cycle :  int
        Total number of non-terminal periods in the consumer's sequence of periods.

    Returns
    -------
    IncomeDstn :  [[np.array]]
        A list with T_cycle elements, each of which is a list of three arrays
        representing a discrete approximation to the income process in a period.
        Order: probabilities, permanent shocks, transitory shocks.
    PermShkDstn : [[np.array]]
        A list with T_cycle elements, each of which is a list of two arrays
        representing a discrete approximation to the permanent income shocks.
    TranShkDstn : [[np.array]]
        A list with T_cycle elements, each of which is a list of two arrays
        representing a discrete approximation to the transitory income shocks.
    '''
    # Unpack the parameters from the input
    PermShkStd    = parameters.PermShkStd
    PermShkCount  = parameters.PermShkCount
    TranShkStd    = parameters.TranShkStd
    TranShkCount  = parameters.TranShkCount
    T_cycle       = parameters.T_cycle
    T_retire      = parameters.T_retire
    UnempPrb      = parameters.UnempPrb
    IncUnemp      = parameters.IncUnemp
    UnempPrbRet   = parameters.UnempPrbRet
    IncUnempRet   = parameters.IncUnempRet

    IncomeDstn    = [] # Discrete approximations to income process in each period
    PermShkDstn   = [] # Discrete approximations to permanent income shocks
    TranShkDstn   = [] # Discrete approximations to transitory income shocks

    # Fill out a simple discrete RV for retirement, with value 1.0 (mean of shocks)
    # in normal times; value 0.0 in "unemployment" times with small prob.
    if T_retire > 0:
        if UnempPrbRet > 0:
            PermShkValsRet  = np.array([1.0, 1.0])    # Permanent income is deterministic in retirement (2 states for temp income shocks)
            TranShkValsRet  = np.array([IncUnempRet,
                                        (1.0-UnempPrbRet*IncUnempRet)/(1.0-UnempPrbRet)])
            ShkPrbsRet      = np.array([UnempPrbRet, 1.0-UnempPrbRet])
        else:
            PermShkValsRet  = np.array([1.0])
            TranShkValsRet  = np.array([1.0])
            ShkPrbsRet      = np.array([1.0])
        IncomeDstnRet = [ShkPrbsRet,PermShkValsRet,TranShkValsRet]

    # Loop to fill in the list of IncomeDstn random variables.
    for t in range(T_cycle): # Iterate over all periods, counting forward

        if T_retire > 0 and t >= T_retire:
            # Then we are in the "retirement period" and add a retirement income object.
            IncomeDstn.append(deepcopy(IncomeDstnRet))
            PermShkDstn.append([np.array([1.0]),np.array([1.0])])
            TranShkDstn.append([ShkPrbsRet,TranShkValsRet])
        else:
            # We are in the "working life" periods.
            TranShkDstn_t    = approxMeanOneLognormal(N=TranShkCount, sigma=TranShkStd[t], tail_N=0)
            if UnempPrb > 0:
                TranShkDstn_t = addDiscreteOutcomeConstantMean(TranShkDstn_t, p=UnempPrb, x=IncUnemp)
            PermShkDstn_t    = approxMeanOneLognormal(N=PermShkCount, sigma=PermShkStd[t], tail_N=0)
            IncomeDstn.append(combineIndepDstns(PermShkDstn_t,TranShkDstn_t)) # mix the independent distributions
            PermShkDstn.append(PermShkDstn_t)
            TranShkDstn.append(TranShkDstn_t)
    return IncomeDstn, PermShkDstn, TranShkDstn


def applyFlatIncomeTax(IncomeDstn,tax_rate,T_retire,unemployed_indices=[],transitory_index=2):
    '''
    Applies a flat income tax rate to all employed income states during the working
    period of life (those before T_retire).  Time runs forward in this function.

    Parameters
    ----------
    IncomeDstn : [income distributions]
        The discrete approximation to the income distribution in each time period.
    tax_rate : float
        A flat income tax rate to be applied to all employed income.
    T_retire : int
        The time index after which the agent retires.
    unemployed_indices : [int]
        Indices of transitory shocks that represent unemployment states (no tax).
    transitory_index : int
        The index of each element of IncomeDstn representing transitory shocks.

    Returns
    -------
    IncomeDstn_new : [income distributions]
        The updated income distributions, after applying the tax.
    '''
    IncomeDstn_new = deepcopy(IncomeDstn)
    i = transitory_index
    for t in range(len(IncomeDstn)):
        if t < T_retire:
            for j in range((IncomeDstn[t][i]).size):
                if j not in unemployed_indices:
                    IncomeDstn_new[t][i][j] = IncomeDstn[t][i][j]*(1-tax_rate)
    return IncomeDstn_new

# =======================================================
# ================ Other useful functions ===============
# =======================================================

def constructAssetsGrid(parameters):
    '''
    Constructs the base grid of post-decision states, representing end-of-period
    assets above the absolute minimum.

    All parameters are passed as attributes of the single input parameters.  The
    input can be an instance of a ConsumerType, or a custom Parameters class.

    Parameters
    ----------
    aXtraMin:                  float
        Minimum value for the a-grid
    aXtraMax:                  float
        Maximum value for the a-grid
    aXtraCount:                 int
        Size of the a-grid
    aXtraExtra:                [float]
        Extra values for the a-grid.
    exp_nest:               int
        Level of nesting for the exponentially spaced grid

    Returns
    -------
    aXtraGrid:     np.ndarray
        Base array of values for the post-decision-state grid.
    '''
    # Unpack the parameters
    aXtraMin     = parameters.aXtraMin
    aXtraMax     = parameters.aXtraMax
    aXtraCount   = parameters.aXtraCount
    aXtraExtra   = parameters.aXtraExtra
    grid_type    = 'exp_mult'
    exp_nest     = parameters.aXtraNestFac

    # Set up post decision state grid:
    aXtraGrid = None
    if grid_type == "linear":
        aXtraGrid = np.linspace(aXtraMin, aXtraMax, aXtraCount)
    elif grid_type == "exp_mult":
        aXtraGrid = makeGridExpMult(ming=aXtraMin, maxg=aXtraMax, ng=aXtraCount, timestonest=exp_nest)
    else:
        raise Exception("grid_type not recognized in __init__." + \
                         "Please ensure grid_type is 'linear' or 'exp_mult'")

    # Add in additional points for the grid:
    for a in aXtraExtra:
        if (a is not None):
            if a not in aXtraGrid:
                j      = aXtraGrid.searchsorted(a)
                aXtraGrid = np.insert(aXtraGrid, j, a)

    return aXtraGrid

# Make a dictionary to specify a lifecycle consumer with a finite horizon
init_lifecycle = copy(init_idiosyncratic_shocks)
init_lifecycle['PermGroFac'] = [1.01,1.01,1.01,1.01,1.01,1.02,1.02,1.02,1.02,1.02]
init_lifecycle['PermShkStd'] = [0.1,0.2,0.1,0.2,0.1,0.2,0.1,0,0,0]
init_lifecycle['TranShkStd'] = [0.3,0.2,0.1,0.3,0.2,0.1,0.3,0,0,0]
init_lifecycle['LivPrb']     = [0.99,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
init_lifecycle['T_cycle']    = 10
init_lifecycle['T_retire']   = 7
init_lifecycle['T_age']      = 11 # Make sure that old people die at terminal age and don't turn into newborns!

# Make a dictionary to specify an infinite consumer with a four period cycle
init_cyclical = copy(init_idiosyncratic_shocks)
init_cyclical['PermGroFac'] = [1.082251, 2.8, 0.3, 1.1]
init_cyclical['PermShkStd'] = [0.1,0.1,0.1,0.1]
init_cyclical['TranShkStd'] = [0.1,0.1,0.1,0.1]
init_cyclical['LivPrb']     = 4*[0.98]
init_cyclical['T_cycle']    = 4

