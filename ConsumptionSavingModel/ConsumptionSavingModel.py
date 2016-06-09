'''
This module contains classes to solve canonical consumption-savings models with CRRA utility.  
It currently solves four models.
   1) A very basic one-period "perfect foresight" consumption-savings model with no uncertainty.
   2) An infinite-horizon OR lifecycle consumption-savings model with uncertainty over transitory
      and permanent income shocks.
   3) The model described in (2), with an interest rate for debt that differs from the interest
      rate for savings.
   4) The model described in (2), with the addition of an exogenous Markov state that affects
      income (e.g. unemployment.)

See NARK for information on variable naming conventions.
See HARK documentation for mathematical descriptions of the models being solved.
'''
import sys 
sys.path.insert(0,'../')

from copy import copy, deepcopy
import numpy as np
from HARKcore import AgentType, Solution, NullFunc
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKinterpolation import CubicInterp, LowerEnvelope, LinearInterp
from HARKsimulation import drawMeanOneLognormal, drawBernoulli
from HARKutilities import approxLognormal, approxMeanOneLognormal, addDiscreteOutcomeConstantMean,\
                          combineIndepDstns, makeGridExpMult, CRRAutility, CRRAutilityP, \
                          CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv, \
                          CRRAutilityP_invP


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
    
    Here and elsewhere in the code, Nrm indicates that variables are normalized by permanent income.
    '''
    def __init__(self, cFunc=NullFunc, vFunc=NullFunc, 
                       vPfunc=NullFunc, vPPfunc=NullFunc,
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
        self.cFunc        = cFunc
        self.vFunc        = vFunc
        self.vPfunc       = vPfunc
        self.vPPfunc      = vPPfunc
        self.mNrmMin      = mNrmMin
        self.hNrm         = hNrm
        self.MPCmin       = MPCmin
        self.MPCmax       = MPCmax
        self.distance_criteria = ['cFunc']

    def appendSolution(self,new_solution):
        '''
        Appends one solution to another to create a ConsumerSolution whose
        attributes are lists.  Used in solveConsumptionSavingMarkov, where we append solutions
        *conditional* on a particular value of a Markov state to each other in order to get the
        entire solution.
        
        Parameters
        ----------
        new_solution : ConsumerSolution
            The solution to a consumption-saving problem conditional on being
            in a particular Markov state to begin the period.
            
        Returns
        -------
        none
        '''
        if type(self.cFunc)!=list:
            # Then we assume that self is an empty initialized solution instance.
            # Begin by checking this is so.            
            assert self.cFunc==NullFunc, 'appendSolution called incorrectly!'         

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

        
class ValueFunc():
    '''
    A class for representing a value function.  The underlying interpolation is
    in the space of (m,u_inv(v)); this class "re-curves" to the value function.
    '''
    def __init__(self,vFuncDecurved,CRRA):
        '''
        Constructor for a new value function object.
        
        Parameters
        ----------
        vFuncDecurved : function
            A real function representing the value function composed with the
            inverse utility function, defined on market resources: u_inv(vFunc(m))
        CRRA : float
            Coefficient of relative risk aversion.
            
        Returns
        -------
        None
        '''
        self.func = deepcopy(vFuncDecurved)
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

     
class MargValueFunc():
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
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
        
        
class MargMargValueFunc():
    '''
    A class for representing a marginal marginal value function in models where
    the standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
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

class PerfectForesightSolver(object):
    '''
    A class for solving a one period perfect foresight consumption-saving problem.
    An instance of this class is created by the function solvePerfForesight in each period.
    '''
    def __init__(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac):
        '''
        Constructor for a new PerfectForesightSolver.
        
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
        PermGroGac : float
            Expected permanent income growth factor at the end of this period.
            
        Returns:
        ----------
        None       
        '''
        # We ask that HARK users define single-letter variables they use in a dictionary
        # attribute called notation.
        # Do that first.
        self.notation = {'a': 'assets after all actions',
                         'm': 'market resources at decision time',
                         'c': 'consumption'}
        self.assignParameters(solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac)
         
    def assignParameters(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac):
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
        PermGroGac : float
            Expected permanent income growth factor at the end of this period.
            
        Returns
        -------
        none    
        '''
        self.solution_next  = solution_next
        self.DiscFac        = DiscFac
        self.LivPrb         = LivPrb
        self.CRRA           = CRRA
        self.Rfree          = Rfree
        self.PermGroFac     = PermGroFac
    
    def defUtilityFuncs(self):
        '''
        Defines CRRA utility function for this period (and its derivatives),
        saving them as attributes of self for other methods to use.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        self.u   = lambda c : utility(c,gam=self.CRRA)  # utility function
        self.uP  = lambda c : utilityP(c,gam=self.CRRA) # marginal utility function
        self.uPP = lambda c : utilityPP(c,gam=self.CRRA)# marginal marginal utility function

    def defValueFuncs(self):
        '''
        Defines the value and marginal value function for this period.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        MPCnvrs      = self.MPC**(-self.CRRA/(1.0-self.CRRA))
        vFuncNvrs    = LinearInterp(np.array([self.mNrmMin, self.mNrmMin+1.0]),np.array([0.0, MPCnvrs]))
        self.vFunc   = ValueFunc(vFuncNvrs,self.CRRA)
        self.vPfunc  = MargValueFunc(self.cFunc,self.CRRA)
        
    def makecFuncPF(self):
        '''
        Makes the (linear) consumption function for this period.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        # Calculate human wealth this period (and lower bound of m)
        self.hNrmNow = (self.PermGroFac/self.Rfree)*(self.solution_next.hNrm + 1.0)
        self.mNrmMin = -self.hNrmNow
        # Calculate the (constant) marginal propensity to consume
        PatFac       = ((self.Rfree*self.DiscFacEff)**(1.0/self.CRRA))/self.Rfree
        self.MPC     = 1.0/(1.0 + PatFac/self.solution_next.MPCmin)
        # Construct the consumption function
        self.cFunc   = LinearInterp([self.mNrmMin, self.mNrmMin+1.0],[0.0, self.MPC])
        
    def solve(self): 
        '''
        Solves the one period perfect foresight consumption-saving problem.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        solution : ConsumerSolution
            The solution to this period's problem.
        '''
        self.defUtilityFuncs()
        self.DiscFacEff = self.DiscFac*self.LivPrb
        self.makecFuncPF()
        self.defValueFuncs()
        solution = ConsumerSolution(cFunc=self.cFunc, vFunc=self.vFunc, vPfunc=self.vPfunc,
                                    mNrmMin=self.mNrmMin, hNrm=self.hNrmNow,
                                    MPCmin=self.MPC, MPCmax=self.MPC)
        return solution


def solvePerfForesight(solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac):
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
    PermGroGac : float
        Expected permanent income growth factor at the end of this period.
        
    Returns
    -------
    solution : ConsumerSolution
            The solution to this period's problem.
    '''
    solver = PerfectForesightSolver(solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac)
    solution = solver.solve()
    return solution


###############################################################################
###############################################################################
class SetupImperfectForesightSolver(PerfectForesightSolver):
    '''
    A superclass for solvers of one period consumption-saving problems with
    constant relative risk aversion utility and permanent and transitory shocks
    to income.
    '''
    def __init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                      PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver for problems with income subject to permanent and transitory
        shocks.
        
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
        self.assignParameters(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.defineUtilityFunctions()

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
        none
        '''
        PerfectForesightSolver.assignParameters(self,solution_next,DiscFac,LivPrb,
                                                CRRA,Rfree,PermGroFac)
        self.BoroCnstArt       = BoroCnstArt
        self.IncomeDstn     = IncomeDstn
        self.aXtraGrid      = aXtraGrid
        self.vFuncBool      = vFuncBool
        self.CubicBool      = CubicBool
        

    def defineUtilityFunctions(self):
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
        PerfectForesightSolver.defUtilityFuncs(self)
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
        none
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
        self.hNrmNow      = self.PermGroFac/self.Rfree*(
                            np.dot(self.ShkPrbsNext,self.TranShkValsNext*self.PermShkValsNext) + 
                            solution_next.hNrm)
        self.MPCmaxNow    = 1.0/(1.0 + (self.WorstIncPrb**(1.0/self.CRRA))*
                                        self.PatFac/solution_next.MPCmax)


    def defineBorrowingConstraint(self,BoroCnstArt):
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
        self.defineBorrowingConstraint(self.BoroCnstArt)


####################################################################################################
####################################################################################################

class ConsumptionSavingSolverENDGBasic(SetupImperfectForesightSolver):
    '''
    This class solves a single period of a standard consumption-saving problem,
    using linear interpolation and without the ability to calculate the value
    function.  ConsumptionSavingSolverENDG inherits from this class and adds the
    ability to perform cubic interpolation and to calculate the value function.
    
    Note that this class does not have its own initializing method.  It initial-
    izes the same problem in the same way as SetupImperfectForesightSolver,
    from which it inherits.
    '''    
    def prepareToGetGothicvP(self):
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
        aNrmNow     = np.asarray(self.aXtraGrid) + self.BoroCnstNat
        ShkCount    = self.TranShkValsNext.size
        aNrm_temp   = np.tile(aNrmNow,(ShkCount,1))

        # Tile arrays of the income shocks and put them into useful shapes
        aNrmCount         = aNrmNow.shape[0]
        PermShkVals_temp  = (np.tile(self.PermShkValsNext,(aNrmCount,1))).transpose()
        TranShkVals_temp  = (np.tile(self.TranShkValsNext,(aNrmCount,1))).transpose()
        ShkPrbs_temp      = (np.tile(self.ShkPrbsNext,(aNrmCount,1))).transpose()
        
        # Get cash on hand next period
        mNrmNext          = self.Rfree/(self.PermGroFac*PermShkVals_temp)*aNrm_temp + TranShkVals_temp

        # Store and report the results
        self.PermShkVals_temp  = PermShkVals_temp
        self.ShkPrbs_temp      = ShkPrbs_temp
        self.mNrmNext          = mNrmNext 
        self.aNrmNow           = aNrmNow               
        return aNrmNow


    def getGothicvP(self):
        '''
        Calculate end-of-period marginal value of assets at each point in aNrmNow
        by taking a weighted sum of next period marginal values across income
        shocks (in a preconstructed grid self.mNrmNext).
        
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


    def getSolution(self,EndOfPrdvP,aNrm,interpolator):
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

        
    def makecFuncLinear(self,mNrm,cNrm):
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
        cFuncUnc = LinearInterp(mNrm,cNrm,self.MPCminNow*self.hNrmNow,self.MPCminNow)
        return cFuncUnc
        
                
    def solve(self):
        '''
        Solves a one period consumption saving problem with risky income.
        
        Parameters
        ----------
        none
            
        Returns
        -------
        none
        '''
        aNrm       = self.prepareToGetGothicvP()           
        EndOfPrdvP = self.getGothicvP()                        
        solution   = self.getSolution(EndOfPrdvP,aNrm,self.makecFuncLinear)
        solution   = self.addMPCandHumanWealth(solution)
        return solution        
       

###############################################################################
###############################################################################

class ConsumptionSavingSolverENDG(ConsumptionSavingSolverENDGBasic):
    '''
    This class solves a single period of a standard consumption-saving problem.
    It inherits from ConsumptionSavingSolverENDGBasic, adding the ability to
    perform cubic interpolation and to calculate the value function.
    '''

    def getConsumptionCubic(self,mNrm,cNrm):
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


    def putVfuncInSolution(self,solution,EndOfPrdvP):
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
        aNrm         = self.prepareToGetGothicvP()           
        EndOfPrdvP   = self.getGothicvP()
        
        # Construct a basic solution for this period
        if self.CubicBool:
            solution   = self.getSolution(EndOfPrdvP,aNrm,interpolator=self.getConsumptionCubic)
        else:
            solution   = self.getSolution(EndOfPrdvP,aNrm,interpolator=self.makecFuncLinear)
        solution       = self.addMPCandHumanWealth(solution) # add a few things
        
        # Add the value function if requested, as well as the marginal marginal
        # value function if cubic splines were used (to prepare for next period)
        if self.vFuncBool:
            solution = self.putVfuncInSolution(solution,EndOfPrdvP)
        if self.CubicBool: 
            solution = self.addvPPfunc(solution)
        return solution        
       

def consumptionSavingSolverENDG(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
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
        solver = ConsumptionSavingSolverENDGBasic(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,
                                                  Rfree,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,
                                                  CubicBool)        
    else: # Use the "advanced" solver if either is requested
        solver = ConsumptionSavingSolverENDG(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                             PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
    solver.prepareToSolve()       # Do some preparatory work
    solution_now = solver.solve() # Solve the period
    return solution_now  


####################################################################################################
####################################################################################################

class ConsumptionSavingSolverKinkedR(ConsumptionSavingSolverENDG):
    '''
    A class to solve a single period consumption-saving problem where the interest
    rate on debt differs from the interest rate on savings.  Inherits from
    ConsumptionSavingSolverENDG, with nearly identical inputs and outputs.  The
    key difference is that Rfree is replaced by Rsave (a>0) and Rboro (a<0).
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
        assert CubicBool==False,'KinkedR will only work with linear interpolation (for now)'

        # Initialize the solver.  Most of the steps are exactly the same as in
        # the non-kinked-R basic case, so start with that.
        ConsumptionSavingSolverENDG.__init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,
                                             Rboro,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,
                                             CubicBool) 

        # Assign the interest rates as class attributes, to use them later.
        self.Rboro   = Rboro
        self.Rsave   = Rsave

    def prepareToGetGothicvP(self):
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
        # Make a grid of end-of-period assets, including *two* copies of a=0
        aNrmNow           = np.sort(np.hstack((np.asarray(self.aXtraGrid) + 
                            self.mNrmMinNow,np.array([0.0,0.0]))))
        aXtraCount        = aNrmNow.size
        
        # Make tiled versions of the assets grid and income shocks
        ShkCount          = self.TranShkValsNext.size
        aNrm_temp         = np.tile(aNrmNow,(ShkCount,1))
        PermShkVals_temp  = (np.tile(self.PermShkValsNext,(aXtraCount,1))).transpose()
        TranShkVals_temp  = (np.tile(self.TranShkValsNext,(aXtraCount,1))).transpose()
        ShkPrbs_temp      = (np.tile(self.ShkPrbsNext,(aXtraCount,1))).transpose()
        
        # Make a 1D array of the interest factor at each asset gridpoint
        Rfree_vec         = self.Rsave*np.ones(aXtraCount)
        Rfree_vec[0:(np.sum(aNrmNow<=0)-1)] = self.Rboro
        self.Rfree        = Rfree_vec
        Rfree_temp        = np.tile(Rfree_vec,(ShkCount,1))
        
        # Make an array of market resources that we could have next period,
        # considering the grid of assets and the income shocks that could occur
        mNrmNext          = Rfree_temp/(self.PermGroFac*PermShkVals_temp)*aNrm_temp + \
                            TranShkVals_temp
        
        # Recalculate the minimum MPC and human wealth using the interest factor on saving.
        # This overwrites values from setAndUpdateValues, which were based on Rboro instead.
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


def consumptionSavingSolverKinkedR(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rboro,Rsave,
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
    assert Rboro>=Rsave, 'Interest factor on debt less than interest factor on savings!'    
    
    solver = ConsumptionSavingSolverKinkedR(solution_next,IncomeDstn,LivPrb,
                                            DiscFac,CRRA,Rboro,Rsave,PermGroFac,BoroCnstArt,
                                            aXtraGrid,vFuncBool,CubicBool)
    solver.prepareToSolve()                                      
    solution = solver.solve()

    return solution                                   

###############################################################################
###############################################################################

class ConsumptionSavingSolverMarkov(ConsumptionSavingSolverENDG):
    '''
    A class to solve a single period consumption-saving problem with risky income
    and stochastic transitions between discrete states, in a Markov fashion.
    Extends ConsumptionSavingSolverENDG, with identical inputs but for a discrete
    Markov state, whose transition rule is summarized in MrkvArray.  Markov
    states can differ in their interest factor, permanent growth factor, and
    income distribution, so the inputs Rfree, PermGroFac, and IncomeDstn are
    now arrays or lists specifying those values in each (succeeding) Markov state.
    '''
    def __init__(self,solution_next,IncomeDstn_list,LivPrb,DiscFac,
                      CRRA,Rfree_list,PermGroFac_list,MrkvArray,BoroCnstArt,
                      aXtraGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver for a one period problem with risky income
        and transitions between discrete Markov states (assume there are N states).
        
        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn_list : [[np.array]]
            A length N list of income distributions in each succeeding Markov
            state.  Each income distribution contains three arrays of floats,
            representing a discrete approximation to the income process at the
            beginning of the succeeding period. Order: event probabilities,
            permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.    
        DiscFac : float
            Intertemporal discount factor for future utility.        
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree_list : np.array
            Risk free interest factor on end-of-period assets for each Markov
            state in the succeeding period.
        PermGroGac_list : float
            Expected permanent income growth factor at the end of this period
            for each Markov state in the succeeding period.
        MrkvArray : numpy.array
            An NxN array representing a Markov transition matrix between discrete
            states.  The i,j-th element of MrkvArray is the probability of
            moving from state i in period t to state j in period t+1.
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
        # Set basic attributes of the problem
        ConsumptionSavingSolverENDG.assignParameters(self,solution_next,np.nan,
                                                     LivPrb,DiscFac,CRRA,np.nan,np.nan,
                                                     BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.defineUtilityFunctions()
        
        # Set additional attributes specific to the Markov model
        self.IncomeDstn_list      = IncomeDstn_list
        self.Rfree_list           = Rfree_list
        self.PermGroFac_list      = PermGroFac_list
        self.StateCount           = len(IncomeDstn_list)
        self.MrkvArray            = MrkvArray

    def solve(self):
        '''
        Solve the one period problem of the consumption-saving model with a Markov state.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem. Includes
            a consumption function cFunc (using cubic or linear splines), a marg-
            inal value function vPfunc, a minimum acceptable level of normalized
            market resources mNrmMin, normalized human wealth hNrm, and bounding
            MPCs MPCmin and MPCmax.  It might also have a value function vFunc
            and marginal marginal value function vPPfunc.  All of these attributes
            are lists or arrays, with elements corresponding to the current
            Markov state.  E.g. solution.cFunc[0] is the consumption function
            when in the i=0 Markov state this period.
        '''
        # Find the natural borrowing constraint in each current state
        self.defBoundary()
        
        # Initialize end-of-period (marginal) value functions
        self.EndOfPrdvFunc_list  = []
        self.EndOfPrdvPfunc_list = []
        self.ExIncNext           = np.zeros(self.StateCount) + np.nan # expected income conditional on the next state
        self.WorstIncPrbAll      = np.zeros(self.StateCount) + np.nan # probability of getting the worst income shock in each next period state

        # Loop through each next-period-state and calculate the end-of-period
        # (marginal) value function
        for j in range(self.StateCount):
            # Condition values on next period's state (and record a couple for later use)
            self.conditionOnState(j)
            self.ExIncNext[j]      = np.dot(self.ShkPrbsNext,
                                            self.PermShkValsNext*self.TranShkValsNext)
            self.WorstIncPrbAll[j] = self.WorstIncPrb
            
            # Construct the end-of-period marginal value function conditional
            # on next period's state and add it to the list of value functions
            EndOfPrdvPfunc_cond = self.makeEndOfPrdvPfuncCond()
            self.EndOfPrdvPfunc_list.append(EndOfPrdvPfunc_cond)
            
            # Construct the end-of-period value functional conditional on next
            # period's state and add it to the list of value functions
            if self.vFuncBool:
                EndOfPrdvFunc_cond = self.makeEndOfPrdvFuncCond()
                self.EndOfPrdvFunc_list.append(EndOfPrdvFunc_cond)
                        
        # EndOfPrdvP_cond is EndOfPrdvP conditional on *next* period's state.
        # Take expectations to get EndOfPrdvP conditional on *this* period's state.
        self.calcEndOfPrdvP()
                
        # Calculate the bounding MPCs and PDV of human wealth for each state
        self.calcHumWealthAndBoundingMPCs()
        
        # Find consumption and market resources corresponding to each end-of-period
        # assets point for each state (and add an additional point at the lower bound)
        aNrm = np.asarray(self.aXtraGrid)[np.newaxis,:] + np.array(self.BoroCnstNat_list)[:,np.newaxis]
        self.getPointsForInterpolation(self.EndOfPrdvP,aNrm)
        cNrm = np.hstack((np.zeros((self.StateCount,1)),self.cNrmNow))
        mNrm = np.hstack((np.reshape(self.mNrmMin_list,(self.StateCount,1)),self.mNrmNow))
        
        # Package and return the solution for this period
        self.BoroCnstNat = self.BoroCnstNat_list
        solution = self.makeSolution(cNrm,mNrm)
        return solution
        
    def defBoundary(self):
        '''
        Find the borrowing constraint for each current state and save it as an
        attribute of self for use by other methods.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        self.BoroCnstNatAll          = np.zeros(self.StateCount) + np.nan
        # Find the natural borrowing constraint conditional on next period's state
        for j in range(self.StateCount):
            PermShkMinNext         = np.min(self.IncomeDstn_list[j][1])
            TranShkMinNext         = np.min(self.IncomeDstn_list[j][2])
            self.BoroCnstNatAll[j] = (self.solution_next.mNrmMin[j] - TranShkMinNext)*\
                                     (self.PermGroFac_list[j]*PermShkMinNext)/self.Rfree_list[j]

        self.BoroCnstNat_list   = np.zeros(self.StateCount) + np.nan
        self.mNrmMin_list       = np.zeros(self.StateCount) + np.nan
        self.BoroCnstDependency = np.zeros((self.StateCount,self.StateCount)) + np.nan
        # The natural borrowing constraint in each current state is the *highest*
        # among next-state-conditional natural borrowing constraints that could
        # occur from this current state.
        for i in range(self.StateCount):
            possible_next_states         = self.MrkvArray[i,:] > 0
            self.BoroCnstNat_list[i]     = np.max(self.BoroCnstNatAll[possible_next_states])
            self.mNrmMin_list[i]         = np.max([self.BoroCnstNat_list[i],self.BoroCnstArt])
            self.BoroCnstDependency[i,:] = self.BoroCnstNat_list[i] == self.BoroCnstNatAll
        # Also creates a Boolean array indicating whether the natural borrowing
        # constraint *could* be hit when transitioning from i to j.
     
    def conditionOnState(self,state_index):
        '''
        Temporarily assume that a particular Markov state will occur in the
        succeeding period, and condition solver attributes on this assumption.
        Allows the solver to construct the future-state-conditional marginal
        value function (etc) for that future state.
        
        Parameters
        ----------
        state_index : int
            Index of the future Markov state to condition on.
        
        Returns
        -------
        none
        '''
        # Set future-state-conditional values as attributes of self
        self.IncomeDstn     = self.IncomeDstn_list[state_index]
        self.Rfree          = self.Rfree_list[state_index]
        self.PermGroFac     = self.PermGroFac_list[state_index]
        self.vPfuncNext     = self.solution_next.vPfunc[state_index]
        self.mNrmMinNow     = self.mNrmMin_list[state_index]
        self.BoroCnstNat    = self.BoroCnstNatAll[state_index]        
        self.setAndUpdateValues(self.solution_next,self.IncomeDstn,self.LivPrb,self.DiscFac)

        # These lines have to come after setAndUpdateValues to override the definitions there
        self.vPfuncNext = self.solution_next.vPfunc[state_index]
        if self.CubicBool:
            self.vPPfuncNext= self.solution_next.vPPfunc[state_index]
        if self.vFuncBool:
            self.vFuncNext  = self.solution_next.vFunc[state_index]
        
    def getGothicvPP(self):
        '''
        Calculates end-of-period marginal marginal value using a pre-defined
        array of next period market resources in self.mNrmNext.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        EndOfPrdvPP = self.DiscFacEff*self.Rfree*self.Rfree*self.PermGroFac**(-self.CRRA-1.0)*\
                      np.sum(self.PermShkVals_temp**(-self.CRRA-1.0)*self.vPPfuncNext(self.mNrmNext)
                      *self.ShkPrbs_temp,axis=0)
        return EndOfPrdvPP
            
    def makeEndOfPrdvFuncCond(self):
        '''
        Construct the end-of-period value function conditional on next period's
        state.  NOTE: It might be possible to eliminate this method and replace
        it with ConsumptionSavingSolverENDG.makeEndOfPrdvFunc, but the self.X_cond
        variables must be renamed.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        EndofPrdvFunc_cond : ValueFunc
            The end-of-period value function conditional on a particular state
            occuring in the next period.            
        '''
        VLvlNext               = (self.PermShkVals_temp**(1.0-self.CRRA)*
                                  self.PermGroFac**(1.0-self.CRRA))*self.vFuncNext(self.mNrmNext)
        EndOfPrdv_cond         = self.DiscFacEff*np.sum(VLvlNext*self.ShkPrbs_temp,axis=0)
        EndOfPrdvNvrs_cond     = self.uinv(EndOfPrdv_cond)
        EndOfPrdvNvrsP_cond    = self.EndOfPrdvP_cond*self.uinvP(EndOfPrdv_cond)
        EndOfPrdvNvrs_cond     = np.insert(EndOfPrdvNvrs_cond,0,0.0)
        EndOfPrdvNvrsP_cond    = np.insert(EndOfPrdvNvrsP_cond,0,EndOfPrdvNvrsP_cond[0])
        aNrm_temp              = np.insert(self.aNrm_cond,0,self.BoroCnstNat)
        EndOfPrdvNvrsFunc_cond = CubicInterp(aNrm_temp,EndOfPrdvNvrs_cond,EndOfPrdvNvrsP_cond)
        EndofPrdvFunc_cond     = ValueFunc(EndOfPrdvNvrsFunc_cond,self.CRRA)        
        return EndofPrdvFunc_cond
        
            
    def makeEndOfPrdvPfuncCond(self):
        '''
        Construct the end-of-period marginal value function conditional on next
        period's state.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        EndofPrdvPfunc_cond : MargValueFunc
            The end-of-period marginal value function conditional on a particular
            state occuring in the succeeding period.
        '''
        # Get data to construct the end-of-period marginal value function (conditional on next state) 
        self.aNrm_cond      = self.prepareToGetGothicvP()  
        self.EndOfPrdvP_cond= self.getGothicvP()
        EndOfPrdvPnvrs_cond = self.uPinv(self.EndOfPrdvP_cond) # "decurved" marginal value
        if self.CubicBool:
            EndOfPrdvPP_cond = self.getGothicvPP()
            EndOfPrdvPnvrsP_cond = EndOfPrdvPP_cond*self.uPinvP(self.EndOfPrdvP_cond) # "decurved" marginal marginal value
        
        # Construct the end-of-period marginal value function conditional on the next state.
        if self.CubicBool:
            EndOfPrdvPnvrsFunc_cond = CubicInterp(self.aNrm_cond,EndOfPrdvPnvrs_cond,
                                                  EndOfPrdvPnvrsP_cond,lower_extrap=True)
        else:
            EndOfPrdvPnvrsFunc_cond = LinearInterp(self.aNrm_cond,EndOfPrdvPnvrs_cond,
                                                   lower_extrap=True)            
        EndofPrdvPfunc_cond = MargValueFunc(EndOfPrdvPnvrsFunc_cond,self.CRRA) # "recurve" the interpolated marginal value function
        return EndofPrdvPfunc_cond
            
    def calcEndOfPrdvP(self):
        '''
        Calculates end of period marginal value (and marginal marginal) value
        at each aXtra gridpoint for each current state, unconditional on the
        future Markov state (i.e. weighting conditional end-of-period marginal
        value by transition probabilities).
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        # Find unique values of minimum acceptable end-of-period assets (and the
        # current period states for which they apply).
        aNrmMin_unique, state_inverse = np.unique(self.BoroCnstNat_list,return_inverse=True)
        self.possible_transitions     = self.MrkvArray > 0
        
        # Calculate end-of-period marginal value (and marg marg value) at each
        # asset gridpoint for each current period state
        EndOfPrdvP                    = np.zeros((self.StateCount,self.aXtraGrid.size))
        EndOfPrdvPP                   = np.zeros((self.StateCount,self.aXtraGrid.size))
        for k in range(aNrmMin_unique.size):
            aNrmMin       = aNrmMin_unique[k]   # minimum assets for this pass
            which_states  = state_inverse == k  # the states for which this minimum applies
            aGrid         = aNrmMin + self.aXtraGrid # assets grid for this pass
            EndOfPrdvP_all  = np.zeros((self.StateCount,self.aXtraGrid.size))
            EndOfPrdvPP_all = np.zeros((self.StateCount,self.aXtraGrid.size))
            for j in range(self.StateCount):
                if np.any(np.logical_and(self.possible_transitions[:,j],which_states)): # only consider a future state if one of the relevant states could transition to it
                    EndOfPrdvP_all[j,:] = self.EndOfPrdvPfunc_list[j](aGrid)
                    if self.CubicBool: # Add conditional end-of-period (marginal) marginal value to the arrays
                        EndOfPrdvPP_all[j,:] = self.EndOfPrdvPfunc_list[j].derivative(aGrid)
            # Weight conditional marginal (marginal) values by transition probs
            # to get unconditional marginal (marginal) value at each gridpoint.
            EndOfPrdvP_temp = np.dot(self.MrkvArray,EndOfPrdvP_all)
            EndOfPrdvP[which_states,:] = EndOfPrdvP_temp[which_states,:] # only take the states for which this asset minimum applies
            if self.CubicBool:
                EndOfPrdvPP_temp = np.dot(self.MrkvArray,EndOfPrdvPP_all)
                EndOfPrdvPP[which_states,:] = EndOfPrdvPP_temp[which_states,:]
                
        # Store the results as attributes of self
        self.EndOfPrdvP = EndOfPrdvP
        if self.CubicBool:
            self.EndOfPrdvPP = EndOfPrdvPP
            
    def calcHumWealthAndBoundingMPCs(self):
        '''
        Calculates human wealth and the maximum and minimum MPC for each current
        period statem storing them as attributes of self for use by other methods.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        # Upper bound on MPC at lower m-bound
        WorstIncPrb_array = self.BoroCnstDependency*np.tile(np.reshape(self.WorstIncPrbAll,
                            (1,self.StateCount)),(self.StateCount,1))
        temp_array        = self.MrkvArray*WorstIncPrb_array
        WorstIncPrbNow    = np.sum(temp_array,axis=1) # Probability of getting the "worst" income shock and transition from each current state
        ExMPCmaxNext      = (np.dot(temp_array,self.Rfree_list**(1.0-self.CRRA)*
                            self.solution_next.MPCmax**(-self.CRRA))/WorstIncPrbNow)**\
                            (-1.0/self.CRRA)
        self.MPCmaxNow    = 1.0/(1.0 + ((self.DiscFacEff*WorstIncPrbNow)**
                            (1.0/self.CRRA))/ExMPCmaxNext)
        self.MPCmaxEff    = self.MPCmaxNow
        self.MPCmaxEff[self.BoroCnstNat_list < self.mNrmMin_list] = 1.0
        # State-conditional PDV of human wealth
        hNrmPlusIncNext   = self.ExIncNext + self.solution_next.hNrm
        self.hNrmNow      = np.dot(self.MrkvArray,(self.PermGroFac_list/self.Rfree_list)*
                            hNrmPlusIncNext)
        # Lower bound on MPC as m gets arbitrarily large
        temp              = (self.DiscFacEff*np.dot(self.MrkvArray,self.solution_next.MPCmin**
                            (-self.CRRA)*self.Rfree_list**(1.0-self.CRRA)))**(1.0/self.CRRA)
        self.MPCminNow    = 1.0/(1.0 + temp)

    def makeSolution(self,cNrm,mNrm):
        '''
        Construct an object representing the solution to this period's problem.
        
        Parameters
        ----------
        cNrm : np.array
            Array of normalized consumption values for interpolation.  Each row
            corresponds to a Markov state for this period.
        mNrm : np.array
            Array of normalized market resource values for interpolation.  Each
            row corresponds to a Markov state for this period.
        
        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem. Includes
            a consumption function cFunc (using cubic or linear splines), a marg-
            inal value function vPfunc, a minimum acceptable level of normalized
            market resources mNrmMin, normalized human wealth hNrm, and bounding
            MPCs MPCmin and MPCmax.  It might also have a value function vFunc
            and marginal marginal value function vPPfunc.  All of these attributes
            are lists or arrays, with elements corresponding to the current
            Markov state.  E.g. solution.cFunc[0] is the consumption function
            when in the i=0 Markov state this period.
        '''
        solution = ConsumerSolution() # An empty solution to which we'll add state-conditional solutions
        # Calculate the MPC at each market resource gridpoint in each state (if desired)
        if self.CubicBool:
            dcda          = self.EndOfPrdvPP/self.uPP(np.array(self.cNrmNow))
            MPC           = dcda/(dcda+1.0)
            self.MPC_temp = np.hstack((np.reshape(self.MPCmaxNow,(self.StateCount,1)),MPC))  
            interpfunc    = self.makeCubiccFunc            
        else:
            interpfunc    = self.makeLinearcFunc
        
        # Loop through each current period state and add its solution to the overall solution
        for i in range(self.StateCount):
            # Set current-period-conditional human wealth and MPC bounds
            self.hNrmNow_j   = self.hNrmNow[i]
            self.MPCminNow_j = self.MPCminNow[i]
            if self.CubicBool:
                self.MPC_temp_j  = self.MPC_temp[i,:]
                
            # Construct the consumption function by combining the constrained and unconstrained portions
            self.cFuncNowCnst = LinearInterp([self.mNrmMin_list[i], self.mNrmMin_list[i]+1.0],
                                             [0.0,1.0])
            cFuncNowUnc       = interpfunc(mNrm[i,:],cNrm[i,:])
            cFuncNow          = LowerEnvelope(cFuncNowUnc,self.cFuncNowCnst)

            # Make the marginal value function and pack up the current-state-conditional solution
            vPfuncNow     = MargValueFunc(cFuncNow,self.CRRA)
            solution_cond = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, 
                                             mNrmMin=self.mNrmMinNow)
            if self.CubicBool: # Add the state-conditional marginal marginal value function (if desired)    
                solution_cond = self.addvPPfunc(solution_cond)

            # Add the current-state-conditional solution to the overall period solution
            solution.appendSolution(solution_cond)
        
        # Add the lower bounds of market resources, MPC limits, human resources,
        # and the value functions to the overall solution
        solution.mNrmMin = self.mNrmMin_list
        solution         = self.addMPCandHumanWealth(solution)
        if self.vFuncBool:
            vFuncNow = self.makevFunc(solution)
            solution.vFunc = vFuncNow
        
        # Return the overall solution to this period
        return solution
        
    
    def makeLinearcFunc(self,mNrm,cNrm):
        '''
        Make a linear interpolation to represent the (unconstrained) consumption
        function conditional on the current period state.
        
        Parameters
        ----------
        mNrm : np.array
            Array of normalized market resource values for interpolation.
        cNrm : np.array
            Array of normalized consumption values for interpolation.
                
        Returns
        -------
        cFuncUnc: an instance of HARKinterpolation.LinearInterp
        '''
        cFuncUnc = LinearInterp(mNrm,cNrm,self.MPCminNow_j*self.hNrmNow_j,self.MPCminNow_j)
        return cFuncUnc


    def makeCubiccFunc(self,mNrm,cNrm):
        '''
        Make a cubic interpolation to represent the (unconstrained) consumption
        function conditional on the current period state.
        
        Parameters
        ----------
        mNrm : np.array
            Array of normalized market resource values for interpolation.
        cNrm : np.array
            Array of normalized consumption values for interpolation.
                
        Returns
        -------
        cFuncUnc: an instance of HARKinterpolation.CubicInterp
        '''
        cFuncUnc = CubicInterp(mNrm,cNrm,self.MPC_temp_j,self.MPCminNow_j*self.hNrmNow_j,
                               self.MPCminNow_j)
        return cFuncUnc
        
    def makevFunc(self,solution):
        '''
        Construct the value function for each current state.
        
        Parameters
        ----------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem. Must
            have a consumption function cFunc (using cubic or linear splines) as
            a list with elements corresponding to the current Markov state.  E.g.
            solution.cFunc[0] is the consumption function when in the i=0 Markov
            state this period.
            
        Returns
        -------
        vFuncNow : [ValueFunc]
            A list of value functions (defined over normalized market resources
            m) for each current period Markov state.
        '''
        vFuncNow = [] # Initialize an empty list of value functions
        # Loop over each current period state and construct the value function
        for i in range(self.StateCount):
            # Make state-conditional grids of market resources and consumption
            mNrmMin       = self.mNrmMin_list[i]
            mGrid         = mNrmMin + self.aXtraGrid
            cGrid         = solution.cFunc[i](mGrid)
            aGrid         = mGrid - cGrid
            
            # Calculate end-of-period value at each gridpoint
            EndOfPrdv_all   = np.zeros((self.StateCount,self.aXtraGrid.size))
            for j in range(self.StateCount):
                if self.possible_transitions[i,j]:
                    EndOfPrdv_all[j,:] = self.EndOfPrdvFunc_list[j](aGrid)
            EndOfPrdv     = np.dot(self.MrkvArray[i,:],EndOfPrdv_all)
            
            # Calculate (normalized) value and marginal value at each gridpoint
            vNrmNow       = self.u(cGrid) + EndOfPrdv
            vPnow         = self.uP(cGrid)
            
            # Make a "decurved" value function with the inverse utility function
            vNvrs        = self.uinv(vNrmNow) # value transformed through inverse utility
            vNvrsP       = vPnow*self.uinvP(vNrmNow)
            mNrm_temp    = np.insert(mGrid,0,mNrmMin) # add the lower bound
            vNvrs        = np.insert(vNvrs,0,0.0)
            vNvrsP       = np.insert(vNvrsP,0,self.MPCmaxEff[i]**(-self.CRRA/(1.0-self.CRRA)))
            MPCminNvrs   = self.MPCminNow[i]**(-self.CRRA/(1.0-self.CRRA))
            vNvrsFunc_i  = CubicInterp(mNrm_temp,vNvrs,vNvrsP,MPCminNvrs*self.hNrmNow[i],MPCminNvrs)
            
            # "Recurve" the decurved value function and add it to the list
            vFunc_i     = ValueFunc(vNvrsFunc_i,self.CRRA)
            vFuncNow.append(vFunc_i)
        return vFuncNow


def solveConsumptionSavingMarkov(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                                 MrkvArray,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
    '''
    Solves a single period consumption-saving problem with risky income and
    stochastic transitions between discrete states, in a Markov fashion.  Has
    identical inputs as solveConsumptionSavingENDG, except for a discrete 
    Markov transitionrule MrkvArray.  Markov states can differ in their interest 
    factor, permanent growth factor, and income distribution, so the inputs Rfree, PermGroFac, and
    IncomeDstn are arrays or lists specifying those values in each (succeeding) Markov state.
    
    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn_list : [[np.array]]
        A length N list of income distributions in each succeeding Markov
        state.  Each income distribution contains three arrays of floats,
        representing a discrete approximation to the income process at the
        beginning of the succeeding period. Order: event probabilities,
        permanent shocks, transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.    
    DiscFac : float
        Intertemporal discount factor for future utility.        
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree_list : np.array
        Risk free interest factor on end-of-period assets for each Markov
        state in the succeeding period.
    PermGroGac_list : float
        Expected permanent income growth factor at the end of this period
        for each Markov state in the succeeding period.
    MrkvArray : numpy.array
        An NxN array representing a Markov transition matrix between discrete
        states.  The i,j-th element of MrkvArray is the probability of
        moving from state i in period t to state j in period t+1.
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
    solution : ConsumerSolution
        The solution to the single period consumption-saving problem. Includes
        a consumption function cFunc (using cubic or linear splines), a marg-
        inal value function vPfunc, a minimum acceptable level of normalized
        market resources mNrmMin, normalized human wealth hNrm, and bounding
        MPCs MPCmin and MPCmax.  It might also have a value function vFunc
        and marginal marginal value function vPPfunc.  All of these attributes
        are lists or arrays, with elements corresponding to the current
        Markov state.  E.g. solution.cFunc[0] is the consumption function
        when in the i=0 Markov state this period.
    '''                                       
    solver = ConsumptionSavingSolverMarkov(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                           PermGroFac,MrkvArray,BoroCnstArt,aXtraGrid,vFuncBool,
                                           CubicBool)              
    solution_now = solver.solve()
    return solution_now             
          
# ============================================================================
# == A class for representing types of consumer agents (and things they do) ==
# ============================================================================

class ConsumerType(AgentType):
    '''
    An agent in the consumption-saving model.  His problem is defined by a sequence
    of income distributions, survival probabilities, discount factors, and permanent
    income growth rates, as well as time invariant values for risk aversion, the
    interest rate, the grid of end-of-period assets, and he is borrowing constrained.
    '''    
    
    # Define some universal values for all consumer types
    cFunc_terminal_      = LinearInterp([0.0, 1.0],[0.0,1.0]) # c=m in terminal period
    vFunc_terminal_      = LinearInterp([0.0, 1.0],[0.0,0.0]) # This is overwritten
    cFuncCnst_terminal_  = LinearInterp([0.0, 1.0],[0.0,1.0])
    solution_terminal_   = ConsumerSolution(cFunc=LowerEnvelope(cFunc_terminal_,cFuncCnst_terminal_),
                                            vFunc = vFunc_terminal_, mNrmMin=0.0, hNrm=0.0, 
                                            MPCmin=1.0, MPCmax=1.0)
    time_vary_ = ['LivPrb','DiscFac','PermGroFac']
    time_inv_ = ['CRRA','Rfree','aXtraGrid','BoroCnstArt','vFuncBool','CubicBool']
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data, and construct objects
        to be used during solution (income distribution, assets grid, etc).
        See SetupConsumerParameters.init_consumer_objects for a dictionary of
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
        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=deepcopy(ConsumerType.solution_terminal_),
                           cycles=cycles,time_flow=time_flow,pseudo_terminal=False,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary      = deepcopy(ConsumerType.time_vary_)
        self.time_inv       = deepcopy(ConsumerType.time_inv_)
        self.solveOnePeriod = consumptionSavingSolverENDG # solver can be changed depending on model
        self.update() # make income distributions, an assets grid, and update the terminal period solution
        self.a_init = np.zeros(self.Nagents) # initialize assets for simulation
        self.p_init = np.ones(self.Nagents)  # initialize permanent income for simulation

    def unpack_cFunc(self):
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
        if not ('cFunc' in self.time_vary):
            self.time_vary.append('cFunc')
            
    def makeIncShkHist(self):
        '''
        Makes histories of simulated income shocks for this consumer type by
        drawing from the discrete income distributions, storing them as attributes
        of self for use by simulation methods.  Non-Markov version.
        
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
        
        # Initialize the shock histories
        PermShkHist = np.zeros((self.sim_periods,self.Nagents)) + np.nan
        TranShkHist = np.zeros((self.sim_periods,self.Nagents)) + np.nan
        PermShkHist[0,:] = 1.0
        TranShkHist[0,:] = 1.0
        t_idx = 0
        
        # Loop through each simulated period
        for t in range(1,self.sim_periods):
            IncomeDstnNow    = self.IncomeDstn[t_idx] # set current income distribution
            PermGroFacNow    = self.PermGroFac[t_idx] # and permanent growth factor
            Events           = np.arange(IncomeDstnNow[0].size) # just a list of integers
            Cutoffs          = np.round(np.cumsum(IncomeDstnNow[0])*self.Nagents)
            top = 0
            # Make a list of event indices that closely matches the discrete income distribution
            EventList        = []
            for j in range(Events.size):
                bot = top
                top = Cutoffs[j]
                EventList += (top-bot)*[Events[j]]
            # Randomly permute the event indices and store the corresponding results
            EventDraws       = self.RNG.permutation(EventList)
            PermShkHist[t,:] = IncomeDstnNow[1][EventDraws]*PermGroFacNow # permanent "shock" includes expected growth
            TranShkHist[t,:] = IncomeDstnNow[2][EventDraws]
            # Advance the time index, looping if we've run out of income distributions
            t_idx += 1
            if t_idx >= len(self.IncomeDstn):
                t_idx = 0
        
        # Store the results as attributes of self and restore time to its original flow
        self.PermShkHist = PermShkHist
        self.TranShkHist = TranShkHist
        if not orig_time:
            self.timeRev()
            
    def makeIncShkHistMrkv(self):
        '''
        Makes histories of simulated income shocks for this consumer type by
        drawing from the discrete income distributions, respecting the Markov
        state for each agent in each period.  Should be run after makeMrkvHist().
        
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
        
        # Initialize the shock histories
        N = self.MrkvArray.shape[0]
        PermShkHist = np.zeros((self.sim_periods,self.Nagents)) + np.nan
        TranShkHist = np.zeros((self.sim_periods,self.Nagents)) + np.nan
        PermShkHist[0,:] = 1.0
        TranShkHist[0,:] = 1.0
        t_idx = 0
        
        # Draw income shocks for each simulated period, respecting the Markov state
        for t in range(1,self.sim_periods):
            MrkvNow = self.MrkvHist[t,:]
            IncomeDstn_list    = self.IncomeDstn[t_idx]
            PermGroFac_list    = self.PermGroFac[t_idx]
            for n in range(N):
                these = MrkvNow == n
                IncomeDstnNow = IncomeDstn_list[n]
                PermGroFacNow = PermGroFac_list[n]
                Events           = np.arange(IncomeDstnNow[0].size) # just a list of integers
                Cutoffs          = np.round(np.cumsum(IncomeDstnNow[0])*np.sum(these))
                top = 0
                # Make a list of event indices that closely matches the discrete income distribution
                EventList        = []
                for j in range(Events.size):
                    bot = top
                    top = Cutoffs[j]
                    EventList += (top-bot)*[Events[j]]
                # Randomly permute the event indices and store the corresponding results
                EventDraws       = self.RNG.permutation(EventList)
                PermShkHist[t,these] = IncomeDstnNow[1][EventDraws]*PermGroFacNow
                TranShkHist[t,these] = IncomeDstnNow[2][EventDraws]
            # Advance the time index, looping if we've run out of income distributions
            t_idx += 1
            if t_idx >= len(self.IncomeDstn):
                t_idx = 0
        
        # Store the results as attributes of self and restore time to its original flow        
        self.PermShkHist = PermShkHist
        self.TranShkHist = TranShkHist
        if not orig_time:
            self.timeRev()
        
            
    def makeMrkvHist(self):
        '''
        Makes a history of simulated discrete Markov states, starting from the
        initial states in markov_init.  Assumes that MrkvArray is constant.

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
        
        # Initialize the Markov state history
        MrkvHist = np.zeros((self.sim_periods,self.Nagents),dtype=int)
        MrkvNow = self.Mrkv_init
        MrkvHist[0,:] = MrkvNow
        base_draws = np.arange(self.Nagents,dtype=float)/self.Nagents + 1.0/(2*self.Nagents)
        
        # Make an array of Markov transition cutoffs
        N = self.MrkvArray.shape[0] # number of states
        Cutoffs = np.cumsum(self.MrkvArray,axis=1)
        
        # Draw Markov transitions for each period
        for t in range(1,self.sim_periods):
            draws_now = self.RNG.permutation(base_draws)
            MrkvNext = np.zeros(self.Nagents) + np.nan
            for n in range(N):
                these = MrkvNow == n
                MrkvNext[these] = np.searchsorted(Cutoffs[n,:],draws_now[these])
            MrkvHist[t,:] = MrkvNext
            MrkvNow = MrkvNext
        
        # Store the results and return time to its original flow
        self.MrkvHist = MrkvHist
        if not orig_time:
            self.timeRev()
            
    def updateIncomeProcess(self):
        '''
        Updates this agent's income process based on his own attributes.  The
        function that generates the discrete income process can be swapped out
        for a different process.
        
        Parameters
        ----------
        none
        
        Returns:
        -----------
        none
        '''
        original_time = self.time_flow
        self.timeFwd()
        IncomeDstn = constructLognormalIncomeProcessUnemployment(self)
        self.IncomeDstn = IncomeDstn
        if not 'IncomeDstn' in self.time_vary:
            self.time_vary.append('IncomeDstn')
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
        self.solution_terminal.vFunc   = ValueFunc(self.solution_terminal.cFunc,self.CRRA)
        self.solution_terminal.vPfunc  = MargValueFunc(self.solution_terminal.cFunc,self.CRRA)
        self.solution_terminal.vPPfunc = MargMargValueFunc(self.solution_terminal.cFunc,self.CRRA)
        
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
        self.RfreeNow = self.Rfree
        
        # Initialize the history arrays
        self.aNow     = a_init
        self.pNow     = p_init
        self.RfreeNow = self.Rfree
        blank_history = np.zeros((sim_prds,self.Nagents)) + np.nan
        self.pHist    = copy(blank_history)
        self.bHist    = copy(blank_history)
        self.mHist    = copy(blank_history)
        self.cHist    = copy(blank_history)
        self.MPChist  = copy(blank_history)
        self.aHist    = copy(blank_history)
        
    def simConsHistory(self):
        '''
        Simulates a history of bank balances, market resources, consumption,
        marginal propensity to consume, assets (after all actions), and permanent
        income giveninitial assets (normalized by permanent income).
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        orig_time = self.time_flow
        self.timeFwd()
        
        # Simulate a history of this consumer type
        for t in range(self.aHist.shape[0]):
            self.advanceIncShks()
            self.advancecFunc()
            self.simOnePrd()
            self.pHist[t,:] = self.pNow
            self.bHist[t,:] = self.bNow
            self.mHist[t,:] = self.mNow
            self.cHist[t,:] = self.cNow
            self.MPChist[t,:] = self.MPCnow
            self.aHist[t,:] = self.aNow
            
        # Restore the original flow of time
        if not orig_time:
            self.timeRev()
                
    def simOnePrd(self):
        '''
        Simulate a single period of a consumption-saving model with permanent
        and transitory income shocks.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        if self.solveOnePeriod is solveConsumptionSavingMarkov:
            is_markov = True
            N = self.MrkvArray.shape[0]
        else:
            is_markov = False
        
        # Unpack objects from self for convenience
        aPrev          = self.aNow
        pPrev          = self.pNow
        TranShkNow     = self.TranShkNow
        PermShkNow     = self.PermShkNow
        if is_markov:
            RfreeNow   = self.RfreeNow[self.MrkvNow]
        else:
            RfreeNow   = self.RfreeNow
        cFuncNow       = self.cFuncNow
        
        # Simulate the period
        pNow    = pPrev*PermShkNow      # Updated permanent income level
        ReffNow = RfreeNow/PermShkNow   # "effective" interest factor on normalized assets
        bNow    = ReffNow*aPrev         # Bank balances before labor income
        mNow    = bNow + TranShkNow     # Market resources after income
        if is_markov:
            cNow = np.zeros_like(mNow)
            MPCnow = np.zeros_like(mNow)
            for n in range(N):
                these = self.MrkvNow == n
                cNow[these], MPCnow[these] = cFuncNow[n].eval_with_derivative(mNow[these])
        else:
            cNow,MPCnow = cFuncNow.eval_with_derivative(mNow) # Consumption and maginal propensity to consume
        aNow    = mNow - cNow           # Assets after all actions are accomplished
        
        # Store the new state and control variables
        self.pNow   = pNow
        self.bNow   = bNow
        self.mNow   = mNow
        self.cNow   = cNow
        self.MPCnow = MPCnow
        self.aNow   = aNow
        
        
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
        if self.solveOnePeriod is solveConsumptionSavingMarkov:
            self.MrkvNow = self.MrkvHist[self.Shk_idx,:]
        self.PermShkNow = self.PermShkHist[self.Shk_idx]
        self.TranShkNow = self.TranShkHist[self.Shk_idx]
        self.Shk_idx += 1
        if self.Shk_idx >= self.PermShkHist.shape[0]:
            self.Shk_idx = 0 # Reset to zero if we've run out of shocks
            
    def advancecFunc(self):
        '''
        Advance the consumption function to the next period in the solution.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        self.cFuncNow  = self.solution[self.cFunc_idx].cFunc
        self.cFunc_idx += 1
        if self.cFunc_idx >= len(self.solution):
            self.cFunc_idx = 0 # Reset to zero if we've run out of cFuncs
                
    def calcBoundingValues(self):
        '''
        Calculate the PDV of human wealth (after receiving income this period)
        in an infinite horizon model with only one period repeated indefinitely.
        Also calculates MPCmin and MPCmax.  Outputs are np.array if the model
        has a Markov state process.
        
        THIS IS BROKEN AND NEEDS FIXING
        
        Parameters
        ----------
        none
        
        Returns
        -------
        hNrm : float or np.array
            Human wealth, the present discounted value of expected future income
            after receiving income this period, ignoring mortality.  
        MPCmax : float
            Upper bound on the marginal propensity to consume as m --> mNrmMin.
        MPCmin : float
            Lower bound on the marginal propensity to consume as m --> infty.
        '''
        assert False, 'calcBoundingValues IS BROKEN AND NEEDS FIXING'
        if hasattr(self,'MrkvArray'):
            StateCount = self.IncomeDstn[0].size
            ExIncNext = np.zeros(StateCount) + np.nan
            for j in range(StateCount):
                PermShkValsNext = self.IncomeDstn[0][j][1]
                TranShkValsNext = self.IncomeDstn[0][j][2]
                ShkPrbsNext     = self.IncomeDstn[0][j][0]
                ExIncNext[j] = np.dot(ShkPrbsNext,PermShkValsNext*TranShkValsNext)                
            hNrm        = np.dot(np.dot(np.linalg.inv((self.Rfree/self.PermGroFac[0])*np.eye(StateCount) -
                              self.MrkvArray),self.MrkvArray),ExIncNext)
            
            p_zero_income_now = np.dot(self.MrkvArray,self.p_zero_income[0])
            PatFac            = (self.DiscFac[0]*self.Rfree)**(1.0/self.CRRA)/self.Rfree
            MPCmax            = 1.0 - p_zero_income_now**(1.0/self.CRRA)*PatFac # THIS IS WRONG
            
        else:
            PermShkValsNext   = self.IncomeDstn[0][1]
            TranShkValsNext   = self.IncomeDstn[0][2]
            ShkPrbsNext       = self.IncomeDstn[0][0]
            ExIncNext         = np.dot(ShkPrbsNext,PermShkValsNext*TranShkValsNext)
            hNrm              = (ExIncNext*self.PermGroFac[0]/self.Rfree)/(1.0-self.PermGroFac[0]/self.Rfree)
            
            PatFac    = (self.DiscFac[0]*self.Rfree)**(1.0/self.CRRA)/self.Rfree
            MPCmax    = 1.0 - self.p_zero_income[0]**(1.0/self.CRRA)*PatFac
        
        MPCmin = 1.0 - PatFac
        return hNrm, MPCmax, MPCmin

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
    T_total :  int
        Total number of non-terminal periods in the consumer's sequence of periods.

    Returns
    -------
    IncomeDstn:  [[np.array]]
        A list with T_total elements, each of which is a list of three arrays
        representing a discrete approximation to the income process in a period.
        Order: probabilities, permanent shocks, transitory shocks.
    '''
    # Unpack the parameters from the input
    PermShkStd    = parameters.PermShkStd
    PermShkCount  = parameters.PermShkCount
    TranShkStd    = parameters.TranShkStd
    TranShkCount  = parameters.TranShkCount
    T_total       = parameters.T_total
    T_retire      = parameters.T_retire
    UnempPrb      = parameters.UnempPrb
    IncUnemp      = parameters.IncUnemp
    UnempPrbRet   = parameters.UnempPrbRet        
    IncUnempRet   = parameters.IncUnempRet
    
    IncomeDstn    = [] # Discrete approximations to income process in each period

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
    for t in range(T_total): # Iterate over all periods, counting forward

        if T_retire > 0 and t >= T_retire:
            # Then we are in the "retirement period" and add a retirement income object.
            IncomeDstn.append(deepcopy(IncomeDstnRet))
        else:
            # We are in the "working life" periods.
            TranShkDstn     = approxLognormal(N=TranShkCount, sigma=TranShkStd[t], tail_N=0)
            if UnempPrb > 0:
                TranShkDstn = addDiscreteOutcomeConstantMean(TranShkDstn, p=UnempPrb, x=IncUnemp)
            PermShkDstn     = approxLognormal(N=PermShkCount, sigma=PermShkStd[t], tail_N=0)
            IncomeDstn.append(combineIndepDstns(PermShkDstn,TranShkDstn)) # mix the independent distributions
    return IncomeDstn
    

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
    Constructs the grid of post-decision states, representing end-of-period assets.

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
    grid_type:              string
        String indicating the type of grid. "linear" or "exp_mult"
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
    aXtraCount    = parameters.aXtraCount
    aXtraExtra   = parameters.aXtraExtra
    grid_type = 'exp_mult'
    exp_nest  = parameters.exp_nest
    
    # Set up post decision state grid:
    aXtraGrid = None
    if grid_type == "linear":
        aXtraGrid = np.linspace(aXtraMin, aXtraMax, aXtraCount)
    elif grid_type == "exp_mult":
        aXtraGrid = makeGridExpMult(ming=aXtraMin, maxg=aXtraMax, ng=aXtraCount, timestonest=exp_nest)
    else:
        raise Exception, "grid_type not recognized in __init__." + \
                         "Please ensure grid_type is 'linear' or 'exp_mult'"

    # Add in additional points for the grid:
    for a in aXtraExtra:
        if (a is not None):
            if a not in aXtraGrid:
                j      = aXtraGrid.searchsorted(a)
                aXtraGrid = np.insert(aXtraGrid, j, a)

    return aXtraGrid

####################################################################################################     
    
if __name__ == '__main__':
    import SetupConsumerParameters as Params
    from HARKutilities import plotFunc, plotFuncDer, plotFuncs
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)

    do_markov_type          = True
    do_perfect_foresight    = True
    do_simulation           = True

####################################################################################################    
    
    # Make and solve a finite consumer type
    LifecycleType = ConsumerType(**Params.init_consumer_objects)
    LifecycleType.solveOnePeriod = consumptionSavingSolverENDG
    
    start_time = clock()
    LifecycleType.solve()
    end_time = clock()
    print('Solving a lifecycle consumer took ' + mystr(end_time-start_time) + ' seconds.')
    LifecycleType.unpack_cFunc()
    LifecycleType.timeFwd()
    
    # Plot the consumption functions during working life
    print('Consumption functions while working:')
    mMin = min([LifecycleType.solution[t].mNrmMin for t in range(40)])
    plotFuncs(LifecycleType.cFunc[:40],mMin,5)

    # Plot the consumption functions during retirement
    print('Consumption functions while retired:')
    plotFuncs(LifecycleType.cFunc[40:],0,5)
    LifecycleType.timeRev()
    
    # Simulate some data
    if do_simulation:
        LifecycleType.sim_periods = LifecycleType.T_total + 1
        LifecycleType.makeIncShkHist()
        LifecycleType.initializeSim()
        LifecycleType.simConsHistory()
    
####################################################################################################    
    
    
    # Make and solve an infinite horizon consumer
    InfiniteType = deepcopy(LifecycleType)
    InfiniteType.assignParameters(    LivPrb = [0.98],
                                      DiscFac = [0.96],
                                      PermGroFac = [1.01],
                                      cycles = 0) # This is what makes the type infinite horizon
    InfiniteType.IncomeDstn = [LifecycleType.IncomeDstn[-1]]
    
    start_time = clock()
    InfiniteType.solve()
    end_time = clock()
    print('Solving an infinite horizon consumer took ' + mystr(end_time-start_time) + ' seconds.')
    InfiniteType.timeFwd()
    InfiniteType.unpack_cFunc()
    
    # Plot the consumption function and MPC for the infinite horizon consumer
    print('Consumption function:')
    plotFunc(InfiniteType.cFunc[0],InfiniteType.solution[0].mNrmMin,5)    # plot consumption
    print('Marginal consumption function:')
    plotFuncDer(InfiniteType.cFunc[0],InfiniteType.solution[0].mNrmMin,5) # plot MPC
    if InfiniteType.vFuncBool and not do_perfect_foresight:
        print('Value function:')
        plotFunc(InfiniteType.solution[0].vFunc,InfiniteType.solution[0].mNrmMin+0.5,10)
        
    if do_simulation:
        InfiniteType.sim_periods = 120
        InfiniteType.makeIncShkHist()
        InfiniteType.initializeSim()
        InfiniteType.simConsHistory()


#################################################################################################### 

    if do_perfect_foresight:
        # Make and solve a perfect foresight consumer type
        PerfectForesightType = deepcopy(InfiniteType)    
        PerfectForesightType.solveOnePeriod = solvePerfForesight
        
        start_time = clock()
        PerfectForesightType.solve()
        end_time = clock()
        print('Solving a perfect foresight consumer took ' + mystr(end_time-start_time) + 
              ' seconds.')
        PerfectForesightType.unpack_cFunc()
        PerfectForesightType.timeFwd()
        
        print('Consumption functions for perfect foresight vs risky income:')            
        plotFuncs([PerfectForesightType.cFunc[0],InfiniteType.cFunc[0]],
                  InfiniteType.solution[0].mNrmMin,100)
        if InfiniteType.vFuncBool:
            print('Value functions for perfect foresight vs risky income:')
            plotFuncs([PerfectForesightType.solution[0].vFunc,InfiniteType.solution[0].vFunc],
                      InfiniteType.solution[0].mNrmMin+0.5,10)
            
    
####################################################################################################    


    # Make and solve an agent with a kinky interest rate
    KinkyType = deepcopy(InfiniteType)

    KinkyType.time_inv.remove('Rfree')
    KinkyType.time_inv += ['Rboro','Rsave']
    KinkyType(Rboro = 1.2, Rsave = 1.03, BoroCnstArt = None, aXtraCount = 48, cycles=0, 
              CubicBool = False)

    KinkyType.solveOnePeriod = consumptionSavingSolverKinkedR
    KinkyType.updateAssetsGrid()
    
    start_time = clock()
    KinkyType.solve()
    end_time = clock()
    print('Solving a kinky consumer took ' + mystr(end_time-start_time) + ' seconds.')
    KinkyType.unpack_cFunc()
    print('Kinky consumption function:')
    KinkyType.timeFwd()
    plotFunc(KinkyType.cFunc[0],KinkyType.solution[0].mNrmMin,5)

    if do_simulation:
        KinkyType.sim_periods = 120
        KinkyType.makeIncShkHist()
        KinkyType.initializeSim()
        KinkyType.simConsHistory()
    
####################################################################################################    


    
    # Make and solve a "cyclical" consumer type who lives the same four quarters repeatedly.
    # The consumer has income that greatly fluctuates throughout the year.
    CyclicalType = deepcopy(LifecycleType)
    CyclicalType.assignParameters(LivPrb = [0.98]*4,
                                      DiscFac = [0.96]*4,
                                      PermGroFac = [1.1, 0.3, 2.8, 1.082251],
                                      cycles = 0) # This is what makes the type (cyclically) infinite horizon)
    CyclicalType.IncomeDstn = [LifecycleType.IncomeDstn[-1]]*4
    
    start_time = clock()
    CyclicalType.solve()
    end_time = clock()
    print('Solving a cyclical consumer took ' + mystr(end_time-start_time) + ' seconds.')
    CyclicalType.unpack_cFunc()
    CyclicalType.timeFwd()
    
    # Plot the consumption functions for the cyclical consumer type
    print('Quarterly consumption functions:')
    mMin = min([X.mNrmMin for X in CyclicalType.solution])
    plotFuncs(CyclicalType.cFunc,mMin,5)
    
    if do_simulation:
        CyclicalType.sim_periods = 480
        CyclicalType.makeIncShkHist()
        CyclicalType.initializeSim()
        CyclicalType.simConsHistory()
    
    
####################################################################################################    
    
    

    # Make and solve a type that has serially correlated unemployment   
    if do_markov_type:
        # Define the Markov transition matrix
        unemp_length = 5
        urate_good = 0.05
        urate_bad = 0.12
        bust_prob = 0.01
        recession_length = 20
        p_reemploy =1.0/unemp_length
        p_unemploy_good = p_reemploy*urate_good/(1-urate_good)
        p_unemploy_bad = p_reemploy*urate_bad/(1-urate_bad)
        boom_prob = 1.0/recession_length
        MrkvArray = np.array([[(1-p_unemploy_good)*(1-bust_prob),p_unemploy_good*(1-bust_prob),
                               (1-p_unemploy_good)*bust_prob,p_unemploy_good*bust_prob],
                              [p_reemploy*(1-bust_prob),(1-p_reemploy)*(1-bust_prob),
                               p_reemploy*bust_prob,(1-p_reemploy)*bust_prob],
                              [(1-p_unemploy_bad)*boom_prob,p_unemploy_bad*boom_prob,
                               (1-p_unemploy_bad)*(1-boom_prob),p_unemploy_bad*(1-boom_prob)],
                              [p_reemploy*boom_prob,(1-p_reemploy)*boom_prob,
                               p_reemploy*(1-boom_prob),(1-p_reemploy)*(1-boom_prob)]])
        
        MarkovType = deepcopy(InfiniteType)
        xi_dist = approxMeanOneLognormal(MarkovType.TranShkCount, 0.1)
        psi_dist = approxMeanOneLognormal(MarkovType.PermShkCount, 0.1)
        employed_income_dist = combineIndepDstns(psi_dist, xi_dist)
        employed_income_dist = [np.ones(1),np.ones(1),np.ones(1)]
        unemployed_income_dist = [np.ones(1),np.ones(1),np.zeros(1)]
        
        MarkovType.solution_terminal.cFunc = 4*[MarkovType.solution_terminal.cFunc]
        MarkovType.solution_terminal.vFunc = 4*[MarkovType.solution_terminal.vFunc]
        MarkovType.solution_terminal.vPfunc = 4*[MarkovType.solution_terminal.vPfunc]
        MarkovType.solution_terminal.vPPfunc = 4*[MarkovType.solution_terminal.vPPfunc]
        MarkovType.solution_terminal.mNrmMin = 4*[MarkovType.solution_terminal.mNrmMin]
        MarkovType.solution_terminal.MPCmax = np.array(4*[1.0])
        MarkovType.solution_terminal.MPCmin = np.array(4*[1.0])
        
        MarkovType.Rfree = np.array(4*[MarkovType.Rfree])
        MarkovType.PermGroFac = [np.array(4*MarkovType.PermGroFac)]
        
        MarkovType.IncomeDstn = [[employed_income_dist,unemployed_income_dist,employed_income_dist,
                                  unemployed_income_dist]]
        MarkovType.MrkvArray = MrkvArray
        MarkovType.time_inv.append('MrkvArray')
        MarkovType.solveOnePeriod = solveConsumptionSavingMarkov
        MarkovType.cycles = 0        
        #MarkovType.vFuncBool = False
        
        MarkovType.timeFwd()
        start_time = clock()
        MarkovType.solve()
        end_time = clock()
        print('Solving a Markov consumer took ' + mystr(end_time-start_time) + ' seconds.')
        print('Consumption functions for each discrete state:')
        plotFuncs(MarkovType.solution[0].cFunc,0,50)
        if MarkovType.vFuncBool:
            print('Value functions for each discrete state:')
            plotFuncs(MarkovType.solution[0].vFunc,5,50)

        if do_simulation:
            MarkovType.Mrkv_init = np.zeros(MarkovType.Nagents,dtype=int)
            MarkovType.makeMrkvHist()
            MarkovType.makeIncShkHistMrkv()
            MarkovType.initializeSim()
            MarkovType.simConsHistory()
