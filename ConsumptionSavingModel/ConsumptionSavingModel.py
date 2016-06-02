# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. 
import sys 
sys.path.insert(0,'../')

import numpy as np
from HARKcore import AgentType, Solution, NullFunc
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKutilities import approxLognormal, approxMeanOneLognormal, addDiscreteOutcomeConstantMean, combineIndepDstns, makeGridExpMult, CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv, CRRAutilityP_invP
from HARKinterpolation import CubicInterp, LowerEnvelope, LinearInterp
from HARKsimulation import drawMeanOneLognormal, drawBernoulli
from copy import copy, deepcopy

utility      = CRRAutility
utilityP     = CRRAutilityP
utilityPP    = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv  = CRRAutility_inv
utilityP_invP= CRRAutilityP_invP

# =====================================================================
# === Classes and functions used to solve consumption-saving models ===
# =====================================================================

class ConsumerSolution(Solution):
    '''
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function and marginal
    value function.
    '''
    def __init__(self, cFunc=NullFunc, vFunc=NullFunc, 
                       vPfunc=NullFunc, vPPfunc=NullFunc,
                       mNrmMin=None, hNrm=None, MPCmin=None, MPCmax=None):
        '''
        The constructor for a new ConsumerSolution object.
        
        Parameters:
        ------------
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
            
        Returns:
        ----------
        new instance of ConsumerSolution        
        '''
        self.cFunc        = cFunc
        self.vFunc        = vFunc
        self.vPfunc       = vPfunc
        self.vPPfunc      = vPPfunc
        self.mNrmMin      = mNrmMin
        self.hNrm         = hNrm
        self.MPCmin       = MPCmin
        self.MPCmax       = MPCmax
        self.convergence_criteria = ['cFunc']

    def getEulerEquationErrorFunction(self,uPfunc):
        '''
        Return the Euler Equation Error function, to check that the solution is
        "good enough".  Note right now this method needs to be passed uPfunc,
        which I find awkward and annoying.
        
        Parameters:
        ------------
        uPunc : function
            The instantaneous marginal utility function.
            
        Returns:
        ----------
        eulerEquationErrorFunction : function
            Function yielding the absolute difference between marginal utility
            and end-of-period marginal value.
        '''
        def eulerEquationErrorFunction(m):
            return np.abs(uPfunc(self.cFunc(m)) - self.EndOfPrdvPfunc(m))            
        return eulerEquationErrorFunction
            
    def appendSolution(self,new_solution):
        '''
        Appends one solution to another to create a ConsumerSolution whose
        attributes are lists.  Used in solveConsumptionSavingMarkov.
        
        Parameters:
        ------------
        new_solution : ConsumerSolution
            The solution to a consumption-saving problem conditional on being
            in a particular Markov state to begin the period.
            
        Returns:
        ----------
        none
        '''
        if type(self.cFunc)!=list:
            assert self.cFunc==NullFunc            
            # Then the assumption is self is an empty initialized instance, we need to start a list
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
        
        Parameters:
        ------------
        vFuncDecurved : function
            A real function representing the value function composed with the
            inverse utility function, defined on market resources: u_inv(vFunc(m))
        CRRA : float
            Coefficient of relative risk aversion.
            
        Returns:
        ----------
        new instance of ValueFunc
        '''
        self.func = deepcopy(vFuncDecurved)
        self.CRRA = CRRA
        
    def __call__(self,m):
        '''
        Evaluate the value function at given levels of market resources m.
        
        Parameters:
        ------------
        m : float or np.array
            Market resources (normalized by permanent income) whose value is to
            be found.
            
        Returns:
        ----------
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
        
        Parameters:
        ------------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources: uP_inv(vPfunc(m)).  Called cFunc because when standard
            envelope condition applies, uP_inv(vPfunc(m)) = cFunc(m).
        CRRA : float
            Coefficient of relative risk aversion.
            
        Returns:
        ----------
        new instance of MargValueFunc
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        
    def __call__(self,m):
        '''
        Evaluate the marginal value function at given levels of market resources m.
        
        Parameters:
        ------------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            value is to be found.
            
        Returns:
        ----------
        vP : float or np.array
            Marginal lifetime value of beginning this period with market
            resources m; has same size as input m.
        '''
        return utilityP(self.cFunc(m),gam=self.CRRA)
        
    def derivative(self,m):
        '''
        Evaluate the derivative of the marginal value function at given levels
        of market resources m; this is the marginal marginal value function.
        
        Parameters:
        ------------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            marginal value is to be found.
            
        Returns:
        ----------
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
        
        Parameters:
        ------------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources: uP_inv(vPfunc(m)).  Called cFunc because when standard
            envelope condition applies, uP_inv(vPfunc(m)) = cFunc(m).
        CRRA : float
            Coefficient of relative risk aversion.
            
        Returns:
        ----------
        new instance of MargMargValueFunc
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        
    def __call__(self,m):
        '''
        Evaluate the marginal marginal value function at given levels of market
        resources m.
        
        Parameters:
        ------------
        m : float or np.array
            Market resources (normalized by permanent income) whose marginal
            marginal value is to be found.
            
        Returns:
        ----------
        vPP : float or np.array
            Marginal lifetime value of beginning this period with market
            resources m; has same size as input m.
        '''
        c, MPC = self.cFunc.eval_with_derivative(m)
        return MPC*utilityPP(c,gam=self.CRRA)


####################################################################################################
####################################################################################################

class PerfectForesightSolver(object):
    '''
    A class for solving a one period perfect foresight consumption-saving problem.
    An instance of this class is created by solvePerfForesight in each period.
    '''
    def __init__(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac):
        '''
        Constructor for a new PerfectForesightSolver.
        
        Parameters:
        -------------
        solution_next : ConsumerSolution
            The solution to the succeeding one period problem.
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
            
        Returns:
        ----------
        new instance of PerfectForesightSolver        
        '''
        self.notation = {'a': 'assets after all actions',
                         'm': 'market resources at decision time',
                         'c': 'consumption'}
        self.assignParameters(solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac)
         
    def assignParameters(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac):
        '''
        Saves necessary parameters as attributes of self for use by other methods.
        
        Parameters:
        -------------
        solution_next : ConsumerSolution
            The solution to the succeeding one period problem.
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
            
        Returns:
        ----------
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
        
        Parameters:
        ------------
        none
        
        Returns:
        ----------
        none
        '''
        self.u   = lambda c : utility(c,gam=self.CRRA)  # utility function
        self.uP  = lambda c : utilityP(c,gam=self.CRRA) # marginal utility function
        self.uPP = lambda c : utilityPP(c,gam=self.CRRA)# marginal marginal utility function

    def defValueFuncs(self):
        '''
        Defines the value and marginal value function for this period.
        
        Parameters:
        ------------
        none
        
        Returns:
        ----------
        none
        '''
        MPCnvrs = self.MPC**(-self.CRRA/(1.0-self.CRRA))
        vFuncNvrs = LinearInterp(np.array([self.mNrmMin, self.mNrmMin+1.0]),np.array([0.0, MPCnvrs]))
        self.vFunc   = ValueFunc(vFuncNvrs,self.CRRA)
        self.vPfunc  = MargValueFunc(self.cFunc,self.CRRA)
        
    def makecFuncPF(self):
        '''
        Makes the (linear) consumption function for this period.
        
        Parameters:
        ------------
        none
        
        Returns:
        ----------
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
        
        Parameters:
        ------------
        none
        
        Returns:
        ----------
        solution : ConsumerSolution
            The solution to this period's problem.
        '''
        self.defUtilityFuncs()
        self.DiscFacEff = self.DiscFac*self.LivPrb
        self.makecFuncPF()
        self.defValueFuncs()
        solution = ConsumerSolution(cFunc=self.cFunc, vFunc=self.vFunc, 
                       vPfunc=self.vPfunc,
                       mNrmMin=self.mNrmMin, hNrm=self.hNrmNow,
                       MPCmin=self.MPC, MPCmax=self.MPC)
        return solution


def solvePerfForesight(solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac):
    '''
    Solves a single period consumption-saving problem for a consumer with perfect foresight.
    
    Parameters:
    -------------
    solution_next : ConsumerSolution
        The solution to the succeeding one period problem.
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
        
    Returns:
    ----------
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
        Constructor for a new solver for problems with risky income.
        
        Parameters:
        -------------
        solution_next : ConsumerSolution
            The solution to the succeeding one period problem.
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
                        
        Returns:
        ----------
        new instance of SetupImperfectForesightSolver  
        '''
        self.assignParameters(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.defineUtilityFunctions()

    def assignParameters(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
        '''
        Assigns period parameters as attributes of self for use by other methods
        
        Parameters:
        -------------
        solution_next : ConsumerSolution
            The solution to the succeeding one period problem.
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
                        
        Returns:
        ----------
        none
        '''
        PerfectForesightSolver.assignParameters(self,solution_next,DiscFac,LivPrb,CRRA,Rfree,PermGroFac)
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
        
        Parameters:
        ------------
        none
        
        Returns:
        ----------
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
        
        Parameters:
        ------------
        solution_next : ConsumerSolution
            The solution to the succeeding one period problem.
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
            
        Returns:
        ----------
        none
        '''
        self.DiscFacEff       = DiscFac*LivPrb # "effective" discount factor
        self.ShkPrbsNext      = IncomeDstn[0]
        self.PermShkValsNext  = IncomeDstn[1]
        self.TranShkValsNext  = IncomeDstn[2]
        self.PermShkMinNext   = np.min(self.PermShkValsNext)    
        self.TranShkMinNext   = np.min(self.TranShkValsNext)
        self.vPfuncNext       = solution_next.vPfunc        
        self.WorstIncPrb      = np.sum(self.ShkPrbsNext[(self.PermShkValsNext*self.TranShkValsNext)==(self.PermShkMinNext*self.TranShkMinNext)]) 

        if self.CubicBool:
            self.vPPfuncNext  = solution_next.vPPfunc
            
        if self.vFuncBool:
            self.vFuncNext    = solution_next.vFunc
            
        # Update the bounding MPCs and PDV of human wealth:
        self.PatFac       = ((self.Rfree*self.DiscFacEff)**(1.0/self.CRRA))/self.Rfree
        self.MPCminNow    = 1.0/(1.0 + self.PatFac/solution_next.MPCmin)
        self.hNrmNow      = self.PermGroFac/self.Rfree*(np.dot(self.ShkPrbsNext,self.TranShkValsNext*self.PermShkValsNext) + solution_next.hNrm)
        self.MPCmaxNow    = 1.0/(1.0 + (self.WorstIncPrb**(1.0/self.CRRA))*self.PatFac/solution_next.MPCmax)


    def defineBorrowingConstraint(self,BoroCnstArt):
        '''
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Uses the artificial and natural borrowing constraints.
        
        Parameters:
        -------------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.
            
        Returns:
        ----------
        none
        '''
        # Calculate the minimum allowable value of money resources in this period
        self.BoroCnstNat = (self.solution_next.mNrmMin - self.TranShkMinNext)*(self.PermGroFac*self.PermShkMinNext)/self.Rfree
        self.mNrmMinNow = np.max([self.BoroCnstNat,BoroCnstArt])
        if self.BoroCnstNat < self.mNrmMinNow: 
            self.MPCmaxEff = 1.0 # If actually constrained, MPC near limit is 1
        else:
            self.MPCmaxEff = self.MPCmaxNow
    
        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(np.array([self.mNrmMinNow, self.mNrmMinNow+1]), np.array([0.0, 1.0]))


    def prepareToSolve(self):
        '''
        Perform preparatory work before calculating the unconstrained consumption
        function.
        
        Parameters:
        ------------
        none
        
        Returns:
        ------------
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
        
        Parameters:
        ------------
        none
        
        Returns:
        ------------
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
        
        Parameters:
        ------------
        none
        
        Returns:
        ------------
        none
        '''        
        EndOfPrdvP  = self.DiscFacEff*self.Rfree*self.PermGroFac**(-self.CRRA)*np.sum(
                    self.PermShkVals_temp**(-self.CRRA)*self.vPfuncNext(self.mNrmNext)*self.ShkPrbs_temp,axis=0)                    
        return EndOfPrdvP
                    

    def getPointsForInterpolation(self,EndOfPrdvP,aNrmNow):
        '''
        Finds interpolation points for the consumption function (m,c).
        
        Parameters:
        ------------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrmNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
            
        Returns:
        ---------
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
        
        Parameters:
        ------------
        cNrm : np.array
            Consumption points for interpolation.
        mNrm : np.array
            Corresponding market resource points for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.
            
        Returns:
        ----------
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
        
        Parameters:
        ------------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrm : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
            
        interpolator : function
            A function that constructs and returns a consumption function.
            
        Returns:
        ----------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        '''
        cNrm,mNrm  = self.getPointsForInterpolation(EndOfPrdvP,aNrm)       
        solution_now = self.usePointsForInterpolation(cNrm,mNrm,interpolator)
        return solution_now

        
    def addMPCandHumanWealth(self,solution):
        '''
        Take a solution and add human wealth and the bounding MPCs to it.
        
        Parameters:
        ------------
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
        
        Parameters:
        ------------
        mNrm : np.array
            Corresponding market resource points for interpolation.
        cNrm : np.array
            Consumption points for interpolation.
            
        Returns:
        -----------
        cFuncUnc : LinearInterp
            The unconstrained consumption function for this period.
        '''
        cFuncUnc = LinearInterp(mNrm,cNrm,self.MPCminNow*self.hNrmNow,self.MPCminNow)
        return cFuncUnc
        
                
    def solve(self):
        '''
        Solves a one period consumption saving problem with risky income.
        
        Parameters:
        ------------
        none
            
        Returns:
        -----------
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
    """
    Method that adds value function, cubic interpolation to ENDG 
    """

    def getConsumptionCubic(self,mNrm,cNrm):
        """
        Interpolate the unconstrained consumption function with cubic splines
        """        
        EndOfPrdvPP   = self.DiscFacEff*self.Rfree*self.Rfree*self.PermGroFac**(-self.CRRA-1.0)* \
                      np.sum(self.PermShkVals_temp**(-self.CRRA-1.0)*self.vPPfuncNext(self.mNrmNext)*self.ShkPrbs_temp,
                             axis=0)    
        dcda        = EndOfPrdvPP/self.uPP(np.array(cNrm[1:]))
        MPC         = dcda/(dcda+1.)
        MPC         = np.insert(MPC,0,self.MPCmaxNow)

        cFuncNowUnc = CubicInterp(mNrm,cNrm,MPC,self.MPCminNow*self.hNrmNow,self.MPCminNow)
        return cFuncNowUnc
        
        
    def makeEndOfPrdvFunc(self,EndOfPrdvP):
        '''
        Construct the end-of-period value function.
        '''
        VLvlNext       = (self.PermShkVals_temp**(1.0-self.CRRA)*self.PermGroFac**(1.0-self.CRRA))*self.vFuncNext(self.mNrmNext)
        EndOfPrdv      = self.DiscFacEff*np.sum(VLvlNext*self.ShkPrbs_temp,axis=0)
        EndOfPrdvNvrs  = self.uinv(EndOfPrdv) # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP*self.uinvP(EndOfPrdv)
        EndOfPrdvNvrs  = np.insert(EndOfPrdvNvrs,0,0.0)
        EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP,0,EndOfPrdvNvrsP[0]) # This is *very* slightly wrong
        aNrm_temp      = np.insert(self.aNrmNow,0,self.BoroCnstNat)
        EndOfPrdvNvrsFunc = CubicInterp(aNrm_temp,EndOfPrdvNvrs,EndOfPrdvNvrsP)
        self.EndOfPrdvFunc = ValueFunc(EndOfPrdvNvrsFunc,self.CRRA)


    def putVfuncInSolution(self,solution,EndOfPrdvP):
        self.makeEndOfPrdvFunc(EndOfPrdvP)
        solution.vFunc = self.makevFunc(solution)
        return solution
        

    def makevFunc(self,solution):        
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
        """
        Take a solution, and add in vPPfunc to it, so that the next solver can
        evaluate vPP and thus use cubic interpolation.
        """
        vPPfuncNow        = MargMargValueFunc(solution.cFunc,self.CRRA)
        solution.vPPfunc  = vPPfuncNow
        return solution

       
    def solve(self):        
        aNrm         = self.prepareToGetGothicvP()           
        EndOfPrdvP   = self.getGothicvP()
        
        if self.CubicBool:
            solution   = self.getSolution(EndOfPrdvP,aNrm,interpolator=self.getConsumptionCubic)
        else:
            solution   = self.getSolution(EndOfPrdvP,aNrm,self.makecFuncLinear)
        solution       = self.addMPCandHumanWealth(solution)
        
        if self.vFuncBool:
            solution = self.putVfuncInSolution(solution,EndOfPrdvP)
        if self.CubicBool: 
            solution = self.addvPPfunc(solution)
                   
        #print('Solved a period with ENDG!')
        return solution        
       

def consumptionSavingSolverENDG(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
                                       
    if (not CubicBool) and (not vFuncBool):
        solver = ConsumptionSavingSolverENDGBasic(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,BoroCnstArt,aXtraGrid,
                                             vFuncBool,CubicBool)        
    else:
        solver = ConsumptionSavingSolverENDG(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,BoroCnstArt,aXtraGrid,
                                             vFuncBool,CubicBool)
    solver.prepareToSolve()                      
    solution = solver.solve()
    return solution   


####################################################################################################
####################################################################################################


class ConsumptionSavingSolverKinkedR(ConsumptionSavingSolverENDG):
    """
    A class to solve a consumption-savings problem where the interest rate on debt differs
    from the interest rate on savings, using the method of endogenous gridpoints.
    
    See documentation for ConsumptionSavingSolverENDG.  Inputs and outputs here are identical,
    except there are two interest rates as inputs (R_save and R_borrow) instead of one (Rfree).

    Parameters:
    -----------
    solution_next: ConsumerSolution
        The solution to the following period.
    IncomeDstn: [np.array]
        A list containing three lists of floats, representing a discrete approximation to the income
        process between the period being solved and the one immediately following (in solution_next).
        Order: probs, psi, xi
    LivPrb: float
        Probability of surviving to succeeding period.
    DiscFac: float
        Discount factor between this period and the succeeding period.
    CRRA: float
        The coefficient of relative risk aversion
    Rboro: float
        Interest factor on assets between this period and the succeeding period
        when assets are negative.
    Rsave: float
        Interest factor on assets between this period and the succeeding period
        when assets are positive.
    PermGroFac: float
        Expected growth factor for permanent income between this period and the succeeding period.
    BoroCnstArt: float
        Borrowing constraint for the minimum allowable assets to end the period
        with.  If it is less than the natural borrowing constraint, then it is
        irrelevant; BoroCnstArt=None indicates no artificial borrowing constraint.
    aXtraGrid: [float]
        A list of end-of-period asset values (post-decision states) at which to solve for optimal
        consumption.

    Returns:
    -----------
    solution_now: ConsumerSolution
        The solution to this period's problem, obtained using the method of endogenous gridpoints.


    """
    def __init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,
                      Rboro,Rsave,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):        

        assert CubicBool==False,'KinkedR will only work with linear interpolation (for now)'

        # Initialize the solver.  Most of the steps are exactly the same as in the Endogenous Grid
        # linear case, so start with that.
        ConsumptionSavingSolverENDG.__init__(self,solution_next,IncomeDstn,
                                                   LivPrb,DiscFac,CRRA,Rboro,PermGroFac,BoroCnstArt,
                                                   aXtraGrid,vFuncBool,CubicBool) 

        # Assign the interest rates as class attributes, to use them later.
        self.Rboro   = Rboro
        self.Rsave   = Rsave


    def prepareToGetGothicvP(self):
        """
        Method to prepare for calculating EndOfPrdvP.
        
        This differs from the baseline case because different savings choices yield different
        interest rates.
        """        
        aNrmNow           = np.sort(np.hstack((np.asarray(self.aXtraGrid) + 
                            self.mNrmMinNow,np.array([0.0,0.0]))))
        aXtraCount        = aNrmNow.size
        ShkCount          = self.TranShkValsNext.size
        aNrm_temp         = np.tile(aNrmNow,(ShkCount,1))
        PermShkVals_temp  = (np.tile(self.PermShkValsNext,(aXtraCount,1))).transpose()
        TranShkVals_temp  = (np.tile(self.TranShkValsNext,(aXtraCount,1))).transpose()
        ShkPrbs_temp      = (np.tile(self.ShkPrbsNext,(aXtraCount,1))).transpose()

        Rfree_vec         = self.Rsave*np.ones(aXtraCount)
        Rfree_vec[0:(np.sum(aNrmNow<=0)-1)] = self.Rboro
        self.Rfree        = Rfree_vec

        Rfree_temp        = np.tile(Rfree_vec,(ShkCount,1))
        mNrmNext          = Rfree_temp/(self.PermGroFac*PermShkVals_temp)*aNrm_temp + TranShkVals_temp
        
        PatFacTop         = ((self.Rsave*self.DiscFacEff)**(1.0/self.CRRA))/self.Rsave
        self.MPCminNow    = 1.0/(1.0 + PatFacTop/self.solution_next.MPCmin)
        self.hNrmNow      = self.PermGroFac/self.Rsave*(np.dot(self.ShkPrbsNext,self.TranShkValsNext*self.PermShkValsNext) + self.solution_next.hNrm)

        self.PermShkVals_temp = PermShkVals_temp
        self.ShkPrbs_temp     = ShkPrbs_temp
        self.mNrmNext         = mNrmNext
        self.aNrmNow          = aNrmNow

        return aNrmNow


def consumptionSavingSolverKinkedR(solution_next,IncomeDstn,
                                   LivPrb,DiscFac,CRRA,Rboro,Rsave,PermGroFac,BoroCnstArt,
                                   aXtraGrid,vFuncBool,CubicBool):

    solver = ConsumptionSavingSolverKinkedR(solution_next,IncomeDstn,LivPrb,
                                            DiscFac,CRRA,Rboro,Rsave,PermGroFac,BoroCnstArt,aXtraGrid,
                                            vFuncBool,CubicBool)
    solver.prepareToSolve()                                      
    solution = solver.solve()

    return solution                                   


####################################################################################################
####################################################################################################

class ConsumptionSavingSolverMarkov(ConsumptionSavingSolverENDG):
    '''
    Solves a single period of a standard consumption-saving problem, representing
    the consumption function as a cubic spline interpolation if CubicBool is
    True and as a linear interpolation if it is False.  Problem is solved using
    the method of endogenous gridpoints.  Solver allows for exogenous transitions
    between discrete states; future states only differ in their income distri-
    butions, should generalize this later.

    Parameters:
    -----------
    solution_next: ConsumerSolution
        The solution to the following period.
    MrkvArray : numpy.array
        An NxN array representing a Markov transition matrix between discrete
        states.  The i,j-th element of MrkvArray is the probability of
        moving from state i in period t to state j in period t+1.
    IncomeDstn: [[numpy.array]]
        A list of lists containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved and
        the one immediately following (in solution_next).  Order: probs, psi, xi.
        The n-th element of IncomeDstn is the income distribution for the n-th
        discrete state.
    LivPrb: float
        Probability of surviving to succeeding period.
    DiscFac: float
        Discount factor between this period and the succeeding period.
    CRRA: float
        The coefficient of relative risk aversion
    Rfree: float
        Interest factor on assets between this period and the succeeding period: w_tp1 = a_t*Rfree
    PermGroFac: float
        Expected growth factor for permanent income between this period and the succeeding period.
    BoroCnstArt: float
        Borrowing constraint for the minimum allowable assets to end the period
        with.  If it is less than the natural borrowing constraint, then it is
        irrelevant; BoroCnstArt=None indicates no artificial borrowing constraint.
    aXtraGrid: [float]
        A list of end-of-period asset values (post-decision states) at which to solve for optimal 
        consumption.
    vFuncBool: Boolean
        An indicator for whether the value function should be computed and included
        in the reported solution
    CubicBool: Boolean
        An indicator for whether the solver should use cubic or linear interpolation
    

    Returns:
    -----------
    solution_t: ConsumerSolution
        The solution to this period's problem, obtained using the method of endogenous gridpoints.
    '''

    def __init__(self,solution_next,IncomeDstn_list,LivPrb,DiscFac,
                      CRRA,Rfree_list,PermGroFac_list,MrkvArray,BoroCnstArt,
                      aXtraGrid,vFuncBool,CubicBool):

        ConsumptionSavingSolverENDG.assignParameters(self,solution_next,np.nan,
                                                     LivPrb,DiscFac,CRRA,np.nan,np.nan,
                                                     BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.defineUtilityFunctions()
        self.IncomeDstn_list      = IncomeDstn_list
        self.Rfree_list           = Rfree_list
        self.PermGroFac_list      = PermGroFac_list
        self.StateCount           = len(IncomeDstn_list)
        self.MrkvArray            = MrkvArray

    def solve(self):
        '''
        Solve the one period problem of the consumption-saving model with a Markov state.
        '''
        # Find the natural borrowing constraint in each current state
        self.defBoundary()
        
        # Initialize end-of-period (marginal) value functions
        self.EndOfPrdvFunc_list = []
        self.EndOfPrdvPfunc_list = []
        self.ExIncNext      = np.zeros(self.StateCount) + np.nan # expected income conditional on the next state
        self.WorstIncPrbAll = np.zeros(self.StateCount) + np.nan # # probability of getting the worst income shock in each next period state

        # Loop through each next-period-state and calculate the end-of-period
        # (marginal) value function
        for j in range(self.StateCount):
            # Condition values on next period's state (and record a couple for later use)
            self.conditionOnState(j)
            self.ExIncNext[j]      = np.dot(self.ShkPrbsNext,self.PermShkValsNext*self.TranShkValsNext)
            self.WorstIncPrbAll[j] = self.WorstIncPrb
            
            # Construct the end-of-period value marginal functional conditional
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
        Find the borrowing constraint for each current state.
        '''
        self.BoroCnstNatAll          = np.zeros(self.StateCount) + np.nan
        # Find the natural borrowing constraint conditional on next period's state
        for j in range(self.StateCount):
            PermShkMinNext         = np.min(self.IncomeDstn_list[j][1])
            TranShkMinNext         = np.min(self.IncomeDstn_list[j][2])
            self.BoroCnstNatAll[j] = (self.solution_next.mNrmMin[j] - TranShkMinNext)*(self.PermGroFac_list[j]*PermShkMinNext)/self.Rfree_list[j]

        self.BoroCnstNat_list   = np.zeros(self.StateCount) + np.nan
        self.mNrmMin_list       = np.zeros(self.StateCount) + np.nan
        self.BoroCnstDependency = np.zeros((self.StateCount,self.StateCount)) + np.nan
        # The natural borrowing constraint in each current state is the *highest*
        # among next-state-conditional natural borrowing constraints that could
        # occur from this current state.
        for i in range(self.StateCount):
            possible_next_states     = self.MrkvArray[i,:] > 0
            self.BoroCnstNat_list[i] = np.max(self.BoroCnstNatAll[possible_next_states])
            self.mNrmMin_list[i]     = np.max([self.BoroCnstNat_list[i],self.BoroCnstArt])
            self.BoroCnstDependency[i,:] = self.BoroCnstNat_list[i] == self.BoroCnstNatAll
        # Also creates a Boolean array indicating whether the natural borrowing
        # constraint *could* be hit when transitioning from i to j.
     
    def conditionOnState(self,state_index):
        """
        Find the income distribution, etc., conditional on a given state next period.
        """
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
        Calculates end-of-period marginal marginal value using pre-defined array
        of next period market resources.
        '''
        EndOfPrdvPP = self.DiscFacEff*self.Rfree*self.Rfree*self.PermGroFac**(-self.CRRA-1.0)*np.sum(self.PermShkVals_temp**(-self.CRRA-1.0)*
                      self.vPPfuncNext(self.mNrmNext)*self.ShkPrbs_temp,axis=0)
        return EndOfPrdvPP
        
    
    def makeEndOfPrdvFuncCond(self):
        '''
        Construct the end-of-period value function conditional on next period's state.
        '''
        VLvlNext           = (self.PermShkVals_temp**(1.0-self.CRRA)*self.PermGroFac**(1.0-self.CRRA))*self.vFuncNext(self.mNrmNext)
        EndOfPrdv_cond     = self.DiscFacEff*np.sum(VLvlNext*self.ShkPrbs_temp,axis=0)
        EndOfPrdvNvrs_cond = self.uinv(EndOfPrdv_cond)
        EndOfPrdvNvrsP_cond= self.EndOfPrdvP_cond*self.uinvP(EndOfPrdv_cond)
        EndOfPrdvNvrs_cond = np.insert(EndOfPrdvNvrs_cond,0,0.0)
        EndOfPrdvNvrsP_cond= np.insert(EndOfPrdvNvrsP_cond,0,EndOfPrdvNvrsP_cond[0])
        aNrm_temp          = np.insert(self.aNrm_cond,0,self.BoroCnstNat)
        EndOfPrdvNvrsFunc_cond = CubicInterp(aNrm_temp,EndOfPrdvNvrs_cond,EndOfPrdvNvrsP_cond)
        return ValueFunc(EndOfPrdvNvrsFunc_cond,self.CRRA)
        
            
    def makeEndOfPrdvPfuncCond(self):
        '''
        Construct the end-of-period marginal value function conditional on next period's state.
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
            EndOfPrdvPnvrsFunc_cond = CubicInterp(self.aNrm_cond,EndOfPrdvPnvrs_cond,EndOfPrdvPnvrsP_cond,lower_extrap=True)
        else:
            EndOfPrdvPnvrsFunc_cond = LinearInterp(self.aNrm_cond,EndOfPrdvPnvrs_cond,lower_extrap=True)            
        return MargValueFunc(EndOfPrdvPnvrsFunc_cond,self.CRRA) # "recurve" the interpolated marginal value function
            
    def calcEndOfPrdvP(self):
        '''
        Calculates end of period marginal value (and marginal marginal) value
        at each aXtra gridpoint for each *current* state.
        '''
        aNrmMin_unique, state_inverse = np.unique(self.BoroCnstNat_list,return_inverse=True)
        self.possible_transitions     = self.MrkvArray > 0
        EndOfPrdvP                    = np.zeros((self.StateCount,self.aXtraGrid.size))
        EndOfPrdvPP                   = np.zeros((self.StateCount,self.aXtraGrid.size))
        for k in range(aNrmMin_unique.size):
            aNrmMin       = aNrmMin_unique[k]
            which_states  = state_inverse == k
            aGrid         = aNrmMin + self.aXtraGrid
            EndOfPrdvP_all  = np.zeros((self.StateCount,self.aXtraGrid.size))
            EndOfPrdvPP_all = np.zeros((self.StateCount,self.aXtraGrid.size))
            for j in range(self.StateCount):
                if np.any(np.logical_and(self.possible_transitions[:,j],which_states)):
                    EndOfPrdvP_all[j,:] = self.EndOfPrdvPfunc_list[j](aGrid)
                    if self.CubicBool:
                        EndOfPrdvPP_all[j,:] = self.EndOfPrdvPfunc_list[j].derivative(aGrid)
            EndOfPrdvP_temp = np.dot(self.MrkvArray,EndOfPrdvP_all)
            EndOfPrdvP[which_states,:] = EndOfPrdvP_temp[which_states,:]
            if self.CubicBool:
                EndOfPrdvPP_temp = np.dot(self.MrkvArray,EndOfPrdvPP_all)
                EndOfPrdvPP[which_states,:] = EndOfPrdvPP_temp[which_states,:]
        self.EndOfPrdvP = EndOfPrdvP
        if self.CubicBool:
            self.EndOfPrdvPP = EndOfPrdvPP
            
    def calcHumWealthAndBoundingMPCs(self):
        '''
        Calculates human wealth and the maximum and minimum MPC for each current period state.
        '''
        # Upper bound on MPC at lower m-bound
        WorstIncPrb_array = self.BoroCnstDependency*np.tile(np.reshape(self.WorstIncPrbAll,(1,self.StateCount)),(self.StateCount,1))
        temp_array = self.MrkvArray*WorstIncPrb_array
        WorstIncPrbNow    = np.sum(temp_array,axis=1) # Probability of getting the "worst" income shock and transition from each current state
        ExMPCmaxNext      = (np.dot(temp_array,self.Rfree_list**(1.0-self.CRRA)*self.solution_next.MPCmax**(-self.CRRA))/WorstIncPrbNow)**(-1.0/self.CRRA)
        self.MPCmaxNow    = 1.0/(1.0 + ((self.DiscFacEff*WorstIncPrbNow)**(1.0/self.CRRA))/ExMPCmaxNext)
        self.MPCmaxEff    = self.MPCmaxNow
        self.MPCmaxEff[self.BoroCnstNat_list < self.mNrmMin_list] = 1.0
        # State-conditional PDV of human wealth
        hNrmPlusIncNext   = self.ExIncNext + self.solution_next.hNrm
        self.hNrmNow      = np.dot(self.MrkvArray,(self.PermGroFac_list/self.Rfree_list)*hNrmPlusIncNext)
        # Lower bound on MPC as m gets arbitrarily large
        temp = (self.DiscFacEff*np.dot(self.MrkvArray,self.solution_next.MPCmin**(-self.CRRA)*self.Rfree_list**(1.0-self.CRRA)))**(1.0/self.CRRA)
        self.MPCminNow = 1.0/(1.0 + temp)

    def makeSolution(self,cNrm,mNrm):
        '''
        Construct an object representing the solution to this period's problem.
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
            self.hNrmNow_j   = self.hNrmNow[i]
            self.MPCminNow_j = self.MPCminNow[i]
            if self.CubicBool:
                self.MPC_temp_j  = self.MPC_temp[i,:]
                
            # Make the constrained portion of the consumption function for this state
            self.cFuncNowCnst = LinearInterp([self.mNrmMin_list[i], self.mNrmMin_list[i]+1.0],[0.0,1.0])
            
            # Construct the unconstrained consumption function for this state
            cFuncNowUnc = interpfunc(mNrm[i,:],cNrm[i,:])

            # Combine the constrained and unconstrained functions into the true consumption function
            cFuncNow = LowerEnvelope(cFuncNowUnc,self.cFuncNowCnst)

            # Make the marginal value function and the marginal marginal value function
            vPfuncNow = MargValueFunc(cFuncNow,self.CRRA)

            # Pack up the state conditional solution
            solution_cond = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow)
            
            # Add the state-conditional marginal marginal value function (if desired)            
            if self.CubicBool: 
                solution_cond = self.addvPPfunc(solution_cond)

            # Add the state-conditional solution to the overall period solution
            solution.appendSolution(solution_cond)
        
        # Add the lower bounds of market resources, MPC limits, human resources, and the value functions
        solution.mNrmMin = self.mNrmMin_list
        solution = self.addMPCandHumanWealth(solution)
        if self.vFuncBool:
            vFuncNow = self.makevFunc(solution)
            solution.vFunc = vFuncNow
        
        # Return the overall solution to this period
        return solution
        
    
    def makeLinearcFunc(self,mNrm,cNrm):
        '''
        Makes a linear interpolation to represent the (unconstrained) consumption function.
        '''
        cFuncUnc = LinearInterp(mNrm,cNrm,self.MPCminNow_j*self.hNrmNow_j,self.MPCminNow_j)
        return cFuncUnc


    def makeCubiccFunc(self,mNrm,cNrm):
        """
        Interpolate the unconstrained consumption function with cubic splines
        """
        cFuncUnc = CubicInterp(mNrm,cNrm,self.MPC_temp_j,self.MPCminNow_j*self.hNrmNow_j,self.MPCminNow_j)
        return cFuncUnc
        
    def makevFunc(self,solution):
        '''
        Construct the value function for each current state.
        '''
        vFuncNow = []
        for i in range(self.StateCount):
            mNrmMin       = self.mNrmMin_list[i]
            mGrid         = mNrmMin + self.aXtraGrid
            cGrid         = solution.cFunc[i](mGrid)
            aGrid         = mGrid - cGrid
            EndOfPrdv_all   = np.zeros((self.StateCount,self.aXtraGrid.size))
            for j in range(self.StateCount):
                if self.possible_transitions[i,j]:
                    EndOfPrdv_all[j,:] = self.EndOfPrdvFunc_list[j](aGrid)
            EndOfPrdv       = np.dot(self.MrkvArray[i,:],EndOfPrdv_all)
            vNrmNow       = self.u(cGrid) + EndOfPrdv
            vPnow         = self.uP(cGrid)
            
            vNvrs        = self.uinv(vNrmNow) # value transformed through inverse utility
            vNvrsP       = vPnow*self.uinvP(vNrmNow)
            mNrm_temp   = np.insert(mGrid,0,mNrmMin)
            vNvrs        = np.insert(vNvrs,0,0.0)
            vNvrsP       = np.insert(vNvrsP,0,self.MPCmaxEff[i]**(-self.CRRA/(1.0-self.CRRA)))
            MPCminNvrs   = self.MPCminNow[i]**(-self.CRRA/(1.0-self.CRRA))
            vNvrsFunc_i  = CubicInterp(mNrm_temp,vNvrs,vNvrsP,MPCminNvrs*self.hNrmNow[i],MPCminNvrs)
            vFunc_i     = ValueFunc(vNvrsFunc_i,self.CRRA)
            vFuncNow.append(vFunc_i)
        return vFuncNow


def solveConsumptionSavingMarkov(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,MrkvArray,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
                                       
    solver = ConsumptionSavingSolverMarkov(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,MrkvArray,BoroCnstArt,aXtraGrid,
                                               vFuncBool,CubicBool)              
    solution = solver.solve()
    return solution             
        
  
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
    cFunc_terminal_      = LinearInterp([0.0, 1.0],[0.0,1.0])
    vFunc_terminal_      = LinearInterp([0.0, 1.0],[0.0,0.0])
    cFuncCnst_terminal_  = LinearInterp([0.0, 1.0],[0.0,1.0])
    solution_terminal_   = ConsumerSolution(cFunc=LowerEnvelope(cFunc_terminal_,cFuncCnst_terminal_),
                                            vFunc = vFunc_terminal_, mNrmMin=0.0, hNrm=0.0, 
                                            MPCmin=1.0, MPCmax=1.0)
    time_vary_ = ['LivPrb','DiscFac','PermGroFac']
    time_inv_ = ['CRRA','Rfree','aXtraGrid','BoroCnstArt','vFuncBool','CubicBool']
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data.
        '''       
        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=deepcopy(ConsumerType.solution_terminal_),
                           cycles=cycles,time_flow=time_flow,pseudo_terminal=False,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary    = deepcopy(ConsumerType.time_vary_)
        self.time_inv     = deepcopy(ConsumerType.time_inv_)
        self.solveOnePeriod = consumptionSavingSolverENDG # this can be swapped for consumptionSavingSolverEXOG or another solver
        self.update()
        self.a_init = np.zeros(self.Nagents)
        self.p_init = np.ones(self.Nagents)

    def unpack_cFunc(self):
        '''
        "Unpacks" the consumption functions into their own field for easier access.
        After the model has been solved, the consumption functions reside in the
        attribute cFunc of each element of ConsumerType.solution.  This method
        creates a (time varying) attribute cFunc that contains a list of consumption
        functions.
        '''
        self.cFunc = []
        for solution_t in self.solution:
            self.cFunc.append(solution_t.cFunc)
        if not ('cFunc' in self.time_vary):
            self.time_vary.append('cFunc')
          
    def addIncomeShockPaths(self,PermShks,TranShks):
        '''
        Adds paths of simulated shocks to the agent as attributes.
        '''
        original_time = self.time_flow
        self.timeFwd()
        self.PermShks = PermShks
        self.TranShks = TranShks
        if not 'PermShks' in self.time_vary:
            self.time_vary.append('PermShks')
        if not 'TranShks' in self.time_vary:
            self.time_vary.append('TranShks')
        if not original_time:
            self.timeRev()
            
    def makeIncShkHist(self):
        '''
        Makes histories of simulated income shocks for this consumer type by
        drawing from the discrete income distributions.  Non-Markov version.
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
        
        for t in range(1,self.sim_periods):
            IncomeDstnNow    = self.IncomeDstn[t_idx]
            PermGroFacNow    = self.PermGroFac[t_idx]
            Events           = np.arange(IncomeDstnNow[0].size) # just a list of integers
            Cutoffs          = np.round(np.cumsum(IncomeDstnNow[0])*self.Nagents)
            top = 0
            EventList        = []
            for j in range(Events.size):
                bot = top
                top = Cutoffs[j]
                EventList += (top-bot)*[Events[j]]
            EventDraws       = self.RNG.permutation(EventList)
            PermShkHist[t,:] = IncomeDstnNow[1][EventDraws]*PermGroFacNow
            TranShkHist[t,:] = IncomeDstnNow[2][EventDraws]
            t_idx += 1
            if t_idx >= len(self.IncomeDstn):
                t_idx = 0
                
        self.PermShkHist = PermShkHist
        self.TranShkHist = TranShkHist
        if not orig_time:
            self.timeRev()
            
    def makeIncShkHistMrkv(self):
        '''
        Makes histories of simulated income shocks for this consumer type by
        drawing from the discrete income distributions, respecting the Markov
        state for each agent in each period.  Should be run after makeMrkvHist().
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
                EventList        = []
                for j in range(Events.size):
                    bot = top
                    top = Cutoffs[j]
                    EventList += (top-bot)*[Events[j]]
                EventDraws       = self.RNG.permutation(EventList)
                PermShkHist[t,these] = IncomeDstnNow[1][EventDraws]*PermGroFacNow
                TranShkHist[t,these] = IncomeDstnNow[2][EventDraws]
            t_idx += 1
            if t_idx >= len(self.IncomeDstn):
                t_idx = 0
                
        self.PermShkHist = PermShkHist
        self.TranShkHist = TranShkHist
        if not orig_time:
            self.timeRev()
        
            
    def makeMrkvHist(self):
        '''
        Makes a history of simulated discrete Markov states, starting from the
        initial states in markov_init.  Assumes that MrkvArray is constant
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
       Updates this agent's end-of-period assets grid.
        '''
        aXtraGrid = constructAssetsGrid(self)
        self.aXtraGrid = aXtraGrid
        
    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution
        '''
        self.solution_terminal.vFunc   = ValueFunc(self.solution_terminal.cFunc,self.CRRA)
        self.solution_terminal.vPfunc  = MargValueFunc(self.solution_terminal.cFunc,self.CRRA)
        self.solution_terminal.vPPfunc = MargMargValueFunc(self.solution_terminal.cFunc,self.CRRA)
        
    def update(self):
        '''
        Update the income process, the assets grid, and the terminal solution.
        '''
        self.updateIncomeProcess()
        self.updateAssetsGrid()
        self.updateSolutionTerminal()
                    
    def simulate(self,w_init,t_first,t_last,which=['w']):
        '''
        Simulate the model forward from initial conditions w_init, beginning in
        t_first and ending in t_last.
        '''
        original_time = self.time_flow
        self.timeFwd()
        if self.cycles > 0:
            cFuncs = self.cFunc[t_first:t_last]
        else:
            cFuncs = t_last*self.cFunc # This needs to be fixed for IH models
        simulated_history = simulateConsumerHistory(cFuncs, w_init, self.PermShks[t_first:t_last],
                                                    self.TranShks[t_first:t_last],which)
        if not original_time:
            self.timeRev()
        return simulated_history
                
    def initializeSim(self,a_init=None,p_init=None,t_init=0,sim_prds=None):
        '''
        Readies this type for simulation by clearing its history, initializing
        state variables, and setting time indices to their correct position.
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
        marginal propensity to consume, and assets (after all actions), given
        initial assets (normalized by permanent income).  User can specify which
        period of life to begin the simulation, and how many periods to simulate.
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
        '''
        self.cFuncNow  = self.solution[self.cFunc_idx].cFunc
        self.cFunc_idx += 1
        if self.cFunc_idx >= len(self.solution):
            self.cFunc_idx = 0 # Reset to zero if we've run out of cFuncs
                
    def calcBoundingValues(self):
        '''
        Calculate the PDV of human wealth (after receiving income this period)
        in an infinite horizon model with only one period repeated indefinitely.
        Also calculates MPCmin and MPCmax for infinite horizon.
        '''
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



def simulateConsumerHistory(cFunc,w0,PermShk,TranShk,which):
    """
    Generates simulated consumer histories.  Agents begin with W/Y ratio of of
    w0 and follow the consumption rules in cFunc each period. Permanent and trans-
    itory shocks are provided in scriptR and theta.  Note that
    PermShk represents R*psi_{it}/PermGroFac_t, the "effective interest factor" for
    agent i in period t.  Further, the object of interest w is the wealth-to
    permanent-income ratio at the beginning of the period, before income is received.
    
    The histories returned by the simulator are determined by which, a list of
    strings that can include 'w', 'm', 'c', 'a', and 'kappa'.  Other strings will
    cause an error on return.  Outputs are returned in the order listed by the user.
    """
    # Determine the size of potential simulated histories
    periods_to_simulate = len(TranShk)
    N_agents = len(TranShk[0])
    
    # Initialize arrays to hold simulated histories as requested
    if 'w' in which:
        w = np.zeros([periods_to_simulate+1,N_agents]) + np.nan
        do_w = True
    else:
        do_w = False
    if 'm' in which:
        m = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_m = True
    else:
        do_m = False
    if 'c' in which:
        c = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_c = True
    else:
        do_c = False
    if 'a' in which:
        a = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_a = True
    else:
        do_a = False
    if 'kappa' in which:
        kappa = np.zeros([periods_to_simulate,N_agents]) + np.nan
        do_k = True
    else:
        do_k = False

    # Initialize the simulation
    w_t = w0
    if do_w:
        w[0,] = w_t
    
    # Run the simulation for all agents:
    for t in range(periods_to_simulate):
        m_t = w_t + TranShk[t]
        if do_k:
            c_t, kappa_t = cFunc[t].eval_with_derivative(m_t)
        else:
            c_t = cFunc[t](m_t)
        a_t = m_t - c_t
        w_t = PermShk[t]*a_t
        
        # Store the requested variables in the history arrays
        if do_w:
            w[t+1,] = w_t
        if do_m:
            m[t,] = m_t
        if do_c:
            c[t,] = c_t
        if do_a:
            a[t,] = a_t
        if do_k:
            kappa[t,] = kappa_t
            
    # Return the simulated histories as requested
    return_list = ''
    for var in which:
        return_list = return_list + var + ', '
    x = len(return_list)
    return_list = return_list[0:(x-2)]
    return eval(return_list)



# ==================================================================================
# = Functions for generating discrete income processes and simulated income shocks =
# ==================================================================================

def constructLognormalIncomeProcessUnemployment(parameters):
    """
    Generates a list of discrete approximations to the income process for each
    life period, from end of life to beginning of life.  Permanent shocks are mean
    one lognormally distributed with standard deviation PermShkStd[t] during the
    working life, and degenerate at 1 in the retirement period.  Transitory shocks
    are mean one lognormally distributed with a point mass at IncUnemp with
    probability UnempPrb while working; they are mean one with a point mass at
    IncUnempRet with probability UnempPrbRet.  Retirement occurs
    after t=final_work_index periods of retirement.

    Note 1: All time in this function runs forward, from t=0 to t=T
    
    Note 2: All parameters are passed as attributes of the input parameters.

    Parameters:
    -----------
    PermShkStd:    [float]
        Array of standard deviations in _permanent_ income uncertainty during
        the agent's life.
    PermShkCount:      int
        The number of approximation points to be used in the equiprobable
        discrete approximation to the permanent income shock distribution.
    TranShkStd      [float]
        Array of standard deviations in _temporary_ income uncertainty during
        the agent's life.
    TranShkCount:       int
        The number of approximation points to be used in the equiprobable
        discrete approximation to the permanent income shock distribution.
    UnempPrb:             float
        The probability of becoming unemployed
    UnempPrbRet:      float
        The probability of not receiving typical retirement income in any retired period
    T_retire:       int
        The index value i equal to the final working period in the agent's life.
        If T_retire <= 0 then there is no retirement.
    IncUnemp:         float
        Income received when unemployed. Often zero.
    IncUnempRet:  float
        Income received while "unemployed" when retired. Often zero.
    T_total:       int
        Total number of non-terminal periods in this consumer's life.

    Returns
    =======
    IncomeDstn:  [income distribution]
        Each element contains the joint distribution of permanent and transitory
        income shocks, as a set of vectors: psi_shock, xi_shock, and pmf. The
        first two are the points in the joint state space, and final vector is
        the joint pmf over those points. For example,
               psi_shock[20], xi_shock[20], and pmf[20]
        refers to the (psi, xi) point indexed by 20, with probability p = pmf[20].
    """
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
    
    IncomeDstn = [] # Discrete approximation to income process

    # Fill out a simple discrete RV for retirement, with value 1.0 (mean of shocks)
    # in normal times; value 0.0 in "unemployment" times with small prob.
    if T_retire > 0:
        if UnempPrbRet > 0:
            PermShkValsRet  = np.array([1.0, 1.0])    # Permanent income is deterministic in retirement (2 states for temp income shocks)
            TranShkValsRet  = np.array([IncUnempRet, (1.0-UnempPrbRet*IncUnempRet)/(1.0-UnempPrbRet)])
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
            IncomeDstn.append(combineIndepDstns(PermShkDstn,TranShkDstn))

    return IncomeDstn
    

def applyFlatIncomeTax(IncomeDstn,tax_rate,T_retire,unemployed_indices=[],transitory_index=2):
    '''
    Applies a flat income tax rate to all employed income states during the working
    period of life (those before T_retire).  Time runs forward in this function.
    
    Parameters:
    -------------
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
        
    Returns:
    ------------
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
   
    
    
def generateIncomeShockHistoryLognormalUnemployment(parameters):
    '''
    Creates arrays of permanent and transitory income shocks for Nagents simulated
    consumers for the entire duration of the lifecycle.  All inputs are assumed
    to be given in ordinary chronological order, from terminal period to t=0.
    Output is also returned in ordinary chronological order.
    
    Arguments:
    ----------
    PermShkStd : [float]
        Permanent income standard deviations for the consumer by age.
    TranShkStd : [float]
        Transitory income standard devisions for the consumer by age.
    PermGroFac : [float]
        Permanent income growth rates for the consumer by age.
    Rfree : float
        The time-invariant interest factor
    UnempPrb : float
        The probability of becoming unemployed
    UnempPrbRet : float
        The probability of not receiving typical retirement income in any retired period
    T_retire : int
        The index value for final working period in the agent's life.
        If T_retire <= 0 then there is no retirement.
    IncUnemp : float
        Income received when unemployed. Often zero.
    IncUnempRet : float
        Income received while "unemployed" when retired. Often zero.
    Nagents : int
        The number of consumers to generate shocks for.
    tax_rate : float
        An income tax rate applied to employed income.
    RNG : numpy.random.RandomState
        A random number generator for this type.

    Returns:
    ----------
    PermShkHist : np.array
        A total_periods x Nagents array of permanent income shocks.  Each element
        is a value representing Rfree/(psi_{it}*PermGroFac_t), so that w_{t+1} = scriptR_{it}*a_t
    TranShkHist : np.array
        A total_periods x Nagents array of transitory income shocks.
    '''
    # Unpack the parameters
    PermShkStd               = parameters.PermShkStd
    TranShkStd               = parameters.TranShkStd
    PermGroFac               = parameters.PermGroFac
    Rfree                    = parameters.Rfree
    UnempPrb                 = parameters.UnempPrb
    UnempPrbRet              = parameters.UnempPrbRet
    IncUnemp                 = parameters.IncUnemp
    IncUnempRet              = parameters.IncUnempRet
    T_retire                 = parameters.T_retire
    Nagents                  = parameters.Nagents
    RNG                      = parameters.RNG
    tax_rate                 = parameters.tax_rate
    
    # Set the seeds we'll need for random draws 
    PermShk_seed             = RNG.randint(low=1, high=2**31-1)
    TranShk_seed             = RNG.randint(low=1, high=2**31-1)
    Unemp_seed               = RNG.randint(low=1, high=2**31-1)

    # Truncate the lifecycle vectors to the working life
    PermShkStdWork   = PermShkStd[0:T_retire]
    TranShkStdWork   = TranShkStd[0:T_retire]
    PermGroFacWork    = PermGroFac[0:T_retire]
    PermGroFacRet    = PermGroFac[T_retire:]
    working_periods  = len(PermGroFacWork) + 1
    retired_periods  = len(PermGroFacRet)
    
    # Generate transitory shocks in the working period (needs one extra period)
    TranShkHistWork = drawMeanOneLognormal(TranShkStdWork, Nagents, TranShk_seed)
    TranShkHistWork.insert(0,RNG.permutation(TranShkHistWork[0]))
    
    # Generate permanent shocks in the working period
    PermShkHistWork = drawMeanOneLognormal(PermShkStdWork, Nagents, PermShk_seed)
    for t in range(working_periods-1):
        PermShkHistWork[t] = Rfree/(PermShkHistWork[t]*PermGroFacWork[t])

    # Generate permanent and transitory shocks for the retired period
    TranShkHistRet = []
    PermShkHistRet = []
    for t in range(retired_periods):
        TranShkHistRet.append(np.ones([Nagents]))
        PermShkHistRet.append(Rfree*np.ones([Nagents])/PermGroFacRet[t])
    PermShkHistRet.append(Rfree*np.ones([Nagents]))
    
    # Generate draws of unemployment
    UnempPrbLife = [UnempPrb]*working_periods + [UnempPrbRet]*retired_periods
    IncUnempLife = [IncUnemp]*working_periods + [IncUnempRet]*retired_periods
    IncUnempScaleLife = [(1-tax_rate)*(1-UnempPrb*IncUnemp)/(1-UnempPrb)]*\
                          working_periods + [(1-UnempPrbRet*IncUnempRet)/
                          (1-UnempPrbRet)]*retired_periods
    UnempHist = drawBernoulli(UnempPrbLife,Nagents,Unemp_seed)   
    
    # Combine working and retired histories and apply unemployment
    TranShkHist         = TranShkHistWork + TranShkHistRet
    PermShkHist         = PermShkHistWork + PermShkHistRet
    for t in range(len(TranShkHist)):
        TranShkHist[t]               = TranShkHist[t]*IncUnempScaleLife[t]
        TranShkHist[t][UnempHist[t]] = IncUnempLife[t]
    
    return PermShkHist, TranShkHist
    
    
def generateIncomeShockHistoryInfiniteSimple(parameters):
    '''
    Creates arrays of permanent and transitory income shocks for Nagents simulated
    consumers for T identical infinite horizon periods.
    
    Arguments:
    ----------
    PermShkStd : float
        Permanent income standard deviation for the consumer.
    TranShkStd : float
        Transitory income standard deviation for the consumer.
    PermGroFac : float
        Permanent income growth rate for the consumer.
    Rfree : float
        The time-invariant interest factor
    UnempPrb : float
        The probability of becoming unemployed
    IncUnemp : float
        Income received when unemployed. Often zero.
    Nagents : int
        The number of consumers to generate shocks for.
    RNG : numpy.random.RandomState
        A random number generator for this type.
    sim_periods : int
        Number of periods of shocks to generate.
    
    Returns:
    ----------
    PermShkHist : np.array
        A sim_periods x Nagents array of permanent income shocks.  Each element
        is a value representing Rfree*psi_{it}/PermGroFac_t, so that w_{t+1} = scriptR_{it}*a_t
    TranShkHist : np.array
        A sim_periods x Nagents array of transitory income shocks.
    '''
    # Unpack the parameters
    PermShkStd     = parameters.PermShkStd
    TranShkStd     = parameters.TranShkStd
    PermGroFac     = parameters.PermGroFac
    Rfree          = parameters.Rfree
    UnempPrb       = parameters.UnempPrb
    IncUnemp       = parameters.IncUnemp
    Nagents        = parameters.Nagents
    sim_periods    = parameters.sim_periods
    RNG            = parameters.RNG
    
    # Set the seeds we'll need for random draws 
    PermShk_seed   = RNG.randint(low=1, high=2**31-1)
    TranShk_seed   = RNG.randint(low=1, high=2**31-1)
    Unemp_seed     = RNG.randint(low=1, high=2**31-1)
    
    TranShkHist    = drawMeanOneLognormal(sim_periods*TranShkStd, Nagents, TranShk_seed)
    UnempHist      = drawBernoulli(sim_periods*[UnempPrb],Nagents,Unemp_seed)
    PermShkHist    = drawMeanOneLognormal(sim_periods*PermShkStd, Nagents, PermShk_seed)
    for t in range(sim_periods):
        PermShkHist[t] = Rfree/(PermShkHist[t]*PermGroFac)
        TranShkHist[t] = TranShkHist[t]*(1-UnempPrb*IncUnemp)/(1-UnempPrb)
        TranShkHist[t][UnempHist[t]] = IncUnemp
        
    return PermShkHist, TranShkHist

# =======================================================
# ================ Other useful functions ===============
# =======================================================

def constructAssetsGrid(parameters):
    '''
    Constructs the grid of post-decision states, representing end-of-period assets.

    All parameters are passed as attributes of the single input parameters.  The
    input can be an instance of a ConsumerType, or a custom Parameters class.    
    
    Parameters:
    -----------
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
        
    Returns:
    ----------
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
    
#    # Make and solve a finite consumer type
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
        print('Solving a perfect foresight consumer took ' + mystr(end_time-start_time) + ' seconds.')
        PerfectForesightType.unpack_cFunc()
        PerfectForesightType.timeFwd()
        
        print('Consumption functions for perfect foresight vs risky income:')            
        plotFuncs([PerfectForesightType.cFunc[0],InfiniteType.cFunc[0]],InfiniteType.solution[0].mNrmMin,100)
        if InfiniteType.vFuncBool:
            print('Value functions for perfect foresight vs risky income:')
            plotFuncs([PerfectForesightType.solution[0].vFunc,InfiniteType.solution[0].vFunc],InfiniteType.solution[0].mNrmMin+0.5,10)
            
    
####################################################################################################    


    # Make and solve an agent with a kinky interest rate
    KinkyType = deepcopy(InfiniteType)

    KinkyType.time_inv.remove('Rfree')
    KinkyType.time_inv += ['Rboro','Rsave']
    KinkyType(Rboro = 1.2, Rsave = 1.03, BoroCnstArt = None, aXtraCount = 48, cycles=0, CubicBool = False)

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
        MrkvArray = np.array([[(1-p_unemploy_good)*(1-bust_prob),p_unemploy_good*(1-bust_prob),(1-p_unemploy_good)*bust_prob,p_unemploy_good*bust_prob],
                                      [p_reemploy*(1-bust_prob),(1-p_reemploy)*(1-bust_prob),p_reemploy*bust_prob,(1-p_reemploy)*bust_prob],
                                      [(1-p_unemploy_bad)*boom_prob,p_unemploy_bad*boom_prob,(1-p_unemploy_bad)*(1-boom_prob),p_unemploy_bad*(1-boom_prob)],
                                      [p_reemploy*boom_prob,(1-p_reemploy)*boom_prob,p_reemploy*(1-boom_prob),(1-p_reemploy)*(1-boom_prob)]])
        
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
        
        MarkovType.IncomeDstn = [[employed_income_dist,unemployed_income_dist,employed_income_dist,unemployed_income_dist]]
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
