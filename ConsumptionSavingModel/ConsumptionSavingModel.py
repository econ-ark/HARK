# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. 
import sys 
sys.path.insert(0,'../')

import numpy as np
from HARKcore import AgentType, NullFunc
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKutilities import approxMeanOneLognormal, addDiscreteOutcomeConstantMean, combineIndepDists, makeGridExpMult, CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv
from HARKinterpolation import CubicInterp, LowerEnvelope, LinearInterp
from HARKsimulation import drawMeanOneLognormal, drawBernoulli
from scipy.optimize import newton, brentq
from copy import deepcopy, copy

utility      = CRRAutility
utilityP     = CRRAutilityP
utilityPP    = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv  = CRRAutility_inv

# =====================================================================
# === Classes and functions used to solve consumption-saving models ===
# =====================================================================

class ConsumerSolution():
    '''
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function, but may also include
    the minimum allowable money resources mRtoMin, expected human wealth hRto,
    and the lower and upper bounds on the MPC MPCmin and MPCmax.  A value
    function can also be included, as well as marginal value and marg marg value.
    '''

    def __init__(self, cFunc=NullFunc, vFunc=NullFunc, 
                       vPfunc=NullFunc, vPPfunc=NullFunc,
                       mRtoMin=None, hRto=None, MPCmin=None, MPCmax=None):
        '''
        The constructor for a new ConsumerSolution object.
        '''
        self.cFunc        = cFunc
        self.vFunc        = vFunc
        self.vPfunc       = vPfunc
        self.vPPfunc      = vPPfunc
        self.mRtoMin      = mRtoMin
        self.hRto         = hRto
        self.MPCmin       = MPCmin
        self.MPCmax       = MPCmax

    def distance(self,solution_other):
        '''
        Returns the distance between single period solutions as the distance
        between their consumption functions.
        '''
        if type(self.cFunc) is list:
            dist_vec = np.zeros(len(self.cFunc)) + np.nan
            for i in range(len(self.cFunc)):
                dist_vec[i] = self.cFunc[i].distance(solution_other.cFunc[i])
            return np.max(dist_vec)
        else:
            return self.cFunc.distance(solution_other.cFunc)

    def getEulerEquationErrorFunction(self,uPfunc):
        """
        Return the Euler Equation Error function, to check that the solution is "good enough".
        
        Note right now this method needs to be passed uPfunc, which I find awkward and annoying.
        """

        def eulerEquationErrorFunction(m):
            return np.abs(uPfunc(self.cFunc(m)) - self.gothicvPfunc(m))
            
        return eulerEquationErrorFunction
            
    def appendSolution(self,instance_of_ConsumerSolution):
        """
        Used in consumptionSavingSolverMarkov.  Appends one solution to another to create
        a ConsumerSolution consisting of lists.
        """
        if type(self.cFunc)!=list:
            assert self.cFunc==NullFunc            
            # Then the assumption is self is an empty initialized instance,
            # we need to start a list
            self.cFunc       = [instance_of_ConsumerSolution.cFunc]
            self.vFunc       = [instance_of_ConsumerSolution.vFunc]
            self.vPfunc      = [instance_of_ConsumerSolution.vPfunc]
            self.vPPfunc     = [instance_of_ConsumerSolution.vPPfunc]
            self.mRtoMin     = [instance_of_ConsumerSolution.mRtoMin]
        
        else:
            self.cFunc.append(instance_of_ConsumerSolution.cFunc)
            self.vFunc.append(instance_of_ConsumerSolution.vFunc)
            self.vPfunc.append(instance_of_ConsumerSolution.vPfunc)
            self.vPPfunc.append(instance_of_ConsumerSolution.vPPfunc)
            self.mRtoMin.append(instance_of_ConsumerSolution.mRtoMin)

        
class ValueFunc():
    '''
    A class for representing a value function.  The underlying interpolation is
    in the space of (m,u_inv(v)); this class "re-curves" to the value function.
    '''
    def __init__(self,vFuncDecurved,CRRA):
        self.func = deepcopy(vFuncDecurved)
        self.CRRA = CRRA
        
    def __call__(self,m):
        return utility(self.func(m),gam=self.CRRA)

     
class MargValueFunc():
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
    def __init__(self,cFunc,CRRA):
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        
    def __call__(self,m):
        return utilityP(self.cFunc(m),gam=self.CRRA)
        
        
class MargMargValueFunc():
    '''
    A class for representing a marginal marginal value function in models where
    the standard envelope condition of v'(m) = u'(c(m)) holds (with CRRA utility).
    '''
    def __init__(self,cFunc,CRRA):
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        
    def __call__(self,m):
        c, MPC = self.cFunc.eval_with_derivative(m)
        return MPC*utilityPP(c,gam=self.CRRA)


####################################################################################################
####################################################################################################

class PerfectForesightSolver(object):

    def __init__(self,solution_next,DiscFac,CRRA,Rfree,PermGroFac,BoroCnst):
        self.notation = {'a': 'assets after all actions',
                         'm': 'market resources at decision time',
                         'c': 'consumption'}


        self.assignParameters(solution_next,DiscFac,CRRA,Rfree,PermGroFac,BoroCnst)
         
    def assignParameters(self,solution_next,DiscFac,CRRA,Rfree,PermGroFac,BoroCnst):
        self.solution_next  = solution_next        
        self.DiscFac        = DiscFac
        self.CRRA           = CRRA
        self.Rfree          = Rfree
        self.PermGroFac     = PermGroFac
        self.BoroCnst       = BoroCnst

    def defineBorrowingConstraint(self,BoroCnst):
        if BoroCnst is not None:
            print 'The constrained solution for the Perfect Foresight solution has not been' + \
                  ' implemented yet.  Solving the unconstrained problem.'
        self.BoroCnst = None   
    
    def defineUtilityFunctions(self):
        self.u   = lambda c : utility(c,gam=self.CRRA)
        self.uP  = lambda c : utilityP(c,gam=self.CRRA)
        self.uPP = lambda c : utilityPP(c,gam=self.CRRA)

    def defineValueFunctions(self):
        self.vFunc   = lambda m: self.u(self.cFunc(m))
        self.vPfunc  = lambda m: self.uP(self.cFunc(m))
        self.vPPfunc = lambda m: self.MPC*self.uPP(self.cFunc(m)) 

    def getcFunc(self):
        self.MPC = (((self.Rfree/self.PermGroFac) - 
                ((self.Rfree/self.PermGroFac)*((self.PermGroFac**(1-self.CRRA))*self.DiscFac)) ** \
                (1/self.CRRA))/(self.Rfree/self.PermGroFac))
        self.cFunc = lambda m: self.MPC*(m - 1 + (1/(1-(1/(self.Rfree/self.PermGroFac)))))

    def getSolution(self):
        # infinte horizon simplification.  This should hold (I think) because the range for kappa
        # and  whatever the greek sybol for Return Patience Factor is specified in the 
        # ConsumerSolution class.__init__
        
        solution_now = ConsumerSolution(cFunc=self.cFunc, vFunc=self.vFunc, 
                                      vPfunc=self.vPfunc, vPPfunc=self.vPPfunc, 
                                      mRtoMin=self.mRtoMinNow, hRto=0.0, MPCmin=1.0, 
                                      MPCmax=1.0)            
        return solution_now

    def prepareToSolve(self):
        self.defineUtilityFunctions()
        self.defineBorrowingConstraint(self.BoroCnst)

    def solve(self):        
        self.getcFunc()
        self.defineValueFunctions()
        solution = self.getSolution()        
        return solution




def perfectForesightSolver(solution_next,DiscFac,CRRA,Rfree,PermGroFac,BoroCnst):
    '''
    Solves a single period consumption - savings problem for a consumer with perfect foresight.
    '''
    solver = PerfectForesightSolver(solution_next,DiscFac,CRRA,Rfree,PermGroFac,BoroCnst)
    
    solver.prepareToSolve()
    solution = solver.solve()
    
    return solution
    

####################################################################################################
####################################################################################################
class SetupImperfectForesightSolver(PerfectForesightSolver):
    def __init__(self,solution_next,IncomeDist,LivFac,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnst,aDispGrid,vFuncBool,CubicBool):

        self.assignParameters(solution_next,IncomeDist,LivFac,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnst,aDispGrid,vFuncBool,CubicBool)
        self.defineUtilityFunctions()

    def assignParameters(self,solution_next,IncomeDist,LivFac,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnst,aDispGrid,vFuncBool,CubicBool):
        PerfectForesightSolver.assignParameters(self,solution_next,DiscFac,CRRA,Rfree,PermGroFac,BoroCnst)
        self.IncomeDist     = IncomeDist
        self.LivFac         = LivFac
        self.aDispGrid      = aDispGrid
        self.vFuncBool      = vFuncBool
        self.CubicBool      = CubicBool
        

    def defineUtilityFunctions(self):
        PerfectForesightSolver.defineUtilityFunctions(self)
        self.uPinv     = lambda u : utilityP_inv(u,gam=self.CRRA)
        self.uinvP     = lambda u : utility_invP(u,gam=self.CRRA)        
        if self.vFuncBool:
            self.uinv      = lambda u : utility_inv(u,gam=self.CRRA)


    def setAndUpdateValues(self,solution_next,IncomeDist,LivFac,DiscFac):

        self.DiscFacEff       = DiscFac*LivFac # "effective" discount factor
        self.ShkPrbsNext      = IncomeDist[0]
        self.PermShkValsNext  = IncomeDist[1]
        self.TranShkValsNext  = IncomeDist[2]
        self.PermShkMinNext   = np.min(self.PermShkValsNext)    
        self.TranShkMinNext   = np.min(self.TranShkValsNext)
        self.vPfuncNext       = solution_next.vPfunc        
        self.WorstIncPrb      = np.sum(self.ShkPrbsNext[(self.PermShkValsNext*self.TranShkValsNext)==(self.PermShkMinNext*self.TranShkMinNext)]) 

        if self.CubicBool:
            self.vPPfuncNext  = solution_next.vPPfunc
    
        # Update the bounding MPCs and PDV of human wealth:
        if self.CubicBool or self.vFuncBool:
            self.vFuncNext    = solution_next.vFunc
            self.PatFac       = ((self.Rfree*self.DiscFacEff)**(1/self.CRRA))/self.Rfree
            self.MPCminNow    = 1.0/(1.0 + self.PatFac/solution_next.MPCmin)
            self.hRtoNow      = self.PermGroFac/self.Rfree*(1.0 + solution_next.hRto)
            self.MPCmaxNow    = 1.0/(1.0 + (self.WorstIncPrb**(1.0/self.CRRA))* \
                               self.PatFac/solution_next.MPCmax)


    def defineBorrowingConstraint(self,BoroCnst):
        # Calculate the minimum allowable value of money resources in this period
        # Use np.max instead of max for future compatibility with Markov solver
        BoroCnstNat = (self.solution_next.mRtoMin - self.TranShkMinNext)*(self.PermGroFac*self.PermShkMinNext)/self.Rfree
        self.mRtoMinNow = np.max([BoroCnstNat,BoroCnst])
        if BoroCnstNat < self.mRtoMinNow: 
            self.MPCmaxNow = 1.0 # If actually constrained, MPC near limit is 1
    
        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = lambda m: m - self.mRtoMinNow


    def prepareToSolve(self):
        self.setAndUpdateValues(self.solution_next,self.IncomeDist,self.LivFac,self.DiscFac)
        self.defineBorrowingConstraint(self.BoroCnst)


####################################################################################################
####################################################################################################

class ConsumptionSavingSolverEXOG(SetupImperfectForesightSolver):
    """
    Class to set up and solve the consumption-savings problem using the method of exogenous
    gridpoints.  
    
    Everything to set up the problem is inherited from SetupImprefectForesightSolver.
    The part that solves, using the method of exogenous gridpoints specifically, is the only new 
    stuff this class.
    """

    def getSolution(self):
        # First, "unpack" things we'll need
        mRtoMinNow      = self.mRtoMinNow
        aDispGrid       = self.aDispGrid 
        Rfree           = self.Rfree
        PermGroFac      = self.PermGroFac
        CRRA            = self.CRRA
        PermShkValsNext = self.PermShkValsNext
        ShkPrbsNext     = self.ShkPrbsNext
        DiscFacEff      = self.DiscFacEff
        vPfuncNext      = self.vPfuncNext
        TranShkValsNext = self.TranShkValsNext
        cFuncNowCnst    = self.cFuncNowCnst
        vFuncBool       = self.vFuncBool
        CubicBool       = self.CubicBool     
        uP              = self.uP            
        uPP             = self.uPP
        
        if vFuncBool:        
            u           = self.u
            uinv        = self.uinv
            uinvP       = self.uinvP
            vFuncNext   = self.vFuncNext

        if CubicBool:
            vPPfuncNext = self.vPPfuncNext
    
        # Update the bounding MPCs and PDV of human wealth:
        if CubicBool or vFuncBool:
            MPCminNow   = self.MPCminNow
            MPCmaxNow    = self.MPCmaxNow
            hRtoNow     = self.hRtoNow
        
        # Find data for the unconstrained consumption function in this period
        c_temp = [0.0]  # Limiting consumption is zero as m approaches mRtoMin
        m_temp = [mRtoMinNow]
        if CubicBool:
            MPC_temp = [MPCmaxNow]
        if vFuncBool:
            vAlt_temp  = []
            vPAlt_temp = []
        for x in aDispGrid:
            mRtoNow = x + mRtoMinNow
            firstOrderCondition = lambda c : uP(c) - DiscFacEff*Rfree*PermGroFac**(-CRRA)*\
                                 np.sum(PermShkValsNext**(-CRRA)*vPfuncNext(Rfree/(PermGroFac*PermShkValsNext)*(mRtoNow-c) + TranShkValsNext)*ShkPrbsNext)
            cRtoNow = brentq(firstOrderCondition,0.001*x,0.999*x)
            c_temp.append(cRtoNow)
            m_temp.append(mRtoNow)
            if vFuncBool or CubicBool:
                mRtoNext    = Rfree/(PermGroFac*PermShkValsNext)*(mRtoNow-cRtoNow) + TranShkValsNext
            if vFuncBool:
                VLvlNext    = (PermShkValsNext**(1.0-CRRA)*PermGroFac**(1.0-CRRA))*vFuncNext(mRtoNext)
                vRtoNow     = u(cRtoNow) + DiscFacEff*np.sum(VLvlNext*ShkPrbsNext)
                vAlt_temp.append(uinv(vRtoNow)) # value transformed through inverse utility
                vPAlt_temp.append(uP(cRtoNow)*uinvP(vRtoNow))
            if CubicBool:
                gothicvPP   = DiscFacEff*Rfree*Rfree*PermGroFac**(-CRRA-1.0)*np.sum(PermShkValsNext**(-CRRA-1.0)*
                              vPPfuncNext(mRtoNext)*ShkPrbsNext)    
                dcda        = gothicvPP/uPP(cRtoNow)
                MPCnow      = dcda/(dcda+1.0)
                MPC_temp.append(MPCnow)
        
        # Construct the unconstrained consumption function
        if CubicBool:
            cFuncNowUnc = CubicInterp(m_temp,c_temp,MPC_temp,MPCminNow*hRtoNow,MPCminNow)
        else:
            cFuncNowUnc = LinearInterp(m_temp,c_temp)
    
        # Combine the constrained and unconstrained functions into the true consumption function
        cFuncNow = LowerEnvelope(cFuncNowUnc,cFuncNowCnst)
        
        # Construct the value function if requested
        if vFuncBool:
            MPCminAlt      = MPCminNow**(-CRRA/(1-CRRA))
            mRtoGrid       = np.asarray(aDispGrid) + mRtoMinNow
            vAltFuncNow    = CubicInterp(mRtoGrid,vAlt_temp,vPAlt_temp,MPCminAlt*hRtoNow,MPCminAlt)
            vFuncNow       = lambda m : u(vAltFuncNow(m))
            
        # Make the marginal value function and the marginal marginal value function
        vPfuncNow = lambda m : uP(cFuncNow(m))
        if CubicBool:
            vPPfuncNow = lambda m : cFuncNow.derivative(m)*uPP(cFuncNow(m))
    
        # Store the results in a solution object and return it
        if CubicBool or vFuncBool:
            solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mRtoMin=mRtoMinNow, 
                                          hRto=hRtoNow, MPCmin=MPCminNow, 
                                          MPCmax=MPCmaxNow)
        else:
            solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mRtoMin=mRtoMinNow)
        if vFuncBool:
            solution_now.vFunc = vFuncNow
        if CubicBool:
            solution_now.vPPfunc=vPPfuncNow
        #print('Solved a period with EXOG!')
        return solution_now


def consumptionSavingSolverEXOG(solution_next,IncomeDist,LivFac,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnst,aDispGrid,vFuncBool,CubicBool):
    '''
    Solves a single period consumption - savings problem for a consumer with perfect foresight.
    '''
    solver = ConsumptionSavingSolverEXOG(solution_next,IncomeDist,LivFac,
                                         DiscFac,CRRA,Rfree,PermGroFac,BoroCnst,aDispGrid,vFuncBool,CubicBool)    
    solver.prepareToSolve()        
    solution = solver.solve()    
    return solution

####################################################################################################
####################################################################################################

class ConsumptionSavingSolverENDGBasic(SetupImperfectForesightSolver):
    """
    This class solves a single period of a standard consumption-saving problem, using linear 
    interpolation and without the ability to calculate the value function.  
    ConsumptionSavingSolverENDG inherits from this class and adds the ability to perform
    cubic interpolation and to calculate the value function.
    
    Note that this class does not have its own initializing method.  It initializes the same 
    problem in the same way as ConsumptionSavingSolverEXOG, which it inherits from.  
    It just solves the problem differently.

    Parameters:
    -----------
    solution_next: ConsumerSolution
        The solution to the following period.
    IncomeDist: [[float]]
        A list containing three lists of floats, representing a discrete approximation to the income
        process between the period being solved and the one immediately following (in solution_next).
        Order: probs, psi, xi
    LivFac: float
        Probability of surviving to succeeding period.
    DiscFac: float
        Discount factor between this period and the succeeding period.
    CRRA: float
        The coefficient of relative risk aversion
    Rfree: float
        Interest factor on assets between this period and the succeeding period: w_tp1 = a_t*Rfree
    PermGroFac: float
        Expected growth factor for permanent income between this period and the succeeding period.
    BoroCnst: float
        Borrowing constraint for the minimum allowable assets to end the period
        with.  If it is less than the natural borrowing constraint, then it is
        irrelevant; BoroCnst=None indicates no artificial borrowing constraint.
    aDispGrid: np.array
        An array of end-of-period asset values (post-decision states) at which to solve for optimal 
        consumption.
    vFuncBool: Boolean
        An indicator for whether the value function should be computed and included
        in the reported solution.  Should be false for an instance of this class.
    CubicBool: Boolean
        An indicator for whether the solver should use cubic or linear interpolation
        Should be false for an instance of this class.

    Returns:
    -----------
    solution_now: ConsumerSolution
        The solution to this period's problem, obtained using the method of endogenous gridpoints.
    """    

    def prepareToGetGothicvP(self):
        """
        Create and return arrays of assets and income shocks we will need to compute GothicvP.
        """               
        aRtoNow     = np.asarray(self.aDispGrid) + self.mRtoMinNow
        ShkCount    = self.TranShkValsNext.size
        aRto_temp   = np.tile(aRtoNow,(ShkCount,1))

        # Tile arrays of the income shocks and put them into useful shapes
        aRtoCount         = aRtoNow.shape[0]
        PermShkVals_temp  = (np.tile(self.PermShkValsNext,(aRtoCount,1))).transpose()
        TranShkVals_temp  = (np.tile(self.TranShkValsNext,(aRtoCount,1))).transpose()
        ShkPrbs_temp      = (np.tile(self.ShkPrbsNext,(aRtoCount,1))).transpose()
        
        # Get cash on hand next period
        mRtoNext          = self.Rfree/(self.PermGroFac*PermShkVals_temp)*aRto_temp + TranShkVals_temp

        # Store and report the results
        self.PermShkVals_temp  = PermShkVals_temp
        self.ShkPrbs_temp      = ShkPrbs_temp
        self.mRtoNext          = mRtoNext                
        return aRtoNow


    def getGothicvP(self):
        """
        Find data for the unconstrained consumption function in this period.
        """        
        gothicvP  = self.DiscFacEff*self.Rfree*self.PermGroFac**(-self.CRRA)*np.sum(
                    self.PermShkVals_temp**(-self.CRRA)*self.vPfuncNext(self.mRtoNext)*self.ShkPrbs_temp,axis=0)
                    
        return gothicvP
                    

    def getPointsForInterpolation(self,gothicvP,aRtoNow):

        cRtoNow = self.uPinv(gothicvP)
        mRtoNow = cRtoNow + aRtoNow

        # Limiting consumption is zero as m approaches mRtoMin
        c_for_interpolation = np.insert(cRtoNow,0,0.,axis=-1)
        m_for_interpolation = np.insert(mRtoNow,0,self.mRtoMinNow,axis=-1)
        
        # Store these for calcvFunc
        self.cRtoNow = cRtoNow
        self.mRtoNow = mRtoNow
        
        return c_for_interpolation,m_for_interpolation

    def usePointsForInterpolation(self,cRto,mRto,interpolator):

        # Construct the unconstrained consumption function
        cFuncNowUnc = interpolator(mRto,cRto)

        # Combine the constrained and unconstrained functions into the true consumption function
        cFuncNow = LowerEnvelope(cFuncNowUnc,self.cFuncNowCnst)

        # Make the marginal value function and the marginal marginal value function
        vPfuncNow = lambda m : self.uP(cFuncNow(m))

        # Pack up the solution and return it
        solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mRtoMin=self.mRtoMinNow)
        return solution_now        


    def getSolution(self,gothicvP,aRto,interpolator = LinearInterp):
        """
        Given a and gothicvP, return the solution for this period.
        """
        cRto,mRto  = self.getPointsForInterpolation(gothicvP,aRto)       
        solution_now = self.usePointsForInterpolation(cRto,mRto,interpolator)
        return solution_now
        
                
    def solve(self):
        aRto       = self.prepareToGetGothicvP()           
        gothicvP   = self.getGothicvP()                        
        solution   = self.getSolution(gothicvP,aRto)
        #print('Solved a period with ENDG!')
        return solution        
       

####################################################################################################
####################################################################################################

class ConsumptionSavingSolverENDG(ConsumptionSavingSolverENDGBasic):
    """
    Method that adds value function, cubic interpolation to ENDG 
    """

    def getConsumptionCubic(self,mRto,cRto):
        """
        Interpolate the unconstrained consumption function with cubic splines
        """        
        gothicvPP   = self.DiscFacEff*self.Rfree*self.Rfree*self.PermGroFac**(-self.CRRA-1.0)* \
                      np.sum(self.PermShkVals_temp**(-self.CRRA-1.0)*self.vPPfuncNext(self.mRtoNext)*self.ShkPrbs_temp,
                             axis=0)    
        dcda        = gothicvPP/self.uPP(np.array(cRto[1:]))
        MPC         = dcda/(dcda+1.)
        MPC         = np.insert(MPC,0,self.MPCmaxNow)

        cFunc_t_unconstrained = CubicInterp(mRto,cRto,MPC,self.MPCminNow*self.hRtoNow,self.MPCminNow)
        return cFunc_t_unconstrained


    def putVfuncInSolution(self,solution,gothicvP):
        # Construct the value function if requested

        VLvlNext    = (self.PermShkVals_temp**(1.0-self.CRRA)*self.PermGroFac**(1.0-self.CRRA))*self.vFuncNext(self.mRtoNext)
        gothicv     = self.DiscFacEff*np.sum(VLvlNext*self.ShkPrbs_temp,axis=0)
        vRtoNow     = self.u(self.cRtoNow) + gothicv
        vAlt_temp   = self.uinv(vRtoNow) # value transformed through inverse utility
        vAltP_temp  = gothicvP*self.uinvP(vRtoNow)
        MPCminAlt   = self.MPCminNow**(-self.CRRA/(1.0-self.CRRA))
        vAltFuncNow = CubicInterp(self.mRtoNow,vAlt_temp,vAltP_temp,MPCminAlt*self.hRtoNow,MPCminAlt)
        vFuncNow    = lambda m : self.u(vAltFuncNow(m))        

        solution.vFunc = vFuncNow        
        return solution


    def prepForCubicSplines(self,solution):
        """
        Take a solution, and add in vPPfunc to it, to prepare for cubic splines
        """
        vPPfuncNow = MargMargValueFunc(solution.cFunc,self.CRRA)
        solution.vPPfunc=vPPfuncNow
        return solution


    def addMPCandHumanWealth(self,solution):
        """
        Take a solution, and other things to it
        """
        solution.hRto = self.hRtoNow
        solution.MPCmin = self.MPCminNow
        solution.MPCmax = self.MPCmaxNow
        return solution

       
    def solve(self):        
        aRto       = self.prepareToGetGothicvP()           
        gothicvP   = self.getGothicvP()
        
        if self.CubicBool:
            solution   = self.getSolution(gothicvP,aRto,interpolator=self.getConsumptionCubic)
        else:
            solution   = self.getSolution(gothicvP,aRto)
        
        if self.vFuncBool:
            solution = self.putVfuncInSolution(solution,gothicvP)
        if self.CubicBool: 
            solution = self.prepForCubicSplines(solution)
            
        if self.vFuncBool or self.CubicBool:
            solution = self.addMPCandHumanWealth(solution)
        #print('Solved a period with ENDG!')
        return solution        
       


def consumptionSavingSolverENDG(solution_next,IncomeDist,LivFac,DiscFac,CRRA,Rfree,PermGroFac,BoroCnst,aDispGrid,vFuncBool,CubicBool):
                                       
    if (not CubicBool) and (not vFuncBool):
        solver = ConsumptionSavingSolverENDGBasic(solution_next,IncomeDist,LivFac,DiscFac,CRRA,Rfree,PermGroFac,BoroCnst,aDispGrid,
                                             vFuncBool,CubicBool)        
    else:

        solver = ConsumptionSavingSolverENDG(solution_next,IncomeDist,LivFac,DiscFac,CRRA,Rfree,PermGroFac,BoroCnst,aDispGrid,
                                             vFuncBool,CubicBool)

    solver.prepareToSolve()                      
    solution                   = solver.solve()

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
    IncomeDist: [[float]]
        A list containing three lists of floats, representing a discrete approximation to the income
        process between the period being solved and the one immediately following (in solution_next).
        Order: probs, psi, xi
    LivFac: float
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
    BoroCnst: float
        Borrowing constraint for the minimum allowable assets to end the period
        with.  If it is less than the natural borrowing constraint, then it is
        irrelevant; BoroCnst=None indicates no artificial borrowing constraint.
    aDispGrid: [float]
        A list of end-of-period asset values (post-decision states) at which to solve for optimal
        consumption.

    Returns:
    -----------
    solution_now: ConsumerSolution
        The solution to this period's problem, obtained using the method of endogenous gridpoints.


    """


    def __init__(self,solution_next,IncomeDist,LivFac,DiscFac,CRRA,
                      Rboro,Rsave,PermGroFac,BoroCnst,aDispGrid,vFuncBool,CubicBool):        

        assert CubicBool==False,'KinkedR will only work with linear interpolation'

        # Initialize the solver.  Most of the steps are exactly the same as in the Endogenous Grid
        # linear case, so start with that.
        ConsumptionSavingSolverENDG.__init__(self,solution_next,IncomeDist,
                                                   LivFac,DiscFac,CRRA,Rboro,PermGroFac,BoroCnst,
                                                   aDispGrid,vFuncBool,CubicBool) 

        # Assign the interest rates as class attributes, to use them later.
        self.Rboro   = Rboro
        self.Rsave   = Rsave


    def prepareToGetGothicvP(self):
        """
        Method to prepare for calculating gothicvP.
        
        This differs from the baseline case because different savings choices yield different
        interest rates.
        """
        
        aRtoNow           = np.sort(np.hstack((np.asarray(self.aDispGrid) + 
                            self.mRtoMinNow,np.array([0.0,0.0]))))
        aDispCount        = aRtoNow.size
        ShkCount          = self.TranShkValsNext.size
        aRto_temp         = np.tile(aRtoNow,(ShkCount,1))
        PermShkVals_temp  = (np.tile(self.PermShkValsNext,(aDispCount,1))).transpose()
        TranShkVals_temp  = (np.tile(self.TranShkValsNext,(aDispCount,1))).transpose()
        ShkPrbs_temp      = (np.tile(self.ShkPrbsNext,(aDispCount,1))).transpose()

        Rfree_vec         = self.Rsave*np.ones(aDispCount)
        Rfree_vec[0:(np.sum(aRtoNow<=0)-1)] = self.Rboro
        self.Rfree        = Rfree_vec

        Rfree_temp        = np.tile(Rfree_vec,(ShkCount,1))
        mRtoNext          = Rfree_temp/(self.PermGroFac*PermShkVals_temp)*aRto_temp + TranShkVals_temp

        self.PermShkVals_temp = PermShkVals_temp
        self.ShkPrbs_temp     = ShkPrbs_temp
        self.mRtoNext         = mRtoNext

        return aRtoNow


def consumptionSavingSolverKinkedR(solution_next,IncomeDist,
                                   LivFac,DiscFac,CRRA,Rboro,Rsave,PermGroFac,BoroCnst,
                                   aDispGrid,vFuncBool,CubicBool):

    solver = ConsumptionSavingSolverKinkedR(solution_next,IncomeDist,LivFac,
                                            DiscFac,CRRA,Rboro,Rsave,PermGroFac,BoroCnst,aDispGrid,
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
    transition_array : numpy.array
        An NxN array representing a Markov transition matrix between discrete
        states.  The i,j-th element of transition_array is the probability of
        moving from state i in period t to state j in period t+1.
    IncomeDist: [[[numpy.array]]]
        A list of lists containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved and
        the one immediately following (in solution_next).  Order: probs, psi, xi.
        The n-th element of IncomeDist is the income distribution for the n-th
        discrete state.
    LivFac: float
        Probability of surviving to succeeding period.
    DiscFac: float
        Discount factor between this period and the succeeding period.
    CRRA: float
        The coefficient of relative risk aversion
    Rfree: float
        Interest factor on assets between this period and the succeeding period: w_tp1 = a_t*Rfree
    PermGroFac: float
        Expected growth factor for permanent income between this period and the succeeding period.
    BoroCnst: float
        Borrowing constraint for the minimum allowable assets to end the period
        with.  If it is less than the natural borrowing constraint, then it is
        irrelevant; BoroCnst=None indicates no artificial borrowing constraint.
    aDispGrid: [float]
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

    def __init__(self,solution_next,IncomeDist_list,p_zero_income_array,LivFac,DiscFac,
                      CRRA,Rfree,PermGroFac,transition_array,BoroCnst,aDispGrid,vFuncBool,CubicBool):

        ConsumptionSavingSolverENDG.assignParameters(self,solution_next,np.nan,
                                                     LivFac,DiscFac,CRRA,Rfree,PermGroFac,
                                                     BoroCnst,aDispGrid,vFuncBool,CubicBool)
                                                     
        
        self.defineUtilityFunctions()
        self.IncomeDist_list      = IncomeDist_list
        self.p_zero_income_array  = p_zero_income_array
        self.StateCount           = p_zero_income_array.size
        self.transition_array     = transition_array

    def conditionOnState(self,state_index):
        """
        Find the income distribution, etc., conditional on a given state next period
        """
        self.IncomeDist     = self.IncomeDist_list[state_index]
        self.p_zero_income  = self.p_zero_income_array[state_index] 
        self.vPfuncNext     = self.solution_next.vPfunc[state_index]
        self.mRtoMinNow     = self.mRtoMin_list[state_index] 
        
        self.cFuncNowCnst   = lambda m: m - self.mRtoMinNow

        if self.CubicBool:
            self.vPPfuncNext= self.solution_next.vPPfunc[state_index]
        if self.vFuncBool:
            self.vFuncNext  = self.solution_next.vFunc[state_index]


    def defineBorrowingConstraint(self):

        # Find the borrowing constraint for each current state i as well as the
        # probability of receiving zero income.
        self.p_zero_income_now  = np.dot(self.transition_array,self.p_zero_income_array)
        mRtoMinAll              = np.zeros(self.StateCount) + np.nan
        for j in range(self.StateCount):
            PermShkMinNext      = np.min(self.IncomeDist_list[j][1])
            TranShkMinNext      = np.min(self.IncomeDist_list[j][2])
            mRtoMinAll[j]       = max((self.solution_next.mRtoMin[j] - TranShkMinNext)*
                                  (self.PermGroFac*PermShkMinNext)/self.Rfree, self.BoroCnst)
        self.mRtoMin_list       = np.zeros(self.StateCount) + np.nan
        for i in range(self.StateCount):
            possible_next_states = self.transition_array[i,:] > 0
            self.mRtoMin_list[i]   = np.max(mRtoMinAll[possible_next_states])

    def solve(self):
        self.defineBorrowingConstraint()
                
        gothicvP_cond      = np.zeros([self.StateCount,self.aDispGrid.size]) + np.nan     
        if self.vFuncBool:
            gothicv_cond   = np.zeros((self.StateCount,self.aDispGrid.size))
        if self.CubicBool:
            gothicvPP_cond = np.zeros((self.StateCount,self.aDispGrid.size))
        if self.vFuncBool or self.CubicBool:
            ExIncNext      = np.zeros(self.StateCount) + np.nan

        for j in range(self.StateCount):
            self.conditionOnState(j)
            self.setAndUpdateValues(self.solution_next,self.IncomeDist,
                                    self.LivFac,self.DiscFac)

            # We need to condition on the state again, because self.setAndUpdateValues sets 
            # self.vPfunc_tp1       = solution_next.vPfunc... may want to fix this later.             
            self.conditionOnState(j)
            if self.vFuncBool or self.CubicBool:
                ExIncNext[j] = np.dot(self.ShkPrbsNext,self.PermShkValsNext*self.TranShkValsNext)
                # Line above might need a PermGroFac in it

            self.prepareToGetGothicvP()  
            gothicvP_cond[j,:] = self.getGothicvP()                        

            if self.vFuncBool:
                VLvlNext            = (self.PermShkVals_temp**(1.0-self.CRRA)*self.PermGroFac**(1.0-self.CRRA))*self.vFuncNext(self.mRtoNext)
                gothicv_cond[j,:]   = self.DiscFacEff*np.sum(VLvlNext*self.ShkPrbs_temp,axis=0)
    
            if self.CubicBool:
                gothicvPP_cond[j,:] = self.DiscFacEff*self.Rfree*self.Rfree*self.PermGroFac**(-self.CRRA-1.0)*np.sum(self.PermShkVals_temp**(-self.CRRA-1.0)*
                                       self.vPPfuncNext(self.mRtoNext)*self.ShkPrbs_temp,axis=0)

        # gothicvP_cond is gothicvP conditional on *next* period's state.
        # Take expectations to get gothicvP conditional on *this* period's state.
        gothicvP      = np.dot(self.transition_array,gothicvP_cond)  
 
        # Calculate the bounding MPCs and PDV of human wealth for each state
        if self.vFuncBool or self.CubicBool:
            ExMPCmaxNext = (np.dot(self.transition_array,self.p_zero_income_array*self.solution_next.MPCmax**(-self.CRRA))/
                                 self.p_zero_income_now)**(-1/self.CRRA) # expectation of upper bound on MPC in t+1 from perspective of t
            self.MPCmaxNow       = 1.0/(1.0 + (self.p_zero_income_now**(1.0/self.CRRA))*self.PatFac/ExMPCmaxNext)
   
        if self.CubicBool:
            self.gothicvPP = np.dot(self.transition_array,gothicvPP_cond)
            self.vPPfuncNext_list = self.vPPfuncNext
        
        # note I'm not sure mRtoMin_list is what it should be... CHECK
        aRto = np.asarray(self.aDispGrid)[np.newaxis,:] + np.array(self.mRtoMin_list)[:,np.newaxis]
        cRto,mRto        = self.getPointsForInterpolation(gothicvP,aRto)
       
        solution = self.usePointsForInterpolation(cRto,mRto,interpolator=LinearInterp)

        if self.vFuncBool or self.CubicBool:
            solution = self.addMPCandHumanWealth(solution)


        return solution    

    def usePointsForInterpolation(self,cRto,mRto,interpolator):
        solution = ConsumerSolution()
        if self.CubicBool:
            dcda          = self.gothicvPP/self.uPP(np.array(self.cRtoNow))
            MPC           = dcda/(dcda+1.0)
            self.MPC_temp = np.hstack((np.reshape(self.MPCmaxNow,(self.StateCount,1)),MPC))  
            interpfunc    = self.getConsumptionCubic            
        else:
            interpfunc = LinearInterp
        
        for j in range(self.StateCount):
            if self.CubicBool:
                self.MPC_temp_j = self.MPC_temp[j,:]

            solution_cond = ConsumptionSavingSolverENDGBasic.usePointsForInterpolation(
                                   self,cRto[j,:],mRto[j,:],interpolator=interpfunc)            
            if self.CubicBool: 
                solution_cond = self.prepForCubicSplines(solution_cond)

            solution.appendSolution(solution_cond)
            
        return solution

    def getConsumptionCubic(self,mRto,cRto):
        """
        Interpolate the unconstrained consumption function with cubic splines
        """
        
        cFuncNowUnc = CubicInterp(mRto,cRto,self.MPC_temp_j,
                                                   self.MPCminNow*self.hRtoNow,
                                                   self.MPCminNow)
        return cFuncNowUnc



def consumptionSavingSolverMarkov(solution_next,IncomeDist,p_zero_income,LivFac,
                                      DiscFac,CRRA,Rfree,PermGroFac,transition_array,BoroCnst,aDispGrid,vFuncBool,CubicBool):
                                       
    solver = ConsumptionSavingSolverMarkov(solution_next,IncomeDist,p_zero_income,
                                               LivFac,DiscFac,CRRA,Rfree,PermGroFac,transition_array,BoroCnst,aDispGrid,
                                               vFuncBool,CubicBool)              
    solution                   = solver.solve()
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
    #cFunc_terminal_ = Cubic1DInterpDecay([0.0, 1.0],[0.0, 1.0],[1.0, 1.0],0,1)
    cFunc_terminal_      = LinearInterp([0.0, 1.0],[0.0,1.0])
    vFunc_terminal_      = LinearInterp([0.0, 1.0],[0.0,0.0])
    cFuncCnst_terminal_  = lambda x: x
    solution_terminal_   = ConsumerSolution(cFunc=LowerEnvelope(cFunc_terminal_,cFuncCnst_terminal_),
                                            vFunc = vFunc_terminal_, mRtoMin=0.0, hRto=0.0, 
                                            MPCmin=1.0, MPCmax=1.0)
    time_vary_ = ['LivFac','DiscFac','PermGroFac']
    time_inv_ = ['CRRA','Rfree','aDispGrid','BoroCnst','vFuncBool','CubicBool']
    
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
            
    def updateIncomeProcess(self):
        '''
        Updates this agent's income process based on his own attributes.  The
        function that generates the discrete income process can be swapped out
        for a different process.
        '''
        original_time = self.time_flow
        self.timeFwd()
        IncomeDist = constructLognormalIncomeProcessUnemployment(self)
        self.IncomeDist             = IncomeDist
        if not 'IncomeDist' in self.time_vary:
            self.time_vary.append('IncomeDist')
        if not original_time:
            self.timeRev()
            
    def updateAssetsGrid(self):
        '''
       Updates this agent's end-of-period assets grid.
        '''
        aDispGrid = constructAssetsGrid(self)
        self.aDispGrid = aDispGrid
        
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
        
    def calcBoundingValues(self):
        '''
        Calculate the PDV of human wealth (after receiving income this period)
        in an infinite horizon model with only one period repeated indefinitely.
        Also calculates MPCmin and MPCmax for infinite horizon.
        '''
        if hasattr(self,'transition_array'):
            StateCount = self.p_zero_income[0].size
            ExIncNext = np.zeros(StateCount) + np.nan
            for j in range(StateCount):
                PermShkValsNext = self.IncomeDist[0][j][1]
                TranShkValsNext = self.IncomeDist[0][j][2]
                ShkPrbsNext     = self.IncomeDist[0][j][0]
                ExIncNext[j] = np.dot(ShkPrbsNext,PermShkValsNext*TranShkValsNext)                
            hRto        = np.dot(np.dot(np.linalg.inv((self.Rfree/self.PermGroFac[0])*np.eye(StateCount) -
                              self.transition_array),self.transition_array),ExIncNext)
            
            p_zero_income_now = np.dot(self.transition_array,self.p_zero_income[0])
            PatFac            = (self.DiscFac[0]*self.Rfree)**(1.0/self.CRRA)/self.Rfree
            MPCmax            = 1.0 - p_zero_income_now**(1.0/self.CRRA)*PatFac # THIS IS WRONG
            
        else:
            PermShkValsNext   = self.IncomeDist[0][1]
            TranShkValsNext   = self.IncomeDist[0][2]
            ShkPrbsNext       = self.IncomeDist[0][0]
            ExIncNext         = np.dot(ShkPrbsNext,PermShkValsNext*TranShkValsNext)
            hRto              = (ExIncNext*self.PermGroFac[0]/self.Rfree)/(1.0-self.PermGroFac[0]/self.Rfree)
            
            PatFac    = (self.DiscFac[0]*self.Rfree)**(1.0/self.CRRA)/self.Rfree
            MPCmax    = 1.0 - self.p_zero_income[0]**(1.0/self.CRRA)*PatFac
        
        MPCmin = 1.0 - PatFac
        return hRto, MPCmax, MPCmin



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
    IncomeDist:  [income distribution]
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
    
    IncomeDist = [] # Discrete approximation to income process

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
        IncomeDistRet = [ShkPrbsRet,PermShkValsRet,TranShkValsRet]

    # Loop to fill in the list of IncomeDist random variables.
    for t in range(T_total): # Iterate over all periods, counting forward

        if T_retire > 0 and t >= T_retire:
            # Then we are in the "retirement period" and add a retirement income object.
            IncomeDist.append(deepcopy(IncomeDistRet))
        else:
            # We are in the "working life" periods.
            TranShkDist     = approxMeanOneLognormal(N=TranShkCount, sigma=TranShkStd[t])
            if UnempPrb > 0:
                TranShkDist = addDiscreteOutcomeConstantMean(TranShkDist, p=UnempPrb, x=IncUnemp)
            PermShkDist     = approxMeanOneLognormal(N=PermShkCount, sigma=PermShkStd[t])
            IncomeDist.append(combineIndepDists(PermShkDist,TranShkDist))

    return IncomeDist
    



def applyFlatIncomeTax(IncomeDist,tax_rate,T_retire,unemployed_indices=[],transitory_index=2):
    '''
    Applies a flat income tax rate to all employed income states during the working
    period of life (those before T_retire).  Time runs forward in this function.
    
    Parameters:
    -------------
    IncomeDist : [income distributions]
        The discrete approximation to the income distribution in each time period.
    tax_rate : float
        A flat income tax rate to be applied to all employed income.
    T_retire : int
        The time index after which the agent retires.
    unemployed_indices : [int]
        Indices of transitory shocks that represent unemployment states (no tax).
    transitory_index : int
        The index of each element of IncomeDist representing transitory shocks.
        
    Returns:
    ------------
    IncomeDist_new : [income distributions]
        The updated income distributions, after applying the tax.
    '''
    IncomeDist_new = deepcopy(IncomeDist)
    i = transitory_index
    for t in range(len(IncomeDist)):
        if t < T_retire:
            for j in range((IncomeDist[t][i]).size):
                if j not in unemployed_indices:
                    IncomeDist_new[t][i][j] = IncomeDist[t][i][j]*(1-tax_rate)
    return IncomeDist_new
   
    
    
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
    Rfree : [float]
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
    psi_seed : int
        Seed for random number generator, permanent income shocks.
    xi_seed : int
        Seed for random number generator, temporary income shocks.
    unemp_seed : int
        Seed for random number generator, unemployment shocks.

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
    TranShkStd                = parameters.TranShkStd
    PermGroFac                   = parameters.PermGroFac
    Rfree                       = parameters.Rfree
    UnempPrb              = parameters.UnempPrb
    UnempPrbRet       = parameters.UnempPrbRet
    IncUnemp         = parameters.IncUnemp
    IncUnempRet  = parameters.IncUnempRet
    T_retire                = parameters.T_retire
    Nagents                 = parameters.Nagents
    psi_seed                = parameters.psi_seed
    xi_seed                 = parameters.xi_seed
    unemp_seed              = parameters.unemp_seed
    tax_rate                = parameters.tax_rate

    # Truncate the lifecycle vectors to the working life
    PermShkStdWork   = PermShkStd[0:T_retire]
    TranShkStdWork   = TranShkStd[0:T_retire]
    PermGroFacWork    = PermGroFac[0:T_retire]
    PermGroFacRet    = PermGroFac[T_retire:]
    working_periods  = len(PermGroFacWork) + 1
    retired_periods  = len(PermGroFacRet)
    
    # Generate transitory shocks in the working period (needs one extra period)
    TranShkHistWork = drawMeanOneLognormal(TranShkStdWork, Nagents, xi_seed)
    np.random.seed(0)
    TranShkHistWork.insert(0,np.random.permutation(TranShkHistWork[0]))
    
    # Generate permanent shocks in the working period
    PermShkHistWork = drawMeanOneLognormal(PermShkStdWork, Nagents, psi_seed)
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
    UnempHist = drawBernoulli(UnempPrbLife,Nagents,unemp_seed)   
    
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
    psi_seed : int
        Seed for random number generator, permanent income shocks.
    xi_seed : int
        Seed for random number generator, temporary income shocks.
    unemp_seed : int
        Seed for random number generator, unemployment shocks.
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
    psi_seed       = parameters.psi_seed
    xi_seed        = parameters.xi_seed
    unemp_seed     = parameters.unemp_seed
    sim_periods    = parameters.sim_periods
    
    TranShkHist    = drawMeanOneLognormal(sim_periods*TranShkStd, Nagents, xi_seed)
    UnempHist      = drawBernoulli(sim_periods*[UnempPrb],Nagents,unemp_seed)
    PermShkHist    = drawMeanOneLognormal(sim_periods*PermShkStd, Nagents, psi_seed)
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
    aDispMin:                  float
        Minimum value for the a-grid
    aDispMax:                  float
        Maximum value for the a-grid
    aDispCount:                 int
        Size of the a-grid
    aDispExtra:                [float]
        Extra values for the a-grid.
    grid_type:              string
        String indicating the type of grid. "linear" or "exp_mult"
    exp_nest:               int
        Level of nesting for the exponentially spaced grid
        
    Returns:
    ----------
    aDispGrid:     np.ndarray
        Base array of values for the post-decision-state grid.
    '''
    # Unpack the parameters
    aDispMin     = parameters.aDispMin
    aDispMax     = parameters.aDispMax
    aDispCount    = parameters.aDispCount
    aDispExtra   = parameters.aDispExtra
    grid_type = 'exp_mult'
    exp_nest  = parameters.exp_nest
    
    # Set up post decision state grid:
    aDispGrid = None
    if grid_type == "linear":
        aDispGrid = np.linspace(aDispMin, aDispMax, aDispCount)
    elif grid_type == "exp_mult":
        aDispGrid = makeGridExpMult(ming=aDispMin, maxg=aDispMax, ng=aDispCount, timestonest=exp_nest)
    else:
        raise Exception, "grid_type not recognized in __init__." + \
                         "Please ensure grid_type is 'linear' or 'exp_mult'"

    # Add in additional points for the grid:
    for a in aDispExtra:
        if (a is not None):
            if a not in aDispGrid:
                j      = aDispGrid.searchsorted(a)
                aDispGrid = np.insert(aDispGrid, j, a)

    return aDispGrid
    
    
if __name__ == '__main__':
    import SetupConsumerParameters as Params
    from HARKutilities import plotFunc, plotFuncDer, plotFuncs
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)

    do_hybrid_type          = False
    do_markov_type          = True
    do_perfect_foresight    = False





####################################################################################################    
    
#    # Make and solve a finite consumer type
    LifecycleType = ConsumerType(**Params.init_consumer_objects)
#    LifecycleType.solveOnePeriod = consumptionSavingSolverEXOG
    LifecycleType.solveOnePeriod = consumptionSavingSolverENDG
    
    start_time = clock()
    LifecycleType.solve()
    end_time = clock()
    print('Solving a lifecycle consumer took ' + mystr(end_time-start_time) + ' seconds.')
    LifecycleType.unpack_cFunc()
    LifecycleType.timeFwd()
    
    # Plot the consumption functions during working life
    print('Consumption functions while working:')
    plotFuncs(LifecycleType.cFunc[:40],0,5)

    # Plot the consumption functions during retirement
    print('Consumption functions while retired:')
    plotFuncs(LifecycleType.cFunc[40:],0,5)
    LifecycleType.timeRev()
    
    
    
    
####################################################################################################    
    
    
    # Make and solve an infinite horizon consumer
    InfiniteType = deepcopy(LifecycleType)
    InfiniteType.assignParameters(    LivFac = [0.98],
                                      DiscFac = [0.96],
                                      PermGroFac = [1.01],
                                      cycles = 0) # This is what makes the type infinite horizon
    InfiniteType.IncomeDist = [LifecycleType.IncomeDist[-1]]
    
    start_time = clock()
    InfiniteType.solve()
    end_time = clock()
    print('Solving an infinite horizon consumer took ' + mystr(end_time-start_time) + ' seconds.')
    InfiniteType.unpack_cFunc()
    
    # Plot the consumption function and MPC for the infinite horizon consumer
    print('Consumption function:')
    plotFunc(InfiniteType.cFunc[0],InfiniteType.solution[0].mRtoMin,5)    # plot consumption
    print('Marginal consumption function:')
    plotFuncDer(InfiniteType.cFunc[0],InfiniteType.solution[0].mRtoMin,5) # plot MPC
    if InfiniteType.vFuncBool:
        print('Value function:')
        plotFunc(InfiniteType.solution[0].vFunc,0.5,10)




    
####################################################################################################    




        
        
    # Make and solve an agent with a kinky interest rate
    KinkyType = deepcopy(InfiniteType)

    KinkyType.time_inv.remove('Rfree')
    KinkyType.time_inv += ['Rboro','Rsave']
    KinkyType(Rboro = 1.2, Rsave = 1.03, BoroCnst = None, aDispCount = 48, cycles=0, CubicBool = False)

    KinkyType.solveOnePeriod = consumptionSavingSolverKinkedR
    KinkyType.updateAssetsGrid()
    
    start_time = clock()
    KinkyType.solve()
    end_time = clock()
    print('Solving a kinky consumer took ' + mystr(end_time-start_time) + ' seconds.')
    KinkyType.unpack_cFunc()
    print('Kinky consumption function:')
    KinkyType.timeFwd()
    plotFunc(KinkyType.cFunc[0],KinkyType.solution[0].mRtoMin,5)


    
####################################################################################################    


    
    # Make and solve a "cyclical" consumer type who lives the same four quarters repeatedly.
    # The consumer has income that greatly fluctuates throughout the year.
    CyclicalType = deepcopy(LifecycleType)
    CyclicalType.assignParameters(LivFac = [0.98]*4,
                                      DiscFac = [0.96]*4,
                                      PermGroFac = [1.1, 0.3, 2.8, 1.1],
                                      cycles = 0) # This is what makes the type (cyclically) infinite horizon)
    CyclicalType.IncomeDist = [LifecycleType.IncomeDist[-1]]*4
    
    start_time = clock()
    CyclicalType.solve()
    end_time = clock()
    print('Solving a cyclical consumer took ' + mystr(end_time-start_time) + ' seconds.')
    CyclicalType.unpack_cFunc()
    CyclicalType.timeFwd()
    
    # Plot the consumption functions for the cyclical consumer type
    print('Quarterly consumption functions:')
    plotFuncs(CyclicalType.cFunc,0,5)
    
    
####################################################################################################    
    
    
    # Make and solve a "hybrid" consumer who solves an infinite horizon problem by
    # alternating between ENDG and EXOG each period.  Yes, this is weird.
    if do_hybrid_type:
        HybridType = deepcopy(InfiniteType)
        HybridType.assignParameters(LivFac = 2*[0.98],
                                      DiscFac = 2*[0.96],
                                      PermGroFac = 2*[1.01])
        HybridType.IncomeDist = 2*[LifecycleType.IncomeDist[-1]]
        HybridType.time_vary.append('solveOnePeriod')
        HybridType.solveOnePeriod = [consumptionSavingSolverENDG,consumptionSavingSolverEXOG] # alternated between ENDG and EXOG
        
        start_time = clock()
        HybridType.solve()
        end_time = clock()
        print('Solving a "hybrid" consumer took ' + mystr(end_time-start_time) + ' seconds.')
        HybridType.unpack_cFunc()
        
        # Plot the consumption function for the cyclical consumer type
        print('"Hybrid solver" consumption function:')
        plotFunc(HybridType.cFunc[0],0,5)
#        
#    
#    # Make and solve a type that has serially correlated unemployment   
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
        transition_array = np.array([[(1-p_unemploy_good)*(1-bust_prob),p_unemploy_good*(1-bust_prob),(1-p_unemploy_good)*bust_prob,p_unemploy_good*bust_prob],
                                      [p_reemploy*(1-bust_prob),(1-p_reemploy)*(1-bust_prob),p_reemploy*bust_prob,(1-p_reemploy)*bust_prob],
                                      [(1-p_unemploy_bad)*boom_prob,p_unemploy_bad*boom_prob,(1-p_unemploy_bad)*(1-boom_prob),p_unemploy_bad*(1-boom_prob)],
                                      [p_reemploy*boom_prob,(1-p_reemploy)*boom_prob,p_reemploy*(1-boom_prob),(1-p_reemploy)*(1-boom_prob)]])
        
        MarkovType = deepcopy(InfiniteType)
        xi_dist = approxMeanOneLognormal(MarkovType.TranShkCount, 0.1)
        psi_dist = approxMeanOneLognormal(MarkovType.PermShkCount, 0.1)
        employed_income_dist = combineIndepDists(psi_dist, xi_dist)
        employed_income_dist = [np.ones(1),np.ones(1),np.ones(1)]
        unemployed_income_dist = [np.ones(1),np.ones(1),np.zeros(1)]
        p_zero_income = [np.array([0.0,1.0,0.0,1.0])]
        
        MarkovType.solution_terminal.cFunc = 4*[MarkovType.solution_terminal.cFunc]
        MarkovType.solution_terminal.vFunc = 4*[MarkovType.solution_terminal.vFunc]
        MarkovType.solution_terminal.vPfunc = 4*[MarkovType.solution_terminal.vPfunc]
        MarkovType.solution_terminal.vPPfunc = 4*[MarkovType.solution_terminal.vPPfunc]
        MarkovType.solution_terminal.mRtoMin = 4*[MarkovType.solution_terminal.mRtoMin]
        
        MarkovType.IncomeDist = [[employed_income_dist,unemployed_income_dist,employed_income_dist,unemployed_income_dist]]
        MarkovType.p_zero_income = p_zero_income
        MarkovType.transition_array = transition_array
        MarkovType.time_inv.append('transition_array')
        MarkovType.time_vary.append('p_zero_income')
        MarkovType.solveOnePeriod = consumptionSavingSolverMarkov
        MarkovType.cycles = 0
        
        MarkovType.vFuncBool = False
        
        MarkovType.timeFwd()
        start_time = clock()
        MarkovType.solve()
        end_time = clock()
        print('Solving a Markov consumer took ' + mystr(end_time-start_time) + ' seconds.')
        print('Consumption functions for each discrete state:')
        plotFuncs(MarkovType.solution[0].cFunc,0,50)
#
#
    if do_perfect_foresight:

        # Make and solve a perfect foresight consumer type who's problem is actually solved analytically,
        # but which can nonetheless be represented in this framework
        
        #PFC_paramteres = (DiscFac = 0.96, PermGroFac = 1.10, Rfree = 1.03 , CRRA = 4, constrained = True)
        PerfectForesightType = deepcopy(LifecycleType)    
        
        #tell the model to use the perfect forsight solver
        PerfectForesightType.solveOnePeriod = perfectForesightSolver
        PerfectForesightType.time_vary = [] #let the model know that there are no longer time varying parameters
        PerfectForesightType.time_inv =  PerfectForesightType.time_inv +['DiscFac','PermGroFac'] #change DiscFac and PermGroFac from time varying to non time varying
        #give the model new DiscFac and PermGroFac parameters to use for the perfect forsight model
        PerfectForesightType.assignParameters(DiscFac = 0.96,
                                              PermGroFac = 1.01)
        #tell the model not to use the terminal solution as a valid result anymore
        PerfectForesightType.pseudo_terminal = True
        
        start_time = clock()
        PerfectForesightType.solve()
        end_time = clock()
        print('Solving a Perfect Foresight consumer took ' + mystr(end_time-start_time) + ' seconds.')
        PerfectForesightType.unpack_cFunc()
        PerfectForesightType.timeFwd()
        
            
        plotFuncs(PerfectForesightType.cFunc[:],0,5)
