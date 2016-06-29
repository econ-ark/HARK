'''
Classes to solve consumption-saving models with idiosyncratic shocks to income
in which shocks are not necessarily fully transitory or fully permanent.  Extends
ConsIndShockModel by explicitly tracking permanent income as a state variable.
'''

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

from copy import copy, deepcopy
import numpy as np
from HARKcore import AgentType, Solution, NullFunc, HARKobject
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKinterpolation import LowerEnvelope, LinearInterp
from HARKsimulation import drawDiscrete
from HARKutilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv,\
                          CRRAutility_invP, CRRAutility_inv, CRRAutilityP_invP
from ConsIndShockModel import ConsIndShockSetup

utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

class MargValueFunc2D():
    '''
    A class for representing a marginal value function in models where the
    standard envelope condition of V'(M,p) = u'(c(M,p)) holds (with CRRA utility).
    This is copied from ConsAggShockModel, with the second state variable re-
    labeled as permanent income p.    
    '''
    def __init__(self,cFunc,CRRA):
        '''
        Constructor for a new marginal value function object.
        
        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on market
            resources and the level of permanent income: uP_inv(VPfunc(M,p)).
            Called cFunc because when standard envelope condition applies,
            uP_inv(VPfunc(M,p)) = cFunc(M,p).
        CRRA : float
            Coefficient of relative risk aversion.
            
        Returns
        -------
        new instance of MargValueFunc
        '''
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA
        
    def __call__(self,M,p):
        return utilityP(self.cFunc(M,p),gam=self.CRRA)
        
###############################################################################
        
class ConsIndShockSolverExplicitPermInc(ConsIndShockSetup):
    '''
    A class for solving the same one period "idiosyncratic shocks" problem as
    ConsIndShock, but with permanent income explicitly tracked as a state variable.
    Can't yet handle borrowing, value function calculation, or cubic spline
    interpolation of the consumption function.
    '''
    def __init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                      PermGroFac,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver-setup for problems with income subject to
        permanent and transitory shocks, with permanent income explicitly
        tracked as a state variable.
        
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
            period with.  Currently ignored, with BoroCnstArt=0 used implicitly.
        aXtraGrid: np.array
            Array of "extra" end-of-period (normalized) asset values-- assets
            above the absolute minimum acceptable level.
        pLvlGrid: np.array
            Array of permanent income levels at which to solve the problem.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.  Can't yet handle vFuncBool=True.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear inter-
            polation.  Can't yet handle CubicBool=True.
                        
        Returns
        -------
        None
        '''
        self.assignParameters(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool)
        self.defUtilityFuncs()
        
    def assignParameters(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool):
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
            period with.  Currently ignored, with BoroCnstArt=0 used implicitly.
        aXtraGrid: np.array
            Array of "extra" end-of-period (normalized) asset values-- assets
            above the absolute minimum acceptable level.
        pLvlGrid: np.array
            Array of permanent income levels at which to solve the problem.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.  Can't yet handle vFuncBool=True.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear inter-
            polation.  Can't yet handle CubicBool=True.
                        
        Returns
        -------
        none
        '''
        ConsIndShockSetup.assignParameters(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.pLvlGrid = pLvlGrid
        self.BoroCnstArt = 0.0
        
    def defBoroCnst(self,BoroCnstArt):
        '''
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Assumes that BoroCnstArt=0 for now.
        
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
        self.mNrmMinNow = 0.0
        self.MPCmaxEff = 1.0
    
        # Define the borrowing constraint (limiting consumption function) at any
        # given permanent income level.
        self.cFuncNowCnst = LinearInterp(np.array([self.mNrmMinNow, self.mNrmMinNow+1]), 
                                         np.array([0.0, 1.0]))
                                         
    def prepareToCalcEndOfPrdvP(self):
        '''
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period normalized assets, the grid of permanent income
        levels, and the distribution of shocks he might experience next period.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        aNrmNow : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.
        '''               
        ShkCount    = self.TranShkValsNext.size
        pLvlCount   = self.pLvlGrid.size
        aNrmCount   = self.aXtraGrid.size
        aNrmNow     = np.tile(np.asarray(self.aXtraGrid) + self.BoroCnstNat,(pLvlCount,1))
        pLvlNow     = np.tile(self.pLvlGrid,(aNrmCount,1)).transpose()
        ALvlNow     = aNrmNow*pLvlNow
        ALvlNow_tiled = np.tile(ALvlNow,(ShkCount,1,1)) # shape = (ShkCount,pLvlCount,aNrmCount)
        
        # Tile arrays of the income shocks and put them into useful shapes
        PermShkVals_tiled = np.transpose(np.tile(self.PermShkValsNext,(aNrmCount,pLvlCount,1)),(2,1,0))
        TranShkVals_tiled = np.transpose(np.tile(self.TranShkValsNext,(aNrmCount,pLvlCount,1)),(2,1,0))
        ShkPrbs_tiled     = np.transpose(np.tile(self.ShkPrbsNext,(aNrmCount,pLvlCount,1)),(2,1,0))
        
        # Get cash on hand next period
        MLvlNext          = self.Rfree/(self.PermGroFac*PermShkVals_tiled)*ALvlNow_tiled + TranShkVals_tiled*pLvlNow_tiled

        # Store and report the results
        self.PermShkVals_temp  = PermShkVals_temp
        self.ShkPrbs_temp      = ShkPrbs_temp
        self.mNrmNext          = mNrmNext 
        self.aNrmNow           = aNrmNow               
        return aNrmNow
    