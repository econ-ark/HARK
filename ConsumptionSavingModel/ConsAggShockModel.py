'''
This module concerns consumption-saving models with aggregate productivity shocks
as well as idiosyncratic income shocks.  Currently only contains one model with
a basic solver.  The model here is implemented in a general equilibrium frame-
work in the /cstwMPC folder, finding the capital-to-labor ratio evolution rule
kNextFunc in a dynamic stochastic general equilibrium.
'''
import sys 
sys.path.insert(0,'../')

import numpy as np
from HARKinterpolation import LinearInterp, LinearInterpOnInterp1D
from HARKutilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv
from ConsumptionSavingModel import ConsumerSolution
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


def solveConsumptionSavingAggShocks(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,PermGroFac,aXtraGrid,kGrid,kNextFunc,Rfunc,wFunc):
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
        