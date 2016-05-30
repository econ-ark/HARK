'''
A consumption-saving solver for a general equilibrium model with aggregate shocks.
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
    #print('Solved a period of the agg shocks model!')
    return solution_now
        