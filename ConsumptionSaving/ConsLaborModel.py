'''
This module contains classes for representing consumers who choose their labor
supply at the beginning of each period.  It currently only has one model: labor
supply choice on the extensive margin with *only* permanent shocks.  This model
is meant to demonstrate the FellaInterp class for handling non-concave problems.
'''

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
from HARKutilities import CRRAutility, CRRAutilityP, CRRAutilityP_inv, CRRAutility_inv
from HARKinterpolation import LinearInterp, CubicInterp, FellaInterp, ConstantFunction
from ConsIndShockModel import IndShockConsumerType, ConsumerSolution, ValueFunc, MargValueFunc

def solveConsBabyLabor(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                                BoroCnstArt,LbrDisutil,aXtraGrid):
    '''
    Solves a single period consumption-saving problem with CRRA utility and risky
    labor productivity (subject to permanent shocks); consumers make a binary
    choice over labor supply each period.  Working brings (normalized) income of
    1, while not working brings income of 0.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn : [np.array]
        A list containing two arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks.  If additional arrays are included,
        they will be ignored.
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
    BoroCnstArt : float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    LbrDisutil : float
        Percentage reduction in "effective consumption" if consumer supplies
        labor.  Must be strictly between 0 and 1.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
        
    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem with labor
        choice.  Includes a consumption function cFunc, a labor supply function
        LbrFunc, a marginal value function vPfunc, a value function vFunc, a min-
        imum acceptable level of normalized bank balances bNrmMin, and bounding
        MPCs MPCmin and MPCmax.
    '''
    # Unpack next period's solution and the income distribution, and define the (inverse) (marginal) utilty function
    vPfuncNext = solution_next.vPfunc
    vFuncNext  = solution_next.vFunc
    ShkPrbsNext = IncomeDstn[0]
    PermShkValsNext = IncomeDstn[1]
    ShockCount  = ShkPrbsNext.size
    u = lambda c : CRRAutility(c,gam=CRRA)
    uP = lambda c : CRRAutilityP(c,gam=CRRA)
    uinv = lambda x : CRRAutility_inv(x,gam=CRRA) 
    uPinv = lambda x : CRRAutilityP_inv(x,gam=CRRA)
    AugPointCount = 10 # Number of gridpoints to add on constrained portion
    
    # Calculate minimum acceptable value of end-of-period normalized assets
    DiscFacEff       = DiscFac*LivPrb # "effective" discount factor
    PermShkMinNext   = np.min(PermShkValsNext)
    WorstIncPrb      = np.sum(ShkPrbsNext[PermShkValsNext==PermShkMinNext])
    BoroCnstNat = solution_next.bNrmMin*(PermGroFac*PermShkMinNext)/Rfree
    BoroCnst    = np.maximum(BoroCnstNat,BoroCnstArt)
    ArtCnstBinds = BoroCnstArt > BoroCnstNat
    bNrmMinLbr0 = BoroCnst # Minimum bank balances if consumer does not work
    bNrmMinLbr1 = BoroCnst - 1. # Minimum bank balances if consumer works
        
    # Update the bounding MPCs and PDV of human wealth:
    PatFac       = ((Rfree*DiscFacEff)**(1.0/CRRA))/Rfree
    MPCminNow    = 1.0/(1.0 + PatFac/solution_next.MPCmin)
    MPCmaxNow    = 1.0/(1.0 + (WorstIncPrb**(1.0/CRRA))*PatFac/solution_next.MPCmax)

    # Make tiled versions of the grid of a_t values and the components of the income distribution
    if ArtCnstBinds:
        aNowGrid = BoroCnst + np.insert(aXtraGrid,0,0.0) # Add a point at aXtra_t = 0
        MPCmaxNow = 1.0
    else:
        aNowGrid = BoroCnst + aXtraGrid
    StateCount = aNowGrid.size
    aNowGrid_rep = np.tile(np.reshape(aNowGrid,(StateCount,1)),(1,ShockCount)) # Replicated aNowGrid for each income shock
    PermShkVals_rep = np.tile(np.reshape(PermShkValsNext,(1,ShockCount)),(StateCount,1)) # Replicated permanent shock values for each a_t state
    ShkPrbs_rep = np.tile(np.reshape(ShkPrbsNext,(1,ShockCount)),(StateCount,1)) # Replicated shock probabilities for each a_t state
    
    # Calculate end-of-period (marginal) value for each a_t in aNowGrid
    Reff_array = Rfree/(PermGroFac*PermShkVals_rep) # Effective interest factor on *normalized* end-of-period assets
    bNext = Reff_array*aNowGrid_rep # Next period's bank balances
    vNext = vFuncNext(bNext)*PermShkVals_rep**(1.-CRRA) # Next period's value
    vPnext = vPfuncNext(bNext)*PermShkVals_rep**(-CRRA) # Next period's marginal value
    EndOfPeriodv  = DiscFacEff*PermGroFac**(1.-CRRA)*np.sum(vNext*ShkPrbs_rep,axis=1) # Value of end-of-period assets
    EndOfPeriodvP = DiscFacEff*Rfree*PermGroFac**(-CRRA)*np.sum(vPnext*ShkPrbs_rep,axis=1) # Marginal value of end-of-period assets

    # Calculate consumption and endogenous b_t gridpoints if consumer does not work
    cNrmLbr0 = uPinv(EndOfPeriodvP)
    bNrmLbr0 = aNowGrid + cNrmLbr0
    vNowLbr0 = u(cNrmLbr0) + EndOfPeriodv
    if ArtCnstBinds: # Add augmented points on constrained portion
        bNrmAug = np.linspace(bNrmMinLbr0,bNrmLbr0[0],num=AugPointCount,endpoint=False)
        cNrmAug = bNrmAug - bNrmMinLbr0
        vNowAug = EndOfPeriodv[0] + u(cNrmAug)
        aug_N = AugPointCount
    else: # Add one point right at natural borrowing constraint
        bNrmAug = np.array([bNrmMinLbr0])
        cNrmAug = np.array([0.])
        if CRRA < 1.:
            temp_u = -np.inf
        else:
            temp_u = 0.
        vNowAug = np.array([temp_u + EndOfPeriodv[0]])
        aug_N = 1
    bNrmLbr0 = np.concatenate([bNrmAug,bNrmLbr0])
    cNrmLbr0 = np.concatenate([cNrmAug,cNrmLbr0])
    vNowLbr0 = np.concatenate([vNowAug,vNowLbr0])
    Lbr0 = np.zeros_like(cNrmLbr0)
    
    # Calculate consumption and endogenous b_t gridpoints if consumer works
    cNrmLbr1 = cNrmLbr0[aug_N:]*(1.-LbrDisutil)**(1./CRRA-1.) # Don't use augmented points
    bNrmLbr1 = aNowGrid + cNrmLbr1 - 1.
    vNowLbr1 = u((1.-LbrDisutil)*cNrmLbr1) + EndOfPeriodv
    if ArtCnstBinds: # Add augmented points on constrained portion
        bNrmAug = np.linspace(bNrmMinLbr1,bNrmLbr1[0],num=AugPointCount,endpoint=False)
        cNrmAug = bNrmAug - bNrmMinLbr1
        vNowAug = EndOfPeriodv[0] + u((1.-LbrDisutil)*cNrmAug)
    else: # Add one point right at natural borrowing constraint
        bNrmAug = np.array([bNrmMinLbr1])
        cNrmAug = np.array([0.])
        if CRRA < 1.:
            temp_u = -np.inf
        else:
            temp_u = 0.
        vNowAug = np.array([temp_u + EndOfPeriodv[0]])
    bNrmLbr1 = np.concatenate([bNrmAug,bNrmLbr1])
    cNrmLbr1 = np.concatenate([cNrmAug,cNrmLbr1])
    vNowLbr1 = np.concatenate([vNowAug,vNowLbr1])
    Lbr1 = np.ones_like(cNrmLbr1)
    
    # Use FellaInterp to take the "first order condition upper envelope"
    # First row of policy input is vPnvrs, second row is cFunc
    SolnFunc = FellaInterp(v0=-1.0, control0=[0.0,0.0,1.0], lower_bound=bNrmMinLbr1, upper_bound=None)
    SolnFunc.addNewPoints(bNrmLbr1,uinv(vNowLbr1),np.vstack(((1.-LbrDisutil)**(1.-1./CRRA)*cNrmLbr1,cNrmLbr1,Lbr1)),True)
    SolnFunc.addNewPoints(bNrmLbr0,uinv(vNowLbr0),np.vstack((cNrmLbr0,cNrmLbr0,Lbr0)),True)
    
    # Construct the value, marginal value, and policy functions
    SolnFunc.makeValueAndPolicyFuncs()
    SolnFunc.makeCRRAvNvrsFunc(CRRA,0)
    
    # Package the solution and return it
    vFuncNow = ValueFunc(SolnFunc.ValueFunc,CRRA)
    vPfuncNow = MargValueFunc(SolnFunc.PolicyFuncs[0],CRRA)
    cFuncNow = SolnFunc.PolicyFuncs[1]
    LbrFuncNow = SolnFunc.PolicyFuncs[2]
    solution_now = ConsumerSolution(cFunc=cFuncNow, vFunc=vFuncNow, vPfunc=vPfuncNow,
                                    hNrm=0.0, MPCmin=MPCminNow, MPCmax=MPCmaxNow)
    solution_now.bNrmMin = bNrmMinLbr1
    solution_now.LbrFunc = LbrFuncNow
    return solution_now
    
    
    
class BabyLaborConsumerType(IndShockConsumerType):
    '''
    A class for representing consumers in the baby labor model.  Agents choose
    consumption and whether to supply a unit of labor each period; their productivity
    is subject to permanent shocks (but not transitory shocks, which would require
    a second continuous state variable).
    '''
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data.
        See ConsumerParameters.init_baby_labor for a dictionary of
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
        # Initialize a basic AgentType and change the solver for this model
        IndShockConsumerType.__init__(self,cycles=cycles,time_flow=time_flow,**kwds)
        self.solveOnePeriod = solveConsBabyLabor # idiosyncratic shocks solver
   
    def update(self):
        IndShockConsumerType.update(self)
        self.updateLbrDisutil()
    
    def updateLbrDisutil(self):
        '''
        Constructs the timepath of disutility from labor based on primitive attributes
        named LbrDisutilCoeffs, representing polynomial coefficients on age for
        transformed labor disutility.  X[t] = sum(LbrDisUtilCoeffs[n]*t**n),
        LbrDisutil = 1/(1 + exp(X)).
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        N = len(self.LbrDisutilCoeffs)
        T = np.arange(self.T_cycle,dtype=float)
        X = np.zeros_like(T)
        for n in range(N):
            X += self.LbrDisutilCoeffs[n]*(T**n)
        LbrDisutil = (1. - 1./(1. + np.exp(X))).tolist()
        if not self.time_flow:
            LbrDisutil.reverse()
        self.LbrDisutil = LbrDisutil
        self.addToTimeVary('LbrDisutil')
        
        
    def getControls(self):
        '''
        Calculates consumption and discrete labor supply for each consumer of
        this type using the policy functions.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        LbrNow = np.zeros(self.AgentCount) + np.nan
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow  = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            LbrNow[these] = self.solution[t].LbrFunc(self.bNrmNow[these])
            cNrmNow[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(self.bNrmNow[these])
        self.LbrNow = LbrNow
        self.cNrmNow = cNrmNow
        self.MPCnow = MPCnow
        self.mNrmNow = self.bNrmNow + LbrNow # Market resources after income (overwriting PerForesightConsumerType.getStates)
        
    
    def updateSolutionTerminal(self):
        '''
        Terminal period solver for the baby labor model.  Assumes a hard borrowing
        constraint at zero, as usual.  For simplicity, assumes that LbrDisutil
        is 1.0 in terminal period, so consumer never works.
        '''
        IndShockConsumerType.updateSolutionTerminal(self)
        self.solution_terminal(LbrFunc = ConstantFunction(0.0))
        self.solution_terminal.bNrmMin = 0.0
        
        
        
        
if __name__ == '__main__':
    import ConsumerParameters as Params
    import matplotlib.pyplot as plt
    from HARKutilities import plotFuncs
    
    # Make and solve an example baby labor consumer type
    BabyLaborExample = BabyLaborConsumerType(**Params.init_baby_labor)
    BabyLaborExample.solve()
    
    