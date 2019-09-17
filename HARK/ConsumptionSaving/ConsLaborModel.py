"""
Subclasses of AgentType representing consumers who make decisions about how much
labor to supply, as well as a consumption-saving decision.

It currently only has
one model: labor supply on the intensive margin (unit interval) with CRRA utility
from a composite good (of consumption and leisure), with transitory and permanent
productivity shocks.  Agents choose their quantities of labor and consumption after
observing both of these shocks, so the transitory shock is a state variable.
"""
import sys 

from copy import copy
import numpy as np
from HARK.core import Solution
from HARK.utilities import CRRAutilityP, CRRAutilityP_inv
from HARK.interpolation import LinearInterp, LinearInterpOnInterp1D, VariableLowerBoundFunc2D, BilinearInterp, ConstantFunction
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType, MargValueFunc
from HARK.ConsumptionSaving.ConsGenIncProcessModel import ValueFunc2D, MargValueFunc2D

class ConsumerLaborSolution(Solution):
    '''
    A class for representing one period of the solution to a Consumer Labor problem.
    '''
    distance_criteria = ['cFunc','LbrFunc']
    
    def __init__(self, cFunc=None, LbrFunc=None, vFunc=None, vPfunc=None, bNrmMin=None):
        '''
        The constructor for a new ConsumerSolution object.
        
        Parameters
        ----------
        cFunc : function
            The consumption function for this period, defined over normalized
            bank balances and the transitory productivity shock: cNrm = cFunc(bNrm,TranShk).
        LbrFunc : function
            The labor supply function for this period, defined over normalized
            bank balances 0.751784276198: Lbr = LbrFunc(bNrm,TranShk).
        vFunc : function
            The beginning-of-period value function for this period, defined over
            normalized bank balances 0.751784276198: v = vFunc(bNrm,TranShk).
        vPfunc : function
            The beginning-of-period marginal value (of bank balances) function for
            this period, defined over normalized bank balances 0.751784276198: vP = vPfunc(bNrm,TranShk).
        bNrmMin: float
            The minimum allowable bank balances for this period, as a function of
            the transitory shock. cFunc, LbrFunc, etc are undefined for bNrm < bNrmMin(TranShk).
        
        Returns
        -------
        None
        '''
        if cFunc is not None:
            self.cFunc = cFunc
        if LbrFunc is not None:
            self.LbrFunc = LbrFunc
        if vFunc is not None:
            self.vFunc = vFunc
        if vPfunc is not None:
            self.vPfunc = vPfunc
        if bNrmMin is not None:
            self.bNrmMin = bNrmMin


def solveConsLaborIntMarg(solution_next,PermShkDstn,TranShkDstn,LivPrb,DiscFac,CRRA,
                          Rfree,PermGroFac,BoroCnstArt,aXtraGrid,TranShkGrid,vFuncBool,
                          CubicBool,WageRte,LbrCost):
    '''
    Solves one period of the consumption-saving model with endogenous labor supply
    on the intensive margin by using the endogenous grid method to invert the first
    order conditions for optimal composite consumption and between consumption and
    leisure, obviating any search for optimal controls.
    
    Parameters 
    ----------
    solution_next : ConsumerLaborSolution
        The solution to the next period's problem; must have the attributes
        vPfunc and bNrmMinFunc representing marginal value of bank balances and
        minimum (normalized) bank balances as a function of the transitory shock.
    PermShkDstn: [np.array]
        Discrete distribution of permanent productivity shocks. 
    TranShkDstn: [np.array]
        Discrete distribution of transitory productivity shocks.       
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period. 
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion over the composite good.  
    Rfree : float
        Risk free interest rate on assets retained at the end of the period.
    PermGroFac : float                                                         
        Expected permanent income growth factor for next period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  Currently not handled, must be None.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    TranShkGrid: np.array
        Grid of transitory shock values to use as a state grid for interpolation.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.  Not yet handled, must be False.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.
        Cubic interpolation is not yet handled, must be False.
    WageRte: float
        Wage rate per unit of labor supplied.
    LbrCost: float
        Cost parameter for supplying labor: u_t = U(x_t), x_t = c_t*z_t^LbrCost,
        where z_t is leisure = 1 - Lbr_t.
        
    Returns
    -------
    solution_now : ConsumerLaborSolution
        The solution to this period's problem, including a consumption function
        cFunc, a labor supply function LbrFunc, and a marginal value function vPfunc;
        each are defined over normalized bank balances and transitory prod shock.
        Also includes bNrmMinNow, the minimum permissible bank balances as a function
        of the transitory productivity shock.
    '''
    # Make sure the inputs for this period are valid: CRRA > LbrCost/(1+LbrCost)
    # and CubicBool = False.  CRRA condition is met automatically when CRRA >= 1.
    frac = 1./(1.+LbrCost)
    if CRRA <= frac*LbrCost:
        print('Error: make sure CRRA coefficient is strictly greater than alpha/(1+alpha).')
        sys.exit()     
    if BoroCnstArt is not None:
        print('Error: Model cannot handle artificial borrowing constraint yet. ')
        sys.exit()
    if vFuncBool or CubicBool is True:
        print('Error: Model cannot handle cubic interpolation yet.')
        sys.exit()

    # Unpack next period's solution and the productivity shock distribution, and define the inverse (marginal) utilty function
    vPfunc_next = solution_next.vPfunc
    TranShkPrbs = TranShkDstn[0]    
    TranShkVals  = TranShkDstn[1]
    PermShkPrbs = PermShkDstn[0]
    PermShkVals  = PermShkDstn[1]        
    TranShkCount  = TranShkPrbs.size
    PermShkCount = PermShkPrbs.size
    uPinv = lambda X : CRRAutilityP_inv(X,gam=CRRA)

    # Make tiled versions of the grid of a_t values and the components of the shock distribution
    aXtraCount = aXtraGrid.size
    bNrmGrid = aXtraGrid # Next period's bank balances before labor income
    bNrmGrid_rep = np.tile(np.reshape(bNrmGrid,(aXtraCount,1)),(1,TranShkCount)) # Replicated axtraGrid of b_t values (bNowGrid) for each transitory (productivity) shock
    TranShkVals_rep = np.tile(np.reshape(TranShkVals,(1,TranShkCount)),(aXtraCount,1)) # Replicated transitory shock values for each a_t state
    TranShkPrbs_rep = np.tile(np.reshape(TranShkPrbs,(1,TranShkCount)),(aXtraCount,1)) # Replicated transitory shock probabilities for each a_t state
    
    # Construct a function that gives marginal value of next period's bank balances *just before* the transitory shock arrives
    vPNext = vPfunc_next(bNrmGrid_rep, TranShkVals_rep) # Next period's marginal value at every transitory shock and every bank balances gridpoint           
    vPbarNext = np.sum(vPNext*TranShkPrbs_rep, axis = 1) # Integrate out the transitory shocks (in TranShkVals direction) to get expected vP just before the transitory shock
    vPbarNvrsNext = uPinv(vPbarNext) # Transformed marginal value through the inverse marginal utility function to "decurve" it
    vPbarNvrsFuncNext = LinearInterp(np.insert(bNrmGrid,0,0.0),np.insert(vPbarNvrsNext,0,0.0)) # Linear interpolation over b_{t+1}, adding a point at minimal value of b = 0. 
    vPbarFuncNext = MargValueFunc(vPbarNvrsFuncNext,CRRA) # "Recurve" the intermediate marginal value function through the marginal utility function

    # Get next period's bank balances at each permanent shock from each end-of-period asset values
    aNrmGrid_rep = np.tile(np.reshape(aXtraGrid,(aXtraCount,1)),(1,PermShkCount)) # Replicated grid of a_t values for each permanent (productivity) shock  
    PermShkVals_rep = np.tile(np.reshape(PermShkVals,(1,PermShkCount)),(aXtraCount,1)) # Replicated permanent shock values for each a_t value   
    PermShkPrbs_rep = np.tile(np.reshape(PermShkPrbs,(1,PermShkCount)),(aXtraCount,1)) # Replicated permanent shock probabilities for each a_t value
    bNrmNext = (Rfree/(PermGroFac*PermShkVals_rep))*aNrmGrid_rep

    # Calculate marginal value of end-of-period assets at each a_t gridpoint  
    vPbarNext = (PermGroFac*PermShkVals_rep)**(-CRRA)*vPbarFuncNext(bNrmNext) # Get marginal value of bank balances next period at each shock
    EndOfPrdvP = DiscFac*Rfree*LivPrb*np.sum(vPbarNext*PermShkPrbs_rep, axis=1, keepdims=True) # Take expectation across permanent income shocks
    
    # Compute scaling factor for each transitory shock
    TranShkScaleFac_temp = frac*(WageRte*TranShkGrid)**(LbrCost*frac)*(LbrCost**(-LbrCost*frac)+LbrCost**(frac))  
    TranShkScaleFac = np.reshape(TranShkScaleFac_temp,(1,TranShkGrid.size)) # Flip it to be a row vector
   
    # Use the first order condition to compute an array of "composite good" x_t values corresponding to (a_t,theta_t) values 
    xNow = (np.dot(EndOfPrdvP,TranShkScaleFac))**(-1./(CRRA-LbrCost*frac))      
    
    # Transform the composite good x_t values into consumption c_t and leisure z_t values
    TranShkGrid_rep = np.tile(np.reshape(TranShkGrid,(1,TranShkGrid.size)),(aXtraCount,1))
    xNowPow = xNow**frac # Will use this object multiple times in math below
    cNrmNow = (((WageRte*TranShkGrid_rep)/LbrCost)**(LbrCost*frac))*xNowPow # Find optimal consumption from optimal composite good
    LsrNow = (LbrCost/(WageRte*TranShkGrid_rep))**frac*xNowPow # Find optimal leisure from optimal composite good
    
    # The zero-th transitory shock is TranShk=0, and the solution is to not work: Lsr = 1, Lbr = 0.
    cNrmNow[:,0] = uPinv(EndOfPrdvP.flatten())
    LsrNow[:,0] = 1.0

    # Agent cannot choose to work a negative amount of time. When this occurs, set
    # leisure to one and recompute consumption using simplified first order condition.
    violates_labor_constraint = LsrNow > 1. # Find where labor would be negative if unconstrained
    EndOfPrdvP_temp = np.tile(np.reshape(EndOfPrdvP,(aXtraCount,1)),(1,TranShkCount))
    cNrmNow[violates_labor_constraint] = uPinv(EndOfPrdvP_temp[violates_labor_constraint])
    LsrNow[violates_labor_constraint] = 1. # Set up z =1, upper limit

    # Calculate the endogenous bNrm states by inverting the within-period transition
    aNrmNow_rep = np.tile(np.reshape(aXtraGrid,(aXtraCount,1)),(1,TranShkGrid.size))
    bNrmNow = aNrmNow_rep - WageRte*TranShkGrid_rep + cNrmNow + WageRte*TranShkGrid_rep*LsrNow
    
    # Add an extra gridpoint at the absolute minimal valid value for b_t for each TranShk;
    # this corresponds to working 100% of the time and consuming nothing.
    bNowArray = np.concatenate((np.reshape(-WageRte*TranShkGrid,(1,TranShkGrid.size)), bNrmNow),axis=0)
    cNowArray = np.concatenate((np.zeros((1,TranShkGrid.size)),cNrmNow),axis=0) # Consume nothing
    LsrNowArray = np.concatenate((np.zeros((1,TranShkGrid.size)),LsrNow),axis=0) # And no leisure!
    LsrNowArray[0,0] = 1.0 # Don't work at all if TranShk=0, even if bNrm=0
    LbrNowArray = 1.0 - LsrNowArray # Labor is the complement of leisure
    
    # Get (pseudo-inverse) marginal value of bank balances using end of period
    # marginal value of assets (envelope condition), adding a column of zeros
    # zeros on the left edge, representing the limit at the minimum value of b_t.
    vPnvrsNowArray = np.concatenate((np.zeros((1,TranShkGrid.size)), uPinv(EndOfPrdvP_temp))) # Concatenate 

    # Construct consumption and marginal value functions for this period
    bNrmMinNow = LinearInterp(TranShkGrid,bNowArray[0,:])

    # Loop over each transitory shock and make a linear interpolation to get lists
    # of optimal consumption, labor and (pseudo-inverse) marginal value by TranShk
    cFuncNow_list = []
    LbrFuncNow_list = []
    vPnvrsFuncNow_list = []
    for j in range(TranShkGrid.size):                   
        bNrmNow_temp = bNowArray[:,j] - bNowArray[0,j] # Adjust bNrmNow for this transitory shock, so bNrmNow_temp[0] = 0
        cFuncNow_list.append(LinearInterp(bNrmNow_temp,cNowArray[:,j])) # Make consumption function for this transitory shock
        LbrFuncNow_list.append(LinearInterp(bNrmNow_temp,LbrNowArray[:,j])) # Make labor function for this transitory shock
        vPnvrsFuncNow_list.append(LinearInterp(bNrmNow_temp,vPnvrsNowArray[:,j])) # Make pseudo-inverse marginal value function for this transitory shock

    # Make linear interpolation by combining the lists of consumption, labor and marginal value functions
    cFuncNowBase = LinearInterpOnInterp1D(cFuncNow_list,TranShkGrid)
    LbrFuncNowBase = LinearInterpOnInterp1D(LbrFuncNow_list,TranShkGrid)
    vPnvrsFuncNowBase = LinearInterpOnInterp1D(vPnvrsFuncNow_list,TranShkGrid)
    
    # Construct consumption, labor, pseudo-inverse marginal value functions with
    # bNrmMinNow as the lower bound.  This removes the adjustment in the loop above.
    cFuncNow = VariableLowerBoundFunc2D(cFuncNowBase,bNrmMinNow)
    LbrFuncNow = VariableLowerBoundFunc2D(LbrFuncNowBase,bNrmMinNow)
    vPnvrsFuncNow = VariableLowerBoundFunc2D(vPnvrsFuncNowBase,bNrmMinNow)
     
    # Construct the marginal value function by "recurving" its pseudo-inverse
    vPfuncNow = MargValueFunc2D(vPnvrsFuncNow,CRRA)  

    # Make a solution object for this period and return it
    solution = ConsumerLaborSolution(cFunc=cFuncNow,LbrFunc=LbrFuncNow,vPfunc=vPfuncNow,bNrmMin=bNrmMinNow)
    return solution

        
class LaborIntMargConsumerType(IndShockConsumerType):
    
    '''        
    A class representing agents who make a decision each period about how much
    to consume vs save and how much labor to supply (as a fraction of their time).
    They get CRRA utility from a composite good x_t = c_t*z_t^alpha, and discount
    future utility flows at a constant factor. 
    '''
    time_vary_ = copy(IndShockConsumerType.time_vary_)
    time_vary_ += ['LbrCost','WageRte']
    time_inv_ = copy(IndShockConsumerType.time_inv_)
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new consumer type with given data.
        See ConsumerParameters.init_labor_intensive for a dictionary of
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
        IndShockConsumerType.__init__(self,cycles = cycles,time_flow=time_flow,**kwds)
        self.pseudo_terminal = False
        self.solveOnePeriod = solveConsLaborIntMarg
        self.update()
    
    def update(self):
        '''
        Update the income process, the assets grid, and the terminal solution.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        self.updateIncomeProcess()
        self.updateAssetsGrid()
        self.updateTranShkGrid()
         
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
            cNrmNow[these] = self.solution[t].cFunc(self.bNrmNow[these], self.TranShkNow[these]) # assign consumtion values
            MPCnow[these] = self.solution[t].cFunc.derivativeX(self.bNrmNow[these], self.TranShkNow[these]) # assign Marginal propensity to consume values (derivative)
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
        self.mNrmNow = self.bNrmNow + LaborIntMargExample.solution[t].LbrFunc(self.bNrmNow, self.TranShkNow)*self.TranShkNow 
        return None


    def updateTranShkGrid(self):
        '''
        Create a time-varying list of arrays for TranShkGrid using TranShkDstn,
        which is created by the method updateIncomeProcess().  Simply takes the
        transitory shock values and uses them as a state grid; can be improved.
        Creates the attribute TranShkGrid.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        time_orig=self.time_flow
        self.timeFwd()
      
        TranShkGrid = []   # Create an empty list for TranShkGrid that will be updated
        for t in range(self.T_cycle):
            TranShkGrid.append(self.TranShkDstn[t][1])  # Update/ Extend the list of TranShkGrid with the TranShkVals for each TranShkPrbs
        self.TranShkGrid = TranShkGrid  # Save that list in self (time-varying)
        self.addToTimeVary('TranShkGrid')   # Run the method addToTimeVary from AgentType to add TranShkGrid as one parameter of time_vary list
        
        if not time_orig:
            self.timeRev()
            
     
    def updateSolutionTerminal(self):
        ''' 
        Updates the terminal period solution and solves for optimal consumption
        and labor when there is no future.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''   
        if self.time_flow: # To make sure we pick the last element of the list, depending on the direction time is flowing
            t=-1
        else:
            t=0
        TranShkGrid = self.TranShkGrid[t]
        LbrCost = self.LbrCost[t]
        WageRte = self.WageRte[t]

        bNrmGrid = np.insert(self.aXtraGrid,0,0.0) # Add a point at b_t = 0 to make sure that bNrmGrid goes down to 0
        bNrmCount = bNrmGrid.size   # 201
        TranShkCount = TranShkGrid.size     # = (7,)   
        bNrmGridTerm = np.tile(np.reshape(bNrmGrid,(bNrmCount,1)),(1,TranShkCount)) # Replicated bNrmGrid for each transitory shock theta_t      
        TranShkGridTerm = np.tile(TranShkGrid,(bNrmCount,1))    # Tile the grid of transitory shocks for the terminal solution. (201,7)  
                                               
        # Array of labor (leisure) values for terminal solution
        LsrTerm = np.minimum((LbrCost/(1.+LbrCost))*(bNrmGridTerm/(WageRte*TranShkGridTerm)+1.),1.0)
        LsrTerm[0,0] = 1.0
        LbrTerm = 1.0 - LsrTerm
        
        # Calculate market resources in terminal period, which is consumption
        mNrmTerm = bNrmGridTerm + LbrTerm*WageRte*TranShkGridTerm
        cNrmTerm = mNrmTerm # Consume everything we have
        
        # Make a bilinear interpolation to represent the labor and consumption functions
        LbrFunc_terminal = BilinearInterp(LbrTerm,bNrmGrid,TranShkGrid)
        cFunc_terminal = BilinearInterp(cNrmTerm,bNrmGrid,TranShkGrid)
        
        # Compute the effective consumption value using consumption value and labor value at the terminal solution
        xEffTerm = LsrTerm**LbrCost*cNrmTerm
        vNvrsFunc_terminal = BilinearInterp(xEffTerm,bNrmGrid,TranShkGrid)
        vFunc_terminal = ValueFunc2D(vNvrsFunc_terminal, self.CRRA)
        
        # Using the envelope condition at the terminal solution to estimate the marginal value function      
        vPterm = LsrTerm**LbrCost*CRRAutilityP(xEffTerm,gam=self.CRRA)        
        vPnvrsTerm = CRRAutilityP_inv(vPterm,gam=self.CRRA)     # Evaluate the inverse of the CRRA marginal utility function at a given marginal value, vP
        
        vPnvrsFunc_terminal = BilinearInterp(vPnvrsTerm,bNrmGrid,TranShkGrid)
        vPfunc_terminal = MargValueFunc2D(vPnvrsFunc_terminal,self.CRRA) # Get the Marginal Value function
            
        bNrmMin_terminal = ConstantFunction(0.)     # Trivial function that return the same real output for any input
        
        self.solution_terminal = ConsumerLaborSolution(cFunc=cFunc_terminal,LbrFunc=LbrFunc_terminal,\
                                 vFunc=vFunc_terminal,vPfunc=vPfunc_terminal,bNrmMin=bNrmMin_terminal)
        
        
    def plotcFunc(self,t,bMin=None,bMax=None,ShkSet=None):
        '''
        Plot the consumption function by bank balances at a given set of transitory shocks.
        
        Parameters
        ----------
        t : int
            Time index of the solution for which to plot the consumption function.
        bMin : float or None
            Minimum value of bNrm at which to begin the plot.  If None, defaults
            to the minimum allowable value of bNrm for each transitory shock.
        bMax : float or None
            Maximum value of bNrm at which to end the plot.  If None, defaults
            to bMin + 20.
        ShkSet : [float] or None
            Array or list of transitory shocks at which to plot the consumption
            function.  If None, defaults to the TranShkGrid for this time period.
        
        Returns
        -------
        None
        '''
        if ShkSet is None:
            ShkSet = self.TranShkGrid[t]
        if bMin is None:
            bMinSet = self.solution[0].bNrmMin(TranShkSet)
        else:
            bMinSet = bMin*np.ones_like(TranShkSet)
        if bMax is None:
            bMaxSet = bMinSet + 20.
        else:
            bMaxSet = bMax*np.ones_like(TranShkSet)
             
        for j in range(len(ShkSet)):
            Shk = ShkSet[j]
            B = np.linspace(bMinSet[j],bMaxSet[j],300)
            C = LaborIntMargExample.solution[t].cFunc(B,Shk*np.ones_like(B))
            plt.plot(B,C)
        plt.xlabel('Beginning of period bank balances')
        plt.ylabel('Normalized consumption level')
        plt.show()
       
               
###############################################################################
          
if __name__ == '__main__':
    import HARK.ConsumptionSaving.ConsumerParameters as Params    # Parameters for a consumer type
    from HARK.utilities import plotFuncsDer, plotFuncs
    import matplotlib.pyplot as plt
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)     # Format numbers as strings
    
    do_simulation           = True
    
###############################################################################

    # Make and solve an idiosyncratic shocks consumer with a finite lifecycle
    LifecycleExample = IndShockConsumerType(**Params.init_lifecycle)
    LifecycleExample.cycles = 1 # Make this consumer live a sequence of periods exactly once

    start_time = clock()
    LifecycleExample.solve()
    end_time = clock()
    print('Solving a lifecycle consumer took ' + mystr(end_time-start_time) + ' seconds.')
    LifecycleExample.unpackcFunc()
    LifecycleExample.timeFwd()

    # Plot the consumption functions during working life
    print('Consumption functions while working:')
    mMin = min([LifecycleExample.solution[t].mNrmMin for t in range(LifecycleExample.T_cycle)])
    plotFuncs(LifecycleExample.cFunc[:LifecycleExample.T_retire],mMin,5)

    # Plot the consumption functions during retirement
    print('Consumption functions while retired:')
    plotFuncs(LifecycleExample.cFunc[LifecycleExample.T_retire:],0,5)
    LifecycleExample.timeRev()
    
    if do_simulation:
        LifecycleExample.T_sim = 120
        LifecycleExample.track_vars = ['mNrmNow','cNrmNow','pLvlNow','t_age']
        LifecycleExample.initializeSim()
        LifecycleExample.simulate()
#        plt.plot(np.linspace(0, 5, 10000), LifecycleExample.cNrmNow_hist[30])
#        plt.show()
        
###############################################################################
    
    # Make and solve a labor intensive margin consumer i.e. a consumer with utility for leisure
    LaborIntMargExample = LaborIntMargConsumerType(**Params.init_labor_intensive)
    LaborIntMargExample.cycles = 0
    
    t_start = clock()
    LaborIntMargExample.solve()
    t_end = clock()
    print('Solving a labor intensive margin consumer took ' + str(t_end-t_start) + ' seconds.')
    
    t = 0
    bMax = 100.
    
    # Plot the consumption function at various transitory productivity shocks
    TranShkSet = LaborIntMargExample.TranShkGrid[t]
    B = np.linspace(0.,bMax,300)
    for Shk in TranShkSet:
        B_temp = B + LaborIntMargExample.solution[t].bNrmMin(Shk)
        C = LaborIntMargExample.solution[t].cFunc(B_temp,Shk*np.ones_like(B_temp))
        plt.plot(B_temp,C)
    plt.xlabel('Beginning of period bank balances')
    plt.ylabel('Normalized consumption level')
    plt.show()
    
    # Plot the marginal consumption function at various transitory productivity shocks
    TranShkSet = LaborIntMargExample.TranShkGrid[t]
    B = np.linspace(0.,bMax,300)
    for Shk in TranShkSet:
        B_temp = B + LaborIntMargExample.solution[t].bNrmMin(Shk)
        C = LaborIntMargExample.solution[t].cFunc.derivativeX(B_temp,Shk*np.ones_like(B_temp))
        plt.plot(B_temp,C)
    plt.xlabel('Beginning of period bank balances')
    plt.ylabel('Marginal propensity to consume')
    plt.show()
    
    # Plot the labor function at various transitory productivity shocks
    TranShkSet = LaborIntMargExample.TranShkGrid[t]
    B = np.linspace(0.,bMax,300)
    for Shk in TranShkSet:
        B_temp = B + LaborIntMargExample.solution[t].bNrmMin(Shk)
        Lbr = LaborIntMargExample.solution[t].LbrFunc(B_temp,Shk*np.ones_like(B_temp))
        plt.plot(B_temp,Lbr)
    plt.xlabel('Beginning of period bank balances')
    plt.ylabel('Labor supply')
    plt.show()
    
    # Plot the marginal value function at various transitory productivity shocks
    pseudo_inverse = True
    TranShkSet = LaborIntMargExample.TranShkGrid[t]
    B = np.linspace(0.,bMax,300)
    for Shk in TranShkSet:
        B_temp = B + LaborIntMargExample.solution[t].bNrmMin(Shk)
        if pseudo_inverse:
            vP = LaborIntMargExample.solution[t].vPfunc.cFunc(B_temp,Shk*np.ones_like(B_temp))
        else:
            vP = LaborIntMargExample.solution[t].vPfunc(B_temp,Shk*np.ones_like(B_temp))
        plt.plot(B_temp,vP)
    plt.xlabel('Beginning of period bank balances')
    if pseudo_inverse:
        plt.ylabel('Pseudo inverse marginal value')
    else:
        plt.ylabel('Marginal value')
    plt.show()
    
    if do_simulation:
        LaborIntMargExample.T_sim = 120 # Set number of simulation periods
        LaborIntMargExample.track_vars = ['bNrmNow', 'cNrmNow']
        LaborIntMargExample.initializeSim()
        LaborIntMargExample.simulate()
#        plt.plot(np.linspace(0,100,10000), LaborIntMargExample.cNrmNow_hist[30])
#        plt.show()
    