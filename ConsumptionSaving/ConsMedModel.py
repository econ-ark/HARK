'''
Consumption-saving models that also include medical spending.
'''
import sys 
sys.path.insert(0,'../')

import numpy as np
from HARKutilities import approxLognormal
from ConsIndShockModel import ConsumerSolution
from HARKinterpolation import BilinearInterpOnInterp1D, BilinearInterp, LowerEnvelope2D
from ConsPersistentShockModel import ConsPersistentShockSolver, PersistentShockConsumerType, MargValueFunc2D

class MedShockConsumerType(PersistentShockConsumerType):
    '''
    A class to represent agents who consume two goods: ordinary composite consumption
    and medical care; both goods yield CRRAutility, and the coefficients on the
    goods might be different.  Agents expect to receive shocks to permanent and
    transitory income as well as multiplicative shocks to utility from medical care.
    '''
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data, and construct objects
        to be used during solution (income distribution, assets grid, etc).
        See ConsumerParameters.init_med_shock for a dictionary of the keywords
        that should be passed to the constructor.
        
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
        PersistentShockConsumerType.__init__(self,**kwds)
        self.solveOnePeriod = solveConsMedShock # Choose correct solver
        self.addToTimeInv('CRRAmed')
        
    def update(self):
        '''
        Updates the assets grid, permanent income grid, medical shock grid, income
        process, terminal period solution, and preference shock process.  A very
        slight extension of PersistentShockConsumerType.update() for the medical
        shock model.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        PersistentShockConsumerType.update(self)  # Update assets grid, income process, terminal solution
        self.updateMedShockProcess()     # Update the discrete medical shock process
        
    def updateMedShockProcess(self):
        '''
        Constructs discrete distributions of medical preference shocks for each
        period in the cycle.  Distributions are saved as attribute MedShkDstn,
        which is added to time_vary.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        # WRITE THE REST OF THIS METHOD
        self.addToTimeVary('MedShkDstn')
        
    def updatePermIncGrid(self):
        '''
        Update the grid of permanent income levels.  Currently only works for
        infinite horizon models (cycles=0) and lifecycle models (cycles=1).  Not
        clear what to do about cycles>1.  Identical to version in persistent
        shocks model, but pLvl=0 is manually added to the grid (because there is
        no closed form lower-bounding cFunc for pLvl=0).
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        # Run basic version of this method
        PersistentShockConsumerType.updatePermIncGrid(self)
        for j in range(len(self.pLvlGrid)): # Then add 0 to the bottom of each pLvlGrid
            this_grid = self.pLvlGrid[j]
            self.pLvlGrid[j] = np.append(this_grid,0,0.0)
        
###############################################################################
        
class ConsMedShockSolver(ConsPersistentShockSolver):
    '''
    Class for solving the one period problem for the "medical shocks" model, in
    which consumers receive shocks to permanent and transitory income as well as
    shocks to "medical need"-- multiplicative utility shocks for a second good.
    '''
    def __init__(self,solution_next,IncomeDstn,MedShkDstn,LivPrb,DiscFac,CRRA,Rfree,MedPrice,
                 PermGroFac,PermIncCorr,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver for a one period problem with idiosyncratic
        shocks to permanent and transitory income and shocks to medical need.
        
        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        MedShkDstn : [np.array]
            Discrete distribution of the multiplicative utility shifter for med-
            ical care. Order: probabilities, preference shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.    
        DiscFac : float
            Intertemporal discount factor for future utility.        
        CRRA : float
            Coefficient of relative risk aversion for composite consumption.
        CRRAmed : float
            Coefficient of relative risk aversion for medical care.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        MedPrice : float
            Price of unit of medical care relative to unit of consumption.
        PermGroGac : float
            Expected permanent income growth factor at the end of this period.
        PermIncCorr : float
            Correlation of permanent income from period to period.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.
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
        ConsPersistentShockSolver.__init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                 PermGroFac,PermIncCorr,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool)
        self.MedShkDstn = MedShkDstn
        self.MedPrice   = MedPrice
        
    def setAndUpdateValues(self,solution_next,IncomeDstn,LivPrb,DiscFac):
        '''
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, medical shocks and probabilities, next
        period's marginal value function (etc), the probability of getting the
        worst income shock next period, the patience factor, human wealth, and
        the bounding MPCs.
        
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
        # Run basic version of this method
        ConsPersistentShockSolver.setAndUpdateValues(self.solution_next,self.IncomeDstn,self.LivPrb,self.DiscFac)
        
        # Also unpack the medical shock distribution
        self.MedShkPrbs = self.MedShkDstn[0]
        self.MedShkVals = self.MedShkDstn[1]
        
    def getPointsForInterpolation(self,EndOfPrdvP,aLvlNow):
        '''
        Finds endogenous interpolation points (c,m) for the consumption function.
        
        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aLvlNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
            
        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        '''
        # Get size of each state dimension
        mCount      = aLvlNow.shape[1]
        pCount      = aLvlNow.shape[0]
        MedCount = self.MedShkVals.size
        
        # Calculate endogenous gridpoints and controls
        cLvlNow = np.tile(np.reshape(self.uPinv(EndOfPrdvP),(1,pCount,mCount)),(MedCount,1,1))
        MedBaseNow = np.tile(np.reshape(self.uMedPinv(self.MedPrice*EndOfPrdvP),(1,pCount,mCount)),(MedCount,1,1))
        MedShkVals_tiled = np.tile(np.reshape(self.MedShkVals**(1.0/self.CRRAmed),(MedCount,1,1)),(pCount,mCount,1))
        MedLvlNow = MedShkVals_tiled*MedBaseNow
        aLvlNow_tiled = np.tile(np.reshape(aLvlNow,(1,pCount,mCount)),(MedCount,1,1))
        mLvlNow = cLvlNow + MedLvlNow + aLvlNow_tiled

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.concatenate((np.zeros((MedCount,pCount,1)),cLvlNow),axis=-1)
        temp = np.tile(self.BoroCnstNat(np.reshape(self.pLvlGrid,(1,self.pLvlGrid.size,1))),(MedCount,1,1))
        m_for_interpolation = np.concatenate((temp,mLvlNow),axis=-1)
        
        return c_for_interpolation, m_for_interpolation
        
    def makevPfunc(self,cFunc):
        '''
        Constructs the marginal value function for this period.
        
        Parameters
        ----------
        cFunc : function
            Consumption function this period, defined over market resources and
            permanent income level.
        
        Returns
        -------
        vPfunc : function
            Marginal value (of market resources) function for this period.
        '''
        # Get state dimension sizes
        mCount = self.aXtraGrid.size
        pCount = self.pLvlGrid.size
        MedCount = self.MedShkVals.size
        
        # Make temporary grids to evaluate the consumption function
        mGrid_temp = np.tile(np.reshape(self.aXtraGrid,(mCount,1,1)),(1,pCount,MedCount))
        pGrid_temp = np.tile(np.reshape(self.pLvlGrid,(1,pCount,1)),(mCount,1,MedCount))
        MedGrid_temp = np.tile(np.reshape(self.MedShkVals,(1,1,MedCount)),(mCount,pCount,1))
        probs_temp = np.tile(np.reshape(self.MedShkPrbs,(1,1,MedCount)),(mCount,pCount,1))
        
        # Calculate expected marginal value by "integrating" across medical shocks
        cGrid_temp = cFunc(mGrid_temp,pGrid_temp,MedGrid_temp)
        vPgrid_temp= self.uP(cGrid_temp)
        vPnow      = np.sum(vPgrid_temp*probs_temp,axis=2)
        vPnvrsNow  = np.concatenate((np.zeros(1,pCount),self.uPinv(vPnow)))
        # ^^^ Add vPnvrs at m=0 to close it off at the bottom
        
        # Construct and return the marginal value function over mLvl,pLvl
        vPnvrsFunc = BilinearInterp(vPnvrsNow,np.insert(self.aXtraGrid,0,0.0),self.pLvlGrid)
        vPfunc = MargValueFunc2D(vPnvrsFunc,self.CRRA)
        return vPfunc
        