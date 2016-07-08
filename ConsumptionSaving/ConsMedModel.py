'''
Consumption-saving models that also include medical spending.
'''
import sys 
sys.path.insert(0,'../')

import numpy as np
from HARKcore import HARKobject
from HARKutilities import approxLognormal, addDiscreteOutcomeConstantMean, CRRAutilityP_inv
from ConsIndShockModel import ConsumerSolution
from HARKinterpolation import BilinearInterpOnInterp1D, BilinearInterp, LinearInterp
from ConsPersistentShockModel import ConsPersistentShockSolver, PersistentShockConsumerType, MargValueFunc2D
from copy import copy, deepcopy

utilityP_inv  = CRRAutilityP_inv

class MedFunc(HARKobject):
    '''
    A class to represent the medical care function for consumers.
    '''
    distance_critera = ['cFunc','CRRAcon','CRRAmed','MedPrice']
    
    def __init__(self,cFunc,CRRAcon,CRRAmed,MedPrice):
        '''
        Make a new medical care function.
        
        Parameters
        -----------
        cFunc : function
            Consumption function, defined over market resources, permanent income,
            and medical need shock.
        CRRAcon : float
            Coefficient of relative risk aversion for consumption.
        CRRAmed : float
            Coefficient of relative risk aversion for medical care.
        MedPrice : float
            Relative price of medical care.
            
        Returns
        -------
        None
        '''
        self.cFunc = cFunc
        self.CRRAcon = CRRAcon
        self.CRRAmed = CRRAmed
        self.MedPrice = MedPrice
        
    def __call__(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the medical care function at given state space points.
        
        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        mLvl : np.array
            Permanent income levels.
        MedShk : np.array
            Medical need shocks.
            
        Returns
        -------
        Med : np.array
            Array of same shape as inputs, containing medical care levels.
        '''
        Med = (MedShk/self.MedPrice)**(1.0/self.CRRAmed) * self.cFunc(mLvl,pLvl,MedShk)**(self.CRRAcon/self.CRRAmed)
        return Med
        
    def derivativeX(self,mLvl,pLvl,MedShk):
        '''
        Evaluate marginal medical care w.r.t market resources at given state
        space points.
        
        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        mLvl : np.array
            Permanent income levels.
        MedShk : np.array
            Medical need shocks.
            
        Returns
        -------
        dMeddm : np.array
            Array of same shape as inputs, containing marginal medical care.
        '''
        # WRITE THIS METHOD
        
    def derivativeY(self,mLvl,pLvl,MedShk):
        '''
        Evaluate marginal medical care w.r.t permanent income at given state
        space points.
        
        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        mLvl : np.array
            Permanent income levels.
        MedShk : np.array
            Medical need shocks.
            
        Returns
        -------
        dMeddp : np.array
            Array of same shape as inputs, containing marginal medical care.
        '''
        # WRITE THIS METHOD
        
    def derivativeZ(self,mLvl,pLvl,MedShk):
        '''
        Evaluate marginal medical care w.r.t medical need shock at given state
        space points.
        
        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        mLvl : np.array
            Permanent income levels.
        MedShk : np.array
            Medical need shocks.
            
        Returns
        -------
        dMeddMedShk : np.array
            Array of same shape as inputs, containing marginal medical care.
        '''
        # WRITE THIS METHOD

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
        self.addToTimeVary('MedPrice')
        
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
        self.updateMedShockProcess() # Update the discrete medical shock process
        
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
        MedShkDstn = [] # empty list for medical shock distribution each period
        for t in range(self.T_total):
            MedShkAvgNow  = self.MedShkAvg[t] # get shock distribution parameters
            MedShkStdNow  = self.MedShkStd[t]
            MedShkDstnNow = approxLognormal(mu=np.log(MedShkAvgNow), sigma=MedShkStdNow, N=self.MedShkCount, tail_N=self.MedShkCountTail, tail_bound=[0,0.9])
            MedShkDstnNow = addDiscreteOutcomeConstantMean(MedShkDstnNow,0.0,0.0,sort=True) # add point at zero with no probability
            MedShkDstn.append(MedShkDstnNow)
        self.MedShkDstn = MedShkDstn
        self.addToTimeVary('MedShkDstn')
        
    def updateIncomeProcess(self):
        '''
        Updates this agent's income process based on his own attributes.  The
        function that generates the discrete income process can be swapped out
        for a different process.
        
        Parameters
        ----------
        None
        
        Returns:
        --------
        None
        '''
        PersistentShockConsumerType.updateIncomeProcess(self)
        tiny_prob = 1e-15
        for j in range(len(self.IncomeDstn)): # Add very tiny event at TranShk=0
            self.IncomeDstn[j][0] = np.insert(self.IncomeDstn[j][0],0,tiny_prob)
            self.IncomeDstn[j][1] = np.insert(self.IncomeDstn[j][1],0,1.0)
            self.IncomeDstn[j][2] = np.insert(self.IncomeDstn[j][2],0,0.0)
        
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
            self.pLvlGrid[j] = np.insert(this_grid,0,0.0)
        
###############################################################################
        
class ConsMedShockSolver(ConsPersistentShockSolver):
    '''
    Class for solving the one period problem for the "medical shocks" model, in
    which consumers receive shocks to permanent and transitory income as well as
    shocks to "medical need"-- multiplicative utility shocks for a second good.
    '''
    def __init__(self,solution_next,IncomeDstn,MedShkDstn,LivPrb,DiscFac,CRRA,CRRAmed,Rfree,MedPrice,
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
        self.CRRAmed    = CRRAmed
        
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
        ConsPersistentShockSolver.setAndUpdateValues(self,self.solution_next,self.IncomeDstn,self.LivPrb,self.DiscFac)
        
        # Also unpack the medical shock distribution
        self.MedShkPrbs = self.MedShkDstn[0]
        self.MedShkVals = self.MedShkDstn[1]
        
    def defUtilityFuncs(self):
        '''
        Defines CRRA utility function for this period (and its derivatives,
        and their inverses), saving them as attributes of self for other methods
        to use.  Extends version from ConsIndShock models by also defining inverse
        marginal utility function over medical care.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        ConsPersistentShockSolver.defUtilityFuncs(self) # Do basic version
        self.uMedPinv = lambda Med : utilityP_inv(Med,gam=self.CRRAmed)
               
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
        p_for_interpolation : np.array
            Corresponding permanent income points for interpolation.
        '''
        # Get size of each state dimension
        mCount      = aLvlNow.shape[1]
        pCount      = aLvlNow.shape[0]
        MedCount = self.MedShkVals.size
        
        # Calculate endogenous gridpoints and controls
        cLvlNow = np.tile(np.reshape(self.uPinv(EndOfPrdvP),(1,pCount,mCount)),(MedCount,1,1))
        MedBaseNow = np.tile(np.reshape(self.uMedPinv(self.MedPrice*EndOfPrdvP),(1,pCount,mCount)),(MedCount,1,1))
        MedShkVals_tiled = np.tile(np.reshape(self.MedShkVals**(1.0/self.CRRAmed),(MedCount,1,1)),(1,pCount,mCount))
        MedLvlNow = MedShkVals_tiled*MedBaseNow
        aLvlNow_tiled = np.tile(np.reshape(aLvlNow,(1,pCount,mCount)),(MedCount,1,1))
        mLvlNow = cLvlNow + MedLvlNow + aLvlNow_tiled

        # Limiting consumption is zero as m approaches the natural borrowing constraint
        c_for_interpolation = np.concatenate((np.zeros((MedCount,pCount,1)),cLvlNow),axis=-1)
        temp = np.tile(self.BoroCnstNat(np.reshape(self.pLvlGrid,(1,self.pLvlGrid.size,1))),(MedCount,1,1))
        m_for_interpolation = np.concatenate((temp,mLvlNow),axis=-1)
        
        # Make a 3D array of permanent income for interpolation
        p_for_interpolation = np.tile(np.reshape(self.pLvlGrid,(1,pCount,1)),(MedCount,1,mCount+1))
        
        return c_for_interpolation, m_for_interpolation, p_for_interpolation
        
    def usePointsForInterpolation(self,cLvl,mLvl,pLvl,MedShk,interpolator):
        '''
        Constructs a basic solution for this period, including the consumption
        function and marginal value function.
        
        Parameters
        ----------
        cLvl : np.array
            Consumption points for interpolation.
        mLvl : np.array
            Corresponding market resource points for interpolation.
        pLvl : np.array
            Corresponding permanent income level points for interpolation.
        MedShk : np.array
            Corresponding medical need shocks for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.
            
        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        '''
        # Construct the unconstrained consumption function
        cFuncNowUnc = interpolator(mLvl,pLvl,MedShk,cLvl)

        # Currently can't handle constrained consumption function, cFunc is just cFuncUnc
        cFuncNow = cFuncNowUnc

        # Make the marginal value function
        vPfuncNow = self.makevPfunc(cFuncNow)

        # Pack up the solution and return it
        solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow)
        return solution_now
        
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
        vPnvrsNow  = np.concatenate((np.zeros((1,pCount)),self.uPinv(vPnow)))
        # ^^^ Add vPnvrs at m=0 to close it off at the bottom
        
        # Construct and return the marginal value function over mLvl,pLvl
        vPnvrsFunc = BilinearInterp(vPnvrsNow,np.insert(self.aXtraGrid,0,0.0),self.pLvlGrid)
        vPfunc = MargValueFunc2D(vPnvrsFunc,self.CRRA)
        return vPfunc
        
    def makeLinearcFunc(self,mLvl,pLvl,MedShk,cLvl):
        '''
        Constructs the (unconstrained) consumption function for this period using
        bilinear interpolation (over permanent income and the medical shock) among
        an array of linear interpolations over market resources.
        
        Parameters
        ----------
        mLvl : np.array
            Corresponding market resource points for interpolation.
        pLvl : np.array
            Corresponding permanent income level points for interpolation.
        MedShk : np.array
            Corresponding medical need shocks for interpolation.
        cLvl : np.array
            Consumption points for interpolation, corresponding to those in mLvl,
            pLvl, and MedShk.
            
        Returns
        -------
        cFuncUnc : BilinearInterpOnInterp1D
            Unconstrained consumption function for this period.
        '''
        # Get state dimensions
        pCount = mLvl.shape[1]
        MedCount = mLvl.shape[0]
        
        # Initialize the empty list of lists of 1D cFuncs
        cFunc_by_pLvl_and_MedShk = []
        
        # Loop over each permanent income level and medical shock and make a linear cFunc
        for i in range(pCount):
            temp_list = []
            for j in range(MedCount):
                m_temp = mLvl[j,i,:]
                c_temp = cLvl[j,i,:]
                temp_list.append(LinearInterp(m_temp,c_temp))
            cFunc_by_pLvl_and_MedShk.append(deepcopy(temp_list))
                
        # Combine the nested list of linear cFuncs into a single function
        pLvl_temp = pLvl[0,:,0]
        MedShk_temp = MedShk[:,0,0]
        cFuncUnc = BilinearInterpOnInterp1D(cFunc_by_pLvl_and_MedShk,pLvl_temp,MedShk_temp)
        return cFuncUnc
        
    def makeBasicSolution(self,EndOfPrdvP,aLvl,interpolator):
        '''
        Given end of period assets and end of period marginal value, construct
        the basic solution for this period.
        
        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aLvl : np.array
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
        cLvl,mLvl,pLvl = self.getPointsForInterpolation(EndOfPrdvP,aLvl)
        MedShk_temp    = np.tile(np.reshape(self.MedShkVals,(self.MedShkVals.size,1,1)),(1,mLvl.shape[1],mLvl.shape[2]))
        solution_now   = self.usePointsForInterpolation(cLvl,mLvl,pLvl,MedShk_temp,interpolator)
        solution_now.MedFunc = MedFunc(solution_now.cFunc,self.CRRA,self.CRRAmed,self.MedPrice)
        return solution_now
        
    def solve(self):
        '''
        Solves a one period consumption saving problem with risky income and 
        shocks to medical need.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem, including a consumption
            function, medical spending function ( both defined over market re-
            sources, permanent income, and medical shock), a marginal value func-
            tion (defined over market resources and permanent income), and human
            wealth as a function of permanent income.
        '''
        aLvl,trash  = self.prepareToCalcEndOfPrdvP()           
        EndOfPrdvP = self.calcEndOfPrdvP()
        if np.all(self.mLvlMinNow(self.pLvlGrid) == 0.0):   
            interpolator = self.makeLinearcFunc
        else: # Solver only works with lower bound of m=0 everywhere at this time
            assert False, "Medical shocks model can't handle mLvlMin < 0 yet!"
        solution   = self.makeBasicSolution(EndOfPrdvP,aLvl,interpolator)
        solution   = self.addMPCandHumanWealth(solution)
        return solution
        
        
def solveConsMedShock(solution_next,IncomeDstn,MedShkDstn,LivPrb,DiscFac,CRRA,CRRAmed,Rfree,MedPrice,
                 PermGroFac,PermIncCorr,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool):
    '''
    Solve the one period problem for a consumer with shocks to permanent and
    transitory income as well as medical need shocks (as multiplicative shifters
    for utility from a second medical care good).
    
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
    solution : ConsumerSolution
        Solution to this period's problem, including a consumption function,
        medical spending function, and marginal value function.  The former two
        are defined over (mLvl,pLvl,MedShk), while the latter is defined only
        on (mLvl,pLvl), with MedShk integrated out.
    '''
    solver = ConsMedShockSolver(solution_next,IncomeDstn,MedShkDstn,LivPrb,DiscFac,CRRA,CRRAmed,Rfree,
                            MedPrice,PermGroFac,PermIncCorr,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool)
    solver.prepareToSolve()       # Do some preparatory work
    solution_now = solver.solve() # Solve the period
    return solution_now


###############################################################################

if __name__ == '__main__':
    import ConsumerParameters as Params
    from time import clock
    import matplotlib.pyplot as plt
    mystr = lambda number : "{:.4f}".format(number)

    # Make an example medical shocks consumer type
    MedicalExample = MedShockConsumerType(**Params.init_medical_shocks)
    MedicalExample.cycles = 100
    t_start = clock()
    MedicalExample.solve()
    t_end = clock()
    print('Solving a medical shocks consumer took ' + mystr(t_end-t_start) + ' seconds.')
    
    # Plot the consumption function
    M = np.linspace(0,30,300)
    P = np.ones_like(M)
    for j in range(MedicalExample.MedShkDstn[0][0].size):
        MedShk = MedicalExample.MedShkDstn[0][1][j]*np.ones_like(M)
        C = MedicalExample.solution[0].cFunc(M,P,MedShk)
        plt.plot(M,C)
    plt.show()
    
    # Plot the medical care function
    for j in range(MedicalExample.MedShkDstn[0][0].size):
        MedShk = MedicalExample.MedShkDstn[0][1][j]*np.ones_like(M)
        Med = MedicalExample.solution[0].MedFunc(M,P,MedShk)
        plt.plot(M,Med)
    plt.show()
    
