'''
This module contains extensions to ConsumptionSavingModel.py concerning models
with preference shocks.  It currently only has one model, in which utility is
subject to an iid lognormal multiplicative shock each period; it assumes that
there are different interest rates on borrowing and saving.
'''
import sys 
sys.path.insert(0,'../')

import numpy as np
from HARKutilities import approxLognormal
from copy import copy
from ConsumptionSavingModel import ConsumerType, ConsumerSolution, ConsumptionSavingSolverKinkedR, ValueFunc, MargValueFunc
from HARKinterpolation import LinearInterpOnInterp1D, LinearInterp, CubicInterp, LowerEnvelope

class PrefShockConsumerType(ConsumerType):
    '''
    A class for representing consumers who experience multiplicative shocks to
    utility each period, specified as iid lognormal.
    '''
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new PrefShockConsumerType with given data.
        '''
        ConsumerType.__init__(self,**kwds)
        self.solveOnePeriod = solveConsPrefShock
        self.time_inv.remove('Rfree')
        self.time_inv.append('Rboro')
        self.time_inv.append('Rsave')
    
    def update(self):
        ConsumerType.update(self)     # Update assets grid, income process, terminal solution
        self.updatePrefShockProcess() # Update the discrete preference shock process
        
    def updatePrefShockProcess(self):
        '''
        Make a discrete shock structure for each period in the cycle for this
        agent type.
        '''
        time_orig = self.time_flow
        self.timeFwd()
        PrefShkDstn = []
        for t in range(len(self.PrefShkStd)):
            PrefShkStd = self.PrefShkStd[t]
            PrefShkDstn.append(approxLognormal(N=self.PrefShkCount,mu=0.0,sigma=PrefShkStd,tail_N=self.PrefShk_tail_N))
        self.PrefShkDstn = PrefShkDstn
        if not 'PrefShkDstn' in self.time_vary:
            self.time_vary.append('PrefShkDstn')
        if not time_orig:
            self.timeRev()
            
    def makePrefShkHist(self):
        '''
        Makes histories of simulated preference shocks for this consumer type by
        drawing from the shock distribution's true lognormal form.
        '''
        orig_time = self.time_flow
        self.timeFwd()
        self.resetRNG()
        
        # Initialize the preference shock history
        PrefShkHist = np.zeros((self.sim_periods,self.Nagents)) + np.nan
        PrefShkHist[0,:] = 1.0
        t_idx = 0
        
        # Make discrete distributions of preference shocks to permute
        base_dstns = []
        for t_idx in range(len(self.PrefShkStd)):
            temp_dstn = approxLognormal(N=self.Nagents,mu=0.0,sigma=self.PrefShkStd[t_idx])
            base_dstns.append(temp_dstn[1]) # only take values, not probs
        
        # Fill in the preference shock history
        for t in range(1,self.sim_periods):
            dstn_now         = base_dstns[t_idx]
            PrefShkHist[t,:] = self.RNG.permutation(dstn_now)
            t_idx += 1
            if t_idx >= len(self.PrefShkStd):
                t_idx = 0
                
        self.PrefShkHist = PrefShkHist
        if not orig_time:
            self.timeRev()
            
    def advanceIncShks(self):
        '''
        Advance the permanent and transitory income shocks to the next period of
        the shock history objects, after first advancing the preference shocks.
        '''
        self.PrefShkNow = self.PrefShkHist[self.Shk_idx,:]
        ConsumerType.advanceIncShks(self)
            
    def simOnePrd(self):
        '''
        Simulate a single period of a consumption-saving model with permanent
        and transitory income shocks plus multiplicative utility shocks.
        '''
        # Unpack objects from self for convenience
        aPrev          = self.aNow
        pPrev          = self.pNow
        TranShkNow     = self.TranShkNow
        PermShkNow     = self.PermShkNow
        PrefShkNow     = self.PrefShkNow
        RfreeNow       = self.RfreeNow
        cFuncNow       = self.cFuncNow
        
        # Simulate the period
        pNow    = pPrev*PermShkNow      # Updated permanent income level
        ReffNow = RfreeNow/PermShkNow   # "effective" interest factor on normalized assets
        bNow    = ReffNow*aPrev         # Bank balances before labor income
        mNow    = bNow + TranShkNow     # Market resources after income
        cNow    = cFuncNow(mNow,PrefShkNow) # Consumption (normalized)
        MPCnow  = cFuncNow.derivativeX(mNow,PrefShkNow) # Marginal propensity to consume
        aNow    = mNow - cNow           # Assets after all actions are accomplished
        
        # Store the new state and control variables
        self.pNow   = pNow
        self.bNow   = bNow
        self.mNow   = mNow
        self.cNow   = cNow
        self.MPCnow = MPCnow
        self.aNow   = aNow


class PrefShockSolver(ConsumptionSavingSolverKinkedR):
    '''
    A one period solver for the preference shock model.
    '''
    def __init__(self,solution_next,IncomeDstn,PrefShkDstn,LivPrb,DiscFac,CRRA,
                      Rboro,Rsave,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
        '''
        Initialize the solver for this period, which largely duplicates the initialization
        with no preference shocks.
        '''
        ConsumptionSavingSolverKinkedR.__init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,
                      Rboro,Rsave,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
        self.PrefShkPrbs = PrefShkDstn[0]
        self.PrefShkVals = PrefShkDstn[1]
    
    def getPointsForInterpolation(self,EndOfPrdvP,aNrmNow):
        '''
        Find endogenous interpolation points for each asset point and each
        discrete preference shock.
        '''
        c_base = self.uPinv(EndOfPrdvP)
        PrefShkCount = self.PrefShkVals.size
        PrefShk_temp = np.tile(np.reshape(self.PrefShkVals**(1.0/self.CRRA),(PrefShkCount,1)),(1,c_base.size))
        self.cNrmNow = np.tile(c_base,(PrefShkCount,1))*PrefShk_temp
        self.mNrmNow = self.cNrmNow + np.tile(aNrmNow,(PrefShkCount,1))
        
        # Add the bottom point to the c and m arrays
        m_for_interpolation = np.concatenate((self.BoroCnstNat*np.ones((PrefShkCount,1)),self.mNrmNow),axis=1)
        c_for_interpolation = np.concatenate((np.zeros((PrefShkCount,1)),self.cNrmNow),axis=1)
        return c_for_interpolation,m_for_interpolation
    
    def usePointsForInterpolation(self,cNrm,mNrm,interpolator):
        '''
        Make a basic solution object with a consumption function and marginal
        value function (unconditional on the preference shock).
        '''
        # Make the preference-shock specific consumption functions
        PrefShkCount = self.PrefShkVals.size
        cFunc_list = []
        for j in range(PrefShkCount):
            MPCmin_j = self.MPCminNow*self.PrefShkVals[j]**(1.0/self.CRRA)
            cFunc_this_shock = LowerEnvelope(LinearInterp(mNrm[j,:],cNrm[j,:],intercept_limit=self.hNrmNow*MPCmin_j,slope_limit=MPCmin_j),self.cFuncNowCnst)
            cFunc_list.append(cFunc_this_shock)
            
        # Combine the list of consumption functions into a single interpolation
        cFuncNow = LinearInterpOnInterp1D(cFunc_list,self.PrefShkVals)
            
        # Make the ex ante marginal value function (before the preference shock)
        m_grid = self.aXtraGrid + self.mNrmMinNow
        vP_vec = np.zeros_like(m_grid)
        for j in range(PrefShkCount): # numeric integration over the preference shock
            vP_vec += self.uP(cFunc_list[j](m_grid))*self.PrefShkPrbs[j]*self.PrefShkVals[j]
        vPnvrs_vec = self.uPinv(vP_vec)
        vPfuncNow = MargValueFunc(LinearInterp(m_grid,vPnvrs_vec),self.CRRA)
    
        # Store the results in a solution object and return it
        solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow)
        return solution_now
        
    def makevFunc(self,solution):
        '''
        Make the beginning-of-period value function (unconditional on the shock).
        '''
        # Compute expected value and marginal value on a grid of market resources,
        # accounting for all of the discrete preference shocks
        PrefShkCount = self.PrefShkVals.size
        mNrm_temp   = self.mNrmMinNow + self.aXtraGrid
        vNrmNow     = np.zeros_like(mNrm_temp)
        vPnow       = np.zeros_like(mNrm_temp)
        for j in range(PrefShkCount):
            this_shock  = self.PrefShkVals[j]
            this_prob   = self.PrefShkPrbs[j]
            cNrmNow     = solution.cFunc(mNrm_temp,this_shock*np.ones_like(mNrm_temp))
            aNrmNow     = mNrm_temp - cNrmNow
            vNrmNow    += this_prob*(this_shock*self.u(cNrmNow) + self.EndOfPrdvFunc(aNrmNow))
            vPnow      += this_prob*this_shock*self.uP(cNrmNow)
        
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
        
        
def solveConsPrefShock(solution_next,IncomeDstn,PrefShkDstn,
                       LivPrb,DiscFac,CRRA,Rboro,Rsave,PermGroFac,BoroCnstArt,
                       aXtraGrid,vFuncBool,CubicBool):
    '''
    Solves a single period of a consumption-saving model with preference shocks
    to marginal utility.  Problem is solved using the method of endogenous gridpoints.

    Parameters:
    -----------
    solution_next: ConsumerSolution
        The solution to the following period.
    IncomeDstn: [np.array]
        A list containing three arrays of floats, representing a discrete approx-
        imation to the income process between the period being solved and the one
        immediately following (in solution_next).  Order: ShkPrbs, PermShkVals, TranShkVals
    PrefShkDstn: [np.array]
        Discrete distribution of the multiplicative utility shifter.
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
        Expected growth factor for permanent income between this period and the
        succeeding period.
    BoroCnstArt: float
        Borrowing constraint for the minimum allowable assets to end the period
        with.  If it is less than the natural borrowing constraint, then it is
        irrelevant; BoroCnstArt=None indicates no artificial borrowing constraint.
    aXtraGrid: [float]
        A list of end-of-period asset values (post-decision states) at which to
        solve for optimal consumption.

    Returns:
    -----------
    solution_now: ConsumerSolution
        The solution to this period's problem, obtained using the method of endogenous gridpoints.
    '''

    solver = PrefShockSolver(solution_next,IncomeDstn,PrefShkDstn,LivPrb,
                             DiscFac,CRRA,Rboro,Rsave,PermGroFac,BoroCnstArt,aXtraGrid,
                             vFuncBool,CubicBool)
    solver.prepareToSolve()                                      
    solution = solver.solve()
    return solution



####################################################################################################     
    
if __name__ == '__main__':
    import SetupConsumerParameters as Params
    import matplotlib.pyplot as plt
    from HARKutilities import plotFunc
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)
    
    do_simulation = True

    # Extend the default initialization dictionary
    ConsPrefShock_dict = copy(Params.init_consumer_objects)
    ConsPrefShock_dict['PrefShkCount'] = 12    # Number of points in discrete approximation to preference shock dist
    ConsPrefShock_dict['PrefShk_tail_N'] = 4   # Number of "tail points" on each end of pref shock dist
    ConsPrefShock_dict['PrefShkStd'] = [0.30]  # Standard deviation of utility shocks
    ConsPrefShock_dict['Rboro'] = 1.20         # Interest factor when borrowing
    ConsPrefShock_dict['Rsave'] = 1.03         # Interest factor when saving
    ConsPrefShock_dict['BoroCnstArt'] = None   # Artificial borrowing constraint
    ConsPrefShock_dict['aXtraCount'] = 64      # Number of asset gridpoints
    ConsPrefShock_dict['aXtraMax'] = 100       # Highest asset gridpoint
    ConsPrefShock_dict['T_total'] = 1
    
    # Make and solve a preference shock consumer
    PrefShockExample = PrefShockConsumerType(**ConsPrefShock_dict)
    PrefShockExample.assignParameters(LivPrb = [0.98],
                                      DiscFac = [0.96],
                                      PermGroFac = [1.02],
                                      CubicBool = False,
                                      cycles = 0)
    t_start = clock()
    PrefShockExample.solve()
    t_end = clock()
    print('Solving a preference shock consumer took ' + str(t_end-t_start) + ' seconds.')
    
    # Plot the consumption function at each discrete shock
    m = np.linspace(PrefShockExample.solution[0].mNrmMin,5,200)
    print('Consumption functions at each discrete shock:')
    for j in range(PrefShockExample.PrefShkDstn[0][1].size):
        PrefShk = PrefShockExample.PrefShkDstn[0][1][j]
        c = PrefShockExample.solution[0].cFunc(m,PrefShk*np.ones_like(m))
        plt.plot(m,c)
    plt.show()
    
    print('Consumption function (and MPC) when shock=1:')
    c = PrefShockExample.solution[0].cFunc(m,np.ones_like(m))
    k = PrefShockExample.solution[0].cFunc.derivativeX(m,np.ones_like(m))
    plt.plot(m,c)
    plt.plot(m,k)
    plt.show()
    
    if PrefShockExample.vFuncBool:
            print('Value function (unconditional on shock):')
            plotFunc(PrefShockExample.solution[0].vFunc,PrefShockExample.solution[0].mNrmMin+0.5,5)
    
    # Test the simulator for the pref shock class
    if do_simulation:
        PrefShockExample.sim_periods = 120
        PrefShockExample.makeIncShkHist()
        PrefShockExample.makePrefShkHist()
        PrefShockExample.initializeSim()
        PrefShockExample.simConsHistory()
        
    