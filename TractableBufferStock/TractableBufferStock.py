'''
This module defines the Tractable Buffer Stock model described in CDC's notes.
'''
# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. 
import sys 
sys.path.insert(0,'../')

from HARKcore import AgentType, NullFunc, Solution
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKutilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityPPP, CRRAutilityPPPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv
from HARKinterpolation import CubicInterp
from copy import copy
from scipy.optimize import newton, brentq
#from cstwMPC import cstwMPCagent

# Define utility function and its derivatives (plus inverses)
utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityPPP = CRRAutilityPPP
utilityPPPP = CRRAutilityPPPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv

class TractableConsumerSolution(Solution):
    '''
    A class representing the solution to a tractable buffer saving problem.
    Attributes include a list of money points mNrm_list, a list of consumption points
    cNrm_list, a list of MPCs MPC_list, a perfect foresight consumption function
    while employed, and a perfect foresight consumption function while unemployed.
    The solution includes a consumption function constructed from the lists.
    '''
    
    def __init__(self, mNrm_list=[], cNrm_list=[], MPC_list=[], cFunc_U=NullFunc, cFunc=NullFunc):
        '''
        The constructor for a new TractableConsumerSolution object. The distance
        between two solutions is the difference in the number of stable arm
        points in each.  This is a very crude measure of distance that captures
        the notion that the process is over when no more points are added.
        '''
        self.mNrm_list = mNrm_list
        self.cNrm_list = cNrm_list
        self.MPC_list = MPC_list
        self.cFunc_U = cFunc_U
        self.cFunc = cFunc
        self.convergence_criteria = ['PointCount']
        


def findNextPoint(DiscFac,Rfree,CRRA,PermGroFacCmp,UnempPrb,Rnrm,Beth,cNext,mNext,MPCNext,PFMPC):
    uPP = lambda x : utilityPP(x,gam=CRRA)
    cNow = PermGroFacCmp*(DiscFac*Rfree)**(-1.0/CRRA)*cNext*(1 + UnempPrb*((cNext/(PFMPC*(mNext-1.0)))**CRRA-1.0))**(-1.0/CRRA)
    mNow = (PermGroFacCmp/Rfree)*(mNext - 1.0) + cNow
    cUNext = PFMPC*(mNow-cNow)*Rnrm
    natural = Beth*Rnrm*(1.0/uPP(cNow))*((1.0-UnempPrb)*uPP(cNext)*MPCNext + UnempPrb*uPP(cUNext)*PFMPC)
    MPCnow = natural / (natural + 1)
    return mNow, cNow, MPCnow
        

def addToStableArmPoints(solution_next,DiscFac,Rfree,CRRA,PermGroFacCmp,UnempPrb,PFMPC,Rnrm,Beth,mLowerBnd,mUpperBnd):
    '''
    This is the solveAPeriod function for the Tractable Buffer Stock model.  If
    the bounding levels of mLowerBnd (lower) and mUpperBnd (upper) have not yet been met
    by a stable arm point in mNrm_list, it adds a point to each end of the arm.  It
    is the contents of the "backshooting" loop.
    '''     
    # Unpack the lists of Euler points
    mNrm_list = copy(solution_next.mNrm_list)
    cNrm_list = copy(solution_next.cNrm_list)
    MPC_list = copy(solution_next.MPC_list)
    
    # Check whether to add a stable arm point to the top
    mNext = mNrm_list[-1]
    if mNext < mUpperBnd:
        # Get the rest of the data for the previous top point
        cNext = solution_next.cNrm_list[-1]
        MPCNext = solution_next.MPC_list[-1]
        
        # Calculate employed levels of c, m, and MPC from next period's values
        mNow, cNow, MPCnow = findNextPoint(DiscFac,Rfree,CRRA,PermGroFacCmp,UnempPrb,Rnrm,Beth,cNext,mNext,MPCNext,PFMPC)
        
        # Add this point to the top of the stable arm list
        mNrm_list.append(mNow)
        cNrm_list.append(cNow)
        MPC_list.append(MPCnow)
    
    # Check whether to add a stable arm point to the bottom
    mNext = mNrm_list[0]
    if mNext > mLowerBnd:
        # Get the rest of the data for the previous bottom point
        cNext = solution_next.cNrm_list[0]
        MPCNext = solution_next.MPC_list[0]
        
        # Calculate employed levels of c, m, and MPC from next period's values
        mNow, cNow, MPCnow = findNextPoint(DiscFac,Rfree,CRRA,PermGroFacCmp,UnempPrb,Rnrm,Beth,cNext,mNext,MPCNext,PFMPC)
        
        # Add this point to the top of the stable arm list
        mNrm_list.insert(0,mNow)
        cNrm_list.insert(0,cNow)
        MPC_list.insert(0,MPCnow)
        
    # Construct and return this period's solution
    solutionNow = TractableConsumerSolution(mNrm_list=mNrm_list, cNrm_list=cNrm_list, MPC_list=MPC_list)
    solutionNow.PointCount = len(mNrm_list)
    return solutionNow
    

class TractableConsumerType(AgentType):

    def __init__(self,cycles=0,time_flow=False,**kwds):
        '''
        Instantiate a new TractableConsumerType with given data.
        '''       
        # Initialize a basic AgentType
        AgentType.__init__(self,cycles=cycles,time_flow=time_flow,pseudo_terminal=True,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = []
        self.time_inv = ['DiscFac','Rfree','CRRA','PermGroFacCmp','UnempPrb','PFMPC','Rnrm','Beth','mLowerBnd','mUpperBnd']
        self.solveOnePeriod = addToStableArmPoints
        
    def preSolve(self):
        '''
        This method calculates all of the solution objects that can be obtained
        before conducting the backshooting routine, including the target levels,
        the perfect foresight solution, (marginal) consumption at m=0, and the
        small perturbations around the steady state.
        '''
        # Define utility functions
        uPP = lambda x : utilityPP(x,gam=self.CRRA)
        uPPP = lambda x : utilityPPP(x,gam=self.CRRA)
        uPPPP = lambda x : utilityPPPP(x,gam=self.CRRA)
        
        # Define some useful constants from model primitives
        self.PermGroFacCmp = self.PermGroFac/(1.0-self.UnempPrb) #"uncertainty compensated" wage growth factor
        self.Rnrm = self.Rfree/self.PermGroFacCmp # net interest factor (Rfree normalized by wage growth)
        self.PFMPC= 1.0-(self.Rfree**(-1.0))*(self.Rfree*self.DiscFac)**(1.0/self.CRRA) # MPC for a perfect forsight consumer
        self.Beth = self.Rnrm*self.DiscFac*self.PermGroFacCmp**(1.0-self.CRRA)
        
        # Verify that this consumer is impatient
        PatFacGrowth = (self.Rfree*self.DiscFac)**(1.0/self.CRRA)/self.PermGroFacCmp 
        PatFacReturn = (self.Rfree*self.DiscFac)**(1.0/self.CRRA)/self.Rfree
        if PatFacReturn >= 1.0:
            raise Exception("Employed consumer not return impatient, cannot solve!")
        if PatFacGrowth >= 1.0:
            raise Exception("Employed consumer not growth impatient, cannot solve!")
            
        # Find target money and consumption
        Pi = (1+(PatFacGrowth**(-self.CRRA)-1.0)/self.UnempPrb)**(1/self.CRRA)
        self.h = (1.0/(1.0-self.PermGroFac/self.Rfree))
        zeta = self.Rnrm*self.PFMPC*Pi
        self.mTarg = 1.0+(self.Rfree/(self.PermGroFacCmp+zeta*self.PermGroFacCmp-self.Rfree))
        self.cTarg = (1.0-self.Rnrm**(-1.0))*self.mTarg+self.Rnrm**(-1.0)
        mTargU = (self.mTarg - self.cTarg)*self.Rnrm
        cTargU = mTargU*self.PFMPC
        self.SSperturbance = self.mTarg*0.1
        
        # Find the MPC, MMPC, and MMMPC at the target
        mpcTargFixedPointFunc = lambda k : k*uPP(self.cTarg) - self.Beth*((1.0-self.UnempPrb)*(1.0-k)*k*self.Rnrm*uPP(self.cTarg)+self.PFMPC*self.UnempPrb*(1.0-k)*self.Rnrm*uPP(cTargU))
        self.MPCtarg = newton(mpcTargFixedPointFunc,0)
        mmpcTargFixedPointFunc = lambda kk : kk*uPP(self.cTarg) + self.MPCtarg**2.0*uPPP(self.cTarg) - self.Beth*(-(1.0 - self.UnempPrb)*self.MPCtarg*kk*self.Rnrm*uPP(self.cTarg)+(1.0-self.UnempPrb)*(1.0 - self.MPCtarg)**2.0*kk*self.Rnrm**2.0*uPP(self.cTarg)-self.PFMPC*self.UnempPrb*kk*self.Rnrm*uPP(cTargU)+(1.0-self.UnempPrb)*(1.0-self.MPCtarg)**2.0*self.MPCtarg**2.0*self.Rnrm**2.0*uPPP(self.cTarg)+self.PFMPC**2.0*self.UnempPrb*(1.0-self.MPCtarg)**2.0*self.Rnrm**2.0*uPPP(cTargU))
        self.MMPCtarg = newton(mmpcTargFixedPointFunc,0)
        mmmpcTargFixedPointFunc = lambda kkk : kkk * uPP(self.cTarg) + 3 * self.MPCtarg * self.MMPCtarg * uPPP(self.cTarg) + self.MPCtarg**3 * uPPPP(self.cTarg) - self.Beth * (-(1 - self.UnempPrb) * self.MPCtarg * kkk * self.Rnrm * uPP(self.cTarg) - 3 * (1 - self.UnempPrb) * (1 - self.MPCtarg) * self.MMPCtarg**2 * self.Rnrm**2 * uPP(self.cTarg) + (1 - self.UnempPrb) * (1 - self.MPCtarg)**3 * kkk * self.Rnrm**3 * uPP(self.cTarg) - self.PFMPC * self.UnempPrb * kkk * self.Rnrm * uPP(cTargU) - 3 * (1 - self.UnempPrb) * (1 - self.MPCtarg) * self.MPCtarg**2 * self.MMPCtarg * self.Rnrm**2 * uPPP(self.cTarg) + 3 * (1 - self.UnempPrb) * (1 - self.MPCtarg)**3 * self.MPCtarg * self.MMPCtarg * self.Rnrm**3 * uPPP(self.cTarg) - 3 * self.PFMPC**2 * self.UnempPrb * (1 - self.MPCtarg) * self.MMPCtarg * self.Rnrm**2 * uPPP(cTargU) + (1 - self.UnempPrb) * (1 - self.MPCtarg)**3 * self.MPCtarg**3 * self.Rnrm**3 * uPPPP(self.cTarg) + self.PFMPC**3 * self.UnempPrb * (1 - self.MPCtarg)**3 * self.Rnrm**3 * uPPPP(cTargU))
        self.MMMPCtarg = newton(mmmpcTargFixedPointFunc,0)
        
        # Find the MPC at m=0
        f_temp = lambda k : self.Beth*self.Rnrm*self.UnempPrb*(self.PFMPC*self.Rnrm*((1.0-k)/k))**(-self.CRRA-1.0)*self.PFMPC
        mpcAtZeroFixedPointFunc = lambda k : k - f_temp(k)/(1 + f_temp(k))
        #self.MPCmax = newton(mpcAtZeroFixedPointFunc,0.5)
        self.MPCmax = brentq(mpcAtZeroFixedPointFunc,self.PFMPC,0.99,xtol=0.00000001,rtol=0.00000001)
        
        # Make the initial list of Euler points: target and perturbation to either side
        mNrm_list = [self.mTarg-self.SSperturbance, self.mTarg, self.mTarg+self.SSperturbance]
        c_perturb_lo = self.cTarg - self.SSperturbance*self.MPCtarg + 0.5*self.SSperturbance**2.0*self.MMPCtarg - (1.0/6.0)*self.SSperturbance**3.0*self.MMMPCtarg
        c_perturb_hi = self.cTarg + self.SSperturbance*self.MPCtarg + 0.5*self.SSperturbance**2.0*self.MMPCtarg + (1.0/6.0)*self.SSperturbance**3.0*self.MMMPCtarg
        cNrm_list = [c_perturb_lo, self.cTarg, c_perturb_hi]
        MPC_perturb_lo = self.MPCtarg - self.SSperturbance*self.MMPCtarg + 0.5*self.SSperturbance**2.0*self.MMMPCtarg
        MPC_perturb_hi = self.MPCtarg + self.SSperturbance*self.MMPCtarg + 0.5*self.SSperturbance**2.0*self.MMMPCtarg
        MPC_list = [MPC_perturb_lo, self.MPCtarg, MPC_perturb_hi]
        
        # Set bounds for money (stable arm construction stops when these are exceeded)
        self.mLowerBnd = 1.0
        self.mUpperBnd = 2.0*self.mTarg
        
        # Make the terminal period solution
        solution_terminal = TractableConsumerSolution(mNrm_list=mNrm_list,cNrm_list=cNrm_list,MPC_list=MPC_list)
        self.solution_terminal = solution_terminal
        
        # Make two linear steady state functions
        self.cSSfunc = lambda m : m*((self.Rnrm*self.PFMPC*Pi)/(1.0+self.Rnrm*self.PFMPC*Pi))
        self.mSSfunc = lambda m : (self.PermGroFacCmp/self.Rfree)+(1.0-self.PermGroFacCmp/self.Rfree)*m
                
    def postSolve(self):
        '''
        This method adds consumption at m=0 to the list of stable arm points,
        then constructs the consumption function as a cubic interpolation over
        those points.  Should be run after the backshooting routine is complete.
        '''
        # Add bottom point to the stable arm points
        self.solution[0].mNrm_list.insert(0,0.0)
        self.solution[0].cNrm_list.insert(0,0.0)
        self.solution[0].MPC_list.insert(0,self.MPCmax)
        
        # Construct an interpolation of the consumption function from the stable arm points
        self.solution[0].cFunc = CubicInterp(self.solution[0].mNrm_list,self.solution[0].cNrm_list,self.solution[0].MPC_list,self.PFMPC*(self.h-1.0),self.PFMPC)
        self.solution[0].cFunc_U = lambda m : self.PFMPC*m
        #self.cFunc = self.solution[0].cFunc
        
    def update():
        '''
        This method does absolutely nothing.
        '''
        