'''
Defines and solves the Tractable Buffer Stock model described in CDC's notes.
The model concerns an agent with constant relative risk aversion utility making
decisions over consumption and saving.  He is subject to only a very particular
sort of risk: the possibility that he will become permanently unemployed until
the day he dies; barring this, his income is certain and grows at a constant rate.

The model has an infinite horizon, but is not solved by backward iteration in a
traditional sense.  Because of the very specific assumptions about risk, it is
possible to find the agent's steady state or target level of market resources
when employed, as well as information about the optimal consumption rule at this
target level.  The full consumption function can then be constructed by "back-
shooting", inverting the Euler equation to find what consumption *must have been*
in the previous period.  The consumption function is thus constructed by repeat-
edly adding "stable arm" points to either end of a growing list until specified
bounds are exceeded.

Despite the non-standard solution method, the iterative process can be embedded
in the HARK framework, as shown below.
'''
# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder.
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))

from HARKcore import AgentType, NullFunc, Solution
from HARK.utilities import warnings  # Because of "patch" to warnings modules
from HARK.utilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityPPP, CRRAutilityPPPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv
from HARK.interpolation import CubicInterp
from HARK.simulation import drawLognormal, drawBernoulli
from copy import copy
from scipy.optimize import newton, brentq

# If you want to run the "tractable" version of cstwMPC, uncomment the two lines below
# and have TractableConsumerType inherit from cstwMPCagent rather than AgentType
#sys.path.insert(0, os.path.abspath('../cstwMPC/'))
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
        The constructor for a new TractableConsumerSolution object.

        Parameters
        ----------
        mNrm_list : [float]
            List of normalized market resources points on the stable arm.
        cNrm_list : [float]
            List of normalized consumption points on the stable arm.
        MPC_list : [float]
            List of marginal propensities to consume on the stable arm, corres-
            ponding to the (mNrm,cNrm) points.
        cFunc_U : function
            The (linear) consumption function when permanently unemployed.
        cFunc : function
            The consumption function when employed.

        Returns
        -------
        new instance of TractableConsumerSolution
        '''
        self.mNrm_list = mNrm_list
        self.cNrm_list = cNrm_list
        self.MPC_list = MPC_list
        self.cFunc_U = cFunc_U
        self.cFunc = cFunc
        self.distance_criteria = ['PointCount']
        # The distance between two solutions is the difference in the number of
        # stable arm points in each.  This is a very crude measure of distance
        # that captures the notion that the process is over when no points are added.

def findNextPoint(DiscFac,Rfree,CRRA,PermGroFacCmp,UnempPrb,Rnrm,Beth,cNext,mNext,MPCnext,PFMPC):
    '''
    Calculates what consumption, market resources, and the marginal propensity
    to consume must have been in the previous period given model parameters and
    values of market resources, consumption, and MPC today.

    Parameters
    ----------
    DiscFac : float
        Intertemporal discount factor on future utility.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFacCmp : float
        Permanent income growth factor, compensated for the possibility of
        permanent unemployment.
    UnempPrb : float
        Probability of becoming permanently unemployed.
    Rnrm : float
        Interest factor normalized by compensated permanent income growth factor.
    Beth : float
        Damned if I know.
    cNext : float
        Normalized consumption in the succeeding period.
    mNext : float
        Normalized market resources in the succeeding period.
    MPCnext : float
        The marginal propensity to consume in the succeeding period.
    PFMPC : float
        The perfect foresight MPC; also the MPC when permanently unemployed.

    Returns
    -------
    mNow : float
        Normalized market resources this period.
    cNow : float
        Normalized consumption this period.
    MPCnow : float
        Marginal propensity to consume this period.
    '''
    uPP = lambda x : utilityPP(x,gam=CRRA)
    cNow = PermGroFacCmp*(DiscFac*Rfree)**(-1.0/CRRA)*cNext*(1 + UnempPrb*((cNext/(PFMPC*(mNext-1.0)))**CRRA-1.0))**(-1.0/CRRA)
    mNow = (PermGroFacCmp/Rfree)*(mNext - 1.0) + cNow
    cUNext = PFMPC*(mNow-cNow)*Rnrm
    natural = Beth*Rnrm*(1.0/uPP(cNow))*((1.0-UnempPrb)*uPP(cNext)*MPCnext + UnempPrb*uPP(cUNext)*PFMPC)
    MPCnow = natural / (natural + 1)
    return mNow, cNow, MPCnow


def addToStableArmPoints(solution_next,DiscFac,Rfree,CRRA,PermGroFacCmp,UnempPrb,PFMPC,Rnrm,Beth,mLowerBnd,mUpperBnd):
    '''
    Adds a one point to the bottom and top of the list of stable arm points if
    the bounding levels of mLowerBnd (lower) and mUpperBnd (upper) have not yet
    been met by a stable arm point in mNrm_list.  This acts as the "one period
    solver" / solveOnePeriod in the tractable buffer stock model.

    Parameters
    ----------
    solution_next : TractableConsumerSolution
        The solution object from the previous iteration of the backshooting
        procedure.  Not the "next period" solution per se.
    DiscFac : float
        Intertemporal discount factor on future utility.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    CRRA : float
        Coefficient of relative risk aversion.
    PermGroFacCmp : float
        Permanent income growth factor, compensated for the possibility of
        permanent unemployment.
    UnempPrb : float
        Probability of becoming permanently unemployed.
    PFMPC : float
        The perfect foresight MPC; also the MPC when permanently unemployed.
    Rnrm : float
        Interest factor normalized by compensated permanent income growth factor.
    Beth : float
        Damned if I know.
    mLowerBnd : float
        Lower bound on market resources for the backshooting process.  If
        min(solution_next.mNrm_list) < mLowerBnd, no new bottom point is found.
    mUpperBnd : float
        Upper bound on market resources for the backshooting process.  If
        max(solution_next.mNrm_list) > mUpperBnd, no new top point is found.

    Returns:
    ---------
    solution_now : TractableConsumerSolution
        A new solution object with new points added to the top and bottom.  If
        no new points were added, then the backshooting process is about to end.
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
    solution_now = TractableConsumerSolution(mNrm_list=mNrm_list, cNrm_list=cNrm_list, MPC_list=MPC_list)
    solution_now.PointCount = len(mNrm_list)
    return solution_now


class TractableConsumerType(AgentType):

    def __init__(self,cycles=0,time_flow=False,**kwds):
        '''
        Instantiate a new TractableConsumerType with given data.

        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.

        Returns:
        -----------
        New instance of TractableConsumerType.
        '''
        # Initialize a basic AgentType
        AgentType.__init__(self,cycles=cycles,time_flow=time_flow,pseudo_terminal=True,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = []
        self.time_inv = ['DiscFac','Rfree','CRRA','PermGroFacCmp','UnempPrb','PFMPC','Rnrm','Beth','mLowerBnd','mUpperBnd']
        self.shock_vars = ['eStateNow']
        self.poststate_vars = ['aLvlNow','eStateNow'] # For simulation
        self.solveOnePeriod = addToStableArmPoints # set correct solver

    def preSolve(self):
        '''
        Calculates all of the solution objects that can be obtained before con-
        ducting the backshooting routine, including the target levels, the per-
        fect foresight solution, (marginal) consumption at m=0, and the small
        perturbations around the steady state.

        Parameters
        ----------
        none

        Returns
        -------
        none
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

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        # Add bottom point to the stable arm points
        self.solution[0].mNrm_list.insert(0,0.0)
        self.solution[0].cNrm_list.insert(0,0.0)
        self.solution[0].MPC_list.insert(0,self.MPCmax)

        # Construct an interpolation of the consumption function from the stable arm points
        self.solution[0].cFunc = CubicInterp(self.solution[0].mNrm_list,self.solution[0].cNrm_list,self.solution[0].MPC_list,self.PFMPC*(self.h-1.0),self.PFMPC)
        self.solution[0].cFunc_U = lambda m : self.PFMPC*m

    def update():
        '''
        This method does absolutely nothing, but should remain here for compati-
        bility with cstwMPC when doing the "tractable" version.
        '''
        return None

    def simBirth(self,which_agents):
        '''
        Makes new consumers for the given indices.  Initialized variables include aNrm, as
        well as time variables t_age and t_cycle.  Normalized assets are drawn from a lognormal
        distributions given by aLvlInitMean and aLvlInitStd.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        '''
        # Get and store states for newly born agents
        N = np.sum(which_agents) # Number of new consumers to make
        self.aLvlNow[which_agents] = drawLognormal(N,mu=self.aLvlInitMean,sigma=self.aLvlInitStd,seed=self.RNG.randint(0,2**31-1))
        self.eStateNow[which_agents] = 1.0 # Agents are born employed
        self.t_age[which_agents]   = 0 # How many periods since each agent was born
        self.t_cycle[which_agents] = 0 # Which period of the cycle each agent is currently in
        return None

    def simDeath(self):
        '''
        Trivial function that returns boolean array of all False, as there is no death.

        Parameters
        ----------
        None

        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        '''
        # Nobody dies in this model
        which_agents = np.zeros(self.AgentCount,dtype=bool)
        return which_agents

    def getShocks(self):
        '''
        Determine which agents switch from employment to unemployment.  All unemployed agents remain
        unemployed until death.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        employed = self.eStateNow == 1.0
        N = int(np.sum(employed))
        newly_unemployed = drawBernoulli(N,p=self.UnempPrb,seed=self.RNG.randint(0,2**31-1))
        self.eStateNow[employed] = 1.0 - newly_unemployed

    def getStates(self):
        '''
        Calculate market resources for all agents this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.bLvlNow = self.Rfree*self.aLvlNow
        self.mLvlNow = self.bLvlNow + self.eStateNow

    def getControls(self):
        '''
        Calculate consumption for each agent this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        employed = self.eStateNow == 1.0
        unemployed = np.logical_not(employed)
        cLvlNow = np.zeros(self.AgentCount)
        cLvlNow[employed] = self.solution[0].cFunc(self.mLvlNow[employed])
        cLvlNow[unemployed] = self.solution[0].cFunc_U(self.mLvlNow[unemployed])
        self.cLvlNow = cLvlNow

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
        self.aLvlNow = self.mLvlNow - self.cLvlNow
        return None


###############################################################################

if __name__ == '__main__':
    # Import the HARK library.  The assumption is that this code is in a folder
    # contained in the HARK folder.  Also import the ConsumptionSavingModel
    import numpy as np                   # numeric Python
    from HARK.utilities import plotFuncs  # basic plotting tools
    from ConsMarkovModel import MarkovConsumerType # An alternative, much longer way to solve the TBS model
    from time import clock               # timing utility

    do_simulation = True

    # Define the model primitives
    base_primitives = {'UnempPrb' : .00625,    # Probability of becoming unemployed
                       'DiscFac' : 0.975,      # Intertemporal discount factor
                       'Rfree' : 1.01,         # Risk-free interest factor on assets
                       'PermGroFac' : 1.0025,  # Permanent income growth factor (uncompensated)
                       'CRRA' : 1.0}           # Coefficient of relative risk aversion

    # Define a dictionary to be used in case of simulation
    simulation_values = {'aLvlInitMean' : 0.0,  # Mean of log initial assets for new agents
                         'aLvlInitStd' : 1.0,   # Stdev of log initial assets for new agents
                         'AgentCount' : 10000,  # Number of agents to simulate
                         'T_sim' : 120,         # Number of periods to simulate
                         'T_cycle' : 1}         # Number of periods in the cycle

    # Make and solve a tractable consumer type
    ExampleType = TractableConsumerType(**base_primitives)
    t_start = clock()
    ExampleType.solve()
    t_end = clock()
    print('Solving a tractable consumption-savings model took ' + str(t_end-t_start) + ' seconds.')

    # Plot the consumption function and whatnot
    m_upper = 1.5*ExampleType.mTarg
    conFunc_PF = lambda m: ExampleType.h*ExampleType.PFMPC + ExampleType.PFMPC*m
    #plotFuncs([ExampleType.solution[0].cFunc,ExampleType.mSSfunc,ExampleType.cSSfunc],0,m_upper)
    plotFuncs([ExampleType.solution[0].cFunc,ExampleType.solution[0].cFunc_U],0,m_upper)

    if do_simulation:
        ExampleType(**simulation_values) # Set attributes needed for simulation
        ExampleType.track_vars = ['mLvlNow']
        ExampleType.makeShockHistory()
        ExampleType.initializeSim()
        ExampleType.simulate()


    # Now solve the same model using backward induction rather than the analytic method of TBS.
    # The TBS model is equivalent to a Markov model with two states, one of them absorbing (permanent unemployment).
    MrkvArray = np.array([[1.0-base_primitives['UnempPrb'],base_primitives['UnempPrb']],[0.0,1.0]]) # Define the two state, absorbing unemployment Markov array
    init_consumer_objects = {"CRRA":base_primitives['CRRA'],
                            "Rfree":np.array(2*[base_primitives['Rfree']]), # Interest factor (same in both states)
                            "PermGroFac":[np.array(2*[base_primitives['PermGroFac']/(1.0-base_primitives['UnempPrb'])])], # Unemployment-compensated permanent growth factor
                            "BoroCnstArt":None,   # Artificial borrowing constraint
                            "PermShkStd":[0.0],   # Permanent shock standard deviation
                            "PermShkCount":1,     # Number of shocks in discrete permanent shock distribution
                            "TranShkStd":[0.0],   # Transitory shock standard deviation
                            "TranShkCount":1,     # Number of shocks in discrete permanent shock distribution
                            "T_cycle":1,          # Number of periods in cycle
                            "UnempPrb":0.0,       # Unemployment probability (not used, as the unemployment here is *permanent*, not transitory)
                            "UnempPrbRet":0.0,    # Unemployment probability when retired (irrelevant here)
                            "T_retire":0,         # Age at retirement (turned off)
                            "IncUnemp":0.0,       # Income when unemployed (irrelevant)
                            "IncUnempRet":0.0,    # Income when unemployed and retired (irrelevant)
                            "aXtraMin":0.001,     # Minimum value of assets above minimum in grid
                            "aXtraMax":ExampleType.mUpperBnd, # Maximum value of assets above minimum in grid
                            "aXtraCount":48,      # Number of points in assets grid
                            "aXtraExtra":[None],  # Additional points to include in assets grid
                            "aXtraNestFac":3,     # Degree of exponential nesting when constructing assets grid
                            "LivPrb":[np.array([1.0,1.0])], # Survival probability
                            "DiscFac":base_primitives['DiscFac'], # Intertemporal discount factor
                            'AgentCount':1,       # Number of agents in a simulation (irrelevant)
                            'tax_rate':0.0,       # Tax rate on labor income (irrelevant)
                            'vFuncBool':False,    # Whether to calculate the value function
                            'CubicBool':True,     # Whether to use cubic splines (False --> linear splines)
                            'MrkvArray':[MrkvArray] # State transition probabilities
                            }
    MarkovType = MarkovConsumerType(**init_consumer_objects)   # Make a basic consumer type
    employed_income_dist = [np.ones(1),np.ones(1),np.ones(1)]    # Income distribution when employed
    unemployed_income_dist = [np.ones(1),np.ones(1),np.zeros(1)] # Income distribution when permanently unemployed
    MarkovType.IncomeDstn = [[employed_income_dist,unemployed_income_dist]]  # set the income distribution in each state
    MarkovType.cycles = 0

    # Solve the "Markov TBS" model
    t_start = clock()
    MarkovType.solve()
    t_end = clock()
    MarkovType.unpackcFunc()

    print('Solving the same model "the long way" took ' + str(t_end-t_start) + ' seconds.')
    #plotFuncs([ExampleType.solution[0].cFunc,ExampleType.solution[0].cFunc_U],0,m_upper)
    plotFuncs(MarkovType.cFunc[0],0,m_upper)
    diffFunc = lambda m : ExampleType.solution[0].cFunc(m) - MarkovType.cFunc[0][0](m)
    print('Difference between the (employed) consumption functions:')
    plotFuncs(diffFunc,0,m_upper)

