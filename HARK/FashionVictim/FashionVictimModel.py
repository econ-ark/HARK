'''
Specifies a dynamic model of fashion selection in a world with only two styles:
jock and punk. Forward-looking agents receive utility from the style they choose
based on the proportion of the population with the same style (as well as direct
preferences each style), and pay switching costs if they change.
'''

import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))

from HARK.core import AgentType, Solution, NullFunc
from HARK.interpolation import LinearInterp
from HARK.utilities import approxUniform, plotFuncs
import numpy as np
import scipy.stats as stats
import FashionVictimParams as Params
from copy import copy

class FashionSolution(Solution):
    '''
    A class for representing the single period solution to a FashionVictim problem.
    '''
    def __init__(self,VfuncJock=None,VfuncPunk=None,switchFuncJock=None,switchFuncPunk=None):
        '''
        Make a new instance of FashionSolution.

        Parameters
        ----------
        VfuncJock : function
            Value when entering the period as a jock as a function of the proportion
            of the population dressed as a punk.
        VfuncPunk : function
            Value when entering the period as a punk as a function of the proportion
            of the population dressed as a punk.
        switchFuncJock : function
            Probability of switching styles (to punk) when entering the period
            as a jock, as a function of the population punk proportion.
        switchFuncPunk : function
            Probability of switching styles (to jock) when entering the period
            as a punkk, as a function of the population punk proportion.

        Returns
        -------
        new instance of FashionSolution
        '''
        # Fill in missing function inputs with trivial defaults
        if VfuncJock is None:
            VfuncJock = NullFunc()
        if VfuncPunk is None:
            VfuncPunk = NullFunc()
        if switchFuncJock is None:
            switchFuncJock = NullFunc()
        if switchFuncPunk is None:
            switchFuncPunk = NullFunc()
        self.VfuncJock = VfuncJock
        self.VfuncPunk = VfuncPunk
        self.switchFuncJock = switchFuncJock
        self.switchFuncPunk = switchFuncPunk
        self.distance_criteria = ['VfuncJock','VfuncPunk']


class FashionEvoFunc(Solution):
    '''
    A class for representing a dynamic fashion rule in the FashionVictim model.
    Fashion victims believe that the proportion of punks evolves according to
    a uniform distribution around a linear function of the previous state.
    '''
    def __init__(self,pNextIntercept=None,pNextSlope=None,pNextWidth=None):
        '''
        Make a new instance of FashionEvoFunc.

        Parameters
        ----------
        pNextIntercept : float
            Intercept of the linear evolution rule.
        pNextSlope : float
            Slope of the linear evolution rule.
        pNextWidth : float
            Width of the uniform distribution around the linear function.

        Returns
        -------
        new instance of FashionEvoFunc
        '''
        self.pNextIntercept    = pNextIntercept
        self.pNextSlope        = pNextSlope
        self.pNextWidth        = pNextWidth
        self.distance_criteria = ['pNextSlope','pNextWidth','pNextIntercept']


class FashionMarketInfo():
    '''
    A class for representing the current distribution of styles in the population.
    '''
    def __init__(self,pNow):
        '''
        Make a new instance of FashionMarketInfo.

        Parameters
        ----------
        pNow : float
            The current proportion of the population that is punk.

        Returns
        -------
        new instance of FashionMarketInfo
        '''
        self.pNow = pNow


class FashionVictimType(AgentType):
    '''
    A class for representing types of agents in the fashion victim model.  Agents
    make a binary decision over which style to wear (jock or punk), subject to
    switching costs.  They receive utility directly from their chosen style and
    from the proportion of the population that wears the same style.
    '''
    _solution_terminal = FashionSolution(VfuncJock=LinearInterp(np.array([0.0, 1.0]),np.array([0.0,0.0])),
                                         VfuncPunk=LinearInterp(np.array([0.0, 1.0]),np.array([0.0,0.0])),
                                         switchFuncJock=NullFunc(),
                                         switchFuncPunk=NullFunc())

    def __init__(self,**kwds):
        '''
        Instantiate a new FashionVictim with given data.

        Parameters
        ----------
        **kwds : keyword arguments
            Any number of keyword arguments key=value; each value will be assigned
            to the attribute key in the new instance.

        Returns
        -------
        new instance of FashionVictimType
        '''
        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=FashionVictimType._solution_terminal,cycles=0,time_flow=True,pseudo_terminal=True,**kwds)

        # Add class-specific features
        self.time_inv = ['DiscFac','conformUtilityFunc','punk_utility','jock_utility','switchcost_J2P','switchcost_P2J','pGrid','pEvolution','pref_shock_mag']
        self.time_vary = []
        self.solveOnePeriod = solveFashion
        self.update()

    def updateEvolution(self):
        '''
        Updates the "population punk proportion" evolution array.  Fasion victims
        believe that the proportion of punks in the subsequent period is a linear
        function of the proportion of punks this period, subject to a uniform
        shock.  Given attributes of self pNextIntercept, pNextSlope, pNextCount,
        pNextWidth, and pGrid, this method generates a new array for the attri-
        bute pEvolution, representing a discrete approximation of next period
        states for each current period state in pGrid.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.pEvolution = np.zeros((self.pCount,self.pNextCount))
        for j in range(self.pCount):
            pNow = self.pGrid[j]
            pNextMean = self.pNextIntercept + self.pNextSlope*pNow
            dist = approxUniform(N=self.pNextCount,bot=pNextMean-self.pNextWidth,top=pNextMean+self.pNextWidth)[1]
            self.pEvolution[j,:] = dist

    def update(self):
        '''
        Updates the non-primitive objects needed for solution, using primitive
        attributes.  This includes defining a "utility from conformity" function
        conformUtilityFunc, a grid of population punk proportions, and an array
        of future punk proportions (for each value in the grid).  Results are
        stored as attributes of self.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.conformUtilityFunc = lambda x : stats.beta.pdf(x,self.uParamA,self.uParamB)
        self.pGrid = np.linspace(0.0001,0.9999,self.pCount)
        self.updateEvolution()

    def reset(self):
        '''
        Resets this agent type to prepare it for a new simulation run.  This
        includes resetting the random number generator and initializing the style
        of each agent of this type.
        '''
        self.resetRNG()
        sNow = np.zeros(self.pop_size)
        Shk  = self.RNG.rand(self.pop_size)
        sNow[Shk < self.p_init] = 1
        self.sNow = sNow

    def preSolve(self):
        '''
        Updates the punk proportion evolution array by calling self.updateEvolution().

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        # This step is necessary in the general equilibrium framework, where a
        # new evolution rule is calculated after each simulation run, but only
        # the sufficient statistics describing it are sent back to agents.
        self.updateEvolution()

    def postSolve(self):
        '''
        Unpack the behavioral and value functions for more parsimonious access.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.switchFuncPunk = self.solution[0].switchFuncPunk
        self.switchFuncJock = self.solution[0].switchFuncJock
        self.VfuncPunk      = self.solution[0].VfuncPunk
        self.VfuncJock      = self.solution[0].VfuncJock

    def simOnePrd(self):
        '''
        Simulate one period of the fashion victom model for this type.  Each
        agent receives an idiosyncratic preference shock and chooses whether to
        change styles (using the optimal decision rule).

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        pNow    = self.pNow
        sPrev   = self.sNow
        J2Pprob = self.switchFuncJock(pNow)
        P2Jprob = self.switchFuncPunk(pNow)
        Shks    = self.RNG.rand(self.pop_size)
        J2P     = np.logical_and(sPrev == 0,Shks < J2Pprob)
        P2J     = np.logical_and(sPrev == 1,Shks < P2Jprob)
        sNow    = copy(sPrev)
        sNow[J2P] = 1
        sNow[P2J] = 0
        self.sNow = sNow

    def marketAction(self):
        '''
        The "market action" for a FashionType in the general equilibrium setting.
        Simulates a single period using self.simOnePrd().

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.simOnePrd()


def solveFashion(solution_next,DiscFac,conformUtilityFunc,punk_utility,jock_utility,switchcost_J2P,switchcost_P2J,pGrid,pEvolution,pref_shock_mag):
    '''
    Solves a single period of the fashion victim model.

    Parameters
    ----------
    solution_next: FashionSolution
        A representation of the solution to the subsequent period's problem.
    DiscFac: float
        The intertemporal discount factor.
    conformUtilityFunc: function
        Utility as a function of the proportion of the population who wears the
        same style as the agent.
    punk_utility: float
        Direct utility from wearing the punk style this period.
    jock_utility: float
        Direct utility from wearing the jock style this period.
    switchcost_J2P: float
        Utility cost of switching from jock to punk this period.
    switchcost_P2J: float
        Utility cost of switching from punk to jock this period.
    pGrid: np.array
        1D array of "proportion of punks" states spanning [0,1], representing
        the fraction of agents *currently* wearing punk style.
    pEvolution: np.array
        2D array representing the distribution of next period's "proportion of
        punks".  The pEvolution[i,:] contains equiprobable values of p for next
        period if p = pGrid[i] today.
    pref_shock_mag: float
        Standard deviation of T1EV preference shocks over style.

    Returns
    -------
    solution_now: FashionSolution
        A representation of the solution to this period's problem.
    '''
    # Unpack next period's solution
    VfuncPunkNext = solution_next.VfuncPunk
    VfuncJockNext = solution_next.VfuncJock

    # Calculate end-of-period expected value for each style at points on the pGrid
    EndOfPrdVpunk = DiscFac*np.mean(VfuncPunkNext(pEvolution),axis=1)
    EndOfPrdVjock = DiscFac*np.mean(VfuncJockNext(pEvolution),axis=1)

    # Get current period utility flow from each style (without switching cost)
    Upunk = punk_utility + conformUtilityFunc(pGrid)
    Ujock = jock_utility + conformUtilityFunc(1.0 - pGrid)

    # Calculate choice-conditional value for each combination of current and next styles (at each)
    V_J2J = Ujock                  + EndOfPrdVjock
    V_J2P = Upunk - switchcost_J2P + EndOfPrdVpunk
    V_P2J = Ujock - switchcost_P2J + EndOfPrdVjock
    V_P2P = Upunk                  + EndOfPrdVpunk

    # Calculate the beginning-of-period expected value of each p-state when punk
    Vboth_P = np.vstack((V_P2J,V_P2P))
    Vbest_P = np.max(Vboth_P,axis=0)
    Vnorm_P = Vboth_P - np.tile(np.reshape(Vbest_P,(1,pGrid.size)),(2,1))
    ExpVnorm_P = np.exp(Vnorm_P/pref_shock_mag)
    SumExpVnorm_P = np.sum(ExpVnorm_P,axis=0)
    V_P = np.log(SumExpVnorm_P)*pref_shock_mag + Vbest_P
    switch_P = ExpVnorm_P[0,:]/SumExpVnorm_P

    # Calculate the beginning-of-period expected value of each p-state when jock
    Vboth_J = np.vstack((V_J2J,V_J2P))
    Vbest_J = np.max(Vboth_J,axis=0)
    Vnorm_J = Vboth_J - np.tile(np.reshape(Vbest_J,(1,pGrid.size)),(2,1))
    ExpVnorm_J = np.exp(Vnorm_J/pref_shock_mag)
    SumExpVnorm_J = np.sum(ExpVnorm_J,axis=0)
    V_J = np.log(SumExpVnorm_J)*pref_shock_mag + Vbest_J
    switch_J = ExpVnorm_J[1,:]/SumExpVnorm_J

    # Make value and policy functions for each style
    VfuncPunkNow = LinearInterp(pGrid,V_P)
    VfuncJockNow = LinearInterp(pGrid,V_J)
    switchFuncPunkNow = LinearInterp(pGrid,switch_P)
    switchFuncJockNow = LinearInterp(pGrid,switch_J)

    # Make and return this period's solution
    solution_now = FashionSolution(VfuncJock=VfuncJockNow,
                                   VfuncPunk=VfuncPunkNow,
                                   switchFuncJock=switchFuncJockNow,
                                   switchFuncPunk=switchFuncPunkNow)
    return solution_now


def calcPunkProp(sNow):
    '''
    Calculates the proportion of punks in the population, given data from each type.

    Parameters
    ----------
    pNow : [np.array]
        List of arrays of binary data, representing the fashion choice of each
        agent in each type of this market (0=jock, 1=punk).
    pop_size : [int]
        List with the number of agents of each type in the market.  Unused.
    '''
    sNowX = np.asarray(sNow).flatten()
    pNow  = np.mean(sNowX)
    return FashionMarketInfo(pNow)


def calcFashionEvoFunc(pNow):
    '''
    Calculates a new approximate dynamic rule for the evolution of the proportion
    of punks as a linear function and a "shock width".

    Parameters
    ----------
    pNow : [float]
        List describing the history of the proportion of punks in the population.

    Returns
    -------
    (unnamed) : FashionEvoFunc
        A new rule for the evolution of the population punk proportion, based on
        the history in input pNow.
    '''
    pNowX = np.array(pNow)
    T = pNowX.size
    p_t   = pNowX[100:(T-1)]
    p_tp1 = pNowX[101:T]
    pNextSlope, pNextIntercept, trash1, trash2, trash3 = stats.linregress(p_t,p_tp1)
    pPopExp  = pNextIntercept + pNextSlope*p_t
    pPopErrSq= (pPopExp - p_tp1)**2
    pNextStd  = np.sqrt(np.mean(pPopErrSq))
    print(str(pNextIntercept) + ', ' + str(pNextSlope) + ', ' + str(pNextStd))
    return FashionEvoFunc(pNextIntercept,pNextSlope,2*pNextStd)


###############################################################################
###############################################################################
if __name__ == '__main__':
    from time import clock
    from HARK.core import Market
    mystr = lambda number : "{:.4f}".format(number)
    import matplotlib.pyplot as plt
    from copy import deepcopy

    do_many_types = True

    # Make a test case and solve the micro model
    TestType = FashionVictimType(**Params.default_params)
    print('Utility function:')
    plotFuncs(TestType.conformUtilityFunc,0,1)

    t_start = clock()
    TestType.solve()
    t_end = clock()
    print('Solving a fashion victim micro model took ' + mystr(t_end-t_start) + ' seconds.')

    print('Jock value function:')
    plotFuncs(TestType.VfuncJock,0,1)
    print('Punk value function:')
    plotFuncs(TestType.VfuncPunk,0,1)
    print('Jock switch probability:')
    plotFuncs(TestType.switchFuncJock,0,1)
    print('Punk switch probability:')
    plotFuncs(TestType.switchFuncPunk,0,1)

    # Make a list of different types
    AltType = deepcopy(TestType)
    AltType(uParamA = Params.uParamB, uParamB = Params.uParamA, seed=20)
    AltType.update()
    AltType.solve()
    type_list = [TestType,AltType]
    u_vec = np.linspace(0.02,0.1,5)
    if do_many_types:
        for j in range(u_vec.size):
            ThisType = deepcopy(TestType)
            ThisType(punk_utility=u_vec[j])
            ThisType.solve()
            type_list.append(ThisType)
            ThisType = deepcopy(AltType)
            ThisType(punk_utility=u_vec[j])
            ThisType.solve()
            type_list.append(ThisType)
        for j in range(u_vec.size):
            ThisType = deepcopy(TestType)
            ThisType(jock_utility=u_vec[j])
            ThisType.solve()
            type_list.append(ThisType)
            ThisType = deepcopy(AltType)
            ThisType(jock_utility=u_vec[j])
            ThisType.solve()
            type_list.append(ThisType)

    # Now run the simulation inside a Market
    TestMarket = Market(agents        = type_list,
                        sow_vars      = ['pNow'],
                        reap_vars     = ['sNow'],
                        track_vars    = ['pNow'],
                        dyn_vars      = ['pNextIntercept','pNextSlope','pNextWidth'],
                        millRule      = calcPunkProp,
                        calcDynamics  = calcFashionEvoFunc,
                        act_T         = 1000,
                        tolerance     = 0.01)
    TestMarket.pNow_init = 0.5

    TestMarket.solve()
    plt.plot(TestMarket.pNow_hist)
    plt.show()
