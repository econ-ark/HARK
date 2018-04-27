'''
Classes to wrap HANK code from Ralph Luetticke's students Seungmoon Park and
Seungcheol Lee.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('./Luetticke_code/'))
from HARKcore import AgentType, Market
from SteadyStateOneAssetIOUs import SteadyStateOneAssetIOU
from FluctuationsOneAssetIOUs import FluctuationsOneAssetIOUs, SGU_solver
from copy import copy, deepcopy
import numpy as np

class LuettickeAgent(AgentType):
    '''
    An agent who lives in a LuettickeMarket. This agent has no solve method,
    but instead inherits it from the Luetticke code.
    '''
    poststate_vars_ = ['aNow','incStateNow']
    
    def __init__(self,AgentCount):
        '''
        Instantiate a new LuettickeType with solution from Luetticke_code.

        Parameters
        ----------
        AgentCount : int
            Number of agents of this type for simulation
        Returns
        -------
        None
        '''       
        self.AgentCount = AgentCount
        
        self.poststate_vars = deepcopy(self.poststate_vars_)
        self.track_vars     = []
        
    def simBirth(self,which_agents):
        '''
        Agents do not die in this model, so birth only happens at time 0.
        Agents get given levels of labor income and assets according to the
        steady state distribution
        
        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".
            Note in this model birth only happens once at time zero, for all agents
        
        Returns
        -------
        None
        '''
        # Get and store states for newly born agents
        N = np.sum(which_agents) # Number of new consumers to make
        
        
        self.aNrmNow[which_agents] = drawLognormal(N,mu=self.aNrmInitMean,sigma=self.aNrmInitStd,seed=self.RNG.randint(0,2**31-1))
        pLvlInitMeanNow = self.pLvlInitMean + np.log(self.PlvlAggNow) # Account for newer cohorts having higher permanent income
        self.pLvlNow[which_agents] = drawLognormal(N,mu=pLvlInitMeanNow,sigma=self.pLvlInitStd,seed=self.RNG.randint(0,2**31-1))
        self.t_age[which_agents]   = 0 # How many periods since each agent was born
        self.t_cycle[which_agents] = 0 # Which period of the cycle each agent is currently in
        return None
    
    def getEconomyData(self,Economy):
        '''
        Imports economy-determined objects into self from a Market.
        In this case the full market solution is imported
        
        Parameters
        ----------
        Economy : LuettickeEconomy
            The "macroeconomy" in which this instance "lives".             
        Returns
        -------
        None
        '''
        self.SR = Economy.SR
        self.SGUresult = Economy.SGUresult

class LuettickeEconomy(Market):
    '''
    A class to wrap the solution of Luetticke's code for a simple HANK model.
    '''
    def __init__(self,FluctuationsOneAssetIOU,agents=[],act_T=1000):
        '''
        Make a new instance of LuettickeEconomy by filling in attributes
        specific to this kind of market.
        
        Parameters
        ----------
        FluctuationsOneAssetIOU : FluctuationsOneAssetIOUs
            Class from Luetticke_code that solves the model
        agents : [ConsumerType]
            List of types of consumers that live in this economy.
        act_T : int
            Number of periods to simulate when making a history of of the market.
            
        Returns
        -------
        None
        '''
        Market.__init__(self,agents=agents,
                            sow_vars=['XNow'],
                            reap_vars=[],
                            track_vars=[],
                            dyn_vars=[],
                            tolerance=1e-10,
                            act_T=act_T)
        self.FluctuationsOneAssetIOU = deepcopy(FluctuationsOneAssetIOU)
    
    def solve(self):
        '''
        Sovles the model using Luetticke's code
        '''
        # First do state reduction
        SR = self.FluctuationsOneAssetIOU.StateReduc()
        # Now solve the model
        SGUresult=SGU_solver(SR['Xss'],SR['Yss'],SR['Gamma_state'],SR['Gamma_control'],SR['InvGamma'],SR['Copula'],
                             SR['par'],SR['mpar'],SR['grid'],SR['targets'],SR['P_H'],SR['aggrshock'],SR['oc'])
        self.SR = SR
        self.SGUresult = SGUresult
        
        for agent in self.agents:
            agent.getEconomyData(self)

###############################################################################

if __name__ == '__main__':
    import defineSSParameters as Params
    from copy import copy
    import pickle
    
    solve_ss = False

    #First calculate the steady state
    if solve_ss:
        EX1param = copy(Params.parm_one_asset_IOU)
        EX1 = SteadyStateOneAssetIOU(**EX1param)
        EX1SS = EX1.SolveSteadyState()
    else:
        EX1SS=pickle.load(open("Luetticke_code/EX1SS.p", "rb"))
    #Build Luetticke's object
    FluctuationsOneAssetIOU=FluctuationsOneAssetIOUs(**EX1SS)
        
    #Create agent object
    LuettickeExampleAgent = LuettickeAgent(AgentCount=10000)
    #Create Market object
    LuettickeExampleEconomy = LuettickeEconomy(FluctuationsOneAssetIOU,agents=[LuettickeExampleAgent])
    #Solve the market
    LuettickeExampleEconomy.solve()
    
    #Simulate
    
