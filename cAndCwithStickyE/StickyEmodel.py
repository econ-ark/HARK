'''
A first attempt at the models described in cAndCwithStickyE.
'''

# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. Also import ConsumptionSavingModel
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
from copy import copy, deepcopy
from HARKsimulation import drawUniform
from ConsAggShockModel import AggShockConsumerType
from RepAgentModel import RepAgentConsumerType

# Make an extension of the base type for the SOE
class StickyEconsumerSOEType(AggShockConsumerType):
    '''
    A consumer who has sticky expectations about the macroeconomy because he does not observe
    aggregate variables every period; this version is for a small open economy.
    ''' 
    def simBirth(self,which_agents):
        '''
        Makes new consumers for the given indices.  Slightly extends base method by also setting
        pLvlErrNow = 1.0 for new agents, indicating that they correctly perceive their productivity.
        
        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".
        
        Returns
        -------
        None
        '''
        AggShockConsumerType.simBirth(self,which_agents)
        if hasattr(self,'pLvlErrNow'):
            self.pLvlErrNow[which_agents] = 1.0
        else:
            self.pLvlErrNow = np.ones(self.AgentCount)
            
    def getShocks(self):
        '''
        Gets permanent and transitory shocks (combining idiosyncratic and aggregate shocks), but
        only consumers who update their macroeconomic beliefs this period notice the aggregate
        transitory shocks and incorporate all previously unnoticed aggregate permanent shocks.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        AggShockConsumerType.getShocks(self) # Get permanent and transitory combined shocks
        
        # Randomly draw which agents will update their beliefs 
        how_many_update = int(round(self.UpdatePrb*self.AgentCount))
        base_bool = np.zeros(self.AgentCount,dtype=bool)
        base_bool[0:how_many_update] = True
        update = self.RNG.permutation(base_bool)
        
        # Non-updaters misperception of their productivity gets worse, but updaters incorporate all the news they've missed
        pLvlErrNew = self.PermShkAggNow/self.PermGroFacAgg # new missed news
        self.PermShkNow = self.PermShkNow/pLvlErrNew
        self.pLvlErrNow = self.pLvlErrNow*pLvlErrNew # pLvlErrNow accumulates all of the missed news
        self.PermShkNow[update] = self.PermShkNow[update]*self.pLvlErrNow[update]
        self.pLvlErrNow[update] = 1.0
        
    def getStates(self):
        '''
        Gets simulated consumers pLvl and mNrm for this period, but with the alteration that these
        represent perceived rather than actual values.  Also calculates mLvlTrue, the true level of
        market resources that the individual has on hand.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        # Update consumers' perception of their permanent income level
        pLvlPrev = self.pLvlNow
        self.pLvlNow = pLvlPrev*self.PermShkNow # Updated permanent income level (only correct if macro state is observed this period)
        
        # Calculate what the consumers perceive their normalized market resources to be
        RfreeNow = self.getRfree()
        bLvlNow = RfreeNow*self.aLvlNow # This is the true level
        
        yLvlTrueNow = self.pLvlNow*self.pLvlErrNow*self.TranShkNow
        mLvlTrueNow = bLvlNow + yLvlTrueNow
        mNrmPcvdNow = mLvlTrueNow/self.pLvlNow
        self.mNrmNow = mNrmPcvdNow
        self.mLvlTrueNow = mLvlTrueNow
        
    def getPostStates(self):
        '''
        Slightly extends the base version of this method by recalculating aLvlNow to account for the
        consumer's (potential) misperception about their productivity level.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        AggShockConsumerType.getPostStates(self)
        self.cLvlNow = self.cNrmNow*self.pLvlNow
        self.aLvlNow = self.mLvlTrueNow - self.cLvlNow
        self.aNrmNow = self.aLvlNow/self.pLvlNow # This is perceived

      
class StickyEconsumerDSGEType(RepAgentConsumerType):
    '''
    A consumer who has sticky expectations about the macroeconomy because he does not observe
    aggregate variables every period; this version is for a Cobb Douglas economy with a rep agent.
    '''
    def simBirth(self,which_agents):
        '''
        Makes new consumers for the given indices.  Slightly extends base method by also setting
        pLvlErrNow = 1.0 for new agents, indicating that they correctly perceive their productivity.
        
        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".
        
        Returns
        -------
        None
        '''
        RepAgentConsumerType.simBirth(self,which_agents)
        if hasattr(self,'pLvlErrNow'):
            self.pLvlErrNow[which_agents] = 1.0
        else:
            self.pLvlErrNow = np.ones(self.AgentCount)
            
    def getShocks(self):
        '''
        Gets permanent and transitory shocks (combining idiosyncratic and aggregate shocks), but
        only consumers who update their macroeconomic beliefs this period notice the aggregate
        transitory shocks and incorporate all previously unnoticed aggregate permanent shocks.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        RepAgentConsumerType.getShocks(self) # Get permanent and transitory shocks
        
        # Randomly choose whether rep agent updates this period
        draw = drawUniform(N=1, seed=self.RNG.randint(0,2**31-1))
        if draw < self.UpdatePrb: # Updaters get all information they've missed
            self.PermShkNow = self.PermShkNow*self.pLvlErrNow
            self.pLvlErrNow[:] = 1.0
        else: # Non-updaters misperception of their productivity gets worse
            pLvlErrNew = self.PermShkNow/self.PermGroFac[0]
            self.PermShkNow[:] = self.PermGroFac[0]
            self.pLvlErrNow = self.pLvlErrNow*pLvlErrNew
    
        
    def getStates(self):
        '''
        Calculates updated values of normalized market resources and permanent income level.
        Uses pLvlNow, aNrmNow, PermShkNow, TranShkNow.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        RepAgentConsumerType.getStates(self) # Calculate perceived normalized market resources
        
        # Calculate actual market resource level
        pLvlPrev = self.pLvlNow/self.PermShkNow
        aNrmPrev = self.aNrmNow
        pLvlTrue = pLvlPrev*(self.PermShkNow*self.pLvlErrNow) # Updated permanent income level
        self.kNrmNow = aNrmPrev/(self.PermShkNow*self.pLvlErrNow)
        self.yNrmNow = self.kNrmNow**self.CapShare*self.TranShkNow**(1.-self.CapShare)
        self.Rfree = 1. + self.CapShare*self.kNrmNow**(self.CapShare-1.)*self.TranShkNow**(1.-self.CapShare) - self.DeprFac
        self.wRte  = (1.-self.CapShare)*self.kNrmNow**self.CapShare*self.TranShkNow**(-self.CapShare)
        mNrmNowTrue = self.Rfree*self.kNrmNow + self.wRte*self.TranShkNow
        self.mLvlTrueNow = mNrmNowTrue*pLvlTrue
        self.yLvlNow = self.yNrmNow*pLvlTrue
        
    def getPostStates(self):
        '''
        Slightly extends the base version of this method by recalculating aLvlNow to account for the
        consumer's (potential) misperception about their productivity level.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        RepAgentConsumerType.getPostStates(self)
        self.cLvlNow = self.cNrmNow*self.pLvlNow
        self.aLvlNow = self.mLvlTrueNow - self.cLvlNow
        self.aNrmNow = self.aLvlNow/self.pLvlNow # This is perceived
        