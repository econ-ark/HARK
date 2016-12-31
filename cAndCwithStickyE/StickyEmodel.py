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
from HARKcore import Market
from ConsAggShockModel import CobbDouglasEconomy, AggShockConsumerType

# Specify which subclass of AgentType the model should modify
BaseAgentType = AggShockConsumerType

# Make an extension of the base type for the SOE
class StickyEconsumerSOEType(BaseAgentType):
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
        BaseAgentType.simBirth(self,which_agents)
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
        BaseAgentType.getShocks(self) # Get permanent and transitory combined shocks
        
        # Randomly draw which agents will update their beliefs 
        how_many_update = int(round(self.UpdatePrb*self.AgentCount))
        base_bool = np.zeros(self.AgentCount,dtype=bool)
        base_bool[0:how_many_update] = True
        update = self.RNG.permutation(base_bool)
        dont = np.logical_not(update)
        
        # For agents who don't update, the aggregate transitory shocks aren't observed
        self.TranShkNow[dont] = self.TranShkNow[dont]/self.TranShkAggNow
        
        # Non-updaters misperception of their productivity gets worse, but updaters incorporate all the news they've missed
        self.pLvlErrNow = self.pLvlErrNow/self.PermShkAggNow
        self.PermShkNow[update] = self.PermShkNow[update]/self.pLvlErrNow[update]
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
        yLvlPcvdNow = self.pLvlNow*self.TranShkNow # This is only correct if individual updated this period
        mLvlPcvdNow = bLvlNow + yLvlPcvdNow # Consumers' perception of their 
        mNrmPcvdNow = mLvlPcvdNow/self.pLvlNow
        self.mNrmNow = mNrmPcvdNow
        
        # And calculate consumers' true level of market resources
        yLvlTrueNow = yLvlPcvdNow/self.pLvlErrNow # This is same as Pcvd if we updated, as pLvlErrNow = 1.0
        mLvlTrueNow = bLvlNow + yLvlTrueNow
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
        BaseAgentType.getPostStates(self)
        self.aLvlNow = self.mLvlTrueNow - self.cNrmNow*self.pLvlNow
        
class StickyEconsumerDSGEType(StickyEconsumerSOEType):
    '''
    A consumer who has sticky expectations about the macroeconomy because he does not observe
    aggregate variables every period; this version is for a Cobb Douglas economy with a rep agent.
    '''
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
        # Calculate what the consumers perceive their normalized market resources to be
        KtoLnowPcvd = self.KtoLnow/self.pLvlErrNow
        yNrmPcvdNow = KtoLnowPcvd**self.CapShare*self.TranShkNow**(1.0-self.CapShare) # This is only correct if individual updated this period
        mNrmPcvdNow = KtoLnowPcvd + yNrmPcvdNow # Consumers' perception of their 
        self.mNrmNow = mNrmPcvdNow
        
        # And calculate consumers' true level of market resources
        yNrmTrueNow = self.KtoLnow**self.CapShare*self.TranShkAggNow**(1.0-self.CapShare) # This is same as Pcvd if we updated
        mLvlTrueNow = self.KtoLnow + yNrmTrueNow
        self.mLvlTrueNow = mLvlTrueNow
        