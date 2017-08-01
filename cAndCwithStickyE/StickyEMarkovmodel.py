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
from ConsMarkovModel import MarkovSOEType

# Specify which subclass of AgentType the model should modify
BaseAgentType = MarkovSOEType

# Make an extension of the base type for the SOE
class StickyEMarkovSOEType(BaseAgentType):
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
        if hasattr(self,'MrkvNow'):
            self.MrkvNow[which_agents] = self.MktMrkvNow
        else:
            self.MrkvNow = np.ones(self.AgentCount)*self.MktMrkvNow
            
    def getShocks(self):
        '''
        Gets permanent and transitory shocks , but
        only consumers who update their macroeconomic beliefs this period notice 
        the change in the markov state.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        MrkvPrev = self.MrkvNow
        BaseAgentType.getShocks(self) # Get permanent and transitory combined shocks
        
        # Randomly draw which agents will update their beliefs 
        how_many_update = int(round(self.UpdatePrb*self.AgentCount))
        base_bool = np.zeros(self.AgentCount,dtype=bool)
        base_bool[0:how_many_update] = True
        update = self.RNG.permutation(base_bool)
        dont = np.logical_not(update)
        
        # For agents who don't update, the Markov state remains as before
        self.MrkvNow[dont] = MrkvPrev[dont]
        perm_growth_fac_belief = self.PermGroFac[0][self.MrkvNow]*(1.0 + (self.PermShkAggNow-1.0)*update)
        perm_growth_fac_actual = self.PermGroFac[0][self.MktMrkvNow]*self.PermShkAggNow
        # Non-updaters misperception of their productivity gets worse, but updaters incorporate all the news they've missed
        self.PermShkNow = self.PermShkNow*perm_growth_fac_belief/perm_growth_fac_actual
        self.pLvlErrNow = self.pLvlErrNow*perm_growth_fac_belief/perm_growth_fac_actual
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
        
        yLvlTrueNow = self.pLvlNow/self.pLvlErrNow*self.TranShkNow
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
        BaseAgentType.getPostStates(self)
        self.aLvlNow = self.mLvlTrueNow - self.cNrmNow*self.pLvlNow
#        self.aLvlNow = np.maximum(self.mLvlTrueNow - self.cNrmNow*self.pLvlNow,0.0) #Fix so that savings are never negative. This should really be fixed in the output for consumption too...
        
