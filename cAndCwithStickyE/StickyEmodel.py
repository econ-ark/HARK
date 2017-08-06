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
from ConsAggShockModel import AggShockConsumerType, AggShockMarkovConsumerType
from RepAgentModel import RepAgentConsumerType

# Make an extension of the base type for the heterogeneous agents versions
class StickyEconsumerType(AggShockConsumerType):
    '''
    A class for representing consumers who have sticky expectations about the macroeconomy
    because they does not observe aggregate variables every period.
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
        self.PlvlAggNow *= self.PermShkAggNow # Updated aggregate permanent productivity level
        
        # Calculate what the consumers perceive their normalized market resources to be
        RfreeNow = self.getRfree()
        bLvlNow = RfreeNow*self.aLvlNow # This is the true level
        
        yLvlTrueNow = self.pLvlNow*self.pLvlErrNow*self.TranShkNow
        mLvlTrueNow = bLvlNow + yLvlTrueNow
        mNrmPcvdNow = mLvlTrueNow/self.pLvlNow
        self.mNrmNow = mNrmPcvdNow
        self.mLvlTrueNow = mLvlTrueNow
        
    def getMaggNow(self): # Agents know the true level of aggregate market resources, but
        MaggPcvdNow = self.MaggNow*self.pLvlErrNow # have erroneous perception of pLvlAgg.
        return MaggPcvdNow
        
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
        
        
class StickyEmarkovConsumerType(AggShockMarkovConsumerType,StickyEconsumerType):
    '''
    A class for representing consumers who have sticky expectations about the macroeconomy
    because they does not observe aggregate variables every period.  This version lives
    in an economy subject to Markov shocks to the aggregate income process.
    '''
    def simBirth(self,which_agents):
        StickyEconsumerType.simBirth(self,which_agents)
        
    def getStates(self):
        StickyEconsumerType.getStates(self)
        
    def getPostStates(self):
        StickyEconsumerType.getPostStates(self)
        
    def getMaggNow(self):
        return StickyEconsumerType.getMaggNow(self)
        
    def getMrkvNow(self):
        return self.MrkvNowPcvd
    
    def getShocks(self):
        '''
        Gets permanent and transitory shocks (combining idiosyncratic and aggregate shocks), but
        only consumers who update their macroeconomic beliefs this period notice the aggregate
        transitory shocks and incorporate all previously unnoticed aggregate permanent shocks.
        Also handles perceptions of the macroeconomic Markov state.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        AggShockMarkovConsumerType.getShocks(self) # Get permanent and transitory combined shocks
        
        # Randomly draw which agents will update their beliefs 
        how_many_update = int(round(self.UpdatePrb*self.AgentCount))
        base_bool = np.zeros(self.AgentCount,dtype=bool)
        base_bool[0:how_many_update] = True
        update = self.RNG.permutation(base_bool)
        dont = np.logical_not(update)
        
        # Only updaters change their perception of the Markov state
        if hasattr(self,'MrkvNowPcvd'):
            self.MrkvNowPcvd[update] = self.MrkvNow
        else: # This only triggers in the first simulated period
            self.MrkvNowPcvd = np.ones(self.AgentCount,dtype=int)*self.MrkvNow
            
        # Calculate innovation to the productivity level perception error
        pLvlErrNew = np.ones(self.AgentCount)
        pLvlErrNew[dont] = self.PermShkAggNow/self.PermGroFacAgg[self.MrkvNowPcvd[dont]]
        self.pLvlErrNow *= pLvlErrNew # Perception error accumulation
        
        # Calculate perceptions of the permanent shock
        PermShkPcvd = self.PermShkNow/pLvlErrNew
        PermShkPcvd[update] *= self.pLvlErrNow[update] # Updaters see the true permanent shock and all missed news        
        self.pLvlErrNow[update] = 1.0
        self.PermShkNow = PermShkPcvd
        

      
class StickyErepAgent(RepAgentConsumerType):
    '''
    A representative consumer who has sticky expectations about the macroeconomy because
    he does not observe aggregate variables every period.  Agent lives in a Cobb-Douglas economy.
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
        self.pLvlTrue = np.ones(self.AgentCount)
        self.aLvlNow = self.aNrmNow*self.pLvlTrue
            
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
        RepAgentConsumerType.getShocks(self) # Get actual permanent and transitory shocks
        
        # Handle the perceived vs actual transitory shock
        TranShkPcvd = self.UpdatePrb*self.TranShkNow + (1.0-self.UpdatePrb)*1.0
        self.TranShkTrue = self.TranShkNow
        self.TranShkNow = TranShkPcvd
        
         
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
        # Calculate perceived and true productivity level
        pLvlPrev = self.pLvlNow
        aLvlPrev = self.aLvlNow
        pLvlTrue = self.pLvlTrue*self.PermShkNow
        pLvlPcvd = self.UpdatePrb*pLvlTrue + (1.0-self.UpdatePrb)*(pLvlPrev*self.getExPermShk())
        self.pLvlNow = pLvlPcvd
        self.pLvlTrue = pLvlTrue
        
        # Calculate perceptions of normalized variables
        self.kNrmNow = aLvlPrev/pLvlPcvd
        self.yNrmNow = self.kNrmNow**self.CapShare*self.TranShkNow**(1.-self.CapShare) - self.kNrmNow*self.DeprFac
        self.Rfree = 1. + self.CapShare*self.kNrmNow**(self.CapShare-1.)*self.TranShkNow**(1.-self.CapShare) - self.DeprFac
        self.wRte  = (1.-self.CapShare)*self.kNrmNow**self.CapShare*self.TranShkNow**(-self.CapShare)
        self.mNrmNow = self.Rfree*self.kNrmNow + self.wRte*self.TranShkNow
        
        # Calculate true values of normalized variables
        self.kNrmTrue = aLvlPrev/pLvlTrue
        self.yNrmTrue = self.kNrmTrue**self.CapShare*self.TranShkTrue**(1.-self.CapShare) - self.kNrmTrue*self.DeprFac
        self.Rfree = 1. + self.CapShare*self.kNrmTrue**(self.CapShare-1.)*self.TranShkTrue**(1.-self.CapShare) - self.DeprFac
        self.wRte  = (1.-self.CapShare)*self.kNrmTrue**self.CapShare*self.TranShkTrue**(-self.CapShare)
        self.mNrmTrue = self.Rfree*self.kNrmTrue + self.wRte*self.TranShkTrue
        self.mLvlTrue = self.mNrmTrue*pLvlTrue
        
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
        self.cLvlNow = self.cNrmNow*self.pLvlNow # This is true
        self.aLvlNow = self.mLvlTrue - self.cLvlNow # This is true
        self.aNrmNow = self.aLvlNow/self.pLvlTrue # This is true
        
    def getExPermShk(self):
        '''
        Returns the expected permanent income shock, including permanent growth factor.
        In this model, that is simply the permanent growth factor for this period.
        '''
        t = self.t_cycle[0]
        return self.PermGroFac[t-1]
        