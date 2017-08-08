'''
A first attempt at the models described in cAndCwithStickyE.
'''
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
from ConsAggShockModel import AggShockConsumerType, AggShockMarkovConsumerType
from RepAgentModel import RepAgentConsumerType, RepAgentMarkovConsumerType

# Make an extension of the base type for the heterogeneous agents versions
class StickyEconsumerType(AggShockConsumerType):
    '''
    A class for representing consumers who have sticky expectations about the macroeconomy
    because they do not observe aggregate variables every period.
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
            
    def getUpdaters(self):
        '''
        Determine which agents update this period vs which don't.  Fills in the
        attributes updaters and dont as boolean arrays of size AgentCount.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        how_many_update = int(round(self.UpdatePrb*self.AgentCount))
        base_bool = np.zeros(self.AgentCount,dtype=bool)
        base_bool[0:how_many_update] = True
        self.update = self.RNG.permutation(base_bool)
        self.dont = np.logical_not(self.update)

        
    def getpLvlError(self):
        '''
        Calculates and returns the misperception of this period's shocks.  Updaters
        have no misperception this period, while those who don't update don't see
        the value of the aggregate permanent shock and assume it is 1.
        
        
        Parameters
        ----------
        None
        
        Returns
        -------
        pLvlErr : np.array
            Array of size AgentCount with this period's (new) misperception.
        '''
        pLvlErr = np.ones(self.AgentCount)
        pLvlErr[self.dont] = self.PermShkAggNow/self.PermGroFacAgg
        return pLvlErr
        
            
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
        # The strange syntax here is so that both StickyEconsumerType and StickyEmarkovConsumerType
        # run the getShocks method of their first superclass: AggShockConsumerType and
        # AggShockMarkovConsumerType respectively.  This will be simplified in Python 3.
        super(self.__class__,self).getShocks() # Get permanent and transitory combined shocks
        self.getUpdaters() # Randomly draw which agents will update their beliefs 
        
        # Calculate innovation to the productivity level perception error
        pLvlErrNew = self.getpLvlError()
        self.pLvlErrNow *= pLvlErrNew # Perception error accumulation
        
        # Calculate (mis)perceptions of the permanent shock
        PermShkPcvd = self.PermShkNow/pLvlErrNew
        PermShkPcvd[self.update] *= self.pLvlErrNow[self.update] # Updaters see the true permanent shock and all missed news        
        self.pLvlErrNow[self.update] = 1.0
        self.PermShkNow = PermShkPcvd
        
        
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
        self.pLvlNow = pLvlPrev*self.PermShkNow # Perceived permanent income level (only correct if macro state is observed this period)
        self.PlvlAggNow *= self.PermShkAggNow # Updated aggregate permanent productivity level
        self.pLvlTrue = self.pLvlNow*self.pLvlErrNow
        
        # Calculate what the consumers perceive their normalized market resources to be
        RfreeNow = self.getRfree()
        bLvlNow = RfreeNow*self.aLvlNow # This is the true level
        
        yLvlNow = self.pLvlTrue*self.TranShkNow # This is true income level
        mLvlTrueNow = bLvlNow + yLvlNow # This is true market resource level
        mNrmPcvdNow = mLvlTrueNow/self.pLvlNow
        self.mNrmNow = mNrmPcvdNow
        self.mLvlTrueNow = mLvlTrueNow
        self.yLvlNow = mLvlTrueNow - self.aLvlNow # Includes capital and labor income 
        
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
        
    def getShocks(self):
        StickyEconsumerType.getShocks(self)
        
    def getStates(self):
        StickyEconsumerType.getStates(self)
        
    def getPostStates(self):
        StickyEconsumerType.getPostStates(self)
        
    def getMaggNow(self):
        return StickyEconsumerType.getMaggNow(self)
        
    def getMrkvNow(self):
        return self.MrkvNowPcvd
    
    def getUpdaters(self):
        '''
        Determine which agents update this period vs which don't.  Fills in the
        attributes updaters and dont as boolean arrays of size AgentCount.  This
        version also updates perceptions of the Markov state.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        StickyEconsumerType.getUpdaters(self)
        # Only updaters change their perception of the Markov state
        if hasattr(self,'MrkvNowPcvd'):
            self.MrkvNowPcvd[self.update] = self.MrkvNow
        else: # This only triggers in the first simulated period
            self.MrkvNowPcvd = np.ones(self.AgentCount,dtype=int)*self.MrkvNow
       
    def getpLvlError(self):
        '''
        Calculates and returns the misperception of this period's shocks.  Updaters
        have no misperception this period, while those who don't update don't see
        the value of the aggregate permanent shock and assume it is 1.  Moreover,
        non-updaters base their belief about aggregate growth on the last Markov
        state that they actually observed.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        pLvlErr : np.array
            Array of size AgentCount with this period's (new) misperception.
        '''
        pLvlErr = np.ones(self.AgentCount)
        pLvlErr[self.dont] = self.PermShkAggNow/self.PermGroFacAgg[self.MrkvNowPcvd[self.dont]]
        return pLvlErr
    


class StickyErepAgent(RepAgentConsumerType):
    '''
    A representative consumer who has sticky expectations about the macroeconomy because
    he does not observe aggregate variables every period.  Agent lives in a Cobb-Douglas economy.
    '''
    def simBirth(self,which_agents):
        '''
        Makes new consumers for the given indices.  Slightly extends base method by also setting
        pLvlTrue = 1.0 for new agents, indicating that they correctly perceive their productivity.
        
        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".
        
        Returns
        -------
        None
        '''
        super(self.__class__,self).simBirth(which_agents)
        if self.t_sim == 0:
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
        super(self.__class__,self).getShocks() # Get actual permanent and transitory shocks
        
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
        aLvlPrev = self.aLvlNow
        self.pLvlTrue = self.pLvlTrue*self.PermShkNow
        self.pLvlNow = self.getpLvlPcvd()
        
        # Calculate perceptions of normalized variables
        self.kNrmNow = aLvlPrev/self.pLvlNow
        self.yNrmNow = self.kNrmNow**self.CapShare*self.TranShkNow**(1.-self.CapShare) - self.kNrmNow*self.DeprFac
        self.Rfree = 1. + self.CapShare*self.kNrmNow**(self.CapShare-1.)*self.TranShkNow**(1.-self.CapShare) - self.DeprFac
        self.wRte  = (1.-self.CapShare)*self.kNrmNow**self.CapShare*self.TranShkNow**(-self.CapShare)
        self.mNrmNow = self.Rfree*self.kNrmNow + self.wRte*self.TranShkNow
        
        # Calculate true values of normalized variables
        self.kNrmTrue = aLvlPrev/self.pLvlTrue
        self.yNrmTrue = self.kNrmTrue**self.CapShare*self.TranShkTrue**(1.-self.CapShare) - self.kNrmTrue*self.DeprFac
        self.Rfree = 1. + self.CapShare*self.kNrmTrue**(self.CapShare-1.)*self.TranShkTrue**(1.-self.CapShare) - self.DeprFac
        self.wRte  = (1.-self.CapShare)*self.kNrmTrue**self.CapShare*self.TranShkTrue**(-self.CapShare)
        self.mNrmTrue = self.Rfree*self.kNrmTrue + self.wRte*self.TranShkTrue
        self.mLvlTrue = self.mNrmTrue*self.pLvlTrue
        
    def getControls(self):
        super(self.__class__,self).getControls()
        self.cLvlNow = self.cNrmNow*self.pLvlNow # This is true
        
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
        self.aLvlNow = self.mLvlTrue - self.cLvlNow # This is true
        self.aNrmNow = self.aLvlNow/self.pLvlTrue # This is true
        
    def getpLvlPcvd(self):
        '''
        Finds the representative agent's (average) perceived productivity level.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        pLvlPcvd : np.array
            Size 1 array with average perception of productivity level.
        '''
        pLvlPcvd = self.UpdatePrb*self.pLvlTrue + (1.0-self.UpdatePrb)*(self.pLvlNow*self.PermGroFac[self.t_cycle[0]-1])
        return pLvlPcvd
    
    
class StickyEmarkovRepAgent(RepAgentMarkovConsumerType,StickyErepAgent):
    '''
    A representative consumer who has sticky expectations about the macroeconomy because
    he does not observe aggregate variables every period.  Agent lives in a Cobb-Douglas
    economy that has a discrete Markov state.
    '''
    def simBirth(self,which_agents):
        RepAgentMarkovConsumerType.simBirth(self,which_agents)
        if self.t_sim == 0: # Initialize perception distribution for Markov state
            self.pLvlTrue = np.ones(self.AgentCount)
            self.aLvlNow = self.aNrmNow*self.pLvlTrue
            StateCount = self.MrkvArray.shape[0]
            self.pLvlNow = np.ones(StateCount)
            self.MrkvPcvd = np.zeros(StateCount)
            self.MrkvPcvd[self.MrkvNow[0]] = 1.0
        
    def getShocks(self):
        StickyErepAgent.getShocks(self)
        
    def getStates(self):
        StickyErepAgent.getStates(self)
        
    def getPostStates(self):
        StickyErepAgent.getPostStates(self)
        
    def getpLvlPcvd(self):
        '''
        Finds the representative agent's (average) perceived productivity level
        for each Markov state, as well as the distribution of the representative
        agent's perception of the Markov state.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        pLvlPcvd : np.array
            Array with average perception of productivity level by Markov state.
        '''
        StateCount = self.MrkvArray.shape[0]
        t = self.t_cycle[0]
        i = self.MrkvNow[0]
        
        dont_mass = self.MrkvPcvd*(1.-self.UpdatePrb) # pmf of non-updaters
        update_mass = np.zeros(StateCount)
        update_mass[i] = self.UpdatePrb # pmf of updaters
        
        dont_pLvlPcvd = self.pLvlNow*self.PermGroFac[t-1] # those that don't update think pLvl grows at PermGroFac for last observed state
        update_pLvlPcvd = np.zeros(StateCount)
        update_pLvlPcvd[i] = self.pLvlTrue # those that update see the true pLvl
        
        # Combine updaters and non-updaters to get average pLvl perception by Markov state
        self.MrkvPcvd = dont_mass + update_mass
        pLvlPcvd = (dont_mass*dont_pLvlPcvd + update_mass*update_pLvlPcvd)/self.MrkvPcvd
        pLvlPcvd[self.MrkvPcvd==0.] = 1.0 # fix division by zero problem when MrkvPcvd[i]=0
        return pLvlPcvd
        
    def getControls(self):
        '''
        Calculates consumption for the representative agent using the consumption functions.
        Takes the weighted average of cLvl across perceived Markov states.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        StateCount = self.MrkvArray.shape[0]
        t = self.t_cycle[0]
        
        cNrmNow = np.zeros(StateCount)
        for i in range(StateCount):
            cNrmNow[i] = self.solution[t].cFunc[i](self.mNrmNow[i])
        self.cNrmNow = cNrmNow
        self.cLvlNow = np.dot(cNrmNow*self.pLvlNow,self.MrkvPcvd)
        