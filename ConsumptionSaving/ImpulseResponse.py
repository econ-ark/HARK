# -*- coding: utf-8 -*-
"""
Contains impulse response code for ConsIndShockModel and ConsAggShockModel
"""
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
from ConsIndShockModel import  IndShockConsumerType
from ConsAggShockModel import AggShockConsumerType, CobbDouglasEconomy
from copy import deepcopy
import matplotlib.pyplot as plt

def impulseResponseInd(IndShockConsumer, PermShk, TranShk, TimeBefore = 100, TimeAfter = 20):
    '''
    Creates an impulse response times series if consumers of type 
    IndShockConsumerType are all shocked with an additional PermShk and TranShk
    
    Parameters
    ----------
    IndShockConsumer : IndShockConsumerType
        An instance of the consumer to be shocked
    PermShk : [np.array]
        An array containing the additional permanent shock to be applied to the 
        consumers. 
    TranShk : [np.array]
        An array containing the additional transitory shock to be applied to the 
        consumers. 
    TimeBefore : float
        Number of time periods to run the simulation before applying the shocks
            TimeBefore : float
        Number of time periods to run the simulation after applying the shocks
    Returns
    -------
    impulse_response : (IndShockConsumerType, IndShockConsumerType)
        A tupel containing the shocked consumer and the un-shocked consumer
    '''
    # create a standard and shocked version of the consumer type
    NoShockConsumer = deepcopy(IndShockConsumer)
    PermShkFull = np.append(np.ones(TimeBefore),PermShk)
    PermShkFull = np.append(PermShkFull, np.ones(TimeAfter))
    TranShkFull = np.append(np.ones(TimeBefore),TranShk)
    TranShkFull = np.append(TranShkFull, np.ones(TimeAfter))
    sim_periods = len(PermShkFull)
    NoShockConsumer.sim_periods = sim_periods
    NoShockConsumer.makeIncShkHist()
    ShockedConsumer = deepcopy(NoShockConsumer)
    # Shock the idiosyncratic shock history with the required shock
    ShockedConsumer.PermShkHist = np.multiply(ShockedConsumer.PermShkHist.transpose(),PermShkFull).transpose()
    ShockedConsumer.TranShkHist = np.multiply(ShockedConsumer.TranShkHist.transpose(),TranShkFull).transpose()
    # simulate consumption history
    NoShockConsumer.initializeSim()
    NoShockConsumer.simConsHistory()
    
    ShockedConsumer.initializeSim()
    ShockedConsumer.simConsHistory()

    return (ShockedConsumer, NoShockConsumer)
    
def AggCons_impulseResponseInd(IndShockConsumer, PermShk, TranShk, TimeBefore = 100, TimeAfter = 20):
    '''
    Routine to calculate the aggregate consumption impulse response for consumers
    of type IndShockConsumerType
    
    Parameters
    ----------
    IndShockConsumer : IndShockConsumerType
        An instance of the consumer to be shocked
    PermShk : [np.array]
        An array containing the additional permanent shock to be applied to the 
        consumers. 
    TranShk : [np.array]
        An array containing the additional permanent shock to be applied to the 
        consumers. 
    TimeBefore : float
        Number of time periods to run the simulation before applying the shocks
            TimeBefore : float
        Number of time periods to run the simulation after applying the shocks
    Returns
    -------
    impulse_response : [np.array]
        An array containing the aggregate consumption response
    '''
    (ShockedConsumer, NoShockConsumer) = impulseResponseInd(IndShockConsumer, PermShk, TranShk, TimeBefore, TimeAfter)
    
    pNoShkHist = NoShockConsumer.pHist
    cNoShkHist = NoShockConsumer.cHist
    cAggNoShkHist = np.sum(pNoShkHist*cNoShkHist,1)
    
    pShkHist = ShockedConsumer.pHist
    cShkHist = ShockedConsumer.cHist
    cAggShkHist = np.sum(pShkHist*cShkHist,1)
    
    cAggImpulseResponse = (cAggShkHist - cAggNoShkHist)/cAggNoShkHist
    
    # Remove TimeBefore periods
    cAggImpulseResponse = cAggImpulseResponse[TimeBefore-1:]

    return cAggImpulseResponse
    
def impulseResponseAgg(Market, PermShk, TranShk, TimeBefore = 100, TimeAfter = 20):
    '''
    Creates an impulse response times series if an economy is hit with permanent
    and transitory aggregate shocks
    
    Parameters
    ----------
    Market : Market
        An instance of the Market to be shocked
    PermShk : [np.array]
        An array containing the permanent aggregate shocks
    TranShk : [np.array]
        An array containing the transitory aggregate shocks 
    TimeBefore : float
        Number of time periods to run the simulation before applying the shocks
            TimeBefore : float
        Number of time periods to run the simulation after applying the shocks
    Returns
    -------
    impulse_response : (Market, Market)
        A tupel containing the shocked market and the un-shocked market
    '''
    # create a standard and shocked version of the market
    NoShockMarket = deepcopy(Market)
    PermShkFull = np.append(np.ones(TimeBefore),PermShk)
    PermShkFull = np.append(PermShkFull, np.ones(TimeAfter))
    TranShkFull = np.append(np.ones(TimeBefore),TranShk)
    TranShkFull = np.append(TranShkFull, np.ones(TimeAfter))
    sim_periods = len(PermShkFull)
    NoShockMarket.act_T = sim_periods
    
    NoShockMarket.PermShkAggHist = np.ones_like(PermShkFull)
    NoShockMarket.TranShkAggHist = np.ones_like(TranShkFull)
    for this_type in NoShockMarket.agents:
        this_type.sim_periods = sim_periods
        this_type.makeIncShkHist()
        this_type.initializeSim()

    ShockedMarket = deepcopy(NoShockMarket)
    ShockedMarket.PermShkAggHist = PermShkFull
    ShockedMarket.TranShkAggHist = TranShkFull
    
    NoShockMarket.makeHistory()
    ShockedMarket.makeHistory()
    
    return (ShockedMarket, NoShockMarket)
    
def AggCons_impulseResponseAgg(Market, PermShk, TranShk, TimeBefore = 100, TimeAfter = 20):
    '''
    Routine to calculate the aggregate consumption impulse response for a Market
    
    Parameters
    ----------
    Market : Market
        An instance of the Market to be shocked
    PermShk : [np.array]
        An array containing the permanent aggregate shocks
    TranShk : [np.array]
        An array containing the transitory aggregate shocks 
    TimeBefore : float
        Number of time periods to run the simulation before applying the shocks
            TimeBefore : float
        Number of time periods to run the simulation after applying the shocks
    Returns
    -------
    impulse_response : [np.array]
        An array containing the aggregate consumption response
    '''
    (ShockedMarket, NoShockMarket) = impulseResponseAgg(Market, PermShk, TranShk, TimeBefore, TimeAfter)
    
    cAggNoShkHist = np.zeros(NoShockMarket.act_T)
    for this_type in NoShockMarket.agents:
        pNoShkHist = this_type.pHist
        cNoShkHist = this_type.cHist
        cAggNoShkHist += np.sum(pNoShkHist*cNoShkHist,1)
    
    cAggShkHist = np.zeros(ShockedMarket.act_T)
    for this_type in ShockedMarket.agents:
        pShkHist = this_type.pHist
        cShkHist = this_type.cHist
        cAggShkHist += np.sum(pShkHist*cShkHist,1)
    
    cAggImpulseResponse = (cAggShkHist - cAggNoShkHist)/cAggNoShkHist
    
    # Remove TimeBefore periods
    cAggImpulseResponse = cAggImpulseResponse[TimeBefore:]

    return cAggImpulseResponse
    
    
###############################################################################
     
if __name__ == '__main__':
    import ConsumerParameters as Params
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)
    
    # Make and solve an example consumer with idiosyncratic income shocks
    IndShockExample = IndShockConsumerType(**Params.init_idiosyncratic_shocks)
    IndShockExample.cycles = 0 # Make this type have an infinite horizon
    IndShockExample.DiePrb = 1-IndShockExample.LivPrb[0]  # Not sure why this is not already in the object
    
    start_time = clock()
    IndShockExample.solve()
    end_time = clock()
    print('Solving a consumer with idiosyncratic shocks took ' + mystr(end_time-start_time) + ' seconds.')
    IndShockExample.unpackcFunc()
    IndShockExample.timeFwd()
    
    PermShk = 1.1
    TranShk = 1
    cPermImpulseResponse = AggCons_impulseResponseInd(IndShockExample,PermShk,TranShk,100,25)
    plt.plot(cPermImpulseResponse)
    
    PermShk = 1
    TranShk = 1.1
    cTranImpulseResponse = AggCons_impulseResponseInd(IndShockExample,PermShk,TranShk,100,25)
    plt.plot(cTranImpulseResponse)
    print('Impulse response to a one time permanent and transitive shock to income of 10%:')
    plt.show()
    
##########################################

# Now do aggregate shocks of a market
    # Make an aggregate shocks consumer
    AggShockExample = AggShockConsumerType(**Params.init_agg_shocks)
    AggShockExample.cycles = 0
    AggShockExample.sim_periods = 3000
    AggShockExample.makeIncShkHist()  # Simulate a history of idiosyncratic shocks
    # Make a Cobb-Douglas economy for the agents
    EconomyExample = CobbDouglasEconomy(agents = [AggShockExample],act_T=AggShockExample.sim_periods,**Params.init_cobb_douglas)
    EconomyExample.makeAggShkHist() # Simulate a history of aggregate shocks
    
    # Have the consumers inherit relevant objects from the economy
    AggShockExample.getEconomyData(EconomyExample)
    
    # Solve the microeconomic model for the aggregate shocks example type (and display results)
    t_start = clock()
    AggShockExample.solve()
    t_end = clock()
    print('Solving an aggregate shocks consumer took ' + mystr(t_end-t_start) + ' seconds.')
    
#    # Solve the "macroeconomic" model by searching for a "fixed point dynamic rule"
#    t_start = clock()
#    EconomyExample.solve()
#    t_end = clock()
#    print('Solving the "macroeconomic" aggregate shocks model took ' + str(t_end - t_start) + ' seconds.')
    
    PermShk = 1.1
    TranShk = 1
    cMarketPermImpulseResponse = AggCons_impulseResponseAgg(EconomyExample,PermShk,TranShk,100,25)
    plt.plot(cMarketPermImpulseResponse)
    
    PermShk = 1
    TranShk = 1.1
    cMarketTranImpulseResponse = AggCons_impulseResponseAgg(EconomyExample,PermShk,TranShk,100,25)
    plt.plot(cMarketTranImpulseResponse)
    print('Impulse response to a one time permanent and transitive shock to income of 10%:')
    plt.show()
    
    
    
