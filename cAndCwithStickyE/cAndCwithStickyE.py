'''
This module runs the exercises and regressions for the cAndCwithStickyE paper.
User can choose which among the six model variations are actually run.  Descriptive
statistics and regression results are both output to screen and saved in a log
file in the ./results directory.  See StickyEparams for calibrated model parameters.
'''

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))

import numpy as np
from time import clock
from copy import deepcopy
from StickyEmodel import StickyEconsumerType, StickyEmarkovConsumerType, StickyErepAgent,\
                         StickyEmarkovRepAgent, StickyCobbDouglasEconomy, StickyCobbDouglasMarkovEconomy
from ConsAggShockModel import SmallOpenEconomy, SmallOpenMarkovEconomy
from HARKutilities import plotFuncs
import matplotlib.pyplot as plt
import StickyEparams as Params
from StickyEtools import makeStickyEdataFile, runStickyEregressions, makeResultsTable,\
                  runStickyEregressionsInStata, makeParameterTable, makeEquilibriumTable,\
                  makeMicroRegressionTable, extractSampleMicroData

# Choose which models to do work for
do_SOE_simple  = False
do_SOE_markov  = True
do_DSGE_simple = False
do_DSGE_markov = False
do_RA_simple   = False
do_RA_markov   = False

# Choose what kind of work to do for each model
run_models = True        # Whether to solve models and generate new simulated data
calc_micro_stats = True  # Whether to calculate microeconomic statistics (only matters when run_models is True)
make_tables = True       # Whether to make LaTeX tables in the /Tables folder
use_stata = True         # Whether to use Stata to run regressions
save_data = True         # Whether to save data for use in Stata (as a tab-delimited text file)

ignore_periods = Params.ignore_periods # Number of simulated periods to ignore as a "burn-in" phase
interval_size = Params.interval_size   # Number of periods in each non-overlapping subsample
total_periods = Params.periods_to_sim  # Total number of periods in simulation
interval_count = (total_periods-ignore_periods)/interval_size # Number of intervals in the macro regressions
periods_to_sim_micro = Params.periods_to_sim_micro #To save memory, micro regressions are run on a smaller sample
AgentCount_micro = Params.AgentCount_micro #To save memory, micro regressions are run on a smaller sample
my_counts = [interval_size,interval_count]
mystr = lambda number : "{:.3f}".format(number)

# Define the function to run macroeconomic regressions, depending on whether Stata is used
if use_stata:
    runRegressions = lambda a,b,c,d : runStickyEregressionsInStata(a,b,c,d,Params.stata_exe)
else:
    runRegressions = lambda a,b,c,d : runStickyEregressions(a,b,c,d)



# Run models and save output if this module is called from main
if __name__ == '__main__':
    ###############################################################################
    ################# SMALL OPEN ECONOMY ##########################################
    ###############################################################################
    
    if do_SOE_simple:
        if run_models:
            # Make a small open economy and the consumers who live in it
            StickySOEbaseType = StickyEconsumerType(**Params.init_SOE_consumer)
            StickySOEbaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age','TranShkNow']
            StickySOEconsumers = []
            for n in range(Params.TypeCount):
                StickySOEconsumers.append(deepcopy(StickySOEbaseType))
                StickySOEconsumers[-1].seed = n
                StickySOEconsumers[-1].DiscFac = Params.DiscFacSetSOE[n]
            StickySOEconomy = SmallOpenEconomy(agents=StickySOEconsumers, **Params.init_SOE_market)
            StickySOEconomy.makeAggShkHist()
            for n in range(Params.TypeCount):
                StickySOEconsumers[n].getEconomyData(StickySOEconomy)
            
            # Solve the small open economy and display some output
            t_start = clock()
            StickySOEconomy.solveAgents()
            t_end = clock()
            print('Solving the small open economy took ' + str(t_end-t_start) + ' seconds.')
            
            print('Consumption function for one type in the small open economy:')
            cFunc = lambda m : StickySOEconsumers[0].solution[0].cFunc(m,np.ones_like(m))
            plotFuncs(cFunc,0.0,20.0)
            
            # Simulate the frictionless small open economy
            t_start = clock()
            for agent in StickySOEconomy.agents:
                agent(UpdatePrb = 1.0)
            StickySOEconomy.makeHistory()
            t_end = clock()
            print('Simulating the frictionless small open economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Make results for the frictionless representative agent economy
            desc = 'Results for the frictionless small open economy (update probability 1.0)'
            name = 'SOEsimpleFrictionless'
            makeStickyEdataFile(StickySOEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            
            # Simulate the sticky small open economy
            t_start = clock()
            for agent in StickySOEconomy.agents:
                agent(UpdatePrb = Params.UpdatePrb)
            StickySOEconomy.makeHistory()
            t_end = clock()
            print('Simulating the sticky small open economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Make results for the sticky small open economy
            desc = 'Results for the sticky small open economy with update probability ' + mystr(Params.UpdatePrb)
            name = 'SOEsimpleSticky'
            makeStickyEdataFile(StickySOEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
        
        if make_tables:
            # Process the coefficients, standard errors, etc into a LaTeX table
            t_start = clock()
            frictionless_panel = runRegressions('SOEsimpleFrictionlessData',interval_size,False,False)
            frictionless_me_panel = runRegressions('SOEsimpleFrictionlessData',interval_size,True,False)
            sticky_panel = runRegressions('SOEsimpleStickyData',interval_size,False,True)
            sticky_me_panel = runRegressions('SOEsimpleStickyData',interval_size,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in SOE Model',[frictionless_me_panel,sticky_panel,sticky_me_panel],my_counts,'SOEsimReg','tPESOEsim')
            t_end = clock()
            print('Running time series regressions for the small open economy took ' + mystr(t_end-t_start) + ' seconds.')
    
    
    ###############################################################################
    ########## SMALL OPEN ECONOMY WITH MACROECONOMIC MARKOV STATE##################
    ###############################################################################
    
    if do_SOE_markov:
        if run_models:
            # Make consumer types to inhabit the small open Markov economy
            StickySOEmarkovBaseType = StickyEmarkovConsumerType(**Params.init_SOE_mrkv_consumer)
            StickySOEmarkovBaseType.IncomeDstn[0] = Params.StateCount*[StickySOEmarkovBaseType.IncomeDstn[0]]
            StickySOEmarkovBaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age','TranShkNow']
            StickySOEmarkovConsumers = []
            for n in range(Params.TypeCount):
                StickySOEmarkovConsumers.append(deepcopy(StickySOEmarkovBaseType))
                StickySOEmarkovConsumers[-1].seed = n
                StickySOEmarkovConsumers[-1].DiscFac = Params.DiscFacSetSOE[n]
            
            # Make a small open economy for the agents
            StickySOmarkovEconomy = SmallOpenMarkovEconomy(agents=StickySOEmarkovConsumers, **Params.init_SOE_mrkv_market)
            StickySOmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
            for n in range(Params.TypeCount):
                StickySOEmarkovConsumers[n].getEconomyData(StickySOmarkovEconomy) # Have the consumers inherit relevant objects from the economy
            
            # Solve the small open Markov model
            t_start = clock()
            StickySOmarkovEconomy.solveAgents()
            t_end = clock()
            print('Solving the small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Plot the consumption function in each Markov state
            print('Consumption function for one type in the small open Markov economy:')
            m = np.linspace(0,20,500)
            M = np.ones_like(m)
            c = np.zeros((Params.StateCount,m.size))
            for i in range(Params.StateCount):
                c[i,:] = StickySOEmarkovConsumers[0].solution[0].cFunc[i](m,M)
                plt.plot(m,c[i,:])
            plt.show()
            
            # Simulate the frictionless small open Markov economy
            t_start = clock()
            for agent in StickySOmarkovEconomy.agents:
                agent(UpdatePrb = 1.0)
            StickySOmarkovEconomy.makeHistory()
            t_end = clock()
            print('Simulating the frictionless small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Make results for the frictionless small open Markov economy
            desc = 'Results for the frictionless small open Markov economy (update probability 1.0)'
            name = 'SOEmarkovFrictionless'
            makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            if calc_micro_stats:
                frictionless_SOEmarkov_micro_data = extractSampleMicroData(StickySOmarkovEconomy, np.minimum(StickySOmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickySOmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)
            
            # Simulate the frictionless small open Markov economy
            t_start = clock()
            for agent in StickySOmarkovEconomy.agents:
                agent(UpdatePrb = Params.UpdatePrb)
            StickySOmarkovEconomy.makeHistory()
            t_end = clock()
            print('Simulating the sticky small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Make results for the sticky small open Markov economy
            desc = 'Results for the sticky small open Markov economy with update probability ' + mystr(Params.UpdatePrb)
            name = 'SOEmarkovSticky'
            makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            
            if calc_micro_stats:
                sticky_SOEmarkov_micro_data = extractSampleMicroData(StickySOmarkovEconomy, np.minimum(StickySOmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickySOmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)
                makeMicroRegressionTable('CGrowCross.tex', [frictionless_SOEmarkov_micro_data,sticky_SOEmarkov_micro_data])
        
        # Process the coefficients, standard errors, etc into a LaTeX table
        if make_tables:
            # Process the coefficients, standard errors, etc into a LaTeX table
            t_start = clock()
            frictionless_panel = runRegressions('SOEmarkovFrictionlessData',interval_size,False,False)
            frictionless_me_panel = runRegressions('SOEmarkovFrictionlessData',interval_size,True,False)
            sticky_panel = runRegressions('SOEmarkovStickyData',interval_size,False,True)
            sticky_me_panel = runRegressions('SOEmarkovStickyData',interval_size,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in SOE Model',[frictionless_me_panel,sticky_panel,sticky_me_panel],my_counts,'SOEmrkvSimReg','tPESOEsim')
            t_end = clock()
            print('Running time series regressions for the small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
    
    ###############################################################################
    ################# COBB-DOUGLAS ECONOMY ########################################
    ###############################################################################
    
    if do_DSGE_simple:
        if run_models:
            # Make consumers who will live in a Cobb-Douglas economy
            StickyDSGEbaseType = StickyEconsumerType(**Params.init_DSGE_consumer)
            StickyDSGEbaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','pLvlNow','t_age','TranShkNow']
            StickyDSGEconsumers = []
            for n in range(Params.TypeCount):
                StickyDSGEconsumers.append(deepcopy(StickyDSGEbaseType))
                StickyDSGEconsumers[-1].seed = n
                StickyDSGEconsumers[-1].DiscFac = Params.DiscFacSetDSGE[n]
                
            # Make a Cobb-Douglas economy and put the agents in it
            StickyDSGEeconomy = StickyCobbDouglasEconomy(agents=StickyDSGEconsumers,**Params.init_DSGE_market)
            StickyDSGEeconomy.makeAggShkHist()
            for n in range(Params.TypeCount):
                StickyDSGEconsumers[n].getEconomyData(StickyDSGEeconomy)
                StickyDSGEconsumers[n](UpdatePrb = 1.0)
                
            # Solve the frictionless HA-DSGE model
            t_start = clock()
            StickyDSGEeconomy.solve()
            t_end = clock()
            print('Solving the frictionless Cobb-Douglas economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Plot the consumption function
            print('Consumption function for the frictionless Cobb-Douglas economy:')
            m = np.linspace(0.,20.,300)
            for M in StickyDSGEconsumers[0].Mgrid:
                c = StickyDSGEconsumers[0].solution[0].cFunc(m,M*np.ones_like(m))
                plt.plot(m,c)
            plt.show()
            
            # Make results for the frictionless Cobb-Douglas economy
            desc = 'Results for the frictionless Cobb-Douglas economy (update probability 1.0)'
            name = 'DSGEsimpleFrictionless'
            makeStickyEdataFile(StickyDSGEeconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            
            # Solve the sticky HA-DSGE model
            for agent in StickyDSGEeconomy.agents:
                agent(UpdatePrb = Params.UpdatePrb)
            t_start = clock()
            StickyDSGEeconomy.solve()
            t_end = clock()
            print('Solving the sticky Cobb-Douglas economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Plot the consumption function
            print('Consumption function for the sticky Cobb-Douglas economy:')
            m = np.linspace(0.,20.,300)
            for M in StickyDSGEconsumers[0].Mgrid:
                c = StickyDSGEconsumers[0].solution[0].cFunc(m,M*np.ones_like(m))
                plt.plot(m,c)
            plt.show()
            
            # Make results for the sticky Cobb-Douglas economy
            desc = 'Results for the sticky Cobb-Douglas economy with update probability ' + mystr(Params.UpdatePrb)
            name = 'DSGEsimpleSticky'
            makeStickyEdataFile(StickyDSGEeconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
        
        # Process the coefficients, standard errors, etc into a LaTeX table
        if make_tables:
            # Process the coefficients, standard errors, etc into a LaTeX table
            t_start = clock()
            frictionless_panel = runRegressions('DSGEsimpleFrictionlessData',interval_size,False,False)
            frictionless_me_panel = runRegressions('DSGEsimpleFrictionlessData',interval_size,True,False)
            sticky_panel = runRegressions('DSGEsimpleStickyData',interval_size,False,True)
            sticky_me_panel = runRegressions('DSGEsimpleStickyData',interval_size,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in HA-DSGE Model',[frictionless_me_panel,sticky_panel,sticky_me_panel],my_counts,'DSGEsimReg','tDSGEsim')
            t_end = clock()
            print('Running time series regressions for the Cobb-Douglas economy took ' + mystr(t_end-t_start) + ' seconds.')
    
    ###############################################################################
    ########## COBB-DOUGLAS ECONOMY WITH MACROECONOMIC MARKOV STATE ###############
    ###############################################################################
    
    if do_DSGE_markov:
        if run_models:
            # Make consumers who will live in the Cobb-Douglas Markov economy
            StickyDSGEmarkovBaseType = StickyEmarkovConsumerType(**Params.init_DSGE_mrkv_consumer)
            StickyDSGEmarkovBaseType.IncomeDstn[0] = Params.StateCount*[StickyDSGEmarkovBaseType.IncomeDstn[0]]
            StickyDSGEmarkovBaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age','TranShkNow']
            StickyDSGEmarkovConsumers = []
            for n in range(Params.TypeCount):
                StickyDSGEmarkovConsumers.append(deepcopy(StickyDSGEmarkovBaseType))
                StickyDSGEmarkovConsumers[-1].seed = n
                StickyDSGEmarkovConsumers[-1].DiscFac = Params.DiscFacSetDSGE[n]
            
            # Make a Cobb-Douglas economy for the agents
            StickyDSGEmarkovEconomy = StickyCobbDouglasMarkovEconomy(agents = StickyDSGEmarkovConsumers,**Params.init_DSGE_mrkv_market)
            StickyDSGEmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
            for n in range(Params.TypeCount):
                StickyDSGEmarkovConsumers[n].getEconomyData(StickyDSGEmarkovEconomy) # Have the consumers inherit relevant objects from the economy
                StickyDSGEmarkovConsumers[n](UpdatePrb = 1.0)
            
            # Solve the frictionless heterogeneous agent DSGE model
            t_start = clock()
            StickyDSGEmarkovEconomy.solve()
            t_end = clock()
            print('Solving the frictionless Cobb-Douglas Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            print('Displaying the consumption functions for the Cobb-Douglas Markov economy would be too much.')
            
            # Make results for the Cobb-Douglas Markov economy
            desc = 'Results for the frictionless Cobb-Douglas Markov economy (update probability 1.0)'
            name = 'DSGEmarkovFrictionless'
            makeStickyEdataFile(StickyDSGEmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            if calc_micro_stats:
                frictionless_DSGEmarkov_micro_data = extractSampleMicroData(StickyDSGEmarkovEconomy, np.minimum(StickyDSGEmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickyDSGEmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)
            
            # Solve the sticky heterogeneous agent DSGE model
            for agent in StickyDSGEmarkovEconomy.agents:
                agent(UpdatePrb = Params.UpdatePrb)
            t_start = clock()
            StickyDSGEmarkovEconomy.solve()
            t_end = clock()
            print('Solving the sticky Cobb-Douglas Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            print('Displaying the consumption functions for the Cobb-Douglas Markov economy would be too much.')
            
            # Make results for the Cobb-Douglas Markov economy
            desc = 'Results for the sticky Cobb-Douglas Markov economy with update probability ' + mystr(Params.UpdatePrb)
            name = 'DSGEmarkovSticky'
            makeStickyEdataFile(StickyDSGEmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
        
            if calc_micro_stats:
                sticky_DSGEmarkov_micro_data = extractSampleMicroData(StickyDSGEmarkovEconomy, np.minimum(StickyDSGEmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickyDSGEmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)
                makeMicroRegressionTable('CGrowCrossDSGE.tex', [frictionless_DSGEmarkov_micro_data,sticky_DSGEmarkov_micro_data])
        
        # Process the coefficients, standard errors, etc into a LaTeX table
        if make_tables:
            # Process the coefficients, standard errors, etc into a LaTeX table
            t_start = clock()
            frictionless_panel = runRegressions('DSGEmarkovFrictionlessData',interval_size,False,False)
            frictionless_me_panel = runRegressions('DSGEmarkovFrictionlessData',interval_size,True,False)
            sticky_panel = runRegressions('DSGEmarkovStickyData',interval_size,False,True)
            sticky_me_panel = runRegressions('DSGEmarkovStickyData',interval_size,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in HA-DSGE Model',[frictionless_me_panel,sticky_panel,sticky_me_panel],my_counts,'DSGEmrkvSimReg','tDSGEsim')
            t_end = clock()
            print('Running time series regressions for the Cobb-Douglas Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
       
    
    ###############################################################################
    ################# REPRESENTATIVE AGENT ECONOMY ################################
    ###############################################################################
    
    if do_RA_simple:
        if run_models:
            # Make a representative agent consumer, then solve and simulate the model
            StickyRAconsumer = StickyErepAgent(**Params.init_RA_consumer)
            StickyRAconsumer.track_vars = ['cLvlNow','yNrmTrue','aLvlNow','pLvlTrue','TranShkNow']
            
            # Solve the representative agent's problem        
            t_start = clock()
            StickyRAconsumer.solve()
            t_end = clock()
            print('Solving the representative agent economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            print('Consumption function for the representative agent:')
            plotFuncs(StickyRAconsumer.solution[0].cFunc,0,50)
            
            # Simulate the representative agent with frictionless expectations
            t_start = clock()
            StickyRAconsumer(UpdatePrb = 1.0)
            StickyRAconsumer.initializeSim()
            StickyRAconsumer.simulate()
            t_end = clock()
            print('Simulating the frictionless representative agent economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Make results for the frictionless representative agent economy
            desc = 'Results for the frictionless representative agent economy (update probability 1.0)'
            name = 'RAsimpleFrictionless'
            makeStickyEdataFile(StickyRAconsumer,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            
            # Simulate the representative agent with sticky expectations
            t_start = clock()
            StickyRAconsumer(UpdatePrb = Params.UpdatePrb)
            StickyRAconsumer.initializeSim()
            StickyRAconsumer.simulate()
            t_end = clock()
            print('Simulating the sticky representative agent economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Make results for the sticky representative agent economy
            desc = 'Results for the sticky representative agent economy with update probability ' + mystr(Params.UpdatePrb)
            name = 'RAsimpleSticky'
            makeStickyEdataFile(StickyRAconsumer,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
        
        if make_tables:
            # Process the coefficients, standard errors, etc into a LaTeX table
            t_start = clock()
            frictionless_panel = runRegressions('RAsimpleFrictionlessData',interval_size,False,False)
            frictionless_me_panel = runRegressions('RAsimpleFrictionlessData',interval_size,True,False)
            sticky_panel = runRegressions('RAsimpleStickyData',interval_size,False,True)
            sticky_me_panel = runRegressions('RAsimpleStickyData',interval_size,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in RA Model',[frictionless_me_panel,sticky_panel,sticky_me_panel],my_counts,'RepAgentSimReg','tRAsim')
            t_end = clock()
            print('Running time series regressions for the representative agent economy took ' + mystr(t_end-t_start) + ' seconds.')
    
    ###############################################################################
    ########### REPRESENTATIVE AGENT ECONOMY WITH MARKOV STATE ####################
    ###############################################################################
    
    if do_RA_markov:
        if run_models:
            # Make a representative agent consumer, then solve and simulate the model
            StickyRAmarkovConsumer = StickyEmarkovRepAgent(**Params.init_RA_mrkv_consumer)
            StickyRAmarkovConsumer.IncomeDstn[0] = Params.StateCount*[StickyRAmarkovConsumer.IncomeDstn[0]]
            StickyRAmarkovConsumer.track_vars = ['cLvlNow','yNrmTrue','aLvlNow','pLvlTrue','TranShkNow','MrkvNow']
            
            # Solve the representative agent Markov economy
            t_start = clock()
            StickyRAmarkovConsumer.solve()
            t_end = clock()
            print('Solving the representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            print('Consumption functions for the Markov representative agent:')
            plotFuncs(StickyRAmarkovConsumer.solution[0].cFunc,0,50)
            
            # Simulate the frictionless representative agent MarkovEconomy
            t_start = clock()
            StickyRAmarkovConsumer(UpdatePrb = 1.0)
            StickyRAmarkovConsumer.initializeSim()
            StickyRAmarkovConsumer.simulate()
            t_end = clock()
            print('Simulating the frictionless representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Make results for the frictionless representative agent economy
            desc = 'Results for the frictionless representative agent Markov economy (update probability 1.0)'
            name = 'RAmarkovFrictionless'
            makeStickyEdataFile(StickyRAmarkovConsumer,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            
            # Simulate the sticky representative agent MarkovEconomy
            t_start = clock()
            StickyRAmarkovConsumer(UpdatePrb = Params.UpdatePrb)
            StickyRAmarkovConsumer.initializeSim()
            StickyRAmarkovConsumer.simulate()
            t_end = clock()
            print('Simulating the sticky representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
            
            # Make results for the frictionless representative agent economy
            desc = 'Results for the sticky representative agent Markov economy'
            name = 'RAmarkovSticky'
            makeStickyEdataFile(StickyRAmarkovConsumer,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
        
        if make_tables:
            # Process the coefficients, standard errors, etc into a LaTeX table
            t_start = clock()
            #frictionless_panel = runRegressions('RAmarkovFrictionlessData',interval_size,False,False)
            frictionless_me_panel = runRegressions('RAmarkovFrictionlessData',interval_size,True,False)
            sticky_panel = runRegressions('RAmarkovStickyData',interval_size,False,True)
            sticky_me_panel = runRegressions('RAmarkovStickyData',interval_size,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in RA Model',[frictionless_me_panel,sticky_panel,sticky_me_panel],my_counts,'RepAgentMrkvSimReg','tRAsim')
            t_end = clock()
            print('Running time series regressions for the representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
        
    ###############################################################################
    ########### MAKE OTHER TABLES #################################################
    ###############################################################################
    if make_tables:
        makeEquilibriumTable('Eqbm.tex', ['SOEmarkovFrictionless','SOEmarkovSticky','DSGEmarkovFrictionless','DSGEmarkovSticky'],Params.init_SOE_consumer['CRRA'])
        makeParameterTable('Calibration.tex', Params)