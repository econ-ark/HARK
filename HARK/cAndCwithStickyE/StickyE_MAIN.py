'''
This module runs the exercises and regressions for the cAndCwithStickyE paper.
User can choose which among the three models are actually run.  Descriptive
statistics and regression results are both output to screen and saved in a log
file in the results directory.  TeX code for tables in the paper are saved in
the tables directory.  See StickyEparams for calibrated model parameters.
'''

import os
import numpy as np
import csv
from time import clock
from copy import deepcopy
from StickyEmodel import StickyEmarkovConsumerType, StickyEmarkovRepAgent, StickyCobbDouglasMarkovEconomy
from HARK.ConsumptionSaving.ConsAggShockModel import SmallOpenMarkovEconomy
from HARK.utilities import plotFuncs
import matplotlib.pyplot as plt
import StickyEparams as Params
from StickyEtools import makeStickyEdataFile, runStickyEregressions, makeResultsTable,\
                  runStickyEregressionsInStata, makeParameterTable, makeEquilibriumTable,\
                  makeMicroRegressionTable, extractSampleMicroData, makeuCostVsPiFig, \
                  makeValueVsAggShkVarFig, makeValueVsPiFig

# Choose which models to do work for
do_SOE  = False
do_DSGE = False
do_RA   = False

# Choose what kind of work to do for each model
run_models = False       # Whether to solve models and generate new simulated data
calc_micro_stats = False # Whether to calculate microeconomic statistics (only matters when run_models is True)
make_tables = False      # Whether to make LaTeX tables in the /Tables folder
use_stata = False        # Whether to use Stata to run the simulated time series regressions
save_data = False        # Whether to save data for use in Stata (as a tab-delimited text file)
run_ucost_vs_pi = False  # Whether to run an exercise that finds the cost of stickiness as it varies with update probability
run_value_vs_aggvar = False # Whether to run an exercise to find value at birth vs variance of aggregate permanent shocks

ignore_periods = Params.ignore_periods # Number of simulated periods to ignore as a "burn-in" phase
interval_size = Params.interval_size   # Number of periods in each non-overlapping subsample
total_periods = Params.periods_to_sim  # Total number of periods in simulation
interval_count = (total_periods-ignore_periods)/interval_size # Number of intervals in the macro regressions
periods_to_sim_micro = Params.periods_to_sim_micro # To save memory, micro regressions are run on a smaller sample
AgentCount_micro = Params.AgentCount_micro # To save memory, micro regressions are run on a smaller sample
my_counts = [interval_size,interval_count]
alt_counts = [interval_size*interval_count,1]
mystr = lambda number : "{:.3f}".format(number)
results_dir = Params.results_dir

# Define the function to run macroeconomic regressions, depending on whether Stata is used
if use_stata:
    runRegressions = lambda a,b,c,d,e : runStickyEregressionsInStata(a,b,c,d,e,Params.stata_exe)
else:
    runRegressions = lambda a,b,c,d,e : runStickyEregressions(a,b,c,d,e)



# Run models and save output if this module is called from main
if __name__ == '__main__':

    ###############################################################################
    ########## SMALL OPEN ECONOMY WITH MACROECONOMIC MARKOV STATE##################
    ###############################################################################

    if do_SOE:
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
            StickySOmarkovEconomy.track_vars += ['TranShkAggNow','wRteNow']
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

            # Simulate the sticky small open Markov economy
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
            DeltaLogC_stdev = np.genfromtxt(results_dir + 'SOEmarkovStickyResults.csv', delimiter=',')[3] # For use in frictionless spec

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
            makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats,meas_err_base=DeltaLogC_stdev)
            if calc_micro_stats:
                frictionless_SOEmarkov_micro_data = extractSampleMicroData(StickySOmarkovEconomy, np.minimum(StickySOmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickySOmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)
                makeMicroRegressionTable('CGrowCross', [frictionless_SOEmarkov_micro_data,sticky_SOEmarkov_micro_data])

            if run_ucost_vs_pi:
                # Find the birth value and cost of stickiness as it varies with updating probability
                UpdatePrbVec = np.linspace(0.025,1.0,40)
                CRRA = StickySOmarkovEconomy.agents[0].CRRA
                vBirth_F = np.genfromtxt(results_dir + 'SOEmarkovFrictionlessBirthValue.csv', delimiter=',')
                uCostVec = np.zeros_like(UpdatePrbVec)
                vVec = np.zeros_like(UpdatePrbVec)
                for j in range(UpdatePrbVec.size):
                    for agent in StickySOmarkovEconomy.agents:
                        agent(UpdatePrb = UpdatePrbVec[j])
                    StickySOmarkovEconomy.makeHistory()
                    makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description='trash',filename='TEMP',save_data=False,calc_micro_stats=True)
                    vBirth_S = np.genfromtxt(results_dir + 'TEMPBirthValue.csv', delimiter=',')
                    uCost = np.mean(1. - (vBirth_S/vBirth_F)**(1./(1.-CRRA)))
                    uCostVec[j] = uCost
                    vVec[j] = np.mean(vBirth_S)
                    print('Found that uCost=' + str(uCost) + ' for Pi=' + str(UpdatePrbVec[j]))
                with open(results_dir + 'SOEuCostbyUpdatePrb.csv','w') as f:
                    my_writer = csv.writer(f, delimiter = ',')
                    my_writer.writerow(UpdatePrbVec)
                    my_writer.writerow(uCostVec)
                    f.close()
                with open(results_dir + 'SOEvVecByUpdatePrb.csv','w') as f:
                    my_writer = csv.writer(f, delimiter = ',')
                    my_writer.writerow(UpdatePrbVec)
                    my_writer.writerow(vVec)
                    f.close()
                os.remove(results_dir + 'TEMPResults.csv')
                os.remove(results_dir + 'TEMPBirthValue.csv')

            if run_value_vs_aggvar:
                # Find value as it varies with updating probability
                PermShkAggVarBase = np.linspace(0.5,1.5,40)
                PermShkAggVarVec = PermShkAggVarBase*Params.PermShkAggVar
                vVec = np.zeros_like(PermShkAggVarVec)
                for j in range(PermShkAggVarVec.size):
                    StickySOmarkovEconomy.PermShkAggStd = Params.StateCount*[np.sqrt(PermShkAggVarVec[j])]
                    StickySOmarkovEconomy.makeAggShkDstn()
                    StickySOmarkovEconomy.makeAggShkHist()
                    for agent in StickySOmarkovEconomy.agents:
                        agent(UpdatePrb = 1.0)
                        agent.getEconomyData(StickySOmarkovEconomy)
                    StickySOmarkovEconomy.solveAgents()
                    StickySOmarkovEconomy.makeHistory()
                    makeStickyEdataFile(StickySOmarkovEconomy,ignore_periods,description='trash',filename='TEMP',save_data=False,calc_micro_stats=True)
                    vBirth_S = np.genfromtxt(results_dir + 'TEMPBirthValue.csv', delimiter=',')
                    v = np.mean(vBirth_S)
                    vVec[j] = v
                    print('Found that v=' + str(v) + ' for PermShkAggVar=' + str(PermShkAggVarVec[j]))
                with open(results_dir + 'SOEvVecByPermShkAggVar.csv','w') as f:
                    my_writer = csv.writer(f, delimiter = ',')
                    my_writer.writerow(PermShkAggVarVec)
                    my_writer.writerow(vVec)
                    f.close()
                os.remove(results_dir + 'TEMPResults.csv')
                os.remove(results_dir + 'TEMPBirthValue.csv')

        # Process the coefficients, standard errors, etc into a LaTeX table
        if make_tables:
            t_start = clock()
            frictionless_panel = runRegressions('SOEmarkovFrictionlessData',interval_size,False,False,True)
            frictionless_me_panel = runRegressions('SOEmarkovFrictionlessData',interval_size,True,False,True)
            frictionless_long_panel = runRegressions('SOEmarkovFrictionlessData',interval_size*interval_count,True,False,True)
            sticky_panel = runRegressions('SOEmarkovStickyData',interval_size,False,True,True)
            sticky_me_panel = runRegressions('SOEmarkovStickyData',interval_size,True,True,True)
            sticky_long_panel = runRegressions('SOEmarkovStickyData',interval_size*interval_count,True,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in SOE Model',[frictionless_me_panel,sticky_me_panel],my_counts,'SOEmrkvSimReg','tPESOEsim')
            makeResultsTable('Aggregate Consumption Dynamics in SOE Model',[frictionless_panel,sticky_panel],my_counts,'SOEmrkvSimRegNoMeasErr','tPESOEsimNoMeasErr')
            makeResultsTable('Aggregate Consumption Dynamics in SOE Model',[frictionless_long_panel,sticky_long_panel],alt_counts,'SOEmrkvSimRegLong','tSOEsimLong')
            makeResultsTable(None,[frictionless_me_panel],my_counts,'SOEmrkvSimRegF','tPESOEsimF')
            makeResultsTable(None,[sticky_me_panel],my_counts,'SOEmrkvSimRegS','tPESOEsimS')
            t_end = clock()
            print('Running time series regressions for the small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')


    ###############################################################################
    ########## COBB-DOUGLAS ECONOMY WITH MACROECONOMIC MARKOV STATE ###############
    ###############################################################################

    if do_DSGE:
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
            StickyDSGEmarkovEconomy.track_vars += ['RfreeNow','wRteNow','TranShkAggNow']
            StickyDSGEmarkovEconomy.overwrite_hist = False
            StickyDSGEmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
            for n in range(Params.TypeCount):
                StickyDSGEmarkovConsumers[n].getEconomyData(StickyDSGEmarkovEconomy) # Have the consumers inherit relevant objects from the economy
                StickyDSGEmarkovConsumers[n](UpdatePrb = Params.UpdatePrb)

            # Solve the sticky heterogeneous agent DSGE model
            t_start = clock()
            StickyDSGEmarkovEconomy.solve()
            t_end = clock()
            print('Solving the sticky Cobb-Douglas Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

            print('Displaying the consumption functions for the Cobb-Douglas Markov economy would be too much.')

            # Make results for the sticky Cobb-Douglas Markov economy
            desc = 'Results for the sticky Cobb-Douglas Markov economy with update probability ' + mystr(Params.UpdatePrb)
            name = 'DSGEmarkovSticky'
            makeStickyEdataFile(StickyDSGEmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            DeltaLogC_stdev = np.genfromtxt(results_dir + 'DSGEmarkovStickyResults.csv', delimiter=',')[3] # For use in frictionless spec
            if calc_micro_stats:
                sticky_DSGEmarkov_micro_data = extractSampleMicroData(StickyDSGEmarkovEconomy, np.minimum(StickyDSGEmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickyDSGEmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)

            # Store the histories of MaggNow, wRteNow, and Rfree now in _overwrite attributes
            StickyDSGEmarkovEconomy.MaggNow_overwrite = deepcopy(StickyDSGEmarkovEconomy.MaggNow_hist)
            StickyDSGEmarkovEconomy.wRteNow_overwrite = deepcopy(StickyDSGEmarkovEconomy.wRteNow_hist)
            StickyDSGEmarkovEconomy.RfreeNow_overwrite = deepcopy(StickyDSGEmarkovEconomy.RfreeNow_hist)

            # Calculate the lifetime value of being frictionless when all other agents are sticky
            if calc_micro_stats:
                StickyDSGEmarkovEconomy.overwrite_hist = True # History will be overwritten by sticky outcomes
                for agent in StickyDSGEmarkovEconomy.agents:
                    agent(UpdatePrb = 1.0) # Make agents frictionless
                StickyDSGEmarkovEconomy.makeHistory() # Simulate a history one more time

                # Save the birth value file in a temporary file and delete the other generated results files
                makeStickyEdataFile(StickyDSGEmarkovEconomy,ignore_periods,description=desc,filename=name+'TEMP',save_data=False,calc_micro_stats=calc_micro_stats)
                os.remove(results_dir + name + 'TEMP' + 'Results.csv')
                sticky_name = name

            # Solve the frictionless heterogeneous agent DSGE model
            StickyDSGEmarkovEconomy.overwrite_hist = False
            for agent in StickyDSGEmarkovEconomy.agents:
                agent(UpdatePrb = 1.0)
            t_start = clock()
            StickyDSGEmarkovEconomy.solve()
            t_end = clock()
            print('Solving the frictionless Cobb-Douglas Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

            print('Displaying the consumption functions for the Cobb-Douglas Markov economy would be too much.')

            # Make results for the frictionless Cobb-Douglas Markov economy
            desc = 'Results for the frictionless Cobb-Douglas Markov economy (update probability 1.0)'
            name = 'DSGEmarkovFrictionless'
            makeStickyEdataFile(StickyDSGEmarkovEconomy,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats,meas_err_base=DeltaLogC_stdev)
            if calc_micro_stats:
                os.remove(results_dir + name + 'BirthValue.csv') # Delete the frictionless birth value file
                os.rename(results_dir + sticky_name + 'TEMPBirthValue.csv',results_dir + name + 'BirthValue.csv') # Replace just deleted file with "alternate" value calculation
                frictionless_DSGEmarkov_micro_data = extractSampleMicroData(StickyDSGEmarkovEconomy, np.minimum(StickyDSGEmarkovEconomy.act_T-ignore_periods-1,periods_to_sim_micro), np.minimum(StickyDSGEmarkovEconomy.agents[0].AgentCount,AgentCount_micro), ignore_periods)
                makeMicroRegressionTable('CGrowCrossDSGE', [frictionless_DSGEmarkov_micro_data,sticky_DSGEmarkov_micro_data])

        # Process the coefficients, standard errors, etc into a LaTeX table
        if make_tables:
            t_start = clock()
            frictionless_panel = runRegressions('DSGEmarkovFrictionlessData',interval_size,False,False,True)
            frictionless_me_panel = runRegressions('DSGEmarkovFrictionlessData',interval_size,True,False,True)
            frictionless_long_panel = runRegressions('DSGEmarkovFrictionlessData',interval_size*interval_count,True,False,True)
            sticky_panel = runRegressions('DSGEmarkovStickyData',interval_size,False,True,True)
            sticky_me_panel = runRegressions('DSGEmarkovStickyData',interval_size,True,True,True)
            sticky_long_panel = runRegressions('DSGEmarkovStickyData',interval_size*interval_count,True,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in HA-DSGE Model',[frictionless_me_panel,sticky_me_panel],my_counts,'DSGEmrkvSimReg','tDSGEsim')
            makeResultsTable('Aggregate Consumption Dynamics in HA-DSGE Model',[frictionless_panel,sticky_panel],my_counts,'DSGEmrkvSimRegNoMeasErr','tDSGEsimNoMeasErr')
            makeResultsTable('Aggregate Consumption Dynamics in HA-DSGE Model',[frictionless_long_panel,sticky_long_panel],alt_counts,'DSGEmrkvSimRegLong','tDSGEsimLong')
            makeResultsTable(None,[frictionless_me_panel],my_counts,'DSGEmrkvSimRegF','tDSGEsimF')
            makeResultsTable(None,[sticky_me_panel],my_counts,'DSGEmrkvSimRegS','tDSGEsimS')
            t_end = clock()
            print('Running time series regressions for the Cobb-Douglas Markov economy took ' + mystr(t_end-t_start) + ' seconds.')



    ###############################################################################
    ########### REPRESENTATIVE AGENT ECONOMY WITH MARKOV STATE ####################
    ###############################################################################

    if do_RA:
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

            # Simulate the sticky representative agent Markov economy
            t_start = clock()
            StickyRAmarkovConsumer(UpdatePrb = Params.UpdatePrb)
            StickyRAmarkovConsumer.initializeSim()
            StickyRAmarkovConsumer.simulate()
            t_end = clock()
            print('Simulating the sticky representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

            # Make results for the sticky representative agent economy
            desc = 'Results for the sticky representative agent Markov economy'
            name = 'RAmarkovSticky'
            makeStickyEdataFile(StickyRAmarkovConsumer,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats)
            DeltaLogC_stdev = np.genfromtxt(results_dir + 'RAmarkovStickyResults.csv', delimiter=',')[3] # For use in frictionless spec

            # Simulate the frictionless representative agent Markov economy
            t_start = clock()
            StickyRAmarkovConsumer(UpdatePrb = 1.0)
            StickyRAmarkovConsumer.initializeSim()
            StickyRAmarkovConsumer.simulate()
            t_end = clock()
            print('Simulating the frictionless representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

            # Make results for the frictionless representative agent economy
            desc = 'Results for the frictionless representative agent Markov economy (update probability 1.0)'
            name = 'RAmarkovFrictionless'
            makeStickyEdataFile(StickyRAmarkovConsumer,ignore_periods,description=desc,filename=name,save_data=save_data,calc_micro_stats=calc_micro_stats,meas_err_base=DeltaLogC_stdev)


        if make_tables:
            # Process the coefficients, standard errors, etc into a LaTeX table
            t_start = clock()
            frictionless_panel = runRegressions('RAmarkovFrictionlessData',interval_size,False,False,True)
            frictionless_me_panel = runRegressions('RAmarkovFrictionlessData',interval_size,True,False,True)
            frictionless_long_panel = runRegressions('RAmarkovFrictionlessData',interval_size*interval_count,True,False,True)
            sticky_panel = runRegressions('RAmarkovStickyData',interval_size,False,True,True)
            sticky_me_panel = runRegressions('RAmarkovStickyData',interval_size,True,True,True)
            sticky_long_panel = runRegressions('RAmarkovStickyData',interval_size*interval_count,True,True,True)
            makeResultsTable('Aggregate Consumption Dynamics in RA Model',[frictionless_me_panel,sticky_me_panel],my_counts,'RepAgentMrkvSimReg','tRAsim')
            makeResultsTable('Aggregate Consumption Dynamics in RA Model',[frictionless_panel,sticky_panel],my_counts,'RepAgentMrkvSimRegNoMeasErr','tRAsimNoMeasErr')
            makeResultsTable('Aggregate Consumption Dynamics in RA Model',[frictionless_long_panel,sticky_long_panel],alt_counts,'RepAgentMrkvSimRegLong','tRAsimLong')
            t_end = clock()
            print('Running time series regressions for the representative agent Markov economy took ' + mystr(t_end-t_start) + ' seconds.')

    ###############################################################################
    ########### MAKE OTHER TABLES AND FIGURES #####################################
    ###############################################################################
    if make_tables:
        makeEquilibriumTable('Eqbm', ['SOEmarkovFrictionless','SOEmarkovSticky','DSGEmarkovFrictionless','DSGEmarkovSticky'],Params.init_SOE_consumer['CRRA'])
        makeParameterTable('Calibration', Params)

    if run_ucost_vs_pi:
        makeuCostVsPiFig('SOEuCostbyUpdatePrb')
        makeValueVsPiFig('SOEvVecByUpdatePrb')

    if run_value_vs_aggvar:
        makeValueVsAggShkVarFig('SOEvVecByPermShkAggVar')


