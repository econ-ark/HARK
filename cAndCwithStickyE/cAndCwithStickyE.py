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
import csv

import numpy as np
from time import clock
from copy import deepcopy
from StickyEmodel import StickyEconsumerType, StickyEmarkovConsumerType, StickyErepAgent, StickyEmarkovRepAgent
from ConsAggShockModel import SmallOpenEconomy, SmallOpenMarkovEconomy, CobbDouglasEconomy,CobbDouglasMarkovEconomy
from HARKutilities import plotFuncs, getLorenzShares
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.sandbox.regression.gmm as smsrg
import StickyEparams as Params
from MakeTable import makeResultsTable

ignore_periods = Params.ignore_periods # Number of simulated periods to ignore as a "burn-in" phase
interval_size = Params.interval_size   # Number of periods in each non-overlapping subsample
mystr = lambda number : "{:.3f}".format(number)

# Choose which models to run
do_SOE_simple  = True
do_SOE_markov  = True
do_DSGE_simple = True
do_DSGE_markov = True
do_RA_simple   = True
do_RA_markov   = True

# Choose whether to save data for use in Stata (as a tab-delimited text file)
save_data = False

def makeStickyEresults(Economy,description='',filename=None,save_data=False):
    '''
    Makes descriptive statistics and regression results for a model after it has
    been solved and simulated. Behaves slightly differently for heterogeneous agents
    vs representative agent models.
    
    Parameters
    ----------
    Economy : Market or AgentType
        A representation of the model economy.  For heterogeneous agents specifications,
        this will be an instance of a subclass of Market.  For representative agent
        specifications, this will be an instance of an AgentType subclass.
    description : str
        Description of the economy that is prepended on the output string.
    filename : str
        Name of the output log file, if any; .txt will be appended automatically.
    save_data : bool
        When True, save simulation data to filename + 'Data.txt' for use in Stata.
        
    Returns
    -------
    output_string : str
        Large string with descriptive statistics and regression results.  Also
        saved to a logfile if filename is not None.
    '''
    # Extract time series data from the economy
    if hasattr(Economy,'agents'): # If this is a heterogeneous agent specification...
        PlvlAgg_hist = np.cumprod(Economy.PermShkAggHist)
        pLvlAll_hist = np.concatenate([this_type.pLvlTrue_hist for this_type in Economy.agents],axis=1)
        aLvlAll_hist = np.concatenate([this_type.aLvlNow_hist for this_type in Economy.agents],axis=1)
        AlvlAgg_hist = np.mean(aLvlAll_hist,axis=1) # Level of aggregate assets
        AnrmAgg_hist = AlvlAgg_hist/PlvlAgg_hist # Normalized level of aggregate assets
        cLvlAll_hist = np.concatenate([this_type.cLvlNow_hist for this_type in Economy.agents],axis=1)
        ClvlAgg_hist = np.mean(cLvlAll_hist,axis=1) # Level of aggregate consumption
        CnrmAgg_hist = ClvlAgg_hist/PlvlAgg_hist # Normalized level of aggregate consumption
        yLvlAll_hist = np.concatenate([this_type.yLvlNow_hist for this_type in Economy.agents],axis=1)
        YlvlAgg_hist = np.mean(yLvlAll_hist,axis=1) # Level of aggregate income
        YnrmAgg_hist = YlvlAgg_hist/PlvlAgg_hist # Normalized level of aggregate income
        
        not_newborns = (np.concatenate([this_type.t_age_hist[(ignore_periods+1):,:] for this_type in Economy.agents],axis=1) > 1).flatten()
        Logc = np.log(cLvlAll_hist[ignore_periods:,:])
        DeltaLogc = (Logc[1:] - Logc[0:-1]).flatten()
        DeltaLogc_trimmed = DeltaLogc[not_newborns]
        Loga = np.log(aLvlAll_hist[ignore_periods:,:])
        DeltaLoga = (Loga[1:] - Loga[0:-1]).flatten()
        DeltaLoga_trimmed = DeltaLoga[not_newborns]
        Logp = np.log(pLvlAll_hist[ignore_periods:,:])
        DeltaLogp = (Logp[1:] - Logp[0:-1]).flatten()
        DeltaLogp_trimmed = DeltaLogp[not_newborns]
        Logy = np.log(yLvlAll_hist[ignore_periods:,:])
        Logy_trimmed = Logy
        Logy_trimmed[np.isinf(Logy)] = np.nan
        BigTheta_hist = Economy.TranShkAggHist
        UpdatePrb = Economy.agents[0].UpdatePrb
        
    else: # If this is a representative agent specification...
        PlvlAgg_hist = Economy.pLvlTrue_hist.flatten()
        ClvlAgg_hist = Economy.cLvlNow_hist.flatten()
        CnrmAgg_hist = ClvlAgg_hist/PlvlAgg_hist.flatten()
        YnrmAgg_hist = Economy.yNrmTrue_hist.flatten()
        YlvlAgg_hist = YnrmAgg_hist*PlvlAgg_hist.flatten()
        AlvlAgg_hist = Economy.aLvlNow_hist.flatten()
        AnrmAgg_hist = AlvlAgg_hist/PlvlAgg_hist.flatten()
        BigTheta_hist = Economy.TranShkNow_hist.flatten()
        UpdatePrb = Economy.UpdatePrb
        
    # Process aggregate data into forms used by regressions
    LogC = np.log(ClvlAgg_hist[ignore_periods:])
    LogA = np.log(AlvlAgg_hist[ignore_periods:])
    LogY = np.log(YlvlAgg_hist[ignore_periods:])
    DeltaLogC = LogC[1:] - LogC[0:-1]
    DeltaLogA = LogA[1:] - LogA[0:-1]
    DeltaLogY = LogY[1:] - LogY[0:-1]
    A = AnrmAgg_hist[(ignore_periods+1):] # This is a relabeling for the regression code
    BigTheta = BigTheta_hist[(ignore_periods+1):]
    
    # Add measurement error to LogC
    sigma_meas_err = np.std(DeltaLogC)/2.0
    np.random.seed(10)
    LogC_me = LogC + sigma_meas_err*np.random.normal(0.,1.,LogC.size)
    DeltaLogC_me = LogC_me[1:] - LogC_me[0:-1]
    
    # Determine how many subsample intervals to run (and initialize array of coefficients)
    N = DeltaLogC.size/interval_size
    CoeffsArray = np.zeros((N,8)) # Order: DeltaLogC_OLS, DeltaLogC_me_OLS, DeltaLogC_me_IV, DeltaLogY_IV, A_OLS, DeltaLogC_HR, DeltaLogY_HR, A_HR
    StdErrArray = np.zeros((N,8)) # Same order as above
    
    # Loop through subsample intervals, running various regressions
    for n in range(N):
        # Select the data subsample
        start = n*interval_size
        end = (n+1)*interval_size
        DeltaLogC_n = DeltaLogC[start:end]
        DeltaLogC_me_n = DeltaLogC_me[start:end]
        DeltaLogY_n = DeltaLogY[start:end]
        A_n = A[start:end]
        BigTheta_n = BigTheta[start:end]
        
        # Run OLS on log consumption (no measurement error)
        mod = sm.OLS(DeltaLogC_n[1:],sm.add_constant(DeltaLogC_n[0:-1]))
        res = mod.fit()
        CoeffsArray[n,0] = res._results.params[1]
        StdErrArray[n,0] = res._results.bse[1]
        
        # Run OLS on log consumption (with measurement error)
        if UpdatePrb < 1.0:
            mod = sm.OLS(DeltaLogC_me_n[1:],sm.add_constant(DeltaLogC_me_n[0:-1]))
            res = mod.fit()
            CoeffsArray[n,1] = res._results.params[1]
            StdErrArray[n,1] = res._results.bse[1]
        else:
            CoeffsArray[n,1] = np.nan
            StdErrArray[n,1] = np.nan
        
        # Define instruments for IV regressions
        temp = np.transpose(np.vstack([DeltaLogC_me_n[1:-3],DeltaLogC_me_n[:-4],DeltaLogY_n[1:-3],DeltaLogY_n[:-4],A_n[1:-3],A_n[:-4]]))
        instruments = sm.add_constant(temp) # With measurement error
        temp2 = np.transpose(np.array([DeltaLogC_n[1:-3],DeltaLogC_n[:-4],DeltaLogY_n[1:-3],DeltaLogY_n[:-4],A_n[1:-3],A_n[:-4]]))
        instruments2 = sm.add_constant(temp2) # No measurement error
        
        # Run IV on log consumption with measurement error (if sticky expectations)
        if UpdatePrb < 1.0:
            mod = smsrg.IV2SLS(DeltaLogC_me_n[4:],sm.add_constant(DeltaLogC_me_n[3:-1]),instruments)
            res = mod.fit()
            CoeffsArray[n,2] = res._results.params[1]
            StdErrArray[n,2] = res._results.bse[1]
        else:
            CoeffsArray[n,2] = np.nan
            StdErrArray[n,2] = np.nan
        
        # Run IV on log income (with measurement error if sticky expectations)
        if UpdatePrb < 1.0:
            mod = smsrg.IV2SLS(DeltaLogC_me_n[4:],sm.add_constant(DeltaLogY_n[4:]),instruments)
            res = mod.fit()
        else:
            mod = smsrg.IV2SLS(DeltaLogC_n[4:],sm.add_constant(DeltaLogY_n[4:]),instruments2)
            res = mod.fit()
        CoeffsArray[n,3] = res._results.params[1]
        StdErrArray[n,3] = res._results.bse[1]
        
        # Run IV on assets (with measurement error if sticky expectations)
        if UpdatePrb < 1.0:
            mod = smsrg.IV2SLS(DeltaLogC_me_n[4:],sm.add_constant(A_n[3:-1]),instruments)
            res = mod.fit()
        else:
            mod = smsrg.IV2SLS(DeltaLogC_n[4:],sm.add_constant(A_n[3:-1]),instruments2)
            res = mod.fit()
        CoeffsArray[n,4] = res._results.params[1]
        StdErrArray[n,4] = res._results.bse[1]
        
        # Run horserace IV (with measurement error if sticky expectations)
        if UpdatePrb < 1.0:
            regressors = sm.add_constant(np.transpose(np.array([DeltaLogC_me_n[3:-1],DeltaLogY_n[4:],A_n[3:-1]])))
            mod = smsrg.IV2SLS(DeltaLogC_me_n[4:],regressors,instruments)
            res = mod.fit()
        else:
            regressors = sm.add_constant(np.transpose(np.array([DeltaLogC_n[3:-1],DeltaLogY_n[4:],A_n[3:-1]])))
            mod = smsrg.IV2SLS(DeltaLogC_n[4:],regressors,instruments2)
            res = mod.fit()
        CoeffsArray[n,5] = res._results.params[1]
        CoeffsArray[n,6] = res._results.params[2]
        CoeffsArray[n,7] = res._results.params[3]
        StdErrArray[n,5] = res._results.bse[1]
        StdErrArray[n,6] = res._results.bse[2]
        StdErrArray[n,7] = res._results.bse[3]

    # Now run regressions on entire dataset
    # Run OLS on log consumption (no measurement error)
    mod = sm.OLS(DeltaLogC[1:],sm.add_constant(DeltaLogC[0:-1]))
    res1 = mod.fit()
    
    # Run OLS on log consumption (with measurement error)
    mod = sm.OLS(DeltaLogC_me[1:],sm.add_constant(DeltaLogC_me[0:-1]))
    res2 = mod.fit()
    
    # Define instruments for IV regressions
    temp = np.transpose(np.vstack([DeltaLogC_me[1:-3],DeltaLogC_me[:-4],DeltaLogY[1:-3],DeltaLogY[:-4],A[1:-3],A[:-4]]))
    instruments = sm.add_constant(temp)
    #instruments = sm.add_constant(np.transpose(np.array([DeltaLogC_me[1:-3],DeltaLogC_me[:-4],DeltaLogY[1:-3],DeltaLogY[:-4]])))
    
    # Run IV on log consumption (with measurement error)
    mod = smsrg.IV2SLS(DeltaLogC_me[4:],sm.add_constant(DeltaLogC_me[3:-1]),instruments)
    res3 = mod.fit()
    
    # Run IV on log income (with measurement error)
    mod = smsrg.IV2SLS(DeltaLogC_me[4:],sm.add_constant(DeltaLogY[4:]),instruments)
    res4 = mod.fit()
    
#    mod = sm.OLS(DeltaLogY[4:],instruments)
#    res_temp = mod.fit()
#    print(res_temp.summary())
#    
#    mod = sm.OLS(DeltaLogC[4:],instruments)
#    res_temp = mod.fit()
#    print(res_temp.summary())
#    
#    mod = sm.OLS(DeltaLogC,sm.add_constant(DeltaLogY))
#    res_temp = mod.fit()
#    print(res_temp.summary())
    
    # Run IV on assets (with measurement error)
    mod = smsrg.IV2SLS(DeltaLogC_me[4:],sm.add_constant(A[3:-1]),instruments)
    res5 = mod.fit()
    
    # Run horserace IV (with measurement error)
    regressors = sm.add_constant(np.transpose(np.array([DeltaLogC_me[3:-1],DeltaLogY[4:],A[3:-1]])))
    mod = smsrg.IV2SLS(DeltaLogC_me[4:],regressors,instruments)
    res6 = mod.fit()
    
    # Also report frictionless results with no measurement error
    temp2 = np.transpose(np.array([DeltaLogC[1:-3],DeltaLogC[:-4],DeltaLogY[1:-3],DeltaLogY[:-4],A[1:-3],A[:-4]]))
    instruments2 = sm.add_constant(temp2)
    
    # Run IV on log income (no measurement error)
    mod = smsrg.IV2SLS(DeltaLogC[4:],sm.add_constant(DeltaLogY[4:]),instruments2)
    res7 = mod.fit()
    
    # Run IV on assets (no measurement error)
    mod = smsrg.IV2SLS(DeltaLogC[4:],sm.add_constant(A[3:-1]),instruments2)
    res8 = mod.fit()
    
    # Run horserace IV (with no measurement error)
    regressors = sm.add_constant(np.transpose(np.array([DeltaLogC[3:-1],DeltaLogY[4:],A[3:-1]])))
    mod = smsrg.IV2SLS(DeltaLogC[4:],regressors,instruments2)
    res9 = mod.fit()
    
    # Make and return the output string, beginning with descriptive statistics
    output_string = description + '\n\n\n'
    output_string += 'Average aggregate asset-to-productivity ratio = ' + str(np.mean(AnrmAgg_hist[ignore_periods:])) + '\n'
    output_string += 'Average aggregate consumption-to-productivity ratio = ' + str(np.mean(CnrmAgg_hist[ignore_periods:])) + '\n'
    output_string += 'Stdev of log aggregate asset-to-productivity ratio = ' + str(np.std(np.log(AnrmAgg_hist[ignore_periods:]))) + '\n'
    output_string += 'Stdev of change in log aggregate consumption level = ' + str(np.std(DeltaLogC)) + '\n'
    output_string += 'Stdev of change in log aggregate output level = ' + str(np.std(DeltaLogY)) + '\n'
    output_string += 'Stdev of change in log aggregate assets level = ' + str(np.std(DeltaLogA)) + '\n'
    if hasattr(Economy,'agents'): # This block only runs for heterogeneous agents specifications
        output_string += 'Cross section stdev of log individual assets = ' + str(np.mean(np.std(Loga,axis=1))) + '\n'
        output_string += 'Cross section stdev of log individual consumption = ' + str(np.mean(np.std(Logc,axis=1))) + '\n'
        output_string += 'Cross section stdev of log individual productivity = ' + str(np.mean(np.std(Logp,axis=1))) + '\n'
        output_string += 'Cross section stdev of log individual non-zero income = ' + str(np.mean(np.std(Logy_trimmed,axis=1))) + '\n'
        output_string += 'Cross section stdev of change in log individual assets = ' + str(np.std(DeltaLoga_trimmed)) + '\n'
        output_string += 'Cross section stdev of change in log individual consumption = ' + str(np.std(DeltaLogc_trimmed)) + '\n'
        output_string += 'Cross section stdev of change in log individual productivity = ' + str(np.std(DeltaLogp_trimmed)) + '\n'
    output_string += '\n\n'    
        
    # Add regression results to the output string
    output_string += str(res1.summary(yname='DeltaLogC_t',xname=['constant','DeltaLogC_tm1'],title='OLS on log consumption (no measurement error)')) + '\n\n\n'
    output_string += str(res2.summary(yname='DeltaLogC_t',xname=['constant','DeltaLogC_tm1'],title='OLS on log consumption (with measurement error)')) + '\n\n\n'
    output_string += str(res3.summary(yname='DeltaLogC_t',xname=['constant','DeltaLogC_tm1'],title='IV on log consumption (with measurement error)')) + '\n\n\n'
    output_string += str(res4.summary(yname='DeltaLogC_t',xname=['constant','DeltaLogY_t'],title='IV on log income (with measurement error)')) + '\n\n\n'
    output_string += str(res5.summary(yname='DeltaLogC_t',xname=['constant','A_tm1'],title='IV on asset ratio (with measurement error)')) + '\n\n\n'
    output_string += str(res6.summary(yname='DeltaLogC_t',xname=['constant','DeltaLogC_tm1','DeltaLogY_t','A_tm1'],title='Horserace IV (with measurement error)')) + '\n\n\n'
    output_string += str(res7.summary(yname='DeltaLogC_t',xname=['constant','DeltaLogY_t'],title='IV on log income (no measurement error)')) + '\n\n\n'
    output_string += str(res8.summary(yname='DeltaLogC_t',xname=['constant','A_tm1'],title='IV on asset ratio (no measurement error)')) + '\n\n\n'
    output_string += str(res9.summary(yname='DeltaLogC_t',xname=['constant','DeltaLogC_tm1','DeltaLogY_t','A_tm1'],title='Horserace IV (no measurement error)')) + '\n\n\n'
     
    # Save the results to a logfile if requested
    if filename is not None:
        with open('./Results/' + filename + '.txt','w') as f:
            f.write(output_string)
            f.close()
            
        if save_data:
            DataArray = (np.vstack((np.arange(DeltaLogC.size),DeltaLogC_me,DeltaLogC,DeltaLogY,A))).transpose()
            VarNames = ['time_period','DeltaLogC_me','DeltaLogC','DeltaLogY','A']
            if hasattr(Economy,'MrkvNow'):
                Mrkv_hist = Economy.MrkvNow_hist[ignore_periods+1:]
                DataArray = np.hstack((DataArray,np.reshape(Mrkv_hist,(Mrkv_hist.size,1))))
                VarNames.append('MrkvState')
            with open('./Results/' + filename + 'Data.txt.','wb') as f:
                my_writer = csv.writer(f, delimiter = '\t')
                my_writer.writerow(VarNames)
                for i in range(DataArray.shape[0]):
                    my_writer.writerow(DataArray[i,:])
                f.close()
    
    t_stat_array = CoeffsArray/StdErrArray
    if UpdatePrb < 1.0:
        C_successes_95 = np.sum(t_stat_array[:,5] > 1.96)
        C_successes_90 = np.sum(t_stat_array[:,5] > 1.645)
        N_out = [C_successes_95,C_successes_90,N]
    else:
        N_out = [0,0,0]
    print(np.mean(t_stat_array,axis=0))
    
    return output_string, np.mean(CoeffsArray,axis=0), np.mean(StdErrArray,axis=0), N_out # Return output string and subsample regression summaries


def evalLorenzDistance(Economy):
    '''
    Calculates the Lorenz distance and the wealth level difference bewtween a
    given economy and some specified targets.
    
    Parameters
    ----------
    Economy : Market
        Economy with one or more agent types (with simulated data).
        
    Returns
    -------
    wealth_difference : float
        Difference between economy and target aggregate wealth level.
    lorenz_distance : float
        Distance between economy and targets in terms of Lorenz.
    '''
    target_wealth = 10.26
    pctiles = [0.2,0.4,0.6,0.8]
    target_lorenz = np.array([-0.002, 0.01, 0.053,0.171])
    A = np.concatenate([Economy.agents[i].aLvlNow for i in range(len(Economy.agents))])
    sim_lorenz = getLorenzShares(A,percentiles=pctiles)
    lorenz_distance = np.sqrt(np.sum((sim_lorenz - target_lorenz)**2))
    wealth_difference = Economy.KtoYnow - target_wealth
    return wealth_difference, lorenz_distance


# Run models and save output if this module is called from main
if __name__ == '__main__':
    ###############################################################################
    ################# SMALL OPEN ECONOMY ##########################################
    ###############################################################################
    
    if do_SOE_simple:
        # Make a small open economy and the consumers who live in it
        StickySOEbaseType = StickyEconsumerType(**Params.init_SOE_consumer)
        StickySOEbaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age']
        StickySOEconsumers = []
        for n in range(Params.TypeCount):
            StickySOEconsumers.append(deepcopy(StickySOEbaseType))
            StickySOEconsumers[-1].seed = n
            StickySOEconsumers[-1].DiscFac = Params.DiscFacSet[n]
        StickySOEconomy = SmallOpenEconomy(agents=StickySOEconsumers, **Params.init_SOE_market)
        StickySOEconomy.makeAggShkHist()
        for n in range(Params.TypeCount):
            StickySOEconsumers[n].getEconomyData(StickySOEconomy)
        
        # Solve the small open economy and display some output
        t_start = clock()
        StickySOEconomy.solveAgents()
        t_end = clock()
        print('Solving the small open economy took ' + str(t_end-t_start) + ' seconds.')
        
        print('Consumption function one type in the small open economy:')
        cFunc = lambda m : StickySOEconsumers[0].solution[0].cFunc(m,np.ones_like(m))
        plotFuncs(cFunc,0.0,20.0)
        
        # Simulate the frictionless small open economy
        t_start = clock()
        for agent in StickySOEconomy.agents:
            agent(UpdatePrb = 1.0)
        StickySOEconomy.makeHistory()
        t_end = clock()
        print('Simulating the frictionless small open economy took ' + mystr(t_end-t_start) + ' seconds.')
        
        # Make results for the frictionless small open economy
        desc = 'Results for the frictionless small open economy'
        name = 'SOEsimpleFrictionlessResults'
        ResultsStringF, CoeffsF, StdErrsF, trash = makeStickyEresults(StickySOEconomy,description=desc,filename=name,save_data=save_data)
        
        # Simulate the sticky small open economy
        t_start = clock()
        for agent in StickySOEconomy.agents:
            agent(UpdatePrb = Params.UpdatePrb)
        StickySOEconomy.makeHistory()
        t_end = clock()
        print('Simulating the sticky small open economy took ' + mystr(t_end-t_start) + ' seconds.')
        
        # Make results for the sticky small open economy
        desc = 'Results for the sticky small open economy with update probability ' + mystr(Params.UpdatePrb)
        name = 'SOEsimpleStickyResults'
        ResultsStringS, CoeffsS, StdErrsS, Counts = makeStickyEresults(StickySOEconomy,description=desc,filename=name,save_data=save_data)
        
        # Process the coefficients, standard errors, etc into a LaTeX table
        Coeffs = np.zeros(14)
        Coeffs[0:6] = CoeffsF[np.array([0,3,4,5,6,7])]
        Coeffs[6:14] = CoeffsS
        StdErrs = np.zeros(14)
        StdErrs[0:6] = CoeffsS[np.array([0,3,4,5,6,7])]
        StdErrs[6:14] = StdErrsS
        Rsq = np.zeros(10) + np.nan
        Pvals = np.zeros(10) + np.nan
        OID = np.zeros(10) + np.nan
        makeResultsTable(Coeffs,StdErrs,Rsq,Pvals,OID,Counts,'Aggregate Consumption Dynamics in Small Open Economy','SOEsimReg')
    
    
    ###############################################################################
    ########## SMALL OPEN ECONOMY WITH MACROECONOMIC MARKOV STATE##################
    ###############################################################################
    
    if do_SOE_markov:
        # Make consumer types to inhabit the small open Markov economy
        StickySOEmarkovBaseType = StickyEmarkovConsumerType(**Params.init_SOE_mrkv_consumer)
        StickySOEmarkovBaseType.IncomeDstn[0] = Params.StateCount*[StickySOEmarkovBaseType.IncomeDstn[0]]
        StickySOEmarkovBaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age']
        StickySOEmarkovConsumers = []
        for n in range(Params.TypeCount):
            StickySOEmarkovConsumers.append(deepcopy(StickySOEmarkovBaseType))
            StickySOEmarkovConsumers[-1].seed = n
            StickySOEmarkovConsumers[-1].DiscFac = Params.DiscFacSet[n]
        
        # Make a small open economy for the agents
        StickySOmarkovEconomy = SmallOpenMarkovEconomy(agents=StickySOEmarkovConsumers, **Params.init_SOE_mrkv_market)
        StickySOmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
        for n in range(Params.TypeCount):
            StickySOEmarkovConsumers[n].getEconomyData(StickySOmarkovEconomy) # Have the consumers inherit relevant objects from the economy
        
        # Solve the small open Markov model
        t_start = clock()
        StickySOmarkovEconomy.solveAgents()
        t_end = clock()
        print('Solving the small open Markov economy took ' + str(t_end-t_start) + ' seconds.')
        
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
        desc = 'Results for the frictionless small open Markov economy'
        name = 'SOEmarkovFrictionlessResults'
        ResultsStringF, CoeffsF, StdErrsF, trash = makeStickyEresults(StickySOmarkovEconomy,description=desc,filename=name,save_data=save_data)
        
        # Simulate the frictionless small open Markov economy
        t_start = clock()
        for agent in StickySOmarkovEconomy.agents:
            agent(UpdatePrb = Params.UpdatePrb)
        StickySOmarkovEconomy.makeHistory()
        t_end = clock()
        print('Simulating the sticky small open Markov economy took ' + mystr(t_end-t_start) + ' seconds.')
        
        # Make results for the sticky small open Markov economy
        desc = 'Results for the sticky small open Markov economy with update probability ' + mystr(Params.UpdatePrb)
        name = 'SOEmarkovStickyResults'
        ResultsStringS, CoeffsS, StdErrsS, Counts = makeStickyEresults(StickySOmarkovEconomy,description=desc,filename=name,save_data=save_data)
        
        # Process the coefficients, standard errors, etc into a LaTeX table
        Coeffs = np.zeros(14)
        Coeffs[0:6] = CoeffsF[np.array([0,3,4,5,6,7])]
        Coeffs[6:14] = CoeffsS
        StdErrs = np.zeros(14)
        StdErrs[0:6] = CoeffsS[np.array([0,3,4,5,6,7])]
        StdErrs[6:14] = StdErrsS
        Rsq = np.zeros(10) + np.nan
        Pvals = np.zeros(10) + np.nan
        OID = np.zeros(10) + np.nan
        makeResultsTable(Coeffs,StdErrs,Rsq,Pvals,OID,Counts,'Aggregate Consumption Dynamics in Small Open Markov Economy (' + str(Params.StateCount) + ' states)','SOEmrkvSimReg')
        
    
    ###############################################################################
    ################# COBB-DOUGLAS ECONOMY ########################################
    ###############################################################################
    
    if do_DSGE_simple:
        # Make consumers who will live in a Cobb-Douglas economy
        StickyDSGEbaseType = StickyEconsumerType(**Params.init_DSGE_consumer)
        StickyDSGEbaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age']
        StickyDSGEconsumers = []
        for n in range(Params.TypeCount):
            StickyDSGEconsumers.append(deepcopy(StickyDSGEbaseType))
            StickyDSGEconsumers[-1].seed = n
            StickyDSGEconsumers[-1].DiscFac = Params.DiscFacSet[n]
            
        # Make a Cobb-Douglas economy and put the agents in it
        StickyDSGEeconomy = CobbDouglasEconomy(agents=StickyDSGEconsumers,**Params.init_DSGE_market)
        StickyDSGEeconomy.makeAggShkHist()
        for n in range(Params.TypeCount):
            StickyDSGEconsumers[n].getEconomyData(StickyDSGEeconomy)
            StickyDSGEconsumers[n](UpdatePrb = 1.0)
        
        # Solve the frictionless HA-DSGE model
        t_start = clock()
        StickyDSGEeconomy.solve()
        t_end = clock()
        print('Solving the frictionless Cobb-Douglas economy took ' + str(t_end-t_start) + ' seconds.')
        
        # Plot the consumption function
        print('Consumption function for the frictionless Cobb-Douglas economy:')
        m = np.linspace(0.,20.,300)
        for M in StickyDSGEconsumers[0].Mgrid:
            c = StickyDSGEconsumers[0].solution[0].cFunc(m,M*np.ones_like(m))
            plt.plot(m,c)
        plt.show()
        
        # Make results for the frictionless Cobb-Douglas economy
        desc = 'Results for the frictionless Cobb-Douglas economy'
        name = 'DSGEsimpleFrictionlessResults'
        ResultsStringF, CoeffsF, StdErrsF, trash = makeStickyEresults(StickyDSGEeconomy,description=desc,filename=name,save_data=save_data)
        
        # Solve the sticky HA-DSGE model
        for agent in StickyDSGEeconomy.agents:
            agent(UpdatePrb = Params.UpdatePrb)
        t_start = clock()
        StickyDSGEeconomy.solve()
        t_end = clock()
        print('Solving the sticky Cobb-Douglas economy took ' + str(t_end-t_start) + ' seconds.')
        
        # Plot the consumption function
        print('Consumption function for the sticky Cobb-Douglas economy:')
        m = np.linspace(0.,20.,300)
        for M in StickyDSGEconsumers[0].Mgrid:
            c = StickyDSGEconsumers[0].solution[0].cFunc(m,M*np.ones_like(m))
            plt.plot(m,c)
        plt.show()
        
        # Make results for the sticky Cobb-Douglas economy
        desc = 'Results for the sticky Cobb-Douglas economy with update probability ' + mystr(Params.UpdatePrb)
        name = 'DSGEsimpleStickyResults'
        ResultsStringS, CoeffsS, StdErrsS, Counts = makeStickyEresults(StickyDSGEeconomy,description=desc,filename=name,save_data=save_data)
        
        # Process the coefficients, standard errors, etc into a LaTeX table
        Coeffs = np.zeros(14)
        Coeffs[0:6] = CoeffsF[np.array([0,3,4,5,6,7])]
        Coeffs[6:14] = CoeffsS
        StdErrs = np.zeros(14)
        StdErrs[0:6] = CoeffsS[np.array([0,3,4,5,6,7])]
        StdErrs[6:14] = StdErrsS
        Rsq = np.zeros(10) + np.nan
        Pvals = np.zeros(10) + np.nan
        OID = np.zeros(10) + np.nan
        makeResultsTable(Coeffs,StdErrs,Rsq,Pvals,OID,Counts,'Aggregate Consumption Dynamics in HA-DSGE Economy','DSGEsimReg')
    
    
    ###############################################################################
    ########## COBB-DOUGLAS ECONOMY WITH MACROECONOMIC MARKOV STATE ###############
    ###############################################################################
    
    if do_DSGE_markov:
        # Make consumers who will live in the Cobb-Douglas Markov economy
        StickyDSGEmarkovBaseType = StickyEmarkovConsumerType(**Params.init_DSGE_mrkv_consumer)
        StickyDSGEmarkovBaseType.IncomeDstn[0] = Params.StateCount*[StickyDSGEmarkovBaseType.IncomeDstn[0]]
        StickyDSGEmarkovBaseType.track_vars = ['aLvlNow','cLvlNow','yLvlNow','pLvlTrue','t_age']
        StickyDSGEmarkovConsumers = []
        for n in range(Params.TypeCount):
            StickyDSGEmarkovConsumers.append(deepcopy(StickyDSGEmarkovBaseType))
            StickyDSGEmarkovConsumers[-1].seed = n
            StickyDSGEmarkovConsumers[-1].DiscFac = Params.DiscFacSet[n]
        
        # Make a Cobb-Douglas economy for the agents
        StickyDSGEmarkovEconomy = CobbDouglasMarkovEconomy(agents = StickyDSGEmarkovConsumers,**Params.init_DSGE_mrkv_market)
        StickyDSGEmarkovEconomy.makeAggShkHist() # Simulate a history of aggregate shocks
        for n in range(Params.TypeCount):
            StickyDSGEmarkovConsumers[n].getEconomyData(StickyDSGEmarkovEconomy) # Have the consumers inherit relevant objects from the economy
            StickyDSGEmarkovConsumers[n](UpdatePrb = 1.0)
        
        # Solve the frictionless heterogeneous agent DSGE model
        t_start = clock()
        StickyDSGEmarkovEconomy.solve()
        t_end = clock()
        print('Solving the frictionless Cobb-Douglas Markov economy took ' + str(t_end-t_start) + ' seconds.')
        
        print('Displaying the consumption functions for the Cobb-Douglas Markov economy would be too much.')
        
        # Make results for the Cobb-Douglas Markov economy
        desc = 'Results for the frictionless Cobb-Douglas Markov economy'
        name = 'DSGEmarkovFrictionlessResults'
        ResultsStringF, CoeffsF, StdErrsF, trash = makeStickyEresults(StickyDSGEmarkovEconomy,description=desc,filename=name,save_data=save_data)
        
        # Solve the sticky heterogeneous agent DSGE model
        for agent in StickyDSGEmarkovEconomy.agents:
            agent(UpdatePrb = Params.UpdatePrb)
        t_start = clock()
        StickyDSGEmarkovEconomy.solve()
        t_end = clock()
        print('Solving the sticky Cobb-Douglas Markov economy took ' + str(t_end-t_start) + ' seconds.')
        
        print('Displaying the consumption functions for the Cobb-Douglas Markov economy would be too much.')
        
        # Make results for the Cobb-Douglas Markov economy
        desc = 'Results for the sticky Cobb-Douglas Markov economy with update probability ' + mystr(Params.UpdatePrb)
        name = 'DSGEmarkovStickyResults'
        ResultsStringS, CoeffsS, StdErrsS, Counts = makeStickyEresults(StickyDSGEmarkovEconomy,description=desc,filename=name,save_data=save_data)
        
        # Process the coefficients, standard errors, etc into a LaTeX table
        Coeffs = np.zeros(14)
        Coeffs[0:6] = CoeffsF[np.array([0,3,4,5,6,7])]
        Coeffs[6:14] = CoeffsS
        StdErrs = np.zeros(14)
        StdErrs[0:6] = CoeffsS[np.array([0,3,4,5,6,7])]
        StdErrs[6:14] = StdErrsS
        Rsq = np.zeros(10) + np.nan
        Pvals = np.zeros(10) + np.nan
        OID = np.zeros(10) + np.nan
        makeResultsTable(Coeffs,StdErrs,Rsq,Pvals,OID,Counts,'Aggregate Consumption Dynamics in HA-DSGE Markov Economy (' + str(Params.StateCount) + ' states)','DSGEmrkvSimReg')
        
    
    ###############################################################################
    ################# REPRESENTATIVE AGENT ECONOMY ################################
    ###############################################################################
    
    if do_RA_simple:
        # Make a representative agent consumer, then solve and simulate the model
        StickyRAconsumer = StickyErepAgent(**Params.init_RA_consumer)
        StickyRAconsumer.track_vars = ['cLvlNow','yNrmTrue','aLvlNow','pLvlTrue','TranShkNow']
        
        # Solve the representative agent's problem        
        t_start = clock()
        StickyRAconsumer.solve()
        t_end = clock()
        print('Solving the representative agent economy took ' + str(t_end-t_start) + ' seconds.')
        
        print('Consumption function for the representative agent:')
        plotFuncs(StickyRAconsumer.solution[0].cFunc,0,50)
        
        # Simulate the representative agent with frictionless expectations
        t_start = clock()
        StickyRAconsumer(UpdatePrb = 1.0)
        StickyRAconsumer.initializeSim()
        StickyRAconsumer.simulate()
        t_end = clock()
        print('Simulating the frictionless representative agent economy took ' + str(t_end-t_start) + ' seconds.')
        
        # Make results for the frictionless representative agent economy
        desc = 'Results for the frictionless representative agent economy'
        name = 'RAsimpleFrictionlessResults'
        ResultsStringF, CoeffsF, StdErrsF, trash = makeStickyEresults(StickyRAconsumer,description=desc,filename=name,save_data=save_data)
        
        # Simulate the representative agent with sticky expectations
        t_start = clock()
        StickyRAconsumer(UpdatePrb = Params.UpdatePrb)
        StickyRAconsumer.initializeSim()
        StickyRAconsumer.simulate()
        t_end = clock()
        print('Simulating the sticky representative agent economy took ' + str(t_end-t_start) + ' seconds.')
        
        # Make results for the sticky representative agent economy
        desc = 'Results for the sticky representative agent economy with update probability ' + mystr(Params.UpdatePrb)
        name = 'RAsimpleStickyResults'
        ResultsStringS, CoeffsS, StdErrsS, Counts = makeStickyEresults(StickyRAconsumer,description=desc,filename=name,save_data=save_data)
        
        # Process the coefficients, standard errors, etc into a LaTeX table
        Coeffs = np.zeros(14)
        Coeffs[0:6] = CoeffsF[np.array([0,3,4,5,6,7])]
        Coeffs[6:14] = CoeffsS
        StdErrs = np.zeros(14)
        StdErrs[0:6] = CoeffsS[np.array([0,3,4,5,6,7])]
        StdErrs[6:14] = StdErrsS
        Rsq = np.zeros(10) + np.nan
        Pvals = np.zeros(10) + np.nan
        OID = np.zeros(10) + np.nan
        makeResultsTable(Coeffs,StdErrs,Rsq,Pvals,OID,Counts,'Aggregate Consumption Dynamics in Rep Agent Economy','RepAgentSimReg')
    
    ###############################################################################
    ########### REPRESENTATIVE AGENT ECONOMY WITH MARKOV STATE ####################
    ###############################################################################
    
    if do_RA_markov:
        # Make a representative agent consumer, then solve and simulate the model
        StickyRAmarkovConsumer = StickyEmarkovRepAgent(**Params.init_RA_mrkv_consumer)
        StickyRAmarkovConsumer.IncomeDstn[0] = Params.StateCount*[StickyRAmarkovConsumer.IncomeDstn[0]]
        StickyRAmarkovConsumer.track_vars = ['cLvlNow','yNrmTrue','aLvlNow','pLvlTrue','TranShkNow']
        
        # Solve the representative agent Markov economy
        t_start = clock()
        StickyRAmarkovConsumer.solve()
        t_end = clock()
        print('Solving the representative agent Markov economy took ' + str(t_end-t_start) + ' seconds.')
        
        print('Consumption functions for the Markov representative agent:')
        plotFuncs(StickyRAmarkovConsumer.solution[0].cFunc,0,50)
        
        # Simulate the frictionless representative agent MarkovEconomy
        t_start = clock()
        StickyRAmarkovConsumer(UpdatePrb = 1.0)
        StickyRAmarkovConsumer.initializeSim()
        StickyRAmarkovConsumer.simulate()
        t_end = clock()
        print('Simulating the frictionless representative agent Markov economy took ' + str(t_end-t_start) + ' seconds.')
        
        # Make results for the frictionless Markov representative agent economy
        desc = 'Results for the frictionless Markov representative agent economy'
        name = 'RAmarkovFrictionlessResults'
        ResultsStringF, CoeffsF, StdErrsF, trash = makeStickyEresults(StickyRAmarkovConsumer,description=desc,filename=name,save_data=save_data)
        
        # Simulate the sticky representative agent MarkovEconomy
        t_start = clock()
        StickyRAmarkovConsumer(UpdatePrb = Params.UpdatePrb)
        StickyRAmarkovConsumer.initializeSim()
        StickyRAmarkovConsumer.simulate()
        t_end = clock()
        print('Simulating the sticky representative agent Markov economy took ' + str(t_end-t_start) + ' seconds.')
        
        # Make results for the sticky Markov representative agent economy
        desc = 'Results for the sticky Markov representative agent economy with update probability ' + mystr(Params.UpdatePrb)
        name = 'RAmarkovStickyResults'
        ResultsStringS, CoeffsS, StdErrsS, Counts = makeStickyEresults(StickyRAmarkovConsumer,description=desc,filename=name,save_data=save_data)
        
        # Process the coefficients, standard errors, etc into a LaTeX table
        Coeffs = np.zeros(14)
        Coeffs[0:6] = CoeffsF[np.array([0,3,4,5,6,7])]
        Coeffs[6:14] = CoeffsS
        StdErrs = np.zeros(14)
        StdErrs[0:6] = CoeffsS[np.array([0,3,4,5,6,7])]
        StdErrs[6:14] = StdErrsS
        Rsq = np.zeros(10) + np.nan
        Pvals = np.zeros(10) + np.nan
        OID = np.zeros(10) + np.nan
        makeResultsTable(Coeffs,StdErrs,Rsq,Pvals,OID,Counts,'Aggregate Consumption Dynamics in Rep Agent Markov Economy (' + str(Params.StateCount) + ' states)','RepAgentMrkvSimReg')
        