"""
This module holds some data tools used in the cAndCwithStickyE project.
"""

import os
import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.sandbox.regression.gmm as smsrg
import matplotlib.pyplot as plt
from copy import deepcopy
import subprocess
from HARK.utilities import CRRAutility
from HARKinterpolation import LinearInterp
from StickyEparams import results_dir, tables_dir, figures_dir, UpdatePrb, PermShkAggVar
UpdatePrbBase = UpdatePrb
PermShkAggVarBase = PermShkAggVar

def mystr1(number):
    if not np.isnan(number):
        out = "{:.3f}".format(number)
    else:
        out = ''
    return out

def mystr2(number):
    if not np.isnan(number):
        out = "{:1.2f}".format(number*10000) + '\\text{e--4}'
    else:
        out = ''
    return out

def mystr3(number):
    if not np.isnan(number):
        out = "{:1.2f}".format(number*1000000) + '\\text{e--6}'
    else:
        out = ''
    return out


def makeStickyEdataFile(Economy,ignore_periods,description='',filename=None,save_data=False,calc_micro_stats=True,meas_err_base=None):
    '''
    Makes descriptive statistics and macroeconomic data file. Behaves slightly
    differently for heterogeneous agents vs representative agent models.

    Parameters
    ----------
    Economy : Market or AgentType
        A representation of the model economy.  For heterogeneous agents specifications,
        this will be an instance of a subclass of Market.  For representative agent
        specifications, this will be an instance of an AgentType subclass.
    ignore_periods : int
        Number of periods at the start of the simulation to throw out.
    description : str
        Description of the economy that is prepended on the output string.
    filename : str
        Name of the output log file, if any; .txt will be appended automatically.
    save_data : bool
        When True, save simulation data to filename + 'Data.txt' for use in Stata.
    calc_micro_stats : bool
        When True, calculate microeconomic statistics like in Table 2 of the
        paper draft.
    meas_err_base : float or None
        Base value of measurement error standard deviation, which will be adjusted.
        When None (default), value is calculated as stdev(DeltaLogC).

    Returns
    -------
    None
    '''
    # Extract time series data from the economy
    if hasattr(Economy,'agents'): # If this is a heterogeneous agent specification...
        if len(Economy.agents) > 1:
            pLvlAll_hist = np.concatenate([this_type.pLvlTrue_hist for this_type in Economy.agents],axis=1)
            aLvlAll_hist = np.concatenate([this_type.aLvlNow_hist for this_type in Economy.agents],axis=1)
            cLvlAll_hist = np.concatenate([this_type.cLvlNow_hist for this_type in Economy.agents],axis=1)
            yLvlAll_hist = np.concatenate([this_type.yLvlNow_hist for this_type in Economy.agents],axis=1)
        else: # Don't duplicate the data unless necessary (with one type, concatenating is useless)
            pLvlAll_hist = Economy.agents[0].pLvlTrue_hist
            aLvlAll_hist = Economy.agents[0].aLvlNow_hist
            cLvlAll_hist = Economy.agents[0].cLvlNow_hist
            yLvlAll_hist = Economy.agents[0].yLvlNow_hist
        # PermShkAggHist needs to be shifted one period forward
        PlvlAgg_hist = np.cumprod(np.concatenate(([1.0],Economy.PermShkAggHist[:-1]),axis=0))
        AlvlAgg_hist = np.mean(aLvlAll_hist,axis=1) # Level of aggregate assets
        AnrmAgg_hist = AlvlAgg_hist/PlvlAgg_hist # Normalized level of aggregate assets
        ClvlAgg_hist = np.mean(cLvlAll_hist,axis=1) # Level of aggregate consumption
        CnrmAgg_hist = ClvlAgg_hist/PlvlAgg_hist # Normalized level of aggregate consumption
        YlvlAgg_hist = np.mean(yLvlAll_hist,axis=1) # Level of aggregate income
        YnrmAgg_hist = YlvlAgg_hist/PlvlAgg_hist # Normalized level of aggregate income

        if calc_micro_stats: # Only calculate stats if requested.  This is a memory hog with many simulated periods
            micro_stat_periods = int((Economy.agents[0].T_sim-ignore_periods)*0.1)
            not_newborns = (np.concatenate([this_type.t_age_hist[(ignore_periods+1):(ignore_periods+micro_stat_periods),:] for this_type in Economy.agents],axis=1) > 1).flatten()
            Logc = np.log(cLvlAll_hist[ignore_periods:(ignore_periods+micro_stat_periods),:])
            DeltaLogc = (Logc[1:] - Logc[0:-1]).flatten()
            DeltaLogc_trimmed = DeltaLogc[not_newborns]
            Loga = np.log(aLvlAll_hist[ignore_periods:(ignore_periods+micro_stat_periods),:])
            DeltaLoga = (Loga[1:] - Loga[0:-1]).flatten()
            DeltaLoga_trimmed = DeltaLoga[not_newborns]
            Logp = np.log(pLvlAll_hist[ignore_periods:(ignore_periods+micro_stat_periods),:])
            DeltaLogp = (Logp[1:] - Logp[0:-1]).flatten()
            DeltaLogp_trimmed = DeltaLogp[not_newborns]
            Logy = np.log(yLvlAll_hist[ignore_periods:(ignore_periods+micro_stat_periods),:])
            Logy_trimmed = Logy
            Logy_trimmed[np.isinf(Logy)] = np.nan
            birth_events = np.concatenate([this_type.t_age_hist == 1 for this_type in Economy.agents],axis=1)
            vBirth = calcValueAtBirth(cLvlAll_hist[ignore_periods:,:],birth_events[ignore_periods:,:],PlvlAgg_hist[ignore_periods:],Economy.MrkvNow_hist[ignore_periods:],Economy.agents[0].DiscFac,Economy.agents[0].CRRA)

        BigTheta_hist = Economy.TranShkAggHist
        if hasattr(Economy,'MrkvNow'):
            Mrkv_hist = Economy.MrkvNow_hist
            if ~hasattr(Economy,'Rfree'): # If this is a markov DSGE specification...
                # Find the expected interest rate - approximate by assuming growth = expected growth
                ExpectedGrowth_hist = Economy.PermGroFacAgg[Mrkv_hist]
                ExpectedKLRatio_hist = AnrmAgg_hist/ExpectedGrowth_hist
                ExpectedR_hist = Economy.Rfunc(ExpectedKLRatio_hist)

    else: # If this is a representative agent specification...
        PlvlAgg_hist = Economy.pLvlTrue_hist.flatten()
        ClvlAgg_hist = Economy.cLvlNow_hist.flatten()
        CnrmAgg_hist = ClvlAgg_hist/PlvlAgg_hist.flatten()
        YnrmAgg_hist = Economy.yNrmTrue_hist.flatten()
        YlvlAgg_hist = YnrmAgg_hist*PlvlAgg_hist.flatten()
        AlvlAgg_hist = Economy.aLvlNow_hist.flatten()
        AnrmAgg_hist = AlvlAgg_hist/PlvlAgg_hist.flatten()
        BigTheta_hist = Economy.TranShkNow_hist.flatten()
        if hasattr(Economy,'MrkvNow'):
            Mrkv_hist = Economy.MrkvNow_hist

    # Process aggregate data into forms used by regressions
    LogC = np.log(ClvlAgg_hist[ignore_periods:])
    LogA = np.log(AlvlAgg_hist[ignore_periods:])
    LogY = np.log(YlvlAgg_hist[ignore_periods:])
    DeltaLogC = LogC[1:] - LogC[0:-1]
    DeltaLogA = LogA[1:] - LogA[0:-1]
    DeltaLogY = LogY[1:] - LogY[0:-1]
    A = AnrmAgg_hist[(ignore_periods+1):] # This is a relabeling for the regression code
    BigTheta = BigTheta_hist[(ignore_periods+1):]
    if hasattr(Economy,'MrkvNow'):
        Mrkv = Mrkv_hist[(ignore_periods+1):] # This is a relabeling for the regression code
        if ~hasattr(Economy,'Rfree') and hasattr(Economy,'agents'): # If this is a markov DSGE specification...
            R = ExpectedR_hist[(ignore_periods+1):]
    Delta8LogC = (np.log(ClvlAgg_hist[8:]) - np.log(ClvlAgg_hist[:-8]))[(ignore_periods-7):]
    Delta8LogY = (np.log(YlvlAgg_hist[8:]) - np.log(YlvlAgg_hist[:-8]))[(ignore_periods-7):]

    # Add measurement error to LogC
    if meas_err_base is None:
        meas_err_base = np.std(DeltaLogC)
    sigma_meas_err = meas_err_base*0.375 # This approximately matches the change in IV vs OLS in U.S. empirical coefficients
    np.random.seed(10)
    Measurement_Error = sigma_meas_err*np.random.normal(0.,1.,LogC.size)
    LogC_me = LogC + Measurement_Error
    DeltaLogC_me = LogC_me[1:] - LogC_me[0:-1]

    # Apply measurement error to long delta LogC
    LogC_long = np.log(ClvlAgg_hist)
    LogC_long_me = LogC_long + sigma_meas_err*np.random.normal(0.,1.,LogC_long.size)
    Delta8LogC_me = (LogC_long_me[8:] - LogC_long_me[:-8])[(ignore_periods-7):]

    # Make summary statistics for the results file
    csv_output_string = str(np.mean(AnrmAgg_hist[ignore_periods:])) +","+ str(np.mean(CnrmAgg_hist[ignore_periods:]))+ ","+str(np.std(np.log(AnrmAgg_hist[ignore_periods:])))+ ","+str(np.std(DeltaLogC))+ ","+str(np.std(DeltaLogY)) +","+ str(np.std(DeltaLogA))
    if hasattr(Economy,'agents') and calc_micro_stats: # This block only runs for heterogeneous agents specifications
        csv_output_string += ","+str(np.mean(np.std(Loga,axis=1)))+ ","+str(np.mean(np.std(Logc,axis=1))) + ","+str(np.mean(np.std(Logp,axis=1))) +","+ str(np.mean(np.nanstd(Logy_trimmed,axis=1))) +","+ str(np.std(DeltaLoga_trimmed))+","+ str(np.std(DeltaLogc_trimmed))+ ","+str(np.std(DeltaLogp_trimmed))

    # Save the results to a logfile if requested
    if filename is not None:
        with open(results_dir + filename + 'Results.csv','w') as f:
            f.write(csv_output_string)
            f.close()
        if calc_micro_stats and hasattr(Economy,'agents'):
            with open(results_dir + filename + 'BirthValue.csv','w') as f:
                my_writer = csv.writer(f, delimiter = ',')
                my_writer.writerow(vBirth)
                f.close()

        if save_data:
            DataArray = (np.vstack((np.arange(DeltaLogC.size),DeltaLogC_me,DeltaLogC,DeltaLogY,A,BigTheta,Delta8LogC,Delta8LogY,Delta8LogC_me,Measurement_Error[1:]))).transpose()
            VarNames = ['time_period','DeltaLogC_me','DeltaLogC','DeltaLogY','A','BigTheta','Delta8LogC','Delta8LogY','Delta8LogC_me','Measurement_Error']
            if hasattr(Economy,'MrkvNow'):
                DataArray = np.hstack((DataArray,np.reshape(Mrkv,(Mrkv.size,1))))
                VarNames.append('MrkvState')
            if hasattr(Economy,'MrkvNow') & ~hasattr(Economy,'Rfree') and hasattr(Economy,'agents'):
                DataArray = np.hstack((DataArray,np.reshape(R,(R.size,1))))
                VarNames.append('R')
            with open(results_dir + filename + 'Data.txt','wb') as f:
                my_writer = csv.writer(f, delimiter = '\t')
                my_writer.writerow(VarNames)
                for i in range(DataArray.shape[0]):
                    my_writer.writerow(DataArray[i,:])
                f.close()


def runStickyEregressions(infile_name,interval_size,meas_err,sticky,all_specs):
    '''
    Runs regressions for the main tables of the StickyC paper and produces a LaTeX
    table with results for one "panel".

    Parameters
    ----------
    infile_name : str
        Name of tab-delimited text file with simulation data.  Assumed to be in
        the results directory, and was almost surely generated by makeStickyEdataFile
        unless we resort to fabricating simulated data.  THAT'S A JOKE, FUTURE REFEREES.
    interval_size : int
        Number of periods in each sub-interval.
    meas_err : bool
        Indicator for whether to add measurement error to DeltaLogC.
    sticky : bool
        Indicator for whether these results used sticky expectations.
    all_specs : bool
        Indicator for whether this panel should include all specifications or
        just the OLS on lagged consumption growth.

    Returns
    -------
    panel_text : str
        String with one panel's worth of LaTeX input.
    '''
    # Read in the data from the infile
    with open(results_dir + infile_name + '.txt') as f:
        my_reader = csv.reader(f, delimiter='\t')
        all_data = list(my_reader)

    # Unpack the data into numpy arrays
    obs = len(all_data) - 1
    DeltaLogC_me = np.zeros(obs)
    DeltaLogC = np.zeros(obs)
    DeltaLogY = np.zeros(obs)
    A = np.zeros(obs)
    BigTheta = np.zeros(obs)
    Delta8LogC = np.zeros(obs)
    Delta8LogY = np.zeros(obs)
    Delta8LogC_me = np.zeros(obs)
    Measurement_Error = np.zeros(obs)
    Mrkv_hist = np.zeros(obs,dtype=int)
    R = np.zeros(obs)

    has_mrkv = 'MrkvState' in all_data[0]
    has_R = 'R' in all_data[0]
    for i in range(obs):
        j = i+1
        DeltaLogC_me[i] = float(all_data[j][1])
        DeltaLogC[i] = float(all_data[j][2])
        DeltaLogY[i] = float(all_data[j][3])
        A[i] = float(all_data[j][4])
        BigTheta[i] = float(all_data[j][5])
        Delta8LogC[i] = float(all_data[j][6])
        Delta8LogY[i] = float(all_data[j][7])
        Delta8LogC_me[i] = float(all_data[j][8])
        Measurement_Error[i] = float(all_data[j][9])
        if has_mrkv:
            Mrkv_hist[i] = int(float(all_data[j][10]))
        if has_R:
            R[i] = float(all_data[j][11])

    # Determine how many subsample intervals to run (and initialize array of coefficients)
    N = DeltaLogC.size/interval_size
    CoeffsArray = np.zeros((N,7)) # Order: DeltaLogC_OLS, DeltaLogC_IV, DeltaLogY_IV, A_OLS, DeltaLogC_HR, DeltaLogY_HR, A_HR
    StdErrArray = np.zeros((N,7)) # Same order as above
    RsqArray = np.zeros((N,5))
    PvalArray = np.zeros((N,5))
    OIDarray = np.zeros((N,5)) + np.nan
    InstrRsqVec = np.zeros(N)

    # Loop through subsample intervals, running various regressions
    for n in range(N):
        # Select the data subsample
        start = n*interval_size
        end = (n+1)*interval_size
        if meas_err:
            DeltaLogC_n = DeltaLogC_me[start:end]
            Delta8LogC_n = Delta8LogC_me[start:end]
        else:
            DeltaLogC_n = DeltaLogC[start:end]
            Delta8LogC_n = Delta8LogC[start:end]
        DeltaLogY_n = DeltaLogY[start:end]
        A_n = A[start:end]
        Delta8LogY_n = Delta8LogY[start:end]

        # Run OLS on log consumption
        mod = sm.OLS(DeltaLogC_n[1:],sm.add_constant(DeltaLogC_n[0:-1]))
        res = mod.fit()
        CoeffsArray[n,0] = res._results.params[1]
        StdErrArray[n,0] = res._results.HC0_se[1]
        RsqArray[n,0] = res._results.rsquared_adj
        PvalArray[n,0] = res._results.f_pvalue

        # Define instruments for IV regressions
        temp = np.transpose(np.vstack([DeltaLogC_n[1:-3],DeltaLogC_n[:-4],DeltaLogY_n[1:-3],DeltaLogY_n[:-4],A_n[1:-3],A_n[:-4],Delta8LogC_n[1:-3],Delta8LogY_n[1:-3]]))
        instruments = sm.add_constant(temp) # With measurement error

        # Run IV on log consumption
        mod = sm.OLS(DeltaLogC_n[3:-1],instruments)
        res = mod.fit()
        DeltaLogC_predict = res.predict()
        mod_2ndStage = sm.OLS(DeltaLogC_n[4:],sm.add_constant(DeltaLogC_predict))
        res_2ndStage = mod_2ndStage.fit()
        mod_IV = smsrg.IV2SLS(DeltaLogC_n[4:], sm.add_constant(DeltaLogC_n[3:-1]),instruments)
        res_IV = mod_IV.fit()
        CoeffsArray[n,1] = res_IV._results.params[1]
        StdErrArray[n,1] = res_IV.bse[1]
        RsqArray[n,1] = res_2ndStage._results.rsquared_adj
        PvalArray[n,1] = res._results.f_pvalue

        # Run IV on log income
        mod = sm.OLS(DeltaLogY_n[4:],instruments)
        res = mod.fit()
        DeltaLogY_predict = res.predict()
        mod_2ndStage = sm.OLS(DeltaLogC_n[4:],sm.add_constant(DeltaLogY_predict))
        res_2ndStage = mod_2ndStage.fit()
        mod_IV = smsrg.IV2SLS(DeltaLogC_n[4:], sm.add_constant(DeltaLogY_n[4:]),instruments)
        res_IV = mod_IV.fit()
        CoeffsArray[n,2] = res_IV._results.params[1]
        StdErrArray[n,2] = res_IV.bse[1]
        RsqArray[n,2] = res_2ndStage._results.rsquared_adj
        PvalArray[n,2] = res._results.f_pvalue

        # Run IV on assets
        mod = sm.OLS(A_n[3:-1],instruments)
        res = mod.fit()
        A_predict = res.predict()
        mod_2ndStage = sm.OLS(DeltaLogC_n[4:],sm.add_constant(A_predict))
        res_2ndStage = mod_2ndStage.fit()
        mod_IV = smsrg.IV2SLS(DeltaLogC_n[4:], sm.add_constant(A_n[3:-1]),instruments)
        res_IV = mod_IV.fit()
        CoeffsArray[n,3] = res_IV._results.params[1]
        StdErrArray[n,3] = res_IV.bse[1]
        RsqArray[n,3] = res_2ndStage._results.rsquared_adj
        PvalArray[n,3] = res._results.f_pvalue

        # Run horserace IV
        regressors = sm.add_constant(np.transpose(np.array([DeltaLogC_predict,DeltaLogY_predict,A_predict])))
        mod_2ndStage = sm.OLS(DeltaLogC_n[4:],regressors)
        res_2ndStage = mod_2ndStage.fit()
        mod_IV = smsrg.IV2SLS(DeltaLogC_n[4:], sm.add_constant(np.transpose(np.array([DeltaLogC_n[3:-1],DeltaLogY_n[4:],A_n[3:-1]]))),instruments)
        res_IV = mod_IV.fit()
        CoeffsArray[n,4] = res_IV._results.params[1]
        CoeffsArray[n,5] = res_IV._results.params[2]
        CoeffsArray[n,6] = res_IV._results.params[3]
        StdErrArray[n,4] = res_IV._results.bse[1]
        StdErrArray[n,5] = res_IV._results.bse[2]
        StdErrArray[n,6] = res_IV._results.bse[3]
        RsqArray[n,4] = res_2ndStage._results.rsquared_adj
        PvalArray[n,4] = np.nan    #Need to put in KP stat here, may have to do this in Stata

        # Regress Delta C_{t+1} on instruments
        mod = sm.OLS(DeltaLogC_n[4:],instruments)
        res = mod.fit()
        InstrRsqVec[n] = res._results.rsquared_adj

    # Count the number of times we reach significance in each variable
    t_stat_array = CoeffsArray/StdErrArray
    C_successes_95 = np.sum(t_stat_array[:,4] > 1.96)
    Y_successes_95 = np.sum(t_stat_array[:,5] > 1.96)

    sigma_meas_err = np.std(Measurement_Error)

    N_out = [C_successes_95,Y_successes_95,N,np.mean(InstrRsqVec),sigma_meas_err**2]

    # Make results table and return it
    panel_text = makeResultsPanel(Coeffs=np.mean(CoeffsArray,axis=0),
                     StdErrs=np.mean(StdErrArray,axis=0),
                     Rsq=np.mean(RsqArray,axis=0),
                     Pvals=np.mean(PvalArray,axis=0),
                     OID=np.mean(OIDarray,axis=0),
                     Counts=N_out,
                     meas_err=meas_err,
                     sticky=sticky,
                     all_specs=all_specs)
    return panel_text


def runStickyEregressionsInStata(infile_name,interval_size,meas_err,sticky,all_specs,stata_exe):
    '''
    Runs regressions for the main tables of the StickyC paper in Stata and produces a
    LaTeX table with results for one "panel". Running in Stata allows production of
    the KP-statistic, for which there is currently no command in statsmodels.api.

    Parameters
    ----------
    infile_name : str
        Name of tab-delimited text file with simulation data.  Assumed to be in
        the results directory, and was almost surely generated by makeStickyEdataFile
        unless we resort to fabricating simulated data.  THAT'S A JOKE, FUTURE REFEREES.
    interval_size : int
        Number of periods in each regression sample (or interval).
    meas_err : bool
        Indicator for whether to add measurement error to DeltaLogC.
    sticky : bool
        Indicator for whether these results used sticky expectations.
    all_specs : bool
        Indicator for whether this panel should include all specifications or
        just the OLS on lagged consumption growth.
    stata_exe : str
        Absolute location where the Stata executable can be found on the computer
        running this code.  Usually set at the top of StickyEparams.py.

    Returns
    -------
    panel_text : str
        String with one panel's worth of LaTeX input.
    '''
    dofile = "StickyETimeSeries.do"
    infile_name_full = os.path.abspath(results_dir + infile_name + ".txt")
    temp_name_full = os.path.abspath(results_dir + "temp.txt")
    if meas_err:
        meas_err_stata = 1
    else:
        meas_err_stata = 0

    # Define the command to run the Stata do file
    cmd = [stata_exe, "do", dofile, infile_name_full, temp_name_full, str(interval_size), str(meas_err_stata)]

    # Run Stata do-file
    stata_status = subprocess.call(cmd,shell = 'true')
    if stata_status!=0:
        raise ValueError('Stata code could not run. Check the stata_exe in StickyEparams.py')
    stata_output = pd.read_csv(temp_name_full, sep=',',header=0)

    # Make results table and return it
    panel_text = makeResultsPanel(Coeffs=stata_output.CoeffsArray,
                     StdErrs=stata_output.StdErrArray,
                     Rsq=stata_output.RsqArray,
                     Pvals=stata_output.PvalArray,
                     OID=stata_output.OIDarray,
                     Counts=stata_output.ExtraInfo,
                     meas_err=meas_err,
                     sticky=sticky,
                     all_specs=all_specs)
    return panel_text


def calcValueAtBirth(cLvlHist,BirthBool,PlvlHist,MrkvHist,DiscFac,CRRA):
    '''
    Calculate expected value of being born in each Markov state using the realizations
    of consumption for a history of many consumers.  The histories should already be
    trimmed of the "burn in" periods.

    Parameters
    ----------
    cLvlHist : np.array
        TxN array of consumption level history for many agents across many periods.
        Agents who die are replaced by newborms.
    BirthBool : np.array
        TxN boolean array indicating when agents are born, replacing one who died.
    PlvlHist : np.array
        T length vector of aggregate permanent productivity levels.
    MrkvHist : np.array
        T length vector of integers for the Markov index in each period.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion.

    Returns
    -------
    vAtBirth : np.array
        J length vector of average lifetime value at birth by Markov state.
    '''
    J = np.max(MrkvHist) + 1 # Number of Markov states
    T = MrkvHist.size        # Length of simulation
    I = cLvlHist.shape[1]    # Number of agent indices in histories
    u = lambda c : CRRAutility(c,gam=CRRA)

    # Initialize an array to hold each agent's lifetime utility
    BirthsByPeriod = np.sum(BirthBool,axis=1)
    BirthsByState = np.zeros(J,dtype=int)
    for j in range(J):
        these = MrkvHist == j
        BirthsByState[j] = np.sum(BirthsByPeriod[these])
    N = np.max(BirthsByState) # Array must hold this many agents per row at least
    vArray = np.zeros((J,N)) + np.nan
    n = np.zeros(J,dtype=int)

    # Loop through each agent index
    DiscVec = DiscFac**np.arange(T)
    for i in range(I):
        birth_t = np.where(BirthBool[:,i])[0]
        # Loop through each agent who lived and died in this index
        for k in range(birth_t.size-1): # Last birth event has no death, so ignore
            # Get lifespan of this agent and circumstances at birth
            t0 = birth_t[k]
            t1 = birth_t[k+1]
            span = t1-t0
            j = MrkvHist[t0]
            # Calculate discounted flow of utility for this agent and store it
            cVec = cLvlHist[t0:t1,i]/PlvlHist[t0]
            uVec = u(cVec)
            v = np.dot(DiscVec[:span],uVec)
            vArray[j,n[j]] = v
            n[j] += 1

    # Calculate expected value at birth by state and return it
    vAtBirth = np.nanmean(vArray,axis=1)
    return vAtBirth


def makeResultsPanel(Coeffs,StdErrs,Rsq,Pvals,OID,Counts,meas_err,sticky,all_specs):
    '''
    Make one panel of simulated results table.  A panel has all results with/out
    measurement error for the sticky or frictionless version, and might have only
    one specification (OLS on lagged consumption growth) or many specifications.

    Parameters
    ----------
    Coeffs : np.array
        Array with 7 entries with regression coefficients.
    StdErrs : np.array
        Array with 7 entries with coefficient standard errors.
    Rsq : np.array
        Array with 5 entries with R^2 values.
    Pvals : np.array
        Array with 5 entries with P values (for the first stage).
    OID : np.array
        Array with 5 entries with overidentification statistics.
    Counts : [int]
        List with 3 elements: [C_successes,Y_successes,N_intervals].
    meas_err : bool
        Indicator for whether this panel used measurement error.
    sticky : bool
        Indicator for whether this panel used sticky expectations.
    all_specs : bool
        Indicator for whether this panel should include all specifications or
        just the OLS on lagged consumption growth.

    Returns
    -------
    output : str
        Text string with one panel of LaTeX input.
    '''
    # Define Delta log C text and expectations text
    if meas_err:
        DeltaLogC = '$\Delta \log \mathbf{C}_{t}^*$'
        DeltaLogC1 = '$\Delta \log \mathbf{C}_{t+1}^*$'
        MeasErr = ' (with measurement error $\mathbf{C}_{t}^* =\mathbf{C}_{t}\\times \\xi_t$);'
    else:
        DeltaLogC = '$\Delta \log \mathbf{C}_{t}$'
        DeltaLogC1 = '$\Delta \log \mathbf{C}_{t+1}$'
        MeasErr = ' (no measurement error)'

    if sticky:
        Expectations = 'Sticky'
    else:
        Expectations = 'Frictionless'

    DeltaLogY1 = '$\Delta \log \mathbf{Y}_{t+1}$'
    A_t = '$A_{t}$'


    # Define significance symbols
    if Counts[2] > 1:
        sig_symb = '\\bullet '
    else:
        sig_symb = '*'
    def sigFunc(coeff,stderr):
        z_stat = np.abs(coeff/stderr)
        cuts = np.array([1.645,1.96,2.576])
        N = np.sum(z_stat > cuts)
        if N > 0:
            sig_text = '^{' + N*sig_symb + '}'
        else:
            sig_text = ''
        return sig_text

    # Make the memo text
    memo = ''
    if all_specs:
        memo += '\\\\ \multicolumn{6}{l}{Memo: For instruments $\mathbf{Z}_{t}$, ' + DeltaLogC + ' $= \mathbf{Z}_{t} \zeta, ~\\bar{R}^{2}=$ ' + mystr1(Counts[3])
        if meas_err:
            memo += '; ~$\\var(\\log(\\xi_t))=$ ' + mystr3(Counts[4])
        memo += ' }  \n'

    # Make the top of the panel
    output = '\\\\ \\midrule \multicolumn{6}{l}{' + Expectations + ' : ' + DeltaLogC1 + MeasErr + '} \n'
    output += '\\\\ \multicolumn{1}{c}{' + DeltaLogC + '} & \multicolumn{1}{c}{' + DeltaLogY1 +'} & \multicolumn{1}{c}{'+A_t+'} & & & \n'

    # OLS on just lagged consumption growth
    output += '\\\\ ' + mystr1(Coeffs[0]) + sigFunc(Coeffs[0],StdErrs[0]) + ' & & & OLS & ' + mystr1(Rsq[0]) + ' & ' + mystr1(np.nan) + '\n'
    output += '\\\\ (' + mystr1(StdErrs[0]) + ') & & & & & \n'

    # Add the rest of the specifications if requested
    if all_specs:
        # IV on lagged consumption growth
        output += '\\\\ ' + mystr1(Coeffs[1]) + sigFunc(Coeffs[1],StdErrs[1]) + ' & & & IV & ' + mystr1(Rsq[1]) + ' & ' + mystr1(Pvals[1]) + '\n'
        output += '\\\\ (' + mystr1(StdErrs[1]) + ') & & & & &' + mystr1(OID[1]) + '\n'

        # IV on expected income growth
        output += '\\\\ & ' + mystr1(Coeffs[2]) + sigFunc(Coeffs[2],StdErrs[2]) + ' & & IV & ' + mystr1(Rsq[2]) + ' & ' + mystr1(Pvals[2]) + '\n'
        output += '\\\\ & (' + mystr1(StdErrs[2]) + ') & & & &' + mystr1(OID[2]) + '\n'

        # IV on aggregate assets
        output += '\\\\ & & ' + mystr2(Coeffs[3]) + sigFunc(Coeffs[3],StdErrs[3]) + ' & IV & ' + mystr1(Rsq[3]) + ' & ' + mystr1(Pvals[3]) + '\n'
        output += '\\\\ & & (' + mystr2(StdErrs[3]) + ') & & &' + mystr1(OID[3]) + '\n'

        # Horse race
        output += '\\\\ ' + mystr1(Coeffs[4]) + sigFunc(Coeffs[4],StdErrs[4]) + ' & ' + mystr1(Coeffs[5]) + sigFunc(Coeffs[5],StdErrs[5]) + ' & ' + mystr2(Coeffs[6]) + sigFunc(Coeffs[6],StdErrs[6]) + ' & IV & ' + mystr1(Rsq[4]) + ' & ' + mystr1(Pvals[4]) + '\n'
        output += '\\\\ (' + mystr1(StdErrs[4]) + ') & (' + mystr1(StdErrs[5]) + ') & (' + mystr2(StdErrs[6]) + ') & & & ' + mystr1(OID[4]) + '\n'
    output += memo

    if Counts[0] is not None and Counts[2] > 1 and False:
        output += '\\\\ \multicolumn{6}{c}{Horserace coefficient on ' + DeltaLogC + ' significant at 95\% level for ' + str(Counts[0]) + ' of ' + str(Counts[2]) + ' subintervals.} \n'
        output += '\\\\ \multicolumn{6}{c}{Horserace coefficient on $\mathbb{E}[\Delta \log \mathbf{Y}_{t+1}]$ significant at 95\% level for ' + str(Counts[1]) + ' of ' + str(Counts[2]) + ' subintervals.} \n'

    return output


def makeResultsTable(caption,panels,counts,filename,label):
    '''
    Make a time series regression results table by piecing together one or more panels.
    Saves a tex file to disk in the tables directory.

    Parameters
    ----------
    caption : str or None
        Text to apply at the start of the table as a title.  If None, the table
        environment is not invoked and the output is formatted for slides.
    panels : [str]
        List of strings with one or more panels, usually made by makeResultsPanel.
    counts : int
        List of two integers: [interval_length, interval_count]
    filename : str
        Name of the file in which to save output (in the ./Tables/ directory).
    label : str
        LaTeX \label, for internal reference in the paper text.

    Returns
    -------
    None
    '''
    if caption is not None:
        note_size = '\\footnotesize'
    else:
        note_size = '\\tiny'
    note = '\\multicolumn{6}{p{0.95\\textwidth}}{' + note_size + ' \\textbf{Notes:} '
    if counts[1] > 1:
        note += 'Reported statistics are the average values for ' + str(counts[1]) + ' samples of ' + str(counts[0]) + ' simulated quarters each.  '
        note += 'Bullets indicate that the average sample coefficient divided by average sample standard error is outside of the inner 90\%, 95\%, and 99\% of the standard normal distribution.  '
    else:
        note += 'Reported statistics are for a single simulation of ' + str(counts[0]) + ' quarters.  '
        note += 'Stars indicate statistical significance at the 90\%, 95\%, and 99\% levels, respectively.  '
    note += 'Instruments $\\textbf{Z}_t = \\{\Delta \log \mathbf{C}_{t-2}, \Delta \log \mathbf{C}_{t-3}, \Delta \log \mathbf{Y}_{t-2}, \Delta \log \mathbf{Y}_{t-3}, A_{t-2}, A_{t-3}, \Delta_8 \log \mathbf{C}_{t-2}, \Delta_8 \log \mathbf{Y}_{t-2}   \\}$.'
    note += '}'

    if caption is not None:
        output = '\\begin{minipage}{\\textwidth}\n'
        output += '\\begin{table} \caption{' + caption + '} \\label{' + label + '} \n'
        output += '  \\centerline{$ \Delta \log \mathbf{C}_{t+1} = \\varsigma + \chi \Delta \log \mathbf{C}_t + \eta \mathbb{E}_t[\Delta \log \mathbf{Y}_{t+1}] + \\alpha A_t + \epsilon_{t+1} $}\n'
    else:
        output = '\\begin{center} \n'
        output += '$ \Delta \log \mathbf{C}_{t+1} = \\varsigma + \chi \Delta \log \mathbf{C}_t + \eta \mathbb{E}_t[\Delta \log \mathbf{Y}_{t+1}] + \\alpha A_t + \epsilon_{t+1} $ \\\\  \n'
    output += '\\begin{tabular}{d{4}d{4}d{5}cd{4}c}\n \\toprule \n'
    output += '\multicolumn{3}{c}{Expectations : Dep Var} & OLS &  \multicolumn{1}{c}{2${}^{\\text{nd}}$ Stage}  &  \multicolumn{1}{c}{KP $p$-val} \n'
    output += '\\\\ \multicolumn{3}{c}{Independent Variables} & or IV & \multicolumn{1}{c}{$\\bar{R}^{2} $} & \multicolumn{1}{c}{Hansen J $p$-val} \n'

    for panel in panels:
        output += panel

    output += '\\\\ \\bottomrule \n ' + note + '\n'
    output += '\end{tabular}\n'

    if caption is not None:
        output += '\end{table}\n'
        output += '\end{minipage}\n'
    else:
        output += '\end{center}\n'

    with open(tables_dir + filename + '.tex','w') as f:
        f.write(output)
        f.close()


def makeParameterTable(filename, params):
    '''
    Makes the parameter table for the paper, saving it to a tex file in the tables folder.
    Also makes two partial parameter tables for the slides.

    Parameters
    ----------

    filename : str
        Name of the file in which to save output (in the tables directory).
        Suffix .tex is automatically added.
    params :
        Object containing the parameter values.

    Returns
    -------
    None
    '''
    # Calibrated macroeconomic parameters
    macro_panel = "\multicolumn{3}{c}{\\textbf{Macroeconomic Parameters} }  \n"
    macro_panel += "\\\\ $\\kapShare$ & " + "{:.2f}".format(params.CapShare) + " & Capital's Share of Income   \n"
    macro_panel += "\\\\ $\\daleth$ & " + "{:.2f}".format(params.DeprFacAnn) + "^{1/4} & Depreciation Factor   \n"
    macro_panel += "\\\\ $\sigma_{\Theta}^{2}$ & "+ "{:.5f}".format(params.TranShkAggVar) +" & Variance Aggregate Transitory Shocks \n"
    macro_panel += "\\\\ $\sigma_{\Psi}^{2}$ & "+ "{:.5f}".format(params.PermShkAggVar) +" & Variance Aggregate Permanent Shocks \n"

    # Steady state values
    SS_panel = "\multicolumn{3}{c}{ \\textbf{Steady State of Perfect Foresight DSGE Model} } \\  \n"
    SS_panel += "\\\\ \multicolumn{3}{c}{ $(\\sigma_{\\Psi}=\\sigma_{\\Theta}=\\sigma_{\\psi}=\\sigma_{\\theta}=\wp=\\PDies=0$, $\\Phi_t = 1)$} \\  \n"
    SS_panel += "\\\\ $\\breve{K}/\\breve{K}^{\\kapShare}$ & " + "{:.1f}".format(params.KYratioSS) + " & SS Capital to Output Ratio  \n"
    SS_panel += "\\\\ $\\breve{K}$ & " + "{:.2f}".format(params.KSS) + " & SS Capital to Labor Productivity Ratio ($=12^{1/(1-\\kapShare)}$) \n"
    SS_panel += "\\\\ $\\breve{\\Wage}$ &  " + "{:.2f}".format(params.wRteSS) + " & SS Wage Rate ($=(1-\\kapShare)\\breve{K}^{\\kapShare}$) \n"
    SS_panel += "\\\\ $\\breve{\\mathsf{r}}$ & " + "{:.2f}".format(params.rFreeSS) + " & SS Interest Rate ($=\\kapShare \\breve{K}^{\\kapShare-1}$) \n"
    SS_panel += "\\\\ $\\breve{\\Rprod}$ & " + "{:.3f}".format(params.RfreeSS) + "& SS Between-Period Return Factor ($=\\daleth + \\breve{\\mathsf{r}}$) \n"

    # Calibrated preference parameters
    pref_panel = "\multicolumn{3}{c}{ \\textbf{Preference Parameters} }  \n"
    pref_panel += "\\\\ $\\rho$ & "+ "{:.0f}".format(params.CRRA) +". & Coefficient of Relative Risk Aversion \n"
    pref_panel += "\\\\ $\\beta_{SOE}$ &  " + "{:.3f}".format(params.DiscFacSOE) +" & SOE Discount Factor \n" #($=0.99 \\cdot \\PLives / (\\breve{\\mathcal{R}} \\Ex [\\pmb{\\psi}^{-\CRRA}])$)\n"
    pref_panel += "\\\\ $\\beta_{DSGE}$ &  " + "{:.3f}".format(params.DiscFacDSGE) +" & HA-DSGE Discount Factor ($=\\breve{\\Rprod}^{-1}$) \n"
    pref_panel += "\\\\ $\Pi$                    & " + "{:.2f}".format(params.UpdatePrb) +"  & Probability of Updating Expectations (if Sticky) \n"

    # Idiosyncratic shock parameters
    idio_panel = "\multicolumn{3}{c}{ \\textbf{Idiosyncratic Shock Parameters} }  \n"
    idio_panel += "\\\\ $\sigma_{\\theta}^{2}$    & " + "{:.3f}".format(params.TranShkVar) +"     & Variance Idiosyncratic Tran Shocks (=$4 \\times$ Annual) \n"
    idio_panel += "\\\\ $\sigma_{\psi}^{2}$      &" + "{:.3f}".format(params.PermShkVar) +"      & Variance Idiosyncratic Perm Shocks (=$\\frac{1}{4} \\times$ Annual) \n"
    idio_panel += "\\\\ $\wp$                    & " + "{:.3f}".format(params.UnempPrb) +"  & Probability of Unemployment Spell \n"
    idio_panel += "\\\\ $\PDies$             & " + "{:.3f}".format(params.DiePrb) +"  & Probability of Mortality \n"

    # Make full parameter table for paper
    paper_output = "\provideboolean{Slides} \setboolean{Slides}{false}  \n"

    paper_output += "\\begin{minipage}{\\textwidth}\n"
    paper_output += "  \\begin{table}\n"
    paper_output += "    \\caption{Calibration}\label{table:calibration}\n"

    paper_output += "\\begin{tabular}{cd{5}l}  \n"
    paper_output += "\\\\ \\toprule  \n"
    paper_output += macro_panel
    paper_output += "\\\\ \\midrule  \n"
    paper_output += SS_panel
    paper_output += "\\\\ \\midrule  \n"
    paper_output += pref_panel
    paper_output += "\\\\ \\midrule  \n"
    paper_output += idio_panel
    paper_output += "\\\\ \\bottomrule  \n"
    paper_output += "\end{tabular}\n"
    paper_output += "\end{table}\n"
    paper_output += "\end{minipage}\n"
    paper_output += "\ifthenelse{\\boolean{StandAlone}}{\end{document}}{}    \n"
    with open(tables_dir + filename + '.tex','w') as f:
        f.write(paper_output)
        f.close()

    # Make two partial parameter tables for the slides
    slides1_output = "\\begin{center}\label{table:calibration1}  \n"
    slides1_output += "\\begin{tabular}{cd{5}l}  \n"
    slides1_output += "\\\\ \\toprule  \n"
    slides1_output += macro_panel
    slides1_output += "\\\\ \\midrule  \n"
    slides1_output += SS_panel
    slides1_output += "\\\\ \\bottomrule  \n"
    slides1_output += "\end{tabular}  \n"
    slides1_output += "\end{center}  \n"
    with open(tables_dir + filename + '_1.tex','w') as f:
        f.write(slides1_output)
        f.close()

    slides2_output = "\\begin{center}\label{table:calibration2}  \n"
    slides2_output += "\\begin{tabular}{cd{5}l}  \n"
    slides2_output += "\\\\ \\toprule  \n"
    slides2_output += pref_panel
    slides2_output += "\\\\ \\midrule  \n"
    slides2_output += idio_panel
    slides2_output += "\\\\ \\bottomrule  \n"
    slides2_output += "\end{tabular}  \n"
    slides2_output += "\end{center}  \n"
    with open(tables_dir + filename + '_2.tex','w') as f:
        f.write(slides2_output)
        f.close()


def makeEquilibriumTable(out_filename, four_in_files, CRRA):
    '''
    Make the equilibrium statistics table for the paper, saving it as a tex file
    in the tables folder.  Also makes a version for the slides that doesn't use
    the table environment, nor include the note at bottom.

    Parameters
    ----------

    out_filename : str
        Name of the file in which to save output (in the tables directory).
        Suffix .tex appended automatically.
    four_in_files: [str]
        A list with four csv files. 0) SOE frictionless 1) SOE Sticky 2) DSGE frictionless 3) DSGE sticky
    CRRA : float
        Coefficient of relative risk aversion

    Returns
    -------
    None
    '''
    # Read in statistics from the four files
    SOEfrictionless = np.genfromtxt(results_dir + four_in_files[0] + 'Results.csv', delimiter=',')
    SOEsticky = np.genfromtxt(results_dir + four_in_files[1] + 'Results.csv', delimiter=',')
    DSGEfrictionless = np.genfromtxt(results_dir + four_in_files[2] + 'Results.csv', delimiter=',')
    DSGEsticky = np.genfromtxt(results_dir + four_in_files[3] + 'Results.csv', delimiter=',')

    # Read in value at birth from the four files
    vBirth_SOE_F = np.genfromtxt(results_dir + four_in_files[0] + 'BirthValue.csv', delimiter=',')
    vBirth_SOE_S = np.genfromtxt(results_dir + four_in_files[1] + 'BirthValue.csv', delimiter=',')
    vBirth_DSGE_F = np.genfromtxt(results_dir + four_in_files[2] + 'BirthValue.csv', delimiter=',')
    vBirth_DSGE_S = np.genfromtxt(results_dir + four_in_files[3] + 'BirthValue.csv', delimiter=',')

    # Calculate the cost of stickiness in the SOE and DSGE models
    StickyCost_SOE = np.mean(1. - (vBirth_SOE_S/vBirth_SOE_F)**(1./(1.-CRRA)))
    StickyCost_DSGE = np.mean(1. - (vBirth_DSGE_S/vBirth_DSGE_F)**(1./(1.-CRRA)))

    paper_top = "\\begin{minipage}{\\textwidth}\n"
    paper_top += "    \\begin{table}  \n"
    paper_top += "\caption{Equilibrium Statistics}  \n"
    paper_top += "\label{table:Eqbm}  \n"
    paper_top += "\\newsavebox{\EqbmBox}  \n"
    paper_top += "\sbox{\EqbmBox}{  \n"
    paper_top += "\\newcommand{\EqDir}{\TablesDir/Eqbm}  \n"

    slides_top = '\\begin{center} \n'

    main_table = "\\begin{tabular}{lllcccc}  \n"
    main_table += "\\toprule \n"
    main_table += "&&& \multicolumn{2}{c}{SOE Model} & \multicolumn{2}{c}{HA-DSGE Model}   \n"
    main_table += "\\\\ %\cline{4-5}   \n"
    main_table += "   &&& \multicolumn{1}{c}{Frictionless} & \multicolumn{1}{c}{Sticky} & \multicolumn{1}{c}{Frictionless} & \multicolumn{1}{c}{Sticky}  \n"
    main_table += "\\\\ \\midrule   \n"
    main_table += "  \multicolumn{3}{l}{Means}  \n"
    main_table += "%\\\\  & & $M$  \n"
    main_table += "%\\\\  & & $K$  \n"
    main_table += "\\\\  & & $A$ & {:.2f}".format(SOEfrictionless[0]) +" &{:.2f}".format(SOEsticky[0]) +" & {:.2f}".format(DSGEfrictionless[0]) +" & {:.2f}".format(DSGEsticky[0]) +"   \n"
    main_table += "\\\\  & & $C$ & {:.2f}".format(SOEfrictionless[1]) +" &{:.2f}".format(SOEsticky[1]) +" & {:.2f}".format(DSGEfrictionless[1]) +" & {:.2f}".format(DSGEsticky[1]) +"   \n"
    main_table += "\\\\ \\midrule  \n"
    main_table += "  \multicolumn{3}{l}{Standard Deviations}  \n"
    main_table += "\\\\ &    \multicolumn{4}{l}{Aggregate Time Series (`Macro')}  \n"
    main_table += "%\\  & & $\Delta \log \mathbf{M}$   \n"
    main_table += "\\\\ & & $\log A $         & {:.3f}".format(SOEfrictionless[2]) +" & {:.3f}".format(SOEsticky[2]) +" & {:.3f}".format(DSGEfrictionless[2]) +" & {:.3f}".format(DSGEsticky[2]) +" \n"
    main_table += "\\\\ & & $\Delta \log \\CLevBF $  & {:.3f}".format(SOEfrictionless[3]) +" & {:.3f}".format(SOEsticky[3]) +" & {:.3f}".format(DSGEfrictionless[3]) +" & {:.3f}".format(DSGEsticky[3]) +" \n"
    main_table += "\\\\ & & $\Delta \log \\YLevBF $  & {:.3f}".format(SOEfrictionless[4]) +" & {:.3f}".format(SOEsticky[4]) +" & {:.3f}".format(DSGEfrictionless[4]) +" & {:.3f}".format(DSGEsticky[4]) +" \n"
    main_table += "\\\\ &   \multicolumn{3}{l}{Individual Cross Sectional (`Micro')}  \n"
    main_table += "\\\\ & & $\log \\aLevBF $  & {:.3f}".format(SOEfrictionless[6]) +" & {:.3f}".format(SOEsticky[6]) +" & {:.3f}".format(DSGEfrictionless[6]) +" & {:.3f}".format(DSGEsticky[6]) +" \n"
    main_table += "\\\\ & & $\log \\cLevBF $  & {:.3f}".format(SOEfrictionless[7]) +" & {:.3f}".format(SOEsticky[7]) +" & {:.3f}".format(DSGEfrictionless[7]) +" & {:.3f}".format(DSGEsticky[7]) +" \n"
    main_table += "\\\\ & & $\log p $  & {:.3f}".format(SOEfrictionless[8]) +" & {:.3f}".format(SOEsticky[8]) +" & {:.3f}".format(DSGEfrictionless[8]) +" & {:.3f}".format(DSGEsticky[8]) +" \n"
    main_table += "\\\\ & & $\log \\yLevBF | \\yLevBF > 0 $  & {:.3f}".format(SOEfrictionless[9]) +" & {:.3f}".format(SOEsticky[9]) +" & {:.3f}".format(DSGEfrictionless[9]) +" & {:.3f}".format(DSGEsticky[9]) +" \n"
    main_table += "\\\\ & & $\Delta \log \\cLevBF $  & {:.3f}".format(SOEfrictionless[11]) +" & {:.3f}".format(SOEsticky[11]) +" & {:.3f}".format(DSGEfrictionless[11]) +" & {:.3f}".format(DSGEsticky[11]) +" \n"
    main_table += "  \n"
    main_table += "  \n"
    main_table += "\\\\ \\midrule \multicolumn{3}{l}{Cost of Stickiness}  \n"
    main_table += " & \multicolumn{2}{c}{" + mystr2(StickyCost_SOE) + "}  \n"
    main_table += " & \multicolumn{2}{c}{" + mystr2(StickyCost_DSGE) + "}  \n"
    main_table += "\\\\ \\bottomrule  \n"
    main_table += " \end{tabular}   \n"

    paper_bot = " } \n "
    paper_bot += "\usebox{\EqbmBox}  \n"
    paper_bot += "\ifthenelse{\\boolean{StandAlone}}{\\newlength\TableWidth}{}  \n"
    paper_bot += "\settowidth\TableWidth{\usebox{\EqbmBox}} % Calculate width of table so notes will match  \n"
    paper_bot += "\medskip\medskip \\vspace{0.0cm} \parbox{\TableWidth}{\\footnotesize\n"
    paper_bot += "\\textbf{Notes}: The cost of stickiness is calculated as the proportion by which the permanent income of a newborn frictionless consumer would need to be reduced in order to achieve the same reduction of expected value associated with forcing them to become a sticky expectations consumer.}  \n"
    paper_bot += "\end{table}\n"
    paper_bot += "\end{minipage}\n"
    paper_bot += "\ifthenelse{\\boolean{StandAlone}}{\end{document}}{}  \n"

    slides_bot = '\\end{center} \n'

    paper_output = paper_top + main_table + paper_bot
    with open(tables_dir + out_filename + '.tex','w') as f:
        f.write(paper_output)
        f.close()

    slides_output = slides_top + main_table + slides_bot
    with open(tables_dir + out_filename + 'Slides.tex','w') as f:
        f.write(slides_output)
        f.close()


def extractSampleMicroData(Economy, num_periods, AgentCount, ignore_periods):
    '''
    Extracts sample micro data to be used in micro (cross section) regression.

    Parameters
    ----------

    Economy : Economy
        An economy (with one AgentType) for for which history has already been calculated.
    num_periods : int
        Number of periods to be stored (should be less than the number of periods calculated)
    AgentCouunt : int
        Number of agent histories to be stored (should be less than the AgentCount property of the agent)
    ignore_periods : int
        Number of periods at the beginning of the history to be discarded

    Returns
    -------
    micro_data : np.array
        Array with rows 1) logc_diff 2) log_trans_shk 3) top_assets
    '''
    # First pull out economy common data.
    # Note indexing on Economy tracked vars is one ahead of agent.
    agg_trans_shk_matrix = deepcopy(Economy.TranShkAggNow_hist[ignore_periods:ignore_periods+num_periods])
    wRte_matrix = deepcopy(Economy.wRteNow_hist[ignore_periods:ignore_periods+num_periods])

    # Now pull out agent data
    agent = Economy.agents[0]
    c_matrix = deepcopy(agent.cLvlNow_hist[(ignore_periods+1):ignore_periods+num_periods+1,0:AgentCount])
    y_matrix = deepcopy(agent.yLvlNow_hist[(ignore_periods+1):ignore_periods+num_periods+1,0:AgentCount])
    total_trans_shk_matrix = deepcopy(agent.TranShkNow_hist[(ignore_periods+1):ignore_periods+num_periods+1,0:AgentCount])
    trans_shk_matrix = total_trans_shk_matrix/(np.array(agg_trans_shk_matrix)*np.array(wRte_matrix))[:,None]
    a_matrix = deepcopy(agent.aLvlNow_hist[(ignore_periods+1):ignore_periods+num_periods+1,0:AgentCount])
    pLvlTrue_matrix = deepcopy(agent.pLvlTrue_hist[(ignore_periods+1):ignore_periods+num_periods+1,0:AgentCount])
    a_matrix_nrm = a_matrix/pLvlTrue_matrix
    age_matrix = deepcopy(agent.t_age_hist[(ignore_periods+1):ignore_periods+num_periods+1,0:AgentCount])

    # Put nan's in so that we do not regress over periods where agents die
    newborn = age_matrix == 1
    c_matrix[newborn] = np.nan
    c_matrix[0:0,:] = np.nan
    y_matrix[newborn] = np.nan
    y_matrix[0,0:] = np.nan
    y_matrix[trans_shk_matrix==0.0] = np.nan
    c_matrix[trans_shk_matrix==0.0] = np.nan
    trans_shk_matrix[trans_shk_matrix==0.0] = np.nan

    top_assets = a_matrix_nrm > np.transpose(np.tile(np.percentile(a_matrix_nrm,1,axis=1),(np.shape(a_matrix_nrm)[1],1)))
    logc_diff = np.log(c_matrix[1:,:])-np.log(c_matrix[:-1,:])
    logy_diff = np.log(y_matrix[1:,:])-np.log(y_matrix[:-1,:])
    logc_diff = logc_diff.flatten('F')
    logy_diff = logy_diff.flatten('F')
    log_trans_shk = np.log(trans_shk_matrix[1:,:].flatten('F'))
    top_assets = top_assets[1:,:].flatten('F')

    # Put nan's in where they exist in logc_diff
    log_trans_shk = log_trans_shk + logc_diff*0.0
    top_assets = top_assets + logc_diff*0.0

    return np.stack((logc_diff,log_trans_shk,top_assets),1)


def makeMicroRegressionTable(out_filename, micro_data):
    '''
    Make the micro regression or (cross section regression) table for the paper, saving
    it to a tex file in the tables folder.  Also makes two partial tables for the slides.

    Parameters
    ----------

    out_filename : str
        Name of the file in which to save output (in the tables directory).
        The suffix .tex is automatically added.
    micro_data : [np.array]
        A list of two np.array's each containing micro data array as returned by extractSampleMicroData

    Returns
    -------
    None
    '''
    coeffs = np.zeros((6,2)) + np.nan
    stdevs = np.zeros((6,2)) +np.nan
    r_sq = np.zeros((4,2)) +np.nan
    obs = np.zeros((4,2)) +np.nan
    for i in range(2):
        this_micro_data = micro_data[i]
        logc_diff = this_micro_data[:,0]
        log_trans_shk = this_micro_data[:,1]
        top_assets = this_micro_data[:,2]

        #Lagged consumption regression
        mod = sm.OLS(logc_diff[1:],sm.add_constant(np.transpose(np.vstack([logc_diff[0:-1]]))), missing='drop')
        res = mod.fit()
        coeffs[0,i] = res._results.params[1]
        stdevs[0,i] = res._results.HC0_se[1]
        r_sq[0,i] = res._results.rsquared_adj
        obs[0,i] = res.nobs

        #Expected income regression
        mod = sm.OLS(logc_diff[1:],sm.add_constant(np.transpose(np.vstack([-log_trans_shk[0:-1]]))), missing='drop')
        res = mod.fit()
        coeffs[1,i] = res._results.params[1]
        stdevs[1,i] = res._results.HC0_se[1]
        r_sq[1,i] = res._results.rsquared_adj
        obs[1,i] = res.nobs

        #Assets regression
        mod = sm.OLS(logc_diff[1:],sm.add_constant(np.transpose(np.vstack([top_assets[0:-1]]))), missing='drop')
        res = mod.fit()
        coeffs[2,i] = res._results.params[1]
        stdevs[2,i] = res._results.HC0_se[1]
        r_sq[2,i] = res._results.rsquared_adj
        obs[2,i] = res.nobs

        #Horeserace regression
        mod = sm.OLS(logc_diff[1:],sm.add_constant(np.transpose(np.vstack([logc_diff[0:-1],-log_trans_shk[0:-1],top_assets[0:-1]]))), missing='drop')
        res = mod.fit()
        coeffs[3,i] = res._results.params[1]
        stdevs[3,i] = res._results.HC0_se[1]
        coeffs[4,i] = res._results.params[2]
        stdevs[4,i] = res._results.HC0_se[2]
        coeffs[5,i] = res._results.params[3]
        stdevs[5,i] = res._results.HC0_se[3]
        r_sq[3,i] = res._results.rsquared_adj
        obs[3,i] = res.nobs

    paper_top = "\\begin{minipage}{\TableWidth}\n"
    paper_top += "  \\begin{table}\n"
    paper_top += "    \\caption{Micro Consumption Regression on Simulated Data} \\label{table:CGrowCross}\n"
    paper_top += "    \\begin{eqnarray} \n"
    paper_top += "\\CGrowCross    \\nonumber %\\\CGrowCrossBar \\nonumber \n"
    paper_top += "    \end{eqnarray}\n"

    slides_top = '\\begin{center} \n'

    header = "\\begin{tabular}{cd{4}d{4}d{5}ccc}  \n"
    header += "\\toprule  \n"
    header += "Model of     &                                &                                &                                 &                                       &                 \\\\  \n"
    header += "Expectations & \multicolumn{1}{c}{$ \chi $} & \multicolumn{1}{c}{$ \eta $} & \multicolumn{1}{c}{$ \\alpha $} & \multicolumn{1}{c}{$\\bar{R}^{2}$} &                   \n"

    F_panel = "\\\\ \\midrule \n \multicolumn{2}{l}{Frictionless}  \n"
    F_panel += "\\\\ &  {:.3f}".format(coeffs[0,0]) +"  &        &        & {:.3f}".format(r_sq[0,0]) +" &  "+ " %NotOnSlide   \n"
    F_panel += "\\\\ &  \multicolumn{1}{c}{(--)}  &        &        &  &  "+ " %NotOnSlide   \n"
    F_panel += "\\\\ &    &    {:.3f}".format(coeffs[1,0]) +"    &        & {:.3f}".format(r_sq[1,0])  +" &  "+" %NotOnSlide   \n"
    F_panel += "\\\\ &    &  \multicolumn{1}{c}{(--)}  &        &  &  "+ " %NotOnSlide   \n"
    F_panel += "\\\\ &    &        &     {:.3f}".format(coeffs[2,0]) +"   & {:.3f}".format(r_sq[2,0]) +" &  " +" %NotOnSlide   \n"
    F_panel += "\\\\ &    &       &  \multicolumn{1}{c}{(--)}  &  &  "+ " %NotOnSlide   \n"
    F_panel += "\\\\ &  {:.3f}".format(coeffs[3,0]) +"  &    {:.3f}".format(coeffs[4,0]) +"    &     {:.3f}".format(coeffs[5,0]) +"   & {:.3f}".format(r_sq[3,0]) +" &    \n"
    F_panel += "\\\\ & \multicolumn{1}{c}{(--)} &  \multicolumn{1}{c}{(--)} &  \multicolumn{1}{c}{(--)}  &  &  "+ " %NotOnSlide   \n"

    S_panel = "\\\\ \\midrule \n \multicolumn{2}{l}{Sticky}  \n"
    S_panel += "\\\\ &  {:.3f}".format(coeffs[0,1]) +"  &        &        & {:.3f}".format(r_sq[0,1]) +" &   %NotOnSlide   \n"
    S_panel += "\\\\ &  \multicolumn{1}{c}{(--)}  &        &        &  &  "+ " %NotOnSlide   \n"
    S_panel += "\\\\ &    &    {:.3f}".format(coeffs[1,1]) +"    &        & {:.3f}".format(r_sq[1,1]) +" &   %NotOnSlide   \n"
    S_panel += "\\\\ &    &  \multicolumn{1}{c}{(--)}  &        &  &  "+ " %NotOnSlide   \n"
    S_panel += "\\\\ &    &        &     {:.3f}".format(coeffs[2,1]) +"   & {:.3f}".format(r_sq[2,1]) +" &   %NotOnSlide   \n"
    S_panel += "\\\\ &    &       &  \multicolumn{1}{c}{(--)}  &  &  "+ " %NotOnSlide   \n"
    S_panel += "\\\\ &  {:.3f}".format(coeffs[3,1]) +"  &    {:.3f}".format(coeffs[4,1]) +"    &     {:.3f}".format(coeffs[5,1]) +"   & {:.3f}".format(r_sq[3,1]) +" &    \n"
    S_panel += "\\\\ & \multicolumn{1}{c}{(--)} &  \multicolumn{1}{c}{(--)} &  \multicolumn{1}{c}{(--)}  &  &  "+ " %NotOnSlide   \n"

    paper_bot = "  \\\\ \\bottomrule \\\\\n"
    paper_bot += "  \multicolumn{5}{p{0.9\\textwidth}}{\\footnotesize \\textbf{Notes}: $\\mathbb{E}_{t,i}$ is the expectation from the perspective of person $i$ in period $t$; $\\bar{a}$ is a dummy variable indicating that agent $i$ is in the top 99 percent of the normalized $a$ distribution.  Simulated sample size is large enough such that standard errors are effectively zero.  Sample is restricted to households with positive income in period $t$. The notation ``(---)'' indicates that standard errors are close to zero, given the very large simulated sample size.}\n"
    paper_bot += "\end{tabular}  \n"
    paper_bot += "\end{table}\n"
    paper_bot += "\end{minipage}\n"
    paper_bot += "\ifthenelse{\\boolean{StandAlone}}{\end{document}}{} \n"

    slides_bot = "\\\\ \\bottomrule  \n"
    slides_bot += "\end{tabular}  \n"
    slides_bot += '\\end{center} \n'

    # Make table for paper
    paper_output = paper_top + header + F_panel + S_panel + paper_bot
    with open(tables_dir + out_filename + '.tex','w') as f:
        f.write(paper_output)
        f.close()

    # Make tables for slides
    slidesF_output = slides_top + header + F_panel + slides_bot
    with open(tables_dir + out_filename + '_SlidesF.tex','w') as f:
        f.write(slidesF_output)
        f.close()

    slidesS_output = slides_top + header + S_panel + slides_bot
    with open(tables_dir + out_filename + '_SlidesS.tex','w') as f:
        f.write(slidesS_output)
        f.close()


def makeuCostVsPiFig(uCost_filename):
    '''
    Make two versions of a figure that plots the cost of stickiness vs updating probability.
    Saves pdf files to the figures directory.

    Parameters
    ----------
    uCost_filename : str
        Name of data file, as a two line csv.  First line is UpdatePrb, second line is uCost.

    Returns
    -------
    None
    '''
    data = np.genfromtxt(results_dir + uCost_filename +'.csv',delimiter=',')
    UpdatePrbVec = data[0,:]
    uCostVec = data[1,:]

    # Plot uCost vs Pi
    f = LinearInterp(UpdatePrbVec,uCostVec*10000)
    plt.plot(UpdatePrbVec,uCostVec*10000,color='#1f77b4')
    plt.plot([UpdatePrbBase,UpdatePrbBase],[0.,f(UpdatePrbBase)],'--k') # Add dashed line at Pi=0.25
    plt.xlim([0.05,1.0])
    plt.ylim([0.0,30.0])
    plt.xlabel(r'Probability of updating information $\Pi$')
    plt.ylabel('Cost of stickiness $\omega$ ($10^{-4}$)')
    plt.savefig(figures_dir + 'uCostvsPi.pdf')
    plt.savefig(figures_dir + 'uCostvsPi.png')
    plt.savefig(figures_dir + 'uCostvsPi.jpg')
    plt.savefig(figures_dir + 'uCostvsPi.svg')
    plt.show()
    plt.close()

    # Plot uCost vs 1/Pi
    plt.plot(1./UpdatePrbVec,uCostVec*10000,color='#1f77b4')
    plt.plot([1./UpdatePrbBase,1./UpdatePrbBase],[0.,f(UpdatePrbBase)],'--k') # Add dashed line at Pi=0.25
    plt.xlim([1.0,16.0])
    plt.ylim([0.0,35.0])
    plt.xlabel('Expected periods between information updates $\Pi^{-1}$')
    plt.ylabel('Cost of stickiness $\omega$ ($10^{-4}$)')
    plt.savefig(figures_dir + 'uCostvsPiInv.pdf')
    plt.savefig(figures_dir + 'uCostvsPiInv.png')
    plt.savefig(figures_dir + 'uCostvsPiInv.jpg')
    plt.savefig(figures_dir + 'uCostvsPiInv.svg')
    plt.show()
    plt.close()


def makeValueVsAggShkVarFig(value_filename):
    '''
    Parameters
    ----------
    value_filename : str
        Name of data file, as a two line csv.  First line is PermShkAggVar, second line is birth value.

    Returns
    -------
    None
    '''
    data = np.genfromtxt(results_dir + value_filename +'.csv',delimiter=',')
    PermShkAggVarVec = data[0,:]
    vVec = data[1,:]

    # Plot birth value vs PermShkAggVar
    f = LinearInterp(PermShkAggVarVec*10**5,vVec)
    vBot = np.min(vVec)
    vTop = np.max(vVec)
    plt.plot(PermShkAggVarVec*10**5,vVec,color='#1f77b4')
    plt.plot([PermShkAggVarBase*10**5,PermShkAggVarBase*10**5],[vBot,f(PermShkAggVarBase*10**5)],'--k')
    plt.ylim(vBot,vTop)
    plt.xlabel('Variance of aggregate permanent shocks $\sigma^2_\Psi$ ($10^{-5}$)')
    plt.ylabel('Expected value at birth $V(W,\cdot)$')
    plt.tight_layout()
    plt.savefig(figures_dir + 'ValueVsPermShkAggVar.pdf')
    plt.savefig(figures_dir + 'ValueVsPermShkAggVar.png')
    plt.savefig(figures_dir + 'ValueVsPermShkAggVar.jpg')
    plt.savefig(figures_dir + 'ValueVsPermShkAggVar.svg')
    plt.show()
    plt.close()


def makeValueVsPiFig(value_filename):
    '''
    Parameters
    ----------
    value_filename : str
        Name of data file, as a two line csv.  First line is UpdatePrb, second line is birth value.

    Returns
    -------
    None
    '''
    data = np.genfromtxt(results_dir + value_filename +'.csv',delimiter=',')
    UpdatePrbVec = data[0,:]
    vVec = data[1,:]

    # Plot birth value vs 1/UpdatePrb
    f = LinearInterp(UpdatePrbVec,vVec)
    vBot = f(1./16)
    vTop = np.max(vVec)
    plt.plot(1./UpdatePrbVec,vVec,color='#1f77b4')
    plt.plot([1./UpdatePrbBase,1./UpdatePrbBase],[vBot,f(UpdatePrbBase)],'--k')
    plt.xlim([1.0,16.0])
    plt.ylim(vBot,vTop)
    plt.xlabel('Expected periods between information updates $\Pi^{-1}$')
    plt.ylabel('Expected value at birth $V(W,\cdot)$')
    plt.tight_layout()
    plt.savefig(figures_dir + 'ValueVsPi.pdf')
    plt.savefig(figures_dir + 'ValueVsPi.png')
    plt.savefig(figures_dir + 'ValueVsPi.jpg')
    plt.savefig(figures_dir + 'ValueVsPi.svg')
    plt.show()
    plt.close()



