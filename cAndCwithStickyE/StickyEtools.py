"""
This module holds some data tools used in the cAndCwithStickyE project.
"""

import os
import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.sandbox.regression.gmm as smsrg
import subprocess
from HARKutilities import getLorenzShares

def mystr1(number):
    if not np.isnan(number):
        out = "{:.3f}".format(number)
    else:
        out = ''
    return out


def mystr2(number):
    if not np.isnan(number):
        out = "{:1.2f}".format(number*10000) + '\\text{e-4}'
    else:
        out = ''
    return out


def makeStickyEdataFile(Economy,ignore_periods,description='',filename=None,save_data=False,calc_micro_stats=True):
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
        paper draft.  This causes huge memory issues 
        
    Returns
    -------
    output_string : str
        Large string with descriptive statistics. Also saved to a logfile if
        filename is not None.
    '''
    # Extract time series data from the economy
    if hasattr(Economy,'agents'): # If this is a heterogeneous agent specification...
        #PermShkAggHist needs to be shifted one period forward
        PlvlAgg_hist = np.cumprod(np.concatenate(([1.0],Economy.PermShkAggHist[:-1]),axis=0))
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
        
        if calc_micro_stats: # Only calculate stats if requested.  This is a memory hog with many simulated periods
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
    sigma_meas_err = np.std(DeltaLogC)*0.375
    np.random.seed(10)
    Measurement_Error = sigma_meas_err*np.random.normal(0.,1.,LogC.size)
    LogC_me = LogC + Measurement_Error
    DeltaLogC_me = LogC_me[1:] - LogC_me[0:-1]
    
    # Apply measurement error to long delta LogC
    LogC_long = np.log(ClvlAgg_hist)
    LogC_long_me = LogC_long + sigma_meas_err*np.random.normal(0.,1.,LogC_long.size)
    Delta8LogC_me = (LogC_long_me[8:] - LogC_long_me[:-8])[(ignore_periods-7):]
    
    # Make and return the output string, beginning with descriptive statistics
    output_string = description + '\n\n\n'
    output_string += 'Average aggregate asset-to-productivity ratio = ' + str(np.mean(AnrmAgg_hist[ignore_periods:])) + '\n'
    output_string += 'Average aggregate consumption-to-productivity ratio = ' + str(np.mean(CnrmAgg_hist[ignore_periods:])) + '\n'
    output_string += 'Stdev of log aggregate asset-to-productivity ratio = ' + str(np.std(np.log(AnrmAgg_hist[ignore_periods:]))) + '\n'
    output_string += 'Stdev of change in log aggregate consumption level = ' + str(np.std(DeltaLogC)) + '\n'
    output_string += 'Stdev of change in log aggregate output level = ' + str(np.std(DeltaLogY)) + '\n'
    output_string += 'Stdev of change in log aggregate assets level = ' + str(np.std(DeltaLogA)) + '\n'
    if hasattr(Economy,'agents') and calc_micro_stats: # This block only runs for heterogeneous agents specifications
        output_string += 'Cross section stdev of log individual assets = ' + str(np.mean(np.std(Loga,axis=1))) + '\n'
        output_string += 'Cross section stdev of log individual consumption = ' + str(np.mean(np.std(Logc,axis=1))) + '\n'
        output_string += 'Cross section stdev of log individual productivity = ' + str(np.mean(np.std(Logp,axis=1))) + '\n'
        output_string += 'Cross section stdev of log individual non-zero income = ' + str(np.mean(np.std(Logy_trimmed,axis=1))) + '\n'
        output_string += 'Cross section stdev of change in log individual assets = ' + str(np.std(DeltaLoga_trimmed)) + '\n'
        output_string += 'Cross section stdev of change in log individual consumption = ' + str(np.std(DeltaLogc_trimmed)) + '\n'
        output_string += 'Cross section stdev of change in log individual productivity = ' + str(np.std(DeltaLogp_trimmed)) + '\n'
    output_string += '\n\n'    
     
    # Save the results to a logfile if requested
    if filename is not None:
        with open('./Results/' + filename + 'Results.txt','w') as f:
            f.write(output_string)
            f.close()
            
        if save_data:
            DataArray = (np.vstack((np.arange(DeltaLogC.size),DeltaLogC_me,DeltaLogC,DeltaLogY,A,BigTheta,Delta8LogC,Delta8LogY,Delta8LogC_me,Measurement_Error))).transpose()
            VarNames = ['time_period','DeltaLogC_me','DeltaLogC','DeltaLogY','A','BigTheta','Delta8LogC','Delta8LogY','Delta8LogC_me']
            if hasattr(Economy,'MrkvNow'):
                DataArray = np.hstack((DataArray,np.reshape(Mrkv,(Mrkv.size,1))))
                VarNames.append('MrkvState')
            if hasattr(Economy,'MrkvNow') & ~hasattr(Economy,'Rfree') and hasattr(Economy,'agents'):
                DataArray = np.hstack((DataArray,np.reshape(R,(R.size,1))))
                VarNames.append('R')
            with open('./Results/' + filename + 'Data.txt.','wb') as f:
                my_writer = csv.writer(f, delimiter = '\t')
                my_writer.writerow(VarNames)
                for i in range(DataArray.shape[0]):
                    my_writer.writerow(DataArray[i,:])
                f.close()
    
    return output_string # Return output string


def runStickyEregressions(infile_name,interval_size,meas_err,sticky):
    '''
    Runs regressions for the main tables of the StickyC paper and produces a LaTeX
    table with results (one "panel" at a time).
    
    Parameters
    ----------
    infile_name : str
        Name of tab-delimited text file with simulation data.  Assumed to be in
        the directory ./Results/, and was almost surely generated by makeStickyEdataFile
        unless we resort to fabricating simulated data.  THAT'S A JOKE, FUTURE REFEREES.
    interval_size : int
        Number of periods in each sub-interval.
    meas_err : bool
        Indicator for whether to add measurement error to DeltaLogC.
    sticky : bool
        Indicator for whether these results used sticky expectations.
        
    Returns
    -------
    panel_text : str
        String with one panel's worth of LaTeX input.
    '''
    # Read in the data from the infile
    with open('./Results/' + infile_name + '.txt') as f:
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
        if has_mrkv:
            Mrkv_hist[i] = int(float(all_data[j][9]))
        if has_R:
            R[i] = float(all_data[j][10])
    
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
        BigTheta_n = BigTheta[start:end]
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
    
    #Hard code variance of measurement error - better to pass this in from the data file
    #Can replace this once new data files are produced
    sigma_meas_err = np.std(DeltaLogC)*0.375
    
    N_out = [C_successes_95,Y_successes_95,N,np.mean(InstrRsqVec),sigma_meas_err**2]
    
    # Make results table and return it
    panel_text = makeResultsPanel(Coeffs=np.mean(CoeffsArray,axis=0),
                     StdErrs=np.mean(StdErrArray,axis=0),
                     Rsq=np.mean(RsqArray,axis=0),
                     Pvals=np.mean(PvalArray,axis=0),
                     OID=np.mean(OIDarray,axis=0),
                     Counts=N_out,
                     meas_err=meas_err,
                     sticky=sticky)
    return panel_text

def runStickyEregressionsInStata(infile_name,interval_size,meas_err,sticky):
    '''
    Runs regressions for the main tables of the StickyC paper in Stata
    and produces a LaTeX table with results (one "panel" at a time).
    Running in Stata allows production of the KP-statistic
    
    Parameters
    ----------
    infile_name : str
        Name of tab-delimited text file with simulation data.  Assumed to be in
        the directory ./Results/, and was almost surely generated by makeStickyEdataFile
        unless we resort to fabricating simulated data.  THAT'S A JOKE, FUTURE REFEREES.
    interval_size : int
        Number of periods in each sub-interval.
    meas_err : bool
        Indicator for whether to add measurement error to DeltaLogC.
    sticky : bool
        Indicator for whether these results used sticky expectations.
        
    Returns
    -------
    panel_text : str
        String with one panel's worth of LaTeX input.
    '''
    dofile = "StataRegressions.do"
    infile_name_full = os.path.abspath("results\\"+infile_name+".txt")
    if meas_err:
        meas_err_stata = 1
    else:
        meas_err_stata = 0
    #Is there are way to make a more robust call to Stata than this?
    cmd = ["C:\Program Files (x86)\Stata14\stataMP-64", "do", dofile, infile_name_full,str(interval_size),str(meas_err_stata)]
    ## Run do-file
    subprocess.call(cmd,shell = 'true') 
    stata_output = pd.read_csv(os.path.abspath("results\\temp.txt"), sep=',',header=0)
    
    # Make results table and return it
    panel_text = makeResultsPanel(Coeffs=stata_output.CoeffsArray,
                     StdErrs=stata_output.StdErrArray,
                     Rsq=stata_output.RsqArray,
                     Pvals=stata_output.PvalArray,
                     OID=stata_output.OIDarray,
                     Counts=stata_output.ExtraInfo,
                     meas_err=meas_err,
                     sticky=sticky)
    return panel_text
    
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


def makeResultsPanel(Coeffs,StdErrs,Rsq,Pvals,OID,Counts,meas_err,sticky):
    '''
    Make one panel of simulated results table.  A panel has all results with/out
    measurement error for the sticky or frictionless version
    
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
        
    Returns
    -------
    output : str
        Text string with one panel of LaTeX input.
    '''
    # Define Delta log C text and expectations text
    if sticky & meas_err:
        DeltaLogC = '$\Delta \log \perc{\mathbf{C}}_{t}^*$'
        DeltaLogC1 = '$\Delta \log \perc{\mathbf{C}}_{t+1}^*$'
    else:
        if sticky:
            DeltaLogC = '$\Delta \log \perc{\mathbf{C}}_{t}$'
            DeltaLogC1 = '$\Delta \log \perc{\mathbf{C}}_{t+1}$'
        else:
            DeltaLogC = '$\Delta \log \mathbf{C}_{t}$'
            DeltaLogC1 = '$\Delta \log \mathbf{C}_{t+1}$'
    if sticky:
        Expectations = 'Sticky'
        DeltaLogY1 = '$\Delta \log \perc{\mathbf{Y}}_{t+1}$'
    else:
        Expectations = 'Frictionless'
        DeltaLogY1 = '$\Delta \log \mathbf{Y}_{t+1}$'
    if sticky:
        if meas_err:
            MeasErr = ' (with measurement error); $\perc{\mathbf{C}}_{t+1}^* =\perc{\mathbf{C}}_{t+1}\\times \\xi_t$'
        else:
            MeasErr = ' (no measurement error)'
    else:
        MeasErr = ''
        
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

    output = '\\\\ \hline \multicolumn{6}{l}{' + Expectations + ' : ' + DeltaLogC1 + MeasErr + '} \n'
    output += '\\\\ \multicolumn{1}{c}{' + DeltaLogC + '} & \multicolumn{1}{c}{' + DeltaLogY1 +'} & \multicolumn{1}{c}{$A_{t}$} & & & \n'
    output += '\\\\ ' + mystr1(Coeffs[0]) + sigFunc(Coeffs[0],StdErrs[0]) + ' & & & OLS & ' + mystr1(Rsq[0]) + ' & ' + mystr1(np.nan) + '\n'   
    output += '\\\\ (' + mystr1(StdErrs[0]) + ') & & & & & \n'   
    if sticky & meas_err:
        output += '\\\\ ' + mystr1(Coeffs[1]) + sigFunc(Coeffs[1],StdErrs[1]) + ' & & & IV & ' + mystr1(Rsq[1]) + ' & ' + mystr1(Pvals[1]) + '\n'   
        output += '\\\\ (' + mystr1(StdErrs[1]) + ') & & & & &' + mystr1(OID[1]) + '\n'   
    if (not sticky) or meas_err:
        output += '\\\\ & ' + mystr1(Coeffs[2]) + sigFunc(Coeffs[2],StdErrs[2]) + ' & & IV & ' + mystr1(Rsq[2]) + ' & ' + mystr1(Pvals[2]) + '\n'     
        output += '\\\\ & (' + mystr1(StdErrs[2]) + ') & & & &' + mystr1(OID[2]) + '\n'    
        output += '\\\\ & & ' + mystr2(Coeffs[3]) + sigFunc(Coeffs[3],StdErrs[3]) + ' & IV & ' + mystr1(Rsq[3]) + ' & ' + mystr1(Pvals[3]) + '\n'   
        output += '\\\\ & & (' + mystr2(StdErrs[3]) + ') & & &' + mystr1(OID[3]) + '\n'    
        output += '\\\\ ' + mystr1(Coeffs[4]) + sigFunc(Coeffs[4],StdErrs[4]) + ' & ' + mystr1(Coeffs[5]) + sigFunc(Coeffs[5],StdErrs[5]) + ' & ' + mystr2(Coeffs[6]) + sigFunc(Coeffs[6],StdErrs[6]) + ' & IV & ' + mystr1(Rsq[4]) + ' & ' + mystr1(Pvals[4]) + '\n'     
        output += '\\\\ (' + mystr1(StdErrs[4]) + ') & (' + mystr1(StdErrs[5]) + ') & (' + mystr2(StdErrs[6]) + ') & & & \n'
        output += '\\\\ \multicolumn{6}{l}{Memo: For instruments $\mathbf{Z}_{t}$, ' + DeltaLogC1 + ' $= \mathbf{Z}_{t} \zeta,~~\\bar{R}^{2}=$' + mystr1(Counts[3]) +' }  \n'
    if meas_err & sticky:
        output += '\\\\ \multicolumn{6}{l}{$\\var(\\xi_t)=$' + mystr2(Counts[4]) + '} \n'
    if Counts[0] is not None and Counts[2] > 1 and False:
        output += '\\\\ \multicolumn{6}{c}{Horserace coefficient on ' + DeltaLogC + ' significant at 95\% level for ' + str(Counts[0]) + ' of ' + str(Counts[2]) + ' subintervals.} \n'
        output += '\\\\ \multicolumn{6}{c}{Horserace coefficient on $\mathbb{E}[\Delta \log \mathbf{Y}_{t+1}]$ significant at 95\% level for ' + str(Counts[1]) + ' of ' + str(Counts[2]) + ' subintervals.} \n'
    
    return output
        
        
def makeResultsTable(caption,panels,counts,filename):
    '''
    Make a results table by piecing together one or more panels.
    
    Parameters
    ----------
    caption : str
        Text to apply at the start of the table as a title.
    panels : [str]
        List of strings with one or more panels, usually made by makeResultsPanel.
    counts : int
        List of two integers: [interval_length, interval_count]
    filename : str
        Name of the file in which to save output (in the ./Tables/ directory).
        
    Returns
    -------
    None
    '''
    note = '\\multicolumn{6}{p{0.8\\textwidth}}{\\footnotesize \\textbf{Notes:} '
    if counts[1] > 1:
        note += 'Reported statistics are the average values for ' + str(counts[1]) + ' subsamples of ' + str(counts[0]) + ' simulated quarters each.  '
        note += 'Bullets indicate that the average subsample coefficient divided by average subsample standard error is outside of the inner 90\%, 95\%, and 99\% of the standard normal distribution.  '
    else:
        note += 'Reported statistics are for a single simulation of ' + str(counts[0]) + ' quarters.  '
        note += 'Stars indicate statistical significance at the 90\%, 95\%, and 99\% levels, respectively.  '
    note += 'Instruments $\\textbf{Z}_t = \\{\Delta \log \mathbf{C}_{t-2}, \Delta \log \mathbf{C}_{t-3}, \Delta \log \mathbf{Y}_{t-2}, \Delta \log \mathbf{Y}_{t-3}, A_{t-2}, A_{t-3}, \Delta_8 \log \mathbf{C}_{t-2}, \Delta_8 \log \mathbf{Y}_{t-2}   \\}$.'
    note += '}'
        
    
    output = '\\begin{table} \caption{' + caption + '}\n'
    output += '\centering \n'
    output += '$ \Delta \log \mathbf{C}_{t+1} = \\varsigma + \chi \Delta \log \mathbf{C}_t + \eta \mathbb{E}_t[\Delta \log \mathbf{Y}_{t+1}] + \\alpha A_t + \epsilon_{t+1} $ \\\\  \n'
    output += '\\begin{tabular}{d{4}d{4}d{5}cd{4}d{5}}\n \\toprule \n'
    output += '\multicolumn{3}{c}{Expectations : Dep Var} & OLS &  \multicolumn{1}{c}{2${}^{\\text{nd}}$ Stage}  &  \multicolumn{1}{c}{KP $p$-val} \n'
    output += '\\\\ \multicolumn{3}{c}{Independent Variables} & or IV & \multicolumn{1}{c}{$\\bar{R}^{2} $} & \multicolumn{1}{c}{Hansen J $p$ val} \n'
    
    for panel in panels:
        output += panel
        
    output += '\\\\ \hline \n ' + note + ' \n \\\\ \hline \hline \n'
    output += '\end{tabular} \n'
    output += '\end{table} \n'
    
    with open('./Tables/' + filename + '.txt','w') as f:
        f.write(output)
        f.close()
        
 