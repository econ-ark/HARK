This document decribes the file structure and how to run the code in cAndCwithStickyE.
The vast majority of the project is written in Python 2.7, but some empirical work uses Stata.

To run the code, you should install the Anaconda distribution of Python 2.7 (from
www.continuum.io), run Spyder, and execute the MAIN file; see notes below.  All
packages used by the StickyE project are included in the base distribution of Anaconda
or included in this archive.  The only exception occurs if you try to run the "beta
dist" specification of the HA-DSGE model, which is commented out in the parameters
file.  To successfully run this exercise, you must install the packages joblib and
dill by typing "conda install joblib" and "conda install dill" at a command prompt.

The MAIN file includes a boolean called use_stata.  If set to True, the time series
regressions will be run in Stata; if False, they will be run using the statsmodels.api
package in Python.  The only difference is that the Stata code is able to produce the
KP statistic, but Python is not.  To successfully use this option, you must set the
stata_exe variable in the parameters file to point to a valid Stata executable.

StickyE project files are listed below.  Other Python files are part of the HARK
toolkit, distributed by the Econ-ARK project.

1) StickyE_MAIN.py
This is the file to run to produce all the results and tables in the paper.
There are a number of switches at the beginning of the file for the user to choose
which results and files to produce.  Note: Setting do_DSGE_markov=True will result
a in VERY long run time.

2) StickyEparams.py
This file contains all the calibration data
It also sets:
i) The number of periods and number of agents to simulate. This has direct consequences for run time.
ii) The Stata path - this must be set to a valid Stata executable on the user's
machine if you want to reproduce the Stata parts of the code.

3) StickyEmodel.py
This file contains the model code, building heavily off the rest of the HARK toolbox.

4) StickyEtools.py
This file does most of the post-processing after simulation.  It runs regression, calculates equilibrium outcomes, etc.
It is a collection of functions that are called by StickyE_MAIN.py

5) StickyETimeSeries.do
This is a Stata do file. The regressions are run in Stata in order to produce the KP statistic.

6) StickyE_NO_MARKOV.py
An alternate version of the MAIN file, using versions of the models without the Markov growth
process.  Results for these models do not appear in the paper.

The subfolders include:

1) Results
This folder contains saved data from the model simulations

2) Tables
This folder contains the Latex code output by functions in StickyEtools.py.
Some of these tables appear in the paper.

3) Figures
This folder contains figures produced by functions in StickyEtools.py.
Some of these figures appear in the paper.

4) Calibration
This folder contains one line parameter values that are read by StickyEparams.py
and by the LaTeX file that produces the paper.