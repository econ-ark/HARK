This document decribes the file structure and how to run the code in cAndCwithStickyE.
The vast majority of the project is written in Python 2.7, but some empirical work uses Stata.

To run the code, you should install the Anaconda distribution of Python 2.7 (from
www.continuum.io), run Spyder, and execute the MAIN file; see notes below.  All
packages used by the StickyE project are included in the base distribution of Anaconda
or included in this archive.  The only exception occurs if you try to run the "beta
dist" specification of the HA-DSGE model, which is commented out in the parameters
file.  To successfully run this exercise, you must install the packages joblib and
dill by typing "conda install joblib" and "conda install dill" at a command prompt.

NOTE ON SPYDER: Anaconda comes with an IDE called Spyder.  On most operating systems,
Spyder can be run by typing "spyder" at a command prompt.  On Windows 10, this does
not seem to work, and Spyder needs to be opened from the Start menu (usually in the
Anaconda2 submenu).

NOTE ON STATA: The MAIN file includes a boolean called use_stata.  If set to True, the
regressions will be run in Stata; if False, they will be run using the statsmodels.api
package in Python.  The only difference is that the Stata code is able to produce the
KP statistic, but Python is not.  To successfully use this option, you must set the
stata_exe variable in the StickyEparams.py to point to a valid Stata executable.
You must set a valid path in stata_exe if the make_emp_table or make_histogram booleans
are set to True.

NOTE ON MEMORY: The heterogeneous agents models are set to simulate 20,000 households
for about 21,000 periods, saving the entire history of several household-level variables
in in double precision.  As such, running the SOE or HA-DSGE models is very memory
intensive.  We recommend that you run the code on a computer with at least 32GB of RAM
to ensure that memory problems are not encountered.  To run the SOE and HA-DSGE models
back-to-back on the same call to the MAIN file, we recommend that you have at least
64 GB of memory.

NOTE ON TIME: On a reasonably modern CPU, the RA model can solve and simulate in about
25-30 seconds (and has basically no memory requirement).  The SOE model takes about
15-20 minutes to solve and simulate (but see note on memory).  The HA-DSGE mode takes
roughly 16 hours to solve the sticky and frictionless versions, and will display progress
toward finding the equilibrium aggregate saving rule after each iteration.  The very
long run time for HA-DSGE is largely due to the extremely large number of periods.
Exercises that examine the relationship between parameters and value-at-birth (e.g. the
"cost of stickiness" exercise) take about 4 hours each to run.  Time series regressions
take a few seconds to run in Python or two minutes in Stata (if use_stata is True).


StickyE project files include:

1) StickyE_MAIN.py
This is the file to run to produce all the results and tables in the paper.
There are a number of switches at the beginning of the file for the user to choose
which results and files to produce.  Note: Setting do_DSGE_markov=True will result in VERY long run time.

2) StickyEparams.py
This file loads the calibrated parameters, defines non-economic parameters (grid sizes,
etc), constructs parameters calculated from the primitive calibrated parameters, and
defines dictionaries for constructing objects in the MAIN file.  It also sets:
i) The number of periods and number of agents to simulate. This has direct consequences for run time.
ii) The Stata path - this must be set to a valid Stata executable on the user's
machine if you want to reproduce the Stata parts of the code.

3) StickyEmodel.py
This file contains the model code, building heavily off the rest of the HARK toolbox.

4) StickyEtools.py
This file does most of the post-processing after simulation.  It runs regression, calculates
equilibrium outcomes, etc.  It is a collection of functions that are called by StickyE_MAIN.py

5) StickyETimeSeries.do
This is a Stata do file. The regressions are run in Stata in order to produce the KP statistic.

Relative locations of other folders are defined near the top of StickyEparams.py.
Referenced folders include:

1) ./Results (results_dir)
This folder contains saved data from the model simulations.

2) ./Tables (tables_dir)
This folder contains the Latex code output by functions in StickyEtools.py.
Some of these tables appear in the paper.

3) ./Figures (figures_dir)
This folder contains figures produced by functions in StickyEtools.py.
Some of these figures appear in the paper.

4) ./Calibration (calibration_dir)
This folder contains one line parameter values that are read by StickyEparams.py
and by the LaTeX file that produces the paper.
