This document decribes the file structure and how to run the code in cAndCwithStickyE

1) StickyE_MAIN.py
This is the file to run to produce all the results and tables in the paper.
There are a number of switches at the beginning of the file for the user to choose which results and files to produce.
Note - setting do_DSGE_markov=True will results in VERY long run time.

2) StickyEparams.py
This file contains all the calibration data
It also sets:
i) The number of periods and number of agents to simulate. This has direct consequences for run time.
ii) The Stata path - this must be set to a valid Stata executable on the user's machine if you want to reproduce the Stata parts of the code.

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