# Heterogeneous Agents Resources and toolKit (HARK)
pre-release 0.8.0 - October 2, 2017

Click the Badge for Citation Info.
[![DOI](https://zenodo.org/badge/50448254.svg)](https://zenodo.org/badge/latestdoi/50448254)


Table of Contents:

* [I.   Introduction](#i-introduction)
* [II.  Quick start guide](#ii-quick-start-guide)
* [III. List of files in repository](#iii-list-of-files-in-repository)
* [IV.  Warnings and disclaimers](#iv-warnings-and-disclaimers)
* [V.   License Information](#v-license)


## I. INTRODUCTION

Welcome to HARK!  We are tremendously excited you're here.  HARK is
very much a work in progress, but we hope you find it valuable.  We 
*really* hope you find it so valuable that you decide to contribute
to it yourself.  This document will tell you how to get HARK up and
running on your machine, and what you will find in HARK once you do.

If you have any comments on the code or documentation, we'd love to
hear from you!  Our email addresses are:

* Chris Carroll: ccarroll@llorracc.org
* Matthew White: mnwhite@gmail.com
* Nathan Palmer: Nathan.Palmer@ofr.treasury.gov
* David Low: David.Low@cfpb.gov
* Alexander Kaufman: akaufman10@gmail.com

GitHub repository:    https://github.com/econ-ark/HARK

Online documentation: https://econ-ark.github.io/HARK

User guide: /Documentation/HARKmanual.pdf (in the repository)


## II. QUICK START GUIDE

This is going to be easy, friend.  HARK is written in Python, specifically the
Anaconda distribution of Python.  Follow these easy steps to get HARK going:

1) Go to https://www.continuum.io/downloads and download Anaconda for your
operating system; be sure to get the version for Python 2.7

2) Install Anaconda, using the instructions provided on that page.  Now you have
installed everything you need to run most of HARK.  But you still need to get HARK
on your machine.

3) To get HARK on your machine, you should know that HARK is managed with version 
control software called "Git".  HARK is hosted on a website called "GitHub" devoted
to hosting projects managed with Git. 
   
   If you don't want to know more than that, you don't have to. Go to HARK's page
on GitHub (https://github.com/econ-ark/HARK), click the "Clone or download" button
in the upper right hand corner of the page, then click "Download ZIP". Unzip it
into an empty directory. Maybe call that directory /HARK ?  The choice is yours.
   
   You can also clone HARK off GitHub using Git.  This is slightly more difficult,
because it involves installing Git on your machine and learning a little about
how to use Git.  We believe this is an investment worth making, but it is up to you.
To learn more about Git, read the documentation at https://git-scm.com/documentation 
or visit many other great Git resources on the internet.

4) Open Spyder, an interactive development environment (IDE) for Python
(specifically, iPython).  On Windows, open a command prompt and type "spyder".
On Linux, open the command line and type "spyder".  On Mac, open the command
line and type "spyder".

5) Navigate to the directory where you put the HARK files.  This can be done
within Spyder by doing "import os" and then using os.chdir() to change directories.  
chdir works just like cd at a command prompt on most operating systems, except that
it takes a string as input: os.chdir('Music') moves to the Music subdirectory
of the current working directory.

6) Run one of HARK's modules.  You can either type "run MODULENAME" after navigating
to the correct directory (see step 5), or click the green arrow "run" button in
Spyder's toolbar after opening the module in the editor.  Every module should
do *something* when run, but that something might not be very interesting in
some cases.  For starters, check out /ConsumptionSavingModel/ConsIndShockModel.py
See section III below for a full list of modules that produce non-trivial output.

7) The Python environment can be cleared or reset with ctrl+.  Note that
this will also change the current working directory back to its default.
To change the default directory (the "global working directory"), see
Tools-->Preferences-->Global working directory; you might need to restart
Spyder for the change to take effect.

8) Read the more complete documentation in [HARKmanual.pdf](https://github.com/econ-ark/HARK/blob/master/Documentation/HARKmanual.pdf).

9) OPTIONAL: If you want to use HARK's multithreading capabilities, you will
need two Python packages that do not come automatically with Anaconda: joblib
and dill.  Assuming you have the necessary permissions on your machine, the 
easiest way to do this is through Anaconda.  Go to the command line, and type
"conda install joblib" and then "conda install dill" (accept defaults if prompted).
If this doesn't work, but you have Git, you can just clone the packages directly
off GitHub.  Go to the command line and navigate to the directory you want to put
these packages in.  Then type "git clone https://github.com/joblib/joblib.git"
and then "git clone https://github.com/uqfoundation/dill".  Joblib should work
after this, but there is one more step to get dill working.  Navigate to dill's
directory in the command line, and then type "python setup.py build".  Then you
should have joblib and dill working on your machine.

Note: If you did not put joblib and dill in one of the paths in sys.path, you will
need to add the joblib and dill directories to sys.path.  The easiest way to do this
is to open up Anaconda, and type:

```python
import sys
sys.path.append('path_to_joblib_directory')
sys.path.append('path_to_dill_directory')
```

## III. LIST OF FILES IN REPOSITORY

This section contains descriptions of every file included in the HARK
repository at the time of the beta release, categorized for convenience.

Documentation files:
* [README.md](https://github.com/econ-ark/HARK/blob/master/README.md): The file you are currently reading.
* [Documentation/HARKdoc.pdf](https://github.com/econ-ark/HARK/blob/master/Documentation/HARKdoc.pdf): A mini-user guide produced for a December 2015 workshop on HARK, unofficially representing the alpha version.  Somewhat out of date.
* [Documentation/HARKmanual.pdf](https://github.com/econ-ark/HARK/blob/master/Documentation/HARKmanual.pdf): A user guide for HARK, written for the beta release at CEF 2016 in Bordeaux.  Should contain 90% fewer lies relative to HARKdoc.pdf.
    * [Documentation/HARKmanual.tex](https://github.com/econ-ark/HARK/blob/master/Documentation/HARKmanual.tex): LaTeX source for the user guide.  Open source code probably requires an open source manual as well.
* [Documentation/ConsumptionSavingModels.pdf](https://github.com/econ-ark/HARK/blob/master/Documentation/ConsumptionSavingModels.pdf): Mathematical descriptions of the various consumption-saving models in HARK and how they map into the code.
    * [Documentation/ConsumptionSavingModels.tex](https://github.com/econ-ark/HARK/blob/master/Documentation/ConsumptionSavingModels.tex): LaTeX source for the "models" writeup.
* [Documentation/NARK.pdf](https://github.com/econ-ark/HARK/blob/master/Documentation/NARK.pdf): Variable naming conventions for HARK, plus concordance with LaTeX variable definitions.  Still in development.   

Tool modules:
* HARKcore.py:
    Frameworks for "microeconomic" and "macroeconomic" models in HARK.
    We somewhat abuse those terms as shorthand; see the user guide for a
    description of what we mean.  Every model in HARK extends the classes
    AgentType and Market in this module.  Does nothing when run.
* HARKutilities.py:
    General purpose tools and utilities.  Contains literal utility functions
    (in the economic sense), functions for making discrete approximations
    to continuous distributions, basic plotting functions for convenience,
    and a few unclassifiable things.  Does nothing when run.
* HARKestimation.py:
    Functions for estimating models.  As is, it only has a few wrapper 
    functions for scipy.optimize optimization routines.  Will be expanded
    in the future with more interesting things.  Does nothing when run.
* HARKsimulation.py:
    Functions for generating simulated data.  Functions in this module have
    names like drawUniform, generating (lists of) arrays of draws from
    various distributions.  Does nothing when run.
* HARKinterpolation.py:
    Classes for representing interpolated function approximations.  Has
    1D-4D interpolation methods, mostly based on linear or cubic spline
    interpolation.  Will have ND methods in the future.  Does nothing when run.
* HARKparallel.py:
    Early version of parallel processing in HARK.  Works with instances of
    the AgentType class (or subclasses of it), distributing commands (as
    methods) to be run on a list of AgentTypes.  Only works with local CPU.
    The module also contains a parallel implentation of the Nelder-Mead
    simplex algorithm, poached from Wiswall and Lee (2011).  Does nothing
    when run.

Model modules:
* ConsumptionSavingModel/TractableBufferStockModel.py:
    A "tractable" model of consumption and saving in which agents face one
    simple risk with constant probability: that they will become permanently
    unemployed and receive no further income.  Unlike other models in HARK,
    this one is not solved by iterating on a sequence of one period problems.
    Instead, it uses a "backshooting" routine that has been shoehorned into
    the AgentType.solve framework.  Solves an example of the model when run,
    then solves the same model again using MarkovConsumerType.
* ConsumptionSavingModel/ConsIndShockModel.py:
    Consumption-saving models with idiosyncratic shocks to income.  Shocks
    are fully transitory or fully permanent.  Solves perfect foresight model,
    a model with idiosyncratic income shocks, and a model with idiosyncratic
    income shocks and a different interest rate on borrowing vs saving.  When
    run, solves several examples of these models, including a standard infinite
    horizon problem, a ten period lifecycle model, a four period "cyclical"
    model, and versions with perfect foresight and "kinked R".
* ConsumptionSavingModel/ConsPrefShockModel.py:
    Consumption-saving models with idiosyncratic shocks to income and multi-
    plicative shocks to utility.  Currently has two models: one that extends
    the idiosyncratic shocks model, and another that extends the "kinked R"
    model.  The second model has very little new code, and is created merely
    by merging the two "parent models" via multiple inheritance.  When run,
    solves examples of the preference shock models.
* ConsumptionSavingModel/ConsMarkovModel.py:
    Consumption-saving models with a discrete state that evolves according to
    a Markov rule.  Discrete states can vary by their income distribution,
    interest factor, and/or expected permanent income growth rate.  When run,
    solves four example models: (1) A serially correlated unemployment model
    with boom and bust cycles (4 states). (2) An "unemployment immunity" model
    in which the consumer occasionally learns that he is immune to unemployment
    shocks for the next N periods.  (3) A model with a time-varying permanent
    income growth rate that is serially correlated.  (4) A model with a time-
    varying interest factor that is serially correlated.
* ConsumptionSavingModel/ConsAggShockModel.py:
    Consumption-saving models with idiosyncratic and aggregate income shocks.
    Currently has a micro model with a basic solver (linear spline consumption
    function only, no value function), and a Cobb-Douglas economy for the
    agents to "live" in (as a "macroeconomy").  When run, solves an example of
    the micro model in partial equilibrium, then solves the general equilibrium
    problem to find an evolution rule for the capital-to-labor ratio that is
    justified by consumers' collective actions.
* FashionVictim/FashionVictimModel.py:
    A very serious model about choosing to dress as a jock or a punk.  Used to
    demonstrate micro and macro framework concepts from HARKcore.  It might be
    the simplest model possible for this purpose, or close to it.  When run,
    the module solves the microeconomic problem of a "fashion victim" for an
    example parameter set, then solves the general equilibrium model for an
    entire "fashion market" constituting many types of agents, finding a rule
    for the evolution of the style distribution in the population that is justi-
    fied by fashion victims' collective actions.
   
Application modules:
* SolvingMicroDSOPs/StructEstimation.py:
    Conducts a very simple structural estimation using the idiosyncratic shocks
    model in ConsIndShocksModel.  Estimates an adjustment factor to an age-varying
    sequence of discount factors (taken from Cagetti (2003)) and a coefficient
    of relative risk aversion that makes simulated agents' wealth profiles best
    match data from the 2004 Survey of Consumer Finance.  Also demonstrates
    the calculation of standard errors by bootstrap and can construct a contour
    map of the objective function.  Based on section 9 of Chris Carroll's
    lecture notes "Solving Microeconomic Dynamic Stochastic Optimization Problems".
* cstwMPC/cstwMPC.py:
    Conducts the estimations for the paper "The Distribution of Wealth and the
    Marginal Propensity to Consume" by Carroll, Slacalek, Tokuoka, and White (2016).
    Runtime options are set in SetupParamsCSTW.py, specifying choices such as:
    perpetual youth vs lifecycle, beta-dist vs beta-point, liquid assets vs net
    worth, aggregate vs idiosyncratic shocks, etc.  Uses ConsIndShockModel and
    ConsAggShockModel; can demonststrate HARK's "macro" framework on a real model.
* cstwMPC/MakeCSTWfigs.py:
    Makes various figures for the text of the cstwMPC paper.  Requires many output
    files produced by cstwMPC.py, from various specifications, which are not
    distributed with HARK.  Has not been tested in quite some time.
* cstwMPC/MakeCSTWfigsForSlides.py:
    Makes various figures for the slides for the cstwMPC paper.  Requires many
    output files produced by cstwMPC.py, from various specifications, which are not
    distributed with HARK.  Has not been tested in quite some time.

Parameter and data modules:
* ConsumptionSaving/ConsumerParameters.py:
    Defines dictionaries with the minimal set of parameters needed to solve the
    models in ConsIndShockModel, ConsAggShockModel, ConsPrefShockModel, and
    ConsMarkovModel.  These dictionaries are used to make examples when those
    modules are run.  Does nothing when run itself.
* SolvingMicroDSOPs/SetupSCFdata.py:
    Imports 2004 SCF data for use by SolvingMicroDSOPs/StructEstimation.py.
* cstwMPC/SetupParamsCSTW.py:
    Loads calibrated model parameters for cstwMPC.py, chooses specification.
* FashionVictim/FashionVictimParams.py:
    Example parameters for FashionVictimModel.py, loaded when that module is run.

Test modules:
* Testing/ComparisonTests.py:
    Early version of unit testing for HARK, still in development.  Compares
    the perfect foresight model solution to the idiosyncratic shocks model
    solution with shocks turned off; also compares the tractable buffer stock
    model solution to the same model solved using a "Markov" description.
* Testing/ModelTesting.py:
    Early version of unit testing for HARK, still in development.  Defines a
    few wrapper classes to run unit tests on subclasses of AgentType.
* Testing/ModelTestingExample.py
    An example of ModelTesting.py in action, using TractableBufferStockModel.
* Testing/TBSunitTests.py:
    Early version of unit testing for HARK, still in development.  Runs a test
    on TractableBufferStockModel.
* Testing/MultithreadDemo.py:
    Demonstrates the multithreading functionality in HARKparallel.py.  When
    run, it solves oneexample consumption-saving model with idiosyncratic
    shocks to income, then solves *many* such models serially, varying the
    coefficient of relative risk aversion between rho=1 and rho=8, displaying
    the results graphically and presenting the timing.  It then solves the
    same set of many models using multithreading on the local CPU, displays
    the results graphically along with the timing.

Data files:
* SolvingMicroDSOPs/SCFdata.csv:
    SCF 2004 data for use in SolvingMicroDSOPs/StructEstimation.py, loaded by
    SolvingMicroDSOPs/EstimationParameters.py.
* cstwMPC/SCFwealthDataReduced.txt:
    SCF 2004 data with just net worth and data weights, for use by cstwMPC.py
* cstwMPC/USactuarial.txt:
    U.S. mortality data from the Social Security Administration, for use by
    cstwMPC.py when running a lifecycle specification.
* cstwMPC/EducMortAdj.txt:
    Mortality adjusters by education and age (columns by sex and race), for use
    by cstwMPC.py when running a lifecycle specification.  Taken from an
    appendix of PAPER.

Other files that you don't need to worry about:
* */index.py:
    A file used by Sphinx when generating html documentation for HARK.  Users
    don't need to worry about it.  Several copies are found throughout HARK.
* .gitignore:
    A file that tells git which files (or types of files) might be found in
    the repository directory tree, but should be ignored (not tracked) for
    the repo.  Currently ignores compiled Python code, LaTex auxiliary files, etc.
* LICENSE:
    License text for HARK, Apache 2.0.  Read it if you're a lawyer!
* SolvingMicroDSOPs/SMMcontour.png:
    Contour plot of the objective function for SolvingMicroDSOPs/StructEstimation.py.
    Generated when that module is run, along with a PDF version.
* cstwMPC/Figures/placeholder.txt:
    A placeholder file because git doesn't like empty folders, but cstwMPC.py
    needs the /Figures directory to exist when it runs.
* cstwMPC/Results/placeholder.txt:
    A placeholder file because git doesn't like empty folders, but cstwMPC.py
    needs the /Results directory to exist when it runs.
* Documentation/conf.py:
    A configuration file for producing html documentation with Sphinx, generated
    by sphinx-quickstart.
* Documentation/includeme.rst:
    A very small file used by Sphinx to produce documentation.
* Documentation/index.rst:
    A list of modules to be included in HARK's Sphinx documentation.  This should
    be edited if a new tool or model module is added to HARK.
* Documentation/instructions.md:
    A markdown file with instructions for how to set up and run Sphinx.  You
    don't need to read it.
* Documentation/simple-steps-getting-sphinx-working.md:
    Another markdown file with instructions for how to set up and run Sphinx.
* Documentation/make.bat:
    A batch file for producing Sphinx documentation, generated by sphinx-quickstart.
* Documentation/Makefile:
    Another Sphinx auxiliary file generated by sphinx-quickstart.
* Documentation/econtex.sty:
    LaTeX style file with notation definitions.
* Documentation/econtex.cls:
    LaTeX class file with document layout for the user manual.
* Documentation/econtexSetup.sty:
    LaTeX style file with notation definitions.
* Documentation/econtexShortcuts.sty:
    LaTeX style file with notation definitions.
* Documentation/UserGuidePic.pdf:
    Image for the front cover of the user guide, showing the consumption
    function for the KinkyPref model.


## IV. WARNINGS AND DISCLAIMERS

This is an early beta version of HARK.  The code has not been
extensively tested as it should be.  We hope it is useful, but 
there are absolutely no guarantees (expressed or implied) that 
it works or will do what you want.  Use at your own risk.  And 
please, let us know if you find bugs by posting an issue to the
GitHub page!


## V. License

All of HARK is licensed under the Apache License, Version 2.0 (ALv2). Please see 
the LICENSE file for the text of the license. More information can be found at: 
http://www.apache.org/dev/apply-license.html

