# Heterogeneous Agents Resources and toolKit (HARK)
pre-release 0.10.1.dev1

Click the Badge for Citation Info.
[![DOI](https://zenodo.org/badge/50448254.svg)](https://zenodo.org/badge/latestdoi/50448254)


Table of Contents:

* [1.   Introduction](#i-introduction)
* [2.  Quick start guide](#ii-quick-start-guide)
    * [Installing](#Installing-HARK)
    * [Installing Anaconda](#Using-HARK-with-Anaconda)
    * [Learning HARK](#Learning-HARK)
* [3. List of files in repository](#iii-list-of-files-in-repository)
* [4.  Warnings and disclaimers](#iv-warnings-and-disclaimers)
* [5.   License Information](#v-license)


## I. INTRODUCTION

This document will tell you how to get HARK up and
running on your machine, how to get started using it, and give you an
overview of the main elements of the toolkit.

Other useful resources are:
   * Documentation: [ReadTheDocs](https://hark.readthedocs.io/) 
   * User guide: [Documentation/HARKmanual.pdf](Documentation/HARKmanual.pdf) 
      * In the [HARK repository](https://github.com/econ-ark/HARK)
   * Demonstrations of HARK functionality: [DemARK](https://github.com/econ-ark/DemARK/)
   * Replications and Explorations Made using the ARK : [REMARK](https://github.com/econ-ark/REMARK/)

## II. QUICK START GUIDE

### Installing HARK

HARK is an open source project that is compatible with both python 2 and 3. But we recommend
using python 3; eventually support for python 2 will end.

#### Installing HARK with pip

The simplest way to install HARK is to use [pip](https://pip.pypa.io/en/stable/installing/).

To install HARK with pip, at a command line type `pip install econ-ark`.

(If you want to install a release that is not the default stable release, for instance if you want to install a development release, you'll need to use a "pinned" release number: `pip install econ-ark==0.10.1.dev1`, substituting "0.10.1.dev1" for your desired release number.)

If you are installing via pip, we recommend using a virtual environment such as [virtualenv](https://virtualenv.pypa.io/en/latest/). Creation of a virtual environment isolates the installation of `econ-ark` from the installations of any other python tools and packages.

To install `virtualenv`, then to create an environment named `econ-ark`, and finally to activate that environment:

```
cd [directory where you want to store the econ-ark virtual environment]
pip install virtualenv
virtualenv econ-ark
source econ-ark/bin/activate
```

----
#### Using HARK with Anaconda

If you intend ever to use the toolkit for anything other than running the precooked material we have provided, you should probably install [Anaconda](https://anaconda.com/why-anaconda), which will install python along with many packages that are frequently used in scientific computing.

1. Download Anaconda for your operating system and follow the installation instructions [at Anaconda.com](https://www.anaconda.com/distribution/#download-section).

1. Anaconda includes its own virtual environment system called `conda` which stores environments in a preset location (so you don't have to choose). So in order to create and activate an econ-ark virtual environment:
```
conda create -n econ-ark anaconda
conda activate econ-ark
conda install -c conda-forge econ-ark
```
1. Open Spyder, an interactive development environment (IDE) for Python (specifically, iPython).  You may be able to do this through Anaconda's graphical interface, or you can do so from the command line/prompt.  To do so, simply open a command line/prompt and type `spyder`.

1. To verify that spyder has access to HARK try typing `pip install econ-ark` into the iPython shell within Spyder.  If you have successfully installed HARK as above, you should see a lot of messages saying 'Requirement satisfied'.

    * If that doesn't work, you will need to manually add HARK to your Spyder environment.  To do this, you'll need to get the code from Github and import it into Spyder.  To get the code from Github, you can either clone it or download a zipped file.

    * If you have `git` installed on the command line, type `git clone git@github.com:econ-ark/HARK.git` in your chosen directory ([more details here](https://git-scm.com/documentation)).

		* If you do not have `git` available on your computer, you can download the [GitHub Desktop app](https://desktop.github.com/) and use it to make a local clone

    * If you don't want to clone HARK, but just to download it, go to [the HARK repository on GitHub](https://github.com/econ-ark/HARK).  In the upper righthand corner is a button that says "clone or download".  Click the "Download Zip" option and then unzip the contents into your chosen directory.

    Once you've got a copy of HARK in a directory, return to Spyder and navigate to that directory where you put HARK.  This can be done within Spyder by doing `import os` and then using `os.chdir()` to change directories.  `chdir` works just like cd at a command prompt on most operating systems, except that it takes a string as input: `os.chdir('Music')` moves to the Music subdirectory of the current working directory.

6) Most of the modules in HARK are just collections of tools.  There are a few demonstration
applications that use the tools that you automatically get when you install HARK -- they are listed below in [Application Modules](#application-modules).  A much larger set of uses of HARK can be found at two repositories:
	* [DemARK](https://github.com/econ-ark/DemARK): Demonstrations of the use of HARK
	* [REMARK](https://github.com/econ-ark/REMARK): Replications of existing papers made using HARK

You will want to obtain your own local copy of these repos using:
```
git clone https://github.com/econ-ark/DemARK.git
```
and similarly for the REMARK repo. Once you have downloaded them, you will find that each repo contains a `notebooks` directory that contains a number of [jupyter notebooks](https://jupyter.org/). If you have the jupyter notebook tool installed (it is installed as part of Anaconda), you should be able to launch the
jupyter notebook app from the command line with the command:

```
jupyter notebook
```
and from there you can open the notebooks and execute them.

#### Learning HARK

We have a set of 30-second [Elevator Spiels](https://github.com/econ-ark/PARK/blob/master/Elevator-Spiels.md#capsule-summaries-of-what-the-econ-ark-project-is) describing the project, tailored to people with several different kinds of background.  

The most broadly applicable advice is to go to [Econ-ARK](https://econ-ark.org) and click on "Notebooks", and choose [A Gentle Introduction to HARK](https://github.com/econ-ark/DemARK/blob/master/notebooks/Gentle-Intro-To-HARK.ipynb) which will launch as a [jupyter notebook](https://jupyter.org/).  

##### [For people with a technical/scientific/computing background but little economics background](https://github.com/econ-ark/PARK/blob/master/Elevator-Spiels.md#for-people-with-a-technicalscientificcomputing-background-but-no-economics-background)

* [A Gentle Introduction to HARK](https://github.com/econ-ark/DemARK/blob/master/notebooks/Gentle-Intro-To-HARK.ipynb)

##### [For economists who have done some structural modeling](https://github.com/econ-ark/PARK/blob/master/Elevator-Spiels.md#for-economists-who-have-done-some-structural-modeling)

* A full replication of the [Iskhakov, Jørgensen, Rust, and Schjerning](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE643) paper for solving the discrete-continuous retirement saving problem
   * An informal discussion of the issues involved is [here](https://github.com/econ-ark/DemARK/blob/master/notebooks/DCEGM-Upper-Envelope.ipynb) (part of the [DemARK](https://github.com/econ-ark/DemARK) repo)

* [Structural-Estimates-From-Empirical-MPCs](https://github.com/econ-ark/DemARK/blob/master/notebooks/Structural-Estimates-From-Empirical-MPCs-Fagereng-et-al.ipynb) is an example of the use of the toolkit in a discussion of a well known paper.  (Yes, it is easy enough to use that you can estimate a structural model on somebody else's data in the limited time available for writing a discussion)

##### [For economists who have not yet done any structural modeling but might be persuadable to start](https://github.com/econ-ark/PARK/blob/master/Elevator-Spiels.md#for-economists-who-have-not-yet-done-any-structural-modeling-but-might-be-persuadable-to-start)

* Start with [A Gentle Introduction to HARK](https://github.com/econ-ark/DemARK/blob/master/notebooks/Gentle-Intro-To-HARK.ipynb) to get your feet wet

* A simple indirect inference/simulated method of moments structural estimation along the lines of Gourinchas and Parker's 2002 Econometrica paper or Cagetti's 2003 paper is performed by the [SolvingMicroDSOPs](https://github.com/econ-ark/REMARK/tree/master/REMARKs/SolvingMicroDSOPs) [REMARK](https://github.com/econ-ark/REMARK); this code implements the solution methods described in the corresponding section of [these lecture notes](http://www.econ2.jhu.edu/people/ccarroll/SolvingMicroDSOPs/)

##### [For Other Developers of Software for Computational Economics](https://github.com/econ-ark/PARK/blob/master/Elevator-Spiels.md#for-other-developers-of-software-for-computational-economics)


* Our workhorse module is [ConsIndShockModel.py](https://github.com/econ-ark/HARK/blob/master/HARK/ConsumptionSaving/ConsIndShockModel.py)
   * which is explored and explained (a bit) in [this jupyter notebook](https://github.com/econ-ark/DemARK/blob/master/notebooks/ConsIndShockModel.ipynb)

### Making changes to HARK

If you want to make changes or contributions (yay!) to HARK, you'll need to have access to the source files.  Installing HARK via pip (either at the command line, or inside Spyder) makes it hard to access those files (and it's a bad idea to mess with the original code anyway because you'll likely forget what changes you made).  If you are adept at GitHub, you can [fork](https://help.github.com/en/articles/fork-a-repo) the repo.  If you are less experienced, you should download a personal copy of HARK again using `git clone` (see above) or the GitHub Desktop app.

1.  Navigate to wherever you want to put the repository and type `git clone git@github.com:econ-ark/HARK.git` ([more details here](https://git-scm.com/documentation)). If you get a permission denied error, you may need to setup SSH for GitHub, or you can clone using HTTPS: 'git clone https://github.com/econ-ark/HARK.git'.

2.  Then, create and activate a [virtual environment]([virtualenv]((https://virtualenv.pypa.io/en/latest/))).

For Mac or Linux:

Install virtualenv if you need to and then type:

```
virtualenv econ-ark
source econ-ark/bin/activate
```
For Windows:
```
virtualenv econ-ark
econ-ark\\Scripts\\activate.bat
```

3. Once the virtualenv is activated, you may see `(econ-ark)` in your command prompt (depending on how your machine is configured)

3.  Make sure to change to HARK directory, and install HARK's requirements into the virtual environment with `pip install -r requirements.txt`.

4.  To check that everything has been set up correctly, run HARK's tests with `python -m unittest`.

### Trouble with installation?

We've done our best to give correct, thorough instructions on how to install HARK but we know this information may be inaccurate or incomplete.  Please let us know if you run into trouble so we can update this guide!  Here's a list of platforms and versions this guide has been verified for:

| Installation Type | Platform      | Python Version |  Date Tested  |  Tested By |
| ------------- |:-------------:| -----:| -----:|-----:|
| basic pip install | Linux (16.04) | 3 | 2019-04-24 | @shaunagm |
| anaconda | Linux (16.04) | 3 | 2019-04-24 | @shaunagm |
| basic pip install | MacOS 10.13.2 "High Sierra" | 2.7| 2019-04-26 | @llorracc |

### Next steps

To learn more about how to use HARK, check out our [user manual](Documentation/HARKmanual.pdf).

For help making changes to HARK, check out our [contributing guide](CONTRIBUTING.md).


## III. LIST OF FILES IN REPOSITORY

This section contains descriptions of the main files in the repo.

Documentation files:
* [README.md](README.md): The file you are currently reading.
* [Documentation/HARKdoc.pdf](Documentation/HARKdoc.pdf): A mini-user guide produced for a December 2015 workshop on HARK, unofficially representing the alpha version.  (Substantially out of date).
* [Documentation/HARKmanual.pdf](Documentation/HARKmanual.pdf): A user guide for HARK, written for the beta release at CEF 2016 in Bordeaux.  Should contain 90% fewer lies relative to HARKdoc.pdf.
    * [Documentation/HARKmanual.tex](Documentation/HARKmanual.tex): LaTeX source for the user guide.  Open source code probably requires an open source manual as well.
* [Documentation/ConsumptionSavingModels.pdf](Documentation/ConsumptionSavingModels.pdf): Mathematical descriptions of the various consumption-saving models in HARK and how they map into the code.
    * [Documentation/ConsumptionSavingModels.tex](Documentation/ConsumptionSavingModels.tex): LaTeX source for the "models" writeup.
* [Documentation/NARK.pdf](Documentation/NARK.pdf): Variable naming conventions for HARK, plus concordance with LaTeX variable definitions.  Still in development.

Tool modules:
* [HARK/core.py](HARK/core.py):
    Frameworks for "microeconomic" and "macroeconomic" models in HARK.
    We somewhat abuse those terms as shorthand; see the user guide for a
    description of what we mean.  Every model in HARK extends the classes
    AgentType and Market in this module.  Does nothing when run.
* [HARK/utilities.py](HARK/utilities.py):
    General purpose tools and utilities.  Contains literal utility functions
    (in the economic sense), functions for making discrete approximations
    to continuous distributions, basic plotting functions for convenience,
    and a few unclassifiable things.  Does nothing when run.
* [HARK/estimation.py](HARK/estimation.py):
    Functions for estimating models.  As is, it only has a few wrapper
    functions for scipy.optimize optimization routines.  Will be expanded
    in the future with more interesting things.  Does nothing when run.
* [HARK/simulation.py](HARK/simulation.py):
    Functions for generating simulated data.  Functions in this module have
    names like drawUniform, generating (lists of) arrays of draws from
    various distributions.  Does nothing when run.
* [HARK/interpolation.py](HARK/interpolation.py):
    Classes for representing interpolated function approximations.  Has
    1D-4D interpolation methods, mostly based on linear or cubic spline
    interpolation.  Will have ND methods in the future.  Does nothing when run.
* [HARK/parallel.py](HARK/parallel.py):
    Early version of parallel processing in HARK.  Works with instances of
    the AgentType class (or subclasses of it), distributing commands (as
    methods) to be run on a list of AgentTypes.  Only works with local CPU.
    The module also contains a parallel implentation of the Nelder-Mead
    simplex algorithm, poached from Wiswall and Lee (2011).  Does nothing
    when run.

Model modules:
* [ConsumptionSaving/TractableBufferStockModel.py](HARK/ConsumptionSaving/TractableBufferStockModel.py):
    * A "tractable" model of consumption and saving in which agents face one
    simple risk with constant probability: that they will become permanently
    unemployed and receive no further income.  Unlike other models in HARK,
    this one is not solved by iterating on a sequence of one period problems.
    Instead, it uses a "backshooting" routine that has been shoehorned into
    the AgentType.solve framework.  Solves an example of the model when run,
    then solves the same model again using MarkovConsumerType.
* [ConsumptionSaving/ConsIndShockModel.py](HARK/ConsumptionSaving/ConsIndShockModel.py):
    * Consumption-saving models with idiosyncratic shocks to income.  Shocks
    are fully transitory or fully permanent.  Solves perfect foresight model,
    a model with idiosyncratic income shocks, and a model with idiosyncratic
    income shocks and a different interest rate on borrowing vs saving.  When
    run, solves several examples of these models, including a standard infinite
    horizon problem, a ten period lifecycle model, a four period "cyclical"
    model, and versions with perfect foresight and "kinked R".
* [ConsumptionSaving/ConsPrefShockModel.py](HARK/ConsumptionSaving/ConsPrefShockModel.py):
    * Consumption-saving models with idiosyncratic shocks to income and multi-
    plicative shocks to utility.  Currently has two models: one that extends
    the idiosyncratic shocks model, and another that extends the "kinked R"
    model.  The second model has very little new code, and is created merely
    by merging the two "parent models" via multiple inheritance.  When run,
    solves examples of the preference shock models.
* [ConsumptionSaving/ConsMarkovModel.py](HARK/ConsumptionSaving/ConsMarkovModel.py):
    * Consumption-saving models with a discrete state that evolves according to
    a Markov rule.  Discrete states can vary by their income distribution,
    interest factor, and/or expected permanent income growth rate.  When run,
    solves four example models: (1) A serially correlated unemployment model
    with boom and bust cycles (4 states). (2) An "unemployment immunity" model
    in which the consumer occasionally learns that he is immune to unemployment
    shocks for the next N periods.  (3) A model with a time-varying permanent
    income growth rate that is serially correlated.  (4) A model with a time-
    varying interest factor that is serially correlated.
* [ConsumptionSaving/ConsAggShockModel.py](HARK/ConsumptionSaving/ConsAggShockModel.py):
    * Consumption-saving models with idiosyncratic and aggregate income shocks.
    Currently has a micro model with a basic solver (linear spline consumption
    function only, no value function), and a Cobb-Douglas economy for the
    agents to "live" in (as a "macroeconomy").  When run, solves an example of
    the micro model in partial equilibrium, then solves the general equilibrium
    problem to find an evolution rule for the capital-to-labor ratio that is
    justified by consumers' collective actions.
* [FashionVictim/FashionVictimModel.py](HARK/FashionVictim/FashionVictimModel.py):
    * A very serious model about choosing to dress as a jock or a punk.  Used to
    demonstrate micro and macro framework concepts from HARKcore.  It might be
    the simplest model possible for this purpose, or close to it.  When run,
    the module solves the microeconomic problem of a "fashion victim" for an
    example parameter set, then solves the general equilibrium model for an
    entire "fashion market" constituting many types of agents, finding a rule
    for the evolution of the style distribution in the population that is justi-
    fied by fashion victims' collective actions.

Application modules: <a name="application-modules"></a>

* [SolvingMicroDSOPs/Code/StructEstimation.py](HARK/SolvingMicroDSOPs/Code/StructEstimation.py):
    * Conducts a very simple structural estimation using the idiosyncratic shocks
    model in ConsIndShocksModel.  Estimates an adjustment factor to an age-varying
    sequence of discount factors (taken from Cagetti (2003)) and a coefficient
    of relative risk aversion that makes simulated agents' wealth profiles best
    match data from the 2004 Survey of Consumer Finance.  Also demonstrates
    the calculation of standard errors by bootstrap and can construct a contour
    map of the objective function.  Based on section 9 of Chris Carroll's
    lecture notes "Solving Microeconomic Dynamic Stochastic Optimization Problems".
* [cstwMPC/cstwMPC.py](HARK/cstwMPC/cstwMPC.py):
    * Conducts the estimations for the paper "The Distribution of Wealth and the
    Marginal Propensity to Consume" by Carroll, Slacalek, Tokuoka, and White (2016).
    Runtime options are set in SetupParamsCSTW.py, specifying choices such as:
    perpetual youth vs lifecycle, beta-dist vs beta-point, liquid assets vs net
    worth, aggregate vs idiosyncratic shocks, etc.  Uses ConsIndShockModel and
    ConsAggShockModel; can demonststrate HARK's "macro" framework on a real model.
* [cstwMPC/MakeCSTWfigs.py](HARK/cstwMPC/MakeCSTWfigs.py):
    * Makes various figures for the text of the [cstwMPC](http://econ.jhu.edu/people/ccarroll/papers/cstwMPC) paper.  Requires many output
    files produced by cstwMPC.py, from various specifications, which are not
    distributed with HARK.  Has not been tested in quite some time.
* [cstwMPC/MakeCSTWfigsForSlides.py](HARK/cstwMPC/MakeCSTWfigsForSlides.py):
    * Makes various figures for the slides for the cstwMPC paper.  Requires many
    output files produced by cstwMPC.py, from various specifications, which are not
    distributed with HARK.  Has not been tested in quite some time.

Parameter and data modules:
* [ConsumptionSaving/ConsumerParameters.py](HARK/ConsumptionSaving/ConsumerParameters.py):
    * Defines dictionaries with the minimal set of parameters needed to solve the
    models in ConsIndShockModel, ConsAggShockModel, ConsPrefShockModel, and
    ConsMarkovModel.  These dictionaries are used to make examples when those
    modules are run.  Does nothing when run itself.
* [SolvingMicroDSOPs/SetupSCFdata.py](HARK/SolvingMicroDSOPs/Calibration/SetupSCFdata.py):
    * Imports 2004 SCF data for use by SolvingMicroDSOPs/StructEstimation.py.
* [cstwMPC/SetupParamsCSTW.py](HARK/cstwMPC/SetupParamsCSTW.py):
    * Loads calibrated model parameters for cstwMPC.py, chooses specification.
* [FashionVictim/FashionVictimParams.py](HARK/FashionVictim/FashionVictimParams.py):
    * Example parameters for FashionVictimModel.py, loaded when that module is run.

Test modules:
* [Testing/Comparison_UnitTests.py](Testing/Comparison_UnitTests.py):
    * Early version of unit testing for HARK, still in development.  Compares
    the perfect foresight model solution to the idiosyncratic shocks model
    solution with shocks turned off; also compares the tractable buffer stock
    model solution to the same model solved using a "Markov" description.
* [Testing/ModelTesting.py](Testing/ModelTesting.py):
    * Early version of unit testing for HARK, still in development.  Defines a
    few wrapper classes to run unit tests on subclasses of AgentType.
* [Testing/TractableBufferStockModel_UnitTests.py](Testing/TractableBufferStockModel_UnitTests.py)
    * Early version of unit testing for HARK, still in development.  Runs a test
    on TractableBufferStockModel.
* [Testing/MultithreadDemo.py](Testing/MultithreadDemo.py):
    * Demonstrates the multithreading functionality in HARKparallel.py.  When
    run, it solves oneexample consumption-saving model with idiosyncratic
    shocks to income, then solves *many* such models serially, varying the
    coefficient of relative risk aversion between rho=1 and rho=8, displaying
    the results graphically and presenting the timing.  It then solves the
    same set of many models using multithreading on the local CPU, displays
    the results graphically along with the timing.

Data files:
* [SolvingMicroDSOPs/Calibration/SCFdata.csv](HARK/SolvingMicroDSOPs/Calibration/SCFdata.csv):
    * SCF 2004 data for use in SolvingMicroDSOPs/StructEstimation.py, loaded by
    SolvingMicroDSOPs/EstimationParameters.py.
* [cstwMPC/SCFwealthDataReduced.txt](HARK/cstwMPC/SCFwealthDataReduced.txt):
    * SCF 2004 data with just net worth and data weights, for use by cstwMPC.py
* [cstwMPC/USactuarial.txt](HARK/cstwMPC/USactuarial.txt):
    * U.S. mortality data from the Social Security Administration, for use by
    cstwMPC.py when running a lifecycle specification.
* [cstwMPC/EducMortAdj.txt](HARK/cstwMPC/EducMortAdj.txt):
    * Mortality adjusters by education and age (columns by sex and race), for use
    by cstwMPC.py when running a lifecycle specification.  Taken from [Brown et al. (2002)](https://www.nber.org/chapters/c9757).

Other files that you don't need to worry about:
* /index.py:
    * A file used by Sphinx when generating html documentation for HARK.  Users
    don't need to worry about it.  Several copies are found throughout HARK.
* [.gitignore](.gitignore):
    * A file that tells git which files (or types of files) might be found in
    the repository directory tree, but should be ignored (not tracked) for
    the repo.  Currently ignores compiled Python code, LaTex auxiliary files, etc.
* [LICENSE](LICENSE):
    * License text for HARK, Apache 2.0.  Read it if you're a lawyer!
* [SolvingMicroDSOPs/Figures/SMMcontour.png](HARK/SolvingMicroDSOPs/Figures/SMMcontour.png):
    * Contour plot of the objective function for SolvingMicroDSOPs/StructEstimation.py.
    Generated when that module is run, along with a PDF version.
* [cstwMPC/Figures/placeholder.txt](HARK/cstwMPC/Figures/placeholder.txt):
    * A placeholder file because git doesn't like empty folders, but cstwMPC.py
    needs the /Figures directory to exist when it runs.
* [Documentation/conf.py](Documentation/conf.py):
    * A configuration file for producing html documentation with Sphinx, generated
    by sphinx-quickstart.
* [Documentation/includeme.rst](Documentation/includeme.rst):
    * A very small file used by Sphinx to produce documentation.
* [Documentation/index.rst](Documentation/index.rst):
    * A list of modules to be included in HARK's Sphinx documentation.  This should
    be edited if a new tool or model module is added to HARK.
* [Documentation/instructions.md](Documentation/instructions.md):
    * A markdown file with instructions for how to set up and run Sphinx.  You
    don't need to read it.
* [Documentation/simple-steps-getting-sphinx-working.md](Documentation/simple-steps-getting-sphinx-working.md):
    * Another markdown file with instructions for how to set up and run Sphinx.
* [Documentation/make.bat](Documentation/make.bat):
    * A batch file for producing Sphinx documentation, generated by sphinx-quickstart.
* [Documentation/Makefile](Documentation/Makefile):
    * Another Sphinx auxiliary file generated by sphinx-quickstart.
* [Documentation/econtex.sty](Documentation/econtex.sty):
    * LaTeX style file with notation definitions.
* [Documentation/econtex.cls](Documentation/econtex.cls):
    * LaTeX class file with document layout for the user manual.
* [Documentation/econtexSetup.sty](Documentation/econtexSetup.sty):
    * LaTeX style file with notation definitions.
* [Documentation/econtexShortcuts.sty](Documentation/econtexShortcuts.sty):
    * LaTeX style file with notation definitions.
* [Documentation/UserGuidePic.pdf](Documentation/UserGuidePic.pdf):
    * Image for the front cover of the user guide, showing the consumption
    function for the KinkyPref model.


## IV. WARNINGS AND DISCLAIMERS

This is a beta version of HARK.  The code has not been extensively tested as it should be.  We hope it is useful, but there are absolutely no guarantees (expressed or implied) that it works or will do what you want.  Use at your own risk.  And please, let us know if you find bugs by posting an issue to [the GitHub page](https://github.com/econ-ark/HARK)!


## V. License

All of HARK is licensed under the Apache License, Version 2.0 (ALv2). Please see
the LICENSE file for the text of the license. More information can be found at:
http://www.apache.org/dev/apply-license.html
