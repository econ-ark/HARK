# List of Files in Repository

This section contains descriptions of the main files in the repo.

Documentation files:
* [Documentation/HARKdoc.pdf](Documentation/HARKdoc.pdf): A mini-user guide produced for a December 2015 workshop on HARK, unofficially representing the alpha version.  (Substantially out of date).
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

