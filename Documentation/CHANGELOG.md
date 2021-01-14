# Release Notes

## Introduction

This document contains the release notes of HARK. HARK aims to produce an open source repository of highly modular, easily interoperable code for solving, simulating, and estimating dynamic economic models with heterogeneous agents.

For more information on HARK, see [our Github organization](https://github.com/econ-ark).

## Changes

### 0.10.9

Release Data: TBD

#### Major Changes

* Adds a constructor for LogNormal distributions from mean and standard deviation [#891](https://github.com/econ-ark/HARK/pull/891/)
* Uses new LogNormal constructor in ConsPortfolioModel [#891](https://github.com/econ-ark/HARK/pull/891/)
* calcExpectations method for taking the expectation of a distribution over a function [#884](https://github.com/econ-ark/HARK/pull/884/] (#897)[https://github.com/econ-ark/HARK/pull/897/)
* Centralizes the definition of value, marginal value, and marginal marginal value functions that use inverse-space
interpolation for problems with CRRA utility. See [#888](https://github.com/econ-ark/HARK/pull/888).
* MarkovProcess class [#902](https://github.com/econ-ark/HARK/pull/902)
* replace HARKobject base class with MetricObject and Model classes [#903](https://github.com/econ-ark/HARK/pull/903/)
* Add __repr__ and __eq__ methods to Model class [#903](https://github.com/econ-ark/HARK/pull/903/)
* Adds a SSA life tables and methods to extract survival probabilities from them [#986](https://github.com/econ-ark/HARK/pull/906).
* Fix the return fields of `dcegm/calcCrossPoints`[#909](https://github.com/econ-ark/HARK/pull/909).
* Corrects location of constructor documentation to class string for Sphinx rendering [#908](https://github.com/econ-ark/HARK/pull/908)

#### Minor Changes

* Move AgentType constructor parameters docs to class docstring so it is rendered by Sphinx.
* Remove uses of deprecated time.clock [#887](https://github.com/econ-ark/HARK/pull/887)

### 0.10.8

Release Data: Nov. 05 2020

#### Major Changes

* Namespace variables for the Market class [#765](https://github.com/econ-ark/HARK/pull/765)
* We now have a Numba based implementation of PerfForesightConsumerType model available as PerfForesightConsumerTypeFast [#774](https://github.com/econ-ark/HARK/pull/774)
* Namespace for exogenous shocks [#803](https://github.com/econ-ark/HARK/pull/803)
* Namespace for controls [#855](https://github.com/econ-ark/HARK/pull/855)
* State and poststate attributes replaced with state_now and state_prev namespaces [#836](https://github.com/econ-ark/HARK/pull/836)

#### Minor Changes 

* Use shock_history namespace for pre-evaluated shock history [#812](https://github.com/econ-ark/HARK/pull/812)
* Fixes seed of PrefShkDstn on initialization and add tests for simulation output
* Reformat code style using black

### 0.10.7

Release Date: 08-08-2020

#### Major Changes

- Add a custom KrusellSmith Model [#762](https://github.com/econ-ark/HARK/pull/762)
- Simulations now uses a dictionary `history` to store state history instead of `_hist` attributes [#674](https://github.com/econ-ark/HARK/pull/674)
- Removed time flipping and time flow state, "forward/backward time" through data access [#570](https://github.com/econ-ark/HARK/pull/570)
- Simulation draw methods are now individual distributions like `Uniform`, `Lognormal`, `Weibull` [#624](https://github.com/econ-ark/HARK/pull/624)

#### Minor Changes 

- unpackcFunc is deprecated, use unpack(parameter) to unpack a parameter after solving the model [#784](https://github.com/econ-ark/HARK/pull/784)
- Remove deprecated Solution Class, use HARKObject across the codebase [#772](https://github.com/econ-ark/HARK/pull/772)
- Add option to find crossing points in the envelope step of DCEGM algorithm [#758](https://github.com/econ-ark/HARK/pull/758)
- Fix reset bug in the behaviour of AgentType.resetRNG(), implemented individual resetRNG methods for AgentTypes [#757](https://github.com/econ-ark/HARK/pull/757)
- Seeds are set at initialisation of a distribution object rather than draw method [#691](https://github.com/econ-ark/HARK/pull/691) [#750](https://github.com/econ-ark/HARK/pull/750), [#729](https://github.com/econ-ark/HARK/pull/729)
- Deal with portfolio share of 'bad' assets [#749](https://github.com/econ-ark/HARK/pull/749)
- Fix bug in make_figs utilities function [#755](https://github.com/econ-ark/HARK/pull/755)
- Fix typo bug in Perfect Foresight Model solver [#743](https://github.com/econ-ark/HARK/pull/743)
- Add initial support for logging in ConsIndShockModel [#714](https://github.com/econ-ark/HARK/pull/714)
- Speed up simulation in AggShockMarkovConsumerType [#702](https://github.com/econ-ark/HARK/pull/702)
- Fix logic bug in DiscreteDistribution draw method [#715](https://github.com/econ-ark/HARK/pull/715)
- Implemented distributeParams to distributes heterogeneous values of one parameter to a set of agents [#692](https://github.com/econ-ark/HARK/pull/692)
- NelderMead is now part of estimation [#693](https://github.com/econ-ark/HARK/pull/693)
- Fix typo bug in parallel [#682](https://github.com/econ-ark/HARK/pull/682)
- Fix DiscreteDstn to make it work with multivariate distributions [#646](https://github.com/econ-ark/HARK/pull/646)
- BayerLuetticke removed from HARK, is now a REMARK [#603](https://github.com/econ-ark/HARK/pull/603)
- cstwMPC removed from HARK, is now a REMARK [#666](https://github.com/econ-ark/HARK/pull/666)
- SolvingMicroDSOPs removed from HARK, is now a REMARK [#651](https://github.com/econ-ark/HARK/pull/651)
- constructLogNormalIncomeProcess is now a method of IndShockConsumerType [#661](https://github.com/econ-ark/HARK/pull/661)
- Discretize continuous distributions [#657](https://github.com/econ-ark/HARK/pull/657)
- Data used in cstwMPC is now in HARK.datasets [#622](https://github.com/econ-ark/HARK/pull/622)
- Refactor checkConditions by adding a checkCondition method instead of writing custom checks for each condition [#568](https://github.com/econ-ark/HARK/pull/568)
- Examples update [#768](https://github.com/econ-ark/HARK/pull/768), [#759](https://github.com/econ-ark/HARK/pull/759), [#756](https://github.com/econ-ark/HARK/pull/756), [#727](https://github.com/econ-ark/HARK/pull/727), [#698](https://github.com/econ-ark/HARK/pull/698), [#697](https://github.com/econ-ark/HARK/pull/697), [#561](https://github.com/econ-ark/HARK/pull/561), [#654](https://github.com/econ-ark/HARK/pull/654), [#633](https://github.com/econ-ark/HARK/pull/633), [#775](https://github.com/econ-ark/HARK/pull/775)


### 0.10.6

Release Date: 17-04-2020

#### Major Changes

* Add Bellman equations for cyclical model example [#600](https://github.com/econ-ark/HARK/pull/600)

* read_shocks now reads mortality as well [#613](https://github.com/econ-ark/HARK/pull/613)

* Discrete probability distributions are now classes [#610](https://github.com/econ-ark/HARK/pull/610)

#### Minor Changes 



### 0.10.5

Release Date: 24-03-2020

#### Major Changes
 * Default parameters dictionaries for ConsumptionSaving models have been moved from ConsumerParameters to nearby the classes that use them. [#527](https://github.com/econ-ark/HARK/pull/527)

 * Improvements and cleanup of ConsPortfolioModel, and adding the ability to specify an age-varying list of RiskyAvg and RiskyStd. [#577](https://github.com/econ-ark/HARK/pull/527)

 * Rewrite and simplification of ConsPortfolioModel solver. [#594](https://github.com/econ-ark/HARK/pull/594)

#### Minor Changes 

### 0.10.4

Release Date: 05-03-2020

#### Major Changes
 - Last release to support Python 2.7, future releases of econ-ark will support Python 3.6+ [#478](https://github.com/econ-ark/HARK/pull/478)
 - Move non-reusable model code to examples directory, BayerLuetticke, FashionVictim now in examples instead of in HARK code [#442](https://github.com/econ-ark/HARK/pull/442)
 - Load default parameters for ConsumptionSaving models [#466](https://github.com/econ-ark/HARK/pull/466)
 - Improved implementaion of parallelNelderMead [#300](https://github.com/econ-ark/HARK/pull/300)
 
#### Minor Changes 
 - Notebook utility functions for determining platform, GUI, latex (installation) are available in HARK.utilities [#512](https://github.com/econ-ark/HARK/pull/512)
 - Few DemARKs moved to examples [#472](https://github.com/econ-ark/HARK/pull/472)
 - MaxKinks available in ConsumerParameters again [#486](https://github.com/econ-ark/HARK/pull/486)


### 0.10.3

Release Date: 12-12-2019

#### Major Changes

- Added constrained perfect foresight model solution. ([#299](https://github.com/econ-ark/HARK/pull/299)

#### Minor Changes

- Fixed slicing error in minimizeNelderMead. ([#460](https://github.com/econ-ark/HARK/pull/460))
- Fixed matplotlib GUI error. ([#444](https://github.com/econ-ark/HARK/pull/444))
- Pinned sphinx dependency. ([#436](https://github.com/econ-ark/HARK/pull/436))
- Fixed bug in ConsPortfolioModel in which the same risky rate of return would be drawn over and over. ([#433](https://github.com/econ-ark/HARK/pull/433))
- Fixed sphinx dependency errors. ([#411](https://github.com/econ-ark/HARK/pull/411))
- Refactored simultation.py. ([#408](https://github.com/econ-ark/HARK/pull/408))
- AgentType.simulate() now throws informative errors if
attributes required for simulation do not exist, or initializeSim() has
never been called. ([#320](https://github.com/econ-ark/HARK/pull/320))

### 0.10.2

Release Date: 10-03-2019

#### Minor Changes
- Add some bugfixes and unit tests to HARK.core. ([#401](https://github.com/econ-ark/HARK/pull/401))
- Fix error in discrete portfolio choice's AdjustPrb. ([#391](https://github.com/econ-ark/HARK/pull/391))

### 0.10.1.dev5

Release Date: 09-25-2019

#### Minor Changes
- Added portfolio choice between risky and safe assets (ConsPortfolioModel). ([#241](https://github.com/econ-ark/HARK/pull/241))

### 0.10.1.dev4

Release Date: 09-19-2019

#### Minor Changes
- Fixes cubic interpolation in KinkedRSolver. ([#386](https://github.com/econ-ark/HARK/pull/386))
- Documentes the procedure for constructing value function inverses and fixes bug in which survival rate was not included in absolute patience factor. ([#383](https://github.com/econ-ark/HARK/pull/383))
- Fixes problems that sometimes prevented multiprocessing from working. ([#377](https://github.com/econ-ark/HARK/pull/377))

### 0.10.1.dev3

Release Date: 07-23-2019

#### Minor Changes
- Missed pre-solve fix (see [#363](https://github.com/econ-ark/HARK/pull/363) for more context). ([#367](https://github.com/econ-ark/HARK/pull/367))

### 0.10.1.dev2

Release Date: 07-22-2019

#### Minor Changes
- Revert pre-solve commit due to bug.  ([#363](https://github.com/econ-ark/HARK/pull/363))

### 0.10.1.dev1

Release Date: 07-20-2019

#### Breaking Changes
- See #302 under minor changes.

#### Major Changes
- Adds BayerLuetticke notebooks and functionality. ([#328](https://github.com/econ-ark/HARK/pull/328))

#### Minor Changes
- Fixes one-asset HANK models for endowment economy (had MP wired in as the shock). ([#355](https://github.com/econ-ark/HARK/pull/355))
- Removes jupytext *.py files. ([#354](https://github.com/econ-ark/HARK/pull/354))
- Reorganizes documentation and configures it to work with Read the Docs. ([#353](https://github.com/econ-ark/HARK/pull/353))
- Adds notebook illustrating dimensionality reduction in Bayer and Luetticke. ([#345](https://github.com/econ-ark/HARK/pull/345))
- Adds notebook illustrating how the Bayer & Luetticke invoke the discrete cosine transformation(DCT) and fixed copula to reduce dimensions of the problem.([#344](https://github.com/econ-ark/HARK/pull/344))
- Makes BayerLuetticke HANK tools importable as a module. ([#342](https://github.com/econ-ark/HARK/pull/342))
- Restores functionality of SGU_solver. ([#341](https://github.com/econ-ark/HARK/pull/341))
- Fixes datafile packaging issue. ([#332](https://github.com/econ-ark/HARK/pull/332))
- Deletes .py file from Bayer-Luetticke folder. ([#329](https://github.com/econ-ark/HARK/pull/329))
- Add an empty method for preSolve called checkRestrictions that can be overwritten in classes inheriting from AgentType to check for illegal parameter values. ([#324](https://github.com/econ-ark/HARK/pull/324))
- Adds a call to updateIncomeProcess() in preSolve() to avoid solutions being based on wrong income process specifications if some parameters change between two solve() calls. ([#323](https://github.com/econ-ark/HARK/pull/323))
- Makes checkConditions() less verbose when the checks are not actually performed by converting a print statement to an inline comment. ([#321](https://github.com/econ-ark/HARK/pull/321))
- Raises more readable exception when simultate() is called without solving first. ([#315](https://github.com/econ-ark/HARK/pull/315)) 
- Removes testing folder (part of ongoing test restructuring). ([#304](https://github.com/econ-ark/HARK/pull/304))
- Fixes unintended behavior in default simDeath(). Previously, all agents would die off in the first period, but they were meant to always survive. ([#302](https://github.com/econ-ark/HARK/pull/302)) __Warning__: Potentially breaking change.

### 0.10.1

Release Date: 05-30-2019

No changes from 0.10.0.dev3.

### 0.10.0.dev3

Release Date: 05-18-2019

#### Major Changes
- Fixes multithreading problems by using Parallels(backend='multiprocessing'). ([287](https://github.com/econ-ark/HARK/pull/287))
- Fixes bug caused by misapplication of check_conditions. ([284](https://github.com/econ-ark/HARK/pull/284))
- Adds functions to calculate quadrature nodes and weights for numerically evaluating expectations in the presence of (log-)normally distributed random variables. ([258](https://github.com/econ-ark/HARK/pull/258))

#### Minor Changes
- Adds method decorator which validates that arguments passed in are not empty. ([282](https://github.com/econ-ark/HARK/pull/282)
- Lints a variety of files.  These PRs include some additional/related minor changes, like replacing an exec function, removing some lambdas, adding some files to .gitignore, etc. ([274](https://github.com/econ-ark/HARK/pull/274), [276](https://github.com/econ-ark/HARK/pull/276), [277](https://github.com/econ-ark/HARK/pull/277), [278](https://github.com/econ-ark/HARK/pull/278), [281](https://github.com/econ-ark/HARK/pull/281))
- Adds vim swp files to gitignore. ([269](https://github.com/econ-ark/HARK/pull/269))
- Adds version dunder in init. ([265](https://github.com/econ-ark/HARK/pull/265))
- Adds flake8 to requirements.txt and config. ([261](https://github.com/econ-ark/HARK/pull/261))
- Adds some unit tests for IndShockConsumerType. ([256](https://github.com/econ-ark/HARK/pull/256))

### 0.10.0.dev2

Release Date: 04-18-2019

#### Major Changes

None

#### Minor Changes

* Fix verbosity check in ConsIndShockModel. ([250](https://github.com/econ-ark/HARK/pull/250))

#### Other Changes

None

### 0.10.0.dev1

Release Date: 04-12-2019

#### Major Changes

* Adds [tools](https://github.com/econ-ark/HARK/blob/master/HARK/dcegm.py) to solve problems that arise from the interaction of discrete and continuous variables, using the [DCEGM](https://github.com/econ-ark/DemARK/blob/master/notebooks/DCEGM-Upper-Envelope.ipynb) method of [Iskhakov et al.](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE643), who apply the their discrete-continuous solution algorithm to the problem of optimal endogenous retirement; their results are replicated using our new tool [here](https://github.com/econ-ark/REMARK/blob/master/REMARKs/EndogenousRetirement/Endogenous-Retirement.ipynb). ([226](https://github.com/econ-ark/HARK/pull/226))
* Parameters of ConsAggShockModel.CobbDouglasEconomy.updateAFunc and ConsAggShockModel.CobbDouglasMarkovEconomy.updateAFunc that govern damping and the number of discarded 'burn-in' periods were previously hardcoded, now proper instance-level parameters. ([244](https://github.com/econ-ark/HARK/pull/244))
* Improve accuracy and performance of functions for evaluating the integrated value function and conditional choice probabilities for models with extreme value type I taste shocks. ([242](https://github.com/econ-ark/HARK/pull/242))
* Add calcLogSum, calcChoiceProbs, calcLogSumChoiceProbs to HARK.interpolation. ([209](https://github.com/econ-ark/HARK/pull/209), [217](https://github.com/econ-ark/HARK/pull/217))
* Create tool to produce an example "template" of a REMARK based on SolvingMicroDSOPs. ([176](https://github.com/econ-ark/HARK/pull/176))

#### Minor Changes

* Moved old utilities tests. ([245](https://github.com/econ-ark/HARK/pull/245))
* Deleted old files related to "cstwMPCold". ([239](https://github.com/econ-ark/HARK/pull/239))
* Set numpy floating point error level to ignore. ([238](https://github.com/econ-ark/HARK/pull/238))
* Fixed miscellaneous imports. ([212](https://github.com/econ-ark/HARK/pull/212), [224](https://github.com/econ-ark/HARK/pull/224), [225](https://github.com/econ-ark/HARK/pull/225))
* Improve the tests of buffer stock model impatience conditions in IndShockConsumerType. ([219](https://github.com/econ-ark/HARK/pull/219))
* Add basic support for Travis continuous integration testing. ([208](https://github.com/econ-ark/HARK/pull/208))
* Add SciPy to requirements.txt. ([207](https://github.com/econ-ark/HARK/pull/207))
* Fix indexing bug in bilinear interpolation. ([194](https://github.com/econ-ark/HARK/pull/194))
* Update the build process to handle Python 2 and 3 compatibility. ([172](https://github.com/econ-ark/HARK/pull/172))
* Add MPCnow attribute to ConsGenIncProcessModel. ([170](https://github.com/econ-ark/HARK/pull/170))
* All standalone demo files have been removed. The content that was in these files can now be found in similarly named Jupyter notebooks in the DEMARK repository. Some of these notebooks are also linked from econ-ark.org.  ([229](https://github.com/econ-ark/HARK/pull/229), [243](https://github.com/econ-ark/HARK/pull/243))

#### Other Notes

* Not all changes from 0.9.1 may be listed in these release notes.  If you are having trouble addressing a breaking change, please reach out to us.

