HARK  
Version 0.10.0.dev3
Release Notes  

# Introduction

This document contains the release notes for the 0.10.0.dev3 version of HARK. HARK aims to produce an open source repository of highly modular, easily interoperable code for solving, simulating, and estimating dynamic economic models with heterogeneous agents.

For more information on HARK, see [our Github organization](https://github.com/econ-ark).

## Changes

### 0.10.0.dev3

Release Date: 05-18-2019

#### Major Changes
- Fixes multithreading problems by using Parallels(backend='multiprocessing'). ([287](https://github.com/econ-ark/HARK/pull/287))
- Fixes bug caused by misapplication of check_conditions. ([284](https://github.com/econ-ark/HARK/pull/284))
- Adds functions to calculate quadrature nodes and weights for numerically evaluating expectations in the presence of (log-)normally distributed random variables. ([258](https://github.com/econ-ark/HARK/pull/258))

#### Minor Changes
- Adds method decorator which validates that arguments passed in are not empty. ([282](https://github.com/econ-ark/HARK/pull/282)
- Lints a variety of files.  These PRs include some additional/related minor changes, like replacing an `exec` function, removing some lambdas, adding some files to .gitignore, etc. ([274](https://github.com/econ-ark/HARK/pull/274), [276](https://github.com/econ-ark/HARK/pull/276), [277](https://github.com/econ-ark/HARK/pull/277), [278](https://github.com/econ-ark/HARK/pull/278), [281](https://github.com/econ-ark/HARK/pull/281))
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

