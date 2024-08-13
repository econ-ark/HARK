# Release Notes

## Introduction

This document contains the release notes of HARK. HARK aims to produce an open source repository of highly modular, easily interoperable code for solving, simulating, and estimating dynamic economic models with heterogeneous agents.

For more information on HARK, see [our Github organization](https://github.com/econ-ark).

## Changes

### 0.16.0 (in development)

Release Date: TBD

### Major Changes

- Adds a discretize method to DBlocks and RBlocks (#1460)[https://github.com/econ-ark/HARK/pull/1460]
- Allows structural equations in model files to be provided in string form [#1427](https://github.com/econ-ark/HARK/pull/1427)
- Introduces `HARK.parser' module for parsing configuration files into models [#1427](https://github.com/econ-ark/HARK/pull/1427)
- Allows construction of shocks with arguments based on mathematical expressions [#1464](https://github.com/econ-ark/HARK/pull/1464)
- YAML configuration file for the normalized consumption and portolio choice [#1465](https://github.com/econ-ark/HARK/pull/1465)

#### Minor Changes

- Fixes bug in `AgentPopulation` that caused discretization of distributions to not work. [1275](https://github.com/econ-ark/HARK/pull/1275)
- Adds support for distributions, booleans, and callables as parameters in the `Parameters` class. [1387](https://github.com/econ-ark/HARK/pull/1387)
- Removes a specific way of accounting for ``employment'' in the idiosyncratic-shocks income process. [1473](https://github.com/econ-ark/HARK/pull/1473)

### 0.15.1

Release Date: June 15, 2024

This minor release was produced prior to CEF 2024 to enable public usage of HARK with the SSJ toolkit.

#### Major Changes

none

#### Minor Changes

- Adds example of integration of HARK with SSJ toolkit. [#1447](https://github.com/econ-ark/HARK/pull/1447)
- Maintains compatibility between EconForge interpolation and numba [#1457](https://github.com/econ-ark/HARK/pull/1457)

### 0.15.0

Release Date: June 4, 2024

Note: Due to major changes on this release, you may need to adjust how AgentTypes are instantiated in your projects using HARK. If you are manually constructing "complicated" objects like MrkvArray, they should be assigned to your instances *after* initialization, not passed as part of the parameter dictionary. See also the new constructor methodology for how to pass parameters for such constructed inputs.

This release drops support for Python 3.8 and 3.9, consistent with SPEC 0, and adds support for Python 3.11 and 3.12. We expect that all HARK features still work with the older versions, but they are no longer part of our testing regimen.

#### Major Changes

- Drop official support for Python 3.8 and 3.9, add support for 3.11 and 3.12. [#1415](https://github.com/econ-ark/HARK/pull/1415)
- Replace object-oriented solvers with single function versions. [#1394](https://github.com/econ-ark/HARK/pull/1394)
- Object-oriented solver code has been moved to /HARK/ConsumptionSaving/LegacyOOsolvers.py, for legacy support of downstream projects.
- AgentTypeMonteCarloSimulator now requires model shock, parameter, and dynamics information to be organized into 'blocks'. The DBlock object is introduced. [#1411](https://github.com/econ-ark/HARK/pull/1411)
- RBlock object allows for recursive composition of DBlocks in models, as demonstrated by the AgentTypeMonteCarloSimulator [#1417](https://github.com/econ-ark/HARK/pull/1417/)
- Transtion, reward, state-rule value function, decision value function, and arrival value function added to DBlock [#1417](https://github.com/econ-ark/HARK/pull/1417/)
- All methods that construct inputs for solvers are now functions that are specified in the dictionary attribute `constructors`. [#1410](https://github.com/econ-ark/HARK/pull/1410)
- Such constructed inputs can use alternate parameterizations / formats by changing the `constructor` function and providing its arguments in `parameters`.
- Move `HARK.datasets` to `HARK.Calibration` for better organization of data and calibration tools. [#1430](https://github.com/econ-ark/HARK/pull/1430)

#### Minor Changes


- Add option to pass pre-built grid to `LinearFast`. [1388](https://github.com/econ-ark/HARK/pull/1388)
- Moves calculation of stable points out of ConsIndShock solver, into method called by post_solve [#1349](https://github.com/econ-ark/HARK/pull/1349)
- Adds cubic spline interpolation and value function construction to "warm glow bequest" models.
- Fixes cubic spline interpolation for ConsMedShockModel.
- Moves computation of "stable points" from inside of ConsIndShock solver to a post-solution method. [#1349](https://github.com/econ-ark/HARK/pull/1349)
- Corrects calculation of "human wealth" under risky returns, providing correct limiting linear consumption function. [#1403](https://github.com/econ-ark/HARK/pull/1403)
- Removed 'parameters' from new block definitions; these are now 'calibrations' provided separately.
- Create functions for well-known and repeated calculations in single-function solvers. [1395](https://github.com/econ-ark/HARK/pull/1395)
- Re-work WealthPortfolioSolver to use approximate EGM method [#1404](https://github.com/econ-ark/HARK/pull/1404)
- Default parameter dictionaries for AgentType subclasses have been "flattened": all parameters appear in one place for each model, rather than inheriting from parent models' dictionaries. The only exception is submodels *within* a file when only 1 or 2 parameters are added or changed. [#1425](https://github.com/econ-ark/HARK/pull/1425)
- Fix minor bug in `HARK.distribution.Bernoulli` to allow conversion into `DiscreteDistributionLabeled`. [#1432](https://github.com/econ-ark/HARK/pull/1432)


### 0.14.1

Release date: February 28, 2024

#### Major Changes

none

#### Minor Changes

- Fixes a bug in make_figs arising from the metadata argument being incompatible with jpg. [#1386](https://github.com/econ-ark/HARK/pull/1386)
- Reverts behavior of the repr method of the Model class, so that long strings aren't generated. Full description is available with describe(). [#1390](https://github.com/econ-ark/HARK/pull/1390)

### 0.14.0

Release Date: February 12, 2024

#### Major Changes

- Adds `HARK.core.AgentPopulation` class to represent a population of agents with ex-ante heterogeneous parametrizations as distributions. [#1237](https://github.com/econ-ark/HARK/pull/1237)
- Adds `HARK.core.Parameters` class to represent a collection of time varying and time invariant parameters in a model. [#1240](https://github.com/econ-ark/HARK/pull/1240)
- Adds `HARK.simulation.monte_carlo` module for generic Monte Carlo simulation functions using Python model configurations. [1296](https://github.com/econ-ark/HARK/pull/1296)

#### Minor Changes

- Adds option `sim_common_Rrisky` to control whether risky-asset models draw common or idiosyncratic returns in simulation. [#1250](https://github.com/econ-ark/HARK/pull/1250),[#1253](https://github.com/econ-ark/HARK/pull/1253)
- Addresses [#1255](https://github.com/econ-ark/HARK/issues/1255). Makes age-varying stochastic returns possible and draws from their discretized version. [#1262](https://github.com/econ-ark/HARK/pull/1262)
- Fixes bug in the metric that compares dictionaries with the same keys. [#1260](https://github.com/econ-ark/HARK/pull/1260)

### 0.13.0

Release Date: February 16, 2023

#### Major Changes

- Updates the DCEGM tools to address the flaws identified in [issue #1062](https://github.com/econ-ark/HARK/issues/1062). PR: [1100](https://github.com/econ-ark/HARK/pull/1100).
- Updates `IndexDstn`, introducing the option to use an existing RNG instead of creating a new one, and creating and storing all the conditional distributions at initialization. [1104](https://github.com/econ-ark/HARK/pull/1104)
- `make_shock_history` and `read_shocks == True` now store and use the random draws that determine newborn's initial states [#1101](https://github.com/econ-ark/HARK/pull/1101).
- `FrameModel` and `FrameSet` classes introduced for more modular construction of framed models. `FrameAgentType` dedicated to simulation. [#1117](https://github.com/econ-ark/HARK/pull/1117)
- General control transitions based on decision rules in `FrameAgentType`. [#1117](https://github.com/econ-ark/HARK/pull/1117)
- Adds `distr_of_function` tool to calculate the distribution of a function of a discrete random variable. [#1144](https://github.com/econ-ark/HARK/pull/1144)
- Changes the `DiscreteDistribution` class to allow for arbitrary array-valued random variables. [#1146](https://github.com/econ-ark/HARK/pull/1146)
- Adds `IndShockRiskyAssetConsumerType` as agent which can invest savings all in safe asset, all in risky asset, a fixed share in risky asset, or optimize its portfolio. [#1107](https://github.com/econ-ark/HARK/issues/1107)
- Updates all HARK models to allow for age-varying interest rates. [#1150](https://github.com/econ-ark/HARK/pull/1150)
- Adds `DiscreteDistribution.expected` method which expects vectorized functions and is faster than `HARK.distribution.calc_expectation`. [#1156](https://github.com/econ-ark/HARK/pull/1156)
- Adds `DiscreteDistributionXRA` class which extends `DiscreteDistribution` to allow for underlying data to be stored in a `xarray.DataArray` object. [#1156](https://github.com/econ-ark/HARK/pull/1156)
- Adds keyword argument `labels` to `expected()` when using `DiscreteDistributionXRA` to allow for expressive functions that use labeled xarrays. [#1156](https://github.com/econ-ark/HARK/pull/1156)
- Adds a wrapper for [`interpolation.py`](https://github.com/EconForge/interpolation.py) for fast multilinear interpolation. [#1151](https://github.com/econ-ark/HARK/pull/1151)
- Adds support for the calculation of dreivatives in the `interpolation.py` wrappers. [#1157](https://github.com/econ-ark/HARK/pull/1157)
- Adds class `DecayInterp` to `econforgeinterp.py`. It implements interpolators that "decay" to some limiting function when extrapolating. [#1165](https://github.com/econ-ark/HARK/pull/1165)
- Add methods to non stochastically simulate an economy by computing transition matrices. Functions to compute transition matrices and ergodic distribution have been added [#1155](https://github.com/econ-ark/HARK/pull/1155).
- Fixes a bug that causes `t_age` and `t_cycle` to get out of sync when reading pre-computed mortality. [#1181](https://github.com/econ-ark/HARK/pull/1181)
- Adds Methods to calculate Heterogenous Agent Jacobian matrices. [#1185](https://github.com/econ-ark/HARK/pull/1185)
- Enhances `combine_indep_dstns` to work with labeled distributions (`DiscreteDistributionLabeled`). [#1191](https://github.com/econ-ark/HARK/pull/1191)
- Updates the `numpy` random generator from `RandomState` to `Generator`. [#1193](https://github.com/econ-ark/HARK/pull/1193)
- Turns the income and income+return distributions into `DiscreteDistributionLabeled` objects. [#1189](https://github.com/econ-ark/HARK/pull/1189)
- Creates `UtilityFuncCRRA` which is an object oriented utility function with a coefficient of constant relative risk aversion and includes derivatives and inverses. Also creates `UtilityFuncCobbDouglas`, `UtilityFuncCobbDouglasCRRA`, and `UtilityFuncConstElastSubs`. [#1168](https://github.com/econ-ark/HARK/pull/1168)
- Reorganizes `HARK.distribution`. All distributions now inherit all features from `scipy.stats`. New `ContinuousFrozenDistribution` and `DiscreteFrozenDistribution` to use `scipy.stats` distributions not yet implemented in HARK. New `Distribution.discretize(N, method = "***")` replaces `Distribution.approx(N)`. New `DiscreteDistribution.limit` attribute describes continuous origin and discretization method. [#1197](https://github.com/econ-ark/HARK/pull/1197).
- Creates new class of _labeled_ models under `ConsLabeledModel` that use xarray for more expressive modeling of underlying mathematical and economics variables. [#1177](https://github.com/econ-ark/HARK/pull/1177)

#### Minor Changes

- Updates the lognormal-income-process constructor from `ConsIndShockModel.py` to use `IndexDistribution`. [#1024](https://github.com/econ-ark/HARK/pull/1024), [#1115](https://github.com/econ-ark/HARK/pull/1115)
- Allows for age-varying unemployment probabilities and replacement incomes with the lognormal income process constructor. [#1112](https://github.com/econ-ark/HARK/pull/1112)
- Option to have newborn IndShockConsumerType agents with a transitory income shock in the first period. Default is false, meaning they only have a permanent income shock in period 1 and permanent AND transitory in the following ones. [#1126](https://github.com/econ-ark/HARK/pull/1126)
- Adds `benchmark` utility to profile the performance of `HARK` solvers. [#1131](https://github.com/econ-ark/HARK/pull/1131)
- Fixes scaling bug in Normal equiprobable approximation method. [1139](https://github.com/econ-ark/HARK/pull/1139)
- Removes the extra-dimension that was returned by `calc_expectations` in some instances. [#1149](https://github.com/econ-ark/HARK/pull/1149)
- Adds `HARK.distribution.expected` alias for `DiscreteDistribution.expected`. [#1156](https://github.com/econ-ark/HARK/pull/1156)
- Renames attributes in `DiscreteDistribution`: `X` to `atoms` and `pmf` to `pmv`. [#1164](https://github.com/econ-ark/HARK/pull/1164), [#1051](https://github.com/econ-ark/HARK/pull/1151), [#1159](https://github.com/econ-ark/HARK/pull/1159).
- Remove or replace automated tests that depend on brittle simulation results. [#1148](https://github.com/econ-ark/HARK/pull/1148)
- Updates asset grid constructor from `ConsIndShockModel.py` to allow for linearly-spaced grids when `aXtraNestFac == -1`. [#1172](https://github.com/econ-ark/HARK/pull/1172)
- Renames `DiscreteDistributionXRA` to `DiscreteDistributionLabeled` and updates methods [#1170](https://github.com/econ-ark/HARK/pull/1170)
- Renames `HARK.numba` to `HARK.numba_tools` [#1183](https://github.com/econ-ark/HARK/pull/1183)
- Adds the RNG seed as a property of `DiscreteDistributionLabeled` [#1184](https://github.com/econ-ark/HARK/pull/1184)
- Updates the `approx` method of `HARK.distributions.Uniform` to include the endpoints of the distribution with infinitesimally small (zero) probability mass. [#1180](https://github.com/econ-ark/HARK/pull/1180)
- Refactors tests to incorporate custom precision `HARK_PRECISION = 4`. [#1193](https://github.com/econ-ark/HARK/pull/1193)
- Cast `DiscreteDistribution.pmv` attribute as a `np.ndarray`. [#1199](https://github.com/econ-ark/HARK/pull/1199)
- Update structure of dynamic interest rate. [#1221](https://github.com/econ-ark/HARK/pull/1221)

### 0.12.0

Release Date: December 14, 2021

#### Major Changes

- FrameAgentType for modular definitions of agents [#865](https://github.com/econ-ark/HARK/pull/865) [#1064](https://github.com/econ-ark/HARK/pull/1064)
- Frame relationships with backward and forward references, with plotting example [#1071](https://github.com/econ-ark/HARK/pull/1071)
- PortfolioConsumerFrameType, a port of PortfolioConsumerType to use Frames [#865](https://github.com/econ-ark/HARK/pull/865)
- Input parameters for cyclical models now indexed by t [#1039](https://github.com/econ-ark/HARK/pull/1039)
- A IndexDistribution class for representing time-indexed probability distributions [#1018](https://github.com/econ-ark/HARK/pull/1018/).
- Adds new consumption-savings-portfolio model `RiskyContrib`, which represents an agent who can save in risky and risk-free assets but faces
  frictions to moving funds between them. To circumvent these frictions, he has access to an income-deduction scheme to accumulate risky assets.
  PR: [#832](https://github.com/econ-ark/HARK/pull/832). See [this forthcoming REMARK](https://github.com/Mv77/RiskyContrib) for the model's details.
- 'cycles' agent property moved from constructor argument to parameter [#1031](https://github.com/econ-ark/HARK/pull/1031)
- Uses iterated expectations to speed-up the solution of `RiskyContrib` when income and returns are independent [#1058](https://github.com/econ-ark/HARK/pull/1058).
- `ConsPortfolioSolver` class for solving portfolio choice model replaces `solveConsPortfolio` method [#1047](https://github.com/econ-ark/HARK/pull/1047)
- `ConsPortfolioDiscreteSolver` class for solving portfolio choice model when allowed share is on a discrete grid [#1047](https://github.com/econ-ark/HARK/pull/1047)
- `ConsPortfolioJointDistSolver` class for solving portfolio chioce model when the income and risky return shocks are not independent [#1047](https://github.com/econ-ark/HARK/pull/1047)

#### Minor Changes

- Using Lognormal.from_mean_std in the forward simulation of the RiskyAsset model [#1019](https://github.com/econ-ark/HARK/pull/1019)
- Fix bug in DCEGM's primary kink finder due to numpy no longer accepting NaN in integer arrays [#990](https://github.com/econ-ark/HARK/pull/990).
- Add a general class for consumers who can save using a risky asset [#1012](https://github.com/econ-ark/HARK/pull/1012/).
- Add Boolean attribute 'PerfMITShk' to consumption models. When true, allows perfect foresight MIT shocks to be simulated. [#1013](https://github.com/econ-ark/HARK/pull/1013).
- Track and update start-of-period (pre-income) risky and risk-free assets as states in the `RiskyContrib` model [1046](https://github.com/econ-ark/HARK/pull/1046).
- distribute_params now uses assign_params to create consistent output [#1044](https://github.com/econ-ark/HARK/pull/1044)
- The function that computes end-of-period derivatives of the value function was moved to the inside of `ConsRiskyContrib`'s solver [#1057](https://github.com/econ-ark/HARK/pull/1057)
- Use `np.fill(np.nan)` to clear or initialize the arrays that store simulations. [#1068](https://github.com/econ-ark/HARK/pull/1068)
- Add Boolean attribute 'neutral_measure' to consumption models. When true, simulations are more precise by allowing permanent shocks to be drawn from a neutral measure (see Harmenberg 2021). [#1069](https://github.com/econ-ark/HARK/pull/1069)
- Fix mathematical limits of model example in `example_ConsPortfolioModel.ipynb` [#1047](https://github.com/econ-ark/HARK/pull/1047)
- Update `ConsGenIncProcessModel.py` to use `calc_expectation` method [#1072](https://github.com/econ-ark/HARK/pull/1072)
- Fix bug in `calc_normal_style_pars_from_lognormal_pars` due to math error. [#1076](https://github.com/econ-ark/HARK/pull/1076)
- Fix bug in `distribute_params` so that `AgentCount` parameter is updated. [#1089](https://github.com/econ-ark/HARK/pull/1089)
- Fix bug in 'vFuncBool' option for 'MarkovConsumerType' so that the value function may now be calculated. [#1095](https://github.com/econ-ark/HARK/pull/1095)

### 0.11.0

Release Date: March 4, 2021

#### Major Changes

- Converts non-mathematical code to PEP8 compliant form [#953](https://github.com/econ-ark/HARK/pull/953)
- Adds a constructor for LogNormal distributions from mean and standard deviation [#891](https://github.com/econ-ark/HARK/pull/891/)
- Uses new LogNormal constructor in ConsPortfolioModel [#891](https://github.com/econ-ark/HARK/pull/891/)
- calcExpectations method for taking the expectation of a distribution over a function [#884](https://github.com/econ-ark/HARK/pull/884/] (#897)[https://github.com/econ-ark/HARK/pull/897/)
- Implements the multivariate normal as a supported distribution, with a discretization method. See [#948](https://github.com/econ-ark/HARK/pull/948).
- Centralizes the definition of value, marginal value, and marginal marginal value functions that use inverse-space
  interpolation for problems with CRRA utility. See [#888](https://github.com/econ-ark/HARK/pull/888).
- MarkovProcess class used in ConsMarkovModel, ConsRepAgentModel, ConsAggShockModel [#902](https://github.com/econ-ark/HARK/pull/902) [#929](https://github.com/econ-ark/HARK/pull/929)
- replace HARKobject base class with MetricObject and Model classes [#903](https://github.com/econ-ark/HARK/pull/903/)
- Add **repr** and **eq** methods to Model class [#903](https://github.com/econ-ark/HARK/pull/903/)
- Adds SSA life tables and methods to extract survival probabilities from them [#986](https://github.com/econ-ark/HARK/pull/906).
- Adds the U.S. CPI research series and tools to extract inflation adjustments from it [#930](https://github.com/econ-ark/HARK/pull/930).
- Adds a module for extracting initial distributions of permanent income (`pLvl`) and normalized assets (`aNrm`) from the SCF [#932](https://github.com/econ-ark/HARK/pull/932).
- Fix the return fields of `dcegm/calcCrossPoints`[#909](https://github.com/econ-ark/HARK/pull/909).
- Corrects location of constructor documentation to class string for Sphinx rendering [#908](https://github.com/econ-ark/HARK/pull/908)
- Adds a module with tools for parsing and using various income calibrations from the literature. It includes the option of using life-cycle profiles of income shock variances from [Sabelhaus and Song (2010)](https://www.sciencedirect.com/science/article/abs/pii/S0304393210000358). See [#921](https://github.com/econ-ark/HARK/pull/921), [#941](https://github.com/econ-ark/HARK/pull/941), [#980](https://github.com/econ-ark/HARK/pull/980).
- remove "Now" from model variable names [#936](https://github.com/econ-ark/HARK/pull/936)
- remove Model.**call**; use Model init in Market and AgentType init to standardize on parameters dictionary [#947](https://github.com/econ-ark/HARK/pull/947)
- Moves state MrkvNow to shocks['Mrkv'] in AggShockMarkov and KrusellSmith models [#935](https://github.com/econ-ark/HARK/pull/935)
- Replaces `ConsIndShock`'s `init_lifecycle` with an actual life-cycle calibration [#951](https://github.com/econ-ark/HARK/pull/951).

#### Minor Changes

- Move AgentType constructor parameters docs to class docstring so it is rendered by Sphinx.
- Remove uses of deprecated time.clock [#887](https://github.com/econ-ark/HARK/pull/887)
- Change internal representation of parameters to Distributions to ndarray type
- Rename IncomeDstn to IncShkDstn
- AgentType simulate() method now returns history. [#916](https://github.com/econ-ark/HARK/pull/916)
- Rename DiscreteDistribution.drawDiscrete() to draw()
- Update documentation and warnings around IncShkDstn [#955](https://github.com/econ-ark/HARK/pull/955)
- Adds csv files to `MANIFEST.in`. [957](https://github.com/econ-ark/HARK/pull/957)

### 0.10.8

Release Date: Nov. 05 2020

#### Major Changes

- Namespace variables for the Market class [#765](https://github.com/econ-ark/HARK/pull/765)
- We now have a Numba based implementation of PerfForesightConsumerType model available as PerfForesightConsumerTypeFast [#774](https://github.com/econ-ark/HARK/pull/774)
- Namespace for exogenous shocks [#803](https://github.com/econ-ark/HARK/pull/803)
- Namespace for controls [#855](https://github.com/econ-ark/HARK/pull/855)
- State and poststate attributes replaced with state_now and state_prev namespaces [#836](https://github.com/econ-ark/HARK/pull/836)

#### Minor Changes

- Use shock_history namespace for pre-evaluated shock history [#812](https://github.com/econ-ark/HARK/pull/812)
- Fixes seed of PrefShkDstn on initialization and add tests for simulation output
- Reformat code style using black

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

- Add Bellman equations for cyclical model example [#600](https://github.com/econ-ark/HARK/pull/600)

- read_shocks now reads mortality as well [#613](https://github.com/econ-ark/HARK/pull/613)

- Discrete probability distributions are now classes [#610](https://github.com/econ-ark/HARK/pull/610)

#### Minor Changes

### 0.10.5

Release Date: 24-03-2020

#### Major Changes

- Default parameters dictionaries for ConsumptionSaving models have been moved from ConsumerParameters to nearby the classes that use them. [#527](https://github.com/econ-ark/HARK/pull/527)

- Improvements and cleanup of ConsPortfolioModel, and adding the ability to specify an age-varying list of RiskyAvg and RiskyStd. [#577](https://github.com/econ-ark/HARK/pull/527)

- Rewrite and simplification of ConsPortfolioModel solver. [#594](https://github.com/econ-ark/HARK/pull/594)

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

- Revert pre-solve commit due to bug. ([#363](https://github.com/econ-ark/HARK/pull/363))

### 0.10.1.dev1

Release Date: 07-20-2019

#### Breaking Changes

- See #302 under minor changes.

#### Major Changes

- Adds BayerLuetticke notebooks and functionality. ([#328](https://github.com/econ-ark/HARK/pull/328))

#### Minor Changes

- Fixes one-asset HANK models for endowment economy (had MP wired in as the shock). ([#355](https://github.com/econ-ark/HARK/pull/355))
- Removes jupytext \*.py files. ([#354](https://github.com/econ-ark/HARK/pull/354))
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
- Fixes unintended behavior in default simDeath(). Previously, all agents would die off in the first period, but they were meant to always survive. ([#302](https://github.com/econ-ark/HARK/pull/302)) **Warning**: Potentially breaking change.

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
- Lints a variety of files. These PRs include some additional/related minor changes, like replacing an exec function, removing some lambdas, adding some files to .gitignore, etc. ([274](https://github.com/econ-ark/HARK/pull/274), [276](https://github.com/econ-ark/HARK/pull/276), [277](https://github.com/econ-ark/HARK/pull/277), [278](https://github.com/econ-ark/HARK/pull/278), [281](https://github.com/econ-ark/HARK/pull/281))
- Adds vim swp files to gitignore. ([269](https://github.com/econ-ark/HARK/pull/269))
- Adds version dunder in init. ([265](https://github.com/econ-ark/HARK/pull/265))
- Adds flake8 to requirements.txt and config. ([261](https://github.com/econ-ark/HARK/pull/261))
- Adds some unit tests for IndShockConsumerType. ([256](https://github.com/econ-ark/HARK/pull/256))

### 0.10.0.dev2

Release Date: 04-18-2019

#### Major Changes

None

#### Minor Changes

- Fix verbosity check in ConsIndShockModel. ([250](https://github.com/econ-ark/HARK/pull/250))

#### Other Changes

None

### 0.10.0.dev1

Release Date: 04-12-2019

#### Major Changes

- Adds [tools](https://github.com/econ-ark/HARK/blob/master/HARK/dcegm.py) to solve problems that arise from the interaction of discrete and continuous variables, using the [DCEGM](https://github.com/econ-ark/DemARK/blob/master/notebooks/DCEGM-Upper-Envelope.ipynb) method of [Iskhakov et al.](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE643), who apply the their discrete-continuous solution algorithm to the problem of optimal endogenous retirement; their results are replicated using our new tool [here](https://github.com/econ-ark/EndogenousRetirement/blob/master/Endogenous-Retirement.ipynb). ([226](https://github.com/econ-ark/HARK/pull/226))
- Parameters of ConsAggShockModel.CobbDouglasEconomy.updateAFunc and ConsAggShockModel.CobbDouglasMarkovEconomy.updateAFunc that govern damping and the number of discarded 'burn-in' periods were previously hardcoded, now proper instance-level parameters. ([244](https://github.com/econ-ark/HARK/pull/244))
- Improve accuracy and performance of functions for evaluating the integrated value function and conditional choice probabilities for models with extreme value type I taste shocks. ([242](https://github.com/econ-ark/HARK/pull/242))
- Add calcLogSum, calcChoiceProbs, calcLogSumChoiceProbs to HARK.interpolation. ([209](https://github.com/econ-ark/HARK/pull/209), [217](https://github.com/econ-ark/HARK/pull/217))
- Create tool to produce an example "template" of a REMARK based on SolvingMicroDSOPs. ([176](https://github.com/econ-ark/HARK/pull/176))

#### Minor Changes

- Moved old utilities tests. ([245](https://github.com/econ-ark/HARK/pull/245))
- Deleted old files related to "cstwMPCold". ([239](https://github.com/econ-ark/HARK/pull/239))
- Set numpy floating point error level to ignore. ([238](https://github.com/econ-ark/HARK/pull/238))
- Fixed miscellaneous imports. ([212](https://github.com/econ-ark/HARK/pull/212), [224](https://github.com/econ-ark/HARK/pull/224), [225](https://github.com/econ-ark/HARK/pull/225))
- Improve the tests of buffer stock model impatience conditions in IndShockConsumerType. ([219](https://github.com/econ-ark/HARK/pull/219))
- Add basic support for Travis continuous integration testing. ([208](https://github.com/econ-ark/HARK/pull/208))
- Add SciPy to requirements.txt. ([207](https://github.com/econ-ark/HARK/pull/207))
- Fix indexing bug in bilinear interpolation. ([194](https://github.com/econ-ark/HARK/pull/194))
- Update the build process to handle Python 2 and 3 compatibility. ([172](https://github.com/econ-ark/HARK/pull/172))
- Add MPCnow attribute to ConsGenIncProcessModel. ([170](https://github.com/econ-ark/HARK/pull/170))
- All standalone demo files have been removed. The content that was in these files can now be found in similarly named Jupyter notebooks in the DEMARK repository. Some of these notebooks are also linked from econ-ark.org. ([229](https://github.com/econ-ark/HARK/pull/229), [243](https://github.com/econ-ark/HARK/pull/243))

#### Other Notes

- Not all changes from 0.9.1 may be listed in these release notes. If you are having trouble addressing a breaking change, please reach out to us.
