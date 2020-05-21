<div align="center">
  <a href="https://econ-ark.org">
    <img src="https://econ-ark.org/assets/img/econ-ark-logo.png" align="center">
  </a>
  <br>
  <br>

[![Anaconda Cloud](https://anaconda.org/conda-forge/econ-ark/badges/version.svg?style=flat)](https://anaconda.org/conda-forge/econ-ark)
[![PyPi](https://img.shields.io/pypi/v/econ-ark.png?style=flat)](https://pypi.org/project/econ-ark/)
[![Documentation Status](https://readthedocs.org/projects/hark/badge/?style=flat&version=latest)](https://hark.readthedocs.io/en/latest/?badge=latest)
[![GitHub Good First Issues](https://img.shields.io/github/issues/econ-ark/HARK/good%20first%20issue.svg)](https://github.com/econ-ark/HARK/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/econ-ark/DemARK/master)
[![DOI](https://zenodo.org/badge/50448254.svg)](https://zenodo.org/badge/latestdoi/50448254)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/)
[![Donate](https://img.shields.io/badge/donate-$2-brightgreen.svg)](https://numfocus.salsalabs.org/donate-to-econ-ark/index.html)
[![Actions Status](https://github.com/econ-ark/hark/workflows/HARK%20build%20on%20MacOS%2C%20Ubuntu%20and%20Windows/badge.svg
)](https://github.com/econ-ark/hark/actions)


<!--
   Badges to be created:

[![Azure](https://dev.azure.com/econ-ark/HARK/_apis/build/status/azure-pipeline%20econ-ark.hark)](
    https://dev.azure.com/econ-ark/hark/_build/latest?definitionId=5)

[![codecov](https://codecov.io/gh/econ-ark/hark/branch/master/graph/badge.svg)](
    https://codecov.io/gh/econ-ark/hark)

-->

</div>

# Heterogeneous Agents Resources and toolKit (HARK)

HARK is a toolkit for the structural modeling of economic choices of optimizing and non-optimizing heterogeneous agents. For more information on using HARK, see the [Econ-ARK Website](https://econ-ark.org).

The Econ-ARK project uses an [open governance model](./GOVERNANCE.md) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/). Consider making a [tax-deductible donation](https://numfocus.salsalabs.org/donate-to-econ-ark/index.html) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
  <a href="https://numfocus.org/project/econ-ark">
    <img height="60px" src="https://numfocus.org/wp-content/uploads/2018/01/optNumFocus_LRG.png" align="center">
  </a>
</div>
<br>

**This project is bound by a [Code of Conduct](/.github/CODE_OF_CONDUCT.md).**

# Table of Contents

* [Install](#install)
* [Usage](#usage)
* [Citation](#citation)
* [Support](#support)
* [Release Types](#release-types)
* [API Documentation](#api-documentation)
* [Introduction](#introduction)
  * [For Students: A Gentle Introduction to Hark](#for-students-a-gentle-introduction-to-hark)
  * [For Economists: Structural Modeling with Hark](#for-economists-structural-modeling-with-hark)
  * [For Computational Economics Developers](#for-computational-economics-developers)
* [Contributing to HARK](#contributing-to-hark)
* [Current Project Team Members](#current-project-team-members)
  * [Founders](#founders)
  * [TSC (Technical Steering Commitee)](#tsc-technical-steering-committee)
  * [Collaborators](#collaborators)
  * [Release Team](#release-team)
* [Contributors](#contributors)
* [Disclaimer](#disclaimer)

## Install

Install from [Anaconda Cloud](https://docs.anaconda.com/anaconda/install/) by running:

`conda install -c conda-forge econ-ark`

Install from [PyPi](https://pypi.org/) by running:

`pip install econ-ark`

## Usage

We start with almost the simplest possible consumption model: A consumer with CRRA utility 

<div align="center">
  <img height="30px" src="https://github.com/econ-ark/HARK/blob/master/doc/images/usage-crra-utility-function.png">
</div> 

has perfect foresight about everything except the (stochastic) date of death.

The agent's problem can be written in [Bellman form](https://en.wikipedia.org/wiki/Bellman_equation) as:

<div align="center">
  <img height="80px" src="https://github.com/econ-ark/HARK/blob/master/doc/images/usage-agent-problem-bellman-form.png">
</div>

<br>

To model the above problem, start by importing the `PerfForesightConsumerType` model from the appropriate `HARK` module then create an agent instance using the appropriate paramaters:

```python
import HARK 

from HARK.ConsumptionSaving.ConsIndShockModel import PerfForesightConsumerType

PF_params = {
    'CRRA' : 2.5,           # Relative risk aversion
    'DiscFac' : 0.96,       # Discount factor
    'Rfree' : 1.03,         # Risk free interest factor
    'LivPrb' : [0.98],      # Survival probability
    'PermGroFac' : [1.01],  # Income growth factor
    'T_cycle' : 1,
    'cycles' : 0,
    'AgentCount' : 10000
}

# Create an instance of a Perfect Foresight agent with the above paramaters
PFexample = PerfForesightConsumerType(**PF_params) 

```
Once the model is created, ask the the agent to solve the problem with `.solve()`:

```python
# Tell the agent to solve the problem
PFexample.solve()
```

Solving the problem populates the agent's `.solution` list attribute with solutions to each period of the problem. In the case of an infinite horizon model, there is just one element in the list, at **index-zero**. 

You can retrieve the solution's consumption function from the `.cFunc` attribute:

```python
# Retrieve the consumption function of the solution
PFexample.solution[0].cFunc
```

Or you can retrieve the solved value for human wealth normalized by permanent income from the solution's `.hNrm` attribute:

```python
# Retrieve the solved value for human wealth normalized by permanent income 
PFexample.solution[0].hNrm
```
For a detailed explanation of the above example please see the demo notebook [*A Gentle Introduction to HARK*](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks/Gentle-Intro-To-HARK.ipynb).

For more examples please visit the [econ-ark/DemARK](https://github.com/econ-ark/DemARK) repository.

## Citation

If using Econ-ARK in your work or research please [cite our Digital Object Identifier](http://doi.org/10.5281/zenodo.1001068) or copy the BibTex below.

[![DOI](https://zenodo.org/badge/50448254.svg)](https://zenodo.org/badge/latestdoi/50448254)

[1] Carroll, Christopher D, Palmer, Nathan, White, Matthew N., Kazil, Jacqueline, Low, David C, & Kaufman, Alexander. (2017, October 3). *econ-ark/HARK*

**BibText**

```
@InProceedings{carroll_et_al-proc-scipy-2018,
  author    = { {C}hristopher {D}. {C}arroll and {A}lexander {M}. {K}aufman and {J}acqueline {L}. {K}azil and {N}athan {M}. {P}almer and {M}atthew {N}. {W}hite },
  title     = { {T}he {E}con-{A}{R}{K} and {H}{A}{R}{K}: {O}pen {S}ource {T}ools for {C}omputational {E}conomics },
  booktitle = { {P}roceedings of the 17th {P}ython in {S}cience {C}onference },
  pages     = { 25 - 30 },
  year      = { 2018 },
  editor    = { {F}atih {A}kici and {D}avid {L}ippa and {D}illon {N}iederhut and {M} {P}acer },
  doi       = { 10.25080/Majora-4af1f417-004 }
}
```

For more on acknowledging Econ-ARK [visit our website](https://econ-ark.org/acknowledging/).

## Support

Looking for help? Please open a [GitHub issue](https://github.com/econ-ark/HARK/issues/new) or reach out to the [TSC](#tsc-technical-steering-committee).

For more support options see [SUPPORT.md](./.github/SUPPORT.md).

## Release Types

* **Current**: Under active development. Code for the Current release is in the branch for its major version number (for example, v1.x).
* **Development**: Under active development. Code for the Current release is in the development.
* **Nightly**: Code from the master branch built every night when there are changes. Use with caution.

Current releases follow [Semantic Versioning](https://semver.org/). For more information please see the [Release documentation](doc/release/README.md).

## API Documentation

Documentation for the latest release is at [hark.readthedocs.io](https://hark.readthedocs.io/en/latest/). Version-specific documentation is available from the same source.

## Introduction

### For Students: A Gentle Introduction to HARK

Most of what economists have done so far with 'big data' has been like what Kepler did with astronomical data: Organizing the data, and finding patterns and regularities and interconnections. 

An alternative approach called 'structural modeling' aims to do, for economic data, what Newton did for astronomical data: Provide a deep and rigorous mathematical (or computational) framework that distills the essence of the underlying behavior that produces the 'big data.'

The notebook [*A Gentle Introduction to HARK*](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks/Gentle-Intro-To-HARK.ipynb) details how you can easily utilize our toolkit for structural modeling. Starting with a simple [Perfect Foresight Model](https://en.wikipedia.org/wiki/Rational_expectations) we solve an agent problem, then experiment with adding [income shocks](https://en.wikipedia.org/wiki/Shock_(economics)) and changing constructed attributes.

### For Economists: Structural Modeling with HARK

Dissatisfaction with the ability of Representative Agent models to answer important questions raised by the Great Recession has led to a strong movement in the macroeconomics literature toward 'Heterogeneous Agent' models, in which microeconomic agents (consumers; firms) solve a structural problem calibrated to match microeconomic data; aggregate outcomes are derived by explicitly simulating the equilibrium behavior of populations solving such models.

The same kinds of modeling techniques are also gaining popularity among microeconomists, in areas ranging from labor economics to industrial organization. In both macroeconomics and structural micro, the chief barrier to the wide adoption of these techniques has been that programming a structural model has, until now, required a great deal of specialized knowledge and custom software development.

HARK provides a robust, well-designed, open-source toolkit for building such models much more efficiently than has been possible in the past.

Our [*DCEGM Upper Envelope*](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks%2FDCEGM-Upper-Envelope.ipynb) notebook illustrates using HARK to replicate the [Iskhakov, Jørgensen, Rust, and Schjerning paper](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE643) for solving the discrete-continuous retirement saving problem. 

The notebook [*Making Structural Estimates From Empirical Results*](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=notebooks%2FStructural-Estimates-From-Empirical-MPCs-Fagereng-et-al.ipynb) is another demonstration of using HARK to conduct a quick structural estimation based on Table 9 of [*MPC Heterogeneity and Household Balance Sheets* by Fagereng, Holm, and Natvik](https://www.ssb.no/en/forskning/discussion-papers/_attachment/286054?_ts=158af859c98).

### For Computational Economics Developers

HARK provides a modular and extensible open-source toolkit for solving heterogeneous-agent partial-and general-equilibrium structural models. The code for such models has always been handcrafted, idiosyncratic, poorly documented, and sometimes not generously shared from leading researchers to outsiders. The result being that it can take years for a new researcher to become proficient. By building an integrated system from the bottom up using object-oriented programming techniques and other tools, we aim to provide a platform that will become a focal point for people using such models. 

HARK is written in Python, making significant use of libraries such as numpy and scipy which offer a wide array of mathematical and statistical functions and tools. Our modules are generally categorized into Tools (mathematical functions and techniques), Models (particular economic models and solvers) and Applications (use of tools to simulate an economic phenomenon). 

For more information on how you can create your own Models or use Tools and Model to create Applications please see the [Architecture documentation](./doc/architecture/README.md).

### Contributing to HARK

**We want contributing to Econ-ARK to be fun, enjoyable, and educational for anyone, and everyone.**

Contributions go far beyond pull requests and commits. Although we love giving you the opportunity to put your stamp on HARK, we are also thrilled to receive a variety of other contributions including:
* Documentation updates, enhancements, designs, or bugfixes
* Spelling or grammar fixes
* REAME.md corrections or redesigns
* Adding unit, or functional tests
* [Triaging GitHub issues](https://github.com/econ-ark/HARK/issues?utf8=%E2%9C%93&q=label%3A%E2%80%9DTag%3A+Triage+Needed%E2%80%9D+) -- e.g. pointing out the relevant files, checking for reproducibility
* [Searching for #econ-ark on twitter](https://twitter.com/search?q=webpack) and helping someone else who needs help
* Answering questions from StackOverflow tagged with [econ-ark](https://stackoverflow.com/questions/tagged/econ-ark)
* Teaching others how to contribute to HARK
* Blogging, speaking about, or creating tutorials about HARK
* Helping others in our mailing list

If you are worried or don’t know how to start, you can always reach out to the [TSC](#tsc-technical-steering-committee) or simply submit [an issue](https://github.com/econ-ark/HARK/issues/new) and a member can help give you guidance!

**After your first contribution please let us know and we will add you to the Contributors list below!**

To install for development see the [Quickstart Guide](./Documentation/quick-start).

For more information on contributing to HARK please see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Current Project Team Members

For information about the governance of the Econ-ARK project, see
[GOVERNANCE.md](./GOVERNANCE.md).

Collaborators follow the [COLLABORATOR_GUIDE.md](./COLLABORATOR_GUIDE.md) in maintaining the Econ-ARK project.

### Founders
Econ-ARK was created by [**Christopher D. Carroll**](http://www.econ2.jhu.edu/people/ccarroll/), Professor of Economics at the Johns Hopkins University. 

Founders of the current repository also include:
* [shaunagm](https://github.com/shaunagm) - **Shauna Gordon-McKeon** &lt;shaunagm@gmail.com&gt; (she/her)
* [sbrice](https://github.com/sbrice) - [**Samuel Brice**](https://medium.com/@sbrice) &lt;brices@gmail.com&gt; (he/him)

### TSC (Technical Steering Committee)
* [llorracc](https://github.com/llorracc) - [**Christopher “Chris” D. Carroll**]((http://www.econ2.jhu.edu/people/ccarroll/)) &lt;ccarroll@llorracc.org&gt; (he/him)
* [sbrice](https://github.com/sbrice) - [**Samuel Brice**](https://medium.com/@sbrice) &lt;brices@gmail.com&gt; (he/him)
* [shaunagm](https://github.com/shaunagm) - **Shauna Gordon-McKeon** &lt;shaunagm@gmail.com&gt; (she/her)

### Collaborators
* [albop](https://github.com/albop) - **Pablo Winant** &lt;pablo.winant@gmail.com&gt; (he/him)
* [DrDrij](https://github.com/DrDrij) - **Andrij Stachurski** &lt;dr.drij@gmail.com&gt; (he/him)

### Release Team
* [shaunagm](https://github.com/shaunagm) - **Shauna Gordon-McKeon** &lt;shaunagm@gmail.com&gt; (she/her)

## Contributors
* [ericholscher](https://github.com/ericholscher) - **Eric Holscher** (he/him)

## Disclaimer

This is a beta version of HARK.  The code has not been extensively tested as it should be.  We hope it is useful, but there are absolutely no guarantees (expressed or implied) that it works or will do what you want.  Use at your own risk.  And please, let us know if you find bugs by posting an issue to [the GitHub page](https://github.com/econ-ark/HARK)!
