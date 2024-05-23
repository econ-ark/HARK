# Quick Start Guide

## Installing HARK

HARK is an open source project that is compatible with Python 3. Currently, we recommend using version 3.10 or lower.

The simplest way to install HARK is to use [pip](https://pip.pypa.io/en/stable/installation/).

Before installing HARK, we recommend creating a new virtual environment, which isolates the installation of `econ-ark` from the installations of any other Python tools and packages, thus avoiding conflicts.

The easiest way to get started with managing environments is to use `conda`, which is packaged with either the [Anaconda](https://anaconda.com/) distribution or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). To create a new virtual environment and install `econ-ark`, enter this in your command line:

```
conda create -n econ-ark python=3.10
conda activate econ-ark
pip install econ-ark
```

## Learning HARK

We have a set of 30-second [Elevator Spiels](https://github.com/econ-ark/PARK/blob/master/Elevator-Spiels.md#capsule-summaries-of-what-the-econ-ark-project-is) describing the project, tailored to people with several different kinds of background.

The most broadly applicable advice is to go to [Econ-ARK](https://econ-ark.org) and click on "Notebooks", and choose [A Gentle Introduction to HARK](https://docs.econ-ark.org/example_notebooks/Gentle-Intro-To-HARK.html) which will launch as a [jupyter notebook](https://jupyter.org/).

### [For people with a technical/scientific/computing background but little economics background](https://github.com/econ-ark/PARK/blob/master/Elevator-Spiels.md#for-people-with-a-technicalscientificcomputing-background-but-no-economics-background)

- A good starting point is [A Gentle Introduction to HARK](https://docs.econ-ark.org/example_notebooks/Gentle-Intro-To-HARK.html) which provides a light economic intuition.

### [For economists who have done some structural modeling](https://github.com/econ-ark/PARK/blob/master/Elevator-Spiels.md#for-economists-who-have-done-some-structural-modeling)

- A full replication of the [Iskhakov, Jørgensen, Rust, and Schjerning](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE643) paper for solving the discrete-continuous retirement saving problem

  - An informal discussion of the issues involved is [here](https://github.com/econ-ark/DemARK/blob/master/notebooks/DCEGM-Upper-Envelope.ipynb) (part of the [DemARK](https://github.com/econ-ark/DemARK) repo)

- [Structural-Estimates-From-Empirical-MPCs](https://github.com/econ-ark/DemARK/blob/master/notebooks/Structural-Estimates-From-Empirical-MPCs-Fagereng-et-al.ipynb) is an example of the use of the toolkit in a discussion of a well known paper. (Yes, it is easy enough to use that you can estimate a structural model on somebody else's data in the limited time available for writing a discussion)

### [For economists who have not yet done any structural modeling but might be persuadable to start](https://github.com/econ-ark/PARK/blob/master/Elevator-Spiels.md#for-economists-who-have-not-yet-done-any-structural-modeling-but-might-be-persuadable-to-start)

- Start with [A Gentle Introduction to HARK](https://docs.econ-ark.org/example_notebooks/Gentle-Intro-To-HARK.html) to get your feet wet

- A simple indirect inference/simulated method of moments structural estimation along the lines of Gourinchas and Parker's 2002 Econometrica paper or Cagetti's 2003 paper is performed by the [SolvingMicroDSOPs](https://github.com/econ-ark/SolvingMicroDSOPs/) [REMARK](https://github.com/econ-ark/REMARK); this code implements the solution methods described in the corresponding section of [these lecture notes](https://llorracc.github.io/SolvingMicroDSOPs/).

### [For Other Developers of Software for Computational Economics](https://github.com/econ-ark/PARK/blob/master/Elevator-Spiels.md#for-other-developers-of-software-for-computational-economics)

- Our workhorse module is [ConsIndShockModel.py](https://github.com/econ-ark/HARK/blob/master/HARK/ConsumptionSaving/ConsIndShockModel.py) which includes the IndShockConsumerType. A short explanation about the Agent Type can be found [here](https://docs.econ-ark.org/example_notebooks/IndShockConsumerType.html) and an introduction how it is solved [here](https://docs.econ-ark.org/example_notebooks/HowWeSolveIndShockConsumerType.html).

### Demonstrations on using HARK

Most of the modules in HARK are just collections of tools. There are a few demonstrations/applications that use the tools that you automatically get when you install HARK -- they are available in [Overview & Examples](https://docs.econ-ark.org/overview/index.html). A much larger set of uses of HARK can be found at two repositories:

- [DemARK](https://github.com/econ-ark/DemARK): Demonstrations of the use of HARK
- [REMARK](https://github.com/econ-ark/REMARK): Replications of existing papers made using HARK

You will want to obtain your own local copy of these repos using:

```
git clone https://github.com/econ-ark/DemARK.git
git clone https://github.com/econ-ark/REMARK.git
```

Once you have downloaded them, you will find that each repo contains a `notebooks` directory that contains a number of [jupyter notebooks](https://jupyter.org/). You can either view them in your integrated development environment (IDE) -- such as [VS Code](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/) -- or if you have `jupyter` installed, launch the Jupyter notebook tool using the command line:

```
cd [directory containing the repository]
jupyter notebook
```

## Next steps

To learn more about how to use HARK, check the next sections in this documentation, in particular the example notebooks. For instructions on making changes to HARK, refer to our [contributing guide](https://docs.econ-ark.org/guides/contributing.html).
