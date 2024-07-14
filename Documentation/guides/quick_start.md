# Quick Start Guide

## Installing HARK

HARK is an open source project that is compatible with Python 3. Currently, we recommend using version 3.10 or higher.

The simplest way to install HARK is to use [pip](https://pip.pypa.io/en/stable/installation/) by running `pip install econ-ark` in your command line.

Before installing HARK, we recommend creating a new virtual environment, which isolates the installation of `econ-ark` from the installations of any other Python tools and packages, thus avoiding conflicts.

The easiest way to get started with managing environments is to use `conda`, which is packaged with either the [Anaconda](https://anaconda.com/) distribution or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). To create a new virtual environment and install `econ-ark`, enter this in your command line:

```
conda create -n econ-ark python=3.10
conda activate econ-ark
pip install econ-ark
```

## Learning HARK

We've prepared a set of 30-second Elevator Spiels describing the project, tailored to people with several different kinds of background.

To start learning HARK we recommend working through the [Overview and Examples](https://docs.econ-ark.org/overview/index.html) section starting with [A Gentle Introduction to HARK](https://docs.econ-ark.org/example_notebooks/Gentle-Intro-To-HARK.html).

:::{dropdown} For people without a technical/scientific/computing background
:color: secondary
:icon: info
Recent years have seen major advances in the ability of computational tools to explain the economic behavior of households, firms, and whole economies. But a major impediment to the widespread adoption of these techniques among economists has been the extent to which the advances are the culmination of years of development of intricate and hand-crafted (but mutually incomprehensible) computational tools by a few pioneering scholars and their students. The aim of the Econ-ARK project is to make it much easier for new scholars to begin using these techniques, by providing a modern, robust, open-source set of high-quality computational tools with components that can be mixed, matched, and extended to address the wide variety of problems across all fields of economics that can be effectively studied using such tools.
:::

:::{dropdown} For people with a technical/scientific/computing background but little economics background
:color: secondary
:icon: info

Most of what economists have done so far with 'big data' has been like what Kepler did with astronomical data: Organizing the data, and finding patterns and regularities and interconnections. An alternative approach called 'structural modeling' aims to do, for economic data, what Newton did for astronomical data: Provide a deep and rigorous mathematical (or computational) framework that distills the essence of the underlying behavior that produces the 'big data.' But structural techniques are so novel and computationally difficult that few economists have mastered them. The aim of the Econ-ARK project is to make it much, much easier for new scholars to do structural modeling, by providing a well-documented, open source codebase that contains the core techniques and can be relatively easily adapted to address many different questions.
:::

:::{dropdown} For economists who have done some structural modeling
:color: secondary
:icon: info

The Econ-ARK project is motivated by a sense that quantitative structural modeling of economic agents' behavior (consumers; firms), at present, is roughly like econometric modeling in the 1960s: Lots of theoretical results are available and a great deal can be done in principle, but actually using such tools for any specific research question requires an enormous investment of a scholar's time and attention to learn techniques that are fundamentally not related to economics but instead are algorithmic/computational (in the 1960s, e.g., inverting matrices; now, e.g., solving dynamic stochastic optimization problems). The toolkit is built using the suite of open source, transparent tools for collaborative software development that have become ubiquitous in other fields in the last few years: Github, object-oriented coding, and methods that make it much easier to produce plug-and-play software modules that can be (relatively) easily combined, enhanced and adapted to address new problems.

After working through the [Overview and Examples](https://docs.econ-ark.org/overview/index.html) section:
- A full replication of the [Iskhakov, Jørgensen, Rust, and Schjerning](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE643) paper for solving the discrete-continuous retirement saving problem

  - An informal discussion of the issues involved is [here](https://github.com/econ-ark/DemARK/blob/master/notebooks/DCEGM-Upper-Envelope.ipynb) (part of the [DemARK](https://github.com/econ-ark/DemARK) repo)

- [Structural-Estimates-From-Empirical-MPCs](https://github.com/econ-ark/DemARK/blob/master/notebooks/Structural-Estimates-From-Empirical-MPCs-Fagereng-et-al.ipynb) is an example of the use of the toolkit in a discussion of a well known paper. (Yes, it is easy enough to use that you can estimate a structural model on somebody else's data in the limited time available for writing a discussion)
:::

:::{dropdown} For economists who have not yet done any structural modeling but might be persuadable to start
:color: secondary
:icon: info

Dissatisfaction with the ability of Representative Agent models to answer important questions raised by the Great Recession has led to a strong movement in the macroeconomics literature toward 'Heterogeneous Agent' models, in which microeconomic agents (consumers; firms) solve a structural problem calibrated to match microeconomic data; aggregate outcomes are derived by explicitly simulating the equilibrium behavior of populations solving such models. The same kinds of modeling techniques are also gaining popularity among microeconomists, in areas ranging from labor economics to industrial organization. In both macroeconomics and structural micro, the chief barrier to the wide adoption of these techniques has been that programming a structural model has, until now, required a great deal of specialized knowledge and custom software development. The aim of the Econ-ARK project is to provide a robust, well-designed, open-source infrastructure for building such models much more efficiently than has been possible in the past.

After working through the [Overview and Examples](https://docs.econ-ark.org/overview/index.html) section:
- A simple indirect inference/simulated method of moments structural estimation along the lines of Gourinchas and Parker's 2002 Econometrica paper or Cagetti's 2003 paper is performed by the [SolvingMicroDSOPs](https://github.com/econ-ark/SolvingMicroDSOPs/) [REMARK](https://github.com/econ-ark/REMARK); this code implements the solution methods described in the corresponding section of [these lecture notes](https://llorracc.github.io/SolvingMicroDSOPs/).
:::

:::{dropdown} For Other Developers of Software for Computational Economics
:icon: info
:color: secondary

The Econ-ARK project's aim is to create a modular and extensible open-source toolkit for solving heterogeneous-agent partial-and general-equilibrium structural models. The code for such models has always been handcrafted, idiosyncratic, poorly documented, and sometimes not generously shared from leading researchers to outsiders. The result that it can take years for a new researcher to become proficient. Building an integrated system from the bottom up using object-oriented programming techniques and other tools (GitHub, open source licensing, unit testing, etc), we aim to provide a platform that will become a focal point for people using such models. At present, the project contains: A set of general purpose tools for solving such models; a number of tutorials and examples of how to use the tools; and complete archives of several papers whose main contribution is structural modeling results, and whose modeling work has been done using the toolkit.
:::

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
