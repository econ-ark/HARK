# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # `DDXRA`: Using `xarray` in `DiscreteDistribution`
#

# %% [markdown]
# First we import relevant libraries and tools, including the new `DiscreteDistributionXRA` class.
#

# %%
import numpy as np
from HARK.distribution import (
    MeanOneLogNormal,
    DiscreteDistributionXRA,
    calc_expectation,
    combine_indep_dstns,
)

# %% [markdown]
# We create a distribution of shocks to income from continuous distributions.
#

# %%
PermShkDstn = MeanOneLogNormal().approx(200)
TranShkDstn = MeanOneLogNormal().approx(200)
IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

# %% [markdown]
# Taking the components of `IncShkDstn`, we can now create a `DiscreteDistributionXRA` object. As a demonstration of additional features, we can add a name attribute to the `DDXRA` object, as well as named dimensions and coordinates.
#

# %%
x_dist = DiscreteDistributionXRA(
    IncShkDstn.pmf,
    IncShkDstn.X,
    name="Distribution of Shocks to Income",
    dims=("rv", "x"),
    coords={"rv": ["perm_shk", "tran_shk"]},
)

# %% [markdown]
# As a side note, we can also use set the boolean option `xarray = True` in `combine_indep_dstns` with the same attributes to create an `DDXRA` object in place.
#

# %%
x_dist = combine_indep_dstns(
    PermShkDstn,
    TranShkDstn,
    xarray=True,
    name="Distribution of Shocks to Income",
    dims=("rv", "x"),
    coords={"rv": ["perm_shk", "tran_shk"]},
)

# %% [markdown]
# The underlying object and metadata is stored in a `xarray.DataArray` object which can be accessed using the `.xarray` attribute.
#

# %%
x_dist.xarray

# %% [markdown]
# ### Taking the Expected Value of `DDXRA` objects.
#

# %% [markdown]
# Taking the expectation of a `DDXRA` object is straightforward using the own `expected_value()` method.
#

# %%
x_dist.expected_value()

# %% [markdown]
# As in the `DiscreteDistribution`, we can provide a function and arguments to the `expected_value()` method.
#

# %%
aGrid = np.linspace(0, 20, 100)
R = 1.03

# %%
# %%timeit
x_dist.expected_value(lambda x, a, R: R * a / x[0] + x[1], aGrid, R)

# %% [markdown]
# Compared to the old method of `calc_expectation` which takes a `DiscreteDistribution` object as input, the new method which takes a `DiscreteDistributionXRA` object remains significantly faster.
#
# """
#

# %%
# %%timeit
calc_expectation(IncShkDstn, lambda x, a, R: R * a / x[0] + x[1], aGrid, R)

# %% [markdown]
# ### Using functions with labels to take expresive expectations.
#

# %% [markdown]
# The main difference is that the `expected_value()` method of `DDXRA` objects can take a function that uses the labels of the `xarray.DataArray` object. This allows for clearer and more expresive mathematical functions and transition equations. Surprisingly, using a function with labels does not add much overhead to the function evaluation.
#

# %%
# %%timeit
x_dist.expected_value(
    lambda x, a, R: R * a / x["perm_shk"] + x["tran_shk"], aGrid, R, labels=True
)

# %% [markdown]
# We can also use `HARK.distribution.ExpectedValue`.

# %%
from HARK.distribution import ExpectedValue

# %%
ExpectedValue(
    lambda x, a, R: R * a / x["perm_shk"] + x["tran_shk"],
    dist=x_dist,
    args=(aGrid, R),
    labels=True,
)
