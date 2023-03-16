# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # `DDL`: Using `xarray` in `DiscreteDistribution`
#

# %% [markdown]
# First we import relevant libraries and tools, including the new `DiscreteDistributionLabeled` class.
#

# %%
import numpy as np
from HARK.distribution import (
    MeanOneLogNormal,
    DiscreteDistributionLabeled,
    calc_expectation,
    combine_indep_dstns,
)

# %% [markdown]
# We create a distribution of shocks to income from continuous distributions.
#

# %%
PermShkDstn = MeanOneLogNormal().discretize(200)
TranShkDstn = MeanOneLogNormal().discretize(200)
IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

# %% [markdown]
# Taking the components of `IncShkDstn`, we can now create a `DiscreteDistributionLabeled` object. As a demonstration of additional features, we can add a name attribute to the `DDL` object, as well as named dimensions and coordinates.
#

# %%
x_dist = DiscreteDistributionLabeled.from_unlabeled(
    IncShkDstn,
    name="Distribution of Shocks to Income",
    var_names=["perm_shk", "tran_shk"],
    var_attrs=[
        {
            "name": "Permanent Shocks to Income",
            "limit": {"type": "Lognormal", "mean": -0.5, "variance": 1.0},
        },
        {
            "name": "Transitory Shocks to Income",
            "limit": {"type": "Lognormal", "mean": -0.5, "variance": 1.0},
        },
    ],
)

# %% [markdown]
# The underlying object and metadata is stored in a `xarray.Dataset` object which can be accessed using the `.dataset` attribute.
#

# %%
x_dist.dataset

# %% [markdown]
# ### Using functions with labels to take expresive expectations.
#

# %% [markdown]
# Taking the expectation of a `DDL` object is straightforward using the own `expected()` method.
#

# %%
x_dist.expected()

# %% [markdown]
# As in the `DiscreteDistribution`, we can provide a function and arguments to the `expected()` method.
#

# %%
aGrid = np.linspace(0, 20, 100)
R = 1.03

# %% [markdown]
# The main difference is that the `expected()` method of `DDL` objects can take a function that uses the labels of the `xarray.DataArray` object. This allows for clearer and more expresive mathematical functions and transition equations. Surprisingly, using a function with labels does not add much overhead to the function evaluation.
#

# %%
# %%timeit
x_dist.expected(
    lambda dist, a, R: R * a / dist["perm_shk"] + dist["tran_shk"],
    aGrid,
    R,
)

# %% [markdown]
# Compared to the old method of `calc_expectation` which takes a `DiscreteDistribution` object as input, the new method which takes a `DiscreteDistributionLabeled` object is significantly faster.

# %%
# %%timeit
calc_expectation(IncShkDstn, lambda dist, a, R: R * a / dist[0] + dist[1], aGrid, R)

# %% [markdown]
# We can also use `HARK.distribution.expected`.
#

# %%
from HARK.distribution import expected

# %%
expected(
    func=lambda dist, a, R: R * a / dist["perm_shk"] + dist["tran_shk"],
    dist=x_dist,
    args=(aGrid, R),
)

# %% [markdown]
# Additionally, we can use xarrays as inputs via keyword arguments.

# %%
from xarray import DataArray

aNrm = DataArray(aGrid, name="aNrm", dims=("aNrm"))


# %%
def mNrm_next(dist, R, a=None):
    variables = {}
    variables["mNrm_next"] = R * a / dist["perm_shk"] + dist["tran_shk"]
    return variables


# %%
# %%timeit
expected(
    func=mNrm_next,
    dist=x_dist,
    args=R,
    a=aNrm,
)

# %% [markdown]
# Taking the expectation with xarray inputs and labeled equations is still significantly faster than the old method.
