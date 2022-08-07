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
# ### Taking the Expected Value of `DDXRA` objects.
#

# %% [markdown]
# Taking the expectation of a `DDXRA` object is straightforward using the own `expected()` method.
#

# %%
x_dist.expected()

# %% [markdown]
# As in the `DiscreteDistribution`, we can provide a function and arguments to the `expected()` method.
#

# %%
aGrid = np.linspace(0, 20, 100)
R = 1.03

# %%
# %%timeit
x_dist.expected(lambda dist, a, R: R * a / dist[0] + dist[1], aGrid, R)

# %% [markdown]
# Compared to the old method of `calc_expectation` which takes a `DiscreteDistribution` object as input, the new method which takes a `DiscreteDistributionXRA` object remains significantly faster.
#
# """
#

# %%
# %%timeit
calc_expectation(IncShkDstn, lambda dist, a, R: R * a / dist[0] + dist[1], aGrid, R)

# %% [markdown]
# ### Using functions with labels to take expresive expectations.
#

# %% [markdown]
# The main difference is that the `expected()` method of `DDXRA` objects can take a function that uses the labels of the `xarray.DataArray` object. This allows for clearer and more expresive mathematical functions and transition equations. Surprisingly, using a function with labels does not add much overhead to the function evaluation.
#

# %%
# %%timeit
x_dist.expected(
    lambda dist, a, R: R * a / dist["perm_shk"] + dist["tran_shk"],
    aGrid,
    R,
    labels=True,
)

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
    labels=True,
)
