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
# # Demonstrations and Timings of HARK.expected
#

# %% [markdown]
# First, we import the relevant libraries.
#

# %%
from time import time

import numpy as np
from HARK.distribution import (
    MeanOneLogNormal,
    Normal,
    calc_expectation,
    combine_indep_dstns,
)

# %% [markdown]
# Next, we define discrete distributions as approximations of continuous distributions.
#

# %%
dd_0_1_20 = Normal().discretize(20)
dd_1_1_40 = Normal(mu=1).discretize(40)
dd_10_10_100 = Normal(mu=10, sigma=10).discretize(100)

# %% [markdown]
# ### The **new** `DiscreteDistribution.expected()` method
#

# %% [markdown]
# There are two ways to get the expectation of a distribution. The first is to use the **new** `expected()` method of the distribution shown below.
#

# %%
# %%timeit
ce1 = dd_0_1_20.expected()
ce2 = dd_1_1_40.expected()
ce3 = dd_10_10_100.expected()

# %% [markdown]
# The second is to use `HARK.distribution.calc_expectation()`. Comparing the timings, the first method is significantly faster.
#

# %%
# %%timeit
ce1 = calc_expectation(dd_0_1_20)
ce2 = calc_expectation(dd_1_1_40)
ce3 = calc_expectation(dd_10_10_100)

# %% [markdown]
# ### The Expected Value of a function of a random variable
#

# %% [markdown]
# Both of these methods allow us to calculate the expected value of a function of the distribution. Using the first method, which is the distribution's own method, we only need to provide the function.
#

# %%
# %%timeit
ce4 = dd_0_1_20.expected(lambda x: 2**x)
ce5 = dd_1_1_40.expected(lambda x: 2 * x)

# %% [markdown]
# Using `HARK.distribution.calc_expectation()`, we first provide the distribution and then the function.
#

# %%
# %%timeit
ce4 = calc_expectation(dd_0_1_20, lambda x: 2**x)
ce5 = calc_expectation(dd_1_1_40, lambda x: 2 * x)

# %% [markdown]
# #### The expected value of a function with additional arguments
#

# %% [markdown]
# For both methods, we can also provide a number of arguments to the function `args`, which are passed to the function and gets called as `func(dstn,*args)`.
#

# %%
# %%timeit
ce6 = dd_10_10_100.expected(lambda x, y: 2 * x + y, 20)
ce7 = dd_0_1_20.expected(lambda x, y: x + y, np.hstack([0, 1, 2, 3, 4, 5]))

# %%
# %%timeit
ce6 = calc_expectation(dd_10_10_100, lambda x, y: 2 * x + y, 20)
ce7 = calc_expectation(dd_0_1_20, lambda x, y: x + y, np.hstack([0, 1, 2, 3, 4, 5]))

# %% [markdown]
# ### The expected value of a function in `HARK`
#

# %% [markdown]
# For a more practical demonstration of these methods as they would be used in `HARK`, we can create a distcrete distribution of shocks to income `IncShkDstn`. Given an array of liquid assets `aGrid` and an interest rate `R`, we can calculate the expected value of next period's cash on hand as the function `m_next = R * aGrid / perm_shk + tran_shk`. Below we see how this is done. Notice that the arguments to the function can be multidimensional.
#

# %%
PermShkDstn = MeanOneLogNormal().discretize(200)
TranShkDstn = MeanOneLogNormal().discretize(200)
IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)
aGrid = np.linspace(0, 20, 100)  # aNrm grid
R = 1.05  # interest rate


def m_next(x, aGrid, R):
    return R * aGrid / x[0] + x[1]


# %%
# %%timeit
ce8 = IncShkDstn.expected(m_next, aGrid, R)
ce9 = IncShkDstn.expected(m_next, aGrid.reshape((10, 10)), R)

# %%
# %%timeit
ce8 = calc_expectation(IncShkDstn, m_next, aGrid, R)
ce9 = calc_expectation(IncShkDstn, m_next, aGrid.reshape((10, 10)), R)

# %% [markdown]
# ### Time Comparison of the two methods
#

# %% [markdown]
# As a final comparision of these two methods, we can see how the time difference is affected by the number of points in the distribution.
#

# %%
size = np.arange(1, 11) * 100

t_self = []
t_dist = []

for n in size:
    PermShkDstn = MeanOneLogNormal().discretize(n)
    TranShkDstn = MeanOneLogNormal().discretize(n)
    IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

    m_next = lambda X, a, r: r * a / X[0] + X[1]
    a_grid = np.linspace(0, 20, 100).reshape((10, 10))
    R = 1.05

    start_self = time()
    ce_self = IncShkDstn.expected(m_next, a_grid, R)
    time_self = time() - start_self

    start_dist = time()
    ce_dist = calc_expectation(IncShkDstn, m_next, a_grid, R)
    time_dist = time() - start_dist

    t_self.append(time_self)
    t_dist.append(time_dist)

# %%
import matplotlib.pyplot as plt

plt.plot(size, t_self, label="dist.ev(f)")
plt.plot(size, t_dist, label="ce(dist, f)")
plt.title("Time to calculate expectation of a function of shocks to income.")
plt.ylabel("time (s)")
plt.xlabel("size of grid: $x^2$")
plt.legend()
plt.show()

# %% [markdown]
# ### Aliases for the new `expected()` method
#

# %% [markdown]
# There is a top-level alias for the new `expected()` method to make it clearer as a mathematical expression. The way to access it is as follows:
#
# `expected(func, dstn, *args)`
#

# %%
from HARK.distribution import expected

# %%
expected(func=m_next, dist=IncShkDstn, args=(aGrid, R))

# %%
expected(func=lambda x: 1 / x[0] + x[1], dist=IncShkDstn)

# %%
