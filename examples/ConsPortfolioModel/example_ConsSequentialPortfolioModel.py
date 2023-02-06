# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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

# %%
"""
Example implementations of SequentialPortfolioConsumerType
"""
from copy import copy
from time import time

import matplotlib.pyplot as plt
import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    SequentialPortfolioConsumerType,
    init_portfolio,
)
from HARK.utilities import plot_funcs

# %%
# Make and solve an example portfolio choice consumer type
print("Now solving an example portfolio choice problem; this might take a moment...")
MyType = SequentialPortfolioConsumerType()
MyType.cycles = 0
t0 = time()
MyType.solve()
t1 = time()
MyType.cFunc = [MyType.solution[t].cFuncAdj for t in range(MyType.T_cycle)]
MyType.ShareFunc = [MyType.solution[t].ShareFuncAdj for t in range(MyType.T_cycle)]
MyType.SequentialShareFunc = [
    MyType.solution[t].SequentialShareFuncAdj for t in range(MyType.T_cycle)
]
print(
    "Solving an infinite horizon portfolio choice problem took "
    + str(t1 - t0)
    + " seconds."
)

# %%
# Plot the consumption and risky-share functions
print("Consumption function over market resources:")
plot_funcs(MyType.cFunc[0], 0.0, 20.0)

# %%
# Since we are using a discretization of the lognormal distribution,
# the limit is numerically computed and slightly different from
# the analytical limit obtained by Merton and Samuelson for infinite wealth
print("Risky asset share as a function of liquid assets:")
print("Optimal (blue/orange) versus Theoretical Limit (green)")
plt.xlabel("Normalized Liquid Assets")
plt.ylabel("Portfolio Share")
plt.ylim(0.0, 1.0)
plt.xlim(0.0, 200.0)
mgrid = np.linspace(0.0, 300.0, 300)
cgrid = MyType.cFunc[0](mgrid)
shares = MyType.ShareFunc[0](mgrid)
agrid = mgrid - cgrid
plt.plot(agrid, shares)
plot_funcs(
    [
        MyType.SequentialShareFunc[0],
        lambda a: MyType.ShareLimit * np.ones_like(a),
    ],
    0.0,
    200.0,
)

# Note that the orange line lies right on top of the blue line and they are basically
# indistinguishable. This is expected, as deciding saving and risky share simultaneously
# should give the same result as when doing it sequentially.

# %%
print("\n\n\n")
print("For derivation of the numerical limiting portfolio share")
print("as market resources approach infinity, see")
print(
    "http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/AssetPricing/Portfolio-CRRA/"
)

# %%
print("\n\n\n")

# %%
""
# Make another example type, but this one can only update their risky portfolio
# share in any particular period with 15% probability.
init_sticky_share = init_portfolio.copy()
init_sticky_share["AdjustPrb"] = 0.15

# %%
# Make and solve a discrete portfolio choice consumer type
print(
    'Now solving a portfolio choice problem with "sticky" portfolio shares; this might take a moment...'
)
StickyType = SequentialPortfolioConsumerType(**init_sticky_share)
StickyType.cycles = 0
t0 = time()
StickyType.solve()
t1 = time()
StickyType.cFuncAdj = [
    StickyType.solution[t].cFuncAdj for t in range(StickyType.T_cycle)
]
StickyType.cFuncFxd = [
    StickyType.solution[t].cFuncFxd for t in range(StickyType.T_cycle)
]
StickyType.ShareFunc = [
    StickyType.solution[t].ShareFuncAdj for t in range(StickyType.T_cycle)
]
StickyType.SequentialShareFunc = [
    StickyType.solution[t].SequentialShareFuncAdj for t in range(StickyType.T_cycle)
]
print(
    "Solving an infinite horizon sticky portfolio choice problem took "
    + str(t1 - t0)
    + " seconds."
)

# %%
# Plot the consumption and risky-share functions
print(
    "Consumption function over market resources when the agent can adjust his portfolio:"
)
plot_funcs(StickyType.cFuncAdj[0], 0.0, 50.0)

# %%
print(
    "Consumption function over market resources when the agent CAN'T adjust, by current share:"
)
M = np.linspace(0.0, 50.0, 100)
for s in np.linspace(0.0, 1.0, 5):
    C = StickyType.cFuncFxd[0](M, s * np.ones_like(M))
    plt.plot(M, C)
plt.xlim(0.0, 50.0)
plt.ylim(0.0, None)
plt.show()

# %%
print("Risky asset share function over market resources (when possible to adjust):")
print("Optimal (blue/orange) versus Theoretical Limit (green)")
plt.xlabel("Normalized Market Resources")
plt.ylabel("Portfolio Share")
plt.ylim(0.0, 1.0)
mgrid = np.linspace(0.0, 200.0, 1000)
cgrid = MyType.cFunc[0](mgrid)
shares = MyType.ShareFunc[0](mgrid)
agrid = mgrid - cgrid
plt.plot(agrid, shares)
plot_funcs(
    [
        StickyType.SequentialShareFunc[0],
        lambda a: StickyType.ShareLimit * np.ones_like(a),
    ],
    0.0,
    100.0,
)

# %%
