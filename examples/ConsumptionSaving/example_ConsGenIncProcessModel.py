# %%
from HARK.utilities import plotFuncs
from time import process_time
import matplotlib.pyplot as plt
import numpy as np
from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    IndShockExplicitPermIncConsumerType,
    IndShockConsumerType,
    PersistentShockConsumerType,
    init_explicit_perm_inc
)
def mystr(number):
    return "{:.4f}".format(number)


# %%
do_simulation = True

# %%
# Display information about the pLvlGrid used in these examples
print(
    "The infinite horizon examples presented here use a grid of persistent income levels (pLvlGrid)"
)
print(
    "based on percentiles of the long run distribution of pLvl for the given parameters. These percentiles"
)
print(
    "are specified in the attribute pLvlPctiles. Here, the lowest percentile is "
    + str(init_explicit_perm_inc["pLvlPctiles"][0] * 100)
    + " and the highest"
)
print(
    "percentile is "
    + str(init_explicit_perm_inc["pLvlPctiles"][-1] * 100)
    + ".\n"
)

# %%
# Make and solve an example "explicit permanent income" consumer with idiosyncratic shocks
ExplicitExample = IndShockExplicitPermIncConsumerType()
t_start = process_time()
ExplicitExample.solve()
t_end = process_time()
print(
    "Solving an explicit permanent income consumer took "
    + mystr(t_end - t_start)
    + " seconds."
)

# %%
# Plot the consumption function at various permanent income levels
print("Consumption function by pLvl for explicit permanent income consumer:")
pLvlGrid = ExplicitExample.pLvlGrid[0]
mLvlGrid = np.linspace(0, 20, 300)
for p in pLvlGrid:
    M_temp = mLvlGrid + ExplicitExample.solution[0].mLvlMin(p)
    C = ExplicitExample.solution[0].cFunc(M_temp, p * np.ones_like(M_temp))
    plt.plot(M_temp, C)
plt.xlim(0.0, 20.0)
plt.ylim(0.0, None)
plt.xlabel("Market resource level mLvl")
plt.ylabel("Consumption level cLvl")
plt.show()

# %%
# Now solve the *exact same* problem, but with the permanent income normalization
NormalizedExample = IndShockConsumerType(**init_explicit_perm_inc)
t_start = process_time()
NormalizedExample.solve()
t_end = process_time()
print(
    "Solving the equivalent problem with permanent income normalized out took "
    + mystr(t_end - t_start)
    + " seconds."
)

# %%
# Show that the normalized consumption function for the "explicit permanent income" consumer
# is almost identical for every permanent income level (and the same as the normalized problem's
# cFunc), but is less accurate due to extrapolation outside the bounds of pLvlGrid.
print("Normalized consumption function by pLvl for explicit permanent income consumer:")
pLvlGrid = ExplicitExample.pLvlGrid[0]
mNrmGrid = np.linspace(0, 20, 300)
for p in pLvlGrid:
    M_temp = mNrmGrid * p + ExplicitExample.solution[0].mLvlMin(p)
    C = ExplicitExample.solution[0].cFunc(M_temp, p * np.ones_like(M_temp))
    plt.plot(M_temp / p, C / p)
plt.xlim(0.0, 20.0)
plt.ylim(0.0, None)
plt.xlabel("Normalized market resources mNrm")
plt.ylabel("Normalized consumption cNrm")
plt.show()
print(
    "Consumption function for normalized problem (without explicit permanent income):"
)
mNrmMin = NormalizedExample.solution[0].mNrmMin
plotFuncs(NormalizedExample.solution[0].cFunc, mNrmMin, mNrmMin + 20)

# %% [markdown]
# The "explicit permanent income" solution deviates from the solution to the normalized problem because
# of errors from extrapolating beyond the bounds of the pLvlGrid.
# The error is largest for pLvl values
# near the upper and lower bounds, and propagates toward the center of the distribution.

# %%
# Plot the value function at various permanent income levels
if ExplicitExample.vFuncBool:
    pGrid = np.linspace(0.1, 3.0, 24)
    M = np.linspace(0.001, 5, 300)
    for p in pGrid:
        M_temp = M + ExplicitExample.solution[0].mLvlMin(p)
        C = ExplicitExample.solution[0].vFunc(M_temp, p * np.ones_like(M_temp))
        plt.plot(M_temp, C)
    plt.ylim([-200, 0])
    plt.xlabel("Market resource level mLvl")
    plt.ylabel("Value v")
    plt.show()

# %%
# Simulate some data
if do_simulation:
    ExplicitExample.T_sim = 500
    ExplicitExample.track_vars = ["mLvlNow", "cLvlNow", "pLvlNow"]
    ExplicitExample.makeShockHistory()  # This is optional
    ExplicitExample.initializeSim()
    ExplicitExample.simulate()
    plt.plot(np.mean(ExplicitExample.mLvlNow_hist, axis=1))
    plt.xlabel("Simulated time period")
    plt.ylabel("Average market resources mLvl")
    plt.show()

# %%
# Make and solve an example "persistent idisyncratic shocks" consumer
PersistentExample = PersistentShockConsumerType(**init_persistent_shocks)
t_start = process_time()
PersistentExample.solve()
t_end = process_time()
print(
    "Solving a persistent income shocks consumer took "
    + mystr(t_end - t_start)
    + " seconds."
)

# %%
# Plot the consumption function at various levels of persistent income pLvl
print(
    "Consumption function by persistent income level pLvl for a consumer with AR1 coefficient of "
    + str(PersistentExample.PrstIncCorr)
    + ":"
)
pLvlGrid = PersistentExample.pLvlGrid[0]
mLvlGrid = np.linspace(0, 20, 300)
for p in pLvlGrid:
    M_temp = mLvlGrid + PersistentExample.solution[0].mLvlMin(p)
    C = PersistentExample.solution[0].cFunc(M_temp, p * np.ones_like(M_temp))
    plt.plot(M_temp, C)
plt.xlim(0.0, 20.0)
plt.ylim(0.0, None)
plt.xlabel("Market resource level mLvl")
plt.ylabel("Consumption level cLvl")
plt.show()

# %%
# Plot the value function at various persistent income levels
if PersistentExample.vFuncBool:
    pGrid = PersistentExample.pLvlGrid[0]
    M = np.linspace(0.001, 5, 300)
    for p in pGrid:
        M_temp = M + PersistentExample.solution[0].mLvlMin(p)
        C = PersistentExample.solution[0].vFunc(M_temp, p * np.ones_like(M_temp))
        plt.plot(M_temp, C)
    plt.ylim([-200, 0])
    plt.xlabel("Market resource level mLvl")
    plt.ylabel("Value v")
    plt.show()

# %%
# Simulate some data
if do_simulation:
    PersistentExample.T_sim = 500
    PersistentExample.track_vars = ["mLvlNow", "cLvlNow", "pLvlNow"]
    PersistentExample.initializeSim()
    PersistentExample.simulate()
    plt.plot(np.mean(PersistentExample.mLvlNow_hist, axis=1))
    plt.xlabel("Simulated time period")
    plt.ylabel("Average market resources mLvl")
    plt.show()
