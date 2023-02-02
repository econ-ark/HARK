# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
# ---

# %%
from HARK.ConsumptionSaving.ConsIndShockModel import (
    PerfForesightConsumerType,
    IndShockConsumerType,
    KinkedRconsumerType,
    init_lifecycle,
    init_cyclical,
)
from HARK.utilities import plot_funcs_der, plot_funcs
from time import time

# %%
mystr = lambda number: f"{number:.4f}"


# %%
do_simulation = True

# %%
# Make and solve an example perfect foresight consumer
PFexample = PerfForesightConsumerType()
# Make this type have an infinite horizon
PFexample.cycles = 0

# %%
start_time = time()
PFexample.solve()
end_time = time()
print(
    "Solving a perfect foresight consumer took "
    + mystr(end_time - start_time)
    + " seconds."
)
PFexample.unpack("cFunc")

# %%
# Plot the perfect foresight consumption function
print("Perfect foresight consumption function:")
mMin = PFexample.solution[0].mNrmMin
plot_funcs(PFexample.cFunc[0], mMin, mMin + 10)

# %%
if do_simulation:
    PFexample.T_sim = 120  # Set number of simulation periods
    PFexample.track_vars = ["mNrm"]
    PFexample.initialize_sim()
    PFexample.simulate()

# %%
# Make and solve an example consumer with idiosyncratic income shocks
IndShockExample = IndShockConsumerType()
IndShockExample.cycles = 0  # Make this type have an infinite horizon

# %%
start_time = time()
IndShockExample.solve()
end_time = time()
print(
    "Solving a consumer with idiosyncratic shocks took "
    + mystr(end_time - start_time)
    + " seconds."
)
IndShockExample.unpack("cFunc")

# %%
# Plot the consumption function and MPC for the infinite horizon consumer
print("Concave consumption function:")
plot_funcs(IndShockExample.cFunc[0], IndShockExample.solution[0].mNrmMin, 5)
print("Marginal consumption function:")
plot_funcs_der(IndShockExample.cFunc[0], IndShockExample.solution[0].mNrmMin, 5)

# %%
# Compare the consumption functions for the perfect foresight and idiosyncratic
# shock types.  Risky income cFunc asymptotically approaches perfect foresight cFunc.
print("Consumption functions for perfect foresight vs idiosyncratic shocks:")
plot_funcs(
    [PFexample.cFunc[0], IndShockExample.cFunc[0]],
    IndShockExample.solution[0].mNrmMin,
    100,
)

# %%
# Compare the value functions for the two types
if IndShockExample.vFuncBool:
    print("Value functions for perfect foresight vs idiosyncratic shocks:")
    plot_funcs(
        [PFexample.solution[0].vFunc, IndShockExample.solution[0].vFunc],
        IndShockExample.solution[0].mNrmMin + 0.5,
        10,
    )

# %%
# Simulate some data; results stored in mNrm_hist, cNrm_hist, and pLvl_hist
if do_simulation:
    IndShockExample.T_sim = 120
    IndShockExample.track_vars = ["mNrm", "cNrm", "pLvl"]
    IndShockExample.make_shock_history()  # This is optional, simulation will draw shocks on the fly if it isn't run.
    IndShockExample.initialize_sim()
    IndShockExample.simulate()

# %%
# Make and solve an idiosyncratic shocks consumer with a finite lifecycle
LifecycleExample = IndShockConsumerType(**init_lifecycle)
LifecycleExample.cycles = (
    1  # Make this consumer live a sequence of periods exactly once
)

# %%
start_time = time()
LifecycleExample.solve()
end_time = time()
print("Solving a lifecycle consumer took " + mystr(end_time - start_time) + " seconds.")
LifecycleExample.unpack("cFunc")

# %%
# Plot the consumption functions during working life
print("Consumption functions while working:")
mMin = min(
    [LifecycleExample.solution[t].mNrmMin for t in range(LifecycleExample.T_cycle)]
)
plot_funcs(LifecycleExample.cFunc[: LifecycleExample.T_retire], mMin, 5)

# %%
# Plot the consumption functions during retirement
print("Consumption functions while retired:")
plot_funcs(LifecycleExample.cFunc[LifecycleExample.T_retire :], 0, 5)

# %%
# Simulate some data; results stored in mNrm_hist, cNrm_hist, pLvl_hist, and t_age_hist
if do_simulation:
    LifecycleExample.T_sim = 120
    LifecycleExample.track_vars = ["mNrm", "cNrm", "pLvl", "t_age"]
    LifecycleExample.initialize_sim()
    LifecycleExample.simulate()

# %%
# Make and solve a "cyclical" consumer type who lives the same four quarters repeatedly.
# The consumer has income that greatly fluctuates throughout the year.
CyclicalExample = IndShockConsumerType(**init_cyclical)
CyclicalExample.cycles = 0

# %%
start_time = time()
CyclicalExample.solve()
end_time = time()
print("Solving a cyclical consumer took " + mystr(end_time - start_time) + " seconds.")
CyclicalExample.unpack("cFunc")

# %%
# Plot the consumption functions for the cyclical consumer type
print("Quarterly consumption functions:")
mMin = min([X.mNrmMin for X in CyclicalExample.solution])
plot_funcs(CyclicalExample.cFunc, mMin, 5)

# %%
# Simulate some data; results stored in cHist, mHist, bHist, aHist, MPChist, and pHist
if do_simulation:
    CyclicalExample.T_sim = 480
    CyclicalExample.track_vars = ["mNrm", "cNrm", "pLvl", "t_cycle"]
    CyclicalExample.initialize_sim()
    CyclicalExample.simulate()

# %%
# Make and solve an agent with a kinky interest rate
KinkyExample = KinkedRconsumerType()
KinkyExample.cycles = 0  # Make the Example infinite horizon

# %%
start_time = time()
KinkyExample.solve()
end_time = time()
print("Solving a kinky consumer took " + mystr(end_time - start_time) + " seconds.")
KinkyExample.unpack("cFunc")
print("Kinky consumption function:")
plot_funcs(KinkyExample.cFunc[0], KinkyExample.solution[0].mNrmMin, 5)

# %%
if do_simulation:
    KinkyExample.T_sim = 120
    KinkyExample.track_vars = ["mNrm", "cNrm", "pLvl"]
    KinkyExample.initialize_sim()
    KinkyExample.simulate()
