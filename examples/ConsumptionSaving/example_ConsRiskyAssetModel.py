# %%
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.utilities import plot_funcs_der, plot_funcs
from time import time

# %%
mystr = lambda number: "{:.4f}".format(number)

# %%
do_simulation = True


# %%
# Make and solve an example consumer with idiosyncratic income shocks
RiskyReturnExample = RiskyAssetConsumerType()
RiskyReturnExample.cycles = 0  # Make this type have an infinite horizon

# %%
start_time = time()
RiskyReturnExample.solve()
end_time = time()
print(
    "Solving a consumer with idiosyncratic shocks took "
    + mystr(end_time - start_time)
    + " seconds."
)
RiskyReturnExample.unpack("cFunc")

# %%
# Plot the consumption function and MPC for the infinite horizon consumer
print("Concave consumption function:")
plot_funcs(RiskyReturnExample.cFunc[0], RiskyReturnExample.solution[0].mNrmMin, 5)
print("Marginal consumption function:")
plot_funcs_der(RiskyReturnExample.cFunc[0], RiskyReturnExample.solution[0].mNrmMin, 5)
