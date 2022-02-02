# %%
from time import time

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.utilities import plot_funcs_der, plot_funcs


# %%
mystr = lambda number: "{:.4f}".format(number)


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
# Make and solve an example consumer with risky returns to savings
init_risky = init_portfolio.copy()
init_risky["PortfolioBool"] = False
RiskyReturnExample = RiskyAssetConsumerType(**init_risky)
RiskyReturnExample.cycles = 0  # Make this type have an infinite horizon


# %%
start_time = time()
RiskyReturnExample.solve()
end_time = time()
print(
    "Solving a consumer with risky returns took "
    + mystr(end_time - start_time)
    + " seconds."
)
RiskyReturnExample.unpack("cFunc")


# %%
# Plot the consumption function and MPC for the risky asset consumer
print("Concave consumption function:")
plot_funcs(RiskyReturnExample.cFunc[0], RiskyReturnExample.solution[0].mNrmMin, 5)
print("Marginal consumption function:")
plot_funcs_der(RiskyReturnExample.cFunc[0], RiskyReturnExample.solution[0].mNrmMin, 5)



# %%
# Make and solve an example consumer with a portfolio choice
init_portfolio["PortfolioBool"] = True
PortfolioChoiceExample = RiskyAssetConsumerType(**init_portfolio)
PortfolioChoiceExample.cycles = 0  # Make this type have an infinite horizon


# %%
start_time = time()
PortfolioChoiceExample.solve()
end_time = time()
print(
    "Solving a consumer with risky returns and portfolio choice took "
    + mystr(end_time - start_time)
    + " seconds."
)
PortfolioChoiceExample.unpack("cFunc")
PortfolioChoiceExample.unpack("ShareFunc")


# %%
# Plot the consumption function and MPC for the portfolio choice consumer
print("Concave consumption function:")
plot_funcs(
    PortfolioChoiceExample.cFunc[0], PortfolioChoiceExample.solution[0].mNrmMin, 5
)
print("Marginal consumption function:")
plot_funcs_der(
    PortfolioChoiceExample.cFunc[0], PortfolioChoiceExample.solution[0].mNrmMin, 5
)



# %%
# Compare the consumption functions for the various agents in this notebook.
print(
    "Consumption functions for idiosyncratic shocks vs risky returns vs portfolio choice:"
)
plot_funcs(
    [
        IndShockExample.cFunc[0],
        RiskyReturnExample.cFunc[0],
        PortfolioChoiceExample.cFunc[0],
    ],
    IndShockExample.solution[0].mNrmMin,
    5,
)


# %%
# Make and solve an example consumer with a portfolio choice
init_portfolio["PortfolioBool"] = True
PortfolioTypeExample = PortfolioConsumerType(**init_portfolio)
PortfolioTypeExample.cycles = 0  # Make this type have an infinite horizon


# %%
start_time = time()
PortfolioTypeExample.solve()
end_time = time()
print(
    "Solving a consumer with portfolio choice took "
    + mystr(end_time - start_time)
    + " seconds."
)
PortfolioTypeExample.unpack("cFuncAdj")
PortfolioTypeExample.unpack("ShareFuncAdj")


# %%
# Plot the consumption function and MPC for the portfolio choice consumer
print("Concave consumption function:")
plot_funcs(PortfolioTypeExample.cFuncAdj[0], 0, 5)
print("Marginal consumption function:")
plot_funcs_der(PortfolioTypeExample.cFuncAdj[0], 0, 5)


# %%
# Compare the consumption functions for the various portfolio choice types.
print(
    "Consumption functions for portfolio choice type vs risky asset with portfolio choice:"
)
plot_funcs([PortfolioTypeExample.cFuncAdj[0], PortfolioChoiceExample.cFunc[0]], 0, 200)


# %%
# Compare the share functions for the various portfolio choice types.
print("Share functions for portfolio choice type vs risky asset with portfolio choice:")
plot_funcs(
    [PortfolioTypeExample.ShareFuncAdj[0], PortfolioChoiceExample.ShareFunc[0]], 0, 200
)


# %%
