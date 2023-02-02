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
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example Implementations of `HARK.ConsumptionSaving.ConsRiskyAssetModel`

# %%
from time import time

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    init_portfolio,
    PortfolioConsumerType,
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    RiskyAssetConsumerType,
    FixedPortfolioShareRiskyAssetConsumerType,
)
from HARK.utilities import plot_funcs_der, plot_funcs


# %%
mystr = lambda number: f"{number:.4f}"


# %% [markdown]
# ## Idiosyncratic Income Shocks Consumer Type

# %%
# Make and solve an example consumer with idiosyncratic income shocks
# Sse init_portfolio parameters to compare to results of PortfolioConsumerType
IndShockExample = IndShockConsumerType(**init_portfolio)
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
plot_funcs(IndShockExample.cFunc[0], 0.0, 5.0)
print("Marginal consumption function:")
plot_funcs_der(IndShockExample.cFunc[0], 0.0, 5.0)

# %% [markdown]
# ## Risky Return Consumer Type

# %%
# Make and solve an example consumer with risky returns to savings
# Use init_portfolio parameters to compare to results of PortfolioConsumerType
RiskyReturnExample = RiskyAssetConsumerType(**init_portfolio)
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
plot_funcs(RiskyReturnExample.cFunc[0], 0.0, 5.0)
print("Marginal consumption function:")
plot_funcs_der(RiskyReturnExample.cFunc[0], 0.0, 5.0)

# %% [markdown]
# ## Compare Idiosyncratic Income Shocks with Risky Return

# %%
# Compare the consumption functions for the various agents in this notebook.
print("Consumption functions for idiosyncratic shocks vs risky returns:")
plot_funcs(
    [
        IndShockExample.cFunc[0],  # blue
        RiskyReturnExample.cFunc[0],  # orange
    ],
    0.0,
    20.0,
)


# %% [markdown]
# ## Risky Return Consumer Type with Portfolio Choice

# %%
# Make and solve an example risky consumer with a portfolio choice
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
plot_funcs(PortfolioChoiceExample.cFunc[0], 0.0, 5.0)
print("Marginal consumption function:")
plot_funcs_der(PortfolioChoiceExample.cFunc[0], 0.0, 5.0)

# %% [markdown]
# ## Compare Income Shocks, Risky Return, and RR w/ Portfolio Choice

# %%
# Compare the consumption functions for the various agents in this notebook.
print(
    "Consumption functions for idiosyncratic shocks vs risky returns vs portfolio choice:"
)
plot_funcs(
    [
        IndShockExample.cFunc[0],  # blue
        RiskyReturnExample.cFunc[0],  # orange
        PortfolioChoiceExample.cFunc[0],  # green
    ],
    0.0,
    20.0,
)


# %% [markdown]
# ## Portfolio Consumer Type

# %%
# Make and solve an example portfolio choice consumer
PortfolioTypeExample = PortfolioConsumerType()
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
plot_funcs(PortfolioTypeExample.cFuncAdj[0], 0.0, 5.0)
print("Marginal consumption function:")
plot_funcs_der(PortfolioTypeExample.cFuncAdj[0], 0.0, 5.0)


# %% [markdown]
# ## Compare RR w/ Portfolio Choice with Portfolio Choice Type

# %%
# Compare the consumption functions for the various portfolio choice types.
print(
    "Consumption functions for portfolio choice type vs risky asset with portfolio choice:"
)
plot_funcs(
    [
        PortfolioTypeExample.cFuncAdj[0],  # blue
        PortfolioChoiceExample.cFunc[0],  # orange
    ],
    0.0,
    20.0,
)


# %%
# Compare the share functions for the various portfolio choice types.
print("Share functions for portfolio choice type vs risky asset with portfolio choice:")
plot_funcs(
    [
        PortfolioTypeExample.ShareFuncAdj[0],  # blue
        PortfolioChoiceExample.ShareFunc[0],  # orange
    ],
    0,
    200,
)


# %% [markdown]
# ## Risky Return Given Fixed Portfolio Share

# %%
FixedShareExample = FixedPortfolioShareRiskyAssetConsumerType(**init_portfolio)
FixedShareExample.cycles = 0


# %%
start_time = time()
FixedShareExample.solve()
end_time = time()
print(
    "Solving a consumer with fixed portfolio share took "
    + mystr(end_time - start_time)
    + " seconds."
)
FixedShareExample.unpack("cFunc")


# %%
# Plot the consumption function and MPC for the infinite horizon consumer
print("Concave consumption function:")
plot_funcs(FixedShareExample.cFunc[0], 0.0, 5.0)
print("Marginal consumption function:")
plot_funcs_der(FixedShareExample.cFunc[0], 0.0, 5.0)


# %% [markdown]
# ## Compare Idiosyncratic Shock Type with Fixed Share at 0.0 Type

# %%
# Compare the consumption functions for the various idiosyncratic shocks
print("Consumption functions for idiosyncratic shocks vs fixed share at 0.0:")
plot_funcs(
    [
        IndShockExample.cFunc[0],  # blue
        FixedShareExample.cFunc[0],  # orange
    ],
    0.0,
    20.0,
)


# %% [markdown]
# ## Fixed Share at 1.0 Type

# %%
init_portfolio["RiskyShareFixed"] = [1.0]
RiskyFixedExample = FixedPortfolioShareRiskyAssetConsumerType(**init_portfolio)
RiskyFixedExample.cycles = 0


# %%
start_time = time()
RiskyFixedExample.solve()
end_time = time()
print(
    "Solving a consumer with share fixed at 1.0 took "
    + mystr(end_time - start_time)
    + " seconds."
)
RiskyFixedExample.unpack("cFunc")


# %%
# Plot the consumption function and MPC for the portfolio choice consumer
print("Concave consumption function:")
plot_funcs(RiskyFixedExample.cFunc[0], 0.0, 5.0)
print("Marginal consumption function:")
plot_funcs_der(RiskyFixedExample.cFunc[0], 0.0, 5.0)


# %% [markdown]
# ## Compare Fixed Share at 1.0 Type with Risky Return Type

# %%
# Compare the consumption functions for the various risky shocks
print("Consumption functions for risky asset vs fixed share at 1.0:")
plot_funcs(
    [
        RiskyReturnExample.cFunc[0],  # blue
        RiskyFixedExample.cFunc[0],  # orange
    ],
    0.0,
    200.0,
)


# %%
