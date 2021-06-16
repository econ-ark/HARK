# %%
"""
Example implementations of HARK.ConsumptionSaving.ConsPortfolioModel
"""
from HARK.ConsumptionSaving.ConsRiskyContribModel import (
    RiskyContribConsumerType,
    init_risky_contrib,
)
from time import time
import numpy as np

# %% Define a plotting function


def plot_slices_3d(
    functions, bot_x, top_x, y_slices, N=300, y_name=None, titles=None, ax_labs=None
):

    import matplotlib.pyplot as plt

    if type(functions) == list:
        function_list = functions
    else:
        function_list = [functions]

    nfunc = len(function_list)

    # Initialize figure and axes
    fig = plt.figure(figsize=plt.figaspect(1.0 / nfunc))

    # Create x grid
    x = np.linspace(bot_x, top_x, N, endpoint=True)

    for k in range(nfunc):
        ax = fig.add_subplot(1, nfunc, k + 1)

        for y in y_slices:

            if y_name is None:
                lab = ""
            else:
                lab = y_name + "=" + str(y)

            z = function_list[k](x, np.ones_like(x) * y)
            ax.plot(x, z, label=lab)

        if ax_labs is not None:
            ax.set_xlabel(ax_labs[0])
            ax.set_ylabel(ax_labs[1])

        # ax.imshow(Z, extent=[bottom[0],top[0],bottom[1],top[1]], origin='lower')
        # ax.colorbar();
        if titles is not None:
            ax.set_title(titles[k])

        ax.set_xlim([bot_x, top_x])

        if y_name is not None:
            ax.legend()

    plt.show()


def plot_slices_4d(
    functions,
    bot_x,
    top_x,
    y_slices,
    w_slices,
    N=300,
    slice_names=None,
    titles=None,
    ax_labs=None,
):

    import matplotlib.pyplot as plt

    if type(functions) == list:
        function_list = functions
    else:
        function_list = [functions]

    nfunc = len(function_list)
    nws = len(w_slices)

    # Initialize figure and axes
    fig = plt.figure(figsize=plt.figaspect(1.0 / nfunc))

    # Create x grid
    x = np.linspace(bot_x, top_x, N, endpoint=True)

    for j in range(nws):
        w = w_slices[j]

        for k in range(nfunc):
            ax = fig.add_subplot(nws, nfunc, j * nfunc + k + 1)

            for y in y_slices:

                if slice_names is None:
                    lab = ""
                else:
                    lab = (
                        slice_names[0]
                        + "="
                        + str(y)
                        + ","
                        + slice_names[1]
                        + "="
                        + str(w)
                    )

                z = function_list[k](x, np.ones_like(x) * y, np.ones_like(x) * w)
                ax.plot(x, z, label=lab)

            if ax_labs is not None:
                ax.set_xlabel(ax_labs[0])
                ax.set_ylabel(ax_labs[1])

            # ax.imshow(Z, extent=[bottom[0],top[0],bottom[1],top[1]], origin='lower')
            # ax.colorbar();
            if titles is not None:
                ax.set_title(titles[k])

            ax.set_xlim([bot_x, top_x])

            if slice_names is not None:
                ax.legend()

    plt.show()


# %%
# Solve an infinite horizon version

# Get initial parameters
par_infinite = init_risky_contrib.copy()
# And make the problem infinite horizon
par_infinite["cycles"] = 0
# and sticky
par_infinite["AdjustPrb"] = 1.0
# and with a withdrawal tax
par_infinite["tau"] = 0.1

par_infinite["DiscreteShareBool"] = False
par_infinite["vFuncBool"] = False

# Create agent and solve it.
inf_agent = RiskyContribConsumerType(tolerance=1e-3, **par_infinite)
print("Now solving infinite horizon version")
t0 = time()
inf_agent.solve(verbose=True)
t1 = time()
print("Converged!")
print("Solving took " + str(t1 - t0) + " seconds.")

# Plot policy functions
periods = [0]
n_slices = [0, 2, 6]
mMax = 20

dfracFunc = [inf_agent.solution[t].stage_sols["Reb"].dfracFunc_Adj for t in periods]
ShareFunc = [inf_agent.solution[t].stage_sols["Sha"].ShareFunc_Adj for t in periods]
cFuncFxd = [inf_agent.solution[t].stage_sols["Cns"].cFunc for t in periods]

# Rebalancing
plot_slices_3d(
    dfracFunc,
    0,
    mMax,
    y_slices=n_slices,
    y_name="n",
    titles=["t = " + str(t) for t in periods],
    ax_labs=["m", "d"],
)
# Share
plot_slices_3d(
    ShareFunc,
    0,
    mMax,
    y_slices=n_slices,
    y_name="n",
    titles=["t = " + str(t) for t in periods],
    ax_labs=["m", "S"],
)

# Consumption
shares = [0.0, 0.9]
plot_slices_4d(
    cFuncFxd,
    0,
    mMax,
    y_slices=n_slices,
    w_slices=shares,
    slice_names=["n_til", "s"],
    titles=["t = " + str(t) for t in periods],
    ax_labs=["m_til", "c"],
)

# %%
# Solve a short, finite horizon version
par_finite = init_risky_contrib.copy()

# Four period model
par_finite["PermGroFac"] = [2.0, 1.0, 0.1, 1.0]
par_finite["PermShkStd"] = [0.1, 0.1, 0.0, 0.0]
par_finite["TranShkStd"] = [0.2, 0.2, 0.0, 0.0]
par_finite["AdjustPrb"] = [0.5, 0.5, 1.0, 1.0]
par_finite["tau"] = [0.1, 0.1, 0.0, 0.0]
par_finite["LivPrb"] = [1.0, 1.0, 1.0, 1.0]
par_finite["T_cycle"] = 4
par_finite["T_retire"] = 0
par_finite["T_age"] = 4

# Adjust discounting and returns distribution so that they make sense in a
# 4-period model
par_finite["DiscFac"] = 0.95 ** 15
par_finite["Rfree"] = 1.03 ** 15
par_finite["RiskyAvg"] = 1.08 ** 15  # Average return of the risky asset
par_finite["RiskyStd"] = 0.20 * np.sqrt(15)  # Standard deviation of (log) risky returns


# Create and solve
contrib_agent = RiskyContribConsumerType(**par_finite)
print("Now solving")
t0 = time()
contrib_agent.solve()
t1 = time()
print("Solving took " + str(t1 - t0) + " seconds.")

# Plot Policy functions
periods = [0, 2, 3]

dfracFunc = [contrib_agent.solution[t].stage_sols["Reb"].dfracFunc_Adj for t in periods]
ShareFunc = [contrib_agent.solution[t].stage_sols["Sha"].ShareFunc_Adj for t in periods]
cFuncFxd = [contrib_agent.solution[t].stage_sols["Cns"].cFunc for t in periods]

# Rebalancing
plot_slices_3d(
    dfracFunc,
    0,
    mMax,
    y_slices=n_slices,
    y_name="n",
    titles=["t = " + str(t) for t in periods],
    ax_labs=["m", "d"],
)
# Share
plot_slices_3d(
    ShareFunc,
    0,
    mMax,
    y_slices=n_slices,
    y_name="n",
    titles=["t = " + str(t) for t in periods],
    ax_labs=["m", "S"],
)
# Consumption
plot_slices_4d(
    cFuncFxd,
    0,
    mMax,
    y_slices=n_slices,
    w_slices=shares,
    slice_names=["n_til", "s"],
    titles=["t = " + str(t) for t in periods],
    ax_labs=["m_til", "c"],
)

# %%  Simulate the finite horizon consumer
contrib_agent.track_vars = [
    "pLvl",
    "t_age",
    "Adjust",
    "mNrm",
    "nNrm",
    "mNrmTilde",
    "nNrmTilde",
    "aNrm",
    "cNrm",
    "Share",
    "dfrac",
]
contrib_agent.T_sim = 4
contrib_agent.AgentCount = 10
contrib_agent.initialize_sim()
contrib_agent.simulate()

# %% Format simulation results

import pandas as pd

df = contrib_agent.history

# Add an id to the simulation results
agent_id = np.arange(contrib_agent.AgentCount)
df["id"] = np.tile(agent_id, (contrib_agent.T_sim, 1))

# Flatten variables
df = {k: v.flatten(order="F") for k, v in df.items()}

# Make dataframe
df = pd.DataFrame(df)
