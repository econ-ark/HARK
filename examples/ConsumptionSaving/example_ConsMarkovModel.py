# %%
from HARK.utilities import plotFuncs
from time import process_time
from copy import deepcopy, copy
import numpy as np
from HARK.ConsumptionSaving.ConsIndShockModel import init_idiosyncratic_shocks
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
mystr = lambda number: "{:.4f}".format(number)
do_simulation = True

# %%
# Define the Markov transition matrix for serially correlated unemployment
unemp_length = 5  # Averange length of unemployment spell
urate_good = 0.05  # Unemployment rate when economy is in good state
urate_bad = 0.12  # Unemployment rate when economy is in bad state
bust_prob = 0.01  # Probability of economy switching from good to bad
recession_length = 20  # Averange length of bad state
p_reemploy = 1.0 / unemp_length
p_unemploy_good = p_reemploy * urate_good / (1 - urate_good)
p_unemploy_bad = p_reemploy * urate_bad / (1 - urate_bad)
boom_prob = 1.0 / recession_length
MrkvArray = np.array(
    [
        [
            (1 - p_unemploy_good) * (1 - bust_prob),
            p_unemploy_good * (1 - bust_prob),
            (1 - p_unemploy_good) * bust_prob,
            p_unemploy_good * bust_prob,
        ],
        [
            p_reemploy * (1 - bust_prob),
            (1 - p_reemploy) * (1 - bust_prob),
            p_reemploy * bust_prob,
            (1 - p_reemploy) * bust_prob,
        ],
        [
            (1 - p_unemploy_bad) * boom_prob,
            p_unemploy_bad * boom_prob,
            (1 - p_unemploy_bad) * (1 - boom_prob),
            p_unemploy_bad * (1 - boom_prob),
        ],
        [
            p_reemploy * boom_prob,
            (1 - p_reemploy) * boom_prob,
            p_reemploy * (1 - boom_prob),
            (1 - p_reemploy) * (1 - boom_prob),
        ],
    ]
)

# %%
# Make a consumer with serially correlated unemployment, subject to boom and bust cycles
init_serial_unemployment = copy(init_idiosyncratic_shocks)
init_serial_unemployment["MrkvArray"] = [MrkvArray]
init_serial_unemployment["UnempPrb"] = 0  # to make income distribution when employed
init_serial_unemployment["global_markov"] = False
SerialUnemploymentExample = MarkovConsumerType(**init_serial_unemployment)
SerialUnemploymentExample.cycles = 0
SerialUnemploymentExample.vFuncBool = False  # for easy toggling here

# %%
# Replace the default (lognormal) income distribution with a custom one
employed_income_dist = [np.ones(1), np.ones(1), np.ones(1)]  # Definitely get income
unemployed_income_dist = [np.ones(1), np.ones(1), np.zeros(1)]  # Definitely don't
SerialUnemploymentExample.IncomeDstn = [
    [
        employed_income_dist,
        unemployed_income_dist,
        employed_income_dist,
        unemployed_income_dist,
    ]
]

# %%
# Interest factor, permanent growth rates, and survival probabilities are constant arrays
SerialUnemploymentExample.Rfree = np.array(4 * [SerialUnemploymentExample.Rfree])
SerialUnemploymentExample.PermGroFac = [
    np.array(4 * SerialUnemploymentExample.PermGroFac)
]
SerialUnemploymentExample.LivPrb = [SerialUnemploymentExample.LivPrb * np.ones(4)]

# %%
# Solve the serial unemployment consumer's problem and display solution
SerialUnemploymentExample.timeFwd()
start_time = process_time()
SerialUnemploymentExample.solve()
end_time = process_time()
print(
    "Solving a Markov consumer with serially correlated unemployment took "
    + mystr(end_time - start_time)
    + " seconds."
)
print("Consumption functions for each discrete state:")
plotFuncs(SerialUnemploymentExample.solution[0].cFunc, 0, 50)
if SerialUnemploymentExample.vFuncBool:
    print("Value functions for each discrete state:")
    plotFuncs(SerialUnemploymentExample.solution[0].vFunc, 5, 50)

# %%
# Simulate some data; results stored in cHist, mNrmNow_hist, cNrmNow_hist, and MrkvNow_hist
if do_simulation:
    SerialUnemploymentExample.T_sim = 120
    SerialUnemploymentExample.MrkvPrbsInit = [0.25, 0.25, 0.25, 0.25]
    SerialUnemploymentExample.track_vars = ["mNrmNow", "cNrmNow"]
    SerialUnemploymentExample.makeShockHistory()  # This is optional
    SerialUnemploymentExample.initializeSim()
    SerialUnemploymentExample.simulate()

# %%
# Make a consumer who occasionally gets "unemployment immunity" for a fixed period
UnempPrb = 0.05  # Probability of becoming unemployed each period
ImmunityPrb = 0.01  # Probability of becoming "immune" to unemployment
ImmunityT = 6  # Number of periods of immunity

# %%
StateCount = ImmunityT + 1  # Total number of Markov states
IncomeDstnReg = [
    np.array([1 - UnempPrb, UnempPrb]),
    np.array([1.0, 1.0]),
    np.array([1.0 / (1.0 - UnempPrb), 0.0]),
]  # Ordinary income distribution
IncomeDstnImm = [
    np.array([1.0]),
    np.array([1.0]),
    np.array([1.0]),
]  # Income distribution when unemployed
IncomeDstn = [IncomeDstnReg] + ImmunityT * [
    IncomeDstnImm
]  # Income distribution for each Markov state, in a list

# %%
# Make the Markov transition array.  MrkvArray[i,j] is the probability of transitioning
# to state j in period t+1 from state i in period t.
MrkvArray = np.zeros((StateCount, StateCount))
MrkvArray[0, 0] = (
    1.0 - ImmunityPrb
)  # Probability of not becoming immune in ordinary state: stay in ordinary state
MrkvArray[
    0, ImmunityT
] = (
    ImmunityPrb
)  # Probability of becoming immune in ordinary state: begin immunity periods
for j in range(ImmunityT):
    MrkvArray[
        j + 1, j
    ] = (
        1.0
    )  # When immune, have 100% chance of transition to state with one fewer immunity periods remaining

# %%
init_unemployment_immunity = copy(init_idiosyncratic_shocks)
init_unemployment_immunity["MrkvArray"] = [MrkvArray]
ImmunityExample = MarkovConsumerType(**init_unemployment_immunity)
ImmunityExample.assignParameters(
    Rfree=np.array(np.array(StateCount * [1.03])),  # Interest factor same in all states
    PermGroFac=[
        np.array(StateCount * [1.01])
    ],  # Permanent growth factor same in all states
    LivPrb=[np.array(StateCount * [0.98])],  # Same survival probability in all states
    BoroCnstArt=None,  # No artificial borrowing constraint
    cycles=0,
)  # Infinite horizon
ImmunityExample.IncomeDstn = [IncomeDstn]

# %%
# Solve the unemployment immunity problem and display the consumption functions
start_time = process_time()
ImmunityExample.solve()
end_time = process_time()
print(
    'Solving an "unemployment immunity" consumer took '
    + mystr(end_time - start_time)
    + " seconds."
)
print("Consumption functions for each discrete state:")
mNrmMin = np.min([ImmunityExample.solution[0].mNrmMin[j] for j in range(StateCount)])
plotFuncs(ImmunityExample.solution[0].cFunc, mNrmMin, 10)

# %%
# Make a consumer with serially correlated permanent income growth
UnempPrb = 0.05  # Unemployment probability
StateCount = 5  # Number of permanent income growth rates
Persistence = (
    0.5
)  # Probability of getting the same permanent income growth rate next period

# %%
IncomeDstnReg = [
    np.array([1 - UnempPrb, UnempPrb]),
    np.array([1.0, 1.0]),
    np.array([1.0, 0.0]),
]
IncomeDstn = StateCount * [
    IncomeDstnReg
]  # Same simple income distribution in each state

# %%
# Make the state transition array for this type: Persistence probability of remaining in the same state, equiprobable otherwise
MrkvArray = Persistence * np.eye(StateCount) + (1.0 / StateCount) * (
    1.0 - Persistence
) * np.ones((StateCount, StateCount))

# %%
init_serial_growth = copy(init_idiosyncratic_shocks)
init_serial_growth["MrkvArray"] = [MrkvArray]
SerialGroExample = MarkovConsumerType(**init_serial_growth)
SerialGroExample.assignParameters(
    Rfree=np.array(
        np.array(StateCount * [1.03])
    ),  # Same interest factor in each Markov state
    PermGroFac=[
        np.array([0.97, 0.99, 1.01, 1.03, 1.05])
    ],  # Different permanent growth factor in each Markov state
    LivPrb=[np.array(StateCount * [0.98])],  # Same survival probability in all states
    cycles=0,
)
SerialGroExample.IncomeDstn = [IncomeDstn]


# %%
# Solve the serially correlated permanent growth shock problem and display the consumption functions
start_time = process_time()
SerialGroExample.solve()
end_time = process_time()
print(
    "Solving a serially correlated growth consumer took "
    + mystr(end_time - start_time)
    + " seconds."
)
print("Consumption functions for each discrete state:")
plotFuncs(SerialGroExample.solution[0].cFunc, 0, 10)

# %%
# Make a consumer with serially correlated interest factors
SerialRExample = deepcopy(SerialGroExample)  # Same as the last problem...
SerialRExample.assignParameters(
    PermGroFac=[
        np.array(StateCount * [1.01])
    ],  # ...but now the permanent growth factor is constant...
    Rfree=np.array([1.01, 1.02, 1.03, 1.04, 1.05]),
)  # ...and the interest factor is what varies across states

# %%
# Solve the serially correlated interest rate problem and display the consumption functions
start_time = process_time()
SerialRExample.solve()
end_time = process_time()
print(
    "Solving a serially correlated interest consumer took "
    + mystr(end_time - start_time)
    + " seconds."
)
print("Consumption functions for each discrete state:")
plotFuncs(SerialRExample.solution[0].cFunc, 0, 10)
