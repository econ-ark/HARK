from copy import deepcopy
from time import time
import numpy as np
from HARK.utilities import plotFuncs
from HARK.ConsumptionSaving.ConsRepAgentModel import (
    RepAgentConsumerType,
    RepAgentMarkovConsumerType,
)

# Make and solve a rep agent model
RAexample = RepAgentConsumerType()
t_start = time()
RAexample.solve()
t_end = time()
print(
    "Solving a representative agent problem took " + str(t_end - t_start) + " seconds."
)
plotFuncs(RAexample.solution[0].cFunc, 0, 20)

# Simulate the representative agent model
RAexample.T_sim = 2000
RAexample.track_vars = ["cNrmNow", "mNrmNow", "Rfree", "wRte"]
RAexample.initializeSim()
t_start = time()
RAexample.simulate()
t_end = time()
print(
    "Simulating a representative agent for "
    + str(RAexample.T_sim)
    + " periods took "
    + str(t_end - t_start)
    + " seconds."
)

# Make and solve a Markov representative agent
RAmarkovExample = RepAgentMarkovConsumerType()
RAmarkovExample.IncomeDstn[0] = 2 * [RAmarkovExample.IncomeDstn[0]]
t_start = time()
RAmarkovExample.solve()
t_end = time()
print(
    "Solving a two state representative agent problem took "
    + str(t_end - t_start)
    + " seconds."
)
plotFuncs(RAmarkovExample.solution[0].cFunc, 0, 10)

# Simulate the two state representative agent model
RAmarkovExample.T_sim = 2000
RAmarkovExample.track_vars = ["cNrmNow", "mNrmNow", "Rfree", "wRte", "MrkvNow"]
RAmarkovExample.initializeSim()
t_start = time()
RAmarkovExample.simulate()
t_end = time()
print(
    "Simulating a two state representative agent for "
    + str(RAexample.T_sim)
    + " periods took "
    + str(t_end - t_start)
    + " seconds."
)
