This folder contains code for the paper of Bayer and Luetikke, "Solving heterogeneous agent models in discrete time with many idiosyncratic states by perturbation methods" found at:

https://cepr.org/active/publications/discussion_papers/dp.php?dpno=13071#

This folder contains preliminary work that has not yet been fully integrated into HARK.

1) BayerLuetticke_wrapper.py creates a wrapper to the one asset version of BayerLuetikke's code. 
This file:
Creates BayerLuettickeAgent and BayerLuettickeEconomy which are simple wrappers to BayerLuetikke's code, presently only the steady state part of this.
Simulates a BayerLuettickeEconomy with 10,000 agents in steady state

2) ConsIndShockModel_extension.py extends ConsIndShockModel to calculate and store a histogram of the distribution of agents.


The BayerLuettike code can also be run directly to recover impulse response functions to aggregate shocks.
The paper above considers the two asset model, the code for which is found in the folder BayerLuetticke_code/TwoAssetCode
To run this code run the two files in order:
1) SteadyStateTwoAsset.py - solves the steady state
2) FluctuationsTwoAsset.py - solves the aggregate shocks and plots impulse response functions
