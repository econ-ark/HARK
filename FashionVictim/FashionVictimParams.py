'''
Defines some default parameters for the fashion victim model.
'''
from __future__ import print_function

DiscFac          = 0.95   # Intertemporal discount factor
uParamA          = 1.0    # Parameter A in the utility function (pdf of the beta distribution)
uParamB          = 5.0    # Parameter B in the utility function (pdf of the beta distribution)
punk_utility     = 0.0    # Direct utility received from dressing as a punk
jock_utility     = 0.0    # Direct utility received from dressing as a jock
switchcost_J2P   = 2.0    # Cost of switching from jock to punk
switchcost_P2J   = 2.0    # Cost of switching from punk to jock
pCount           = 51     # Number of points in the grid of population punk proportion values
pref_shock_mag   = 0.5    # Scaling factor for the magnitude of transitory style preference shocks
pNextIntercept   = 0.1    # Intercept of linear function of beliefs over next period's punk proportion
pNextSlope       = 0.8    # Slope of linear function of beliefs over next period's punk proportion
pNextWidth       = 0.1    # Width of uniform distribution of next period's punk proportion (around linear function)
pNextCount       = 10     # Number of points in discrete approximation to distribution of next period's p
pop_size         = 20     # Number of fashion victims of this type (for simulation)
p_init           = 0.5    # Probability of being dressed as a punk when the simulation begins

# Make a dictionary for convenient type creation
default_params={'DiscFac'      : DiscFac,
               'uParamA'       : uParamA,
               'uParamB'       : uParamB,
               'punk_utility'  : punk_utility,
               'jock_utility'  : jock_utility,
               'switchcost_J2P': switchcost_J2P,
               'switchcost_P2J': switchcost_P2J,
               'pCount'        : pCount,
               'pref_shock_mag': pref_shock_mag,
               'pNextIntercept': pNextIntercept,
               'pNextSlope'    : pNextSlope,
               'pNextWidth'    : pNextWidth,
               'pNextCount'    : pNextCount,
               'pop_size'      : pop_size,
               'p_init'        : p_init
               }
               
               
if __name__ == '__main__':
    print("Sorry, FashionVictimParams doesn't actually do anything on its own.")
    print("This module is imported by FashionVictimModel, providing example")
    print("parameters for the model.  Please see that module if you want more")
    print("interesting output.")
