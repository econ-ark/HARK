'''
This module defines some default parameters for the fashion victim model.
'''

DiscFac          = 0.95
uParamA          = 1.0
uParamB          = 5.0
punk_utility     = 0.0
jock_utility     = 0.0
switchcost_J2P   = 2.0
switchcost_P2J   = 2.0
pCount           = 51
pref_shock_mag   = 0.5
pNextIntercept   = 0.1
pNextSlope       = 0.8
pNextWidth       = 0.1
pNextCount       = 10
pop_size         = 20
p_init           = 0.5

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
