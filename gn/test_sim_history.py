# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:53:31 2016

@author: ganong
"""
import settings
settings.init()
settings.t_rebate = 25
settings.rebate_size = 0
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0,'../ConsumptionSaving')
sys.path.insert(0,'../SolvingMicroDSOPs')

from copy import copy, deepcopy
import numpy as np
from scipy.optimize import newton
from HARKcore import AgentType, Solution, NullFunc, HARKobject
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKinterpolation import CubicInterp, LowerEnvelope, LinearInterp
from HARKsimulation import drawDiscrete
from HARKutilities import approxMeanOneLognormal, addDiscreteOutcomeConstantMean,\
                          combineIndepDstns, makeGridExpMult, CRRAutility, CRRAutilityP, \
                          CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv, \
                          CRRAutilityP_invP

utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

from HARKutilities import plotFuncsDer, plotFuncs
from time import clock
mystr = lambda number : "{:.4f}".format(number)

#code from HARK
#import ConsumerParameters as Params
#from ConsIndShockModel import IndShockConsumerType
#LifecycleType = IndShockConsumerType(**Params.init_lifecycle)

#my modified code

import EstimationParameters as Params
reload(Params)
import ConsumptionSavingModel_gn as Model

baseline_params = Params.init_consumer_objects
baseline_params['DiscFac'] = (np.array(Params.DiscFac_timevary)*0.96).tolist()
LifecycleType = Model.IndShockConsumerType(**baseline_params)


#solve model
LifecycleType.cycles = 1 # Make this consumer live a sequence of periods exactly once
start_time = clock()
LifecycleType.solve()
end_time = clock()
print('Solving a lifecycle consumer took ' + mystr(end_time-start_time) + ' seconds.')
LifecycleType.unpack_cFunc() #xxx why does unpack_cFunc now have an underscore that it did not before?
LifecycleType.timeFwd()

# Simulate some data for num_agents defaults to 10000; results stored in cHist, mHist, bHist, aHist, MPChist, and pHist
LifecycleType.sim_periods = LifecycleType.T_total + 1
LifecycleType.makeIncShkHist()
LifecycleType.initializeSim()
LifecycleType.simConsHistory()

#how do these change with age
np.mean(LifecycleType.cHist,axis=1) #rising w age and then falling right before death
np.mean(LifecycleType.mHist,axis=1) #rising then falling
np.mean(LifecycleType.bHist,axis=1) #rising then falling bank balances before labor income
np.mean(LifecycleType.aHist,axis=1) #rising then falling
np.mean(LifecycleType.MPChist,axis=1) #falling very rapidly in first five years.
np.mean(LifecycleType.MPChist[:40]) #mean during working life is only 0.07, which is pretty low

np.mean(LifecycleType.pHist,axis=1) #rising permanent income level




#np.std(LifecycleType.cHist,axis=1)