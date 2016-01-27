'''
This module runs an example version of the Tractable Buffer Stock model.
'''

import numpy as np
import TractableBufferStock as Model
from HARKutilities import plotFunc, plotFuncs
from time import clock

# Define the model primitives
base_primitives = {'mho' : .00625,
                   'beta' : 0.975,
                   'R' : 1.01,
                   'G' : 1.0025,
                   'rho' : 1.0}
                   
# Make and solve a tractable consumer type
ExampleType = Model.TractableConsumerType(**base_primitives)
t_start = clock()
ExampleType.solve()
t_end = clock()
print('Solving a tractable consumption-savings model took ' + str(t_end-t_start) + ' seconds.')

# Plot the consumption function and whatnot
m_upper = 1.5*ExampleType.m_targ
conFunc_PF = lambda m: ExampleType.h*ExampleType.kappa_PF + ExampleType.kappa_PF*m
plotFuncs([ExampleType.solution[0].cFunc,ExampleType.mSSfunc,ExampleType.cSSfunc],0,m_upper)
