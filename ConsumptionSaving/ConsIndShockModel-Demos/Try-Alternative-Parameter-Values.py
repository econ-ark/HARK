# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:40:49 2017

@author: ccarroll@llorracc.org
"""
from builtins import str
from builtins import range
import sys   
import os    
import pylab # the plotting tools

sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../..'))

xPoints=100  # number of points at which to sample a function when plotting it using pylab
mMinVal = 0. # minimum value of the consumer's cash-on-hand to show in plots
mMaxVal = 5. # maximum value of the consumer's cash-on-hand to show in plots

import ConsumerParameters as Params # Read in the database of parameters
my_dictionary = Params.init_idiosyncratic_shocks # Create a dictionary containing the default values of the parameters
import numpy as np # Get the suite of tools for doing numerical computation in python
from HARKutilities import plotFuncs # Get some tools developed for plotting HARK functions
from ConsIndShockModel import IndShockConsumerType # Set up the tools for solving a consumer's problem

# define a function that generates the plot
def perturbParameterToGetcPlotList(base_dictionary,param_name,param_min,param_max,N=20,time_vary=False):
    param_vec = np.linspace(param_min,param_max,num=N,endpoint=True) # vector of alternative values of the parameter to examine
    thisConsumer = IndShockConsumerType(**my_dictionary) # create an instance of the consumer type
    thisConsumer.cycles = 0 # Make this type have an infinite horizon
    x = np.linspace(mMinVal,mMaxVal,xPoints,endpoint=True) # Define a vector of x values that span the range from the minimum to the maximum values of m
    
    for j in range(N): # loop from the first to the last values of the parameter
        if time_vary: # Some parameters are time-varying; others are not
            setattr(thisConsumer,param_name,[param_vec[j]])
        else:
            setattr(thisConsumer,param_name,param_vec[j])
        thisConsumer.update() # set up the preliminaries required to solve the problem
        thisConsumer.solve()  # solve the problem
        y = thisConsumer.solution[0].cFunc(x) # Get the values of the consumption function at the points in the vector of x points
        pylab.plot(x,y,label=str(round(param_vec[j],3))) # plot it and generate a label indicating the rounded value of the parameter
        pylab.legend(loc='upper right') # put the legend in the upper right 
    return pylab # return the figure

cPlot_by_DiscFac = perturbParameterToGetcPlotList(my_dictionary,'DiscFac',0.899,0.999,5,False) # create the figure
cPlot_by_DiscFac.show() # show it

