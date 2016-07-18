"""
This file solves the Tractable Buffer Stock model for many different parameter values, keeping
track of when the model generates an error.
"""

# First, tell Python what directories we will be using
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))
sys.path.insert(0, os.path.abspath('./'))

# Bring in the HARK model we want to test
import TractableBufferStockModel as Model

# Bring in...
import ModelTesting as test

base_primitives = {'UnempPrb' : .015,
                   'DiscFac' : 0.9,
                   'Rfree' : 1.1,
                   'PermGroFac' : 1.05,
                   'CRRA' : .95}
                   
#assign a model and base parameters to be checked
TBSCheck = test.parameterCheck(Model.TractableConsumerType,base_primitives)

#run the testing function.  This runs the model multiple times
TBSCheck.testParameters()
print("-----------------------------------------------------------------------")

#get a test result and find out more info
test100 = TBSCheck.test_results[0]
print("the test number is : " + str(test100.testNumber))
print("the test parameters were : " + str(test100.tested_primitives))
print("the error code is : " + str(test100.errorCode))
print("the traceback for the error looked like : ")
test100.traceback()
