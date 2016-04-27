# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:08:22 2016

@author: kaufmana
"""


import TractableBufferStock as Model
import ModelTesting as test

base_primitives = {'mho' : .015,
                   'beta' : 0.9,
                   'R' : 1.1,
                   'G' : 1.05,
                   'rho' : .95}

#assign a model and base parameters to be checked
TBSCheck = test.parameterCheck(Model.TractableConsumerType,base_primitives)

#run the testing function.  This runs the model multiple times
TBSCheck.testParamaters()
print("-----------------------------------------------------------------------")

#get a test result and find out more info
test100 = TBSCheck.test_results[100]
print("the test number is : " + str(test100.testNumber))
print("the test parameters were : " + str(test100.tested_primitives))
print("the error code is : " + str(test100.errorCode))
print("the traceback for the error looked like : ")
test100.traceback()
