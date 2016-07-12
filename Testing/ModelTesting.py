import sys
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSavingModel'))
sys.path.insert(0, os.path.abspath('./'))

import traceback
import sys
import itertools
import numpy as np


class parameterCheck(object):
    '''
    A wrapper for an AgentType object (a model). This class automates  
    testing over a range parameter inputs by generating and testing sets of 
    parameters based on the original parameters.
    '''
    
    def __init__(self, model, base_primitives, multiplier = .1, interval = 2):
        '''        
        model: an instance of AgentType with a working .solve() function
        
        base_primitives: a dictionary of input parameters for the the model
        
        multiplier: coefficient that determines the range for each parameter 
        within testing sets.  
        the range for each parameter P is [P-P*multiplier,P+P*multiplier].  All 
        testing parameters will be within this range
        
        interval: the number of parameters to to test within the given range
        '''
        self._model            = model
        self._base_primitives  = base_primitives
        self._multiplier       = multiplier
        self._interval         = interval
        self._iterator         = self.makeParameterIterator()
        self._testParams       = self.findTestParameters()
        self.test_results      = []
        self.validParams       = []
        self.failedParams      = []
        
    def makeParameterIterator(self):
        '''
        create an object that contains all the information needed to generate 
        sets of parameters for testing
        
        returns a dictionary that specifies the range and intervals for each parameter
        
        '''
        mixMaxRangeTupples = {k:(v-self._multiplier*v,v+self._multiplier*v,v/self._interval) for k,v in self._base_primitives.iteritems()}
        totalLoops = self._interval**len(self._base_primitives)
        print("There are " + str(totalLoops)+ " parameter combinations to test.")
        return mixMaxRangeTupples
        
    def findTestParameters(self):
        '''
        this function creates sets (dictionaries) of parameters to test in the model
        it also applies pairwise combination to reduce actual the number of sets
        that are tested.  For more info see pairwise.org
        
        returns a list of parameter sets (dictionaries) for testing
        '''
        parameterLists = []
        keyOrder       = []
        testParams     = []
        for k,v in self._iterator.iteritems():
            parameterRange = np.arange(*v)
            parameterLists.append(parameterRange)
            keyOrder.append(k)
        for param_combination in itertools.product(*parameterLists):
            testParams.append(dict(zip(keyOrder,param_combination)))

        return testParams
    
    def testParameters(self):
        '''
        runs the model on the test parameters and store error results
        print out the error messages that were thrown
        '''        
        
        self.runModel(self._testParams)
        self.printErrors()
        
    def narrowParameters(self):
        '''
        this function needs to be able to identify the valid parameter space

        then it can plug in those values to the makeParameterIterator function and rerun the models        
        
        self._iterator = self.makeParameterIterator()
        
        parameterLists = []
        for k,v in self._iterator.iteritems():
            parameterRange = np.arange(*v)
            parameterLists.append(parameterRange)
        pairwise = list(all_pairs(parameterLists, previously_tested=self._testedParams))
        print("Subsequent round of testing reduced to " + str(len(pairwise)) + " pairwise combinations of parameters")
        
        self.runModel(pairwise)
        '''
        pass
    def runModel(self,paramtersToTest):
        '''
        run the model using each set of test parameters.  for each model, a new
        object (an instance of parameterInstanceCheck) records the results of
        the test.  
        
        Each result is places in the appropriate list (failedParams or validParams)
        '''
        for i in range(len(paramtersToTest)):
            tempDict   = dict(self._base_primitives)
            tempParams = paramtersToTest[i]
            testData   = parameterInstanceCheck(i,tempParams,tempDict)
            Test       = self._model(**tempParams)
            try:
                Test.solve()
            #TODO: Insert allowed exceptions here so they don't count as errors!
            except Exception,e:
                testData.errorBoolean    = True
                testData.errorCode      = str(e)
                testData._tracebackText = sys.exc_info()
            self.test_results.append(testData)
          
        for i in range(len(self.test_results)):
            if self.test_results[i].errorBoolean:
                self.failedParams.append(self.test_results[i])
            else:
                self.validParams.append(self.test_results[i])
                
    def printErrors(self):
        '''
        print out the test numbers and error codes for all failed tests
        '''
        for i in range(len(self.test_results)):
            if self.test_results[i].errorBoolean:
                print("test no " + str(i) + " failed with the following error code:")
                print(self.test_results[i].errorCode)
    
class parameterInstanceCheck(object):
    '''
    this class holds information for a single test of a model
    '''
    def __init__(self,testNumber,base_primitives,original_primitives,errorBoolean=False,errorCode=None,tracebackText=None):
        '''
        testNumber: the test number
        
        base_primitives: the set of parameters that was tested
        
        original_primitives: the original parameters that test parameters were constructed from
                    
        errorBoolean: boolean indicator of an error    

        errorCode: text of the error (exception type included)
        
        tracebackText: full traceback, printable using the traceback.prin_excpetino function
        
        '''
        
        self.testNumber          = testNumber
        self.original_primitives = original_primitives
        self.tested_primitives   = base_primitives
        self.errorBoolean        = errorBoolean
        self.errorCode           = errorCode
        self._tracebackText      = tracebackText
        
        
    def traceback(self):
        '''
        function that prints a traceback for an errror
        '''
        try:
            traceback.print_exception(*self._tracebackText)
        except TypeError:
            print("The test was run successfully - no error generated")
