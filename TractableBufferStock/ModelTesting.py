import traceback
import sys
import numpy as np
import metacomm.combinatorics.all_pairs2
all_pairs = metacomm.combinatorics.all_pairs2.all_pairs2

class parameterCheck(object):
    
    def __init__(self, model, base_primitives, multiplier = 1, interval = 5):
        
        self._model = model
        self._base_primitives = base_primitives
        self._multiplier = multiplier
        self._interval = interval
        self._iterator = self.makeParameterIterator()
        self._testParams = self.findTestParameters()
        self.test_results = []
        self.validParams = []
        self.failedParams = []
        
    def makeParameterIterator(self):
        
        mixMaxRangeTupples = {k:(v-self._multiplier*v,v+self._multiplier*v,v/self._interval) for k,v in self._base_primitives.iteritems()}
        totalLoops = self._interval**len(self._base_primitives)
        print("There are " + str(totalLoops)+ " parameter combinations to test.")
        return mixMaxRangeTupples
        
    def findTestParameters(self):
        
        parameterLists = []
        keyOrder = []
        testParams = []
        for k,v in self._iterator.iteritems():
            parameterRange = np.arange(*v)
            parameterLists.append(parameterRange)
            keyOrder.append(k)
        pairwise = list(all_pairs(parameterLists))
        print("Testing parameters reduced to " + str(len(pairwise)) + " pairwise combinations")
        for i in range(len(pairwise)):
            testParams.append(dict(zip(keyOrder,pairwise[i])))
        return testParams
    
    def testParamaters(self):
        
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
           
        for i in range(len(paramtersToTest)):
            tempDict = dict(self._base_primitives)
            tempParams = paramtersToTest[i]
            testData = parameterInstanceCheck(i,tempParams,tempDict)
            Test = self._model(**tempParams)
            try:
                Test.solve()
            except Exception,e:
                testData.errorBoolean = True
                testData.errorCode = str(e)
                testData._tracebackText = sys.exc_info()
            self.test_results.append(testData)
          
        for i in range(len(self.test_results)):
            if self.test_results[i].errorBoolean:
                self.failedParams.append(self.test_results[i])
            else:
                self.validParams.append(self.test_results[i])
                
    def printErrors(self):
    
        for i in range(len(self.test_results)):
            if self.test_results[i].errorBoolean:
                print("test no " + str(i) + " failed with the following error code:")
                print(self.test_results[i].errorCode)
    
class parameterInstanceCheck(object):
    
    def __init__(self,testNumber,base_primitives,original_primitives,errorBoolean=False,errorCode=None,tracebackText=None):
        
        self.testNumber = testNumber
        self.original_primitives = original_primitives
        self.tested_primitives = base_primitives
        self.errorBoolean = errorBoolean
        self.errorCode = errorCode
        self._tracebackText = tracebackText
        
        
    def traceback(self):
        try:
            traceback.print_exception(*self._tracebackText)
        except TypeError:
            print("The test was run successfully - no error generated")
