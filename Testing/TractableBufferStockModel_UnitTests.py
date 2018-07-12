# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:01:50 2016

@author: kaufmana
"""
from __future__ import print_function, division
from __future__ import absolute_import

from builtins import str
from builtins import zip
from builtins import range
from builtins import object

import HARK.ConsumptionSaving.TractableBufferStockModel as Model
import unittest

class FuncTest(unittest.TestCase):

    def setUp(self):
        base_primitives = {'UnempPrb' : .015,
                           'DiscFac' : 0.9,
                           'Rfree' : 1.1,
                           'PermGroFac' : 1.05,
                           'CRRA' : .95}
        test_model = Model.TractableConsumerType(**base_primitives)
        test_model.solve()
        cNrm_list = [0.0,
                  0.6170411710160961,
                  0.7512931350607787,
                  0.8242071925443384,
                  0.8732633069358244,
                  0.9090443048442146,
                  0.9362584565290604,
                  0.9574865107447327,
                  0.9743235996720729,
                  0.9878347049396029,
                  0.9987694718922687,
                  1.0499840337356576,
                  1.0988370658458553,
                  1.1079081119060201,
                  1.1185500922622567,
                  1.1309953859705277,
                  1.1454986397022289,
                  1.1623357560591763,
                  1.1818022106863713,
                  1.2042108062871855,
                  1.2298890682784422,
                  1.2591765689896088,
                  1.2924225145436121,
                  1.329983925942064,
                  1.372224689976677,
                  1.4195156568037894,
                  1.4722358408529614,
                  1.5307746658958221]
        return test_model.solution[0].cNrm_list,cNrm_list

    def test1(self):
        results = self.setUp()
        self.assertEqual(results[0],results[1])


if __name__ == '__main__':
    unittest.main()
