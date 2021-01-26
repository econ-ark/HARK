# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 10:06:51 2021

@author: Mateo
"""

import unittest
from copy import copy
import numpy as np
from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    RiskyContribConsumerType,
    init_riskyContrib,
)


class test_(unittest.TestCase):
    def setUp(self):

        # A set of finite parameters
        self.par_finite = init_riskyContrib.copy()

        # Four period model
        self.par_finite["PermGroFac"] = [2.0, 1.0, 0.1, 1.0]
        self.par_finite["PermShkStd"] = [0.1, 0.1, 0.0, 0.0]
        self.par_finite["TranShkStd"] = [0.2, 0.2, 0.0, 0.0]
        self.par_finite["AdjustPrb"] = [0.5, 0.5, 1.0, 1.0]
        self.par_finite["tau"] = [0.1, 0.1, 0.0, 0.0]
        self.par_finite["LivPrb"] = [1.0, 1.0, 1.0, 1.0]
        self.par_finite["T_cycle"] = 4
        self.par_finite["T_retire"] = 0
        self.par_finite["T_age"] = 4

        # Adjust discounting and returns distribution so that they make sense in a
        # 4-period model
        self.par_finite["DiscFac"] = 0.95 ** 15
        self.par_finite["Rfree"] = 1.03 ** 15
        self.par_finite["RiskyAvg"] = 1.08 ** 15  # Average return of the risky asset
        self.par_finite["RiskyStd"] = 0.20 * np.sqrt(
            15
        )  # Standard deviation of (log) risky returns

    def test_finite_cont_share(self):
        # Finite horizon with continuous contribution share
        cont_params = copy(self.par_finite)
        cont_params["DiscreteShareBool"] = False
        cont_params["vFuncBool"] = False

        fin_cont_agent = RiskyContribConsumerType(**cont_params)
        fin_cont_agent.solve()

        self.assertAlmostEqual(
            fin_cont_agent.solution[0].stageSols["Reb"].DFuncAdj(3, 4), -0.87757204
        )
        self.assertAlmostEqual(
            fin_cont_agent.solution[0].stageSols["Sha"].ShareFuncAdj(5, 0.1), 0.10846904
        )
        self.assertAlmostEqual(
            fin_cont_agent.solution[0].stageSols["Cns"].cFunc(3, 4, 0.1), 2.46055802
        )

    def test_finite_disc_share(self):
        # Finite horizon with discrete contribution share
        disc_params = copy(self.par_finite)
        disc_params["DiscreteShareBool"] = True
        disc_params["vFuncBool"] = True

        fin_disc_agent = RiskyContribConsumerType(**disc_params)
        fin_disc_agent.solve()

        self.assertAlmostEqual(
            fin_disc_agent.solution[0].stageSols["Reb"].DFuncAdj(3, 4), -0.87755064
        )
        self.assertAlmostEqual(
            fin_disc_agent.solution[0].stageSols["Sha"].ShareFuncAdj(5, 0.1), 0.1
        )
        self.assertAlmostEqual(
            fin_disc_agent.solution[0].stageSols["Cns"].cFunc(3, 4, 0.1), 2.46055803
        )
