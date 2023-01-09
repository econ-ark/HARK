"""
This file implements unit tests for several of the ConsumptionSaving models in HARK.
These tests compare the output of different models in specific cases in which those models
should yield the same output.  The code will pass these tests if and only if the output is close
"enough".
"""

# Bring in modules we need
import unittest
from copy import deepcopy
import numpy as np

# Bring in the HARK models we want to test
from HARK.ConsumptionSaving.ConsIndShockModel import (
    PerfForesightConsumerType,
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.TractableBufferStockModel import TractableConsumerType
from HARK.distribution import DiscreteDistribution, DiscreteDistributionLabeled


class Compare_PerfectForesight_and_Infinite(unittest.TestCase):
    """
    Class to compare output of the perfect foresight and infinite horizon models.
    When income uncertainty is removed from the infinite horizon model, it reduces in theory to
    the perfect foresight model.  This class implements tests to make sure it reduces in practice
    to the perfect foresight model as well.
    """

    def setUp(self):
        """
        Prepare to compare the models by initializing and solving them
        """
        # Set up and solve infinite type

        # Define a test dictionary that should have the same solution in the
        # perfect foresight and idiosyncratic shocks models.
        test_dictionary = deepcopy(init_idiosyncratic_shocks)
        test_dictionary["LivPrb"] = [1.0]
        test_dictionary["DiscFac"] = 0.955
        test_dictionary["PermGroFac"] = [1.0]
        test_dictionary["PermShkStd"] = [0.0]
        test_dictionary["TranShkStd"] = [0.0]
        test_dictionary["UnempPrb"] = 0.0
        test_dictionary["T_cycle"] = 1
        test_dictionary["T_retire"] = 0
        test_dictionary["BoroCnstArt"] = None

        InfiniteType = IndShockConsumerType(**test_dictionary)
        InfiniteType.cycles = 0

        InfiniteType.update_income_process()
        InfiniteType.solve()
        InfiniteType.unpack("cFunc")

        # Make and solve a perfect foresight consumer type with the same parameters
        PerfectForesightType = PerfForesightConsumerType(**test_dictionary)
        PerfectForesightType.cycles = 0

        PerfectForesightType.solve()
        PerfectForesightType.unpack("cFunc")

        self.InfiniteType = InfiniteType
        self.PerfectForesightType = PerfectForesightType

    def test_consumption(self):
        """"
        Now compare the consumption functions and make sure they are "close"
        """
        mNrmMinInf = self.InfiniteType.solution[0].mNrmMin  # mNrm min in inf hor model
        aXtraMin = self.InfiniteType.aXtraMin  # point above min where a grid starts
        aXtraMax = self.InfiniteType.aXtraMax  # point above min where a grid ends

        def diffFunc(m):
            return self.PerfectForesightType.solution[0].cFunc(
                m
            ) - self.InfiniteType.cFunc[0](m)

        points = np.arange(0.5, mNrmMinInf+aXtraMin, mNrmMinInf+aXtraMax)
        difference = diffFunc(points)
        max_difference = np.max(np.abs(difference))

        self.assertLess(max_difference, 0.01)


class Compare_TBS_and_Markov(unittest.TestCase):
    """
    Class to compare output of the Tractable Buffer Stock and Markov models.
    The only uncertainty in the TBS model is over when the agent will enter an absorbing state
    with 0 income.  With the right transition arrays and income processes, this is just a special
    case of the Markov model.  So with the right inputs, we should be able to solve the two
    different models and get the same outputs.
    """

    def setUp(self):
        # Set up and solve TBS
        base_primitives = {
            "UnempPrb": 0.015,
            "DiscFac": 0.9,
            "Rfree": 1.1,
            "PermGroFac": 1.05,
            "CRRA": 0.95,
        }
        TBSType = TractableConsumerType(**base_primitives)
        TBSType.solve()

        # Set up and solve Markov
        MrkvArray = [
            np.array(
                [
                    [1.0 - base_primitives["UnempPrb"], base_primitives["UnempPrb"]],
                    [0.0, 1.0],
                ]
            )
        ]
        Markov_primitives = {
            "CRRA": base_primitives["CRRA"],
            "Rfree": np.array(2 * [base_primitives["Rfree"]]),
            "PermGroFac": [
                np.array(
                    2
                    * [
                        base_primitives["PermGroFac"]
                        / (1.0 - base_primitives["UnempPrb"])
                    ]
                )
            ],
            "BoroCnstArt": None,
            "PermShkStd": [0.0],
            "PermShkCount": 1,
            "TranShkStd": [0.0],
            "TranShkCount": 1,
            "T_total": 1,
            "UnempPrb": 0.0,
            "UnempPrbRet": 0.0,
            "T_retire": 0,
            "IncUnemp": 0.0,
            "IncUnempRet": 0.0,
            "aXtraMin": 0.001,
            "aXtraMax": TBSType.mUpperBnd,
            "aXtraCount": 48,
            "aXtraExtra": [None],
            "aXtraNestFac": 3,
            "LivPrb": [np.array([1.0, 1.0]), ],
            "DiscFac": base_primitives["DiscFac"],
            "Nagents": 1,
            "psi_seed": 0,
            "xi_seed": 0,
            "unemp_seed": 0,
            "tax_rate": 0.0,
            "vFuncBool": False,
            "CubicBool": True,
            "MrkvArray": MrkvArray,
            "T_cycle": 1,
        }

        MarkovType = MarkovConsumerType(**Markov_primitives)
        MarkovType.cycles = 0
        employed_income_dist = DiscreteDistributionLabeled(
            pmv=np.ones(1),
            data=np.array([[1.0], [1.0]]),
            var_names=["PermShk", "TranShk"],
        )
        unemployed_income_dist = DiscreteDistributionLabeled(
            pmv=np.ones(1),
            data=np.array([[1.0], [0.0]]),
            var_names=["PermShk", "TranShk"],
        )
        MarkovType.IncShkDstn = [[employed_income_dist, unemployed_income_dist]]

        MarkovType.solve()
        MarkovType.unpack("cFunc")

        self.TBSType = TBSType
        self.MarkovType = MarkovType

    def test_consumption(self):
        # Now compare the consumption functions and make sure they are "close"

        def diffFunc(m):
            return self.TBSType.solution[0].cFunc(m) - self.MarkovType.cFunc[0][0](m)

        points = np.arange(0.1, 10.0, 0.01)
        difference = diffFunc(points)
        max_difference = np.max(np.abs(difference))

        self.assertLess(max_difference, 0.01)


if __name__ == "__main__":
    # Run all the tests
    unittest.main()
