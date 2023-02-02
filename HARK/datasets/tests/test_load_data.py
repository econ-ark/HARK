import unittest
from HARK.datasets import load_SCF_wealth_weights
from HARK.datasets.cpi.us.CPITools import cpi_deflator
from HARK.datasets.SCF.WealthIncomeDist.SCFDistTools import income_wealth_dists_from_scf


class test_load_SCF_wealth_weights(unittest.TestCase):
    def setUp(self):
        self.SCF_wealth, self.SCF_weights = load_SCF_wealth_weights()

    def test_shape(self):
        self.assertEqual(self.SCF_wealth.shape, (3553,))
        self.assertEqual(self.SCF_weights.shape, (3553,))


# %% US CPI tests
class test_cpi_deflators(unittest.TestCase):
    def test_month_deflators(self):
        # Same year test
        defl_same_year = cpi_deflator(2000, 2000, "SEP")
        self.assertEqual(defl_same_year[0], 1.0)

        # Different year test
        defl_diff_year = cpi_deflator(1998, 2019, "SEP")
        self.assertAlmostEqual(defl_diff_year[0], 1.57279534)

    def test_avg_deflators(self):
        # Same year test
        defl_same_year = cpi_deflator(2000, 2000)
        self.assertEqual(defl_same_year[0], 1.0)

        # Different year test
        defl_diff_year = cpi_deflator(1998, 2019)
        self.assertAlmostEqual(defl_diff_year[0], 1.57202505)


# %% Tests for Survey of Consumer finances initial distributions
class test_SCF_dists(unittest.TestCase):
    def setUp(self):
        self.BaseYear = 1992

    def test_at_21(self):
        # Get stats for various groups and test them
        NoHS = income_wealth_dists_from_scf(
            self.BaseYear, age=21, education="NoHS", wave=1995
        )
        self.assertAlmostEqual(NoHS["aNrmInitMean"], -1.0611984728537684)
        self.assertAlmostEqual(NoHS["aNrmInitStd"], 1.475816500147777)
        self.assertAlmostEqual(NoHS["pLvlInitMean"], 2.5413398571226233)
        self.assertAlmostEqual(NoHS["pLvlInitStd"], 0.7264931123240703)

        HS = income_wealth_dists_from_scf(
            self.BaseYear, age=21, education="HS", wave=2013
        )
        self.assertAlmostEqual(HS["aNrmInitMean"], -1.0812342937817578)
        self.assertAlmostEqual(HS["aNrmInitStd"], 1.7526704743231725)
        self.assertAlmostEqual(HS["pLvlInitMean"], 2.806605268756435)
        self.assertAlmostEqual(HS["pLvlInitStd"], 0.6736467457859727)

        Coll = income_wealth_dists_from_scf(
            self.BaseYear, age=21, education="College", wave=2019
        )
        self.assertAlmostEqual(Coll["aNrmInitMean"], -0.6837248150760165)
        self.assertAlmostEqual(Coll["aNrmInitStd"], 0.8813676761170798)
        self.assertAlmostEqual(Coll["pLvlInitMean"], 3.2790838587291127)
        self.assertAlmostEqual(Coll["pLvlInitStd"], 0.746362502979793)

    def test_at_60(self):
        # Get stats for various groups and test them
        NoHS = income_wealth_dists_from_scf(
            self.BaseYear, age=60, education="NoHS", wave=1995
        )
        self.assertAlmostEqual(NoHS["aNrmInitMean"], 0.1931578281432479)
        self.assertAlmostEqual(NoHS["aNrmInitStd"], 1.6593916577375334)
        self.assertAlmostEqual(NoHS["pLvlInitMean"], 3.3763953392998705)
        self.assertAlmostEqual(NoHS["pLvlInitStd"], 0.61810580085094993)

        HS = income_wealth_dists_from_scf(
            self.BaseYear, age=60, education="HS", wave=2013
        )
        self.assertAlmostEqual(HS["aNrmInitMean"], 0.6300862955841334)
        self.assertAlmostEqual(HS["aNrmInitStd"], 1.7253736778036055)
        self.assertAlmostEqual(HS["pLvlInitMean"], 3.462790681398899)
        self.assertAlmostEqual(HS["pLvlInitStd"], 0.8179188962937205)

        Coll = income_wealth_dists_from_scf(
            self.BaseYear, age=60, education="College", wave=2019
        )
        self.assertAlmostEqual(Coll["aNrmInitMean"], 1.643936802283761)
        self.assertAlmostEqual(Coll["aNrmInitStd"], 1.2685135110865389)
        self.assertAlmostEqual(Coll["pLvlInitMean"], 4.278905678818748)
        self.assertAlmostEqual(Coll["pLvlInitStd"], 1.0776403992280614)
