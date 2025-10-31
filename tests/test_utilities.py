"""
This file implements unit tests for various functions in HARK.utilities
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt

from HARK.utilities import (
    make_assets_grid,
    get_lorenz_shares,
    make_grid_exp_mult,
    calc_subpop_avg,
    kernel_regression,
    determine_platform,
    in_ipynb,
    find_gui,
    make_figs,
)


class testMakeGridExpMult(unittest.TestCase):
    bot = 0.001
    top = 10.0
    N = 50

    def test_posititive_nesting(self):
        grid = make_grid_exp_mult(self.bot, self.top, self.N, timestonest=3)
        self.assertAlmostEqual(grid.size, self.N)
        self.assertAlmostEqual(grid[0], self.bot)
        self.assertAlmostEqual(grid[-1], self.top)

    def test_no_nesting(self):
        grid = make_grid_exp_mult(self.bot, self.top, self.N, timestonest=-1)
        self.assertAlmostEqual(grid.size, self.N)
        self.assertAlmostEqual(grid[0], self.bot)
        self.assertAlmostEqual(grid[-1], self.top)
        self.assertAlmostEqual(grid[1], (self.top - self.bot) / (self.N - 1) + self.bot)

    def test_log_nesting(self):
        grid = make_grid_exp_mult(self.bot, self.top, self.N, timestonest=0)
        self.assertAlmostEqual(grid.size, self.N)
        self.assertAlmostEqual(grid[0], self.bot)
        self.assertAlmostEqual(grid[-1], self.top)


class testGetLorenzShares(unittest.TestCase):
    data = 100.0 * np.random.rand(1000)

    def test_lorenz_default(self):
        out = get_lorenz_shares(self.data)
        self.assertAlmostEqual(out.size, 1)

    def test_lorenz_pctiles(self):
        out = get_lorenz_shares(self.data, percentiles=[0.2, 0.4, 0.6, 0.8])
        self.assertAlmostEqual(out.size, 4)
        self.assertTrue(np.all(out[1:] >= out[:-1]))

    def test_lorenz_sorteds(self):
        out = get_lorenz_shares(
            np.sort(self.data), percentiles=[0.2, 0.4], presorted=True
        )
        self.assertAlmostEqual(out.size, 2)
        self.assertTrue(np.all(out[1:] >= out[:-1]))


class testSubpopAvg(unittest.TestCase):
    def test_calc_subpop(self):
        my_data = 200 * np.random.rand(1000)
        my_ref = 5 * np.random.rand(1000)
        my_cuts = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        cond_avgs = calc_subpop_avg(my_data, my_ref, my_cuts)
        self.assertTrue(len(cond_avgs), len(my_cuts))


class testKernelRegression(unittest.TestCase):
    def setUp(self):
        f = np.sqrt
        X = 10.0 * np.random.rand(1000)
        err = 0.3 * np.random.rand(1000) - 0.15
        Y = f(X) + err
        self.X = X
        self.Y = Y

    def test_regression(self):
        g = kernel_regression(self.X, self.Y, h=0.25)
        Q = 10 * np.random.rand(100)
        out = g(Q)
        self.assertTrue(np.all(np.isreal(out)))


class testEnvironmentStuff(unittest.TestCase):
    def setUp(self):
        self.pf = determine_platform()

    def testInPyNB(self):
        out = in_ipynb()

    def testFindGUI(self):
        out = find_gui()


class testEtc(unittest.TestCase):
    def test_asset_grid(self):
        # Test linear asset grid
        params = {
            "aXtraMin": 0.0,
            "aXtraMax": 1.0,
            "aXtraCount": 5,
            "aXtraExtra": None,
            "aXtraNestFac": -1,
        }

        aXtraGrid = make_assets_grid(**params)
        test = np.unique(np.diff(aXtraGrid).round(decimals=3))
        self.assertEqual(test.size, 1)

    def test_make_figs(self):
        # Test the make_figs() function with a trivial output
        plt.figure()
        plt.plot(np.linspace(1, 5, 40), np.linspace(4, 8, 40))
        make_figs("test", True, False, target_dir="")
        plt.clf()
