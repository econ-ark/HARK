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
    files_in_dir,
    get_percentiles,
    NullFunc,
    apply_fun_to_vals,
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

        self.assertRaises(
            ValueError, get_lorenz_shares, self.data, percentiles=[0.2, 1.1]
        )

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


class testGetPercentiles(unittest.TestCase):
    def test_default(self):
        X = np.array([5.0])
        out = get_percentiles(X)
        self.assertTrue(np.isnan(out))

        Y = np.sort(np.random.rand(100))
        out = get_percentiles(Y, percentiles=[0.2, 0.5, 0.8], presorted=True)
        self.assertTrue(out.size == 3)

        self.assertRaises(ValueError, get_percentiles, Y, percentiles=[0.5, 1.1])


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

    def test_triangle(self):
        g = kernel_regression(self.X, self.Y, kernel="hat")
        Q = 10 * np.random.rand(100)
        out = g(Q)
        self.assertTrue(np.all(np.isreal(out)))

    def test_inextant(self):
        self.assertRaises(
            ValueError, kernel_regression, self.X, self.Y, h=0.25, kernel="hate"
        )


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

        params["aXtraNestFac"] = 2.5  # invalid
        self.assertRaises(ValueError, make_assets_grid, **params)

    def test_make_figs(self):
        # Test the make_figs() function with a trivial output
        plt.figure()
        plt.plot(np.linspace(1, 5, 40), np.linspace(4, 8, 40))
        make_figs("test", True, False, target_dir="")
        plt.clf()

    def test_files_in_dir(self):
        some_list = files_in_dir(".")

    def test_NullFunc(self):
        f = NullFunc()
        self.assertTrue(f() is None)
        self.assertTrue(np.isnan(f(5.0)))
        self.assertAlmostEqual(f.distance(f), 0.0)
        self.assertAlmostEqual(f.distance(np), 1000.0)
        self.assertAlmostEqual(f.distance(5), 1000.0)

    def test_apply_func_to_vals(self):
        def temp_func(x, y, z):
            return 3 * x + 4 * y + 2 * z

        W = np.random.rand(100)
        X = np.random.rand(100)
        Y = np.random.rand(100)
        Z = np.random.rand(100)

        my_dict = {
            "w": W,
            "x": X,
            "y": Y,
            "z": Z,
        }

        vals = temp_func(X, Y, Z)
        out = apply_fun_to_vals(temp_func, my_dict)
        self.assertTrue(np.all(vals == out))
