import unittest

import numpy as np

from HARK.mat_methods import mass_to_grid
from HARK.utilities import jump_to_grid_1D, jump_to_grid_2D


# Compare general mass_to_grid with jump_to_grid_1D
class TestMassToGrid1D(unittest.TestCase):
    def setUp(self):
        n_grid = 30
        n_points = 1000

        # Create 1d grid
        self.grid = np.linspace(0, 10.0, n_grid)
        # Simulate 1d points from a normal distribution
        # large variance to ensure some points are outside the grid
        self.points = np.random.normal(5, 20, n_points)
        # Create weights
        self.weights = np.random.uniform(0, 1, n_points)

    def test_compare_jump_to_grid_1d(self):
        # Compare mass_to_grid with jump_to_grid_1D
        res1 = mass_to_grid(self.points[..., np.newaxis], self.weights, (self.grid,))
        res2 = jump_to_grid_1D(self.points, self.weights, self.grid)

        # Compare results
        self.assertTrue(np.allclose(res1, res2))


# Compare general mass_to_grid with jump_to_grid_2D
class TestMassToGrid2D(unittest.TestCase):
    def setUp(self):
        n_grid = 30
        n_points = 1000

        # Create 2d grid
        self.x_grid = np.linspace(0.0, 10, n_grid)
        self.y_grid = np.linspace(0.0, 10, n_grid)

        # Simulate 2d points from a normal distribution
        # large variance to ensure some points are outside the grid
        mean = np.array([3, 4])
        vcov = np.array([[10.0, -0.5], [-0.5, 3.0]])
        self.points = np.random.multivariate_normal(mean, vcov, n_points)
        self.weights = np.repeat(1.3, n_points)

    def test_compare_jump_to_grid_2d(self):
        # Compare mass_to_grid with jump_to_grid_2D
        res1 = mass_to_grid(self.points, self.weights, (self.x_grid, self.y_grid))
        res2 = jump_to_grid_2D(
            self.points[:, 0], self.points[:, 1], self.weights, self.x_grid, self.y_grid
        )

        # Compare results
        self.assertTrue(np.allclose(res1, res2))


class Test3DMassToGrid(unittest.TestCase):
    def test_simple_3d(self):
        # 3d grid of 2 points in each dimension
        x_grid = np.array([0.0, 1.0])
        y_grid = np.array([0.0, 1.0])
        z_grid = np.array([0.0, 1.0])

        # Some points
        my_points = np.array(
            [
                [0.5, 0.5, 0.5],
                [3.0, 3.0, 3.0],
            ]
        )
        mass = np.array([1.0, 1.0])

        # Compare results
        grid_mass = mass_to_grid(my_points, mass, (x_grid, y_grid, z_grid))

        grid_mass = grid_mass.reshape((2, 2, 2))

        # Check the mass on the 8 points
        self.assertTrue(grid_mass[0, 0, 0] == 1 / 8)
        self.assertTrue(grid_mass[0, 0, 1] == 1 / 8)
        self.assertTrue(grid_mass[0, 1, 0] == 1 / 8)
        self.assertTrue(grid_mass[0, 1, 1] == 1 / 8)
        self.assertTrue(grid_mass[1, 0, 0] == 1 / 8)
        self.assertTrue(grid_mass[1, 0, 1] == 1 / 8)
        self.assertTrue(grid_mass[1, 1, 0] == 1 / 8)
        self.assertTrue(grid_mass[1, 1, 1] == (1 / 8 + 1.0))
