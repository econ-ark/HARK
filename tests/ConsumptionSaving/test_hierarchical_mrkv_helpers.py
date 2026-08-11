"""Tests for the hierarchical-Markov helper functions:
general-format support in make_hierarchical_mrkv_array and the new
extract_cond_mrkv_arrays inverse, plus simple-format invariance."""

import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsAggIndMarkovModel import (
    extract_cond_mrkv_arrays,
    make_hierarchical_mrkv_array,
)


class testHierarchicalMrkvHelpers(unittest.TestCase):
    def setUp(self):
        self.macro = np.array([[0.9, 0.1], [0.3, 0.7]])
        self.micro_a = np.array([[0.8, 0.2], [0.4, 0.6]])
        self.micro_b = np.array([[0.5, 0.5], [0.25, 0.75]])

    def test_simple_format_unchanged(self):
        """The pre-existing simple (destination-conditioned) format
        produces exactly the same matrix as before the generalization."""
        cond = [self.micro_a, self.micro_b]
        full = make_hierarchical_mrkv_array(self.macro, cond)
        expected = np.zeros((4, 4))
        for i in range(2):
            for j in range(2):
                expected[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = (
                    self.macro[i, j] * cond[j]
                )
        np.testing.assert_array_equal(full, expected)
        np.testing.assert_allclose(full.sum(axis=1), 1.0)

    def test_general_format(self):
        """The nested [i][j] (source-and-destination) format is detected
        and each block uses its own conditional matrix."""
        cond = [
            [self.micro_a, self.micro_b],
            [self.micro_b, self.micro_a],
        ]
        full = make_hierarchical_mrkv_array(self.macro, cond)
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(
                    full[2 * i : 2 * i + 2, 2 * j : 2 * j + 2],
                    self.macro[i, j] * cond[i][j],
                )
        np.testing.assert_allclose(full.sum(axis=1), 1.0)

    def test_extract_round_trip(self):
        """extract_cond_mrkv_arrays inverts make_hierarchical_mrkv_array."""
        cond = [
            [self.micro_a, self.micro_b],
            [self.micro_b, self.micro_a],
        ]
        full = make_hierarchical_mrkv_array(self.macro, cond)
        recovered = extract_cond_mrkv_arrays(full, self.macro, 2)
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(recovered[i][j], cond[i][j])

    def test_extract_zero_probability_block(self):
        """Blocks with zero macro probability come back as zeros."""
        macro = np.array([[1.0, 0.0], [0.5, 0.5]])
        cond = [self.micro_a, self.micro_b]
        full = make_hierarchical_mrkv_array(macro, cond)
        recovered = extract_cond_mrkv_arrays(full, macro, 2)
        np.testing.assert_array_equal(recovered[0][1], np.zeros((2, 2)))
        np.testing.assert_allclose(recovered[1][0], cond[0])


class testKSEconomyStoresHierarchicalPieces(unittest.TestCase):
    def test_make_mrkv_array_stores_macro_and_cond(self):
        from HARK.ConsumptionSaving.ConsAggShockModel import KrusellSmithEconomy

        economy = KrusellSmithEconomy()
        economy.make_MrkvArray()
        np.testing.assert_array_equal(economy.MacroMrkvArray, economy.MrkvAggArray)
        rebuilt = make_hierarchical_mrkv_array(
            economy.MacroMrkvArray, economy.CondMrkvArrays
        )
        np.testing.assert_allclose(rebuilt, economy.MrkvIndArray)
