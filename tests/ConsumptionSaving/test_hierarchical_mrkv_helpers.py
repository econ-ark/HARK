"""Tests for the hierarchical-Markov helper functions:
general-format support in make_hierarchical_mrkv_array and the new
extract_cond_mrkv_arrays inverse, plus simple-format invariance."""

import unittest

import numpy as np

import HARK.ConsumptionSaving.ConsAggIndMarkovModel as agg_ind_mrkv_module
from HARK.ConsumptionSaving.ConsAggIndMarkovModel import (
    AggIndMrkvConsumerType,
    extract_cond_mrkv_arrays,
    make_hierarchical_mrkv_array,
)
from HARK.ConsumptionSaving.ConsAggShockModel import KrusellSmithEconomy


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

    def test_extract_rejects_non_hierarchical_input(self):
        """A block that is not the macro probability times a row-stochastic
        matrix is rejected rather than silently returning rows that do not
        sum to one."""
        cond = [
            [self.micro_a, self.micro_b],
            [self.micro_b, self.micro_a],
        ]
        full = make_hierarchical_mrkv_array(self.macro, cond)
        full[0:2, 0:2] = np.array([[0.1, 0.2], [0.05, 0.3]])
        with self.assertRaises(ValueError) as cm:
            extract_cond_mrkv_arrays(full, self.macro, 2)
        self.assertIn("(0,0)", str(cm.exception))

    def test_extract_rejects_wrong_micro_state_count(self):
        """An N inconsistent with the array shape is rejected rather than
        silently slicing mis-sized blocks."""
        cond = [
            [self.micro_a, self.micro_b],
            [self.micro_b, self.micro_a],
        ]
        full = make_hierarchical_mrkv_array(self.macro, cond)
        for N_wrong in [1, 3]:
            with self.subTest(N=N_wrong):
                with self.assertRaises(ValueError) as cm:
                    extract_cond_mrkv_arrays(full, self.macro, N_wrong)
                self.assertIn("shape", str(cm.exception))


class testKSEconomyStoresHierarchicalPieces(unittest.TestCase):
    def test_make_mrkv_array_stores_macro_and_cond(self):
        economy = KrusellSmithEconomy()
        economy.make_MrkvArray()
        np.testing.assert_array_equal(economy.MacroMrkvArray, economy.MrkvAggArray)
        rebuilt = make_hierarchical_mrkv_array(
            economy.MacroMrkvArray, economy.CondMrkvArrays
        )
        np.testing.assert_allclose(rebuilt, economy.MrkvIndArray)


class testAggIndMrkvConsumerTypeBasics(unittest.TestCase):
    """Direct tests of the rewritten hierarchical class (non-shuffle)."""

    def test_from_combined_scalar_and_vector(self):
        a = AggIndMrkvConsumerType.__new__(AggIndMrkvConsumerType)
        a.num_micro_states = 3
        self.assertEqual(a.macro_from_combined(7), 2)
        self.assertIsInstance(a.macro_from_combined(7), int)
        self.assertEqual(a.micro_from_combined(7), 1)
        np.testing.assert_array_equal(
            a.macro_from_combined(np.array([2, 7])), np.array([0, 2])
        )
        np.testing.assert_array_equal(
            a.micro_from_combined(np.array([2, 7])), np.array([2, 1])
        )

    def test_old_name_is_gone(self):
        self.assertFalse(hasattr(agg_ind_mrkv_module, "AggIndMarkovConsumerType"))
