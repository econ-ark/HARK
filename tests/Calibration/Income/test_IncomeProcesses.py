"""
Tests for the Markov income-process input validators.

These helpers exist solely to turn malformed input into a descriptive
Exception, and each one catches a specific tuple of exception types. Without
a test on the raise path, narrowing or widening that tuple is invisible: an
input whose attribute access raises outside the tuple propagates a raw
AttributeError instead of the intended message, and nothing notices.
"""

import unittest

import numpy as np

from HARK.Calibration.Income.IncomeProcesses import (
    _expand_expense_shock_schedule,
    _markov_unemployment_lists,
    _validate_markov_retirement,
    _validate_markov_state_dims,
    _validate_markov_unemployment_ndim,
)


class testValidateMarkovStateDims(unittest.TestCase):
    """K must agree across PermShkStd, TranShkStd, UnempPrb and IncUnemp."""

    def setUp(self):
        # Three periods, two discrete states.
        self.PermShkStd = np.zeros((3, 2))
        self.TranShkStd = np.zeros((3, 2))
        self.UnempPrb = np.zeros(2)
        self.IncUnemp = np.zeros(2)

    def test_consistent_dims_returns_K(self):
        K = _validate_markov_state_dims(
            self.PermShkStd, self.TranShkStd, self.UnempPrb, self.IncUnemp
        )
        self.assertEqual(K, 2)

    def test_mismatched_state_count_raises(self):
        with self.assertRaisesRegex(Exception, "number of discrete states"):
            _validate_markov_state_dims(
                self.PermShkStd, self.TranShkStd, np.zeros(3), self.IncUnemp
            )

    def test_non_array_input_raises_descriptive_error(self):
        """A float has no .shape; the AttributeError must be converted."""
        with self.assertRaisesRegex(Exception, "number of discrete states"):
            _validate_markov_state_dims(
                self.PermShkStd, self.TranShkStd, 0.05, self.IncUnemp
            )

    def test_one_dimensional_permshkstd_raises(self):
        """PermShkStd.shape[1] must exist; an IndexError must be converted."""
        with self.assertRaisesRegex(Exception, "number of discrete states"):
            _validate_markov_state_dims(
                np.zeros(3), self.TranShkStd, self.UnempPrb, self.IncUnemp
            )


class testValidateMarkovRetirement(unittest.TestCase):
    """Retirement arrays are checked only when T_retire is positive."""

    def test_no_retirement_skips_validation(self):
        """With T_retire <= 0 the retirement arrays are unused, so anything goes."""
        self.assertIsNone(_validate_markov_retirement(0, 2, None, None))
        self.assertIsNone(_validate_markov_retirement(-1, 2, "nonsense", None))

    def test_matching_sizes_pass(self):
        self.assertIsNone(_validate_markov_retirement(5, 2, np.zeros(2), np.zeros(2)))

    def test_wrong_size_raises(self):
        with self.assertRaisesRegex(Exception, "UnempPrbRet and IncUnempRet"):
            _validate_markov_retirement(5, 2, np.zeros(3), np.zeros(2))

    def test_non_array_raises(self):
        """A float has no .size; the AttributeError must be converted."""
        with self.assertRaisesRegex(Exception, "UnempPrbRet and IncUnempRet"):
            _validate_markov_retirement(5, 2, 0.05, np.zeros(2))


class testValidateMarkovUnemploymentNdim(unittest.TestCase):
    """All unemployment inputs must share one dimensionality, either 1 or 2."""

    def test_consistent_1d_returns_one(self):
        D = _validate_markov_unemployment_ndim(np.zeros(2), np.zeros(2), 0, None, None)
        self.assertEqual(D, 1)

    def test_consistent_2d_returns_two(self):
        D = _validate_markov_unemployment_ndim(
            np.zeros((3, 2)), np.zeros((3, 2)), 0, None, None
        )
        self.assertEqual(D, 2)

    def test_mixed_ndim_raises(self):
        with self.assertRaisesRegex(Exception, "2D arrays"):
            _validate_markov_unemployment_ndim(
                np.zeros((3, 2)), np.zeros(2), 0, None, None
            )

    def test_retirement_arrays_participate_when_retiring(self):
        """With T_retire > 0 the retirement arrays must match too."""
        with self.assertRaisesRegex(Exception, "2D arrays"):
            _validate_markov_unemployment_ndim(
                np.zeros(2), np.zeros(2), 5, np.zeros((3, 2)), np.zeros(2)
            )

    def test_three_dimensional_raises(self):
        with self.assertRaisesRegex(Exception, "1D or 2D arrays"):
            _validate_markov_unemployment_ndim(
                np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), 0, None, None
            )


class testMarkovUnemploymentLists(unittest.TestCase):
    """Per-period unemployment lists for one Markov state."""

    def test_two_dimensional_uses_state_column(self):
        """D == 2 means the inputs already carry an age profile per state."""
        UnempPrb = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        IncUnemp = np.array([[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]])
        prb, inc = _markov_unemployment_lists(
            2, 1, 0, 3, 0, UnempPrb, IncUnemp, None, None
        )
        self.assertEqual(prb, [0.2, 0.4, 0.6])
        self.assertEqual(inc, [1.2, 1.4, 1.6])

    def test_one_dimensional_without_retirement_repeats_scalar(self):
        prb, inc = _markov_unemployment_lists(
            1, 0, 0, 3, 0, np.array([0.1, 0.9]), np.array([1.1, 9.9]), None, None
        )
        self.assertEqual(prb, [0.1, 0.1, 0.1])
        self.assertEqual(inc, [1.1, 1.1, 1.1])

    def test_one_dimensional_with_retirement_appends_retired_values(self):
        prb, inc = _markov_unemployment_lists(
            1,
            0,
            2,
            2,
            3,
            np.array([0.1, 0.9]),
            np.array([1.1, 9.9]),
            np.array([0.5, 0.6]),
            np.array([5.5, 6.6]),
        )
        self.assertEqual(prb, [0.1, 0.1, 0.5, 0.5, 0.5])
        self.assertEqual(inc, [1.1, 1.1, 5.5, 5.5, 5.5])


class testExpandExpenseShockSchedule(unittest.TestCase):
    """Position-by-position layout of the expense shock schedule.

    The model-init tests assert only that TranShkDstn[0] and [30] have the
    expected atom counts. Every band has the same number of atoms, so
    reversing the band order, duplicating a band, or shifting the leading
    None block by one leaves those assertions green.
    """

    def setUp(self):
        # Distinguishable one-element bands so position errors are visible.
        self.groups = [[float(i)] for i in range(9)]
        self.out = _expand_expense_shock_schedule(self.groups)

    def test_total_length(self):
        # 50 leading None + 8 bands of 5 + a final band of 31.
        self.assertEqual(len(self.out), 50 + 8 * 5 + 31)
        self.assertEqual(len(self.out), 121)

    def test_leading_positions_are_none(self):
        self.assertTrue(all(x is None for x in self.out[:50]))
        self.assertIsNotNone(self.out[50])

    def test_band_order_is_preserved(self):
        """Band i must occupy positions 50 + 5i through 54 + 5i."""
        for i in range(8):
            for offset in range(5):
                np.testing.assert_array_equal(
                    self.out[50 + 5 * i + offset],
                    np.array([float(i)]),
                    err_msg=f"band {i}, offset {offset}",
                )

    def test_final_band_spans_31_positions(self):
        for pos in range(90, 121):
            np.testing.assert_array_equal(self.out[pos], np.array([8.0]))
