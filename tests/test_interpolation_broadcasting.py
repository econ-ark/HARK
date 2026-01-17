"""
Test for array broadcasting in 2D interpolation classes.

This test verifies that HARKinterpolator2D methods correctly handle
mixed scalar/array inputs, e.g., func(array, scalar).

Bug: Prior to this fix, calling a 2D interpolator with (array, scalar)
inputs would fail with IndexError because xa.flatten() and ya.flatten()
produced arrays of different lengths.

Fix: Use np.broadcast_arrays(xa, ya) before flattening.
"""

import numpy as np
import pytest
from HARK.interpolation import (
    LinearInterp,
    BilinearInterp,
    LinearInterpOnInterp1D,
    VariableLowerBoundFunc2D,
    LowerEnvelope2D,
)


class TestInterpolation2DBroadcasting:
    """Test that 2D interpolation handles mixed scalar/array inputs."""

    def test_linear_interp_on_interp1d_array_scalar(self):
        """
        Minimal example demonstrating the bug with LinearInterpOnInterp1D.

        This test fails without the broadcast fix:
        IndexError: boolean index did not match indexed array along dimension 0
        """
        y_list = np.array([0.8, 1.0, 1.2])
        x_interps = [
            LinearInterp([0, 1, 2, 5], [0, 0.8, 1.6, 4.0]),
            LinearInterp([0, 1, 2, 5], [0, 1.0, 2.0, 5.0]),
            LinearInterp([0, 1, 2, 5], [0, 1.2, 2.4, 6.0]),
        ]
        lioi = LinearInterpOnInterp1D(x_interps, y_list)

        x = np.array([1.0, 2.0])

        # This should work: both inputs are arrays of same length
        result_array = lioi(x, np.array([1.0, 1.0]))

        # This was failing with IndexError before the fix
        result_scalar = lioi(x, 1.0)

        np.testing.assert_array_almost_equal(result_scalar, result_array)
        np.testing.assert_array_almost_equal(result_scalar, [1.0, 2.0])

    def test_variable_lower_bound_func2d_array_scalar(self):
        """Test VariableLowerBoundFunc2D with (array, scalar) inputs."""
        y_list = np.array([0.8, 1.0, 1.2])
        x_interps = [
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
        ]
        lioi = LinearInterpOnInterp1D(x_interps, y_list)
        lower_bound = LinearInterp(y_list, np.array([0.0, 0.0, 0.0]))
        vlbf = VariableLowerBoundFunc2D(lioi, lower_bound)

        x = np.array([1.0, 2.0, 5.0])

        result_scalar = vlbf(x, 1.0)
        result_array = vlbf(x, np.array([1.0, 1.0, 1.0]))

        np.testing.assert_array_almost_equal(result_scalar, result_array)

    def test_lower_envelope2d_nested_array_scalar(self):
        """Test LowerEnvelope2D with nested interpolators and (array, scalar)."""
        # Build complex nested structure like consumption-saving solvers
        y_list = np.array([0.8, 1.0, 1.2])
        x_interps = [
            LinearInterp([0, 1, 2, 5, 10], [0, 0.8, 1.6, 4.0, 8.0]),
            LinearInterp([0, 1, 2, 5, 10], [0, 1.0, 2.0, 5.0, 10.0]),
            LinearInterp([0, 1, 2, 5, 10], [0, 1.2, 2.4, 6.0, 12.0]),
        ]
        lioi = LinearInterpOnInterp1D(x_interps, y_list)
        lower_bound = LinearInterp(y_list, np.array([0.0, 0.0, 0.0]))
        vlbf = VariableLowerBoundFunc2D(lioi, lower_bound)

        x_list = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        y_list_bi = np.array([0.8, 1.0, 1.2])
        z_array = np.outer(x_list, y_list_bi)
        bi = BilinearInterp(z_array, x_list, y_list_bi)

        env = LowerEnvelope2D(vlbf, bi)

        x = np.array([1.0, 2.0, 5.0])

        result_scalar = env(x, 1.0)
        result_array = env(x, np.array([1.0, 1.0, 1.0]))

        np.testing.assert_array_almost_equal(result_scalar, result_array)

    def test_derivative_x_array_scalar(self):
        """Test derivativeX with (array, scalar) inputs."""
        y_list = np.array([0.8, 1.0, 1.2])
        x_interps = [
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
        ]
        lioi = LinearInterpOnInterp1D(x_interps, y_list)

        x = np.array([1.0, 2.0])

        dx_scalar = lioi.derivativeX(x, 1.0)
        dx_array = lioi.derivativeX(x, np.array([1.0, 1.0]))

        np.testing.assert_array_almost_equal(dx_scalar, dx_array)

    def test_derivative_y_array_scalar(self):
        """Test derivativeY with (array, scalar) inputs."""
        y_list = np.array([0.8, 1.0, 1.2])
        x_interps = [
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
        ]
        lioi = LinearInterpOnInterp1D(x_interps, y_list)

        x = np.array([1.0, 2.0])

        dy_scalar = lioi.derivativeY(x, 1.0)
        dy_array = lioi.derivativeY(x, np.array([1.0, 1.0]))

        np.testing.assert_array_almost_equal(dy_scalar, dy_array)

    def test_scalar_array_reversed(self):
        """Test (scalar, array) inputs - reversed order."""
        y_list = np.array([0.8, 1.0, 1.2])
        x_interps = [
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
            LinearInterp([0, 1, 2, 5], [0, 1, 2, 5]),
        ]
        lioi = LinearInterpOnInterp1D(x_interps, y_list)

        y = np.array([0.8, 1.0, 1.2])

        result_scalar = lioi(2.0, y)
        result_array = lioi(np.array([2.0, 2.0, 2.0]), y)

        np.testing.assert_array_almost_equal(result_scalar, result_array)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
