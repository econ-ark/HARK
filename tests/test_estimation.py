import numpy as np
from HARK.estimation import (
    bootstrap_sample_from_data,
    minimize_nelder_mead,
    minimize_powell,
)


class TestEstimation:
    def test_bootstrap_sample_from_data(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        new_data = bootstrap_sample_from_data(data, seed=0)

        # Check that the new data has the same shape
        assert data.shape == new_data.shape

        # Check that the new data is a sample of the original data
        for row in new_data:
            assert row in data

    def test_minimize_nelder_mead(self):
        def objective_func(x):
            return x[0] ** 2 + x[1] ** 2

        parameter_guess = [1.0, 1.0]
        result = minimize_nelder_mead(objective_func, parameter_guess)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-4)

    def test_minimize_powell(self):
        def objective_func(x):
            return x[0] ** 2 + x[1] ** 2

        parameter_guess = [1.0, 1.0]
        result = minimize_powell(objective_func, parameter_guess)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-4)

    def test_save_load_nelder_mead(self):
        import os
        from HARK.estimation import save_nelder_mead_data, load_nelder_mead_data

        simplex = np.array([[1.0, 2.0], [3.0, 4.0]])
        fvals = np.array([5.0, 6.0])
        iters = 10
        evals = 20
        name = "test_nm_data"

        save_nelder_mead_data(name, simplex, fvals, iters, evals)

        # Now, load the data back
        simplex_loaded, fvals_loaded, iters_loaded, evals_loaded = (
            load_nelder_mead_data(name)
        )

        np.testing.assert_allclose(simplex, simplex_loaded)
        np.testing.assert_allclose(fvals, fvals_loaded)
        assert iters == iters_loaded
        assert evals == evals_loaded

        # Clean up the test file
        os.remove(name + ".txt")
