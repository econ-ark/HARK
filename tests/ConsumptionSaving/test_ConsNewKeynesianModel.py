import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsNewKeynesianModel import NewKeynesianConsumerType
from tests import HARK_PRECISION


# %% Test Transition Matrix Methods

# Uses class default values


class test_Transition_Matrix_Methods(unittest.TestCase):
    def test_calc_tran_matrix(self):
        example1 = NewKeynesianConsumerType()
        example1.cycles = 0
        example1.solve()

        example1.define_distribution_grid()
        p = example1.dist_pGrid  # Grid of permanent income levels

        example1.calc_transition_matrix()
        c = example1.cPol_Grid  # Normalized Consumption Policy Grid
        asset = example1.aPol_Grid  # Normalized Asset Policy Grid

        example1.calc_ergodic_dist()
        vecDstn = example1.vec_erg_dstn
        # Distribution of market resources and permanent income as a vector (m*p)x1 vector where

        # Compute Aggregate Consumption and Aggregate Assets
        gridc = np.zeros((len(c), len(p)))
        grida = np.zeros((len(asset), len(p)))

        for j in range(len(p)):
            gridc[:, j] = p[j] * c  # unnormalized Consumption policy grid
            grida[:, j] = p[j] * asset  # unnormalized Asset policy grid

        AggC = np.dot(gridc.flatten(), vecDstn)  # Aggregate Consumption
        AggA = np.dot(grida.flatten(), vecDstn)  # Aggregate Assets

        self.assertAlmostEqual(AggA[0], 0.82983, places=4)
        self.assertAlmostEqual(AggC[0], 1.00780, places=4)


# %% Test Heterogenous Agent Jacobian Methods


class test_Jacobian_methods(unittest.TestCase):
    def test_calc_jacobian(self):
        Agent = NewKeynesianConsumerType()
        Agent.compute_pe_steady_state()
        CJAC_Perm, AJAC_Perm = Agent.calc_jacobian("PermShkStd", 50)

        self.assertAlmostEqual(CJAC_Perm.T[30][29], -0.10503, places=HARK_PRECISION)
        self.assertAlmostEqual(CJAC_Perm.T[30][30], 0.10316, places=HARK_PRECISION)
        self.assertAlmostEqual(CJAC_Perm.T[30][31], 0.09059, places=HARK_PRECISION)


class test_assign_dist_mGrid(unittest.TestCase):
    """Branches of _assign_dist_mGrid that the transition-matrix tests miss.

    Every existing caller invokes define_distribution_grid() with no
    arguments, so dist_mGrid is None and m_density is 0 -- the prespecified
    grid branch and the densification loop never run.
    """

    def setUp(self):
        self.agent = NewKeynesianConsumerType()
        self.agent.cycles = 0

    def test_prespecified_grid_is_used_directly(self):
        my_grid = np.array([0.1, 0.5, 1.0, 2.0])
        self.agent.define_distribution_grid(dist_mGrid=my_grid)
        np.testing.assert_array_equal(self.agent.dist_mGrid, my_grid)

    def test_m_density_inserts_midpoints(self):
        self.agent.define_distribution_grid(m_density=0)
        base = self.agent.dist_mGrid.copy()

        self.agent.define_distribution_grid(m_density=1)
        dense = self.agent.dist_mGrid

        # One densification pass adds a midpoint per existing gridpoint.
        self.assertEqual(len(dense), 2 * len(base))
        self.assertTrue(np.all(np.diff(dense) >= 0.0), "grid must stay sorted")
        # Every original point survives densification.
        self.assertTrue(np.all(np.isin(base, dense)))


class test_NeutralMeasureWithGrowth(unittest.TestCase):
    """
    The neutral measure under mortality with permanent income growth: growth enters
    the normalized transition, survivors carry mass LivPrb*PermGroFac/PermGroFacAgg,
    newborns the complement. Checked against the model-file Simulator on the same
    primitives.
    """

    def _agents(self):
        from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType

        A = IndShockConsumerType(
            cycles=0, tolerance=1e-12
        )  # LivPrb 0.98, PermGroFac 1.01
        A.solve()
        params = dict(
            cycles=0,
            CRRA=A.CRRA,
            DiscFac=A.DiscFac,
            Rfree=[float(A.Rfree[0])],
            LivPrb=[float(A.LivPrb[0])],
            PermGroFac=[float(A.PermGroFac[0])],
            PermShkStd=list(A.PermShkStd),
            TranShkStd=list(A.TranShkStd),
            UnempPrb=A.UnempPrb,
            IncUnemp=A.IncUnemp,
            PermShkCount=A.PermShkCount,
            TranShkCount=A.TranShkCount,
            tolerance=1e-12,
            aXtraMax=A.aXtraMax,
            aXtraCount=A.aXtraCount,
        )
        N = NewKeynesianConsumerType(**params)
        N.solve()
        N.neutral_measure = True
        N.construct("IncShkDstn", "TranShkDstn", "PermShkDstn")
        return A, N

    def test_matches_simulator_with_growth(self):
        from HARK.utilities import make_grid_exp_mult

        A, N = self._agents()
        N.define_distribution_grid(dist_mGrid=make_grid_exp_mult(1e-4, 100.0, 1000, 3))
        N.calc_transition_matrix()
        N.calc_ergodic_dist()
        D = np.asarray(N.vec_erg_dstn).ravel()
        c_nk = float(np.dot(D, np.asarray(N.cPol_Grid).ravel()))
        a_nk = float(np.dot(D, np.asarray(N.aPol_Grid).ravel()))
        A.initialize_sym()
        X = A._simulator
        X.make_transition_matrices(
            {
                "kNrm": {"min": 0.0, "max": 100.0, "N": 1001, "nest": 3},
                "cNrm": {"min": 0.0, "max": 6.0, "N": 601},
            },
            norm="G",
        )
        X.find_steady_state()
        c_sim = X.get_long_run_average("cNrm")
        a_sim = X.get_long_run_average("aNrm")
        # residual: this class places newborns at m = 1 rather than at their first income draw
        self.assertLess(abs(c_nk / c_sim - 1.0), 5e-4)
        self.assertLess(abs(a_nk / a_sim - 1.0), 5e-3)

    def test_trend_option_and_error_path(self):
        A, N = self._agents()
        G = float(N.PermGroFac[0])
        N.define_distribution_grid()
        N.calc_transition_matrix()
        T_fix = np.array(N.tran_matrix)
        N.PermGroFacAgg = G  # all growth is a trend that newborns inherit
        N.calc_transition_matrix()
        T_trend = np.array(N.tran_matrix)
        self.assertFalse(np.allclose(T_fix, T_trend))
        self.assertTrue(np.all(np.isclose(np.sum(T_fix, axis=0), 1.0)))
        N.PermGroFacAgg = 1.0
        N.LivPrb = [0.995]
        with self.assertRaises(ValueError):
            N.calc_transition_matrix()
