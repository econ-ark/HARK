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
        Agent.compute_steady_state()
        CJAC_Perm, AJAC_Perm = Agent.calc_jacobian("PermShkStd", 50)

        self.assertAlmostEqual(CJAC_Perm.T[30][29], -0.10503, places=HARK_PRECISION)
        self.assertAlmostEqual(CJAC_Perm.T[30][30], 0.10316, places=HARK_PRECISION)
        self.assertAlmostEqual(CJAC_Perm.T[30][31], 0.09059, places=HARK_PRECISION)
