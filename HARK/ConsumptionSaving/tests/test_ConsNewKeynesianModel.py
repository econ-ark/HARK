import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsNewKeynesianModel import NewKeynesianConsumerType
from HARK.tests import HARK_PRECISION

jacobian_test_dict = {
    # Parameters shared with the perfect foresight model
    "CRRA": 2,  # Coefficient of relative risk aversion
    "Rfree": 1.04**0.25,  # Interest factor on assets
    "DiscFac": 0.9735,  # Intertemporal discount factor
    "LivPrb": [0.99375],  # Survival probability
    "PermGroFac": [1.00],  # Permanent income growth factor
    # Parameters that specify the income distribution over the lifecycle
    "PermShkStd": [
        0.06
    ],  # [(0.01*4/11)**0.5],    # Standard deviation of log permanent shocks to income
    "PermShkCount": 5,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.3],  # Standard deviation of log transitory shocks to income
    "TranShkCount": 5,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.07,  # Probability of unemployment while working
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate
    "UnempPrbRet": 0.0005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraCount": 48,  # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra": None,  # Additional values to add to aXtraGrid
    # A few other parameters
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool": True,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Preference shocks currently only compatible with linear cFunc
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    # Parameters only used in simulation
    "AgentCount": 500,  # Number of agents of this type
    "T_sim": 100,  # Number of periods to simulate
    "aNrmInitMean": np.log(1.3) - (0.5**2) / 2,  # Mean of log initial assets
    "aNrmInitStd": 0.5,  # Standard deviation of log initial assets
    "pLvlInitMean": 0,  # Mean of log initial permanent income
    "pLvlInitStd": 0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "T_age": None,  # Age after which simulated agents are automatically killed
    # Parameters for Transition Matrix Simulation
    "mMin": 0.001,
    "mMax": 20,
    "mCount": 48,
    "mFac": 3,
}


# %% Test Transition Matrix Methods


class test_Transition_Matrix_Methods(unittest.TestCase):
    def test_calc_tran_matrix(self):
        example1 = NewKeynesianConsumerType(**jacobian_test_dict)
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

        self.assertAlmostEqual(AggA[0], 1.19513, places=4)
        self.assertAlmostEqual(AggC[0], 1.00417, places=4)


# %% Test Heterogenous Agent Jacobian Methods


class test_Jacobian_methods(unittest.TestCase):
    def test_calc_jacobian(self):
        Agent = NewKeynesianConsumerType(**jacobian_test_dict)
        Agent.compute_steady_state()
        CJAC_Perm, AJAC_Perm = Agent.calc_jacobian("PermShkStd", 50)

        self.assertAlmostEqual(CJAC_Perm.T[30][29], -0.06120, places=HARK_PRECISION)
        self.assertAlmostEqual(CJAC_Perm.T[30][30], 0.05307, places=HARK_PRECISION)
        self.assertAlmostEqual(CJAC_Perm.T[30][31], 0.04674, places=HARK_PRECISION)
