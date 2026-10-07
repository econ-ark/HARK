"""
Created on Tue Jan 26 10:06:51 2021

@author: Mateo
"""

import unittest
from copy import copy

import numpy as np

from HARK.ConsumptionSaving.ConsRiskyContribModel import (
    RiskyContribConsumerType,
    init_risky_contrib,
    RiskyContribRebSolution,
    RiskyContribShaSolution,
    RiskyContribCnsSolution,
)
from tests import HARK_PRECISION


class testSolutionClasses(unittest.TestCase):
    def test_null_Reb(self):
        soln = RiskyContribRebSolution()

    def test_null_Sha(self):
        soln = RiskyContribShaSolution()

    def test_null_Cns(self):
        soln = RiskyContribCnsSolution()


class test_(unittest.TestCase):
    def setUp(self):
        # A set of finite parameters
        self.par_finite = init_risky_contrib.copy()

        # Four period model
        self.par_finite["PermGroFac"] = [2.0, 1.0, 0.1, 1.0]
        self.par_finite["PermShkStd"] = [0.1, 0.1, 0.0, 0.0]
        self.par_finite["TranShkStd"] = [0.2, 0.2, 0.0, 0.0]
        self.par_finite["AdjustPrb"] = [0.5, 0.5, 1.0, 1.0]
        self.par_finite["WithdrawTax"] = [0.1, 0.1, 0.0, 0.0]
        self.par_finite["LivPrb"] = [1.0, 1.0, 1.0, 1.0]
        self.par_finite["T_cycle"] = 4
        self.par_finite["T_retire"] = 0
        self.par_finite["T_age"] = 4
        self.par_finite["T_sim"] = 20
        self.par_finite["AgentCount"] = 100

        # Adjust discounting and returns distribution so that they make sense in a
        # 4-period model
        self.par_finite["DiscFac"] = 0.95**15
        self.par_finite["Rfree"] = 4 * [1.03**15]
        self.par_finite["RiskyAvg"] = 1.08**15  # Average return of the risky asset
        self.par_finite["RiskyStd"] = 0.0621376926532 * np.sqrt(
            15
        )  # Standard deviation of (log) risky returns

    def test_finite_cont_share(self):
        # Finite horizon with continuous contribution share
        cont_params = copy(self.par_finite)
        cont_params["DiscreteShareBool"] = False
        cont_params["vFuncBool"] = False

        fin_cont_agent = RiskyContribConsumerType(**cont_params)
        self.agent = fin_cont_agent

        # Independent solver
        fin_cont_agent.solve()
        self.assertAlmostEqual(
            fin_cont_agent.solution[0].stage_sols["Reb"].dfracFunc_Adj(3.0, 4.0),
            -0.87671,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            fin_cont_agent.solution[0].stage_sols["Sha"].ShareFunc_Adj(5.0, 0.1),
            0.14641,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            fin_cont_agent.solution[0].stage_sols["Cns"].cFunc(3.0, 4.0, 0.1),
            2.45609,
            places=HARK_PRECISION,
        )

        # Test simulation
        fin_cont_agent.track_vars = ["cNrm", "Share", "aNrm"]
        fin_cont_agent.initialize_sim()
        fin_cont_agent.simulate()

        # This type and DualMeasureMixin are the two overrides of
        # sim_one_period in HARK, and both reach _sim_period_prologue and
        # _sim_period_epilogue rather than inlining them. This side of that
        # contract is checked here: without these assertions a reordered or
        # dropped stage in either helper leaves the test green.
        T_sim = self.par_finite["T_sim"]
        AgentCount = self.par_finite["AgentCount"]
        for var in ["cNrm", "Share", "aNrm"]:
            self.assertEqual(
                fin_cont_agent.history[var].shape, (T_sim, AgentCount), var
            )
            self.assertTrue(np.all(np.isfinite(fin_cont_agent.history[var])), var)

        # Consumption must be strictly positive for every simulated agent.
        self.assertTrue(np.all(fin_cont_agent.history["cNrm"] > 0.0))

        # The epilogue advances t_age and t_cycle and wraps t_cycle at T_cycle.
        self.assertTrue(np.all(fin_cont_agent.t_cycle < fin_cont_agent.T_cycle))
        self.assertTrue(np.all(fin_cont_agent.t_age >= 0))
        self.assertTrue(np.all(fin_cont_agent.t_age <= self.par_finite["T_age"]))

        # General correlated solver
        fin_cont_agent.joint_dist_solver = True
        fin_cont_agent.solve()
        self.assertAlmostEqual(
            fin_cont_agent.solution[0].stage_sols["Reb"].dfracFunc_Adj(3, 4),
            -0.87849,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            fin_cont_agent.solution[0].stage_sols["Sha"].ShareFunc_Adj(5, 0.1),
            0.10658,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            fin_cont_agent.solution[0].stage_sols["Cns"].cFunc(3, 4, 0.1),
            2.45610,
            places=HARK_PRECISION,
        )

    def test_finite_cont_share_value(self):
        # Continuous share with AdjustPrb < 1 builds the share-stage value function,
        # which at grid nodes is consumption-stage value at the optimal share
        cont_params = copy(self.par_finite)
        cont_params["DiscreteShareBool"] = False
        cont_params["vFuncBool"] = True
        agent = RiskyContribConsumerType(**cont_params)
        agent.solve()
        Sha = agent.solution[0].stage_sols["Sha"]
        Cns = agent.solution[0].stage_sols["Cns"]
        mNrm, nNrm = (
            x.ravel()
            for x in np.meshgrid(agent.mNrmGrid[[3, 10, 20]], agent.nNrmGrid[[3, 10]])
        )
        Share = Sha.ShareFunc_Adj(mNrm, nNrm)
        np.testing.assert_allclose(
            Sha.vFunc_Adj(mNrm, nNrm), Cns.vFunc(mNrm, nNrm, Share), rtol=1e-12
        )

    def test_finite_disc_share(self):
        # Finite horizon with discrete contribution share
        disc_params = copy(self.par_finite)
        disc_params["DiscreteShareBool"] = True
        disc_params["vFuncBool"] = True

        fin_disc_agent = RiskyContribConsumerType(**disc_params)

        # Independent solver
        fin_disc_agent.solve()

        self.assertAlmostEqual(
            fin_disc_agent.solution[0].stage_sols["Reb"].dfracFunc_Adj(3.0, 4.0),
            -0.8767603,
        )
        self.assertAlmostEqual(
            fin_disc_agent.solution[0].stage_sols["Sha"].ShareFunc_Adj(5.0, 0.1), 0.1
        )
        self.assertAlmostEqual(
            fin_disc_agent.solution[0].stage_sols["Cns"].cFunc(3.0, 4.0, 0.1),
            2.45609,
            places=HARK_PRECISION,
        )

        # General correlated solver
        fin_disc_agent.joint_dist_solver = True
        fin_disc_agent.solve()

        self.assertAlmostEqual(
            fin_disc_agent.solution[0].stage_sols["Reb"].dfracFunc_Adj(3, 4),
            -0.87846,
            places=HARK_PRECISION,
        )
        self.assertAlmostEqual(
            fin_disc_agent.solution[0].stage_sols["Sha"].ShareFunc_Adj(5, 0.1), 0.1
        )
        self.assertAlmostEqual(
            fin_disc_agent.solution[0].stage_sols["Cns"].cFunc(3, 4, 0.1),
            2.45610,
            places=HARK_PRECISION,
        )


class testLogUtilityValue(unittest.TestCase):
    def test_value_one_period_before_terminal(self):
        """
        With log utility, consumption-stage value one period before the end matches
        the Bellman equation.

        The terminal agent withdraws everything and consumes m' + n' / (1 + WithdrawTax),
        so continuation value is the log of that plus log(PermGroFac * psi), where
        permanent income growth enters additively (issue #75). The check uses grid
        nodes of (m, n, Share), where the consumption function is not interpolated.
        """
        params = copy(init_risky_contrib)
        params.update(
            {
                "CRRA": 1.0,
                "vFuncBool": True,
                "cycles": 1,
                "T_cycle": 1,
                "T_age": None,
                "T_retire": 0,
                "AdjustPrb": [1.0],
                "WithdrawTax": [0.1],
                "LivPrb": [0.98],
                "Rfree": [1.03],
                "PermGroFac": [1.01],
                "PermShkStd": [0.1],
                "TranShkStd": [0.1],
            }
        )
        agent = RiskyContribConsumerType(**params)
        agent.solve()
        solution = agent.solution[0].stage_sols["Cns"]
        ShkDstn = agent.ShockDstn[0]
        PermShk, TranShk, Risky = ShkDstn.atoms
        mNrm, nNrm, Share = (
            x.ravel()
            for x in np.meshgrid(
                agent.mNrmGrid[[5, 10, 20, 30]],
                agent.nNrmGrid[[5, 15]],
                agent.ShareGrid[[0, agent.ShareGrid.size // 2]],
                indexing="ij",
            )
        )
        c = solution.cFunc(mNrm, nNrm, Share)
        growth = agent.PermGroFac[0] * PermShk
        mNext = agent.Rfree[0] * (mNrm - c)[:, None] / growth
        mNext += (1.0 - Share[:, None]) * TranShk
        nNext = Risky * nNrm[:, None] / growth + Share[:, None] * TranShk
        cNext = mNext + nNext / (1.0 + agent.WithdrawTax[0])
        vNext = (np.log(cNext) + np.log(growth)) @ ShkDstn.pmv
        v = np.log(c) + agent.DiscFac * agent.LivPrb[0] * vNext
        vSolved = solution.vFunc(mNrm, nNrm, Share)
        np.testing.assert_allclose(vSolved, v, rtol=0, atol=5e-4)
