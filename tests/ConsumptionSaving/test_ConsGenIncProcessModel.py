import unittest
from copy import copy

import numpy as np

from HARK.ConsumptionSaving.ConsGenIncProcessModel import (
    IndShockExplicitPermIncConsumerType,
    PersistentShockConsumerType,
)
from tests import HARK_PRECISION

GenIncDictionary = {
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability
    "AgentCount": 10000,  # Number of agents of this type (only matters for simulation)
    "kNrmInitMean": 0.0,  # Mean of log initial assets (only matters for simulation)
    "kNrmInitStd": 1.0,  # Standard deviation of log initial assets (only for simulation)
    "pLogInitMean": 0.0,  # Mean of log initial permanent income (only matters for simulation)
    "pLogInitStd": 0.4,  # Standard deviation of log initial permanent income (only matters for simulation)
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor (only matters for simulation)
    "T_age": None,  # Age after which simulated agents are automatically killed
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 30,  # Maximum end-of-period "assets above minimum" value
    # Some other value of "assets above minimum" to add to the grid
    "aXtraExtra": np.array([0.005, 0.01]),
    "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    # Parameters describing the income process
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
    "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
    "tax_rate": 0.0,  # Flat income tax rate
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "CubicBool": False,  # Use cubic spline interpolation when True, linear interpolation when False
    "vFuncBool": True,  # Whether to calculate the value function during solution
    # More parameters specific to "Explicit Permanent income" shock model
    "PermGroFac": [
        1.0
    ],  # Permanent income growth factor - long run permanent income growth doesn't work yet
}


class testIndShockExplicitPermIncConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = IndShockExplicitPermIncConsumerType(cycles=1, **GenIncDictionary)
        self.agent.solve()

    def test_solution(self):
        pLvlGrid = self.agent.pLvlGrid[0]
        self.assertAlmostEqual(
            self.agent.pLvlGrid[0][0], 0.28063, places=HARK_PRECISION
        )

        self.assertAlmostEqual(self.agent.solution[0].mLvlMin(pLvlGrid[0]), 0.0)

        self.assertAlmostEqual(
            self.agent.solution[0].cFunc(10, pLvlGrid[5]).tolist(),
            5.408106,
            places=HARK_PRECISION,
        )


class testPersistentShockConsumerType(unittest.TestCase):
    def setUp(self):
        # "persistent idiosyncratic shocks" model
        PrstIncCorr = 0.98  # Serial correlation coefficient for persistent income
        persistent_shocks = copy(GenIncDictionary)
        persistent_shocks["PrstIncCorr"] = PrstIncCorr

        # "persistent idisyncratic shocks" consumer
        self.agent = PersistentShockConsumerType(cycles=1, **persistent_shocks)
        self.agent.solve()

    def test_solution(self):
        pLvlGrid = self.agent.pLvlGrid[0]

        self.assertAlmostEqual(
            self.agent.solution[0].cFunc(10, pLvlGrid[1]).tolist(),
            5.27723,
            places=HARK_PRECISION,
        )

    def test_value(self):
        pLvlGrid = self.agent.pLvlGrid[0]

        self.assertTrue(self.agent.vFuncBool)
        self.assertAlmostEqual(
            self.agent.solution[0].vFunc(10, pLvlGrid[3]),
            -0.36683,
            places=HARK_PRECISION,
        )

    def test_simulation(self):
        self.agent.T_sim = 25

        self.agent.track_vars = ["aLvl", "mLvl", "cLvl", "pLvl"]
        self.agent.initialize_sim()
        self.agent.simulate()

    def test_aNrm_is_written_every_period(self):
        # aNrm is declared in state_vars but this model works in levels, so
        # only sim_birth used to write it. Continuing agents then reported
        # whatever the per-period blanking left in the buffer -- values like
        # 3.96e-319, i.e. subnormals read out of freed memory. Pin aNrm to
        # its definition so a regression shows up as a mismatch rather than
        # as plausible-looking noise.
        self.agent.T_sim = 10
        self.agent.AgentCount = 100
        self.agent.track_vars = ["aNrm", "aLvl", "pLvl"]
        self.agent.initialize_sim()
        self.agent.simulate()

        aNrm = self.agent.history["aNrm"]
        implied = self.agent.history["aLvl"] / self.agent.history["pLvl"]

        self.assertEqual(aNrm.size, 1000)
        self.assertTrue(np.all(np.isfinite(aNrm)))
        self.assertTrue(np.allclose(aNrm, implied, rtol=1e-12, atol=0.0))

    def test_cubic(self):
        CubicType = PersistentShockConsumerType(CubicBool=True)
        CubicType.solve()
        CubicType.unpack("cFunc")
        self.assertAlmostEqual(
            CubicType.cFunc[0](5.0, 2.0), 3.48222, places=HARK_PRECISION
        )

    def test_IH_constructors(self):
        self.agent.cycles = 0
        self.agent.construct()


class testLogUtilityValue(unittest.TestCase):
    def test_value_one_period_before_terminal(self):
        """
        With log utility, value one period before the end matches the Bellman equation.

        The model is solved in levels, where terminal value is log(mLvl), so value is
        log(c) + DiscFac * LivPrb * E[log(Rfree * a + pLvlNext * theta)].
        """
        for AgentType in (
            IndShockExplicitPermIncConsumerType,
            PersistentShockConsumerType,
        ):
            agent = AgentType(cycles=1, CRRA=1.0, vFuncBool=True)
            agent.solve()
            solution = agent.solution[0]
            PermShk, TranShk = agent.IncShkDstn[0].atoms
            beta = agent.DiscFac * agent.LivPrb[0]
            for pLvl in agent.pLvlGrid[0][[6, 13, 20]]:
                mLvl = solution.mLvlMin(pLvl) + pLvl * np.array([0.5, 2.0, 10.0])
                pLvls = np.full_like(mLvl, pLvl)
                c = solution.cFunc(mLvl, pLvls)
                pLvlNext = agent.pLvlNextFunc[0](pLvl) * PermShk
                mLvlNext = agent.Rfree[0] * (mLvl - c)[:, None] + pLvlNext * TranShk
                v = np.log(c) + beta * np.log(mLvlNext) @ agent.IncShkDstn[0].pmv
                np.testing.assert_allclose(
                    solution.vFunc(mLvl, pLvls), v, rtol=0, atol=1e-5
                )

    def test_value_without_income(self):
        """
        With no income (pLvl = 0) and log utility, value matches the exact solution.

        The consumer spends a share MPC_t of mLvl, where 1 / MPC_t = 1 + beta / MPC_t+1
        and MPC_T = 1, and carries the rest at Rfree. Three periods check that the
        solver extends this row from a value function other than the terminal one.
        """
        periods = 3
        LivPrb, Rfree = 0.98, 1.03
        agent = IndShockExplicitPermIncConsumerType(
            cycles=1,
            T_cycle=periods,
            CRRA=1.0,
            vFuncBool=True,
            PermShkStd=[0.1] * periods,
            TranShkStd=[0.1] * periods,
            PermGroFac=[1.0] * periods,
            Rfree=[Rfree] * periods,
            LivPrb=[LivPrb] * periods,
        )
        agent.solve()
        beta = agent.DiscFac * LivPrb
        MPC = [1.0]
        for _ in range(periods):
            MPC.insert(0, 1.0 / (1.0 + beta / MPC[0]))
        mLvlStart = np.array([0.5, 2.0, 10.0])
        for t in range(periods):
            mLvl, v, discount = mLvlStart.copy(), np.zeros(3), 1.0
            for s in range(t, periods + 1):
                c = MPC[s] * mLvl
                v += discount * np.log(c)
                mLvl, discount = Rfree * (mLvl - c), discount * beta
            vSolved = agent.solution[t].vFunc(mLvlStart, np.zeros(3))
            np.testing.assert_allclose(vSolved, v, rtol=1e-10)
