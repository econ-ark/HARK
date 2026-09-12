import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsSequentialPortfolioModel import (
    SequentialPortfolioConsumerType,
)


class testLogUtilityValue(unittest.TestCase):
    def test_value_one_period_before_terminal(self):
        """
        With log utility, value one period before the end matches the Bellman equation.

        At the chosen share, m' = Rport * a / (PermGroFac * psi) + theta, and
        continuation value is log(m') + log(PermGroFac * psi), where permanent
        income growth enters additively (issue #75).
        """
        agent = SequentialPortfolioConsumerType(cycles=1, CRRA=1.0, vFuncBool=True)
        agent.solve()
        solution = agent.solution[0]
        PermShk, TranShk, Risky = agent.ShockDstn[0].atoms
        m = np.array([0.5, 2.0, 10.0, 19.0])
        c = solution.cFuncAdj(m)
        Share = solution.ShareFuncAdj(m)[:, None]
        Rport = agent.Rfree[0] + Share * (Risky - agent.Rfree[0])
        growth = agent.PermGroFac[0] * PermShk
        mNext = Rport * (m - c)[:, None] / growth + TranShk
        vNext = (np.log(mNext) + np.log(growth)) @ agent.ShockDstn[0].pmv
        v = np.log(c) + agent.DiscFac * agent.LivPrb[0] * vNext
        np.testing.assert_allclose(solution.vFuncAdj(m), v, rtol=0, atol=1e-4)
