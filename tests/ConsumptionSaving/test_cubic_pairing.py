"""
The pairing rule: cubic Hermite interpolation of the consumption function is
refused when the expectation over the income shocks is a sum over discrete
atoms, as every HARK solver takes it but IndShock's and BufferStock's with
IncShkExpectation="continuous", and, under that expectation, when next period's
consumption function is piecewise linear. The demonstration solves with the
check disabled, which no user can do, and measures what the rule prevents.
"""

import unittest
from unittest import mock

import numpy as np
from scipy.optimize import brentq

from HARK.ConsumptionSaving.ConsBufferStockModel import BufferStockConsumerType
from HARK.ConsumptionSaving.ConsIndShockModel import (
    CUBIC_NEXT_PERIOD_MESSAGE,
    CUBIC_PAIRING_MESSAGE,
    IndShockConsumerType,
    KinkedRconsumerType,
    check_cubic_pairing,
)
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsPortfolioModel import PortfolioConsumerType
from HARK.interpolation import (
    CubicHermiteInterp,
    IdentityFunction,
    LinearInterp,
    LowerEnvelope,
)


def concavity_violations_above_the_kink(agent):
    """
    Solve the agent and inspect its unconstrained consumption function above this
    period's kink, where the lower envelope uses the cubic rather than the
    constrained line. Returns the number of segments wholly or partly above the
    kink, the number of those where the cubic violates concavity (its second
    derivative is positive at either end, computed exactly from the knots' values
    and slopes), how many of those segments hold an image of next period's kink (an
    asset level from which some atom lands exactly on it), and for linear
    interpolation the number of knots above the kink where the chord slopes rise.
    """
    agent.solve()
    sol = agent.solution[0]
    unc = sol.cFunc.functions[0]
    m, c = np.asarray(unc.x_list), np.asarray(unc.y_list)
    kink = brentq(lambda x: unc(x) - (x - sol.mNrmMin), m[1], m[-1])
    above = m[1:] > kink
    h, slope = np.diff(m), np.diff(c) / np.diff(m)
    if not agent.CubicBool:
        rises = np.diff(slope)[above[1:]] > 1e-12
        return int(above.sum()), 0, 0, int(rises.sum())
    s = np.asarray(unc.dydx_list)
    c2 = np.maximum(
        (6 * slope - 4 * s[:-1] - 2 * s[1:]) / h,
        (-6 * slope + 2 * s[:-1] + 4 * s[1:]) / h,
    )
    violating = np.flatnonzero((c2 > 1e-9) & above)
    psi, theta = agent.IncShkDstn[0].atoms
    images = (kink - theta) * agent.PermGroFac[0] * psi / agent.Rfree[0]
    a = m - c
    held = sum(((images > a[k]) & (images < a[k + 1])).any() for k in violating)
    return int(above.sum()), int(violating.size), int(held), 0


class testCubicPairingRule(unittest.TestCase):
    def test_refused_with_the_explanation(self):
        for make in (
            lambda: IndShockConsumerType(CubicBool=True),
            lambda: KinkedRconsumerType(CubicBool=True),
            lambda: MarkovConsumerType(CubicBool=True),
            lambda: BufferStockConsumerType(CubicBool=True, verbose=0),
        ):
            agent = make()
            with self.assertRaises(ValueError) as caught:
                agent.solve()
            self.assertEqual(str(caught.exception), CUBIC_PAIRING_MESSAGE)
        # The helper itself, and the pairing that is allowed
        with self.assertRaises(ValueError):
            check_cubic_pairing(True)
        check_cubic_pairing(True, continuous=True)
        check_cubic_pairing(False)

    def test_refused_before_any_period_is_solved(self):
        # check_restrictions, which pre_solve runs, refuses it where the solver
        # takes CubicBool, before the terminal period is built or any period solved
        for agent in (
            IndShockConsumerType(CubicBool=True),
            MarkovConsumerType(CubicBool=True),
        ):
            with self.assertRaises(ValueError) as caught:
                agent.check_restrictions()
            self.assertEqual(str(caught.exception), CUBIC_PAIRING_MESSAGE)
        IndShockConsumerType(
            CubicBool=True, IncShkExpectation="continuous"
        ).check_restrictions()
        # a type whose solver does not take CubicBool is not refused for it
        PortfolioConsumerType(CubicBool=True).check_restrictions()

    def test_allowed_with_the_continuous_expectation(self):
        agent = IndShockConsumerType(
            cycles=3, CubicBool=True, IncShkExpectation="continuous"
        )
        agent.solve()
        self.assertIsInstance(agent.solution[0].cFunc.functions[0], CubicHermiteInterp)

    def test_next_period_must_be_cubic(self):
        # Under the continuous expectation, next period's consumption function
        # must be cubic or a single line: a piecewise linear one is kinked at
        # each of its knots
        m = np.array([0.0, 1.0, 2.0, 4.0])
        c = np.array([0.0, 0.9, 1.6, 2.6])
        line = LinearInterp(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        for cFunc in (
            line,
            IdentityFunction(),
            CubicHermiteInterp(m, c, np.array([1.0, 0.8, 0.6, 0.4])),
            LowerEnvelope(
                CubicHermiteInterp(m, c, np.array([1.0, 0.8, 0.6, 0.4])), line
            ),
        ):
            check_cubic_pairing(True, continuous=True, cFunc_next=cFunc)
        for cFunc in (LinearInterp(m, c), LowerEnvelope(LinearInterp(m, c), line)):
            with self.assertRaises(ValueError) as caught:
                check_cubic_pairing(True, continuous=True, cFunc_next=cFunc)
            self.assertEqual(str(caught.exception), CUBIC_NEXT_PERIOD_MESSAGE)
        check_cubic_pairing(False, continuous=True, cFunc_next=LinearInterp(m, c))
        # in the solver: a cubic period before a linear one is refused
        linear = IndShockConsumerType(cycles=1, IncShkExpectation="continuous")
        linear.solve()
        agent = IndShockConsumerType(
            cycles=1, CubicBool=True, IncShkExpectation="continuous"
        )
        # solve() builds the terminal solution with its constructor
        agent.constructors = dict(
            agent.constructors, solution_terminal=lambda: linear.solution[0]
        )
        with self.assertRaises(ValueError) as caught:
            agent.solve()
        self.assertEqual(str(caught.exception), CUBIC_NEXT_PERIOD_MESSAGE)

    def test_linear_is_unaffected(self):
        agent = IndShockConsumerType(cycles=0)
        agent.solve()
        self.assertFalse(agent.CubicBool)
        self.assertAlmostEqual(agent.solution[0].cFunc(1.0), 0.86546, places=4)

    @mock.patch(
        "HARK.ConsumptionSaving.ConsIndShockModel.check_cubic_pairing",
        lambda *args, **kwargs: None,
    )
    def test_what_the_rule_prevents(self):
        # HARK's default model with the artificial constraint at zero, on a grid
        # fine enough to isolate the images of next period's kink: the cubic
        # violates concavity in segments holding an image, and the linear chords
        # never rise
        above, violating, held, _ = concavity_violations_above_the_kink(
            IndShockConsumerType(cycles=0, CubicBool=True, aXtraCount=200)
        )
        self.assertGreater(above, 100)
        self.assertGreaterEqual(violating, 1)
        self.assertEqual(held, violating)
        above, _, _, rises = concavity_violations_above_the_kink(
            IndShockConsumerType(cycles=0, CubicBool=False, aXtraCount=200)
        )
        self.assertGreater(above, 100)
        self.assertEqual(rises, 0)
        # The same with i.i.d. income on the default grid
        kwds = dict(PermShkStd=[0.0], PermShkCount=1, UnempPrb=0.0, TranShkStd=[0.3])
        above, violating, held, _ = concavity_violations_above_the_kink(
            IndShockConsumerType(cycles=0, CubicBool=True, **kwds)
        )
        self.assertGreaterEqual(violating, 1)
        self.assertEqual(held, violating)
        _, _, _, rises = concavity_violations_above_the_kink(
            IndShockConsumerType(cycles=0, CubicBool=False, **kwds)
        )
        self.assertEqual(rises, 0)
