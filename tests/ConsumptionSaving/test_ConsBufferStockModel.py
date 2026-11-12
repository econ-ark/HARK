import json
import unittest
import warnings

import numpy as np

from HARK.ConsumptionSaving.ConsBufferStockModel import (
    BufferStockConsumerType,
    calc_buffer_stock_conditions,
)
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.distributions import Lognormal, MeanOneLogNormal


def make_agent(**kwds):
    params = {"verbose": 0}
    params.update(kwds)
    return BufferStockConsumerType(**params)


class testBufferStockConditions(unittest.TestCase):
    """The paper's conditions at its calibration, against closed forms."""

    def setUp(self):
        self.agent = make_agent()
        self.agent.check_conditions()

    def test_closed_forms(self):
        bilt = self.agent.bilt
        self.assertAlmostEqual(bilt["APFac"], 0.999200, places=6)
        self.assertAlmostEqual(bilt["RPFac"], 0.960769, places=6)
        self.assertAlmostEqual(bilt["GPFacRaw"], 0.970097, places=6)
        self.assertAlmostEqual(bilt["FHWFac"], 0.990385, places=6)
        self.assertAlmostEqual(bilt["PFVAFac"], 0.932039, places=6)
        self.assertAlmostEqual(bilt["WRPFac"], 0.067937, places=6)
        self.assertAlmostEqual(bilt["MPCmin"], 0.039231, places=6)
        self.assertAlmostEqual(bilt["MPCmax"], 0.932063, places=6)
        self.assertEqual(bilt["WorstPrb"], 0.005)
        results = bilt["BufferStockConditions"]
        self.assertAlmostEqual(results.hNrm, 103.0, places=9)
        self.assertEqual(results.BoroCnstNat, 0.0)

    def test_verdicts_at_the_calibration(self):
        results = self.agent.bilt["BufferStockConditions"]
        self.assertEqual(results.existence, "established")
        self.assertIn("thm:convgtobellman", results.existence_basis)
        self.assertTrue(results.target_guaranteed)
        self.assertTrue(results.pseudo_target_guaranteed)
        self.assertFalse(self.agent.degenerate)
        self.assertTrue(all(self.agent.conditions.values()))

    def test_moments_come_from_the_law(self):
        # With the truncation far out, the moments are the lognormal's
        agent = make_agent(PermShk_inf=-12.0, PermShk_sup=12.0)
        agent.check_conditions()
        self.assertAlmostEqual(agent.bilt["GPFacMod"], 0.979846, places=6)
        self.assertAlmostEqual(agent.bilt["VAFac"], 0.941406, places=6)
        # They do not move with the number of points
        for count in (3, 7, 31):
            other = make_agent(PermShkCount=count, TranShkCount=count)
            other.check_conditions()
            self.assertEqual(other.bilt["GPFacMod"], self.agent.bilt["GPFacMod"])

    def test_function_matches_the_agent(self):
        a = self.agent
        results = calc_buffer_stock_conditions(
            a.Rfree[0],
            a.DiscFac,
            a.CRRA,
            a.PermGroFac[0],
            a.UnempPrb,
            a.IncUnemp,
            a.PermShkLaw,
            a.TranShkLaw,
        )
        self.assertEqual(results, a.bilt["BufferStockConditions"])
        json.dumps(results.to_dict())


class testBufferStockRestrictions(unittest.TestCase):
    """The class refuses what is outside the paper's model."""

    def assertRefused(self, fragment, **kwds):
        with self.assertRaises(ValueError) as caught:
            make_agent(**kwds).check_restrictions()
        self.assertIn(fragment, str(caught.exception))

    def test_preferences_and_returns(self):
        self.assertRefused("CRRA must exceed one", CRRA=1.0)
        self.assertRefused("CRRA must exceed one", CRRA=0.5)
        self.assertRefused("DiscFac must be positive", DiscFac=0.0)
        self.assertRefused("Rfree must be positive", Rfree=[0.0])
        self.assertRefused("PermGroFac must be positive", PermGroFac=[0.0])

    def test_scope(self):
        self.assertRefused("no mortality", LivPrb=[0.98])
        self.assertRefused("on_no_solution", on_no_solution="ignore")

    def test_probability_of_the_zero_income_event(self):
        self.assertRefused("strictly between zero and one", UnempPrb=0.0)
        self.assertRefused("strictly between zero and one", UnempPrb=1.0)
        # Zero needs an artificial constraint that binds, and the message says so
        self.assertRefused("artificial borrowing constraint that binds", UnempPrb=0.0)
        make_agent(UnempPrb=0.0, BoroCnstArt=0.0).check_restrictions()

    def test_an_artificial_constraint_that_binds(self):
        # At most zero
        self.assertRefused("must not be above zero", BoroCnstArt=0.5)
        self.assertRefused("must not be above zero", UnempPrb=0.0, BoroCnstArt=0.5)
        # The zero-income event, if it can happen, is still the worst outcome
        self.assertRefused(
            "at least zero and below one", UnempPrb=1.0, IncUnemp=0.3, BoroCnstArt=0.0
        )
        self.assertRefused(
            "below the worst employed income", IncUnemp=0.95, BoroCnstArt=0.0
        )
        make_agent(IncUnemp=0.3, BoroCnstArt=0.0).check_restrictions()

    def test_a_natural_constraint_with_no_finite_value(self):
        # With positive income in the event and G times the permanent shock's
        # infimum at or above R, no borrowing is ruled out by the worst path
        params = {
            "IncUnemp": 0.3,
            "PermShkStd": 0.01,
            "Rfree": [0.98],
            "PermGroFac": [1.02],
            "DiscFac": 1.0,
        }
        self.assertRefused("no finite value", **params)
        # An artificial constraint is then the one in force
        agent = make_agent(BoroCnstArt=0.0, **params)
        agent.check_restrictions()
        agent.solve()
        self.assertEqual(agent.bilt["BufferStockConditions"].BoroCnstNat, -np.inf)
        self.assertEqual(agent.solution[0].mNrmMin, 0.0)

    def assertLawRefused(self, fragment, **laws):
        # A law of the user's own is assigned after construction
        agent = make_agent()
        for name, law in laws.items():
            setattr(agent, name, law)
        agent.update_income_process()
        with self.assertRaises(ValueError) as caught:
            agent.check_restrictions()
        self.assertIn(fragment, str(caught.exception))

    def test_laws(self):
        self.assertLawRefused(
            "supremum must be finite", PermShkLaw=MeanOneLogNormal(0.1)
        )
        self.assertLawRefused(
            "infimum must be positive", TranShkLaw=MeanOneLogNormal(0.1)
        )
        self.assertLawRefused("mean must be one", TranShkLaw=Lognormal(0.0, 0.1))

    def test_income_in_the_zero_income_event(self):
        agent = make_agent()
        inf = agent.TranShkLaw.infimum[0]
        bound = inf / (1.0 - agent.UnempPrb * (1.0 - inf))
        self.assertRefused("must not be negative", IncUnemp=-0.1)
        self.assertRefused("below the worst employed income", IncUnemp=bound)
        self.assertRefused("below the worst employed income", IncUnemp=0.95)
        make_agent(IncUnemp=0.99 * bound).check_restrictions()

    def test_every_violation_is_named(self):
        with self.assertRaises(ValueError) as caught:
            make_agent(CRRA=0.5, UnempPrb=0.0, LivPrb=[0.9]).check_restrictions()
        message = str(caught.exception)
        for fragment in ("CRRA", "UnempPrb", "LivPrb"):
            self.assertIn(fragment, message)

    def test_a_point_law_is_accepted(self):
        agent = make_agent(PermShkStd=0.0)
        agent.check_restrictions()
        agent.check_conditions()
        self.assertEqual(agent.bilt["GPFacMod"], agent.bilt["GPFacRaw"])


class testBufferStockSolution(unittest.TestCase):
    """The solution, its stable points, and its warnings."""

    def test_same_solution_as_IndShockConsumerType(self):
        # Before the restart at the limits, the solve is IndShockConsumerType's
        agent = make_agent()
        agent.solve(postsolve=False)
        other = IndShockConsumerType(
            cycles=0,
            CRRA=2.0,
            Rfree=[1.04],
            DiscFac=0.96,
            LivPrb=[1.0],
            PermGroFac=[1.03],
            BoroCnstArt=None,
            verbose=0,
        )
        other.IncShkDstn = agent.IncShkDstn
        other.PermShkDstn = agent.PermShkDstn
        other.TranShkDstn = agent.TranShkDstn
        other.aXtraGrid = agent.aXtraGrid
        other.solve(postsolve=False)
        m = np.linspace(0.01, 30.0, 500)
        np.testing.assert_array_equal(
            agent.solution[0].cFunc(m), other.solution[0].cFunc(m)
        )

    def test_stable_points_at_the_papers_calibration(self):
        # The paper's calibration with seven equiprobable points of untruncated
        # lognormals, 400 asset points up to 100, cubic interpolation. The values
        # were computed independently with IndShockConsumerType. Truncating at 12
        # standard deviations reproduces the untruncated discretization.
        agent = make_agent(
            PermShk_inf=-12.0,
            PermShk_sup=12.0,
            TranShk_inf=-12.0,
            TranShk_sup=12.0,
            aXtraCount=400,
            aXtraMax=100.0,
            CubicBool=True,
            tolerance=1e-10,
        )
        agent.solve()
        self.assertAlmostEqual(agent.bilt["mNrmStE"], 1.37732, places=5)
        self.assertAlmostEqual(agent.bilt["mNrmTrg"], 1.39103, places=5)

    def test_stable_points(self):
        agent = make_agent()
        agent.solve()
        cFunc, bilt = agent.solution[0].cFunc, agent.bilt
        m_trg, m_bal = bilt["mNrmTrg"], bilt["mNrmStE"]
        self.assertEqual(len(bilt["mNrmTrg_roots"]), 1)
        self.assertEqual(len(bilt["mNrmStE_roots"]), 1)
        self.assertLess(m_bal, m_trg)
        # Each solves its equation
        RNrm = agent.Rfree[0] / agent.PermGroFac[0]
        psi = agent.PermShkDstn[0]
        Ex_inv = np.dot(psi.pmv, 1.0 / np.ravel(psi.atoms))
        self.assertAlmostEqual(
            RNrm * Ex_inv * (m_trg - cFunc(m_trg)) + 1.0, m_trg, places=9
        )
        self.assertAlmostEqual(RNrm * (m_bal - cFunc(m_bal)) + 1.0, m_bal, places=9)
        self.assertEqual(agent.solution[0].mNrmTrg, m_trg)

    def test_no_target_guaranteed(self):
        # The paper's example: four times the variance of the permanent shocks
        agent = make_agent(PermShkStd=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            agent.solve()
        results = agent.bilt["BufferStockConditions"]
        self.assertFalse(results.conditions["GICMod"].holds)
        self.assertTrue(results.conditions["GICRaw"].holds)
        self.assertTrue(np.isnan(agent.bilt["mNrmTrg"]))
        self.assertNotEqual(len(agent.bilt["mNrmTrg_roots"]), 1)
        self.assertFalse(np.isnan(agent.bilt["mNrmStE"]))

    def test_warns_when_the_wric_fails(self):
        params = {"Rfree": [1.04], "DiscFac": 1.5, "UnempPrb": 0.7}
        agent = make_agent(**params)
        agent.check_conditions()
        self.assertEqual(agent.bilt["BufferStockConditions"].existence, "fails")
        self.assertTrue(agent.degenerate)
        with self.assertWarns(UserWarning):
            agent._respond_to_conditions()
        with self.assertRaises(ValueError):
            make_agent(on_no_solution="raise", **params).solve()

    def test_warns_when_existence_is_not_established(self):
        agent = make_agent(PermGroFac=[0.95])
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertTrue(results.conditions["WRIC"].holds)
        self.assertFalse(results.conditions["FVAC"].holds)
        self.assertEqual(results.existence, "not established")
        with self.assertWarns(UserWarning):
            agent._respond_to_conditions()
        # quiet silences it, and raise does not apply
        quiet = make_agent(PermGroFac=[0.95], quiet=True, on_no_solution="raise")
        quiet.check_conditions()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            quiet._respond_to_conditions()

    def test_positive_income_in_the_zero_income_event(self):
        agent = make_agent(IncUnemp=0.3)
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertEqual(results.WorstPrb, 0.0)
        self.assertEqual(results.MPCmax, 1.0)
        self.assertEqual(len(results.notes), 1)
        self.assertLess(results.BoroCnstNat, 0.0)

    def test_nothing_is_claimed_beyond_the_papers_theorems(self):
        # The paper proves its theorems for zero income in the zero-income event.
        # With positive income there, every condition holds at the calibration,
        # and still nothing is established or guaranteed
        agent = make_agent(IncUnemp=0.3)
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertTrue(all(agent.conditions.values()))
        self.assertEqual(results.existence, "not established")
        self.assertIn("zero income in the zero-income event", results.existence_basis)
        self.assertFalse(results.target_guaranteed)
        self.assertFalse(results.pseudo_target_guaranteed)
        self.assertFalse(agent.degenerate)
        for name in ("WRIC", "FVAC", "GICRaw", "GICMod"):
            self.assertIn("a hypothesis of", results.conditions[name].meaning)
        self.assertIn(results.existence_basis, agent.bilt["conditions_report"])
        # Nothing fails, so there is no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            agent._respond_to_conditions()
        # The same parameters with zero income there are the paper's model
        paper = make_agent()
        paper.check_conditions()
        self.assertNotIn(
            "a hypothesis of",
            paper.bilt["BufferStockConditions"].conditions["GICMod"].meaning,
        )
        # When the FVAC fails as well, the class warns
        agent = make_agent(IncUnemp=0.3, PermGroFac=[0.95])
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertEqual(results.existence, "not established")
        self.assertIn("The FVAC", results.existence_basis)
        with self.assertWarns(UserWarning):
            agent._respond_to_conditions()

    def test_a_constraint_that_cannot_bind(self):
        base = make_agent()
        base.solve()
        m = np.linspace(0.01, 30.0, 500)
        for BoroCnstArt in (-1.0, 0.0):
            agent = make_agent(BoroCnstArt=BoroCnstArt)
            self.assertEqual(agent.constraint_scenario(), "natural")
            agent.solve()
            self.assertEqual(agent.bilt["BufferStockConditions"].scenario, "natural")
            np.testing.assert_allclose(
                agent.solution[0].cFunc(m), base.solution[0].cFunc(m), rtol=1e-12
            )
            self.assertEqual(agent.bilt["mNrmTrg"], base.bilt["mNrmTrg"])

    def test_limits_are_exact_above_the_grid(self):
        # Started at their limits, the limits do not move, and the consumption
        # function above the grid converges as fast as on it
        agent = make_agent(tolerance=1e-8)
        agent.solve()
        results = agent.bilt["BufferStockConditions"]
        self.assertAlmostEqual(agent.solution[0].hNrm, results.hNrm, places=10)
        self.assertAlmostEqual(agent.solution[0].MPCmin, results.MPCmin, places=12)
        self.assertAlmostEqual(agent.solution[0].MPCmax, results.MPCmax, places=12)
        tight = make_agent(tolerance=1e-12)
        tight.solve()
        m = np.linspace(25.0, 60.0, 100)  # above the grid
        np.testing.assert_allclose(
            agent.solution[0].cFunc(m), tight.solution[0].cFunc(m), atol=1e-6
        )

    def test_natural_constraint_of_the_laws(self):
        agent = make_agent(IncUnemp=0.3)
        agent.solve()
        results = agent.bilt["BufferStockConditions"]
        self.assertAlmostEqual(agent.solution[0].mNrmMin, results.BoroCnstNat, places=8)
        self.assertEqual(agent.solution[0].MPCmax, 1.0)
        self.assertEqual(results.MPCmax, 1.0)
        # No theorem of the paper guarantees a target here: the stable points
        # found are listed, and none is recorded as the target
        self.assertTrue(np.isnan(agent.bilt["mNrmTrg"]))
        self.assertTrue(np.isnan(agent.bilt["mNrmStE"]))
        self.assertEqual(len(agent.bilt["mNrmTrg_roots"]), 1)
        self.assertEqual(len(agent.bilt["mNrmStE_roots"]), 1)

    def test_a_point_mass_at_the_worst_income(self):
        # With positive income in the event and laws whose infima are atoms, the
        # worst event has positive probability and the MPC at the bottom is
        # below one, in the solution as in the record
        agent = make_agent(IncUnemp=0.05)
        agent.PermShkLaw = MeanOneLogNormal(0.1).discretize(7, method="equiprobable")
        agent.TranShkLaw = MeanOneLogNormal(0.1).discretize(7, method="equiprobable")
        agent.update_income_process()
        agent.solve()
        results = agent.bilt["BufferStockConditions"]
        self.assertAlmostEqual(results.WorstPrb, 0.005 / 7.0, places=12)
        RPFac = agent.bilt["RPFac"]
        self.assertAlmostEqual(
            results.MPCmax, 1.0 - (0.005 / 7.0) ** 0.5 * RPFac, places=12
        )
        self.assertLess(results.MPCmax, 1.0)
        self.assertAlmostEqual(agent.solution[0].MPCmax, results.MPCmax, places=8)
        self.assertAlmostEqual(
            agent.solution[0].mNrmMin, results.BoroCnstNat, places=10
        )

    def test_note_when_the_perfect_foresight_limit_is_not_finite(self):
        agent = make_agent(Rfree=[0.98], DiscFac=1.0, PermGroFac=[1.02])
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertEqual(results.existence, "established")
        self.assertIn("RIC and FHWC failing", results.notes[0])

    def test_converges_when_the_perfect_foresight_limit_is_not_finite(self):
        # The RIC fails: the solve converges, on the grid and above it
        params = {"Rfree": [0.98], "DiscFac": 1.0, "PermGroFac": [1.02]}
        loose = make_agent(tolerance=1e-8, **params)
        loose.solve()
        tight = make_agent(tolerance=1e-12, **params)
        tight.solve()
        self.assertLess(tight.completed_cycles, 5000)
        m = np.linspace(0.01, 60.0, 600)
        c = loose.solution[0].cFunc(m)
        np.testing.assert_allclose(c, tight.solution[0].cFunc(m), atol=1e-6)
        self.assertTrue(np.all(np.diff(c, 2) <= 1e-12))

    def test_finite_horizon(self):
        agent = make_agent(cycles=10)
        agent.solve()
        self.assertEqual(len(agent.solution), 11)
        self.assertNotIn("mNrmTrg", agent.bilt)


class testBufferStockArtificialConstraint(unittest.TestCase):
    """An artificial borrowing constraint that binds."""

    def test_scenario(self):
        # With zero income in the zero-income event the natural constraint is
        # zero, and an artificial one at zero cannot bind
        self.assertEqual(make_agent(BoroCnstArt=0.0).constraint_scenario(), "natural")
        # It binds where the worst income is positive
        for kwds in ({"UnempPrb": 0.0}, {"IncUnemp": 0.3}):
            agent = make_agent(BoroCnstArt=0.0, **kwds)
            self.assertEqual(agent.constraint_scenario(), "artificial")
        # and only above the natural constraint
        agent = make_agent(IncUnemp=0.3)
        natural = agent.get_buffer_stock_conditions().BoroCnstNat
        self.assertLess(natural, 0.0)
        below = make_agent(IncUnemp=0.3, BoroCnstArt=natural - 0.1)
        self.assertEqual(below.constraint_scenario(), "natural")
        above = make_agent(IncUnemp=0.3, BoroCnstArt=natural + 0.1)
        self.assertEqual(above.constraint_scenario(), "artificial")

    def test_the_worst_income_follows_the_parameters(self):
        agent = make_agent()
        self.assertEqual(agent.WorstInc, 0.0)
        agent.IncUnemp = 0.3
        agent.update_income_process()
        self.assertEqual(agent.WorstInc, 0.3)
        agent.UnempPrb = 0.0
        agent.update_income_process()
        self.assertEqual(agent.WorstInc, agent.TranShkLaw.infimum[0])
        # With no zero-income event the discretization has no point for it
        IncShkDstn = agent.IncShkDstn[0]
        self.assertTrue(np.all(IncShkDstn.pmv > 0.0))
        self.assertGreater(np.min(IncShkDstn.atoms[1]), agent.WorstInc)
        self.assertAlmostEqual(np.dot(IncShkDstn.pmv, IncShkDstn.atoms[1]), 1.0)

    def test_nothing_is_claimed_under_uncertainty(self):
        # The paper proves no theorem where a liquidity constraint binds under
        # uncertainty: every condition holds, and nothing is established
        agent = make_agent(UnempPrb=0.0, BoroCnstArt=0.0)
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertTrue(all(agent.conditions.values()))
        self.assertEqual(results.scenario, "artificial")
        self.assertEqual(results.existence, "not established")
        self.assertIn("Ma, Stachurski and Toda", results.existence_basis)
        self.assertFalse(results.target_guaranteed)
        self.assertFalse(results.pseudo_target_guaranteed)
        self.assertEqual(results.MPCmax, 1.0)
        self.assertEqual(results.WorstInc, agent.TranShkLaw.infimum[0])
        self.assertLess(results.BoroCnstNat, 0.0)
        for name in ("FVAC", "GICRaw", "GICMod"):
            self.assertIn(
                "not proved where an artificial", results.conditions[name].meaning
            )
        self.assertIn("binds", results.notes[-1])
        # Nothing fails, so there is no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            agent._respond_to_conditions()
        # The paper treats no limit other than zero
        other = make_agent(IncUnemp=0.3, BoroCnstArt=-0.3)
        basis = other.get_buffer_stock_conditions().existence_basis
        self.assertIn("no borrowing limit other than zero", basis)
        # When the FVAC fails, the class warns
        agent = make_agent(UnempPrb=0.0, BoroCnstArt=0.0, PermGroFac=[0.95])
        agent.check_conditions()
        self.assertFalse(agent.conditions["FVAC"])
        with self.assertWarns(UserWarning):
            agent._respond_to_conditions()

    def test_same_solution_as_IndShockConsumerType(self):
        # Before the restart at the limits, the solve is IndShockConsumerType's
        # with the same constraint
        agent = make_agent(UnempPrb=0.0, BoroCnstArt=0.0)
        agent.solve(postsolve=False)
        other = IndShockConsumerType(
            cycles=0,
            CRRA=2.0,
            Rfree=[1.04],
            DiscFac=0.96,
            LivPrb=[1.0],
            PermGroFac=[1.03],
            BoroCnstArt=0.0,
            verbose=0,
        )
        other.IncShkDstn = agent.IncShkDstn
        other.PermShkDstn = agent.PermShkDstn
        other.TranShkDstn = agent.TranShkDstn
        other.aXtraGrid = agent.aXtraGrid
        other.solve(postsolve=False)
        m = np.linspace(0.01, 30.0, 500)
        np.testing.assert_array_equal(
            agent.solution[0].cFunc(m), other.solution[0].cFunc(m)
        )

    def test_the_constraint_binds(self):
        for kwds, floor in (
            ({"UnempPrb": 0.0, "BoroCnstArt": 0.0}, 0.0),
            ({"IncUnemp": 0.3, "BoroCnstArt": -0.3}, -0.3),
        ):
            agent = make_agent(**kwds)
            agent.solve()
            solution = agent.solution[0]
            cFunc = solution.cFunc
            self.assertEqual(solution.mNrmMin, floor)
            self.assertEqual(solution.MPCmax, 1.0)
            # Near the bottom the consumer ends the period at the constraint
            m = floor + np.linspace(0.01, 0.3, 30)
            np.testing.assert_allclose(cFunc(m), m - floor, rtol=1e-12)
            # and saves above it further up
            self.assertLess(cFunc(floor + 3.0), 3.0)
            m = np.linspace(floor + 0.01, 30.0, 3000)
            self.assertTrue(np.all(np.diff(cFunc(m), 2) <= 1e-12))
            # No theorem of the paper guarantees a target: the stable points
            # found are listed, and none is recorded as the target
            self.assertTrue(np.isnan(agent.bilt["mNrmTrg"]))
            self.assertTrue(np.isnan(agent.bilt["mNrmStE"]))
            self.assertEqual(len(agent.bilt["mNrmTrg_roots"]), 1)
            self.assertEqual(len(agent.bilt["mNrmStE_roots"]), 1)

    def test_perfect_foresight_against_the_papers_solution(self):
        # prop:PFCExist, and the constrained solution of subsec:PFCon
        pf = {
            "PermShkStd": 0.0,
            "TranShkStd": 0.0,
            "UnempPrb": 0.0,
            "BoroCnstArt": 0.0,
        }

        def factors(agent):
            R, G = agent.Rfree[0], agent.PermGroFac[0]
            APFac = (R * agent.DiscFac) ** (1.0 / agent.CRRA)
            return R, G, APFac

        # The RIC and the GICRaw hold: c(m) = m up to m = 1, then a kink n
        # periods before the constraint binds, at the paper's points
        agent = make_agent(tolerance=1e-10, **pf)
        agent.solve()
        results = agent.bilt["BufferStockConditions"]
        self.assertEqual(results.existence, "established")
        self.assertIn("prop:PFCExist", results.existence_basis)
        self.assertFalse(results.target_guaranteed)
        R, G, APFac = factors(agent)
        RPFac, GPFac, RNrm = APFac / R, APFac / G, R / G
        cFunc = agent.solution[0].cFunc
        self.assertAlmostEqual(float(cFunc(0.5)), 0.5, places=12)
        for n in range(6):
            c_n = GPFac ** (-n)
            b_n = c_n * (1.0 - RPFac ** (n + 1)) / (1.0 - RPFac) - (
                1.0 - RNrm ** (-(n + 1))
            ) / (1.0 - 1.0 / RNrm)
            self.assertAlmostEqual(float(cFunc(b_n + 1.0)), c_n, delta=2e-3)
        # Market resources settle at one, where the constraint binds
        self.assertEqual(len(agent.bilt["mNrmTrg_roots"]), 1)
        self.assertAlmostEqual(agent.bilt["mNrmTrg_roots"][0], 1.0, places=9)

        # The GICRaw fails and the RIC holds: the unconstrained solution, cut
        # off by the constraint
        agent = make_agent(
            Rfree=[1.04], DiscFac=0.98, PermGroFac=[0.99], tolerance=1e-10, **pf
        )
        agent.solve()
        results = agent.bilt["BufferStockConditions"]
        self.assertEqual(results.existence, "established")
        self.assertFalse(results.conditions["GICRaw"].holds)
        R, G, APFac = factors(agent)
        MPCmin = 1.0 - APFac / R
        hNrm = 1.0 / (1.0 - G / R)  # the paper's h, which counts current income
        m = np.array([0.3, 0.59, 1.0, 2.0, 10.0])
        np.testing.assert_allclose(
            agent.solution[0].cFunc(m),
            np.minimum(m, (m - 1.0 + hNrm) * MPCmin),
            atol=1e-8,
        )
        self.assertEqual(agent.bilt["mNrmTrg_roots"], [])

        # The RIC fails and the GICRaw holds, the paper's example
        agent = make_agent(Rfree=[0.98], DiscFac=1.0, PermGroFac=[0.99], **pf)
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertFalse(results.conditions["RIC"].holds)
        self.assertEqual(results.existence, "established")
        self.assertIn("though the RIC fails", results.existence_basis)

        # Both fail: the limiting solution is degenerate
        params = dict(Rfree=[0.98], DiscFac=1.0, PermGroFac=[0.97], **pf)
        agent = make_agent(**params)
        agent.check_conditions()
        self.assertEqual(agent.bilt["BufferStockConditions"].existence, "fails")
        self.assertTrue(agent.degenerate)
        with self.assertWarns(UserWarning):
            agent._respond_to_conditions()
        with self.assertRaises(ValueError):
            make_agent(on_no_solution="raise", **params).solve()
