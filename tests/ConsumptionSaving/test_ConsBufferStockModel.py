import json
import unittest
import warnings

import numpy as np

from HARK.ConsumptionSaving.ConsBufferStockModel import (
    BufferStockConsumerType,
    PointMassAtInfimum,
    calc_buffer_stock_conditions,
    paper_url,
)
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.distributions import DiscreteDistribution, Lognormal, MeanOneLogNormal


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

    def test_links_to_the_paper(self):
        # Each label is an anchor of the paper's web edition, in lower case with a
        # hyphen for the colon
        self.assertEqual(
            paper_url("thm:existencePositiveMinimum"),
            "https://llorracc.github.io/BufferStockTheory/#thm-existencepositiveminimum",
        )
        self.assertIn("paper_url", self.agent.bilt["conditions_report"])

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
            aXtraNestFac=3,
            aXtraExtra=None,
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
        # The permanent shock's default law then has a point mass at its infimum,
        # so that its lowest value has positive probability
        # (ass:shocksPositiveMinimum)
        truncated = make_agent().PermShkLaw
        self.assertNotIsInstance(truncated, PointMassAtInfimum)
        agent = make_agent(IncUnemp=0.3)
        law = agent.PermShkLaw
        self.assertIsInstance(law, PointMassAtInfimum)
        # The lowest value is the truncated lognormal's infimum, with the mass
        # the truncation removes below it; the mean is still one
        self.assertEqual(law.infimum[0], truncated.infimum[0])
        self.assertAlmostEqual(law.infimum_prb, 0.0013498980316301, places=12)
        self.assertGreater(law.scale * truncated.infimum[0], law.infimum[0])
        agent.check_restrictions()
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertEqual(results.WorstInc, 0.3)
        self.assertAlmostEqual(results.WorstPrb, 0.005 * law.infimum_prb, places=15)
        self.assertAlmostEqual(
            results.MPCmax,
            1.0 - (0.005 * law.infimum_prb) ** 0.5 * agent.bilt["RPFac"],
            places=12,
        )
        # Minimum human wealth (eq:FiniteMinHumanWealth)
        x = 1.03 * law.infimum[0] / 1.04
        self.assertAlmostEqual(results.BoroCnstNat, -0.3 * x / (1.0 - x), places=12)
        self.assertEqual(len(results.notes), 1)
        self.assertIn("lemma:NaturalBorrowingLimit", results.notes[0])

    def test_lowest_values_do_not_move_with_the_number_of_points(self):
        # What belongs to the laws (the lowest values, their probabilities, the
        # borrowing limit, the MPC at the bottom) is the same at every number of
        # points of the discretization
        records = []
        for count in (3, 7, 15):
            agent = make_agent(IncUnemp=0.3, PermShkCount=count, TranShkCount=count)
            agent.check_conditions()
            PermShkDstn = agent.PermShkDstn[0]
            self.assertEqual(PermShkDstn.atoms.size, count + 1)
            self.assertEqual(np.min(PermShkDstn.atoms), agent.PermShkLaw.infimum[0])
            self.assertAlmostEqual(np.dot(PermShkDstn.pmv, PermShkDstn.atoms[0]), 1.0)
            results = agent.bilt["BufferStockConditions"]
            records.append(
                (results.BoroCnstNat, results.WorstPrb, results.MPCmax, results.hNrm)
            )
        self.assertEqual(records[0], records[1])
        self.assertEqual(records[1], records[2])
        # The probability of the lowest value is a parameter
        agent = make_agent(IncUnemp=0.3, PermShkMinPrb=0.1)
        agent.check_restrictions()
        self.assertEqual(agent.PermShkLaw.infimum_prb, 0.1)
        self.assertAlmostEqual(agent.get_buffer_stock_conditions().WorstPrb, 0.0005)
        for prb in (-0.1, 1.0):
            with self.assertRaises(ValueError) as caught:
                make_agent(IncUnemp=0.3, PermShkMinPrb=prb).check_restrictions()
            self.assertIn("PermShkMinPrb", str(caught.exception))
        # Zero gives the lognormal itself, without a point mass
        agent = make_agent(IncUnemp=0.3, PermShkMinPrb=0.0)
        agent.check_restrictions()
        self.assertNotIsInstance(agent.PermShkLaw, PointMassAtInfimum)
        self.assertEqual(agent.get_buffer_stock_conditions().WorstPrb, 0.0)

    def test_the_appendix_on_a_positive_minimum_of_income(self):
        # With positive income in the zero-income event the verdicts rest on the
        # paper's appendix (sec:ApndxPositiveMinimumIncome)
        agent = make_agent(IncUnemp=0.3)
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertTrue(all(agent.conditions.values()))
        self.assertEqual(results.existence, "established")
        self.assertIn("thm:existencePositiveMinimum", results.existence_basis)
        self.assertTrue(results.target_guaranteed)
        self.assertTrue(results.pseudo_target_guaranteed)
        self.assertFalse(agent.degenerate)
        self.assertIn(results.existence_basis, agent.bilt["conditions_report"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            agent._respond_to_conditions()
        # When the FVAC fails the theorem establishes nothing, and the class warns
        agent = make_agent(IncUnemp=0.3, PermGroFac=[0.95])
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertFalse(results.conditions["FVAC"].holds)
        self.assertEqual(results.existence, "not established")
        self.assertIn("thm:existencePositiveMinimum", results.existence_basis)
        self.assertFalse(results.target_guaranteed)
        with self.assertWarns(UserWarning):
            agent._respond_to_conditions()

    def test_positive_income_when_the_hypotheses_fail(self):
        # subsec:PositiveMinimumFailures, at extreme parameters and a permanent
        # shock on two points
        PermShkLaw = DiscreteDistribution(np.array([0.5, 0.5]), np.array([0.9, 1.1]))
        TranShkLaw = DiscreteDistribution(np.array([1.0]), np.array([1.0]))

        def verdict(PermGroFac):
            return calc_buffer_stock_conditions(
                0.05, 0.99, 2.0, PermGroFac, 0.9, 0.3, PermShkLaw, TranShkLaw
            )

        # The WRIC fails with the probability of the worst event, and minimum
        # human wealth is finite: the limiting solution is degenerate, which
        # with IncUnemp above zero needs the FVAC to fail
        # (remark:WRICWorstImplied)
        results = verdict(0.04)
        self.assertAlmostEqual(results.WorstPrb, 0.45)
        self.assertFalse(results.conditions["WRIC"].holds)
        self.assertFalse(results.conditions["FVAC"].holds)
        self.assertTrue(np.isfinite(results.BoroCnstNat))
        self.assertEqual(results.existence, "fails")
        self.assertIn("subsec:PositiveMinimumFailures", results.existence_basis)
        self.assertIn("remark:WRICWorstImplied", results.existence_basis)
        self.assertEqual(results.MPCmax, 0.0)
        # Minimum human wealth not finite, and WorstPrb^(1/ρ) Þ above PermGroFac
        # times the lowest permanent shock: consumption goes to zero
        # (prop:InfiniteMinHumanWealth, part ii)
        results = verdict(0.06)
        self.assertEqual(results.BoroCnstNat, -np.inf)
        self.assertEqual(results.existence, "fails")
        self.assertIn("goes to zero", results.existence_basis)
        self.assertIn("prop:InfiniteMinHumanWealth", results.existence_basis)
        self.assertFalse(results.target_guaranteed)
        # Minimum human wealth not finite under the RIC: consumption grows
        # without bound (part i), so the limiting solution is degenerate
        results = calc_buffer_stock_conditions(
            1.04, 0.5, 2.0, 1.2, 0.005, 0.3, PermShkLaw, TranShkLaw
        )
        self.assertTrue(results.conditions["RIC"].holds)
        self.assertEqual(results.BoroCnstNat, -np.inf)
        self.assertEqual(results.existence, "fails")
        self.assertIn("grows without bound", results.existence_basis)

        # Neither part applies: with the FVAC, still no non-degenerate limiting
        # solution; without it, the paper does not say where consumption goes.
        # Here PermGroFac times the lowest permanent shock, 1.08, is between Þ
        # and PermGroFac, and above WorstPrb^(1/ρ) Þ.
        def middle(DiscFac):
            return calc_buffer_stock_conditions(
                1.04, DiscFac, 2.0, 1.2, 0.005, 0.3, PermShkLaw, TranShkLaw
            )

        results = middle(1.15)  # Þ = 1.09
        self.assertTrue(results.conditions["FVAC"].holds)
        self.assertEqual(results.existence, "fails")
        self.assertIn("under the FVAC", results.existence_basis)
        results = middle(1.25)  # Þ = 1.14, and the FVAC fails
        self.assertFalse(results.conditions["FVAC"].holds)
        self.assertEqual(results.existence, "not established")
        self.assertIn("does not say", results.existence_basis)

    def test_a_permanent_shock_without_a_point_mass_at_its_lowest_value(self):
        # The borrowing limit can then be reached and the MPC at the bottom is
        # one; a limiting solution exists under the FVAC and finite minimum
        # human wealth (item:NoPointMass, thm:existenceNoWorst), with the targets
        # as before (prop:NoWorstTargets)
        agent = make_agent(IncUnemp=0.3, PermShkMinPrb=0.0)
        agent.check_restrictions()
        results = agent.get_buffer_stock_conditions()
        self.assertEqual(results.WorstPrb, 0.0)
        self.assertEqual(results.MPCmax, 1.0)
        self.assertEqual(results.existence, "established")
        self.assertIn("thm:existenceNoWorst", results.existence_basis)
        self.assertTrue(results.target_guaranteed)
        self.assertTrue(results.pseudo_target_guaranteed)
        self.assertIn("Not needed", results.conditions["WRIC"].meaning)
        self.assertIn("prop:NoWorstTargets", results.conditions["GICMod"].meaning)
        self.assertIn("subsec:PositiveMinimumNoWorstEvent", results.notes[0])
        # The borrowing limit is the law's, as with the point mass
        x = 1.03 * agent.PermShkLaw.infimum[0] / 1.04
        self.assertAlmostEqual(results.BoroCnstNat, -0.3 * x / (1.0 - x), places=12)
        # The FVAC is the existence theorem's one condition besides finite
        # minimum human wealth
        agent = make_agent(IncUnemp=0.3, PermShkMinPrb=0.0, PermGroFac=[0.95])
        results = agent.get_buffer_stock_conditions()
        self.assertFalse(results.conditions["FVAC"].holds)
        self.assertEqual(results.existence, "not established")
        self.assertFalse(results.target_guaranteed)
        # With zero income in the event the paper's text needs no point mass
        paper = make_agent()
        paper.check_restrictions()
        paper.check_conditions()
        self.assertIn(
            "thm:convgtobellman", paper.bilt["BufferStockConditions"].existence_basis
        )

    def test_solution_without_a_point_mass_at_the_lowest_value(self):
        # The values of the check of the appendix's revision
        # (subsec:PositiveMinimumNoWorstEvent) at seven points: the solver's
        # lowest permanent shock is above the law's infimum, so the consumer
        # ends the period at the limit below a kink
        agent = make_agent(IncUnemp=0.3, PermShkMinPrb=0.0, tolerance=1e-10)
        agent.solve()
        solution, bilt = agent.solution[0], agent.bilt
        results = bilt["BufferStockConditions"]
        self.assertAlmostEqual(solution.mNrmMin, results.BoroCnstNat, places=8)
        self.assertAlmostEqual(float(solution.cFunc(1.0)), 1.1065, places=4)
        self.assertAlmostEqual(bilt["mNrmTrg"], 0.3741, places=4)
        self.assertAlmostEqual(bilt["mNrmStE"], 0.3975, places=4)
        span = np.array([1e-6, 0.5, 0.88])
        np.testing.assert_allclose(
            solution.cFunc(solution.mNrmMin + span), span, rtol=1e-9
        )
        self.assertLess(float(solution.cFunc(solution.mNrmMin + 0.89)), 0.89)

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

    def test_borrowing_against_minimum_income(self):
        # With positive income in the event the consumer may borrow up to
        # minimum human wealth, and here does so at the target and the
        # pseudo-target, which are then negative and in reversed order
        # (lemma:orderingPositiveMinimum)
        agent = make_agent(IncUnemp=0.3)
        agent.solve()
        results = agent.bilt["BufferStockConditions"]
        solution = agent.solution[0]
        cFunc, bilt = solution.cFunc, agent.bilt
        self.assertAlmostEqual(solution.mNrmMin, results.BoroCnstNat, places=8)
        self.assertLess(results.MPCmax, 1.0)
        self.assertAlmostEqual(solution.MPCmax, results.MPCmax, places=8)
        m_trg, m_bal = bilt["mNrmTrg"], bilt["mNrmStE"]
        self.assertEqual([m_trg], bilt["mNrmTrg_roots"])
        self.assertEqual([m_bal], bilt["mNrmStE_roots"])
        self.assertGreater(m_trg, results.BoroCnstNat)
        signs = {
            np.sign(1.0 - cFunc(1.0)),
            np.sign(m_bal - 1.0),
            np.sign(m_trg - 1.0),
            np.sign(m_bal - cFunc(m_bal)),
            np.sign(m_trg - cFunc(m_trg)),
            np.sign(m_trg - m_bal),
        }
        self.assertEqual(signs, {-1.0})
        # The bounds of lemma:MPCBoundsConvgPositiveMinimum
        m = results.BoroCnstNat + np.geomspace(1e-4, 100.0, 500)
        span = m - results.BoroCnstNat
        c = cFunc(m)
        self.assertTrue(np.all(c >= results.MPCmin * span - 1e-9))
        self.assertTrue(np.all(c <= results.MPCmax * span + 1e-9))

    def test_natural_constraint_of_the_laws(self):
        # A period's natural constraint comes from the laws' infima, not from the
        # discretization's lowest points. One period from the end, with a law
        # that has no point mass at its infimum and an artificial constraint
        # that does not yet bind:
        agent = make_agent(IncUnemp=0.3, BoroCnstArt=-0.3, cycles=1)
        agent.PermShkLaw = make_agent().PermShkLaw  # the truncated lognormal
        agent.update_income_process()
        agent.solve()
        infimum = agent.PermShkLaw.infimum[0]
        lowest_point = np.min(agent.PermShkDstn[0].atoms)
        self.assertLess(infimum, lowest_point)
        self.assertAlmostEqual(
            agent.solution[0].mNrmMin, -0.3 * 1.03 * infimum / 1.04, places=12
        )

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
        self.assertEqual(results.existence, "established")

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

    def test_the_constraint_at_zero_under_uncertainty(self):
        # The liquidity constrained consumer, with no zero-income event
        # (item:Restrained with zero minimum income), and a consumer who may not
        # borrow with positive income in the event (item:Artificial): a limiting
        # solution exists under the FVAC (thm:existenceNoWorst), with the targets
        # (prop:NoWorstTargets)
        for kwds, problem in (
            ({"UnempPrb": 0.0}, "item:Restrained"),
            ({"IncUnemp": 0.3}, "item:Artificial"),
        ):
            agent = make_agent(BoroCnstArt=0.0, **kwds)
            agent.check_conditions()
            results = agent.bilt["BufferStockConditions"]
            self.assertTrue(all(agent.conditions.values()))
            self.assertEqual(results.scenario, "artificial")
            self.assertEqual(results.existence, "established")
            self.assertIn("thm:existenceNoWorst", results.existence_basis)
            self.assertIn(problem, results.existence_basis)
            self.assertTrue(results.target_guaranteed)
            self.assertTrue(results.pseudo_target_guaranteed)
            self.assertEqual(results.MPCmax, 1.0)
            self.assertIn("prop:NoWorstTargets", results.conditions["GICMod"].meaning)
            self.assertIn("binds", results.notes[-1])
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                agent._respond_to_conditions()
        results = make_agent(
            UnempPrb=0.0, BoroCnstArt=0.0
        ).get_buffer_stock_conditions()
        self.assertEqual(results.WorstInc, make_agent().TranShkLaw.infimum[0])
        self.assertLess(results.BoroCnstNat, 0.0)
        # When the FVAC fails, nothing is established and the class warns
        agent = make_agent(UnempPrb=0.0, BoroCnstArt=0.0, PermGroFac=[0.95])
        agent.check_conditions()
        results = agent.bilt["BufferStockConditions"]
        self.assertFalse(agent.conditions["FVAC"])
        self.assertEqual(results.existence, "not established")
        self.assertFalse(results.target_guaranteed)
        with self.assertWarns(UserWarning):
            agent._respond_to_conditions()

    def test_what_is_not_claimed_under_uncertainty(self):
        # Without the point mass, the result for a consumer who may not borrow
        # does not apply (item:Artificial assumes ass:shocksPositiveMinimum)
        agent = make_agent(IncUnemp=0.3, PermShkMinPrb=0.0, BoroCnstArt=0.0)
        results = agent.get_buffer_stock_conditions()
        self.assertEqual(results.existence, "not established")
        self.assertIn("ass:shocksPositiveMinimum", results.existence_basis)
        self.assertFalse(results.target_guaranteed)
        for name in ("FVAC", "GICRaw", "GICMod"):
            self.assertIn(
                "not proved where an artificial", results.conditions[name].meaning
            )
        # A constant limit below zero is not the restrained consumer's, whose
        # limit changes with the horizon
        other = make_agent(UnempPrb=0.0, BoroCnstArt=-0.3)
        self.assertEqual(other.constraint_scenario(), "artificial")
        basis = other.get_buffer_stock_conditions().existence_basis
        self.assertIn("item:Restrained", basis)
        self.assertEqual(
            other.get_buffer_stock_conditions().existence, "not established"
        )
        # With a zero-income event the paper treats no limit other than zero
        other = make_agent(IncUnemp=0.3, BoroCnstArt=-0.3)
        basis = other.get_buffer_stock_conditions().existence_basis
        self.assertIn("no artificial borrowing limit other than zero", basis)

    def test_the_grid_starts_at_the_binding_constraint(self):
        # IndShockConsumerType's solver starts the asset grid at a binding
        # artificial constraint, with the constraint as a gridpoint, so the kink
        # where it stops binding is a node. For the liquidity constrained
        # consumer, located by bisection, the kink is then the value of the
        # check of the appendix's revision (item:Restrained), 0.9769
        agent = make_agent(UnempPrb=0.0, BoroCnstArt=0.0, tolerance=1e-10)
        agent.solve()
        cFunc = agent.solution[0].cFunc
        lo, hi = 0.9, 1.05
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if abs(float(cFunc(mid)) - mid) < 1e-12:
                lo = mid
            else:
                hi = mid
        self.assertAlmostEqual(lo, 0.9769, places=4)
        # The solution records that kink, on the constraint line
        solution = agent.solution[0]
        self.assertAlmostEqual(solution.mNrmKink, lo, places=9)
        self.assertAlmostEqual(
            solution.mNrmKink - solution.cNrmKink, solution.mNrmMin, places=12
        )

    def test_the_kink_where_the_laws_limit_binds(self):
        # Without a point mass at the permanent shock's lowest value, the laws'
        # limit binds as an artificial constraint at that level does, and the
        # solution records the kink there: 0.8840 above the limit at seven
        # points, as in the check of the appendix's revision (item:NoPointMass)
        agent = make_agent(IncUnemp=0.3, PermShkMinPrb=0.0, tolerance=1e-10)
        agent.solve()
        solution = agent.solution[0]
        self.assertAlmostEqual(solution.mNrmKink - solution.mNrmMin, 0.8840, delta=5e-4)
        self.assertAlmostEqual(
            solution.mNrmKink - solution.cNrmKink, solution.mNrmMin, places=12
        )
        self.assertLess(solution.MPCaboveKink, 1.0)
        # In the paper's own model the natural constraint binds: no kink
        paper = make_agent()
        paper.solve()
        self.assertTrue(np.isnan(paper.solution[0].mNrmKink))

    def test_no_borrowing_with_positive_income_in_the_event(self):
        # The values of the check of the appendix's revision (item:Artificial),
        # with the default laws: the constraint binds below a positive level of
        # m, and both targets are at least one
        expected = {
            0.05: (0.5604, 0.8860, 1.3179, 1.3068),
            0.3: (0.9459, 0.9677, 1.1001, 1.0967),
            0.5: (0.9688, 0.9814, 1.0523, 1.0509),
        }
        for IncUnemp, (kink, c_one, target, pseudo) in expected.items():
            agent = make_agent(IncUnemp=IncUnemp, BoroCnstArt=0.0, tolerance=1e-10)
            agent.solve()
            solution, bilt = agent.solution[0], agent.bilt
            self.assertEqual(solution.mNrmMin, 0.0)
            self.assertAlmostEqual(float(solution.cFunc(1.0)), c_one, places=4)
            self.assertAlmostEqual(bilt["mNrmTrg"], target, delta=2e-4)
            self.assertAlmostEqual(bilt["mNrmStE"], pseudo, delta=2e-4)
            self.assertGreaterEqual(bilt["mNrmStE"], 1.0)
            below, above = kink - 1e-3, kink + 1e-3
            self.assertAlmostEqual(float(solution.cFunc(below)), below, places=9)
            self.assertLess(float(solution.cFunc(above)), above - 1e-9)

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
            # The stable points found are listed. At zero, without a zero-income
            # event, the paper guarantees them (prop:NoWorstTargets); below zero
            # no theorem does, and none is recorded as the target
            self.assertEqual(len(agent.bilt["mNrmTrg_roots"]), 1)
            self.assertEqual(len(agent.bilt["mNrmStE_roots"]), 1)
            if floor == 0.0:
                self.assertEqual([agent.bilt["mNrmTrg"]], agent.bilt["mNrmTrg_roots"])
                self.assertEqual([agent.bilt["mNrmStE"]], agent.bilt["mNrmStE_roots"])
            else:
                self.assertTrue(np.isnan(agent.bilt["mNrmTrg"]))
                self.assertTrue(np.isnan(agent.bilt["mNrmStE"]))

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
