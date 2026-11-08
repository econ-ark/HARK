import logging
import unittest

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    PerfForesightConsumerType,
    calc_human_wealth,
    calc_human_wealth_closed_form,
    calc_mpc_min,
    calc_mpc_min_closed_form,
    calc_patience_factor,
)
from tests import HARK_PRECISION


class testPerfForesightConsumerType(unittest.TestCase):
    def setUp(self):
        self.agent = PerfForesightConsumerType()
        self.agent_infinite = PerfForesightConsumerType(cycles=0)

        PF_dictionary = {
            "CRRA": 2.5,
            "DiscFac": 0.96,
            "Rfree": [1.03],
            "LivPrb": [0.98],
            "PermGroFac": [1.01],
            "T_cycle": 1,
            "cycles": 0,
            "AgentCount": 10000,
        }
        self.agent_alt = PerfForesightConsumerType(**PF_dictionary)

    def test_default_solution(self):
        self.agent.solve()
        c = self.agent.solution[0].cFunc

        self.assertAlmostEqual(c.x_list[0], -0.98058, places=HARK_PRECISION)
        self.assertAlmostEqual(c.x_list[1], 0.01942, places=HARK_PRECISION)
        self.assertEqual(c.y_list[0], 0)
        self.assertAlmostEqual(c.y_list[1], 0.51132, places=HARK_PRECISION)
        self.assertEqual(c.decay_extrap, False)

    def test_another_solution(self):
        self.agent_alt.DiscFac = 0.90
        self.agent_alt.solve()
        self.assertAlmostEqual(
            self.agent_alt.solution[0].cFunc(10).tolist(),
            3.97501,
            places=HARK_PRECISION,
        )

    def test_check_conditions(self):
        self.agent_infinite.check_conditions()
        self.assertTrue(self.agent_infinite.conditions["AIC"])
        self.assertTrue(self.agent_infinite.conditions["GICRaw"])
        self.assertTrue(self.agent_infinite.conditions["RIC"])
        self.assertTrue(self.agent_infinite.conditions["FHWC"])

    def test_failed_conditions(self):
        TestType = PerfForesightConsumerType(cycles=0, quiet=False, verbose=False)
        TestType.check_conditions()

        # make DiscFac way too big
        TestType = PerfForesightConsumerType(cycles=0, DiscFac=1.06)
        TestType.check_conditions()

        # make PermGroFac big
        TestType = PerfForesightConsumerType(cycles=0, DiscFac=0.96, PermGroFac=[1.1])
        TestType.check_conditions()

        # make Rfree too big
        TestType = PerfForesightConsumerType(cycles=0, Rfree=[1.1])
        TestType.check_conditions()

        # test constrained outcomes
        TestType = PerfForesightConsumerType(cycles=0, Rfree=[0.9], BoroCnstArt=0.0)
        TestType.check_conditions()
        TestType = PerfForesightConsumerType(
            cycles=0, PermGroFac=[0.9], BoroCnstArt=0.0
        )
        TestType.check_conditions()
        TestType = PerfForesightConsumerType(
            cycles=0, Rfree=[0.9], PermGroFac=[0.9], BoroCnstArt=0.0
        )
        TestType.check_conditions()

    def test_simulation(self):
        self.agent_infinite.solve()

        # Create parameter values necessary for simulation
        SimulationParams = {
            "AgentCount": 10000,  # Number of agents of this type
            "T_sim": 120,  # Number of periods to simulate
            "kLogInitMean": -6.0,  # Mean of log initial assets
            "kLogInitStd": 1.0,  # Standard deviation of log initial assets
            "pLogInitMean": 0.0,  # Mean of log initial permanent income
            "pLogInitStd": 0.0,  # Standard deviation of log initial permanent income
            "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
            "T_age": None,  # Age after which simulated agents are automatically killed
        }

        self.agent_infinite.assign_parameters(
            **SimulationParams
        )  # This implicitly uses the assign_parameters method of AgentType

        # Create PFexample object
        self.agent_infinite.track_vars = ["bNrm", "mNrm", "TranShk"]
        self.agent_infinite.initialize_sim()
        self.agent_infinite.simulate()

        self.assertAlmostEqual(
            np.mean(self.agent_infinite.history["mNrm"], axis=1)[40],
            np.mean(self.agent_infinite.history["bNrm"], axis=1)[40]
            + np.mean(self.agent_infinite.history["TranShk"], axis=1)[40],
        )

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    np.mean(self.agent_infinite.history["mNrm"], axis=1)[100],
        #    -27.16461,
        # )

        ## Try now with the manipulation at time step 80

        self.agent_infinite.initialize_sim()
        self.agent_infinite.simulate(80)

        # This actually does nothing because aNrmNow is
        # epiphenomenal. Probably should change mNrmNow instead
        self.agent_infinite.state_now["aNrm"] += -5.0
        self.agent_infinite.simulate(40)

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    np.mean(self.agent_infinite.history["mNrm"], axis=1)[40],
        #    -23.00806,
        # )

        # simulation test -- seed/generator specific
        # self.assertAlmostEqual(
        #    np.mean(self.agent_infinite.history["mNrm"], axis=1)[100],
        #    -29.14026,
        # )

    def test_stable_points(self):
        # Solve the constrained agent. Stable points exists only with a
        # borrowing constraint.
        constrained_agent = PerfForesightConsumerType(cycles=0, BoroCnstArt=0.0)

        constrained_agent.solve()

        # Check against pre-computed values.
        self.assertEqual(constrained_agent.solution[0].mNrmStE, 1.0)
        # Check that they are both the same, since the problem is deterministic
        self.assertEqual(
            constrained_agent.solution[0].mNrmStE, constrained_agent.solution[0].mNrmTrg
        )


class testClosedFormPerfForesight(unittest.TestCase):
    # (CRRA, DiscFac, Rfree, PermGroFac, LivPrb)
    param_sets = [
        (2.5, 0.96, 1.03, 1.01, 1.0),
        (2.0, 0.96, 1.03, 1.01, 0.98),
        (4.0, 1.02, 1.05, 1.03, 1.0),
    ]

    def make_agent(self, params, **kwds):
        CRRA, DiscFac, Rfree, PermGroFac, LivPrb = params
        return PerfForesightConsumerType(
            CRRA=CRRA,
            DiscFac=DiscFac,
            Rfree=[Rfree],
            PermGroFac=[PermGroFac],
            LivPrb=[LivPrb],
            T_cycle=1,
            **kwds,
        )

    def test_matches_iterated_one_step_helpers(self):
        for CRRA, DiscFac, Rfree, PermGroFac, LivPrb in self.param_sets:
            pat_fac = calc_patience_factor(Rfree, DiscFac * LivPrb, CRRA)
            mpc_min, h_nrm = 1.0, 0.0
            for n in range(1, 301):
                mpc_min = calc_mpc_min(mpc_min, pat_fac)
                h_nrm = calc_human_wealth(h_nrm, PermGroFac, Rfree, 1.0)
                np.testing.assert_allclose(
                    calc_mpc_min_closed_form(pat_fac, n), mpc_min, rtol=1e-12
                )
                np.testing.assert_allclose(
                    calc_human_wealth_closed_form(PermGroFac, Rfree, n),
                    h_nrm,
                    rtol=1e-12,
                )

    def test_matches_finite_horizon_solution(self):
        for params in self.param_sets:
            CRRA, DiscFac, Rfree, PermGroFac, LivPrb = params
            pat_fac = calc_patience_factor(Rfree, DiscFac * LivPrb, CRRA)
            for T in [1, 7, 60]:
                agent = self.make_agent(params, cycles=T)
                agent.solve()
                for t, solution in enumerate(agent.solution):
                    periods_left = len(agent.solution) - 1 - t
                    np.testing.assert_allclose(
                        calc_mpc_min_closed_form(pat_fac, periods_left),
                        solution.MPCmin,
                        rtol=1e-12,
                    )
                    np.testing.assert_allclose(
                        calc_human_wealth_closed_form(PermGroFac, Rfree, periods_left),
                        solution.hNrm,
                        rtol=1e-12,
                        atol=1e-12,
                    )

    def test_matches_infinite_horizon_solution(self):
        for params in self.param_sets:
            CRRA, DiscFac, Rfree, PermGroFac, LivPrb = params
            pat_fac = calc_patience_factor(Rfree, DiscFac * LivPrb, CRRA)
            agent = self.make_agent(params, cycles=0, tolerance=1e-12)
            agent.solve()
            agent.check_conditions(verbose=0)
            mpc_min = calc_mpc_min_closed_form(pat_fac)
            h_nrm = calc_human_wealth_closed_form(PermGroFac, Rfree)
            np.testing.assert_allclose(mpc_min, agent.solution[0].MPCmin, rtol=1e-10)
            np.testing.assert_allclose(h_nrm, agent.solution[0].hNrm, rtol=1e-9)
            # bilt holds the same limits, but its hNrm includes this period's income
            np.testing.assert_allclose(mpc_min, agent.bilt["MPCmin"], rtol=1e-12)
            np.testing.assert_allclose(h_nrm + 1.0, agent.bilt["hNrm"], rtol=1e-12)

    def test_edge_cases(self):
        # Terminal period
        self.assertEqual(calc_mpc_min_closed_form(0.98, 0), 1.0)
        self.assertEqual(calc_human_wealth_closed_form(1.01, 1.03, 0), 0.0)
        # Return patience factor of one: each remaining period gets an equal share
        self.assertAlmostEqual(calc_mpc_min_closed_form(1.0, 9), 0.1)
        # RIC fails: the limiting MPC is zero
        self.assertEqual(calc_mpc_min_closed_form(1.0), 0.0)
        self.assertEqual(calc_mpc_min_closed_form(1.01), 0.0)
        # G = R: human wealth is the number of remaining periods
        self.assertEqual(calc_human_wealth_closed_form(1.03, 1.03, 5), 5.0)
        # FHWC fails: infinite human wealth
        self.assertEqual(calc_human_wealth_closed_form(1.03, 1.03), np.inf)
        self.assertEqual(calc_human_wealth_closed_form(1.05, 1.03), np.inf)


class testPerfForesightConditionReport(unittest.TestCase):
    """The conditions and the verdict for every ordering of the absolute patience
    factor, R and G, with and without a borrowing constraint."""

    # name: Rfree, PermGroFac, DiscFac, RIC, FHWC, GICRaw, the limit without a
    # constraint, the limit with a binding one
    orderings = {
        "Pat<G<R": (1.04, 1.01, 0.90, True, True, True, "linear", "kinked"),
        "G<Pat<R": (1.04, 0.98, 0.96, True, True, False, "linear", "kinked"),
        "Pat<R<G": (1.02, 1.05, 0.90, True, False, True, "c(m)=Infinity", "kinked"),
        "G<R<Pat": (1.02, 1.00, 1.05, False, True, False, "c(m) = 0", "c(m)=0"),
        "R<Pat<G": (1.00, 1.05, 1.02, False, False, True, "c(m)=Infinity", "kinked"),
        "R<G<Pat": (1.00, 1.01, 1.10, False, False, False, "c(m)=0", "c(m)=0"),
    }

    def make_agent(self, Rfree, PermGroFac, DiscFac, **kwds):
        return PerfForesightConsumerType(
            cycles=0,
            CRRA=2.0,
            Rfree=[Rfree],
            PermGroFac=[PermGroFac],
            DiscFac=DiscFac,
            LivPrb=[1.0],
            **kwds,
        )

    def verdict(self, agent):
        """The sentence of the report that states the limit of the solution."""
        logger = logging.getLogger("HARK")
        level = logger.level
        agent.check_conditions(verbose=3)
        logger.setLevel(level)
        report = agent.bilt["conditions_report"]
        return [x for x in report.splitlines() if x.startswith("Because")][0]

    def check_limit(self, name, limit, verdict):
        if limit == "linear":
            self.assertIn("non-degenerate linear solution exists", verdict, name)
        elif limit == "kinked":
            self.assertNotIn("degenerate at", verdict, name)
            self.assertIn("non-degenerate", verdict, name)
        else:
            self.assertIn("degenerate at " + limit, verdict, name)

    def test_conditions_and_verdict_unconstrained(self):
        for name, case in self.orderings.items():
            R, G, b, RIC, FHWC, GIC, limit, _ = case
            agent = self.make_agent(R, G, b)
            verdict = self.verdict(agent)
            self.assertEqual(bool(agent.conditions["RIC"]), RIC, name)
            self.assertEqual(bool(agent.conditions["FHWC"]), FHWC, name)
            self.assertEqual(bool(agent.conditions["GICRaw"]), GIC, name)
            self.check_limit(name, limit, verdict)

    def test_verdict_constrained(self):
        for BoroCnstArt in (0.0, -0.5):
            for name, case in self.orderings.items():
                R, G, b, _, _, _, _, limit = case
                agent = self.make_agent(R, G, b, BoroCnstArt=BoroCnstArt)
                self.check_limit(name, limit, self.verdict(agent))

    def test_limit_of_closed_form_when_all_three_fail(self):
        # Consumption at m = 1 with n periods left is kappa_n * (1 + h_n)
        pat_fac = calc_patience_factor(1.00, 1.10, 2.0)
        c = [
            calc_mpc_min_closed_form(pat_fac, n)
            * (1.0 + calc_human_wealth_closed_form(1.01, 1.00, n))
            for n in (10, 100, 1000)
        ]
        self.assertGreater(c[0], c[1])
        self.assertGreater(c[1], c[2])
        self.assertLess(c[2], 1e-12)

    def test_verdict_names_only_conditions_that_hold(self):
        # All three fail: the report must not say that the FHWC is satisfied
        R, G, b = self.orderings["R<G<Pat"][:3]
        verdict = self.verdict(self.make_agent(R, G, b, BoroCnstArt=0.0))
        self.assertNotIn("FHWC is satisfied", verdict)
        # The FHWC fails: there is no unconstrained solution to approach
        R, G, b = self.orderings["Pat<R<G"][:3]
        verdict = self.verdict(self.make_agent(R, G, b, BoroCnstArt=0.0))
        self.assertNotIn("approaching the unconstrained solution", verdict)
        # The FHWC holds: there is one
        R, G, b = self.orderings["Pat<G<R"][:3]
        verdict = self.verdict(self.make_agent(R, G, b, BoroCnstArt=0.0))
        self.assertIn("approaching the unconstrained solution", verdict)

    def test_growth_patience_factor_of_exactly_one(self):
        G = 1.1**0.5
        agent = self.make_agent(1.0, G, 1.1)
        verdict = self.verdict(agent)
        self.assertEqual(agent.bilt["GPFacRaw"], 1.0)
        self.assertIn("degenerate at c(m)=1", verdict)
        verdict = self.verdict(self.make_agent(1.0, G, 1.1, BoroCnstArt=0.0))
        self.assertIn("degenerate at c(m)=min(m-BoroCnstArt,1)", verdict)

    def test_constraint_that_cannot_bind(self):
        R, G, b = self.orderings["Pat<G<R"][:3]
        agent = self.make_agent(R, G, b, BoroCnstArt=-1000.0, MaxKinks=100)
        self.check_limit("not binding", "linear", self.verdict(agent))
        self.assertAlmostEqual(agent.bilt["MPCmax"], agent.bilt["MPCmin"], places=12)
        # A constraint above minus human wealth can bind
        agent = self.make_agent(R, G, b, BoroCnstArt=-20.0)
        self.check_limit("binding", "kinked", self.verdict(agent))
        self.assertEqual(agent.bilt["MPCmax"], 1.0)
        # With infinite human wealth, every finite constraint can bind
        R, G, b = self.orderings["Pat<R<G"][:3]
        agent = self.make_agent(R, G, b, BoroCnstArt=-1000.0)
        self.check_limit("infinite h", "kinked", self.verdict(agent))

    def test_cNrmPDV(self):
        R, G, b = self.orderings["G<R<Pat"][:3]
        agent = self.make_agent(R, G, b, verbose=0)
        agent.check_conditions()
        self.assertEqual(agent.bilt["cNrmPDV"], np.inf)
        # A Return Patience Factor of exactly one no longer divides by zero
        agent = self.make_agent(1.0, 0.99, 1.0, verbose=0)
        agent.check_conditions()
        self.assertEqual(agent.bilt["RPFac"], 1.0)
        self.assertEqual(agent.bilt["cNrmPDV"], np.inf)
        self.assertFalse(agent.conditions["RIC"])

    def test_stable_point_unconstrained(self):
        R, G, b = self.orderings["Pat<G<R"][:3]
        agent = self.make_agent(R, G, b, verbose=0)
        agent.solve()
        hNrm = agent.solution[0].hNrm
        self.assertAlmostEqual(agent.bilt["mNrmStE"], -hNrm, places=3)
        self.assertEqual(agent.bilt["mNrmTrg"], agent.bilt["mNrmStE"])

    def test_stable_point_constrained(self):
        R, G, b = self.orderings["Pat<G<R"][:3]
        for BoroCnstArt in (0.0, -0.5):
            agent = self.make_agent(R, G, b, BoroCnstArt=BoroCnstArt, verbose=0)
            agent.solve()
            target = 1.0 + BoroCnstArt * R / G
            self.assertAlmostEqual(agent.bilt["mNrmStE"], target, places=6)
            self.assertEqual(agent.bilt["mNrmTrg"], agent.bilt["mNrmStE"])
