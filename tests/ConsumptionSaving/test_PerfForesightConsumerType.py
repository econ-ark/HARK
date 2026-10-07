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
