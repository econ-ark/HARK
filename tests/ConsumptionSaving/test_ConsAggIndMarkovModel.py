"""Tests for AggIndMrkvConsumerType hierarchical Markov + shuffle wiring."""

import unittest
from copy import deepcopy

import numpy as np

from HARK.ConsumptionSaving.ConsAggIndMarkovModel import (
    AggIndMrkvConsumerType,
    make_hierarchical_mrkv_array,
)
from HARK.ConsumptionSaving.ConsMarkovModel import init_indshk_markov
from HARK.distributions import DiscreteDistributionLabeled


class testAggIndMarkovShuffle(unittest.TestCase):
    """markov_shuffle on AggIndMrkvConsumerType uses MarkovProcess shuffling per micro source state."""

    def test_hierarchical_markov_shuffle_transition_counts(self):
        TM_micro = np.array([[0.95, 0.05], [0.5, 0.5]])
        MrkvInd = make_hierarchical_mrkv_array(np.array([[1.0]]), [TM_micro])

        init = deepcopy(init_indshk_markov)
        init["cycles"] = 0
        init["MrkvArray"] = [MrkvInd]
        init["constructors"]["MrkvArray"] = None
        init["Rfree"] = [np.array([1.03, 1.03])]
        init["LivPrb"] = [np.array([0.98, 0.98])]
        init["PermGroFac"] = [np.array([1.01, 1.01])]
        init["num_macro_states"] = 1
        init["num_micro_states"] = 2
        init["CondMrkvArrays"] = [TM_micro]
        init["AgentCount"] = 10_000
        init["T_sim"] = 3
        init["markov_shuffle"] = True
        init["MrkvPrbsInit"] = np.array([0.95, 0.05])

        agent = AggIndMrkvConsumerType(**init)
        det_inc = DiscreteDistributionLabeled(
            pmv=np.ones(1),
            atoms=np.array([[1.0], [1.0]]),
            var_names=["PermShk", "TranShk"],
        )
        agent.IncShkDstn = [[det_inc, det_inc]]
        agent.solve()
        agent.initialize_sim()

        agent.shocks["Mrkv"] = np.array([0] * 9500 + [1] * 500, dtype=int)
        agent.t_age[:] = 1
        agent.t_sim = 1
        agent.get_markov_states()
        new = agent.shocks["Mrkv"]
        prev = np.array([0] * 9500 + [1] * 500)
        count_0_to_0 = np.sum((prev == 0) & (new == 0))
        count_0_to_1 = np.sum((prev == 0) & (new == 1))
        count_1_to_0 = np.sum((prev == 1) & (new == 0))
        count_1_to_1 = np.sum((prev == 1) & (new == 1))

        self.assertAlmostEqual(count_0_to_0, 9025, delta=1)
        self.assertAlmostEqual(count_0_to_1, 475, delta=1)
        self.assertAlmostEqual(count_1_to_0, 250, delta=1)
        self.assertAlmostEqual(count_1_to_1, 250, delta=1)

    def test_hierarchical_markov_shuffle_crn_across_conditional_arrays(self):
        """Common random numbers across scenarios with perturbed CondMrkvArrays.

        Two ``AggIndMrkvConsumerType`` instances are run with the same seed but
        one has a perturbed row in its ``CondMrkvArrays``.  Agents whose source
        micro row is *unchanged* between the two scenarios must produce
        bit-identical output, leaving only the perturbed row as a source of
        treatment-effect variance.  This is the end-to-end property required
        for counterfactual / scenario-comparison research; it relies on
        ``markov_shuffle=True`` being honored in ``get_micro_markov_states``
        and on ``MarkovProcess._draw_shuffled`` providing per-source-state
        sub-RNG isolation.
        """
        # Simple 2-macro, 4-micro hierarchical Markov. One conditional row
        # (macro=0, source micro=2) will be perturbed in the policy scenario.
        cond_base_0 = np.array(
            [
                [0.95, 0.05, 0.00, 0.00],
                [0.50, 0.50, 0.00, 0.00],
                [0.30, 0.00, 0.70, 0.00],  # <-- will be perturbed
                [0.30, 0.00, 0.00, 0.70],
            ]
        )
        cond_base_1 = np.array(
            [
                [0.90, 0.10, 0.00, 0.00],
                [0.40, 0.60, 0.00, 0.00],
                [0.20, 0.00, 0.80, 0.00],
                [0.20, 0.00, 0.00, 0.80],
            ]
        )
        cond_policy_0 = cond_base_0.copy()
        # Shift 10pp of mass from target 0 to target 2 in source row 2
        cond_policy_0[2] = np.array([0.20, 0.00, 0.80, 0.00])

        def build_agent(cond_0, cond_1, seed):
            agent = AggIndMrkvConsumerType.__new__(AggIndMrkvConsumerType)
            agent.num_micro_states = 4
            agent.AgentCount = 400
            agent.CondMrkvArrays = [cond_0, cond_1]
            agent.RNG = np.random.default_rng(seed)
            # All agents in macro 0
            agent.MacroMrkvNow = np.zeros(agent.AgentCount, dtype=int)
            # 100 agents in each of the 4 micro source states
            agent.shocks = {
                "Mrkv": np.concatenate(
                    [np.full(100, 4 * 0 + j) for j in range(4)]
                ).astype(int)
            }
            agent.state_now = {}
            agent.markov_shuffle = True
            return agent

        a_base = build_agent(cond_base_0, cond_base_1, seed=42)
        a_policy = build_agent(cond_policy_0, cond_base_1, seed=42)

        a_base.get_micro_markov_states()
        a_policy.get_micro_markov_states()

        # Source micro states 0, 1, 3: unchanged row → bit-identical output.
        # Source micro state 2: perturbed row → exactly floor(100 * 0.10) = 10
        # agents should differ (the ones the policy redirected from target 0
        # to target 2).
        prev_micro = a_base.shocks["Mrkv"] % 4
        for j, expected_diff in [(0, 0), (1, 0), (2, 10), (3, 0)]:
            mask = prev_micro == j
            n_diff = int(
                np.sum(a_base.MicroMrkvNow[mask] != a_policy.MicroMrkvNow[mask])
            )
            self.assertEqual(
                n_diff,
                expected_diff,
                msg=(
                    f"source micro state {j}: expected {expected_diff} "
                    f"agents to differ between base and policy runs, got "
                    f"{n_diff}.  This indicates either that markov_shuffle "
                    f"is not honored in get_micro_markov_states, or that "
                    f"MarkovProcess._draw_shuffled is not providing true "
                    f"CRN across calls with different transition matrices."
                ),
            )


class testMakeShockHistoryShuffleFlag(unittest.TestCase):
    """make_shock_history(shuffle=True) restores income_shuffle / markov_shuffle."""

    def test_restores_flags_after_shuffle_true(self):
        from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType

        agent = MarkovConsumerType(
            MrkvArray=[np.array([[0.9, 0.1], [0.1, 0.9]])],
            AgentCount=200,
            T_sim=5,
            markov_shuffle=False,
            income_shuffle=False,
        )
        agent.cycles = 0
        agent.solve()
        self.assertFalse(agent.markov_shuffle)
        self.assertFalse(agent.income_shuffle)
        agent.make_shock_history(shuffle=True)
        self.assertFalse(agent.markov_shuffle)
        self.assertFalse(agent.income_shuffle)
        self.assertTrue(agent.read_shocks)
