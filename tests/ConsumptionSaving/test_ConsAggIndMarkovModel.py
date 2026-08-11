"""Tests for AggIndMrkvConsumerType hierarchical Markov + shuffle wiring."""

import unittest
import warnings
from copy import deepcopy

import numpy as np

import HARK.ConsumptionSaving.ConsAggIndMarkovModel as agg_ind_mrkv_module
from HARK.ConsumptionSaving.ConsAggIndMarkovModel import (
    AggIndMrkvConsumerType,
    make_hierarchical_mrkv_array,
)
from HARK.ConsumptionSaving.ConsMarkovModel import (
    MarkovConsumerType,
    init_indshk_markov,
)
from HARK.distributions import DiscreteDistributionLabeled, MarkovProcess


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
            agent.state_prev = {}
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


class testGeneralFormatShuffle(unittest.TestCase):
    """The general ``CondMrkvArrays[i][j]`` format under ``markov_shuffle``.

    The simple format conditions micro transitions on the *destination* macro
    state alone; the general format conditions on the (source, destination)
    pair.  These tests keep two distinct pairs live in the same call, with
    deliberately asymmetric arrays, so that reading the pair in the wrong
    order is visible in the per-cell counts.
    """

    # Rows chosen so that CondMrkvArrays[0][1] is the transpose-in-index of
    # CondMrkvArrays[1][0]: swapping (mp, mn) swaps 75/25 for 25/75.
    COND_01 = np.array([[0.75, 0.25], [0.25, 0.75]])
    COND_10 = np.array([[0.25, 0.75], [0.75, 0.25]])
    # Diagonal cells are never exercised by these tests, but must be present
    # and row-stochastic for the array to be a well-formed general format.
    COND_DIAG = np.array([[1.0, 0.0], [0.0, 1.0]])

    def _make_agent(self, cond, macro_prev, macro_next, micro_prev, seed=0):
        """Build a bare agent with the hierarchical attributes set by hand.

        Bypasses ``__init__`` because ``get_micro_markov_states`` reads only
        these attributes, and a full agent would drag in a solver and an
        income process irrelevant to the transition arithmetic.
        """
        agent = AggIndMrkvConsumerType.__new__(AggIndMrkvConsumerType)
        agent.num_macro_states = len(cond)
        agent.num_micro_states = cond[0][0].shape[0]
        agent.AgentCount = macro_prev.size
        agent.CondMrkvArrays = cond
        agent.markov_shuffle = True
        agent.state_now = {}
        agent.state_prev = {}
        agent.RNG = np.random.default_rng(seed)
        N = agent.num_micro_states
        agent.shocks = {"Mrkv": (N * macro_prev + micro_prev).astype(int)}
        agent.MacroMrkvNow = macro_next.astype(int)
        return agent

    def _two_pair_setup(self):
        """200 agents in macro transition (0,1) and 200 in (1,0).

        Within each, 100 start in micro state 0 and 100 in micro state 1, so
        every one of the four (pair, source-micro) cells is populated.
        """
        cond = [
            [self.COND_DIAG, self.COND_01],
            [self.COND_10, self.COND_DIAG],
        ]
        macro_prev = np.repeat([0, 1], 200)
        macro_next = np.repeat([1, 0], 200)
        micro_prev = np.tile(np.repeat([0, 1], 100), 2)
        return cond, macro_prev, macro_next, micro_prev

    def test_general_format_per_cell_counts(self):
        """Each (macro_prev, macro_next, micro_prev) cell draws its own row.

        ``MarkovProcess.draw(shuffle=True)`` is quota-exact, so with 100
        agents per cell the destination counts are deterministic and equal to
        100 times the conditional row.  Reading ``CondMrkvArrays[mn][mp]``
        instead of ``[mp][mn]`` swaps the two off-diagonal cells and flips
        every count below.
        """
        cond, macro_prev, macro_next, micro_prev = self._two_pair_setup()
        agent = self._make_agent(cond, macro_prev, macro_next, micro_prev)
        agent.get_micro_markov_states()
        new_micro = agent.MicroMrkvNow

        expected = {
            # (macro_prev, macro_next, micro_prev): [count to 0, count to 1]
            (0, 1, 0): [75, 25],
            (0, 1, 1): [25, 75],
            (1, 0, 0): [25, 75],
            (1, 0, 1): [75, 25],
        }
        for (mp, mn, mi), counts in expected.items():
            mask = (macro_prev == mp) & (macro_next == mn) & (micro_prev == mi)
            self.assertEqual(int(mask.sum()), 100)
            observed = np.bincount(new_micro[mask], minlength=2)
            np.testing.assert_array_equal(
                observed,
                np.array(counts),
                err_msg=(
                    f"cell (macro_prev={mp}, macro_next={mn}, "
                    f"micro_prev={mi}) drew {observed.tolist()}, expected "
                    f"{counts}.  A mismatch that swaps this cell with "
                    f"(macro_prev={mn}, macro_next={mp}) means the general "
                    f"format is being indexed as CondMrkvArrays[mn][mp]."
                ),
            )

    def test_general_format_all_agents_assigned(self):
        """Every agent leaves with a micro state in range.

        The drawing loops fill a sentinel-initialised buffer, so an agent that
        no loop covers is detectable; before the sentinel it carried whatever
        was in the allocation.
        """
        cond, macro_prev, macro_next, micro_prev = self._two_pair_setup()
        agent = self._make_agent(cond, macro_prev, macro_next, micro_prev)
        agent.get_micro_markov_states()
        new_micro = agent.MicroMrkvNow
        self.assertEqual(new_micro.size, macro_prev.size)
        self.assertTrue(
            np.all((new_micro >= 0) & (new_micro < 2)),
            msg=f"out-of-range micro states drawn: {np.unique(new_micro)}",
        )

    def test_zero_probability_macro_transition_raises(self):
        """A macro transition with no probability mass is an error, not a draw.

        ``extract_cond_mrkv_arrays`` returns a zero matrix where the macro
        probability is zero.  If agents are nonetheless assigned that
        transition, there is no row to draw from, and the model that produced
        the macro states contradicts the conditional arrays.
        """
        cond = [
            [self.COND_DIAG, np.zeros((2, 2))],
            [self.COND_10, self.COND_DIAG],
        ]
        macro_prev = np.zeros(50, dtype=int)
        macro_next = np.ones(50, dtype=int)
        micro_prev = np.zeros(50, dtype=int)
        agent = self._make_agent(cond, macro_prev, macro_next, micro_prev)
        with self.assertRaises(ValueError) as cm:
            agent.get_micro_markov_states()
        message = str(cm.exception)
        self.assertIn("CondMrkvArrays[0][1]", message)
        self.assertIn("zero-probability transition", message)

    def test_zero_probability_simple_format_raises(self):
        """Same guarantee for the simple (destination-conditioned) format."""
        cond = [self.COND_DIAG, np.zeros((2, 2))]
        macro_prev = np.zeros(50, dtype=int)
        macro_next = np.ones(50, dtype=int)
        micro_prev = np.zeros(50, dtype=int)
        agent = self._make_agent(
            [[c, c] for c in cond], macro_prev, macro_next, micro_prev
        )
        agent.CondMrkvArrays = cond
        with self.assertRaises(ValueError) as cm:
            agent.get_micro_markov_states()
        self.assertIn("CondMrkvArrays[1]", str(cm.exception))

    def test_balanced_transitions_without_pLvl_warns(self):
        """Balancing silently does nothing without pLvl, so it must say so."""
        cond, macro_prev, macro_next, micro_prev = self._two_pair_setup()
        agent = self._make_agent(cond, macro_prev, macro_next, micro_prev)
        agent.balanced_transitions = True
        with self.assertWarns(RuntimeWarning) as cm:
            agent.get_micro_markov_states()
        self.assertIn("pLvl", str(cm.warning))

    def test_balanced_transitions_sorts_within_cell_by_pLvl(self):
        """pLvl reaches the draw as a sort key, spacing the destinations evenly.

        Systematic sampling assigns the minority destination at a fixed stride
        in pLvl rank, so within the 100-agent cell that sends 25 agents to
        micro state 1 the ranks of those agents are exactly 4 apart.  Without
        the sort key the same cell draws them in a clumped iid pattern and the
        gaps vary.  No warning may be issued, since pLvl is present.
        """
        cond, macro_prev, macro_next, micro_prev = self._two_pair_setup()
        agent = self._make_agent(cond, macro_prev, macro_next, micro_prev)
        agent.balanced_transitions = True
        pLvl = np.linspace(0.5, 2.0, macro_prev.size)
        agent.state_prev = {"pLvl": pLvl}
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            agent.get_micro_markov_states()

        mask = (macro_prev == 0) & (macro_next == 1) & (micro_prev == 0)
        order = np.argsort(pLvl[mask])
        destinations = agent.MicroMrkvNow[mask][order]
        ranks = np.flatnonzero(destinations == 1)
        self.assertEqual(ranks.size, 25)
        np.testing.assert_array_equal(
            np.diff(ranks),
            np.full(24, 4),
            err_msg=(
                "minority destinations are not evenly spaced in pLvl rank, so "
                "state_prev['pLvl'] is not reaching MarkovProcess.draw as "
                "sort_key and balanced_transitions is a no-op"
            ),
        )

    def test_balanced_transitions_ignores_state_now_pLvl(self):
        """A pLvl in state_now must not satisfy the balanced path.

        This method runs inside ``get_shocks``, which
        ``AgentType._sim_period_prologue`` calls after replacing every ndarray
        in ``state_now`` with ``np.empty``.  Measured during a live
        simulation, ``state_now["pLvl"]`` at this point held values like
        6.6e-310 and a stale 1.178 left over from a previous allocation, while
        ``state_prev["pLvl"]`` held the real previous-period levels.  Sorting
        on the former never raises and never produces NaN, so only an explicit
        check catches it.
        """
        cond, macro_prev, macro_next, micro_prev = self._two_pair_setup()
        agent = self._make_agent(cond, macro_prev, macro_next, micro_prev)
        agent.balanced_transitions = True
        agent.state_now = {"pLvl": np.linspace(0.5, 2.0, macro_prev.size)}
        agent.state_prev = {}
        with self.assertWarns(RuntimeWarning) as cm:
            agent.get_micro_markov_states()
        self.assertIn("state_prev", str(cm.warning))

    def test_unset_sentinel_cannot_be_a_valid_micro_state(self):
        """The sentinel must stay outside the range of real micro states.

        Nothing in the current drawing loops can leave an agent unwritten, so
        the end-of-method check is an instrument rather than a live guard.  It
        only works while the sentinel is unreachable: setting it to 0 would
        make a genuinely unwritten agent indistinguishable from one drawn into
        micro state 0, which is exactly the silent failure it exists to stop.
        """
        self.assertLess(agg_ind_mrkv_module._UNSET_MICRO, 0)


class testBalancedSortKeyDuringSimulation(unittest.TestCase):
    """The balanced sort key comes from state_prev during a live simulation.

    ``MarkovConsumerType.get_markov_states`` runs inside ``get_shocks``, which
    ``AgentType._sim_period_prologue`` calls after replacing every ndarray in
    ``state_now`` with ``np.empty``.  Reading ``state_now["pLvl"]`` there sorts
    agents by uninitialized memory: it never raises, never produces NaN, and
    varies with the allocator, so no assertion elsewhere in the suite notices.
    This test drives a real simulation and captures the sort key actually
    handed to ``MarkovProcess.draw``.
    """

    def _run_and_capture_sort_keys(self):
        init = dict(init_indshk_markov)
        init["cycles"] = 0
        init["MrkvArray"] = [np.array([[0.9, 0.1], [0.1, 0.9]])]
        init["constructors"] = dict(init_indshk_markov["constructors"])
        init["constructors"]["MrkvArray"] = None
        init["Rfree"] = [np.array([1.03, 1.03])]
        init["LivPrb"] = [np.array([0.98, 0.98])]
        init["PermGroFac"] = [np.array([1.01, 1.01])]
        init["AgentCount"] = 100
        init["T_sim"] = 3
        init["markov_shuffle"] = True
        init["balanced_transitions"] = True

        agent = MarkovConsumerType(**init)
        agent.solve()

        captured = []
        real_draw = MarkovProcess.draw

        def spy(self, state_now_indices, shuffle=False, sort_key=None):
            captured.append(None if sort_key is None else np.array(sort_key))
            return real_draw(
                self, state_now_indices, shuffle=shuffle, sort_key=sort_key
            )

        MarkovProcess.draw = spy
        try:
            agent.initialize_sim()
            prev_snapshots = []
            real_get = type(agent).get_markov_states

            def record(self):
                prev_snapshots.append(np.array(self.state_prev["pLvl"]))
                return real_get(self)

            type(agent).get_markov_states = record
            try:
                agent.simulate()
            finally:
                type(agent).get_markov_states = real_get
        finally:
            MarkovProcess.draw = real_draw
        return captured, prev_snapshots

    def test_sort_key_matches_previous_period_pLvl(self):
        captured, prev_snapshots = self._run_and_capture_sort_keys()
        self.assertTrue(captured, "no sort key reached MarkovProcess.draw")
        self.assertEqual(len(captured), len(prev_snapshots))
        for period, (sort_key, pLvl_prev) in enumerate(zip(captured, prev_snapshots)):
            self.assertIsNotNone(sort_key, f"period {period}: sort key was None")
            np.testing.assert_array_equal(
                sort_key,
                pLvl_prev,
                err_msg=(
                    f"period {period}: the sort key handed to "
                    f"MarkovProcess.draw is not state_prev['pLvl'].  If it "
                    f"came from state_now it is uninitialized memory at this "
                    f"point in the simulation loop."
                ),
            )

    def test_sort_key_holds_no_uninitialized_values(self):
        """Independent check that does not depend on the state_prev snapshot.

        Permanent income is strictly positive and O(1) here, so a denormal or
        a zero is a direct signature of a blanked ``np.empty`` buffer.
        """
        captured, _ = self._run_and_capture_sort_keys()
        for period, sort_key in enumerate(captured):
            self.assertIsNotNone(sort_key)
            self.assertTrue(
                np.all(np.isfinite(sort_key)),
                msg=f"period {period}: non-finite sort key {sort_key}",
            )
            self.assertGreater(
                float(np.min(sort_key)),
                1e-6,
                msg=(
                    f"period {period}: sort key contains values at or below "
                    f"1e-6 (min {float(np.min(sort_key))!r}); permanent income "
                    f"cannot be that small, so this is uninitialized memory"
                ),
            )


class testMakeShockHistoryShuffleFlag(unittest.TestCase):
    """make_shock_history(shuffle=True) restores income_shuffle / markov_shuffle."""

    def test_restores_flags_after_shuffle_true(self):
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
